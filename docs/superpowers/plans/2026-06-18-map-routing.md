# Map Routing & Road-Snapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace noisy incident-cluster polylines and fuzzy barricade markers with road-snapped geometry from OpenStreetMap so diversion routes follow actual roads and barricade pins land precisely at road intersections.

**Architecture:** A new `src/road_network.py` module owns the OSMnx graph lifecycle (one-time download → SCC reduction → GraphML cache → nearest-node snap → Dijkstra routing). `src/map_builder.py` consumes it for all geometry. `app.py` loads the graph once inside `@st.cache_data` and passes it through.

**Tech Stack:** `osmnx>=2.0.0`, `networkx` (transitive dep of osmnx), `pandas`, `folium` (existing).

## Global Constraints

- OSMnx **2.x API** throughout: `graph_from_bbox((west, south, east, north), ...)` and `ox.distance.nearest_nodes(G, X=lng, Y=lat)`.
- Bounding box: `(77.28, 12.78, 77.80, 13.30)` = (west, south, east, north). **Do not** use `graph_from_place("Bengaluru")`.
- SCC reduction is **mandatory** after download — use pure NetworkX (`nx.strongly_connected_components`) so there is no OSMnx-version dependency.
- Cache path: `data/bengaluru_drive.graphml`. Subsequent loads skip the download.
- **No fallback** to the old noisy polyline method. `_corridor_path()` is deleted.
- `src/model.py` is **off-limits** — do not touch it.
- All existing 41 tests must pass after **each** task.

---

### Task 1: Install osmnx, create `src/road_network.py` (`load_graph` + `nearest_node`), add `bengaluru_graph` fixture

**Files:**
- Modify: `requirements.txt`
- Create: `src/road_network.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_road_network.py`

**Interfaces:**
- Produces:
  - `load_graph(cache_path: Path) -> nx.MultiDiGraph`
  - `nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int`

- [ ] **Step 1: Install osmnx and lightgbm**

```bash
pip install "osmnx>=2.0.0" "lightgbm>=4.0.0"
```

- [ ] **Step 2: Update `requirements.txt`**

Open `requirements.txt`. It currently contains 7 lines. Replace the entire file with:

```
streamlit>=1.32.0
folium>=0.16.0
streamlit-folium>=0.20.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
pytest>=8.0.0
lightgbm>=4.0.0
osmnx>=2.0.0
```

Run: `pip install -r requirements.txt`
Expected: no errors.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_road_network.py` with this exact content:

```python
# tests/test_road_network.py
import pytest
import networkx as nx
from pathlib import Path
from src.road_network import load_graph, nearest_node


def test_load_graph_saves_and_reloads(tmp_path, bengaluru_graph):
    """Loading from a saved GraphML produces the same graph."""
    import osmnx as ox
    cache = tmp_path / "copy.graphml"
    ox.save_graphml(bengaluru_graph, cache)
    G2 = load_graph(cache)
    assert G2.number_of_nodes() == bengaluru_graph.number_of_nodes()


def test_nearest_node_returns_int(bengaluru_graph):
    node = nearest_node(bengaluru_graph, lat=12.97, lng=77.59)
    assert isinstance(node, int)


def test_nearest_node_in_graph(bengaluru_graph):
    node = nearest_node(bengaluru_graph, lat=12.97, lng=77.59)
    assert node in bengaluru_graph.nodes


def test_all_nodes_strongly_connected(bengaluru_graph):
    assert nx.is_strongly_connected(bengaluru_graph)
```

Add the `bengaluru_graph` fixture to `tests/conftest.py` (after the existing `sample_df` fixture):

```python
@pytest.fixture(scope="session")
def bengaluru_graph():
    from src.road_network import load_graph
    from pathlib import Path
    return load_graph(Path("data/bengaluru_drive.graphml"))
```

Run: `pytest tests/test_road_network.py -v`
Expected: 4 FAIL with `ImportError: cannot import name 'load_graph' from 'src.road_network'`

- [ ] **Step 4: Create `src/road_network.py` with `load_graph` and `nearest_node`**

Create `src/road_network.py` with this exact content:

```python
# src/road_network.py
import math
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

# Bounding box for Bengaluru event data + ~0.03° buffer.
# OSMnx 2.x format: (west, south, east, north) = (min_lng, min_lat, max_lng, max_lat)
_BBOX = (77.28, 12.78, 77.80, 13.30)


def load_graph(cache_path: Path) -> nx.MultiDiGraph:
    """Load Bengaluru drive graph from GraphML cache; download and cache if absent.

    Downloads once (~60 s, internet required). All subsequent loads read from disk (~2 s).
    The graph is reduced to its largest strongly-connected component so every node
    is reachable from every other node — nx.shortest_path never raises NetworkXNoPath.
    """
    if cache_path.exists():
        return ox.load_graphml(cache_path)
    G: nx.MultiDiGraph = ox.graph_from_bbox(_BBOX, network_type="drive")
    sccs = list(nx.strongly_connected_components(G))
    largest_scc = max(sccs, key=len)
    G = G.subgraph(largest_scc).copy()
    ox.save_graphml(G, cache_path)
    return G


def nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    """Snap a lat/lng coordinate to the nearest OSM node in G.

    OSMnx convention: X=longitude, Y=latitude.
    """
    return int(ox.distance.nearest_nodes(G, X=lng, Y=lat))
```

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/test_road_network.py -v`

**Note:** The `bengaluru_graph` session fixture triggers `load_graph("data/bengaluru_drive.graphml")`.
If `data/bengaluru_drive.graphml` does not yet exist, the first run downloads the Bengaluru road
network (~60 s, internet required) and saves it. Every subsequent `pytest` invocation loads from
disk in ~2 s.

Expected: 4 PASS

- [ ] **Step 6: Verify all existing tests still pass**

Run: `pytest tests/ -q --ignore=tests/test_road_network.py`
Expected: `41 passed`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/road_network.py tests/conftest.py tests/test_road_network.py
git commit -m "feat: road_network — load_graph and nearest_node with SCC guarantee"
```

---

### Task 2: Add `route_coords` and `corridor_route_coords` to `src/road_network.py`

**Files:**
- Modify: `src/road_network.py`
- Modify: `tests/test_road_network.py`

**Interfaces:**
- Consumes: `nearest_node(G, lat, lng) -> int` (Task 1)
- Produces:
  - `route_coords(G: nx.MultiDiGraph, orig_node: int, dest_node: int) -> list[tuple[float, float]]`
  - `corridor_route_coords(G: nx.MultiDiGraph, df: pd.DataFrame, corridor: str) -> list[tuple[float, float]]`
  - Each element is `(lat, lng)`. `G.nodes[n]["y"]` = latitude, `G.nodes[n]["x"]` = longitude.

- [ ] **Step 1: Write the failing tests**

Open `tests/test_road_network.py`. Replace the import line at the top:

```python
from src.road_network import load_graph, nearest_node, route_coords, corridor_route_coords
```

Then add these three tests at the bottom of the file (keep all four existing tests):

```python
def test_route_coords_returns_road_following_list(bengaluru_graph):
    n1 = nearest_node(bengaluru_graph, 12.97, 77.59)
    n2 = nearest_node(bengaluru_graph, 12.98, 77.61)
    coords = route_coords(bengaluru_graph, n1, n2)
    assert len(coords) > 1
    assert all(len(c) == 2 for c in coords)


def test_route_coords_endpoints_near_inputs(bengaluru_graph):
    lat1, lng1, lat2, lng2 = 12.97, 77.59, 12.98, 77.61
    n1 = nearest_node(bengaluru_graph, lat1, lng1)
    n2 = nearest_node(bengaluru_graph, lat2, lng2)
    coords = route_coords(bengaluru_graph, n1, n2)
    # First waypoint within ~1 km of start; last within ~1 km of end
    assert abs(coords[0][0] - lat1) < 0.01
    assert abs(coords[-1][0] - lat2) < 0.01


def test_corridor_route_coords_returns_road_path(bengaluru_graph):
    from src.pipeline import load_raw
    df = load_raw()
    coords = corridor_route_coords(bengaluru_graph, df, "CBD 2")
    assert len(coords) > 1
    assert all(len(c) == 2 for c in coords)
```

Run: `pytest tests/test_road_network.py::test_route_coords_returns_road_following_list -v`
Expected: FAIL with `ImportError: cannot import name 'route_coords'`

- [ ] **Step 2: Implement `_haversine_km`, `route_coords`, and `corridor_route_coords`**

Open `src/road_network.py`. Add the following three functions after `nearest_node` (before the end of the file):

```python
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def route_coords(G: nx.MultiDiGraph, orig_node: int, dest_node: int) -> list:
    """Return road-following (lat, lng) waypoints between two OSM node IDs via Dijkstra."""
    node_ids = nx.shortest_path(G, orig_node, dest_node, weight="length")
    return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in node_ids]


def corridor_route_coords(G: nx.MultiDiGraph, df: pd.DataFrame, corridor: str) -> list:
    """Return road-following waypoints along the full length of a named corridor.

    Finds the two most geographically distant historical events on the corridor,
    snaps both to OSM nodes, and returns the Dijkstra shortest path between them.
    """
    sub = (
        df[df["corridor"] == corridor]
        .dropna(subset=["latitude", "longitude"])
        .reset_index(drop=True)
    )
    if sub.empty:
        raise ValueError(f"No data for corridor '{corridor}'")
    if len(sub) > 50:
        sub = sub.sample(50, random_state=42).reset_index(drop=True)

    # Find the two most geographically distant rows (corridor start and end)
    max_dist, i_max, j_max = -1.0, 0, min(1, len(sub) - 1)
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            d = _haversine_km(
                float(sub.loc[i, "latitude"]), float(sub.loc[i, "longitude"]),
                float(sub.loc[j, "latitude"]), float(sub.loc[j, "longitude"]),
            )
            if d > max_dist:
                max_dist, i_max, j_max = d, i, j

    start_node = nearest_node(G, float(sub.loc[i_max, "latitude"]), float(sub.loc[i_max, "longitude"]))
    end_node   = nearest_node(G, float(sub.loc[j_max, "latitude"]), float(sub.loc[j_max, "longitude"]))
    return route_coords(G, start_node, end_node)
```

- [ ] **Step 3: Run all road_network tests**

Run: `pytest tests/test_road_network.py -v`
Expected: 7 PASS

- [ ] **Step 4: Run full suite**

Run: `pytest tests/ -q`
Expected: `44 passed` (41 original + 3 new road_network tests from this task; the 4 from Task 1 were already counted)

Wait — Task 1 added 4 tests. This task adds 3. Total: 41 + 4 + 3 = 48. Run and confirm the actual count.

Run: `pytest tests/ -q`
Expected: `48 passed`

- [ ] **Step 5: Commit**

```bash
git add src/road_network.py tests/test_road_network.py
git commit -m "feat: road_network — route_coords and corridor_route_coords via Dijkstra"
```

---

### Task 3: Rewrite `src/map_builder.py` and update all test files that call `build_map`

**Files:**
- Modify: `src/map_builder.py` (full rewrite)
- Modify: `tests/test_map_builder.py` (full rewrite)
- Modify: `tests/test_integration.py` (update `trained_state` fixture + `build_map` call)

**Interfaces:**
- Consumes:
  - `road_network.nearest_node(G, lat, lng) -> int` (Task 1)
  - `road_network.corridor_route_coords(G, df, corridor) -> list[(lat,lng)]` (Task 2)
- Produces: `build_map(event_lat, event_lng, severity, barricade_junctions, diversion_corridors, officer_info, train_df, event_name, G) -> folium.Map`
  - `G` is the last positional argument (keyword-safe via `G=`).

- [ ] **Step 1: Write the failing tests**

Replace the entire content of `tests/test_map_builder.py`:

```python
# tests/test_map_builder.py
import pytest
import folium  # type: ignore[import]
from src.map_builder import build_map


def test_build_map_returns_folium_map(sample_df, bengaluru_graph):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="HIGH",
        barricade_junctions=["QueensStatueCircle"],
        diversion_corridors=["ORR East 1"],
        officer_info={"total_min": 10, "total_max": 12},
        train_df=sample_df,
        event_name="Test Event",
        G=bengaluru_graph,
    )
    assert isinstance(m, folium.Map)


def test_build_map_low_severity_uses_green(sample_df, bengaluru_graph):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="LOW",
        barricade_junctions=[],
        diversion_corridors=[],
        officer_info={"total_min": 2, "total_max": 4},
        train_df=sample_df,
        event_name="Small Event",
        G=bengaluru_graph,
    )
    assert isinstance(m, folium.Map)


def test_build_map_has_polyline_for_diversion(sample_df, bengaluru_graph):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="MEDIUM",
        barricade_junctions=[],
        diversion_corridors=["ORR East 1"],
        officer_info={"total_min": 4, "total_max": 6},
        train_df=sample_df,
        event_name="Diversion Test",
        G=bengaluru_graph,
    )
    assert isinstance(m, folium.Map)
    children = list(m._children.values())
    has_polyline = any(isinstance(c, folium.PolyLine) for c in children)
    assert has_polyline, "Expected a PolyLine for the diversion route"
```

Run: `pytest tests/test_map_builder.py -v`
Expected: 3 FAIL — `build_map() got an unexpected keyword argument 'G'`

- [ ] **Step 2: Rewrite `src/map_builder.py`**

Replace the **entire** content of `src/map_builder.py` with:

```python
# src/map_builder.py
from collections import deque

import folium  # type: ignore[import]
import networkx as nx
import pandas as pd

import src.road_network as road_network

_SEVERITY_COLOR  = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
_SEVERITY_RADIUS = {"LOW": 500,     "MEDIUM": 1000,     "HIGH": 2000}


def _junction_centroid(df: pd.DataFrame, junction: str):
    """Return mean (lat, lng) of all historical events at a named junction."""
    sub: pd.DataFrame = df[df["junction"] == junction]
    sub = sub.dropna(subset=["latitude", "longitude"])
    if sub.empty:
        return None
    return (float(sub["latitude"].mean()), float(sub["longitude"].mean()))


def _corridor_centroid(df: pd.DataFrame, corridor: str):
    """Return mean (lat, lng) of all historical events on a corridor."""
    sub: pd.DataFrame = df[df["corridor"] == corridor]
    sub = sub.dropna(subset=["latitude", "longitude"])
    if sub.empty:
        return None
    return (float(sub["latitude"].mean()), float(sub["longitude"].mean()))


def _snap_to_intersection(G: nx.MultiDiGraph, lat: float, lng: float) -> tuple:
    """Snap a lat/lng to the nearest road intersection (node degree >= 3) via BFS.

    Walks outward from the nearest node up to 3 hops until a true junction is found.
    Falls back to the original nearest node if no degree-3 node is reachable.
    """
    node_id = road_network.nearest_node(G, lat, lng)
    visited = {node_id}
    queue = deque([(node_id, 0)])
    while queue:
        nid, depth = queue.popleft()
        if G.degree(nid) >= 3:
            return (float(G.nodes[nid]["y"]), float(G.nodes[nid]["x"]))
        if depth < 3:
            neighbors = set(G.successors(nid)) | set(G.predecessors(nid))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return (float(G.nodes[node_id]["y"]), float(G.nodes[node_id]["x"]))


def build_map(
    event_lat: float,
    event_lng: float,
    severity: str,
    barricade_junctions: list,
    diversion_corridors: list,
    officer_info: dict,
    train_df: pd.DataFrame,
    event_name: str,
    G: nx.MultiDiGraph,
) -> folium.Map:
    color  = _SEVERITY_COLOR[severity]
    radius = _SEVERITY_RADIUS[severity]

    m = folium.Map(location=[event_lat, event_lng], zoom_start=14)
    all_coords = [[event_lat, event_lng]]

    # Impact zone
    folium.Circle(
        location=[event_lat, event_lng],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.15,
        popup=f"{severity} impact zone ({radius}m radius)",
    ).add_to(m)

    # Event epicenter marker
    folium.Marker(
        location=[event_lat, event_lng],
        popup=(
            f"{event_name}<br>Severity: {severity}<br>"
            f"Officers: {officer_info['total_min']}-{officer_info['total_max']}"
        ),
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

    # Barricade positions — snapped to real road intersections
    for junction in barricade_junctions:
        centroid = _junction_centroid(train_df, junction)
        if centroid is None:
            continue
        coords = _snap_to_intersection(G, centroid[0], centroid[1])
        folium.Marker(
            location=list(coords),
            popup=f"Barricade: {junction}",
            icon=folium.Icon(color="red", icon="remove-sign"),
        ).add_to(m)
        all_coords.append(list(coords))

    # Diversion routes — road-following polylines via Dijkstra on OSM graph
    for corridor in diversion_corridors:
        path = road_network.corridor_route_coords(G, train_df, corridor)
        folium.PolyLine(
            locations=path,
            color="blue",
            weight=5,
            opacity=0.8,
            tooltip=f"Diversion → {corridor}",
            popup=f"Divert via: {corridor}",
        ).add_to(m)
        centroid = _corridor_centroid(train_df, corridor)
        if centroid:
            folium.Marker(
                location=list(centroid),
                tooltip=f"Diversion → {corridor}",
                popup=f"Diversion route: {corridor}",
                icon=folium.Icon(color="blue", icon="share-alt"),
            ).add_to(m)
            all_coords.extend(path)

    if len(all_coords) > 1:
        m.fit_bounds(all_coords)

    return m
```

- [ ] **Step 3: Run map_builder tests**

Run: `pytest tests/test_map_builder.py -v`
Expected: 3 PASS

- [ ] **Step 4: Update `tests/test_integration.py`**

Replace the **entire** content of `tests/test_integration.py`:

```python
# tests/test_integration.py
"""
Smoke test: full pipeline from raw CSV to prediction and recommendation.
Does not start Streamlit — exercises all four layers directly.
"""
import pandas as pd
import pytest
import folium  # type: ignore[import]
from pathlib import Path

from src.pipeline import load_raw, split_data, corridor_metadata
from src.baseline import (
    compute_window_counts, compute_corridor_baselines,
    compute_excess_scores, compute_tertile_thresholds, label_severity,
)
from src.model import train_model, evaluate_cv, evaluate_test, predict, get_knn_neighbors
from src.recommender import officer_count, barricade_positions, build_diversion_graph, get_diversions
from src.map_builder import build_map
from src.road_network import load_graph


@pytest.fixture(scope="module")
def trained_state():
    df = load_raw()
    df["window_count"] = compute_window_counts(df)
    train_df, val_df, test_df = split_data(df)
    baselines = compute_corridor_baselines(train_df)
    for split in [train_df, val_df, test_df]:
        split["impact_score"] = compute_excess_scores(split, baselines)
    low_t, high_t = compute_tertile_thresholds(train_df)
    for split in [train_df, val_df, test_df]:
        split["severity"] = label_severity(split, low_t, high_t)
    pipeline = train_model(train_df)
    diversion_graph = build_diversion_graph(
        pd.concat([train_df, val_df], ignore_index=True)
    )
    graph = load_graph(Path("data/bengaluru_drive.graphml"))
    return dict(
        train_df=train_df,
        test_df=test_df,
        pipeline=pipeline,
        diversion_graph=diversion_graph,
        graph=graph,
    )


def test_test_f1_above_minimum_bar(trained_state):
    score = evaluate_test(trained_state["pipeline"], trained_state["test_df"])
    assert score >= 0.50, (
        f"Test macro-F1 = {score:.3f} is below minimum bar of 0.50. "
        "Consider switching corridor encoding to geospatial clusters."
    )


def test_end_to_end_cbd2_scenario(trained_state):
    train_df        = trained_state["train_df"]
    pipeline        = trained_state["pipeline"]
    diversion_graph = trained_state["diversion_graph"]
    graph           = trained_state["graph"]

    corridor = "CBD 2"
    zone, police, lat, lng = corridor_metadata(train_df, corridor)

    features = {
        "event_cause":           "public_event",
        "event_type":            "planned",
        "corridor":              corridor,
        "zone":                  zone,
        "police_station":        police,
        "hour_band":             "evening",
        "hour_of_day":           18,
        "day_of_week":           0,
        "requires_road_closure": False,
        "priority":              "High",
        "junction":              "unknown",
        "month":                 2,
        "is_weekend":            0,
    }

    severity, confidence = predict(pipeline, features)
    assert severity in {"LOW", "MEDIUM", "HIGH"}
    assert abs(sum(confidence.values()) - 1.0) < 1e-6

    barricades = barricade_positions(train_df, corridor, event_lat=lat, event_lng=lng, top_n=4)
    n_adj      = min(3, len(barricades))
    officers   = officer_count(severity, n_adjacent_junctions=n_adj)
    assert officers["total_min"] >= 2

    diversions = get_diversions(diversion_graph, corridor, features["hour_band"])

    fmap = build_map(
        lat, lng, severity, barricades, diversions, officers,
        train_df, "CBD 2 Rally", graph,
    )
    assert isinstance(fmap, folium.Map)

    neighbors = get_knn_neighbors(train_df, features, k=5)
    assert len(neighbors) == 5
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: `51 passed` (48 from Tasks 1–2 + 3 map_builder tests; integration tests were already in the 48 count — recount: 41 original + 4 road_network Task1 + 3 road_network Task2 + 3 map_builder = 51)

- [ ] **Step 6: Commit**

```bash
git add src/map_builder.py tests/test_map_builder.py tests/test_integration.py
git commit -m "feat: map_builder — road-snapped barricades and Dijkstra diversion polylines"
```

---

### Task 4: Wire the graph into `app.py`

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `load_graph(cache_path: Path) -> nx.MultiDiGraph` (Task 1), `build_map(..., G) -> folium.Map` (Task 3)

There are no new tests for this task — `test_integration.py` already exercises the full stack including `build_map` with a real graph. This task only wires the already-tested pieces into the Streamlit entrypoint.

- [ ] **Step 1: Add imports to `app.py`**

Open `app.py`. The current import block ends around line 15. Add two lines after the existing `from src.duration_model import ...` import:

```python
from pathlib import Path
from src.road_network import load_graph
```

- [ ] **Step 2: Load graph inside `load_and_train()`**

In `load_and_train()`, find the line:

```python
    train_df, val_df, test_df = split_data(df)
```

Add this line **immediately after** it:

```python
    graph = load_graph(Path("data/bengaluru_drive.graphml"))
```

- [ ] **Step 3: Add graph to the returned dict**

Find the `return {` block at the end of `load_and_train()`. It currently ends with `"dur_model": dur_model,`. Add `"graph"` as the last entry:

```python
    return {
        "train_df":        train_df,
        "baselines":       baselines,
        "low_t":           low_t,
        "high_t":          high_t,
        "pipeline":        pipeline,
        "cv_f1":           cv_f1,
        "test_f1":         test_f1,
        "diversion_graph": diversion_graph,
        "dur_model":       dur_model,
        "graph":           graph,
    }
```

- [ ] **Step 4: Unpack graph from state**

Find the block after `state = load_and_train()` where state keys are unpacked (lines ~72–77). Add:

```python
graph           = state["graph"]
```

- [ ] **Step 5: Pass `G` to `build_map()`**

Find the `build_map(...)` call in the form submission handler (inside `if submitted:`). Replace it with:

```python
fmap = build_map(lat, lng, severity, barricades, diversions, officers, train_df, event_name, graph)
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest tests/ -q`
Expected: `51 passed` (same count — no new tests, no regressions)

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: wire OSMnx road graph into app.py — road-snapped map end-to-end"
```

---

## Self-Review

**Spec coverage:**
1. ✅ `load_graph` with OSMnx 2.x bbox `(77.28, 12.78, 77.80, 13.30)` — Task 1
2. ✅ SCC reduction via `nx.strongly_connected_components` — Task 1
3. ✅ GraphML cache at `data/bengaluru_drive.graphml` — Task 1
4. ✅ `nearest_node` with `X=lng, Y=lat` convention — Task 1
5. ✅ `route_coords` via `nx.shortest_path(weight="length")` — Task 2
6. ✅ `corridor_route_coords` haversine max-distance endpoints + subsample 50 — Task 2
7. ✅ `_snap_to_intersection` BFS degree≥3, successors+predecessors, max 3 hops — Task 3
8. ✅ `_junction_coords` renamed `_junction_centroid` — Task 3
9. ✅ `_corridor_path` deleted — Task 3
10. ✅ `build_map` new signature `(..., G)` — Task 3
11. ✅ Barricades use `_snap_to_intersection` — Task 3
12. ✅ Diversion polylines use `corridor_route_coords` — Task 3
13. ✅ `app.py` loads graph in `load_and_train` — Task 4
14. ✅ `app.py` unpacks `graph` from state and passes to `build_map` — Task 4
15. ✅ `test_road_network.py` — Tasks 1 & 2
16. ✅ `test_map_builder.py` updated — Task 3
17. ✅ `test_integration.py` updated — Task 3
18. ✅ `osmnx>=2.0.0` in `requirements.txt` — Task 1
19. ✅ No fallback — `_corridor_path` deleted in Task 3
20. ✅ `src/model.py` untouched
