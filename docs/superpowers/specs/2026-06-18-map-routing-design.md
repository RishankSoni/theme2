# Map Routing & Road-Snapping Design

## Goal

Replace the noisy incident-cluster polylines and fuzzy barricade markers in the Folium map with road-snapped geometry derived from OpenStreetMap via OSMnx, so that diversion routes follow actual roads and barricade pins land precisely at road intersections.

## Architecture

Four files are touched; one is new.

| File | Change |
|---|---|
| `src/road_network.py` | **New.** Owns the OSMnx graph lifecycle: download, SCC reduction, GraphML cache, nearest-node snap, road-following route. |
| `src/map_builder.py` | **Modified.** Accepts graph `G` as a new argument; uses `road_network` for corridor paths and barricade snapping. |
| `src/recommender.py` | **No algorithm change.** `barricade_positions()` signature and ranking logic are unchanged; only the downstream marker placement in `map_builder` improves. |
| `app.py` | **Minimal.** `load_and_train()` loads the graph once and stores it in the state dict; `build_map()` call gains one argument `G`. |
| `tests/test_road_network.py` | **New.** Four tests covering graph load, node snap, route generation, and SCC guarantee. |

## Global Constraints

- OSMnx graph is downloaded using a bounding box derived from the event data (lat 12.80–13.27, lng 77.31–77.77, small buffer). Do **not** use `graph_from_place("Bengaluru")` — the bbox query is smaller and faster.
- Immediately after download, reduce to the largest strongly-connected component: `ox.utils_graph.get_largest_component(G, strongly=True)`. This is a hard requirement — it guarantees every node is reachable from every other node so `nx.shortest_path` can never raise `NetworkXNoPath`.
- Save the SCC-reduced graph as `data/bengaluru_drive.graphml`. Subsequent loads read from this file (~2 s) without re-downloading.
- There is **no fallback** to the old noisy polyline method. Road-snapped routing is the only code path.
- The severity model in `src/model.py` is **off-limits** — do not touch it.
- All existing 41 tests must continue to pass after implementation.

---

## Component 1 — `src/road_network.py`

### `load_graph(cache_path: Path) -> nx.MultiDiGraph`

- If `cache_path` exists: `return ox.load_graphml(cache_path)`
- Else:
  - `bbox = (13.30, 12.78, 77.80, 77.28)` (north, south, east, west with ~0.03° buffer)
  - `G = ox.graph_from_bbox(bbox, network_type="drive")`
  - `G = ox.utils_graph.get_largest_component(G, strongly=True)`
  - `ox.save_graphml(G, cache_path)`
  - `return G`

### `nearest_node(G, lat: float, lng: float) -> int`

```python
return int(ox.distance.nearest_nodes(G, X=lng, Y=lat))
```

Note: OSMnx convention is `X=longitude, Y=latitude`.

### `route_coords(G, orig_node: int, dest_node: int) -> list[tuple[float, float]]`

```python
node_ids = nx.shortest_path(G, orig_node, dest_node, weight="length")
return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_ids]
```

Returns `(lat, lng)` tuples. `G.nodes[n]["y"]` is latitude, `G.nodes[n]["x"]` is longitude (OSMnx convention).

### `corridor_route_coords(G, df: pd.DataFrame, corridor: str) -> list[tuple[float, float]]`

```python
sub = df[df["corridor"] == corridor].dropna(subset=["latitude", "longitude"])
# Find the two most geographically distant points (corridor start and end)
# Use max haversine distance across all pairs (subsample to 50 points if large)
# Snap both endpoints to nearest OSM node
# Return route_coords(G, start_node, end_node)
```

Full algorithm:
1. If `sub` is empty, raise `ValueError(f"No data for corridor {corridor}")`.
2. Subsample: if `len(sub) > 50`, take `sub.sample(50, random_state=42)`.
3. Compute pairwise haversine distances; find the pair `(i, j)` with maximum distance.
4. `start_node = nearest_node(G, sub.iloc[i]["latitude"], sub.iloc[i]["longitude"])`
5. `end_node   = nearest_node(G, sub.iloc[j]["latitude"], sub.iloc[j]["longitude"])`
6. `return route_coords(G, start_node, end_node)`

---

## Component 2 — `src/map_builder.py` changes

### New helper: `_snap_to_intersection(G, lat, lng) -> tuple[float, float]`

```python
node_id = road_network.nearest_node(G, lat, lng)
# Walk outward (BFS, max 3 hops) until a node with degree >= 3 is found
from collections import deque
visited = {node_id}
queue = deque([(node_id, 0)])
while queue:
    nid, depth = queue.popleft()
    if G.degree(nid) >= 3:
        return (G.nodes[nid]["y"], G.nodes[nid]["x"])
    if depth < 3:
        for neighbor in G.neighbors(nid):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
# If no degree-3 node found within 3 hops, return the original nearest node
return (G.nodes[node_id]["y"], G.nodes[node_id]["x"])
```

### `_corridor_path()` — removed entirely

Replaced by `road_network.corridor_route_coords(G, df, corridor)` called directly in `build_map()`.

### `build_map()` — updated signature

```python
def build_map(
    event_lat, event_lng, severity, barricade_junctions,
    diversion_corridors, officer_info, train_df, event_name, G
) -> folium.Map:
```

**Barricade markers:** replace `_junction_coords(train_df, junction)` with:
```python
raw = _junction_centroid(train_df, junction)   # mean lat/lng as before
coords = _snap_to_intersection(G, raw[0], raw[1])
```

**Diversion polylines:** replace `_corridor_path(train_df, corridor)` with:
```python
path = road_network.corridor_route_coords(G, train_df, corridor)
```

The `PolyLine` call is unchanged — only the `locations=path` source changes.

---

## Component 3 — `app.py` changes

### In `load_and_train()`

Add after `split_data`:
```python
from src.road_network import load_graph
from pathlib import Path
graph = load_graph(Path("data/bengaluru_drive.graphml"))
```

Add `"graph": graph` to the returned dict.

### At call site

```python
graph = state["graph"]
# ...
fmap = build_map(lat, lng, severity, barricades, diversions, officers,
                 train_df, event_name, graph)
```

---

## Testing

### `tests/test_road_network.py` (new file)

```python
def test_load_graph_creates_cache(tmp_path):
    cache = tmp_path / "test_drive.graphml"
    G = load_graph(cache)
    assert cache.exists()
    assert G.number_of_nodes() > 0

def test_load_graph_uses_cache(tmp_path):
    cache = tmp_path / "test_drive.graphml"
    G1 = load_graph(cache)
    G2 = load_graph(cache)   # second call must not re-download
    assert G1.number_of_nodes() == G2.number_of_nodes()

def test_nearest_node_returns_int(bengaluru_graph):
    node = nearest_node(bengaluru_graph, lat=12.97, lng=77.59)
    assert isinstance(node, int)

def test_route_coords_returns_road_following_list(bengaluru_graph):
    n1 = nearest_node(bengaluru_graph, 12.97, 77.59)
    n2 = nearest_node(bengaluru_graph, 12.98, 77.61)
    coords = route_coords(bengaluru_graph, n1, n2)
    assert len(coords) > 1
    assert all(len(c) == 2 for c in coords)

def test_all_nodes_strongly_connected(bengaluru_graph):
    import networkx as nx
    assert nx.is_strongly_connected(bengaluru_graph)
```

`bengaluru_graph` is a module-scoped fixture that calls `load_graph(Path("data/bengaluru_drive.graphml"))` — uses the real cached file.

### `tests/test_map_builder.py` update

Add `test_build_map_returns_polyline_layers(bengaluru_graph, sample_df)`:
- Call `build_map(...)` with the real graph
- Introspect the Folium map's `_children` to assert at least one `PolyLine` child is present

### `tests/test_integration.py` update

Update `test_end_to_end_cbd2_scenario` fixture to load graph and pass it to `build_map()`.

---

## Dependency

```
osmnx>=2.0.0
```

Add to `requirements.txt` (or `pyproject.toml` if present). OSMnx 2.x API is used throughout (`ox.utils_graph`, `ox.distance.nearest_nodes`, `ox.routing` module).
