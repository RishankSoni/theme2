# Three Innovations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add geospatial barricade routing, time-banded diversion graphs, and duration forecasting to the GRIDLOCK 2.0 traffic event planner.

**Architecture:** Three independent enhancements layered onto existing modules — `pipeline.py` adds `duration_h`, `recommender.py` gains geospatial and temporal awareness, and a new `src/duration_model.py` benchmarks three models and exposes a `predict_duration` function. `app.py` wires all three with minimal changes.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, LightGBM, Streamlit

## Global Constraints

- `src/model.py` is **completely off-limits** — the severity LightGBM classifier (macro F1 = 0.74) must not change
- Duration model uses `OrdinalEncoder` (NOT `CorridorStatsTransformer`) in a separate pipeline
- `duration_h = (closed_datetime − start_datetime).dt.total_seconds() / 3600`, kept only when in `(0, 24]`; rows without `closed_datetime` → NaN (excluded from training, allowed at predict time)
- `app.py` already has `lat`, `lng` (from `corridor_metadata`) and `hb` (hour_band) in the form-submit block — just pass them as new args; no form UI changes needed
- 32 tests must continue to pass after each task
- `ALL_FEATURE_COLS = CAT_COLS + NUM_COLS` from `src/model.py` = 13 features (8 cat + 5 num); this is the feature set for the duration model too
- `_DURATION_FEATURES` in `src/duration_model.py` equals `CAT_COLS + NUM_COLS` — import from `src.model`

---

## Task 1: Add `duration_h` to `pipeline.py`

**Files:**
- Modify: `src/pipeline.py` — `load_raw()` only
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `load_raw()` returns DataFrame with `duration_h` column (float, NaN where closed_datetime missing or duration outside (0, 24])

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
from src.pipeline import load_raw


def test_load_raw_has_duration_h():
    df = load_raw()
    assert "duration_h" in df.columns
    valid = df["duration_h"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all()
    assert (valid <= 24).all()


def test_load_raw_duration_h_nan_when_no_closed_datetime():
    df = load_raw()
    # Rows where closed_datetime is NaT should have NaN duration_h
    missing_close = df[df["closed_datetime"].isna()]
    if not missing_close.empty:
        assert missing_close["duration_h"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_pipeline.py -v
```

Expected: FAIL — `KeyError: 'duration_h'`

- [ ] **Step 3: Implement**

In `src/pipeline.py`, inside `load_raw()`, add these two lines **after** the `requires_road_closure` block and **before** the `for col in [...]` fillna loop:

```python
    df["duration_h"] = (
        df["closed_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 3600
    df["duration_h"] = df["duration_h"].where(
        (df["duration_h"] > 0) & (df["duration_h"] <= 24)
    )
```

The full `load_raw()` after the change (lines to insert shown in context):

```python
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["start_datetime", "closed_datetime", "end_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=["start_datetime", "corridor"])
    df["hour_of_day"] = df["start_datetime"].dt.hour.astype(int)
    df["day_of_week"] = df["start_datetime"].dt.dayofweek.astype(int)
    df["hour_band"]   = df["hour_of_day"].apply(_hour_to_band)
    df["month"]      = df["start_datetime"].dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .astype(str).str.strip().str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
        .astype(object)
    )
    df["duration_h"] = (
        df["closed_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 3600
    df["duration_h"] = df["duration_h"].where(
        (df["duration_h"] > 0) & (df["duration_h"] <= 24)
    )
    for col in ["event_cause", "event_type", "corridor", "zone", "police_station", "junction", "priority"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run all tests**

```
pytest tests/test_pipeline.py tests/test_model.py tests/test_recommender.py -v
```

Expected: all pass (32 existing + 2 new = 34 total)

- [ ] **Step 5: Commit**

```
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add duration_h column to load_raw() in pipeline.py"
```

---

## Task 2: Geospatial Barricade Positions (Innovation 1)

**Files:**
- Modify: `src/recommender.py` — replace `barricade_positions`, add `_haversine_km`
- Modify: `tests/test_recommender.py` — update 2 existing barricade tests + add 1 geospatial test

**Interfaces:**
- Consumes: `train_df` with `latitude`, `longitude`, `corridor`, `requires_road_closure`, `junction` columns
- Produces: `barricade_positions(train_df, corridor, event_lat, event_lng, radius_km=2.0, top_n=4) -> list[str]`

- [ ] **Step 1: Write the failing test**

Update `tests/test_recommender.py`. Replace the two existing barricade tests and add a new geospatial test:

```python
def test_barricade_positions_top_junctions(sample_df):
    df = _add_features(sample_df)
    # radius_km=1000 → no geospatial filtering → same as old behavior
    positions = barricade_positions(df, corridor="CBD 2",
                                    event_lat=12.97, event_lng=77.59,
                                    radius_km=1000.0, top_n=2)
    assert "QueensStatueCircle" in positions


def test_barricade_positions_empty_for_corridor_with_no_closures(sample_df):
    df = _add_features(sample_df)
    df["requires_road_closure"] = False
    positions = barricade_positions(df, corridor="CBD 2",
                                    event_lat=12.97, event_lng=77.59, top_n=2)
    assert positions == []


def test_barricade_positions_geospatial_filters_far_junctions():
    """With 2+ near junctions, far junctions must be excluded."""
    near_rows = pd.DataFrame({
        "corridor":              ["CBD 2"] * 4,
        "requires_road_closure": [True] * 4,
        "junction":              ["NearA", "NearA", "NearB", "NearB"],
        "latitude":              [12.97, 12.97, 12.98, 12.98],
        "longitude":             [77.59, 77.59, 77.60, 77.60],
    })
    far_row = pd.DataFrame({
        "corridor":              ["CBD 2"],
        "requires_road_closure": [True],
        "junction":              ["FarJunc"],
        "latitude":              [10.00],
        "longitude":             [76.00],
    })
    df = pd.concat([near_rows, far_row], ignore_index=True)
    positions = barricade_positions(df, "CBD 2",
                                    event_lat=12.97, event_lng=77.59,
                                    radius_km=2.0, top_n=4)
    assert "FarJunc" not in positions
    assert "NearA" in positions
    assert "NearB" in positions
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_recommender.py::test_barricade_positions_top_junctions tests/test_recommender.py::test_barricade_positions_geospatial_filters_far_junctions -v
```

Expected: FAIL — `TypeError: barricade_positions() missing required positional arguments`

- [ ] **Step 3: Implement**

In `src/recommender.py`, add `import math` at the top and replace the `barricade_positions` function:

```python
import math
import pandas as pd
import numpy as np
```

```python
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def barricade_positions(
    train_df: pd.DataFrame,
    corridor: str,
    event_lat: float,
    event_lng: float,
    radius_km: float = 2.0,
    top_n: int = 4,
) -> list:
    """Junctions near the event epicenter most frequently requiring road closure."""
    mask = (
        (train_df["corridor"] == corridor)
        & (train_df["requires_road_closure"] == True)  # noqa: E712
    )
    subset: pd.DataFrame = train_df[mask].copy()
    subset = subset[subset["junction"].notna()]
    subset = subset[subset["junction"] != "unknown"]
    if subset.empty:
        return []

    junction_centroids = (
        subset.dropna(subset=["latitude", "longitude"])
        .groupby("junction")[["latitude", "longitude"]]
        .mean()
    )

    def _nearby(r_km: float) -> list:
        return [
            junc for junc, row in junction_centroids.iterrows()
            if _haversine_km(event_lat, event_lng,
                             float(row["latitude"]), float(row["longitude"])) <= r_km
        ]

    for candidate_radius in [radius_km, radius_km * 2.5]:
        survivors = _nearby(candidate_radius)
        if len(survivors) >= 2:
            return (
                subset[subset["junction"].isin(survivors)]["junction"]
                .value_counts()
                .head(top_n)
                .index.tolist()
            )

    # Fallback: corridor-wide top-N (original behavior)
    return subset["junction"].value_counts().head(top_n).index.tolist()
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all 35 tests pass (34 from Task 1 + 1 new geospatial test)

- [ ] **Step 5: Commit**

```
git add src/recommender.py tests/test_recommender.py
git commit -m "feat: geospatial barricade positioning with haversine distance filter and fallback chain"
```

---

## Task 3: Time-Banded Diversion Graph (Innovation 2)

**Files:**
- Modify: `src/recommender.py` — `build_diversion_graph` keys by `(corridor, hour_band)`; `get_diversions` gains `hour_band` arg
- Modify: `tests/test_recommender.py` — update 2 diversion tests

**Interfaces:**
- Produces: `build_diversion_graph(train_val_df, min_cooccurrences=5) -> dict[(str, str), list[str]]`
- Produces: `get_diversions(diversion_graph, corridor, hour_band) -> list[str]`

- [ ] **Step 1: Update failing tests**

In `tests/test_recommender.py`, update the two diversion tests:

```python
def test_build_diversion_graph_returns_dict(sample_df):
    df = _add_features(sample_df)
    big = pd.concat([df] * 10, ignore_index=True)
    big["window_count"] = compute_window_counts(big)
    baselines = compute_corridor_baselines(big, min_obs=1)
    big["impact_score"] = compute_excess_scores(big, baselines)
    graph = build_diversion_graph(big, min_cooccurrences=1)
    assert isinstance(graph, dict)
    # Keys must be (corridor, hour_band) tuples
    for key in graph:
        assert isinstance(key, tuple)
        assert len(key) == 2


def test_get_diversions_returns_list(sample_df):
    df = _add_features(sample_df)
    big = pd.concat([df] * 10, ignore_index=True)
    big["window_count"] = compute_window_counts(big)
    baselines = compute_corridor_baselines(big, min_obs=1)
    big["impact_score"] = compute_excess_scores(big, baselines)
    graph = build_diversion_graph(big, min_cooccurrences=1)
    divs = get_diversions(graph, "CBD 2", "morning")  # hour_band required
    assert isinstance(divs, list)
    assert len(divs) <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_recommender.py::test_build_diversion_graph_returns_dict tests/test_recommender.py::test_get_diversions_returns_list -v
```

Expected: FAIL — `TypeError` (key check fails on str keys, get_diversions missing arg)

- [ ] **Step 3: Implement**

In `src/recommender.py`, replace `build_diversion_graph` and `get_diversions`:

```python
def build_diversion_graph(
    train_val_df: pd.DataFrame,
    min_cooccurrences: int = 5,
) -> dict:
    """
    Build a co-disruption graph keyed by (corridor, hour_band).
    Returns dict[(str, str), list[str]] — top-2 diversion corridors per key.
    """
    post = pd.Timedelta(hours=1)

    d_means: dict = {}
    for corr_key, grp in train_val_df.groupby("corridor"):
        d_means[corr_key] = float(grp["window_count"].mean())

    global_mean: float = float(train_val_df["window_count"].mean())
    if not global_mean:
        global_mean = 1.0

    raw: dict = {}  # (C, hour_band) -> {D: [count, ...]}

    for idx, row in train_val_df.iterrows():
        C  = row["corridor"]
        t  = row["start_datetime"]
        hb = row["hour_band"]
        key = (C, hb)

        co = train_val_df[
            (train_val_df.index != idx)
            & (train_val_df["corridor"] != C)
            & (train_val_df["start_datetime"] >= t)
            & (train_val_df["start_datetime"] <= t + post)
        ]
        for D, grp in co.groupby("corridor"):
            raw.setdefault(key, {}).setdefault(D, []).append(len(grp))

    result = {}
    for key, neighbors in raw.items():
        elevations = {}
        for D, counts in neighbors.items():
            if len(counts) < min_cooccurrences:
                continue
            mean_count = float(np.mean(counts))
            d_baseline = float(d_means.get(D, global_mean)) or global_mean
            elevations[D] = mean_count / d_baseline
        top2 = sorted(elevations, key=lambda k: elevations[k], reverse=True)[:2]
        if top2:
            result[key] = top2

    return result


def get_diversions(diversion_graph: dict, corridor: str, hour_band: str) -> list:
    """Return recommended diversion corridors for a given corridor and time band."""
    # 1. Exact time-band match
    exact = diversion_graph.get((corridor, hour_band))
    if exact is not None:
        return exact
    # 2. Any band for this corridor
    for key, divs in diversion_graph.items():
        if key[0] == corridor:
            return divs
    # 3. No data
    return []
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all 35 tests pass

- [ ] **Step 5: Commit**

```
git add src/recommender.py tests/test_recommender.py
git commit -m "feat: time-banded diversion graph keyed by (corridor, hour_band)"
```

---

## Task 4: Duration Forecasting Model (Innovation 3)

**Files:**
- Create: `src/duration_model.py`
- Create: `tests/test_duration_model.py`

**Interfaces:**
- Consumes: `train_df` with `duration_h` column (from Task 1) and `ALL_FEATURE_COLS` columns
- Produces:
  - `duration_tertile_thresholds(train_df) -> tuple[float, float]`
  - `compute_duration_labels(df, low_thresh, high_thresh) -> pd.Series`
  - `train_duration_model(train_df) -> dict` with keys `pipeline`, `kind`, `low_thresh`, `high_thresh`
  - `predict_duration(dur_model, features) -> str` — returns `"SHORT"`, `"MEDIUM"`, or `"LONG"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_duration_model.py
import numpy as np
import pandas as pd

from src.duration_model import (
    compute_duration_labels,
    duration_tertile_thresholds,
    predict_duration,
    train_duration_model,
)


def _make_duration_df(n: int = 120) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "event_cause":           ["public_event"] * n,
        "event_type":            ["planned"] * (n // 2) + ["unplanned"] * (n // 2),
        "corridor":              ["CBD 2"] * (n // 2) + ["ORR"] * (n // 2),
        "zone":                  ["Central Zone"] * n,
        "police_station":        ["Cubbon Park"] * n,
        "hour_band":             (
            ["morning"] * (n // 4) + ["afternoon"] * (n // 4)
            + ["evening"] * (n // 4) + ["night"] * (n // 4)
        ),
        "priority":              ["High"] * (n // 2) + ["Low"] * (n // 2),
        "junction":              ["unknown"] * n,
        "hour_of_day":           np.random.randint(0, 24, n).tolist(),
        "day_of_week":           np.random.randint(0, 7, n).tolist(),
        "requires_road_closure": [False] * n,
        "month":                 np.random.randint(1, 13, n).tolist(),
        "is_weekend":            np.random.randint(0, 2, n).tolist(),
        "duration_h":            np.random.exponential(1.5, n).clip(0.01, 24).tolist(),
    })


def test_duration_tertile_thresholds():
    df = _make_duration_df()
    low, high = duration_tertile_thresholds(df)
    assert 0 < low < high < 24


def test_compute_duration_labels_covers_all_classes():
    df = _make_duration_df()
    low, high = duration_tertile_thresholds(df)
    labels = compute_duration_labels(df, low, high)
    valid = labels.dropna()
    assert set(valid.unique()) == {"SHORT", "MEDIUM", "LONG"}


def test_compute_duration_labels_nan_for_null_duration():
    df = _make_duration_df()
    df.loc[0, "duration_h"] = np.nan
    low, high = duration_tertile_thresholds(df)
    labels = compute_duration_labels(df, low, high)
    assert pd.isna(labels.iloc[0])


def test_train_duration_model_returns_valid_dict():
    df = _make_duration_df()
    dur_model = train_duration_model(df)
    assert isinstance(dur_model, dict)
    assert set(dur_model.keys()) == {"pipeline", "kind", "low_thresh", "high_thresh"}
    assert dur_model["kind"] in ("classifier", "regressor", "baseline")
    assert dur_model["low_thresh"] < dur_model["high_thresh"]


def test_predict_duration_returns_valid_label():
    df = _make_duration_df()
    dur_model = train_duration_model(df)
    features = {
        "event_cause":           "public_event",
        "event_type":            "planned",
        "corridor":              "CBD 2",
        "zone":                  "Central Zone",
        "police_station":        "Cubbon Park",
        "hour_band":             "morning",
        "priority":              "High",
        "junction":              "unknown",
        "hour_of_day":           9,
        "day_of_week":           1,
        "requires_road_closure": False,
        "month":                 6,
        "is_weekend":            0,
    }
    result = predict_duration(dur_model, features)
    assert result in ("SHORT", "MEDIUM", "LONG")


def test_predict_duration_handles_unseen_corridor():
    """Model must not crash on a corridor not seen during training."""
    df = _make_duration_df()
    dur_model = train_duration_model(df)
    features = {
        "event_cause":           "accident",
        "event_type":            "unplanned",
        "corridor":              "UNSEEN_CORRIDOR",
        "zone":                  "Unknown Zone",
        "police_station":        "Unknown Station",
        "hour_band":             "night",
        "priority":              "Low",
        "junction":              "unknown",
        "hour_of_day":           23,
        "day_of_week":           6,
        "requires_road_closure": True,
        "month":                 1,
        "is_weekend":            1,
    }
    result = predict_duration(dur_model, features)
    assert result in ("SHORT", "MEDIUM", "LONG")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_duration_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.duration_model'`

- [ ] **Step 3: Create `src/duration_model.py`**

```python
# src/duration_model.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import lightgbm as lgb

from src.model import CAT_COLS, NUM_COLS

_DURATION_FEATURES = CAT_COLS + NUM_COLS  # 13 features, same as severity model


def _to_float(X):
    return X.astype(float)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", FunctionTransformer(_to_float), NUM_COLS),
    ])


def duration_tertile_thresholds(train_df: pd.DataFrame) -> tuple:
    """Returns (low_thresh, high_thresh) from training duration_h distribution."""
    valid = train_df["duration_h"].dropna()
    return float(valid.quantile(1 / 3)), float(valid.quantile(2 / 3))


def _bucket(values, low_thresh: float, high_thresh: float) -> list:
    result = []
    for v in values:
        if v <= low_thresh:
            result.append("SHORT")
        elif v <= high_thresh:
            result.append("MEDIUM")
        else:
            result.append("LONG")
    return result


def compute_duration_labels(df: pd.DataFrame, low_thresh: float, high_thresh: float) -> pd.Series:
    """Classify duration_h into SHORT/MEDIUM/LONG using training tertile thresholds."""
    def _label(v):
        if pd.isna(v):
            return np.nan
        return _bucket([v], low_thresh, high_thresh)[0]
    return df["duration_h"].map(_label)


def train_duration_model(train_df: pd.DataFrame) -> dict:
    """
    Benchmark three models on valid-duration rows, pick the winner, refit on full training data.
    Returns dict with keys: pipeline, kind, low_thresh, high_thresh.
    kind is one of: 'classifier', 'regressor', 'baseline'.
    """
    valid = train_df.dropna(subset=["duration_h"]).copy()
    low_thresh, high_thresh = duration_tertile_thresholds(valid)
    valid["_dur_label"] = compute_duration_labels(valid, low_thresh, high_thresh)
    valid = valid.dropna(subset=["_dur_label"])

    X        = valid[_DURATION_FEATURES]
    y_label  = valid["_dur_label"]
    y_log    = np.log1p(valid["duration_h"])
    y_dur_h  = valid["duration_h"]

    X_tr, X_te, ylab_tr, ylab_te, ylog_tr, ylog_te, ydur_tr, ydur_te = train_test_split(
        X, y_label, y_log, y_dur_h, test_size=0.2, random_state=42
    )

    # Model A: LGBMClassifier (tertile labels)
    pipe_a = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )),
    ])
    pipe_a.fit(X_tr, ylab_tr)
    f1_a = f1_score(ylab_te, pipe_a.predict(X_te), average="macro", zero_division=0)

    # Model B: LGBMRegressor (log1p target)
    pipe_b = Pipeline([
        ("pre", _make_preprocessor()),
        ("reg", lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
    ])
    pipe_b.fit(X_tr, ylog_tr)
    mae_b = mean_absolute_error(ydur_te, np.expm1(pipe_b.predict(X_te)))

    # Model C: ExtraTreesRegressor (log1p target)
    pipe_c = Pipeline([
        ("pre", _make_preprocessor()),
        ("reg", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipe_c.fit(X_tr, ylog_tr)
    mae_c = mean_absolute_error(ydur_te, np.expm1(pipe_c.predict(X_te)))

    # Winner selection — refit on full data
    if f1_a > 0.45:
        winner = Pipeline([
            ("pre", _make_preprocessor()),
            ("clf", lgb.LGBMClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
            )),
        ])
        winner.fit(X, y_label)
        kind = "classifier"
    elif min(mae_b, mae_c) <= 2.0:
        if mae_b <= mae_c:
            winner = Pipeline([
                ("pre", _make_preprocessor()),
                ("reg", lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
            ])
            winner.fit(X, y_log)
        else:
            winner = Pipeline([
                ("pre", _make_preprocessor()),
                ("reg", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ])
            winner.fit(X, y_log)
        kind = "regressor"
    else:
        winner = str(y_label.mode()[0])  # most-frequent label as string
        kind = "baseline"

    return {
        "pipeline":    winner,
        "kind":        kind,
        "low_thresh":  low_thresh,
        "high_thresh": high_thresh,
    }


def predict_duration(dur_model: dict, features: dict) -> str:
    """Returns 'SHORT', 'MEDIUM', or 'LONG'."""
    if dur_model["kind"] == "baseline":
        return dur_model["pipeline"]  # already the most-frequent label string

    X = pd.DataFrame([features])[_DURATION_FEATURES]
    pipeline = dur_model["pipeline"]

    if dur_model["kind"] == "classifier":
        return str(pipeline.predict(X)[0])

    # regressor: back-transform log1p, then bucket
    log_pred = float(pipeline.predict(X)[0])
    dur_pred = float(np.expm1(log_pred))
    return _bucket([dur_pred], dur_model["low_thresh"], dur_model["high_thresh"])[0]
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all 41 tests pass (35 from Task 3 + 6 new duration model tests)

- [ ] **Step 5: Commit**

```
git add src/duration_model.py tests/test_duration_model.py
git commit -m "feat: duration forecasting model — benchmark LGBMClassifier/Regressor/ExtraTrees, expose predict_duration"
```

---

## Task 5: Wire All Three Innovations into `app.py`

**Files:**
- Modify: `app.py` — import duration_model, train in `load_and_train()`, pass lat/lng/hb, display result

**Interfaces:**
- Consumes from Task 2: `barricade_positions(train_df, corridor, lat, lng)` — `lat`, `lng` already in scope
- Consumes from Task 3: `get_diversions(diversion_graph, corridor, hb)` — `hb` already in scope
- Consumes from Task 4: `train_duration_model`, `predict_duration`, `duration_tertile_thresholds`, `compute_duration_labels`

- [ ] **Step 1: No test file to write** — functional validation is done by running the app (Step 4). The existing 41 tests already cover all changed modules.

- [ ] **Step 2: Edit `app.py`**

**2a — Add import** at the top (after the existing `from src.model import ...` line):

```python
from src.duration_model import (
    duration_tertile_thresholds, compute_duration_labels,
    train_duration_model, predict_duration,
)
```

**2b — Inside `load_and_train()`, after severity labeling** (after the `for split in [train_df, val_df, test_df]: split["severity"] = ...` block), add:

```python
    low_d, high_d = duration_tertile_thresholds(train_df)
    train_df["duration_label"] = compute_duration_labels(train_df, low_d, high_d)
    dur_model = train_duration_model(train_df)
```

**2c — Add `dur_model` to the `return` dict** in `load_and_train()`:

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
    }
```

**2d — After `state = load_and_train()`, add**:

```python
dur_model = state["dur_model"]
```

**2e — In the form submission block**, replace the two old calls:

Old:
```python
        barricades = barricade_positions(train_df, corridor, top_n=4)
        ...
        diversions = get_diversions(diversion_graph, corridor)
```

New:
```python
        barricades = barricade_positions(train_df, corridor, lat, lng)
        n_adj      = min(3, len(barricades))
        officers   = officer_count(severity, n_adjacent_junctions=n_adj)
        diversions = get_diversions(diversion_graph, corridor, hb)
        duration   = predict_duration(dur_model, features)
```

**2f — Add `"duration"` to `st.session_state.result_data`**:

```python
        st.session_state.result_data = {
            "event_name": event_name,
            "corridor":   corridor,
            "severity":   severity,
            "confidence": confidence,
            "officers":   officers,
            "barricades": barricades,
            "diversions": diversions,
            "neighbors":  neighbors,
            "fmap":       fmap,
            "duration":   duration,
        }
```

**2g — In the results screen, in the `with left:` block**, add the duration display after the `st.caption(...)` line and before `st.markdown("---")`:

```python
        # Duration forecast
        _low_min  = round(state["dur_model"]["low_thresh"] * 60 / 5) * 5
        _high_min = round(state["dur_model"]["high_thresh"] * 60 / 5) * 5
        _DUR_LABELS = {
            "SHORT":  f"SHORT (<{_low_min} min)",
            "MEDIUM": f"MEDIUM ({_low_min}–{_high_min} min)",
            "LONG":   f"LONG (>{_high_min} min)",
        }
        _dur = r.get("duration", "N/A")
        st.markdown(f"**Duration Forecast:** {_DUR_LABELS.get(_dur, _dur)}")
```

**2h — Add `"Duration"` row to the export CSV** (in the `export_rows` list):

```python
    export_rows = [
        ("Event",         r["event_name"]),
        ("Corridor",      r["corridor"]),
        ("Severity",      severity),
        ("Confidence",    f"{conf_pct:.0f}%"),
        ("Duration",      r.get("duration", "N/A")),
        ("Officers min",  str(officers["total_min"])),
        ("Officers max",  str(officers["total_max"])),
        ("Barricades",    "; ".join(barricades) if barricades else "None"),
        ("Diversions",    "; ".join(diversions) if diversions else "None"),
    ]
```

- [ ] **Step 3: Run all tests**

```
pytest tests/ -v
```

Expected: all 41 tests pass

- [ ] **Step 4: Run the Streamlit app and verify**

```
streamlit run app.py
```

Check:
- App loads without errors
- Submit the form with any corridor/event
- Results screen shows **Duration Forecast: SHORT/MEDIUM/LONG (<N min)**
- Barricades and diversions still display correctly
- Map still renders without blinking

- [ ] **Step 5: Commit**

```
git add app.py
git commit -m "feat: wire geospatial barricades, time-banded diversions, and duration forecast into app.py"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `duration_h` in pipeline.py | Task 1 |
| `_haversine_km` helper + geospatial filter + fallback chain | Task 2 |
| `barricade_positions(train_df, corridor, event_lat, event_lng, radius_km, top_n)` | Task 2 |
| `build_diversion_graph` keys by `(corridor, hour_band)` | Task 3 |
| `get_diversions(graph, corridor, hour_band)` with 3-level fallback | Task 3 |
| Benchmark A (LGBMClassifier), B (LGBMRegressor), C (ExtraTrees) | Task 4 |
| Winner selection rule (F1 > 0.45 → A, else MAE comparison) | Task 4 |
| Fallback baseline (most-frequent label) | Task 4 |
| `predict_duration` returns SHORT/MEDIUM/LONG | Task 4 |
| `app.py` passes `lat`, `lng` to `barricade_positions` | Task 5 |
| `app.py` passes `hb` to `get_diversions` | Task 5 |
| Duration displayed with human-readable range on results screen | Task 5 |
| `src/model.py` untouched | All tasks — confirmed |
| 32 existing tests preserved | All tasks |

**Placeholder scan:** None found — all steps have complete code.

**Type consistency check:**
- `barricade_positions` new args `event_lat`, `event_lng` are `float` in impl and tests ✓
- `get_diversions` new arg `hour_band` is `str` in impl and tests ✓
- `train_duration_model` returns `dict` with keys `pipeline`, `kind`, `low_thresh`, `high_thresh` — used consistently in Task 4 tests and Task 5 app ✓
- `predict_duration(dur_model, features)` — `dur_model` is the dict from `train_duration_model` ✓
- `_DURATION_FEATURES = CAT_COLS + NUM_COLS` — 13 features, matching app.py `features` dict ✓
