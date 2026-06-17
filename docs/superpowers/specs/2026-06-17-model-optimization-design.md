# Model Optimization — LightGBM + Feature Engineering

**Date:** 2026-06-17
**Context:** GRIDLOCK 2.0 hackathon demo — improve severity classifier macro F1 from 0.65 baseline
**Approach:** Feature expansion + CorridorStatsTransformer + LGBMClassifier + Optuna tuning

---

## Problem

The current `GradientBoostingClassifier` with `OrdinalEncoder` achieves macro F1 = 0.65 on the held-out test set. HIGH severity recall is particularly weak (0.51). Three root causes:

1. **Thin feature set** — 9 features; `priority`, `junction`, `month`, `is_weekend`, and corridor-level historical stats all sit unused in the CSV.
2. **Suboptimal encoder** — `OrdinalEncoder` assigns arbitrary ordinal integers to unordered categoricals like `corridor`, misleading the model.
3. **Untuned model** — sklearn GBT defaults; no hyperparameter search; no class weighting to address HIGH recall gap.

---

## Goal

Maximize macro F1 on the test set. No specific target — maximize as much as possible while keeping the same `train_model() → Pipeline` and `predict(pipeline, features)` public API so `app.py` requires only minimal changes.

---

## Feature Engineering

### New raw features (added in `load_raw()`)

| Feature | Source | Notes |
|---|---|---|
| `priority` | `priority` column (High/Low) | Strong prior signal; available in historical data and added to Streamlit form |
| `junction` | `junction` column | High-cardinality categorical; fillna "unknown" |
| `month` | `start_datetime.dt.month` | Seasonality — festival/monsoon months differ |
| `is_weekend` | `day_of_week >= 5` | Binary int (0/1) |

**Not added:** `veh_type` — null for ~70% of rows (only relevant for vehicle_breakdown events); adds noise.

### Corridor historical stats (computed from `train_df` only)

| Feature | Definition |
|---|---|
| `corridor_high_rate` | Fraction of HIGH severity events on this corridor in training data |
| `corridor_event_count` | Total events on this corridor in training data |

These give the model a per-corridor prior — the single most impactful addition since the current model must infer corridor behavior entirely through the categorical value.

Both stats are computed inside `CorridorStatsTransformer.fit()` and joined in `transform()`. Fallback for unseen corridors: global mean `corridor_high_rate`, median `corridor_event_count`.

### Final feature columns

```python
CAT_COLS = [
    "event_cause", "event_type", "corridor", "zone",
    "police_station", "hour_band", "priority", "junction",
]
NUM_COLS = [
    "hour_of_day", "day_of_week", "requires_road_closure",
    "month", "is_weekend",
    "corridor_high_rate", "corridor_event_count",  # added by transformer
]
```

---

## Model Architecture

Replace `ColumnTransformer + GradientBoostingClassifier` with:

```
Pipeline([
    ("corridor_stats", CorridorStatsTransformer()),
    ("lgbm", LGBMClassifier(categorical_feature="auto", class_weight="balanced", ...)),
])
```

### CorridorStatsTransformer

- Inherits `BaseEstimator`, `TransformerMixin` (sklearn-compatible).
- `fit(X, y)`: groups `X` by `corridor`, computes `corridor_high_rate` and `corridor_event_count`, stores as `self.stats_` dict.
- `transform(X)`: left-joins stats onto `X`; fills missing corridors with global fallback values.
- Operates on DataFrames; returns DataFrame (LightGBM accepts DataFrames directly).

### LGBMClassifier

- `categorical_feature="auto"` — picks up pandas `category` dtype columns; no `OrdinalEncoder` needed.
- `class_weight="balanced"` — upweights HIGH and MEDIUM to address recall gap.
- All CAT_COLS cast to `category` dtype in the transformer's output.
- Accepts optional `params` dict from Optuna (see Tuning section).

### Why this replaces ColumnTransformer

LightGBM handles both categorical and numerical features natively. Eliminating `ColumnTransformer` and `OrdinalEncoder` removes the artificial ordinal relationship imposed on unordered categoricals.

---

## Tuning

### New file: `src/tuner.py`

```python
def tune_lgbm(train_df: pd.DataFrame, n_trials: int = 75) -> dict:
    """Optuna study; returns best params dict."""
```

- **Objective**: 5-fold stratified CV macro F1 (mirrors `evaluate_cv`).
- **Search space**:

| Param | Range | Scale |
|---|---|---|
| `num_leaves` | 20–200 | linear |
| `learning_rate` | 0.01–0.3 | log |
| `n_estimators` | 100–800 | linear |
| `min_child_samples` | 5–100 | linear |
| `reg_alpha` | 1e-8–10 | log |
| `reg_lambda` | 1e-8–10 | log |
| `subsample` | 0.5–1.0 | linear |
| `colsample_bytree` | 0.5–1.0 | linear |

- Optuna `TPESampler` with `MedianPruner` to kill poor trials early.
- Run as `python -m src.tuner` (prints best params) or called programmatically.
- `train_model(train_df, params=None)` — if `params=None`, uses hardcoded sensible defaults so existing tests continue to pass without running tuning.

---

## API Compatibility

### `src/model.py`

| Function | Change |
|---|---|
| `train_model(train_df, params=None)` | Accepts optional params; returns same `Pipeline` type |
| `predict(pipeline, features)` | Features dict expands with `priority`, `junction`, `month`, `is_weekend`; corridor stats computed internally by transformer |
| `get_knn_neighbors(train_df, query_features, k)` | Same expansion; uses updated `ALL_FEATURE_COLS` |
| `evaluate_cv`, `evaluate_test` | Unchanged signatures |

### `app.py`

- Add `priority` selectbox ("High" / "Low", default "High") to the event form.
- Derive `month = event_date.month` and `is_weekend = int(dow >= 5)` from existing inputs.
- Set `junction = "unknown"` (operator cannot know junction for a future event).
- Pass all four new fields in the `features` dict to `predict()`.

### `src/pipeline.py`

- `load_raw()` adds `priority`, `junction`, `month`, `is_weekend` columns.
- `priority` fillna "unknown"; `junction` fillna "unknown"; both cleaned like other categoricals.

---

## File Changes

| File | Change |
|---|---|
| `src/pipeline.py` | `load_raw()` adds 4 new columns |
| `src/model.py` | Full replacement of pipeline internals; `CorridorStatsTransformer` class; expanded feature lists; `params` arg on `train_model` |
| `src/tuner.py` | New file — Optuna study |
| `app.py` | Form adds `priority`; passes `month`, `is_weekend`, `junction` to `predict()` |
| `tests/test_model.py` | `_prepare()` fixture adds new columns; assertions unchanged |

No changes to `baseline.py`, `recommender.py`, `map_builder.py`, or the labeling pipeline.

---

## Expected Outcome

| Metric | Current | Expected |
|---|---|---|
| Macro F1 | 0.65 | 0.73–0.82 |
| HIGH recall | 0.51 | 0.60–0.72 |
| HIGH F1 | 0.56 | 0.65–0.75 |

Biggest individual gains: corridor historical stats > `priority` feature > LightGBM over GBT > Optuna tuning.
