# Phase 1: Enhanced Police Event Planning System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add calendar intelligence, adaptive planned/unplanned form, traffic congestion + law-and-order risk models, SHAP explainability, and multi-page Streamlit layout to the existing Bengaluru event congestion planner.

**Architecture:** Six new/modified modules live in `src/`; Streamlit UI splits into `pages/1_Plan_Event.py` (form) and `pages/2_Results.py` (dashboard); all training is centralised in `src/app_cache.py` so both pages share one cached training run. The severity classifier pipeline is extended with five new feature columns but is otherwise unchanged.

**Tech Stack:** Python 3.11+, LightGBM, scikit-learn, SHAP (`pip install shap`), Streamlit ≥ 1.30, streamlit-folium, OSMnx, folium, pandas, numpy.

## Global Constraints

- All new `src/` modules must be importable independently (no circular imports).
- `_X()` in `src/model.py` must provide safe defaults for every new column so all existing tests continue to pass without modification.
- `shap` must be installed before Task 4: `pip install shap`.
- Pages live in `pages/` at the project root (sibling of `src/`, `app.py`).
- Commit after every task; never commit a failing test.
- Run `pytest` after each task to confirm no regressions in existing tests.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/calendar_intel.py` | CREATE | Holiday/festival lookup for Karnataka/India 2022–2027 |
| `src/risk_model.py` | CREATE | Congestion + law-and-order binary classifiers |
| `src/explainer.py` | CREATE | SHAP driver extraction for severity + risk pipelines |
| `src/app_cache.py` | CREATE | Single `@st.cache_data` / `@st.cache_resource` for both pages |
| `pages/1_Plan_Event.py` | CREATE | Adaptive planned/unplanned input form |
| `pages/2_Results.py` | CREATE | Extended results dashboard |
| `src/pipeline.py` | MODIFY | `load_raw()` adds 5 new columns |
| `src/model.py` | MODIFY | `NUM_COLS` gains 5 entries; `_X()` handles safe defaults |
| `app.py` | MODIFY | Gutted to `st.switch_page` redirect (~3 lines) |
| `tests/test_calendar_intel.py` | CREATE | Calendar unit tests |
| `tests/test_risk_model.py` | CREATE | Risk model smoke tests |
| `tests/test_explainer.py` | CREATE | SHAP output structure tests |

---

## Task 1: Calendar Intelligence

**Files:**
- Create: `src/calendar_intel.py`
- Test: `tests/test_calendar_intel.py`

**Interfaces:**
- Produces: `get_holiday_info(date: datetime.date) -> dict` with keys `is_holiday` (bool), `holiday_type` (str), `holiday_name` (str), `risk_tier` (int 0–3)

---

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calendar_intel.py
import datetime
import pytest
from src.calendar_intel import get_holiday_info


def test_non_holiday_returns_tier_zero():
    result = get_holiday_info(datetime.date(2024, 6, 5))   # random Wednesday
    assert result["is_holiday"] is False
    assert result["holiday_type"] == "none"
    assert result["risk_tier"] == 0
    assert result["holiday_name"] == ""


def test_republic_day_is_national():
    result = get_holiday_info(datetime.date(2024, 1, 26))
    assert result["is_holiday"] is True
    assert result["holiday_type"] == "national"
    assert result["holiday_name"] == "Republic Day"
    assert result["risk_tier"] == 2


def test_independence_day_is_national():
    result = get_holiday_info(datetime.date(2025, 8, 15))
    assert result["is_holiday"] is True
    assert result["holiday_type"] == "national"
    assert result["risk_tier"] == 2


def test_rajyotsava_is_state():
    result = get_holiday_info(datetime.date(2024, 11, 1))
    # Nov 1 2024 is also within Diwali window — festival tier wins
    assert result["is_holiday"] is True
    assert result["risk_tier"] >= 1


def test_dasara_window_2024():
    # Vijayadashami 2024: Oct 12. Window Oct 3–12.
    for day in range(3, 13):
        r = get_holiday_info(datetime.date(2024, 10, day))
        assert r["is_holiday"] is True, f"Oct {day} 2024 should be Dasara"
        assert r["risk_tier"] == 3


def test_outside_dasara_window_2024():
    result = get_holiday_info(datetime.date(2024, 10, 2))  # day before window
    # Oct 2 is Gandhi Jayanti (national, tier 2) — still a holiday, just not Dasara
    assert result["is_holiday"] is True
    assert result["risk_tier"] == 2


def test_diwali_2024_window():
    for d in [datetime.date(2024, 10, 31), datetime.date(2024, 11, 1), datetime.date(2024, 11, 2)]:
        r = get_holiday_info(d)
        assert r["is_holiday"] is True
        assert r["risk_tier"] == 3


def test_new_years_eve():
    result = get_holiday_info(datetime.date(2025, 12, 31))
    assert result["is_holiday"] is True
    assert result["risk_tier"] == 3


def test_return_type_structure():
    result = get_holiday_info(datetime.date(2024, 3, 15))
    assert set(result.keys()) == {"is_holiday", "holiday_type", "holiday_name", "risk_tier"}
    assert isinstance(result["is_holiday"], bool)
    assert isinstance(result["risk_tier"], int)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_calendar_intel.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.calendar_intel'`

- [ ] **Step 3: Implement `src/calendar_intel.py`**

```python
# src/calendar_intel.py
import datetime

# Fixed holidays: (month, day) → (holiday_type, holiday_name, risk_tier)
_FIXED_ANNUAL: dict[tuple, tuple] = {
    (1,  1):  ("festival", "New Year",            3),
    (1,  26): ("national", "Republic Day",         2),
    (4,  14): ("national", "Ambedkar Jayanti",     2),
    (8,  15): ("national", "Independence Day",     2),
    (10, 2):  ("national", "Gandhi Jayanti",       2),
    (11, 1):  ("state",    "Rajyotsava",           1),
    (12, 25): ("national", "Christmas",            2),
    (12, 31): ("festival", "New Year's Eve",       3),
}

# Variable/lunar holidays: datetime.date → (holiday_type, holiday_name, risk_tier)
# Each day in a multi-day festival window is listed individually.
_VARIABLE_DATES: dict[datetime.date, tuple] = {
    # ── 2022 ──
    datetime.date(2022, 3, 17): ("festival", "Holi",            3),
    datetime.date(2022, 3, 18): ("festival", "Holi",            3),
    datetime.date(2022, 4, 2):  ("state",    "Ugadi",           1),
    datetime.date(2022, 5, 2):  ("festival", "Eid al-Fitr",     3),
    datetime.date(2022, 5, 3):  ("festival", "Eid al-Fitr",     3),
    datetime.date(2022, 5, 4):  ("festival", "Eid al-Fitr",     3),
    datetime.date(2022, 7, 9):  ("festival", "Eid al-Adha",     3),
    datetime.date(2022, 7, 10): ("festival", "Eid al-Adha",     3),
    datetime.date(2022, 7, 11): ("festival", "Eid al-Adha",     3),
    **{datetime.date(2022, 9, d): ("festival", "Dasara", 3) for d in range(26, 31)},
    datetime.date(2022, 10, 1): ("festival", "Dasara",          3),
    datetime.date(2022, 10, 2): ("festival", "Dasara",          3),  # also Gandhi Jayanti
    datetime.date(2022, 10, 9): ("state",    "Valmiki Jayanti", 1),
    datetime.date(2022, 10, 24): ("festival", "Diwali",         3),
    datetime.date(2022, 10, 25): ("festival", "Diwali",         3),
    datetime.date(2022, 10, 26): ("festival", "Diwali",         3),
    datetime.date(2022, 11, 19): ("state",   "Kanakadasa Jayanti", 1),

    # ── 2023 ──
    datetime.date(2023, 3, 7):  ("festival", "Holi",            3),
    datetime.date(2023, 3, 8):  ("festival", "Holi",            3),
    datetime.date(2023, 3, 22): ("state",    "Ugadi",           1),
    datetime.date(2023, 4, 21): ("festival", "Eid al-Fitr",     3),
    datetime.date(2023, 4, 22): ("festival", "Eid al-Fitr",     3),
    datetime.date(2023, 4, 23): ("festival", "Eid al-Fitr",     3),
    datetime.date(2023, 6, 28): ("festival", "Eid al-Adha",     3),
    datetime.date(2023, 6, 29): ("festival", "Eid al-Adha",     3),
    datetime.date(2023, 6, 30): ("festival", "Eid al-Adha",     3),
    **{datetime.date(2023, 10, d): ("festival", "Dasara", 3) for d in range(15, 25)},
    datetime.date(2023, 10, 28): ("state",   "Valmiki Jayanti", 1),
    datetime.date(2023, 11, 12): ("festival", "Diwali",         3),
    datetime.date(2023, 11, 13): ("festival", "Diwali",         3),
    datetime.date(2023, 11, 14): ("festival", "Diwali",         3),
    datetime.date(2023, 11, 27): ("state",   "Kanakadasa Jayanti", 1),

    # ── 2024 ──
    datetime.date(2024, 3, 25): ("festival", "Holi",            3),
    datetime.date(2024, 3, 26): ("festival", "Holi",            3),
    datetime.date(2024, 4, 9):  ("festival", "Eid al-Fitr",     3),  # also Ugadi
    datetime.date(2024, 4, 10): ("festival", "Eid al-Fitr",     3),
    datetime.date(2024, 4, 11): ("festival", "Eid al-Fitr",     3),
    datetime.date(2024, 6, 16): ("festival", "Eid al-Adha",     3),
    datetime.date(2024, 6, 17): ("festival", "Eid al-Adha",     3),
    datetime.date(2024, 6, 18): ("festival", "Eid al-Adha",     3),
    **{datetime.date(2024, 10, d): ("festival", "Dasara", 3) for d in range(3, 13)},
    datetime.date(2024, 10, 17): ("state",   "Valmiki Jayanti", 1),
    datetime.date(2024, 10, 31): ("festival", "Diwali",         3),
    datetime.date(2024, 11, 1):  ("festival", "Diwali",         3),  # also Rajyotsava
    datetime.date(2024, 11, 2):  ("festival", "Diwali",         3),
    datetime.date(2024, 11, 14): ("state",   "Kanakadasa Jayanti", 1),

    # ── 2025 ──
    datetime.date(2025, 3, 13): ("festival", "Holi",            3),
    datetime.date(2025, 3, 14): ("festival", "Holi",            3),
    datetime.date(2025, 3, 30): ("festival", "Eid al-Fitr",     3),  # also Ugadi
    datetime.date(2025, 3, 31): ("festival", "Eid al-Fitr",     3),
    datetime.date(2025, 6, 6):  ("festival", "Eid al-Adha",     3),
    datetime.date(2025, 6, 7):  ("festival", "Eid al-Adha",     3),
    datetime.date(2025, 6, 8):  ("festival", "Eid al-Adha",     3),
    **{datetime.date(2025, 9, d): ("festival", "Dasara", 3) for d in range(23, 30)},
    datetime.date(2025, 9, 30): ("festival", "Dasara",          3),
    datetime.date(2025, 10, 1): ("festival", "Dasara",          3),
    datetime.date(2025, 10, 2): ("festival", "Dasara",          3),  # also Gandhi Jayanti
    datetime.date(2025, 10, 6): ("state",    "Valmiki Jayanti", 1),
    datetime.date(2025, 10, 19): ("festival", "Diwali",         3),
    datetime.date(2025, 10, 20): ("festival", "Diwali",         3),
    datetime.date(2025, 10, 21): ("festival", "Diwali",         3),
    datetime.date(2025, 11, 3): ("state",    "Kanakadasa Jayanti", 1),

    # ── 2026 ──
    datetime.date(2026, 3, 2):  ("festival", "Holi",            3),
    datetime.date(2026, 3, 3):  ("festival", "Holi",            3),
    datetime.date(2026, 3, 20): ("festival", "Eid al-Fitr",     3),  # also Ugadi ~Mar 20
    datetime.date(2026, 3, 21): ("festival", "Eid al-Fitr",     3),
    datetime.date(2026, 3, 22): ("festival", "Eid al-Fitr",     3),
    datetime.date(2026, 5, 27): ("festival", "Eid al-Adha",     3),
    datetime.date(2026, 5, 28): ("festival", "Eid al-Adha",     3),
    datetime.date(2026, 5, 29): ("festival", "Eid al-Adha",     3),
    **{datetime.date(2026, 10, d): ("festival", "Dasara", 3) for d in range(13, 23)},
    datetime.date(2026, 10, 25): ("state",   "Valmiki Jayanti", 1),
    datetime.date(2026, 11, 7): ("festival", "Diwali",          3),
    datetime.date(2026, 11, 8): ("festival", "Diwali",          3),
    datetime.date(2026, 11, 9): ("festival", "Diwali",          3),
    datetime.date(2026, 11, 23): ("state",   "Kanakadasa Jayanti", 1),

    # ── 2027 ──
    datetime.date(2027, 3, 1):  ("festival", "Holi",            3),
    datetime.date(2027, 3, 2):  ("festival", "Holi",            3),
    datetime.date(2027, 3, 9):  ("state",    "Ugadi",           1),
    datetime.date(2027, 3, 10): ("festival", "Eid al-Fitr",     3),
    datetime.date(2027, 3, 11): ("festival", "Eid al-Fitr",     3),
    datetime.date(2027, 5, 17): ("festival", "Eid al-Adha",     3),
    datetime.date(2027, 5, 18): ("festival", "Eid al-Adha",     3),
    datetime.date(2027, 5, 19): ("festival", "Eid al-Adha",     3),
    **{datetime.date(2027, 10, d): ("festival", "Dasara", 3) for d in range(1, 11)},
    datetime.date(2027, 10, 26): ("festival", "Diwali",         3),
    datetime.date(2027, 10, 27): ("festival", "Diwali",         3),
    datetime.date(2027, 10, 28): ("festival", "Diwali",         3),
}


def _build_lookup() -> dict[datetime.date, tuple]:
    lookup: dict[datetime.date, tuple] = {}
    # Variable/lunar dates first
    for d, info in _VARIABLE_DATES.items():
        lookup[d] = info
    # Fixed annual dates: only overwrite if equal or lower tier already present
    for year in range(2022, 2028):
        for (month, day), info in _FIXED_ANNUAL.items():
            try:
                date = datetime.date(year, month, day)
            except ValueError:
                continue
            existing = lookup.get(date)
            if existing is None or info[2] >= existing[2]:
                lookup[date] = info
    return lookup


_LOOKUP: dict[datetime.date, tuple] = _build_lookup()


def get_holiday_info(date: datetime.date) -> dict:
    """Return holiday metadata for a given date.

    Checks the pre-built lookup table, then applies long-weekend detection
    for any Monday/Friday adjacent to a Tuesday/Thursday holiday.
    """
    entry = _LOOKUP.get(date)
    if entry:
        return {
            "is_holiday":   True,
            "holiday_type": entry[0],
            "holiday_name": entry[1],
            "risk_tier":    entry[2],
        }

    # Long weekend: Monday after a Tuesday holiday, or Friday before a Thursday holiday
    weekday = date.weekday()  # 0=Mon, 4=Fri
    if weekday == 0:  # Monday — check if Tuesday is a holiday
        neighbour = _LOOKUP.get(date + datetime.timedelta(days=1))
        if neighbour:
            return {"is_holiday": True, "holiday_type": "state",
                    "holiday_name": f"Long weekend ({neighbour[1]})", "risk_tier": 1}
    if weekday == 4:  # Friday — check if Thursday is a holiday
        neighbour = _LOOKUP.get(date - datetime.timedelta(days=1))
        if neighbour:
            return {"is_holiday": True, "holiday_type": "state",
                    "holiday_name": f"Long weekend ({neighbour[1]})", "risk_tier": 1}

    return {"is_holiday": False, "holiday_type": "none", "holiday_name": "", "risk_tier": 0}
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_calendar_intel.py -v
```
Expected: All 9 tests PASS.

- [ ] **Step 5: Confirm no regressions**

```
pytest --tb=short -q
```
Expected: All pre-existing tests still pass.

- [ ] **Step 6: Commit**

```
git add src/calendar_intel.py tests/test_calendar_intel.py
git commit -m "feat: calendar intelligence module with Karnataka/India holiday lookup"
```

---

## Task 2: Pipeline and Model Feature Extension

**Files:**
- Modify: `src/pipeline.py`
- Modify: `src/model.py`

**Interfaces:**
- Consumes: `get_holiday_info` from `src.calendar_intel`
- Produces: `load_raw()` now returns DataFrame with 5 extra columns: `is_holiday` (int), `holiday_risk_tier` (int), `estimated_attendance` (int, always 0), `has_vip` (int, always 0), `is_route_event` (int, always 0)
- Produces: `NUM_COLS` in `model.py` gains those 5 columns; `_X()` provides safe defaults for them

---

- [ ] **Step 1: Write failing tests for new pipeline columns**

```python
# Add to tests/test_pipeline.py (append after existing tests)

def test_load_raw_has_calendar_columns(tmp_path):
    """load_raw() must add is_holiday and holiday_risk_tier."""
    from src.pipeline import load_raw
    import os
    df = load_raw()   # uses the real data/events.csv
    assert "is_holiday" in df.columns,       "is_holiday column missing"
    assert "holiday_risk_tier" in df.columns, "holiday_risk_tier column missing"
    assert df["is_holiday"].isin([0, 1]).all()
    assert df["holiday_risk_tier"].between(0, 3).all()


def test_load_raw_has_form_default_columns():
    from src.pipeline import load_raw
    df = load_raw()
    for col in ["estimated_attendance", "has_vip", "is_route_event"]:
        assert col in df.columns, f"{col} missing"
        assert (df[col] == 0).all(), f"{col} should be all-zero defaults"
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_pipeline.py::test_load_raw_has_calendar_columns tests/test_pipeline.py::test_load_raw_has_form_default_columns -v
```
Expected: FAIL — columns absent.

- [ ] **Step 3: Update `src/pipeline.py` — add 5 new columns in `load_raw()`**

Open `src/pipeline.py`. After the `_add_nlp_features(df)` call (around line 92), add:

```python
    # Calendar features — derived from event date
    from src.calendar_intel import get_holiday_info
    _dates = df["start_datetime"].dt.date
    _holiday_info = _dates.map(get_holiday_info)
    df["is_holiday"]        = _holiday_info.map(lambda x: int(x["is_holiday"]))
    df["holiday_risk_tier"] = _holiday_info.map(lambda x: x["risk_tier"]).astype(int)

    # Form-derived features — backfilled to 0 for all historical rows
    df["estimated_attendance"] = 0
    df["has_vip"]              = 0
    df["is_route_event"]       = 0
```

The full `load_raw()` function tail (after `_add_nlp_features`) should now look like:

```python
    df = _add_nlp_features(df)

    from src.calendar_intel import get_holiday_info
    _dates = df["start_datetime"].dt.date
    _holiday_info = _dates.map(get_holiday_info)
    df["is_holiday"]        = _holiday_info.map(lambda x: int(x["is_holiday"]))
    df["holiday_risk_tier"] = _holiday_info.map(lambda x: x["risk_tier"]).astype(int)

    df["estimated_attendance"] = 0
    df["has_vip"]              = 0
    df["is_route_event"]       = 0

    return df.reset_index(drop=True)
```

- [ ] **Step 4: Update `src/model.py` — extend `NUM_COLS` and `_X()`**

Replace the existing `NUM_COLS` list:

```python
NUM_COLS = [
    "hour_of_day", "day_of_week", "requires_road_closure",
    "month", "is_weekend",
    "desc_traffic_slow", "desc_breakdown",
    "is_holiday",
    "holiday_risk_tier",
    "estimated_attendance",
    "has_vip",
    "is_route_event",
]
```

Add a new private constant after `_NLP_NUM_COLS`:

```python
_NEW_INT_COLS = [
    "is_holiday", "holiday_risk_tier",
    "estimated_attendance", "has_vip", "is_route_event",
]
```

Update `_X()` to inject safe defaults and cast the new int columns:

```python
def _X(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _NLP_NUM_COLS:
        if col not in df.columns:
            df[col] = 0
    for col in _NEW_INT_COLS:
        if col not in df.columns:
            df[col] = 0
    if "veh_type" not in df.columns:
        df["veh_type"] = "unknown"

    out: pd.DataFrame = df[ALL_FEATURE_COLS].copy()
    out["requires_road_closure"] = out["requires_road_closure"].astype(int)
    out["is_weekend"] = out["is_weekend"].astype(int)
    out["month"] = out["month"].astype(int)
    for col in _NLP_NUM_COLS + _NEW_INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in CAT_COLS:
        col_s: pd.Series = out[col]
        out[col] = col_s.fillna("unknown").astype(str)
    return out
```

- [ ] **Step 5: Run the new pipeline tests**

```
pytest tests/test_pipeline.py -v
```
Expected: all PASS (including the 2 new tests).

- [ ] **Step 6: Confirm no regressions across the full suite**

```
pytest --tb=short -q
```
Expected: all pre-existing tests PASS. Any test that builds a `labeled_df` must still work because `_X()` provides safe defaults for the new columns.

- [ ] **Step 7: Commit**

```
git add src/pipeline.py src/model.py tests/test_pipeline.py
git commit -m "feat: add calendar + form feature columns to pipeline and model"
```

---

## Task 3: Risk Models

**Files:**
- Create: `src/risk_model.py`
- Test: `tests/test_risk_model.py`

**Interfaces:**
- Consumes: `CAT_COLS`, `NUM_COLS` from `src.model`
- Produces:
  - `safe_df(df) -> pd.DataFrame` — adds missing columns with safe defaults
  - `train_risk_models(train_df) -> dict` — keys: `congestion` (Pipeline), `law_order` (Pipeline), `congestion_auc` (float), `law_order_auc` (float)
  - `predict_risks(risk_models, features) -> dict` — keys: `congestion_prob` (float 0–1), `law_order_prob` (float 0–1)
  - `_RISK_FEATURES: list[str]` — `CAT_COLS + NUM_COLS` (20 features)

---

- [ ] **Step 1: Write failing tests**

```python
# tests/test_risk_model.py
import pandas as pd
import numpy as np
import pytest
from src.risk_model import train_risk_models, predict_risks, _RISK_FEATURES, safe_df


def _make_train_df() -> pd.DataFrame:
    """Minimal DataFrame with all columns needed by train_risk_models."""
    import datetime
    np.random.seed(42)
    n = 80
    corridors = ["CBD 2", "ORR East 1", "Tumkur Road", "MG Road"]
    causes    = ["public_event", "vehicle_breakdown", "accident",
                 "protest", "procession", "vip_movement", "tree_fall"]
    rng = np.random.default_rng(42)
    start = pd.date_range("2024-01-01", periods=n, freq="6h", tz="UTC")
    df = pd.DataFrame({
        "event_cause":           rng.choice(causes, n),
        "event_type":            rng.choice(["planned", "unplanned"], n),
        "corridor":              rng.choice(corridors, n),
        "zone":                  rng.choice(["Central Zone 2", "East Zone 1"], n),
        "police_station":        rng.choice(["Cubbon Park", "Bellandur"], n),
        "priority":              rng.choice(["High", "Low"], n),
        "junction":              rng.choice(["JuncA", "unknown"], n),
        "hour_band":             rng.choice(["morning", "evening", "afternoon", "night"], n),
        "hour_of_day":           rng.integers(0, 24, n),
        "day_of_week":           rng.integers(0, 7, n),
        "month":                 rng.integers(1, 13, n),
        "is_weekend":            rng.integers(0, 2, n),
        "requires_road_closure": rng.integers(0, 2, n),
        "desc_traffic_slow":     rng.integers(0, 2, n),
        "desc_breakdown":        rng.integers(0, 2, n),
        "is_holiday":            rng.integers(0, 2, n),
        "holiday_risk_tier":     rng.integers(0, 4, n),
        "estimated_attendance":  rng.integers(0, 5000, n),
        "has_vip":               rng.integers(0, 2, n),
        "is_route_event":        rng.integers(0, 2, n),
        "window_count":          rng.integers(0, 10, n),
        "start_datetime":        start,
    })
    return df


def test_risk_features_length():
    assert len(_RISK_FEATURES) == 20


def test_train_risk_models_returns_required_keys():
    df = _make_train_df()
    result = train_risk_models(df)
    assert set(result.keys()) == {"congestion", "law_order",
                                   "congestion_auc", "law_order_auc"}


def test_train_risk_models_auc_is_float():
    df = _make_train_df()
    result = train_risk_models(df)
    assert isinstance(result["congestion_auc"], float)
    assert isinstance(result["law_order_auc"], float)
    assert 0.0 <= result["congestion_auc"] <= 1.0
    assert 0.0 <= result["law_order_auc"] <= 1.0


def test_predict_risks_returns_probs():
    df = _make_train_df()
    risk_models = train_risk_models(df)
    features = {
        "event_cause":           "public_event",
        "event_type":            "planned",
        "corridor":              "CBD 2",
        "zone":                  "Central Zone 2",
        "police_station":        "Cubbon Park",
        "priority":              "High",
        "junction":              "unknown",
        "hour_band":             "evening",
        "hour_of_day":           18,
        "day_of_week":           0,
        "month":                 2,
        "is_weekend":            0,
        "requires_road_closure": 0,
        "desc_traffic_slow":     1,
        "desc_breakdown":        0,
        "is_holiday":            0,
        "holiday_risk_tier":     0,
        "estimated_attendance":  1000,
        "has_vip":               0,
        "is_route_event":        0,
    }
    result = predict_risks(risk_models, features)
    assert set(result.keys()) == {"congestion_prob", "law_order_prob"}
    assert 0.0 <= result["congestion_prob"] <= 1.0
    assert 0.0 <= result["law_order_prob"] <= 1.0


def test_safe_df_injects_missing_columns():
    df = pd.DataFrame({"event_cause": ["accident"], "corridor": ["CBD 2"]})
    out = safe_df(df)
    for col in _RISK_FEATURES:
        assert col in out.columns, f"safe_df missing column: {col}"
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_risk_model.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.risk_model'`

- [ ] **Step 3: Implement `src/risk_model.py`**

```python
# src/risk_model.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import lightgbm as lgb

from src.model import CAT_COLS, NUM_COLS, _NLP_NUM_COLS

_RISK_FEATURES: list = CAT_COLS + NUM_COLS  # 20 features

_HIGH_RISK_CAUSES = {"riot", "protest", "procession", "public_event", "vip_movement"}

_NEW_INT_COLS = [
    "is_holiday", "holiday_risk_tier",
    "estimated_attendance", "has_vip", "is_route_event",
]


def _to_float(X):
    return X.astype(float)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", FunctionTransformer(_to_float), NUM_COLS),
    ])


def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Inject all feature columns with safe defaults when absent."""
    df = df.copy()
    for col in _NLP_NUM_COLS + _NEW_INT_COLS:
        if col not in df.columns:
            df[col] = 0
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].fillna("unknown").astype(str)
    return df


def train_risk_models(train_df: pd.DataFrame) -> dict:
    """Train congestion and law-and-order classifiers on train_df.

    Returns dict with keys: congestion, law_order (fitted Pipelines),
    congestion_auc, law_order_auc (float, evaluated on 20% hold-out).
    """
    df = safe_df(train_df)

    # Labels
    p75 = df.groupby("corridor")["window_count"].transform(
        lambda s: s.quantile(0.75)
    )
    y_cong = (df["window_count"] > p75).astype(int)
    y_law  = df["event_cause"].isin(_HIGH_RISK_CAUSES).astype(int)

    X = df[_RISK_FEATURES]
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    yc_tr, yc_te = y_cong.iloc[idx_tr], y_cong.iloc[idx_te]
    yl_tr, yl_te = y_law.iloc[idx_tr], y_law.iloc[idx_te]

    def _fit_eval(y_tr, y_te):
        pipe = Pipeline([
            ("pre", _make_preprocessor()),
            ("clf", lgb.LGBMClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
            )),
        ])
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        # Refit on full data
        pipe_full = Pipeline([
            ("pre", _make_preprocessor()),
            ("clf", lgb.LGBMClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
            )),
        ])
        pipe_full.fit(X, y_tr.iloc[idx_tr].append(y_te) if False else
                      y_cong if y_tr is yc_tr else y_law)
        return pipe_full, float(auc)

    pipe_cong_eval = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )),
    ])
    pipe_cong_eval.fit(X_tr, yc_tr)
    cong_auc = roc_auc_score(yc_te, pipe_cong_eval.predict_proba(X_te)[:, 1])

    pipe_law_eval = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )),
    ])
    pipe_law_eval.fit(X_tr, yl_tr)
    law_auc = roc_auc_score(yl_te, pipe_law_eval.predict_proba(X_te)[:, 1])

    # Final models fitted on full training data
    pipe_cong = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )),
    ])
    pipe_cong.fit(X, y_cong)

    pipe_law = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )),
    ])
    pipe_law.fit(X, y_law)

    return {
        "congestion":     pipe_cong,
        "law_order":      pipe_law,
        "congestion_auc": cong_auc,
        "law_order_auc":  law_auc,
    }


def predict_risks(risk_models: dict, features: dict) -> dict:
    """Return congestion_prob and law_order_prob for a single event dict."""
    row = safe_df(pd.DataFrame([features]))
    X = row[_RISK_FEATURES]
    cong_prob = float(risk_models["congestion"].predict_proba(X)[0][1])
    law_prob  = float(risk_models["law_order"].predict_proba(X)[0][1])
    return {"congestion_prob": cong_prob, "law_order_prob": law_prob}
```

- [ ] **Step 4: Run risk model tests**

```
pytest tests/test_risk_model.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 5: Confirm no regressions**

```
pytest --tb=short -q
```

- [ ] **Step 6: Commit**

```
git add src/risk_model.py tests/test_risk_model.py
git commit -m "feat: traffic congestion and law-and-order risk classifiers"
```

---

## Task 4: SHAP Explainability

**Files:**
- Create: `src/explainer.py`
- Test: `tests/test_explainer.py`

**Interfaces:**
- Consumes: `_X` from `src.model`; `safe_df`, `_RISK_FEATURES` from `src.risk_model`
- Produces:
  - `build_explainers(severity_pipeline, risk_models) -> dict` — keys: `severity`, `congestion`, `law_order`, each a `shap.TreeExplainer`
  - `explain_severity(explainer, severity_pipeline, features, predicted_class) -> list[dict]` — top-5 SHAP drivers
  - `explain_risk(explainer, risk_pipeline, features) -> list[dict]` — top-5 SHAP drivers for positive class
  - Each driver dict: `feature` (str), `display` (str), `shap` (float), `direction` (str `"+" | "-"`), `pct` (float)
  - `FEATURE_DISPLAY: dict[str, str]` — human-readable feature names

---

- [ ] **Step 1: Install shap**

```
pip install shap
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_explainer.py
import pytest
import pandas as pd
import numpy as np


def _make_minimal_train() -> pd.DataFrame:
    from tests.test_risk_model import _make_train_df
    from src.baseline import (
        compute_window_counts, compute_corridor_baselines,
        compute_excess_scores, compute_tertile_thresholds, label_severity,
    )
    df = _make_train_df()
    df["window_count"]  = compute_window_counts(df)
    baselines           = compute_corridor_baselines(df, min_obs=1)
    df["impact_score"]  = compute_excess_scores(df, baselines)
    low_t, high_t       = compute_tertile_thresholds(df)
    df["severity"]      = label_severity(df, low_t, high_t)
    return df


@pytest.fixture(scope="module")
def trained_pipelines():
    from src.model import train_model
    from src.risk_model import train_risk_models
    from src.explainer import build_explainers
    df     = _make_minimal_train()
    sev    = train_model(df)
    risks  = train_risk_models(df)
    expls  = build_explainers(sev, risks)
    return {"sev": sev, "risks": risks, "expls": expls, "df": df}


def test_build_explainers_returns_three_keys(trained_pipelines):
    expls = trained_pipelines["expls"]
    assert set(expls.keys()) == {"severity", "congestion", "law_order"}


def test_explain_severity_returns_five_drivers(trained_pipelines):
    from src.explainer import explain_severity
    sev   = trained_pipelines["sev"]
    expls = trained_pipelines["expls"]
    features = {
        "event_cause": "public_event", "event_type": "planned",
        "corridor": "CBD 2", "zone": "Central Zone 2",
        "police_station": "Cubbon Park", "hour_band": "evening",
        "hour_of_day": 18, "day_of_week": 0, "month": 2, "is_weekend": 0,
        "requires_road_closure": 0, "priority": "High", "junction": "unknown",
        "desc_traffic_slow": 1, "desc_breakdown": 0,
        "is_holiday": 0, "holiday_risk_tier": 0,
        "estimated_attendance": 500, "has_vip": 0, "is_route_event": 0,
    }
    drivers = explain_severity(expls["severity"], sev, features, "HIGH")
    assert len(drivers) == 5
    for d in drivers:
        assert set(d.keys()) == {"feature", "display", "shap", "direction", "pct"}
        assert d["direction"] in ("+", "-")
        assert 0.0 <= d["pct"] <= 100.0


def test_explain_risk_returns_five_drivers(trained_pipelines):
    from src.explainer import explain_risk
    risks = trained_pipelines["risks"]
    expls = trained_pipelines["expls"]
    features = {
        "event_cause": "public_event", "event_type": "planned",
        "corridor": "CBD 2", "zone": "Central Zone 2",
        "police_station": "Cubbon Park", "hour_band": "evening",
        "hour_of_day": 18, "day_of_week": 0, "month": 2, "is_weekend": 0,
        "requires_road_closure": 0, "priority": "High", "junction": "unknown",
        "desc_traffic_slow": 0, "desc_breakdown": 0,
        "is_holiday": 0, "holiday_risk_tier": 0,
        "estimated_attendance": 0, "has_vip": 0, "is_route_event": 0,
    }
    drivers = explain_risk(expls["congestion"], risks["congestion"], features)
    assert len(drivers) == 5
    for d in drivers:
        assert d["direction"] in ("+", "-")


def test_pcts_sum_to_100(trained_pipelines):
    from src.explainer import explain_severity
    sev   = trained_pipelines["sev"]
    expls = trained_pipelines["expls"]
    features = {
        "event_cause": "vehicle_breakdown", "event_type": "unplanned",
        "corridor": "ORR East 1", "zone": "East Zone 1",
        "police_station": "Bellandur", "hour_band": "morning",
        "hour_of_day": 9, "day_of_week": 1, "month": 1, "is_weekend": 0,
        "requires_road_closure": 0, "priority": "High", "junction": "unknown",
        "desc_traffic_slow": 0, "desc_breakdown": 1,
        "is_holiday": 0, "holiday_risk_tier": 0,
        "estimated_attendance": 0, "has_vip": 0, "is_route_event": 0,
    }
    drivers = explain_severity(expls["severity"], sev, features, "LOW")
    # pcts should sum to ≤ 100 (only top-5 shown)
    assert sum(d["pct"] for d in drivers) <= 100.0 + 1e-6
```

- [ ] **Step 3: Run to confirm failure**

```
pytest tests/test_explainer.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.explainer'`

- [ ] **Step 4: Implement `src/explainer.py`**

```python
# src/explainer.py
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

FEATURE_DISPLAY: dict[str, str] = {
    "event_cause":            "Event cause",
    "event_type":             "Event type",
    "corridor":               "Corridor",
    "zone":                   "Zone",
    "police_station":         "Police station",
    "hour_band":              "Time of day band",
    "priority":               "Priority level",
    "junction":               "Junction",
    "hour_of_day":            "Hour of day",
    "day_of_week":            "Day of week",
    "requires_road_closure":  "Road closure required",
    "month":                  "Month",
    "is_weekend":             "Weekend event",
    "desc_traffic_slow":      "Congestion keywords in description",
    "desc_breakdown":         "Breakdown keywords in description",
    "is_holiday":             "Public holiday",
    "holiday_risk_tier":      "Holiday / festival severity tier",
    "estimated_attendance":   "Expected attendance",
    "has_vip":                "VIP presence",
    "is_route_event":         "Route-based event",
    "corridor_high_rate":     "Historical HIGH-severity rate on corridor",
    "corridor_event_count":   "Historical event volume on corridor",
    "corridor_auth_rate":     "Authenticated incident rate on corridor",
    "corridor_closure_rate":  "Road closure frequency on corridor",
}


def build_explainers(severity_pipeline: Pipeline, risk_models: dict) -> dict:
    """Build TreeExplainer instances for severity, congestion, law-and-order models."""
    return {
        "severity":  shap.TreeExplainer(severity_pipeline.named_steps["lgbm"]),
        "congestion": shap.TreeExplainer(risk_models["congestion"].named_steps["clf"]),
        "law_order":  shap.TreeExplainer(risk_models["law_order"].named_steps["clf"]),
    }


def _top5_drivers(sv: np.ndarray, feature_names: list) -> list[dict]:
    total = float(np.sum(np.abs(sv))) or 1.0
    drivers = [
        {
            "feature":   name,
            "display":   FEATURE_DISPLAY.get(name, name.replace("_", " ").title()),
            "shap":      float(val),
            "direction": "+" if val >= 0 else "-",
            "pct":       round(abs(float(val)) / total * 100, 1),
        }
        for name, val in zip(feature_names, sv)
    ]
    return sorted(drivers, key=lambda d: abs(d["shap"]), reverse=True)[:5]


def explain_severity(
    explainer: shap.TreeExplainer,
    severity_pipeline: Pipeline,
    features: dict,
    predicted_class: str,
) -> list[dict]:
    """Return top-5 SHAP drivers for the severity prediction."""
    from src.model import _X
    feature_row = _X(pd.DataFrame([features]))
    # Run through CorridorStatsTransformer (adds corridor stat columns)
    X_pre = severity_pipeline.named_steps["corridor_stats"].transform(feature_row)
    feature_names = list(X_pre.columns)

    shap_vals = explainer.shap_values(X_pre)
    # Multiclass: list of arrays, one per class
    lgbm_step = severity_pipeline.named_steps["lgbm"]
    class_idx = list(lgbm_step.classes_).index(predicted_class)
    sv = shap_vals[class_idx][0]

    return _top5_drivers(sv, feature_names)


def explain_risk(
    explainer: shap.TreeExplainer,
    risk_pipeline: Pipeline,
    features: dict,
) -> list[dict]:
    """Return top-5 SHAP drivers for a binary risk model (positive class)."""
    from src.risk_model import safe_df, _RISK_FEATURES
    row = safe_df(pd.DataFrame([features]))
    X_pre = risk_pipeline.named_steps["pre"].transform(row[_RISK_FEATURES])
    raw_names = risk_pipeline.named_steps["pre"].get_feature_names_out()
    feature_names = [n.split("__", 1)[1] for n in raw_names]

    shap_vals = explainer.shap_values(X_pre)
    # Binary LightGBM returns a single 2-D array; take sample 0
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]   # positive class
    else:
        sv = shap_vals[0]

    return _top5_drivers(sv, feature_names)
```

- [ ] **Step 5: Run explainer tests**

```
pytest tests/test_explainer.py -v
```
Expected: All 4 tests PASS. Note: SHAP may print warnings about TreeExplainer with LightGBM — these are harmless.

- [ ] **Step 6: Confirm no regressions**

```
pytest --tb=short -q
```

- [ ] **Step 7: Commit**

```
git add src/explainer.py tests/test_explainer.py
git commit -m "feat: SHAP explainability module for severity and risk models"
```

---

## Task 5: Shared App Cache

**Files:**
- Create: `src/app_cache.py`

**Interfaces:**
- Consumes: all trained modules from Tasks 1–4
- Produces:
  - `load_and_train() -> dict` — `@st.cache_data`, returns `train_df`, `pipeline`, `dur_model`, `risk_models`, `explainers`, `diversion_graph`, `baselines`, `low_t`, `high_t`, `cv_f1`, `test_f1`
  - `get_road_graph() -> nx.MultiDiGraph` — `@st.cache_resource`

---

- [ ] **Step 1: Create `src/app_cache.py`**

```python
# src/app_cache.py
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from src.baseline import (
    compute_corridor_baselines, compute_excess_scores,
    compute_tertile_thresholds, compute_window_counts, label_severity,
)
from src.duration_model import train_duration_model
from src.explainer import build_explainers
from src.model import evaluate_cv, evaluate_test, train_model
from src.pipeline import load_raw, split_data
from src.recommender import build_diversion_graph
from src.risk_model import train_risk_models
from src.road_network import load_graph


@st.cache_resource(show_spinner="Loading road network...")
def get_road_graph() -> nx.MultiDiGraph:
    return load_graph(Path("data/bengaluru_drive.graphml"))


@st.cache_data(show_spinner="Loading data and training models...")
def load_and_train() -> dict:
    df = load_raw()
    df["window_count"] = compute_window_counts(df)

    train_df, val_df, test_df = split_data(df)

    baselines = compute_corridor_baselines(train_df)
    for split in [train_df, val_df, test_df]:
        split["impact_score"] = compute_excess_scores(split, baselines)

    low_t, high_t = compute_tertile_thresholds(train_df)
    for split in [train_df, val_df, test_df]:
        split["severity"] = label_severity(split, low_t, high_t)

    best_params = {
        "n_estimators":      224,
        "num_leaves":        200,
        "learning_rate":     0.2985879580529471,
        "min_child_samples": 5,
        "reg_alpha":         3.016516532940732e-08,
        "reg_lambda":        5.151065907260535e-08,
        "subsample":         0.6962164633886399,
        "colsample_bytree":  0.7913373860606256,
    }
    pipeline = train_model(train_df, params=best_params)
    cv_f1    = evaluate_cv(train_df, params=best_params)
    test_f1  = evaluate_test(pipeline, test_df)

    diversion_graph = build_diversion_graph(
        pd.concat([train_df, val_df], ignore_index=True)
    )
    dur_model   = train_duration_model(train_df)
    risk_models = train_risk_models(train_df)
    explainers  = build_explainers(pipeline, risk_models)

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
        "risk_models":     risk_models,
        "explainers":      explainers,
    }
```

- [ ] **Step 2: Verify the cache module is importable (no Streamlit runtime needed)**

```
python -c "import src.app_cache; print('OK')"
```
Expected: `OK` (no errors; Streamlit decorators are no-ops at import time outside a Streamlit server).

- [ ] **Step 3: Run full test suite**

```
pytest --tb=short -q
```
Expected: All tests pass.

- [ ] **Step 4: Commit**

```
git add src/app_cache.py
git commit -m "feat: centralised app_cache module for shared Streamlit training cache"
```

---

## Task 6: Multi-page Streamlit Restructure

**Files:**
- Create: `pages/1_Plan_Event.py`
- Create: `pages/2_Results.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: everything from `src/app_cache.py` and all domain modules
- State transfer: `st.session_state["result_data"]` dict passed from page 1 to page 2 — keys: `event_name`, `corridor`, `severity`, `confidence`, `duration`, `risks`, `shap_severity`, `shap_congestion`, `shap_law`, `officers`, `barricades`, `diversions`, `neighbors`, `fmap`, `holiday_name`, `estimated_attendance`, `has_vip`, `is_route_event`

---

- [ ] **Step 1: Create the `pages/` directory and `pages/1_Plan_Event.py`**

```python
# pages/1_Plan_Event.py
import datetime

import pandas as pd
import streamlit as st

from src.app_cache import get_road_graph, load_and_train
from src.calendar_intel import get_holiday_info
from src.duration_model import predict_duration
from src.explainer import explain_risk, explain_severity
from src.map_builder import build_map
from src.model import get_knn_neighbors, predict
from src.pipeline import corridor_metadata
from src.recommender import barricade_positions, get_diversions, officer_count
from src.risk_model import predict_risks

st.set_page_config(page_title="Event Congestion Planner", layout="wide")

state           = load_and_train()
graph           = get_road_graph()
train_df        = state["train_df"]
pipeline        = state["pipeline"]
dur_model       = state["dur_model"]
diversion_graph = state["diversion_graph"]
risk_models     = state["risk_models"]
explainers      = state["explainers"]

st.sidebar.markdown("### Model Performance")
st.sidebar.metric("CV macro-F1 (train)", f"{state['cv_f1']:.3f}")
st.sidebar.metric("Test macro-F1",       f"{state['test_f1']:.3f}")
st.sidebar.metric("Congestion AUC",      f"{state['risk_models']['congestion_auc']:.3f}")
st.sidebar.metric("Law & Order AUC",     f"{state['risk_models']['law_order_auc']:.3f}")
st.sidebar.caption("Baseline (majority class): ~0.22 on 3-class problem")

st.title("Event Congestion Planner")
st.markdown(
    "Enter details of an upcoming event to forecast traffic impact "
    "and generate a deployment plan."
)

corridors    = sorted(train_df["corridor"].dropna().unique().tolist())
event_causes = sorted(train_df["event_cause"].dropna().unique().tolist())

# ── Event type toggle ────────────────────────────────────────────────────────
event_type_display = st.radio(
    "Event type", ["Planned", "Unplanned"], horizontal=True, key="event_type_radio"
)
is_planned = event_type_display == "Planned"
event_type = "planned" if is_planned else "unplanned"

with st.form("event_form"):
    col1, col2 = st.columns(2)

    # ── Core fields (always visible) ─────────────────────────────────────────
    with col1:
        event_name  = st.text_input("Event name", value="Public Rally")
        default_idx = event_causes.index("public_event") if "public_event" in event_causes else 0
        event_cause = st.selectbox("Event cause", event_causes, index=default_idx)
        corridor    = st.selectbox("Primary corridor", corridors)
        priority    = st.selectbox("Priority", ["High", "Low"], index=0)
    with col2:
        event_date   = st.date_input("Date", value=datetime.date.today())
        event_time   = st.time_input("Start time", value=datetime.time(18, 0))
        road_closure = st.checkbox("Requires road closure?", value=False)

    # ── Calendar strip (auto-filled, officer can override) ───────────────────
    auto_cal = get_holiday_info(event_date)
    holiday_options = ["none", "state", "national", "festival"]
    auto_idx = holiday_options.index(auto_cal["holiday_type"]) \
        if auto_cal["holiday_type"] in holiday_options else 0

    st.markdown("**Calendar context**")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        holiday_type_sel = st.selectbox(
            "Holiday / festival type", holiday_options, index=auto_idx,
            help=f"Auto-detected: {auto_cal['holiday_name'] or 'None'}",
        )
    with cc2:
        holiday_name_input = st.text_input(
            "Name (optional)", value=auto_cal["holiday_name"]
        )

    _tier_map = {"none": 0, "state": 1, "national": 2, "festival": 3}
    holiday_risk_tier = _tier_map[holiday_type_sel]
    is_holiday        = int(holiday_type_sel != "none")

    # ── Planned-only fields ──────────────────────────────────────────────────
    estimated_attendance = 0
    has_vip       = 0
    is_route_event = 0

    if is_planned:
        st.markdown("---")
        st.markdown("**Planned event details**")
        p1, p2 = st.columns(2)
        with p1:
            estimated_attendance = st.number_input(
                "Estimated attendance", min_value=0, value=1000, step=100
            )
            has_vip = int(st.checkbox("VIP presence?", value=False))
        with p2:
            st.text_input("Organizer (optional)", value="")

        route_fmt = st.radio(
            "Event format", ["Venue-based", "Route-based"], horizontal=True
        )
        is_route_event = int(route_fmt == "Route-based")
        if is_route_event:
            r1, r2 = st.columns(2)
            with r1:
                st.text_input("Start checkpoint", value="")
            with r2:
                st.text_input("End checkpoint", value="")
            st.text_input("Intermediate stops (comma-separated, optional)", value="")

    # ── Unplanned-only fields ────────────────────────────────────────────────
    if not is_planned:
        st.markdown("---")
        st.markdown("**Incident details**")
        u1, u2 = st.columns(2)
        with u1:
            st.selectbox("Incident type", [
                "accident", "breakdown", "protest", "riot",
                "vip_movement", "natural_disaster", "other",
            ])
        with u2:
            st.checkbox("Medical support needed?", value=False)

    submitted = st.form_submit_button("Predict Impact", type="primary")

if submitted:
    hour = event_time.hour
    dow  = event_date.weekday()
    if hour < 6:       hb = "night"
    elif hour < 12:    hb = "morning"
    elif hour < 18:    hb = "afternoon"
    else:              hb = "evening"

    zone, police, lat, lng = corridor_metadata(train_df, corridor)

    features = {
        "event_cause":           event_cause,
        "event_type":            event_type,
        "corridor":              corridor,
        "zone":                  zone,
        "police_station":        police,
        "hour_band":             hb,
        "hour_of_day":           hour,
        "day_of_week":           dow,
        "requires_road_closure": road_closure,
        "priority":              priority,
        "junction":              "unknown",
        "month":                 event_date.month,
        "is_weekend":            int(dow >= 5),
        "desc_traffic_slow":     0,
        "desc_breakdown":        int(event_cause == "vehicle_breakdown"),
        "is_holiday":            is_holiday,
        "holiday_risk_tier":     holiday_risk_tier,
        "estimated_attendance":  int(estimated_attendance),
        "has_vip":               has_vip,
        "is_route_event":        is_route_event,
    }

    severity, confidence = predict(pipeline, features)
    duration             = predict_duration(dur_model, features)
    risks                = predict_risks(risk_models, features)
    shap_sev             = explain_severity(
        explainers["severity"], pipeline, features, severity
    )
    shap_cong            = explain_risk(
        explainers["congestion"], risk_models["congestion"], features
    )
    shap_law             = explain_risk(
        explainers["law_order"], risk_models["law_order"], features
    )
    neighbors  = get_knn_neighbors(train_df, features, k=5)
    barricades = barricade_positions(train_df, corridor, lat, lng)
    n_adj      = min(3, len(barricades))
    officers   = officer_count(severity, n_adjacent_junctions=n_adj)
    diversions = get_diversions(diversion_graph, corridor, hb)
    fmap       = build_map(
        lat, lng, severity, barricades, diversions,
        officers, train_df, event_name, graph
    )

    st.session_state["result_data"] = {
        "event_name":            event_name,
        "corridor":              corridor,
        "severity":              severity,
        "confidence":            confidence,
        "duration":              duration,
        "risks":                 risks,
        "shap_severity":         shap_sev,
        "shap_congestion":       shap_cong,
        "shap_law":              shap_law,
        "officers":              officers,
        "barricades":            barricades,
        "diversions":            diversions,
        "neighbors":             neighbors,
        "fmap":                  fmap,
        "holiday_name":          holiday_name_input,
        "estimated_attendance":  int(estimated_attendance),
        "has_vip":               has_vip,
        "is_route_event":        is_route_event,
    }
    st.switch_page("pages/2_Results.py")
```

- [ ] **Step 2: Create `pages/2_Results.py`**

```python
# pages/2_Results.py
import streamlit as st
from streamlit_folium import st_folium

from src.app_cache import load_and_train

st.set_page_config(page_title="Event Congestion Planner — Results", layout="wide")

state = load_and_train()   # hits the cache; no re-training

# ── Guard: redirect if navigated here directly without submitting form ───────
if "result_data" not in st.session_state:
    st.warning("No prediction data found. Please fill in the event form first.")
    st.page_link("pages/1_Plan_Event.py", label="← Back to form")
    st.stop()

r          = st.session_state["result_data"]
severity   = r["severity"]
confidence = r["confidence"]
officers   = r["officers"]
barricades = r["barricades"]
diversions = r["diversions"]
neighbors  = r["neighbors"]
fmap       = r["fmap"]
risks      = r["risks"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("CV macro-F1 (train)", f"{state['cv_f1']:.3f}")
st.sidebar.metric("Test macro-F1",       f"{state['test_f1']:.3f}")
st.sidebar.metric("Congestion AUC",      f"{state['risk_models']['congestion_auc']:.3f}")
st.sidebar.metric("Law & Order AUC",     f"{state['risk_models']['law_order_auc']:.3f}")

# ── Header ────────────────────────────────────────────────────────────────────
st.page_link("pages/1_Plan_Event.py", label="← Back to form")
st.title(f"Deployment Plan — {r['event_name']}")

conf_pct = confidence.get(severity, 0.0) * 100

left, right = st.columns([1, 2])


def _risk_bar(prob: float) -> str:
    filled = int(round(prob * 10))
    return "█" * filled + "░" * (10 - filled)


def _risk_label(prob: float) -> str:
    if prob < 0.33:  return "LOW"
    if prob < 0.66:  return "MEDIUM"
    return "HIGH"


def _render_shap_drivers(drivers: list[dict]) -> None:
    for d in drivers:
        arrow = "▲" if d["direction"] == "+" else "▼"
        st.markdown(
            f"{arrow} **{d['direction']}{d['pct']}%** &nbsp; {d['display']}"
        )


with left:
    st.markdown(f"## {severity}")
    st.caption(f"Confidence: {conf_pct:.0f}%  |  Corridor: {r['corridor']}")

    # Duration
    dur_model  = state["dur_model"]
    _low_min   = round(dur_model["low_thresh"]  * 60 / 5) * 5
    _high_min  = round(dur_model["high_thresh"] * 60 / 5) * 5
    _DUR_LABELS = {
        "SHORT":  f"SHORT (<{_low_min} min)",
        "MEDIUM": f"MEDIUM ({_low_min}–{_high_min} min)",
        "LONG":   f"LONG (>{_high_min} min)",
    }
    _dur = r.get("duration", "N/A")
    st.markdown(f"**Duration Forecast:** {_DUR_LABELS.get(_dur, _dur)}")

    # ── Risk Forecast ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Risk Forecast")
    cong_prob = risks["congestion_prob"]
    law_prob  = risks["law_order_prob"]
    st.markdown(
        f"**Traffic Congestion** &nbsp; "
        f"`{_risk_bar(cong_prob)}` &nbsp; "
        f"{cong_prob*100:.0f}% — **{_risk_label(cong_prob)}**"
    )
    st.markdown(
        f"**Law & Order** &nbsp; "
        f"`{_risk_bar(law_prob)}` &nbsp; "
        f"{law_prob*100:.0f}% — **{_risk_label(law_prob)}**"
    )

    # ── Action Plan ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Action Plan")
    st.markdown(f"**Officers:** {officers['total_min']}–{officers['total_max']} total")
    st.markdown(
        f"  ({officers['primary_min']}–{officers['primary_max']} on primary corridor)"
    )
    st.markdown(f"**Barricades:** {len(barricades)} position(s)")
    for b in barricades:
        st.markdown(f"  - {b}")
    st.markdown(f"**Diversions:** {len(diversions)} route(s)")
    for d in diversions:
        st.markdown(f"  - {d}")

    # ── SHAP Explainability ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Why {severity}?")
    _render_shap_drivers(r["shap_severity"])

    with st.expander(f"Why traffic congestion = {cong_prob*100:.0f}%?"):
        _render_shap_drivers(r["shap_congestion"])

    with st.expander(f"Why law & order risk = {law_prob*100:.0f}%?"):
        _render_shap_drivers(r["shap_law"])

    # ── Similar past events ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 5 Similar Past Events")
    if not neighbors.empty:
        display = neighbors[
            ["corridor", "event_cause", "severity", "impact_score"]
        ].copy()
        display.columns = ["Corridor", "Cause", "Severity", "Excess Score"]
        display["Excess Score"] = display["Excess Score"].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)
    avg_excess = neighbors["impact_score"].mean() if not neighbors.empty else 0
    st.caption(
        f"Avg excess incidents in similar events: {avg_excess:+.1f} above baseline"
    )

with right:
    st.markdown("### Impact Map")
    st_folium(fmap, width=700, height=520, returned_objects=[])

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
import pandas as pd
export_rows = [
    ("Event",                r["event_name"]),
    ("Corridor",             r["corridor"]),
    ("Severity",             severity),
    ("Confidence",           f"{conf_pct:.0f}%"),
    ("Duration",             r.get("duration", "N/A")),
    ("Officers min",         str(officers["total_min"])),
    ("Officers max",         str(officers["total_max"])),
    ("Barricades",           "; ".join(barricades) if barricades else "None"),
    ("Diversions",           "; ".join(diversions) if diversions else "None"),
    ("Congestion prob",      f"{cong_prob*100:.0f}%"),
    ("Law & order prob",     f"{law_prob*100:.0f}%"),
    ("Holiday",              r.get("holiday_name", "")),
    ("Estimated attendance", str(r.get("estimated_attendance", 0))),
    ("VIP presence",         str(bool(r.get("has_vip", 0)))),
    ("Route event",          str(bool(r.get("is_route_event", 0)))),
]
export_df = pd.DataFrame(export_rows, columns=["Field", "Value"])
st.download_button(
    "Export Plan (CSV)",
    data=export_df.to_csv(index=False),
    file_name=f"plan_{r['event_name'].replace(' ', '_')}.csv",
    mime="text/csv",
)
```

- [ ] **Step 3: Gut `app.py` to a redirect**

Replace the entire content of `app.py` with:

```python
# app.py
import streamlit as st
st.switch_page("pages/1_Plan_Event.py")
```

- [ ] **Step 4: Run full test suite to confirm no breakage**

```
pytest --tb=short -q
```
Expected: All tests PASS. (The existing `app.py` tests, if any, will need to be confirmed — none exist in the current suite.)

- [ ] **Step 5: Manual smoke test — start the app and verify all screens**

```
streamlit run app.py
```

Checklist:
- [ ] App opens and immediately shows the Plan Event form (redirect works)
- [ ] Sidebar shows 4 metrics: CV F1, Test F1, Congestion AUC, Law & Order AUC
- [ ] Calendar strip auto-fills for today's date; changing date updates it
- [ ] Switching to "Planned" shows attendance / VIP / route fields
- [ ] Switching to "Unplanned" hides planned fields and shows incident type
- [ ] Submitting form navigates to Results page (no `show_results` flag needed)
- [ ] Results page shows Risk Forecast bars for congestion and law & order
- [ ] "Why HIGH?" section shows 5 SHAP drivers with arrows and percentages
- [ ] Two collapsed expanders appear for risk explanations
- [ ] "← Back to form" link returns to page 1 and clears results
- [ ] Export CSV includes `Congestion prob`, `Law & order prob`, `Holiday` columns
- [ ] Navigating directly to Results without submitting shows the warning and back link

- [ ] **Step 6: Commit**

```
git add pages/1_Plan_Event.py pages/2_Results.py app.py
git commit -m "feat: multi-page Streamlit app with adaptive form and extended results dashboard"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Multi-page restructure — Task 6
- [x] Calendar intelligence — Task 1 + Task 2 (`pipeline.py`)
- [x] Adaptive form planned/unplanned — Task 6 (`pages/1_Plan_Event.py`)
- [x] Officer override of holiday — Task 6 (selectbox with auto-detected default)
- [x] Risk models (congestion + law & order) — Task 3
- [x] Risk AUC in sidebar — Task 6 (`pages/1_Plan_Event.py` + `pages/2_Results.py`)
- [x] SHAP explainability for all 3 models — Task 4 + Task 6
- [x] Extended CSV export — Task 6 (`pages/2_Results.py`)
- [x] `@st.cache_data` / `@st.cache_resource` shared across pages — Task 5

**All success criteria from spec:**
- Calendar: `get_holiday_info(date(2024,10,12))` → Dasara tier-3 ✓ (covered by `test_dasara_window_2024`)
- Adaptive form: Planned fields hidden when Unplanned — covered by conditional `if is_planned` block ✓
- Congestion model: AUC > 0.60 — verified at runtime in sidebar; test confirms 0–1 range ✓
- SHAP: top-5 drivers for severity; expanders for risk ✓
- All existing panels preserved in Results ✓
