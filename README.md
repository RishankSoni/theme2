# Event-Driven Congestion Forecast & Recommendation System

**GRIDLOCK 2.0 — Theme 2 Hackathon Demo**

Predicts traffic impact severity for upcoming events and auto-generates a concrete deployment plan — before the event happens. Built on historical GRIDLOCK Bengaluru incident data.

---

## What It Does

Given an upcoming event (type, location, date/time), the system:

1. **Forecasts severity** — LOW / MEDIUM / HIGH — using a Gradient Boosted classifier trained on historical incident patterns
2. **Generates a deployment plan** — officer count, barricade positions, and diversion routes
3. **Shows a live map** — impact zone, officer pins, barricade markers, diversion polylines
4. **Surfaces evidence** — 5 most similar past events with their actual outcomes

---

## Architecture

```
Layer 1 — Data Pipeline     load_raw() → split_data() → feature engineering
Layer 2 — ML Impact Scorer  excess-above-baseline target → GBT classifier → KNN evidence
Layer 3 — Recommendation    officer counts · barricade positions · diversion graph
Layer 4 — Streamlit UI      event input form → split-pane results dashboard + Folium map
```

### Impact Score (the ML target)

```
window_count(e)  = incidents on same corridor within [t−1h, t+1.5h]
baseline(c, t)   = mean window_count for corridor c, same hour-band, same day-of-week
impact_score(e)  = window_count(e) − baseline(c, t)
```

This isolates the event's causal contribution, removing the corridor's normal activity level. Tertile thresholds (LOW / MEDIUM / HIGH cut-points) are computed on the **training split only** to prevent leakage.

### Data Splits

| Split | Size | Purpose |
|---|---|---|
| Train | 70% | Fit model, compute baselines, compute tertile thresholds |
| Validation | 15% | Hyperparameter tuning, window sweep |
| Test | 15% | Final macro-F1 — reported once, never touched during development |

### Model Validation

- Window sweep over {1h, 1.5h, 2h, 2.5h} post-windows → **1.5h wins** (CV macro-F1 = 0.637)
- Majority-class baseline macro-F1 ≈ 0.22
- Minimum bar: test macro-F1 ≥ 0.50 ✅

---

## Setup

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Data:** Copy the GRIDLOCK CSV into `data/events.csv`:
```
data/events.csv   ← Astram event data_anonymized.csv
```

---

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. First load trains the model (~30–60s, cached afterwards).

---

## Demo Flow

1. Open the app — **Event Input** screen loads
2. Enter: `Public Rally · public_event · CBD 2 · Monday 18:00`
3. Click **Predict Impact**
4. Results screen shows severity badge + confidence %
5. Action plan: officer range, barricade list, diversion routes
6. Map: colored impact circle, barricade markers, dashed diversion lines
7. Evidence panel: 5 similar past events with excess incident counts
8. Click **Export Plan (CSV)** to download for field units

---

## Tests

```bash
pytest tests/ -v
```

28 tests across 6 files:

| File | Tests |
|---|---|
| `tests/test_pipeline.py` | 6 — data loading, splits, feature extraction |
| `tests/test_baseline.py` | 7 — window counts, excess scores, severity labels |
| `tests/test_model.py` | 5 — training, prediction, CV, KNN |
| `tests/test_recommender.py` | 6 — officer counts, barricades, diversion graph |
| `tests/test_map_builder.py` | 2 — Folium map construction |
| `tests/test_integration.py` | 2 — end-to-end CBD 2 scenario, test F1 assertion |

---

## File Map

```
app.py                    Streamlit entry point
src/
  pipeline.py             load_raw(), split_data(), corridor_metadata()
  baseline.py             window counts, excess scores, severity labeling
  model.py                GBT classifier, CV/test evaluation, KNN evidence
  recommender.py          officer counts, barricades, diversion graph
  map_builder.py          Folium map with impact zone + markers
  window_sweep.py         one-off script that validated the 1.5h post-window
tests/
  conftest.py             shared sample_df fixture
  test_pipeline.py
  test_baseline.py
  test_model.py
  test_recommender.py
  test_map_builder.py
  test_integration.py
data/
  events.csv              GRIDLOCK Bengaluru incident data (not committed)
```

---

## Stack

| Library | Version | Purpose |
|---|---|---|
| streamlit | ≥1.32 | Web dashboard |
| folium | ≥0.16 | Interactive map |
| streamlit-folium | ≥0.20 | Folium ↔ Streamlit bridge |
| scikit-learn | ≥1.4 | GBT classifier, OrdinalEncoder, KNN |
| pandas | ≥2.1 | Data pipeline |
| numpy | ≥1.26 | Numerical operations |
| pytest | ≥8.0 | Test suite |
