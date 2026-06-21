# Event-Driven Congestion: Forecast & Recommendation System

**Date:** 2026-06-16
**Context:** Hackathon demo — GRIDLOCK 2.0, Theme 2
**Stack:** Python + Streamlit + Folium + scikit-learn

---

## Problem Statement

Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns in Bengaluru. Today:

- Event impact is not quantified in advance
- Resource deployment (manpower, barricades, diversions) is experience-driven
- There is no post-event learning system

**Goal:** Given an upcoming event (type, location, expected date/time), use historical GRIDLOCK incident data to forecast traffic impact severity and automatically generate a concrete deployment plan — before the event happens.

---

## Users

| Role | Interaction |
|---|---|
| Traffic Management Center (TMC) | Plans ahead: enters upcoming events, reviews predictions and action plans |
| Police Field Units | Executes: receives officer count, barricade positions, diversion orders |

Primary interface for this demo: TMC operator at a control-room web dashboard.

---

## Data Source

**File:** `Astram event data_anonymized.csv` (GRIDLOCK system, Bengaluru, ~2023–2024)

**Key fields used:**

| Field | Purpose |
|---|---|
| `event_type` | planned / unplanned |
| `event_cause` | vehicle_breakdown, public_event, construction, accident, tree_fall, water_logging, pot_holes, congestion, others |
| `latitude`, `longitude` | Geolocation of event |
| `corridor` | Named traffic corridor (Tumkur Road, ORR East 1, etc.) |
| `zone` | City zone (North Zone 1, Central Zone 2, etc.) |
| `police_station` | Jurisdiction |
| `junction` | Specific junction name |
| `start_datetime`, `end_datetime` | Event window |
| `requires_road_closure` | Boolean — barricade signal |
| `priority` | High / Low |
| `status` | active / resolved / closed |

---

## Architecture: Four Layers

### Layer 1 — Data Pipeline

Run once at app startup. Steps:

1. Load CSV; parse datetimes; drop rows with null `start_datetime` or `corridor`.
2. Extract features per event:
   - `hour_of_day` (0–23), `day_of_week` (0–6)
   - `corridor`, `zone`, `police_station`
   - `event_cause`, `event_type`, `requires_road_closure`
3. Compute **excess incident target** (see below) — using only the training split's time periods for baseline statistics (see Data Splits section).
4. Three-way stratified split: **70% train / 15% validation / 15% test**, stratified by severity class. See Data Splits section for usage rules.

### Layer 2 — ML Impact Scorer

**Target: Excess Incidents Above Baseline**

For each event `e` on corridor `c` starting at time `t`:

```
window_count(e)   = incidents on corridor c within [t − 1h, t + 2h]
baseline(c, t)    = mean window_count for corridor c, same hour-of-day band,
                    same day-of-week, on non-event days
impact_score(e)   = window_count(e) − baseline(c, t)
```

**Window justification:** The 1h pre-window captures arrival congestion; the 2h post-window is set to the 75th percentile of event durations in the dataset (computed from `closed_datetime − start_datetime` across all closed/resolved events). Both boundaries are fixed constants, not free parameters. During implementation, sweep post-window values of {1h, 1.5h, 2h, 2.5h} and confirm 2h maximises corridor-level correlation with known high-priority events before committing.

This isolates the event's causal contribution to congestion, removing the corridor's normal activity level from the signal.

**Thin corridor fallback:** If corridor `c` has fewer than 10 non-event day observations, fall back to the zone-level baseline for that hour/day-of-week combination.

**Severity classes:** Tertiles of `impact_score` computed on the **training split only**, then applied to validation/test. This prevents test-distribution leakage into the threshold boundaries.

| Class | Meaning |
|---|---|
| LOW | Impact score in bottom tertile — minimal disruption expected |
| MEDIUM | Middle tertile — moderate congestion, partial resource deployment |
| HIGH | Top tertile — significant disruption, full deployment required |

**Model:** Gradient Boosted Classifier (scikit-learn `GradientBoostingClassifier`) trained on the feature set above. Output: severity class + confidence probability.

**Validation:** 5-fold stratified cross-validation on the training split; report macro-F1 (macro averaging used because class sizes after tertile split may not be equal). Comparison baseline: majority-class predictor achieves macro-F1 ≈ 0.22 on a balanced 3-class problem. Minimum bar before the recommendation layer can be trusted: macro-F1 ≥ 0.50. Target: macro-F1 ≥ 0.65. If the model cannot clear 0.50, the most likely fix is replacing the categorical `corridor` encoding with a geospatial cluster derived from `(latitude, longitude)` — the corridor strings in the data are unevenly distributed and may not encode geographic proximity well.

**Evidence Panel:** KNN (k=5) on the same feature space surfaces the 5 most similar historical events, showing their corridor, date, actual severity, and excess incident count — giving the prediction interpretability for judges and operators.

### Layer 3 — Recommendation Engine

Deterministic rules taking `(severity_class, corridor, zone, requires_road_closure)` as input:

**Officer Count:**
| Severity | Officers on primary corridor | Officers at adjacent junctions |
|---|---|---|
| LOW | 2–4 | 0 |
| MEDIUM | 4–6 | 1 per adjacent junction |
| HIGH | 8–12 | 2 per adjacent junction |

**Barricade Positions:** Drawn from the `junction` field of historical events on the same corridor. The junctions most frequently flagged with `requires_road_closure = TRUE` on that corridor become the recommended barricade points.

**Diversion Corridors (data-derived):** When corridor C has a high-priority event at time t, corridors that show elevated incident counts in the window [t, t+1h] are empirically absorbing diverted traffic — they are the natural diversion routes. For each corridor pair (C, D), compute:

```
co_elevation(C, D) = mean excess incidents on D during events on C
                     normalised by D's own baseline
```

Corridors D where `co_elevation` exceeds 1 standard deviation above D's mean are ranked as diversion candidates for C. This co-disruption graph is precomputed from the full dataset at pipeline time and stored as a lookup. At recommendation time, the top-2 co-elevated corridors for the predicted primary corridor are returned as recommended diversions. If fewer than 5 co-occurrence observations exist for a pair, the pair is excluded from the graph.

### Layer 4 — Streamlit UI

**Screen 1: Event Input Form**
- Event type selector (planned / unplanned)
- Event cause dropdown (matches dataset `event_cause` values)
- Location: corridor selector + optional lat/lng picker
- Date and time pickers

**Screen 2: Results Dashboard (Split-Pane layout)**

```
┌──────────────────┬────────────────────────────────────┐
│ Severity Badge   │                                    │
│ HIGH · 78%       │                                    │
├──────────────────│       Folium Map                   │
│ Action Plan      │   · Impact zone (colored circle)   │
│ 👮 10 officers   │   · Officer pins (blue markers)    │
│ 🚧 4 barricades  │   · Barricade points (red markers) │
│ ↩ 2 diversions   │   · Diversion routes (dashed line) │
├──────────────────│                                    │
│ 5 Similar Events │                                    │
│ (evidence panel) │                                    │
└──────────────────┴────────────────────────────────────┘
```

The map centers on the event location. The impact zone is a radius circle color-coded by severity (green / amber / red). Officer and barricade markers are placed at junction coordinates from the dataset.

---

## Map Visualization Details

| Element | Representation |
|---|---|
| Event epicenter | Star marker, labeled with event name |
| Impact zone | Filled circle, radius scaled by severity (LOW: 500m, MEDIUM: 1km, HIGH: 2km), color-coded |
| Recommended officer positions | Blue person markers at junction coordinates |
| Barricade points | Red barrier markers at junction coordinates |
| Diversion routes | Dashed polyline along alternate corridor |

---

## Data Splits

| Split | Size | Purpose |
|---|---|---|
| **Train** | 70% | Fit the model; compute corridor/zone baselines; compute tertile thresholds |
| **Validation** | 15% | Tune hyperparameters: sweep post-window values {1h, 1.5h, 2h, 2.5h}, tune model depth/n_estimators, select features |
| **Test** | 15% | Evaluate once at the end — report final macro-F1. Never used during development |

**Rules:**
- All three splits are stratified by severity class so each has balanced LOW/MEDIUM/HIGH representation.
- Corridor/zone **baseline means** (non-event period averages) are computed from non-event records that fall within training-split time periods only. Applying test-period baselines to test events would leak future knowledge.
- **Tertile thresholds** (the cut points that define LOW/MEDIUM/HIGH from `impact_score`) are computed on the train split only, then applied identically to validation and test.
- The co-disruption graph (diversion corridors) is precomputed from train + validation events only; test events are never used to build it.
- Final reported metric: **macro-F1 on test set**. This is the one number presented to judges.

---

## Implementation Cautions

1. **Thin corridor baselines:** Require a minimum of 10 non-event observations before using a corridor-level baseline. Fall back to zone-level baseline otherwise. This mirrors the location-fallback structure already present in the recommendation layer.

2. **Tertile leakage:** Severity class thresholds (LOW/MEDIUM/HIGH cut points) must be computed on the training split only, then applied to validation and test sets. Computing on the full dataset before splitting leaks test-distribution information into the thresholds — a subtle but real data leakage issue.

---

## Demo Flow (Narrative for Judges)

1. Operator opens dashboard — **Event Input** screen loads.
2. Enters: "Public Event · M. Chinnaswamy Stadium area · CBD 2 corridor · Monday 18:00".
3. Clicks **Predict Impact**.
4. Results screen shows: **HIGH severity · 78% confidence**.
5. Action plan panel: 10 officers, 4 barricades at Queen's Statue Circle / MG Road junction, divert via St. Marks Road.
6. Map shows red impact zone over CBD 2, officer pins at key junctions, dashed diversion route.
7. Evidence panel: "Based on 5 similar events — avg +6.2 excess incidents above baseline on this corridor at this time."
8. Operator exports plan (CSV or PDF) to share with field units.

---

## Out of Scope (for this demo)

- Real-time CCTV or GPS feed integration
- Live traffic speed data
- Mobile app for field officers
- Automated push notifications
- Post-event learning feedback loop (architecture supports it, not built for demo)
