# app.py
import streamlit as st
import pandas as pd
import datetime
from streamlit_folium import st_folium  # type: ignore[import]

from src.pipeline import load_raw, split_data, corridor_metadata
from src.baseline import (
    compute_window_counts, compute_corridor_baselines,
    compute_excess_scores, compute_tertile_thresholds, label_severity,
)
from src.model import train_model, evaluate_cv, evaluate_test, predict, get_knn_neighbors
from src.recommender import officer_count, barricade_positions, build_diversion_graph, get_diversions
from src.map_builder import build_map
from src.duration_model import (
    duration_tertile_thresholds, compute_duration_labels,
    train_duration_model, predict_duration,
)

st.set_page_config(page_title="Event Congestion Planner", layout="wide")

# ── Cached training pipeline ────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data and training model...")
def load_and_train():
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

    low_d, high_d = duration_tertile_thresholds(train_df)
    train_df["duration_label"] = compute_duration_labels(train_df, low_d, high_d)
    dur_model = train_duration_model(train_df)

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

# ── App state ────────────────────────────────────────────────────────────────

state           = load_and_train()
train_df        = state["train_df"]
pipeline        = state["pipeline"]
diversion_graph = state["diversion_graph"]
cv_f1           = state["cv_f1"]
test_f1         = state["test_f1"]
dur_model       = state["dur_model"]

st.sidebar.markdown("### Model Performance")
st.sidebar.metric("CV macro-F1 (train)", f"{cv_f1:.3f}")
st.sidebar.metric("Test macro-F1",       f"{test_f1:.3f}")
st.sidebar.caption("Baseline (majority class): ~0.22 on 3-class problem")

if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "result_data" not in st.session_state:
    st.session_state.result_data = {}

# ── Screen 1: Event Input Form ───────────────────────────────────────────────

if not st.session_state.show_results:
    st.title("Event Congestion Planner")
    st.markdown("Enter details of an upcoming event to forecast traffic impact and generate a deployment plan.")

    corridors    = sorted(train_df["corridor"].dropna().unique().tolist())
    event_causes = sorted(train_df["event_cause"].dropna().unique().tolist())
    event_types  = ["planned", "unplanned"]

    with st.form("event_form"):
        col1, col2 = st.columns(2)
        with col1:
            event_name  = st.text_input("Event name", value="Public Rally")
            event_type  = st.selectbox("Event type", event_types)
            default_cause_idx = event_causes.index("public_event") if "public_event" in event_causes else 0
            event_cause = st.selectbox("Event cause", event_causes, index=default_cause_idx)
            corridor    = st.selectbox("Primary corridor", corridors)
            priority    = st.selectbox("Priority", ["High", "Low"], index=0)
        with col2:
            event_date   = st.date_input("Date", value=datetime.date.today())
            event_time   = st.time_input("Start time", value=datetime.time(18, 0))
            road_closure = st.checkbox("Requires road closure?", value=False)

        submitted = st.form_submit_button("Predict Impact", type="primary")

    if submitted:
        hour = event_time.hour
        dow  = event_date.weekday()
        if hour < 6:    hb = "night"
        elif hour < 12: hb = "morning"
        elif hour < 18: hb = "afternoon"
        else:           hb = "evening"

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
        }

        severity, confidence = predict(pipeline, features)
        neighbors            = get_knn_neighbors(train_df, features, k=5)
        barricades           = barricade_positions(train_df, corridor, lat, lng)
        n_adj                = min(3, len(barricades))
        officers             = officer_count(severity, n_adjacent_junctions=n_adj)
        diversions           = get_diversions(diversion_graph, corridor, hb)
        duration             = predict_duration(dur_model, features)
        fmap                 = build_map(lat, lng, severity, barricades, diversions, officers, train_df, event_name)

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
        st.session_state.show_results = True
        st.rerun()

# ── Screen 2: Results Dashboard ──────────────────────────────────────────────

else:
    r          = st.session_state.result_data
    severity   = r["severity"]
    confidence = r["confidence"]
    officers   = r["officers"]
    barricades = r["barricades"]
    diversions = r["diversions"]
    neighbors  = r["neighbors"]
    fmap       = r["fmap"]

    SEVERITY_COLOR = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
    conf_pct = confidence.get(severity, 0.0) * 100

    if st.button("Back to form"):
        st.session_state.show_results = False
        st.rerun()

    st.title(f"Deployment Plan — {r['event_name']}")

    left, right = st.columns([1, 2])

    with left:
        sev_label = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"}[severity]
        st.markdown(f"## {sev_label}")
        st.caption(f"Confidence: {conf_pct:.0f}%  |  Corridor: {r['corridor']}")

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

        st.markdown("---")
        st.markdown("### Action Plan")
        st.markdown(f"**Officers:** {officers['total_min']}-{officers['total_max']} total")
        st.markdown(f"  ({officers['primary_min']}-{officers['primary_max']} on primary corridor)")
        st.markdown(f"**Barricades:** {len(barricades)} position(s)")
        for b in barricades:
            st.markdown(f"  - {b}")
        st.markdown(f"**Diversions:** {len(diversions)} route(s)")
        for d in diversions:
            st.markdown(f"  - {d}")

        st.markdown("---")
        st.markdown("### 5 Similar Past Events")
        if not neighbors.empty:
            display = neighbors[["corridor", "event_cause", "severity", "impact_score"]].copy()
            display.columns = ["Corridor", "Cause", "Severity", "Excess Score"]
            display["Excess Score"] = display["Excess Score"].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)
        avg_excess = neighbors["impact_score"].mean() if not neighbors.empty else 0
        st.caption(f"Avg excess incidents in similar events: {avg_excess:+.1f} above baseline")

    with right:
        st.markdown("### Impact Map")
        st_folium(fmap, width=700, height=520, returned_objects=[])

    st.markdown("---")
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
    export_df = pd.DataFrame(export_rows, columns=["Field", "Value"])
    st.download_button(
        "Export Plan (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"plan_{r['event_name'].replace(' ', '_')}.csv",
        mime="text/csv",
    )
