# pages/1_Plan_Event.py
import datetime

import streamlit as st

from src.app_cache import get_model_state, get_road_graph, get_train_df
from src.calendar_intel import get_holiday_info
from src.duration_model import predict_duration
from src.explainer import build_explainers, explain_risk, explain_severity
from src.map_builder import build_map
from src.model import get_knn_neighbors, predict
from src.pipeline import corridor_metadata
from src.recommender import barricade_positions, get_diversions, officer_count
from src.risk_model import predict_risks

st.set_page_config(page_title="Event Congestion Planner", layout="wide")

train_df = get_train_df()

st.sidebar.markdown("### Model Performance")
st.sidebar.caption("Models are trained on first prediction and then cached.")

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
    model_state = get_model_state()
    pipeline = model_state["pipeline"]
    dur_model = model_state["dur_model"]
    diversion_graph = model_state["diversion_graph"]
    risk_models = model_state["risk_models"]

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

    shap_sev: list[dict] = []
    shap_cong: list[dict] = []
    shap_law: list[dict] = []
    try:
        explainers = build_explainers(pipeline, risk_models)
        shap_sev = explain_severity(explainers["severity"], pipeline, features, severity)
        shap_cong = explain_risk(explainers["congestion"], risk_models["congestion"], features)
        shap_law = explain_risk(explainers["law_order"], risk_models["law_order"], features)
    except Exception:
        # Keep app usable when SHAP fails on hosted runtimes.
        pass

    neighbors  = get_knn_neighbors(train_df, features, k=5)
    barricades = barricade_positions(train_df, corridor, lat, lng)
    n_adj      = min(3, len(barricades))
    officers   = officer_count(severity, n_adjacent_junctions=n_adj)
    diversions = get_diversions(diversion_graph, corridor, hb)

    with st.spinner("Loading road network for map..."):
        graph = get_road_graph()
    fmap       = build_map(
        lat, lng, severity, barricades, diversions,
        officers, train_df, event_name, graph, corridor=corridor
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
        "test_f1":               model_state["test_f1"],
        "congestion_auc":        model_state["congestion_auc"],
        "law_order_auc":         model_state["law_order_auc"],
        "dur_low_thresh":        dur_model["low_thresh"],
        "dur_high_thresh":       dur_model["high_thresh"],
    }
    st.switch_page("pages/2_Results.py")
