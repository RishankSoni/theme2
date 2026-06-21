# pages/2_Results.py
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Event Congestion Planner — Results", layout="wide")

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
st.sidebar.metric("Test macro-F1",  f"{float(r.get('test_f1', 0.0)):.3f}")
st.sidebar.metric("Congestion AUC", f"{float(r.get('congestion_auc', 0.0)):.3f}")
st.sidebar.metric("Law & Order AUC", f"{float(r.get('law_order_auc', 0.0)):.3f}")

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
    _low_min   = round(float(r.get("dur_low_thresh", 0.0)) * 60 / 5) * 5
    _high_min  = round(float(r.get("dur_high_thresh", 0.0)) * 60 / 5) * 5
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

st.markdown("---")
st.page_link("pages/3_Post_Event_Report.py", label="File Post-Event Report")
