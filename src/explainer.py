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
    lgbm_step = severity_pipeline.named_steps["lgbm"]
    classes = list(lgbm_step.classes_)

    # Determine which class index to use
    if predicted_class in classes:
        class_idx = classes.index(predicted_class)
    else:
        # Fallback: use index 0 (only class available in small training sets)
        class_idx = 0

    # shap_values may be:
    #   - list of 2-D arrays (one per class, multiclass)
    #   - single 2-D ndarray (binary or single-class)
    if isinstance(shap_vals, list):
        sv = shap_vals[class_idx][0]
    else:
        # Single class or binary: ndarray shape (n_samples, n_features)
        sv = shap_vals[0]

    return _top5_drivers(sv, feature_names)


def explain_risk(
    explainer: shap.TreeExplainer,
    risk_pipeline: Pipeline,
    features: dict,
) -> list[dict]:
    """Return top-5 SHAP drivers for a binary risk model (positive class)."""
    from src.risk_model import safe_df, _RISK_FEATURES
    from src.model import CAT_COLS, NUM_COLS
    row = safe_df(pd.DataFrame([features]))
    X_pre = risk_pipeline.named_steps["pre"].transform(row[_RISK_FEATURES])
    # ColumnTransformer orders: CAT_COLS first, then NUM_COLS
    feature_names = CAT_COLS + NUM_COLS

    shap_vals = explainer.shap_values(X_pre)
    # Binary LightGBM: list of two arrays (one per class) or single 2-D array
    if isinstance(shap_vals, list):
        # Take positive class (index 1); if only one array, use index 0
        sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        sv = shap_vals[0]

    return _top5_drivers(sv, feature_names)
