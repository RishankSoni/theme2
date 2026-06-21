# src/explainer.py
import warnings
from typing import Any

import numpy as np
import pandas as pd
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


def _import_shap() -> Any:
    # Import SHAP lazily to avoid heavy startup cost and runtime incompatibility crashes.
    import shap
    return shap


def build_explainers(severity_pipeline: Pipeline, risk_models: dict) -> dict:
    """Build TreeExplainer instances for severity, congestion, law-and-order models."""
    shap = _import_shap()
    return {
        "severity":  shap.TreeExplainer(severity_pipeline.named_steps["lgbm"]),
        "congestion": shap.TreeExplainer(risk_models["congestion"].named_steps["clf"]),
        "law_order":  shap.TreeExplainer(risk_models["law_order"].named_steps["clf"]),
    }


def _extract_class_shap(shap_vals, class_idx: int, n_features: int) -> np.ndarray:
    """Return a 1-D array of SHAP values for one class, handling all SHAP return formats."""
    arr = np.asarray(shap_vals)
    # object array or list: each element is (n_samples, n_features) for one class
    if arr.dtype == object or isinstance(shap_vals, list):
        per_class = list(shap_vals)
        idx = class_idx if class_idx < len(per_class) else 0
        return np.asarray(per_class[idx]).ravel()[:n_features]
    # (n_samples, n_features, n_classes)
    if arr.ndim == 3:
        return arr[0, :, class_idx]
    # (n_samples, n_features) — binary / single class
    if arr.ndim == 2 and arr.shape[-1] == n_features:
        return arr[0]
    # (n_features, n_classes) — squeezed multiclass single sample
    if arr.ndim == 2 and arr.shape[0] == n_features:
        return arr[:, class_idx]
    # fallback: flatten whatever we have to n_features
    return arr.ravel()[:n_features]


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
    explainer: Any,
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
        warnings.warn(
            f"predicted_class '{predicted_class}' not in model classes {list(lgbm_step.classes_)}; "
            f"falling back to class index 0 for SHAP explanation"
        )
        class_idx = 0

    # Normalise: newer SHAP returns numpy object array instead of list
    sv = _extract_class_shap(shap_vals, class_idx, len(feature_names))

    return _top5_drivers(sv, feature_names)


def explain_risk(
    explainer: Any,
    risk_pipeline: Pipeline,
    features: dict,
) -> list[dict]:
    """Return top-5 SHAP drivers for a binary risk model (positive class)."""
    from src.risk_model import safe_df, _RISK_FEATURES
    row = safe_df(pd.DataFrame([features]))
    X_pre = risk_pipeline.named_steps["pre"].transform(row[_RISK_FEATURES])
    # Feature order matches _RISK_FEATURES definition in risk_model
    feature_names = _RISK_FEATURES

    shap_vals = explainer.shap_values(X_pre)
    sv = _extract_class_shap(shap_vals, 1, len(feature_names))

    return _top5_drivers(sv, feature_names)
