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
