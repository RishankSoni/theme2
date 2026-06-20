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
