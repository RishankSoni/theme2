# tests/test_model.py
import pandas as pd
import pytest
from src.model import train_model, evaluate_cv, evaluate_test, predict, get_knn_neighbors
from src.baseline import (
    compute_window_counts, compute_corridor_baselines,
    compute_excess_scores, compute_tertile_thresholds, label_severity,
)

def _prepare(sample_df):
    """Add all columns needed for model training."""
    def _hour_to_band(h):
        if h < 6:   return "night"
        if h < 12:  return "morning"
        if h < 18:  return "afternoon"
        return "evening"
    df = pd.concat([sample_df] * 15, ignore_index=True)
    df["day_of_week"] = df["start_datetime"].dt.dayofweek.astype(int)
    df["hour_of_day"] = df["start_datetime"].dt.hour.astype(int)
    df["hour_band"]   = df["hour_of_day"].apply(_hour_to_band)
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.upper()
        .map({"TRUE": True, "FALSE": False}).fillna(False)
    )
    df["window_count"] = compute_window_counts(df)
    baselines = compute_corridor_baselines(df, min_obs=1)
    df["impact_score"] = compute_excess_scores(df, baselines)
    low_t, high_t = compute_tertile_thresholds(df)
    df["severity"] = label_severity(df, low_t, high_t)
    return df

@pytest.fixture
def labeled_df(sample_df):
    return _prepare(sample_df)

def test_train_model_returns_pipeline(labeled_df):
    pipeline = train_model(labeled_df)
    assert hasattr(pipeline, "predict")

def test_predict_returns_valid_severity(labeled_df):
    pipeline = train_model(labeled_df)
    features = {
        "event_cause": "public_event",
        "event_type": "planned",
        "corridor": "CBD 2",
        "zone": "Central Zone 2",
        "police_station": "Cubbon Park",
        "hour_band": "evening",
        "hour_of_day": 18,
        "day_of_week": 0,
        "requires_road_closure": False,
    }
    severity, confidence = predict(pipeline, features)
    assert severity in {"LOW", "MEDIUM", "HIGH"}
    assert abs(sum(confidence.values()) - 1.0) < 1e-6

def test_evaluate_cv_returns_float(labeled_df):
    score = evaluate_cv(labeled_df)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_evaluate_test_returns_float(labeled_df):
    from src.pipeline import split_data
    test = split_data(labeled_df)[2]
    pipeline = train_model(labeled_df)
    score = evaluate_test(pipeline, test)
    assert isinstance(score, float)

def test_knn_neighbors_returns_k_rows(labeled_df):
    query = {
        "event_cause": "public_event",
        "event_type": "planned",
        "corridor": "CBD 2",
        "zone": "Central Zone 2",
        "police_station": "Cubbon Park",
        "hour_band": "evening",
        "hour_of_day": 18,
        "day_of_week": 0,
        "requires_road_closure": False,
    }
    neighbors = get_knn_neighbors(labeled_df, query, k=3)
    assert len(neighbors) == 3
    assert "severity" in neighbors.columns
    assert "impact_score" in neighbors.columns
