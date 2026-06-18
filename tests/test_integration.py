# tests/test_integration.py
"""
Smoke test: full pipeline from raw CSV to prediction and recommendation.
Does not start Streamlit — exercises all four layers directly.
"""
import pandas as pd
import pytest
import folium  # type: ignore[import]
from pathlib import Path

from src.pipeline import load_raw, split_data, corridor_metadata
from src.baseline import (
    compute_window_counts, compute_corridor_baselines,
    compute_excess_scores, compute_tertile_thresholds, label_severity,
)
from src.model import train_model, evaluate_cv, evaluate_test, predict, get_knn_neighbors
from src.recommender import officer_count, barricade_positions, build_diversion_graph, get_diversions
from src.map_builder import build_map
from src.road_network import load_graph


@pytest.fixture(scope="module")
def trained_state():
    df = load_raw()
    df["window_count"] = compute_window_counts(df)
    train_df, val_df, test_df = split_data(df)
    baselines = compute_corridor_baselines(train_df)
    for split in [train_df, val_df, test_df]:
        split["impact_score"] = compute_excess_scores(split, baselines)
    low_t, high_t = compute_tertile_thresholds(train_df)
    for split in [train_df, val_df, test_df]:
        split["severity"] = label_severity(split, low_t, high_t)
    pipeline = train_model(train_df)
    diversion_graph = build_diversion_graph(
        pd.concat([train_df, val_df], ignore_index=True)
    )
    graph = load_graph(Path("data/bengaluru_drive.graphml"))
    return dict(
        train_df=train_df,
        test_df=test_df,
        pipeline=pipeline,
        diversion_graph=diversion_graph,
        graph=graph,
    )


def test_test_f1_above_minimum_bar(trained_state):
    score = evaluate_test(trained_state["pipeline"], trained_state["test_df"])
    assert score >= 0.50, (
        f"Test macro-F1 = {score:.3f} is below minimum bar of 0.50. "
        "Consider switching corridor encoding to geospatial clusters."
    )


def test_end_to_end_cbd2_scenario(trained_state):
    train_df        = trained_state["train_df"]
    pipeline        = trained_state["pipeline"]
    diversion_graph = trained_state["diversion_graph"]
    graph           = trained_state["graph"]

    corridor = "CBD 2"
    zone, police, lat, lng = corridor_metadata(train_df, corridor)

    features = {
        "event_cause":           "public_event",
        "event_type":            "planned",
        "corridor":              corridor,
        "zone":                  zone,
        "police_station":        police,
        "hour_band":             "evening",
        "hour_of_day":           18,
        "day_of_week":           0,
        "requires_road_closure": False,
        "priority":              "High",
        "junction":              "unknown",
        "month":                 2,
        "is_weekend":            0,
    }

    severity, confidence = predict(pipeline, features)
    assert severity in {"LOW", "MEDIUM", "HIGH"}
    assert abs(sum(confidence.values()) - 1.0) < 1e-6

    barricades = barricade_positions(train_df, corridor, event_lat=lat, event_lng=lng, top_n=4)
    n_adj      = min(3, len(barricades))
    officers   = officer_count(severity, n_adjacent_junctions=n_adj)
    assert officers["total_min"] >= 2

    diversions = get_diversions(diversion_graph, corridor, features["hour_band"])

    fmap = build_map(
        lat, lng, severity, barricades, diversions, officers,
        train_df, "CBD 2 Rally", graph,
    )
    assert isinstance(fmap, folium.Map)

    neighbors = get_knn_neighbors(train_df, features, k=5)
    assert len(neighbors) == 5
