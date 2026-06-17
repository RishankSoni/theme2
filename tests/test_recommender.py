# tests/test_recommender.py
import pandas as pd
import pytest
from src.recommender import (
    officer_count, barricade_positions, build_diversion_graph, get_diversions
)
from src.baseline import compute_window_counts, compute_corridor_baselines, compute_excess_scores


def _add_features(sample_df):
    """Add columns needed by baseline functions."""
    def _hour_to_band(h):
        if h < 6:   return "night"
        if h < 12:  return "morning"
        if h < 18:  return "afternoon"
        return "evening"
    df = sample_df.copy()
    df["day_of_week"] = df["start_datetime"].dt.dayofweek.astype(int)
    df["hour_of_day"] = df["start_datetime"].dt.hour.astype(int)
    df["hour_band"]   = df["hour_of_day"].apply(_hour_to_band)
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.upper()
        .map({"TRUE": True, "FALSE": False}).fillna(False)
    )
    return df


def test_officer_count_high_severity():
    result = officer_count("HIGH", n_adjacent_junctions=2)
    assert result["primary_min"] == 8
    assert result["primary_max"] == 12
    assert result["adjacent_total"] == 4
    assert result["total_min"] == 12
    assert result["total_max"] == 16


def test_officer_count_low_severity():
    result = officer_count("LOW", n_adjacent_junctions=3)
    assert result["primary_min"] == 2
    assert result["primary_max"] == 4
    assert result["adjacent_total"] == 0


def test_barricade_positions_top_junctions(sample_df):
    df = _add_features(sample_df)
    # radius_km=1000 → no geospatial filtering → same as old behavior
    positions = barricade_positions(df, corridor="CBD 2",
                                    event_lat=12.97, event_lng=77.59,
                                    radius_km=1000.0, top_n=2)
    assert "QueensStatueCircle" in positions


def test_barricade_positions_empty_for_corridor_with_no_closures(sample_df):
    df = _add_features(sample_df)
    df["requires_road_closure"] = False
    positions = barricade_positions(df, corridor="CBD 2",
                                    event_lat=12.97, event_lng=77.59, top_n=2)
    assert positions == []


def test_barricade_positions_geospatial_filters_far_junctions():
    """With 2+ near junctions, far junctions must be excluded."""
    near_rows = pd.DataFrame({
        "corridor":              ["CBD 2"] * 4,
        "requires_road_closure": [True] * 4,
        "junction":              ["NearA", "NearA", "NearB", "NearB"],
        "latitude":              [12.97, 12.97, 12.98, 12.98],
        "longitude":             [77.59, 77.59, 77.60, 77.60],
    })
    far_row = pd.DataFrame({
        "corridor":              ["CBD 2"],
        "requires_road_closure": [True],
        "junction":              ["FarJunc"],
        "latitude":              [10.00],
        "longitude":             [76.00],
    })
    df = pd.concat([near_rows, far_row], ignore_index=True)
    positions = barricade_positions(df, "CBD 2",
                                    event_lat=12.97, event_lng=77.59,
                                    radius_km=2.0, top_n=4)
    assert "FarJunc" not in positions
    assert "NearA" in positions
    assert "NearB" in positions


def test_build_diversion_graph_returns_dict(sample_df):
    df = _add_features(sample_df)
    big = pd.concat([df] * 10, ignore_index=True)
    big["window_count"] = compute_window_counts(big)
    baselines = compute_corridor_baselines(big, min_obs=1)
    big["impact_score"] = compute_excess_scores(big, baselines)
    graph = build_diversion_graph(big, min_cooccurrences=1)
    assert isinstance(graph, dict)


def test_get_diversions_returns_list(sample_df):
    df = _add_features(sample_df)
    big = pd.concat([df] * 10, ignore_index=True)
    big["window_count"] = compute_window_counts(big)
    baselines = compute_corridor_baselines(big, min_obs=1)
    big["impact_score"] = compute_excess_scores(big, baselines)
    graph = build_diversion_graph(big, min_cooccurrences=1)
    divs = get_diversions(graph, "CBD 2")
    assert isinstance(divs, list)
    assert len(divs) <= 2
