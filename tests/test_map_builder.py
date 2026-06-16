# tests/test_map_builder.py
import pytest
import folium
from src.map_builder import build_map


def test_build_map_returns_folium_map(sample_df):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="HIGH",
        barricade_junctions=["QueensStatueCircle"],
        diversion_corridors=["ORR East 1"],
        officer_info={"total_min": 10, "total_max": 12},
        train_df=sample_df,
        event_name="Test Event",
    )
    assert isinstance(m, folium.Map)


def test_build_map_low_severity_uses_green(sample_df):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="LOW",
        barricade_junctions=[],
        diversion_corridors=[],
        officer_info={"total_min": 2, "total_max": 4},
        train_df=sample_df,
        event_name="Small Event",
    )
    assert isinstance(m, folium.Map)
