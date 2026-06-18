# tests/test_map_builder.py
import pytest
import folium  # type: ignore[import]
from src.map_builder import build_map


def test_build_map_returns_folium_map(sample_df, bengaluru_graph):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="HIGH",
        barricade_junctions=["QueensStatueCircle"],
        diversion_corridors=["ORR East 1"],
        officer_info={"total_min": 10, "total_max": 12},
        train_df=sample_df,
        event_name="Test Event",
        G=bengaluru_graph,
    )
    assert isinstance(m, folium.Map)


def test_build_map_low_severity_uses_green(sample_df, bengaluru_graph):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="LOW",
        barricade_junctions=[],
        diversion_corridors=[],
        officer_info={"total_min": 2, "total_max": 4},
        train_df=sample_df,
        event_name="Small Event",
        G=bengaluru_graph,
    )
    assert isinstance(m, folium.Map)


def test_build_map_has_polyline_for_diversion(sample_df, bengaluru_graph):
    m = build_map(
        event_lat=12.97,
        event_lng=77.59,
        severity="MEDIUM",
        barricade_junctions=[],
        diversion_corridors=["ORR East 1"],
        officer_info={"total_min": 4, "total_max": 6},
        train_df=sample_df,
        event_name="Diversion Test",
        G=bengaluru_graph,
    )
    assert isinstance(m, folium.Map)
    children = list(m._children.values())
    has_polyline = any(isinstance(c, folium.PolyLine) for c in children)
    assert has_polyline, "Expected a PolyLine for the diversion route"
