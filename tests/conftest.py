# tests/conftest.py
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "event_type": ["planned", "unplanned", "unplanned", "planned", "unplanned", "unplanned"],
        "event_cause": ["public_event", "vehicle_breakdown", "accident", "construction", "tree_fall", "vehicle_breakdown"],
        "latitude":  [12.97, 12.95, 13.00, 12.98, 12.92, 13.04],
        "longitude": [77.59, 77.58, 77.60, 77.61, 77.58, 77.52],
        "corridor": ["CBD 2", "ORR East 1", "CBD 2", "Tumkur Road", "ORR East 1", "CBD 2"],
        "zone": ["Central Zone 2", "East Zone 1", "Central Zone 2", "North Zone 1", "East Zone 1", "Central Zone 2"],
        "police_station": ["Cubbon Park", "Bellandur", "Cubbon Park", "Yeshwanthpura", "Bellandur", "Cubbon Park"],
        "junction": ["QueensStatueCircle", "KadubisanahalliFlyover", None, "GorguntePalyaJunc", None, "QueensStatueCircle"],
        "start_datetime": pd.to_datetime([
            "2024-02-12 18:00:00+00:00",
            "2024-01-30 09:00:00+00:00",
            "2024-02-12 17:30:00+00:00",
            "2024-03-07 08:00:00+00:00",
            "2024-01-30 09:30:00+00:00",
            "2024-02-12 19:00:00+00:00",
        ]),
        "closed_datetime": pd.to_datetime([
            "2024-02-12 20:00:00+00:00",
            "2024-01-30 11:00:00+00:00",
            "2024-02-12 19:30:00+00:00",
            "2024-03-07 10:00:00+00:00",
            "2024-01-30 11:00:00+00:00",
            "2024-02-12 21:00:00+00:00",
        ]),
        "requires_road_closure": ["FALSE", "FALSE", "FALSE", "TRUE", "FALSE", "TRUE"],
        "priority": ["High", "High", "Low", "High", "Low", "High"],
        "status": ["closed", "closed", "closed", "closed", "closed", "closed"],
        # EDA-derived columns
        "authenticated": [1, 0, 1, 1, 0, 1],
        "veh_type": ["heavy_vehicle", "lcv", "unknown", "unknown", "heavy_vehicle", "private_bus"],
        "description": [
            "traffic slow movement near junction",
            "vehicle breakdown on road",
            "accident near signal heavy traffic",
            "road block due to construction work",
            "tree fall on road slow movement",
            "traffic jam starting problem sir",
        ],
    })

@pytest.fixture(scope="session")
def bengaluru_graph():
    from src.road_network import load_graph
    from pathlib import Path
    return load_graph(Path("data/bengaluru_drive.graphml"))
