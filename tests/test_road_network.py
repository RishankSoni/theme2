# tests/test_road_network.py
import pytest
import networkx as nx
from pathlib import Path
from src.road_network import load_graph, nearest_node, route_coords, corridor_route_coords


def test_load_graph_saves_and_reloads(tmp_path, bengaluru_graph):
    """Loading from a saved GraphML produces the same graph."""
    import osmnx as ox
    cache = tmp_path / "copy.graphml"
    ox.save_graphml(bengaluru_graph, cache)
    G2 = load_graph(cache)
    assert G2.number_of_nodes() == bengaluru_graph.number_of_nodes()


def test_nearest_node_returns_int(bengaluru_graph):
    node = nearest_node(bengaluru_graph, lat=12.97, lng=77.59)
    assert isinstance(node, int)


def test_nearest_node_in_graph(bengaluru_graph):
    node = nearest_node(bengaluru_graph, lat=12.97, lng=77.59)
    assert node in bengaluru_graph.nodes


def test_all_nodes_strongly_connected(bengaluru_graph):
    assert nx.is_strongly_connected(bengaluru_graph)


def test_route_coords_returns_road_following_list(bengaluru_graph):
    n1 = nearest_node(bengaluru_graph, 12.97, 77.59)
    n2 = nearest_node(bengaluru_graph, 12.98, 77.61)
    coords = route_coords(bengaluru_graph, n1, n2)
    assert len(coords) > 1
    assert all(len(c) == 2 for c in coords)


def test_route_coords_endpoints_near_inputs(bengaluru_graph):
    lat1, lng1, lat2, lng2 = 12.97, 77.59, 12.98, 77.61
    n1 = nearest_node(bengaluru_graph, lat1, lng1)
    n2 = nearest_node(bengaluru_graph, lat2, lng2)
    coords = route_coords(bengaluru_graph, n1, n2)
    # First waypoint within ~1 km of start; last within ~1 km of end
    assert abs(coords[0][0] - lat1) < 0.01
    assert abs(coords[-1][0] - lat2) < 0.01


def test_corridor_route_coords_returns_road_path(bengaluru_graph):
    from src.pipeline import load_raw
    df = load_raw()
    coords = corridor_route_coords(bengaluru_graph, df, "CBD 2")
    assert len(coords) > 1
    assert all(len(c) == 2 for c in coords)
