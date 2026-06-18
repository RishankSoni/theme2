# tests/test_road_network.py
import pytest
import networkx as nx
from pathlib import Path
from src.road_network import load_graph, nearest_node


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
