# src/road_network.py
import math
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

# Bounding box for Bengaluru event data + ~0.03° buffer.
# OSMnx 2.x format: (west, south, east, north) = (min_lng, min_lat, max_lng, max_lat)
_BBOX = (77.28, 12.78, 77.80, 13.30)


def load_graph(cache_path: Path) -> nx.MultiDiGraph:
    """Load Bengaluru drive graph from GraphML cache; download and cache if absent.

    Downloads once (~60 s, internet required). All subsequent loads read from disk (~2 s).
    The graph is reduced to its largest strongly-connected component so every node
    is reachable from every other node — nx.shortest_path never raises NetworkXNoPath.
    """
    if cache_path.exists():
        return ox.load_graphml(cache_path)
    G: nx.MultiDiGraph = ox.graph_from_bbox(_BBOX, network_type="drive")
    sccs = list(nx.strongly_connected_components(G))
    largest_scc = max(sccs, key=len)
    G = G.subgraph(largest_scc).copy()
    ox.save_graphml(G, cache_path)
    return G


def nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    """Snap a lat/lng coordinate to the nearest OSM node in G.

    OSMnx convention: X=longitude, Y=latitude.
    """
    return int(ox.distance.nearest_nodes(G, X=lng, Y=lat))
