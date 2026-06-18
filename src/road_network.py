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


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def route_coords(G: nx.MultiDiGraph, orig_node: int, dest_node: int) -> list:
    """Return road-following (lat, lng) waypoints between two OSM node IDs via Dijkstra."""
    node_ids = nx.shortest_path(G, orig_node, dest_node, weight="length")
    return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in node_ids]


def corridor_route_coords(G: nx.MultiDiGraph, df: pd.DataFrame, corridor: str) -> list:
    """Return road-following waypoints along the full length of a named corridor.

    Finds the two most geographically distant historical events on the corridor,
    snaps both to OSM nodes, and returns the Dijkstra shortest path between them.
    """
    sub = (
        df[df["corridor"] == corridor]
        .dropna(subset=["latitude", "longitude"])
        .reset_index(drop=True)
    )
    if sub.empty:
        raise ValueError(f"No data for corridor '{corridor}'")
    if len(sub) > 50:
        sub = sub.sample(50, random_state=42).reset_index(drop=True)

    # Find the two most geographically distant rows (corridor start and end)
    max_dist, i_max, j_max = -1.0, 0, min(1, len(sub) - 1)
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            d = _haversine_km(
                float(sub.loc[i, "latitude"]), float(sub.loc[i, "longitude"]),
                float(sub.loc[j, "latitude"]), float(sub.loc[j, "longitude"]),
            )
            if d > max_dist:
                max_dist, i_max, j_max = d, i, j

    start_node = nearest_node(G, float(sub.loc[i_max, "latitude"]), float(sub.loc[i_max, "longitude"]))
    end_node   = nearest_node(G, float(sub.loc[j_max, "latitude"]), float(sub.loc[j_max, "longitude"]))
    return route_coords(G, start_node, end_node)
