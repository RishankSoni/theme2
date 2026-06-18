# src/map_builder.py
from collections import deque

import folium  # type: ignore[import]
import networkx as nx
import pandas as pd

import src.road_network as road_network

_SEVERITY_COLOR  = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
_SEVERITY_RADIUS = {"LOW": 500,     "MEDIUM": 1000,     "HIGH": 2000}


def _junction_centroid(df: pd.DataFrame, junction: str):
    """Return mean (lat, lng) of all historical events at a named junction."""
    sub: pd.DataFrame = df[df["junction"] == junction]
    sub = sub.dropna(subset=["latitude", "longitude"])
    if sub.empty:
        return None
    return (float(sub["latitude"].mean()), float(sub["longitude"].mean()))


def _corridor_centroid(df: pd.DataFrame, corridor: str):
    """Return mean (lat, lng) of all historical events on a corridor."""
    sub: pd.DataFrame = df[df["corridor"] == corridor]
    sub = sub.dropna(subset=["latitude", "longitude"])
    if sub.empty:
        return None
    return (float(sub["latitude"].mean()), float(sub["longitude"].mean()))


def _snap_to_intersection(G: nx.MultiDiGraph, lat: float, lng: float) -> tuple:
    """Snap a lat/lng to the nearest road intersection (node degree >= 3) via BFS.

    Walks outward from the nearest node up to 3 hops until a true junction is found.
    Falls back to the original nearest node if no degree-3 node is reachable.
    """
    node_id = road_network.nearest_node(G, lat, lng)
    visited = {node_id}
    queue = deque([(node_id, 0)])
    while queue:
        nid, depth = queue.popleft()
        if G.degree(nid) >= 3:
            return (float(G.nodes[nid]["y"]), float(G.nodes[nid]["x"]))
        if depth < 3:
            neighbors = set(G.successors(nid)) | set(G.predecessors(nid))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return (float(G.nodes[node_id]["y"]), float(G.nodes[node_id]["x"]))


def build_map(
    event_lat: float,
    event_lng: float,
    severity: str,
    barricade_junctions: list,
    diversion_corridors: list,
    officer_info: dict,
    train_df: pd.DataFrame,
    event_name: str,
    G: nx.MultiDiGraph,
) -> folium.Map:
    color  = _SEVERITY_COLOR[severity]
    radius = _SEVERITY_RADIUS[severity]

    m = folium.Map(location=[event_lat, event_lng], zoom_start=14)
    all_coords = [[event_lat, event_lng]]

    # Impact zone
    folium.Circle(
        location=[event_lat, event_lng],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.15,
        popup=f"{severity} impact zone ({radius}m radius)",
    ).add_to(m)

    # Event epicenter marker
    folium.Marker(
        location=[event_lat, event_lng],
        popup=(
            f"{event_name}<br>Severity: {severity}<br>"
            f"Officers: {officer_info['total_min']}-{officer_info['total_max']}"
        ),
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

    # Barricade positions — snapped to real road intersections
    for junction in barricade_junctions:
        centroid = _junction_centroid(train_df, junction)
        if centroid is None:
            continue
        coords = _snap_to_intersection(G, centroid[0], centroid[1])
        folium.Marker(
            location=list(coords),
            popup=f"Barricade: {junction}",
            icon=folium.Icon(color="red", icon="remove-sign"),
        ).add_to(m)
        all_coords.append(list(coords))

    # Diversion routes — road-following polylines via Dijkstra on OSM graph
    for corridor in diversion_corridors:
        path = road_network.corridor_route_coords(G, train_df, corridor)
        folium.PolyLine(
            locations=path,
            color="blue",
            weight=5,
            opacity=0.8,
            tooltip=f"Diversion → {corridor}",
            popup=f"Divert via: {corridor}",
        ).add_to(m)
        centroid = _corridor_centroid(train_df, corridor)
        if centroid:
            folium.Marker(
                location=list(centroid),
                tooltip=f"Diversion → {corridor}",
                popup=f"Diversion route: {corridor}",
                icon=folium.Icon(color="blue", icon="share-alt"),
            ).add_to(m)
            all_coords.extend(path)

    if len(all_coords) > 1:
        m.fit_bounds(all_coords)

    return m
