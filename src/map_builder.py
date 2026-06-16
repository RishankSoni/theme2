# src/map_builder.py
import folium
import pandas as pd

_SEVERITY_COLOR  = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
_SEVERITY_RADIUS = {"LOW": 500,     "MEDIUM": 1000,     "HIGH": 2000}


def _junction_coords(df: pd.DataFrame, junction: str):
    sub: pd.DataFrame = df[df["junction"] == junction]  # type: ignore[assignment]
    sub = sub.dropna(subset=["latitude", "longitude"])  # type: ignore[assignment]
    if sub.empty:
        return None
    return (float(sub["latitude"].mean()), float(sub["longitude"].mean()))  # type: ignore[arg-type]


def _corridor_centroid(df: pd.DataFrame, corridor: str):
    sub: pd.DataFrame = df[df["corridor"] == corridor]  # type: ignore[assignment]
    sub = sub.dropna(subset=["latitude", "longitude"])  # type: ignore[assignment]
    if sub.empty:
        return None
    return (float(sub["latitude"].mean()), float(sub["longitude"].mean()))  # type: ignore[arg-type]


def build_map(
    event_lat: float,
    event_lng: float,
    severity: str,
    barricade_junctions: list,
    diversion_corridors: list,
    officer_info: dict,
    train_df: pd.DataFrame,
    event_name: str = "Event",
) -> folium.Map:
    color  = _SEVERITY_COLOR[severity]
    radius = _SEVERITY_RADIUS[severity]

    m = folium.Map(location=[event_lat, event_lng], zoom_start=14)

    # Impact zone
    folium.Circle(
        location=[event_lat, event_lng],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.15,
        popup=f"{severity} impact zone ({radius}m radius)",
    ).add_to(m)

    # Event epicenter
    folium.Marker(
        location=[event_lat, event_lng],
        popup=f"{event_name}<br>Severity: {severity}<br>"
              f"Officers: {officer_info['total_min']}-{officer_info['total_max']}",
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

    # Barricade positions
    for junction in barricade_junctions:
        coords = _junction_coords(train_df, junction)
        if coords:
            folium.Marker(
                location=list(coords),
                popup=f"Barricade: {junction}",
                icon=folium.Icon(color="red", icon="remove-sign"),
            ).add_to(m)

    # Diversion routes (event epicenter to corridor centroid, dashed)
    for corridor in diversion_corridors:
        coords = _corridor_centroid(train_df, corridor)
        if coords:
            folium.PolyLine(
                locations=[[event_lat, event_lng], list(coords)],
                color="blue",
                weight=3,
                dash_array="10",
                popup=f"Divert via {corridor}",
            ).add_to(m)

    return m
