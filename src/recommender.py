# src/recommender.py
import math
import pandas as pd
import numpy as np

_OFFICER_TABLE = {
    "LOW":    {"primary": (2, 4),   "per_junction": 0},
    "MEDIUM": {"primary": (4, 6),   "per_junction": 1},
    "HIGH":   {"primary": (8, 12),  "per_junction": 2},
}


def officer_count(severity: str, n_adjacent_junctions: int) -> dict:
    """Returns officer deployment numbers for the given severity level."""
    spec = _OFFICER_TABLE[severity]
    lo, hi = spec["primary"]
    adj    = spec["per_junction"] * n_adjacent_junctions
    return {
        "primary_min":    lo,
        "primary_max":    hi,
        "adjacent_total": adj,
        "total_min":      lo + adj,
        "total_max":      hi + adj,
    }


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def barricade_positions(
    train_df: pd.DataFrame,
    corridor: str,
    event_lat: float,
    event_lng: float,
    radius_km: float = 2.0,
    top_n: int = 4,
) -> list:
    """Junctions near the event epicenter most frequently requiring road closure."""
    mask = (
        (train_df["corridor"] == corridor)
        & (train_df["requires_road_closure"] == True)  # noqa: E712
    )
    subset: pd.DataFrame = train_df[mask].copy()
    subset = subset[subset["junction"].notna()]
    subset = subset[subset["junction"] != "unknown"]
    if subset.empty:
        return []

    junction_centroids = (
        subset.dropna(subset=["latitude", "longitude"])
        .groupby("junction")[["latitude", "longitude"]]
        .mean()
    )

    def _nearby(r_km: float) -> list:
        return [
            junc for junc, row in junction_centroids.iterrows()
            if _haversine_km(event_lat, event_lng,
                             float(row["latitude"]), float(row["longitude"])) <= r_km
        ]

    for candidate_radius in [radius_km, radius_km * 2.5]:
        survivors = _nearby(candidate_radius)
        if len(survivors) >= 2:
            return (
                subset[subset["junction"].isin(survivors)]["junction"]
                .value_counts()
                .head(top_n)
                .index.tolist()
            )

    # Fallback: corridor-wide top-N (original behavior)
    return subset["junction"].value_counts().head(top_n).index.tolist()


def build_diversion_graph(
    train_val_df: pd.DataFrame,
    min_cooccurrences: int = 5,
) -> dict:
    """
    For each primary corridor C, find corridors D that see elevated incident
    counts in [t, t+1h] when C has an event — ranked by co_elevation ratio.
    co_elevation(C,D) = mean co-incidents on D / D's own mean window_count.
    """
    post = pd.Timedelta(hours=1)
    corridors = train_val_df["corridor"].dropna().unique()

    # Per-corridor mean window_count (denominator for normalization)
    d_means: dict = {}
    for corr_key, grp in train_val_df.groupby("corridor"):
        d_means[corr_key] = float(grp["window_count"].mean())  # type: ignore[arg-type]

    global_mean: float = train_val_df["window_count"].mean()  # type: ignore[assignment]
    if not global_mean:
        global_mean = 1.0

    # Accumulate co-incident counts: raw[C][D] = [count, count, ...]
    raw: dict = {c: {} for c in corridors}

    for idx, row in train_val_df.iterrows():
        C = row["corridor"]
        t = row["start_datetime"]

        co = train_val_df[
            (train_val_df.index != idx) &
            (train_val_df["corridor"] != C) &
            (train_val_df["start_datetime"] >= t) &
            (train_val_df["start_datetime"] <= t + post)
        ]
        for D, grp in co.groupby("corridor"):
            raw[C].setdefault(D, []).append(len(grp))

    # Convert to elevation scores and keep top-2 per corridor
    result = {}
    for C, neighbors in raw.items():
        elevations = {}
        for D, counts in neighbors.items():
            if len(counts) < min_cooccurrences:
                continue
            mean_count = float(np.mean(counts))
            d_baseline = float(d_means.get(D, global_mean)) or global_mean
            elevations[D] = mean_count / d_baseline
        top2 = sorted(elevations, key=lambda k: elevations[k], reverse=True)[:2]
        result[C] = top2

    return result


def get_diversions(diversion_graph: dict, corridor: str) -> list:
    """Return recommended diversion corridors for a given primary corridor."""
    return diversion_graph.get(corridor, [])
