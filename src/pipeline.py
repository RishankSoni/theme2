# src/pipeline.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent.parent / "data" / "events.csv"

def _hour_to_band(hour: int) -> str:
    if hour < 6:   return "night"
    if hour < 12:  return "morning"
    if hour < 18:  return "afternoon"
    return "evening"

def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["start_datetime", "closed_datetime", "end_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=["start_datetime", "corridor"])
    df["hour_of_day"] = df["start_datetime"].dt.hour.astype(int)
    df["day_of_week"] = df["start_datetime"].dt.dayofweek.astype(int)
    df["hour_band"]   = df["hour_of_day"].apply(_hour_to_band)
    df["month"]      = df["start_datetime"].dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .astype(str).str.strip().str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
        .astype(object)
    )
    df["duration_h"] = (
        df["closed_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 3600
    df["duration_h"] = df["duration_h"].where(
        (df["duration_h"] > 0) & (df["duration_h"] <= 24)
    )
    for col in ["event_cause", "event_type", "corridor", "zone", "police_station", "junction", "priority"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df.reset_index(drop=True)

def split_data(df: pd.DataFrame, train_frac=0.70, val_frac=0.15, random_state=42):
    """Returns (train_df, val_df, test_df). 70/15/15 random split."""
    test_size = 1.0 - train_frac - val_frac          # 0.15
    val_size  = val_frac / (train_frac + val_frac)    # 0.15 / 0.85 ≈ 0.1765

    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val      = train_test_split(train_val, test_size=val_size, random_state=random_state)
    return train.copy(), val.copy(), test.copy()

def corridor_metadata(df: pd.DataFrame, corridor: str) -> tuple:
    """Returns (zone, police_station, mean_lat, mean_lng) for a corridor."""
    sub = df[df["corridor"] == corridor]
    if sub.empty:
        return ("unknown", "unknown", 12.97, 77.59)
    zone   = sub["zone"].mode().iloc[0]
    police = sub["police_station"].mode().iloc[0]
    lat    = sub["latitude"].mean()
    lng    = sub["longitude"].mean()
    return zone, police, lat, lng
