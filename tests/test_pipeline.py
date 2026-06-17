# tests/test_pipeline.py
import pandas as pd
import pytest
from pathlib import Path
from src.pipeline import load_raw, split_data, corridor_metadata

def test_load_raw_returns_dataframe(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "id,event_type,event_cause,latitude,longitude,corridor,zone,police_station,"
        "junction,start_datetime,closed_datetime,requires_road_closure,priority,status\n"
        "E1,planned,public_event,12.97,77.59,CBD 2,Central Zone 2,Cubbon Park,"
        "QueensStatue,2024-02-12 18:00:00+00:00,2024-02-12 20:00:00+00:00,FALSE,High,closed\n"
        "E2,unplanned,accident,12.95,77.58,ORR East 1,East Zone 1,Bellandur,"
        ",2024-01-30 09:00:00+00:00,2024-01-30 11:00:00+00:00,FALSE,High,closed\n"
    )
    df = load_raw(path=csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

def test_load_raw_drops_null_corridor(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "id,event_type,event_cause,latitude,longitude,corridor,zone,police_station,"
        "junction,start_datetime,closed_datetime,requires_road_closure,priority,status\n"
        "E1,planned,public_event,12.97,77.59,,Central Zone 2,Cubbon Park,"
        ",2024-02-12 18:00:00+00:00,2024-02-12 20:00:00+00:00,FALSE,High,closed\n"
        "E2,unplanned,accident,12.95,77.58,ORR East 1,East Zone 1,Bellandur,"
        ",2024-01-30 09:00:00+00:00,2024-01-30 11:00:00+00:00,FALSE,High,closed\n"
    )
    df = load_raw(path=csv)
    assert len(df) == 1
    assert df.iloc[0]["corridor"] == "ORR East 1"

def test_load_raw_adds_time_features(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "id,event_type,event_cause,latitude,longitude,corridor,zone,police_station,"
        "junction,start_datetime,closed_datetime,requires_road_closure,priority,status\n"
        "E1,planned,public_event,12.97,77.59,CBD 2,Central Zone 2,Cubbon Park,"
        ",2024-02-12 18:00:00+00:00,2024-02-12 20:00:00+00:00,FALSE,High,closed\n"
    )
    df = load_raw(path=csv)
    assert df.iloc[0]["hour_of_day"] == 18
    assert df.iloc[0]["day_of_week"] == 0  # Monday
    assert df.iloc[0]["hour_band"] == "evening"

def test_load_raw_parses_road_closure(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "id,event_type,event_cause,latitude,longitude,corridor,zone,police_station,"
        "junction,start_datetime,closed_datetime,requires_road_closure,priority,status\n"
        "E1,planned,construction,12.97,77.59,CBD 2,Central Zone 2,Cubbon Park,"
        ",2024-02-12 08:00:00+00:00,2024-02-12 10:00:00+00:00,TRUE,High,closed\n"
    )
    df = load_raw(path=csv)
    assert df.iloc[0]["requires_road_closure"] is True

def test_split_data_sizes(sample_df):
    import pandas as pd
    big = pd.concat([sample_df] * 20, ignore_index=True)
    train, val, test = split_data(big)
    total = len(train) + len(val) + len(test)
    assert total == len(big)
    assert abs(len(train) / total - 0.70) < 0.05
    assert abs(len(val)   / total - 0.15) < 0.05
    assert abs(len(test)  / total - 0.15) < 0.05

def test_load_raw_adds_month_and_weekend(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "id,event_type,event_cause,latitude,longitude,corridor,zone,police_station,"
        "junction,start_datetime,closed_datetime,requires_road_closure,priority,status\n"
        "E1,planned,public_event,12.97,77.59,CBD 2,Central Zone 2,Cubbon Park,"
        ",2024-02-12 18:00:00+00:00,2024-02-12 20:00:00+00:00,FALSE,High,closed\n"
        "E2,unplanned,accident,12.95,77.58,ORR East 1,East Zone 1,Bellandur,"
        ",2024-01-06 09:00:00+00:00,2024-01-06 11:00:00+00:00,FALSE,Low,closed\n"
    )
    df = load_raw(path=csv)
    assert df.iloc[0]["month"] == 2
    assert df.iloc[0]["is_weekend"] == 0   # Monday
    assert df.iloc[1]["month"] == 1
    assert df.iloc[1]["is_weekend"] == 1   # Saturday

def test_load_raw_normalises_priority(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "id,event_type,event_cause,latitude,longitude,corridor,zone,police_station,"
        "junction,start_datetime,closed_datetime,requires_road_closure,priority,status\n"
        "E1,planned,public_event,12.97,77.59,CBD 2,Central Zone 2,Cubbon Park,"
        ",2024-02-12 18:00:00+00:00,2024-02-12 20:00:00+00:00,FALSE,,closed\n"
        "E2,unplanned,accident,12.95,77.58,ORR East 1,East Zone 1,Bellandur,"
        ",2024-01-06 09:00:00+00:00,2024-01-06 11:00:00+00:00,FALSE,High,closed\n"
    )
    df = load_raw(path=csv)
    assert df.iloc[0]["priority"] == "unknown"
    assert df.iloc[1]["priority"] == "High"

def test_corridor_metadata_returns_zone(sample_df):
    zone, police, lat, lng = corridor_metadata(sample_df, "CBD 2")
    assert zone == "Central Zone 2"
    assert police == "Cubbon Park"
    assert abs(lat - sample_df[sample_df["corridor"] == "CBD 2"]["latitude"].mean()) < 0.001
    assert abs(lng - sample_df[sample_df["corridor"] == "CBD 2"]["longitude"].mean()) < 0.001

def test_load_raw_has_duration_h():
    df = load_raw()
    assert "duration_h" in df.columns
    valid = df["duration_h"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all()
    assert (valid <= 24).all()


def test_load_raw_duration_h_nan_when_no_closed_datetime():
    df = load_raw()
    # Rows where closed_datetime is NaT should have NaN duration_h
    missing_close = df[df["closed_datetime"].isna()]
    if not missing_close.empty:
        assert missing_close["duration_h"].isna().all()
