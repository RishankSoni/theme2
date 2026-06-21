# src/app_cache.py
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from src.baseline import (
    compute_corridor_baselines, compute_excess_scores,
    compute_tertile_thresholds, compute_window_counts, label_severity,
)
from src.duration_model import train_duration_model
from src.model import evaluate_cv, evaluate_test, train_model
from src.pipeline import load_raw, split_data
from src.recommender import build_diversion_graph
from src.risk_model import train_risk_models
from src.road_network import load_graph


@st.cache_resource(show_spinner="Loading road network...")
def get_road_graph() -> nx.MultiDiGraph:
    return load_graph(Path("data/bengaluru_drive.graphml"))


@st.cache_data(show_spinner="Loading data and training models...")
def load_and_train() -> dict:
    df = load_raw()
    df["window_count"] = compute_window_counts(df)

    train_df, val_df, test_df = split_data(df)

    baselines = compute_corridor_baselines(train_df)
    for split in [train_df, val_df, test_df]:
        split["impact_score"] = compute_excess_scores(split, baselines)

    low_t, high_t = compute_tertile_thresholds(train_df)
    for split in [train_df, val_df, test_df]:
        split["severity"] = label_severity(split, low_t, high_t)

    best_params = {
        "n_estimators":      224,
        "num_leaves":        200,
        "learning_rate":     0.2985879580529471,
        "min_child_samples": 5,
        "reg_alpha":         3.016516532940732e-08,
        "reg_lambda":        5.151065907260535e-08,
        "subsample":         0.6962164633886399,
        "colsample_bytree":  0.7913373860606256,
    }
    pipeline = train_model(train_df, params=best_params)
    cv_f1    = evaluate_cv(train_df, params=best_params)
    test_f1  = evaluate_test(pipeline, test_df)

    diversion_graph = build_diversion_graph(
        pd.concat([train_df, val_df], ignore_index=True)
    )
    dur_model   = train_duration_model(train_df)
    risk_models = train_risk_models(train_df)

    return {
        "train_df":        train_df,
        "baselines":       baselines,
        "low_t":           low_t,
        "high_t":          high_t,
        "pipeline":        pipeline,
        "cv_f1":           cv_f1,
        "test_f1":         test_f1,
        "diversion_graph": diversion_graph,
        "dur_model":       dur_model,
        "risk_models":     risk_models,
        "congestion_auc":  risk_models["congestion_auc"],
        "law_order_auc":   risk_models["law_order_auc"],
    }
