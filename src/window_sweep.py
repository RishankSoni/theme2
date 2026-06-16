# src/window_sweep.py
"""Run once to confirm post_window=2h maximises CV macro-F1. Adjust baseline.py if another wins."""
import pandas as pd
from src.pipeline import load_raw, split_data
from src.baseline import (
    compute_window_counts, compute_corridor_baselines,
    compute_excess_scores, compute_tertile_thresholds, label_severity,
)
from src.model import evaluate_cv

POST_WINDOWS = [1.0, 1.5, 2.0, 2.5]

if __name__ == "__main__":
    df = load_raw()
    train_df, val_df, _ = split_data(df)

    for post_h in POST_WINDOWS:
        full = pd.concat([train_df, val_df], ignore_index=True)
        full["window_count"] = compute_window_counts(full, post_h=post_h)

        t_slice = full.iloc[:len(train_df)].copy()
        baselines = compute_corridor_baselines(t_slice)
        t_slice["impact_score"] = compute_excess_scores(t_slice, baselines)
        low_t, high_t = compute_tertile_thresholds(t_slice)
        t_slice["severity"] = label_severity(t_slice, low_t, high_t)

        # Need at least 3 classes for stratified CV; skip if degenerate
        if t_slice["severity"].nunique() < 2:
            print(f"post_window={post_h}h  ->  skipped (degenerate labels)")
            continue

        cv_f1 = evaluate_cv(t_slice)
        print(f"post_window={post_h}h  ->  CV macro-F1 = {cv_f1:.4f}")
