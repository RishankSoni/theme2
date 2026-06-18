# src/duration_model.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import lightgbm as lgb

from src.model import CAT_COLS, NUM_COLS

_DURATION_FEATURES = CAT_COLS + NUM_COLS  # 13 features, same as severity model


def _to_float(X):
    return X.astype(float)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", FunctionTransformer(_to_float), NUM_COLS),
    ])


def duration_tertile_thresholds(train_df: pd.DataFrame) -> tuple:
    """Returns (low_thresh, high_thresh) from training duration_h distribution."""
    valid = train_df["duration_h"].dropna()
    return float(valid.quantile(1 / 3)), float(valid.quantile(2 / 3))


def _bucket(values, low_thresh: float, high_thresh: float) -> list:
    result = []
    for v in values:
        if v <= low_thresh:
            result.append("SHORT")
        elif v <= high_thresh:
            result.append("MEDIUM")
        else:
            result.append("LONG")
    return result


def compute_duration_labels(df: pd.DataFrame, low_thresh: float, high_thresh: float) -> pd.Series:
    """Classify duration_h into SHORT/MEDIUM/LONG using training tertile thresholds."""
    def _label(v):
        if pd.isna(v):
            return np.nan
        return _bucket([v], low_thresh, high_thresh)[0]
    return df["duration_h"].map(_label)


def train_duration_model(train_df: pd.DataFrame) -> dict:
    """
    Benchmark three models on valid-duration rows, pick the winner, refit on full training data.
    Returns dict with keys: pipeline, kind, low_thresh, high_thresh.
    kind is one of: 'classifier', 'regressor', 'baseline'.
    """
    if "start_datetime" in train_df.columns and "closed_datetime" in train_df.columns:
        _start = pd.to_datetime(train_df["start_datetime"], utc=True, errors="coerce")
        _end   = pd.to_datetime(train_df["closed_datetime"], utc=True, errors="coerce")
        _dur   = (_end - _start).dt.total_seconds() / 3600
        _dur   = _dur.where((_dur > 0) & (_dur <= 24))
        df     = train_df.assign(duration_h=_dur)
    else:
        df = train_df.copy()
    valid  = df.dropna(subset=["duration_h"]).copy()
    low_thresh, high_thresh = duration_tertile_thresholds(valid)
    valid["_dur_label"] = compute_duration_labels(valid, low_thresh, high_thresh)
    valid = valid.dropna(subset=["_dur_label"])

    X        = valid[_DURATION_FEATURES]
    y_label  = valid["_dur_label"]
    y_log    = np.log1p(valid["duration_h"])
    y_dur_h  = valid["duration_h"]

    X_tr, X_te, ylab_tr, ylab_te, ylog_tr, ylog_te, ydur_tr, ydur_te = train_test_split(
        X, y_label, y_log, y_dur_h, test_size=0.2, random_state=42
    )

    # Model A: LGBMClassifier (tertile labels)
    pipe_a = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )),
    ])
    pipe_a.fit(X_tr, ylab_tr)
    f1_a = f1_score(ylab_te, pipe_a.predict(X_te), average="macro", zero_division=0)

    # Model B: LGBMRegressor (log1p target)
    pipe_b = Pipeline([
        ("pre", _make_preprocessor()),
        ("reg", lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
    ])
    pipe_b.fit(X_tr, ylog_tr)
    mae_b = mean_absolute_error(ydur_te, np.expm1(pipe_b.predict(X_te)))

    # Model C: ExtraTreesRegressor (log1p target)
    pipe_c = Pipeline([
        ("pre", _make_preprocessor()),
        ("reg", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipe_c.fit(X_tr, ylog_tr)
    mae_c = mean_absolute_error(ydur_te, np.expm1(pipe_c.predict(X_te)))

    # Winner selection — refit on full data
    if f1_a > 0.45:
        winner = Pipeline([
            ("pre", _make_preprocessor()),
            ("clf", lgb.LGBMClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
            )),
        ])
        winner.fit(X, y_label)
        kind = "classifier"
    elif min(mae_b, mae_c) <= 2.0:
        if mae_b <= mae_c:
            winner = Pipeline([
                ("pre", _make_preprocessor()),
                ("reg", lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
            ])
            winner.fit(X, y_log)
        else:
            winner = Pipeline([
                ("pre", _make_preprocessor()),
                ("reg", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ])
            winner.fit(X, y_log)
        kind = "regressor"
    else:
        winner = str(y_label.mode()[0])  # most-frequent label as string
        kind = "baseline"

    return {
        "pipeline":    winner,
        "kind":        kind,
        "low_thresh":  low_thresh,
        "high_thresh": high_thresh,
    }


def predict_duration(dur_model: dict, features: dict) -> str:
    """Returns 'SHORT', 'MEDIUM', or 'LONG'."""
    if dur_model["kind"] == "baseline":
        return dur_model["pipeline"]  # already the most-frequent label string

    X = pd.DataFrame([features])[_DURATION_FEATURES]
    pipeline = dur_model["pipeline"]

    if dur_model["kind"] == "classifier":
        return str(pipeline.predict(X)[0])

    # regressor: back-transform log1p, then bucket
    log_pred = float(pipeline.predict(X)[0])
    dur_pred = float(np.expm1(log_pred))
    return _bucket([dur_pred], dur_model["low_thresh"], dur_model["high_thresh"])[0]
