# src/risk_model.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import lightgbm as lgb

from src.model import CAT_COLS, NUM_COLS

_RISK_FEATURES: list = CAT_COLS + NUM_COLS  # 20 features

_HIGH_RISK_CAUSES = {"riot", "protest", "procession", "public_event", "vip_movement"}


def _to_float(X):
    return X.astype(float)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", FunctionTransformer(_to_float), NUM_COLS),
    ])


def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Inject all feature columns with safe defaults when absent."""
    df = df.copy()
    # Inject all numeric feature columns with 0 as default
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].fillna("unknown").astype(str)
    return df


def train_risk_models(train_df: pd.DataFrame) -> dict:
    """Train congestion and law-and-order classifiers on train_df.

    Returns dict with keys: congestion, law_order (fitted Pipelines),
    congestion_auc, law_order_auc (float, evaluated on 20% hold-out).
    """
    df = safe_df(train_df)

    # Labels
    p75 = df.groupby("corridor")["window_count"].transform(
        lambda s: s.quantile(0.75)
    )
    y_cong = (df["window_count"] > p75).astype(int)
    y_law  = df["event_cause"].isin(_HIGH_RISK_CAUSES).astype(int)

    X = df[_RISK_FEATURES]
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    yc_tr, yc_te = y_cong.iloc[idx_tr], y_cong.iloc[idx_te]
    yl_tr, yl_te = y_law.iloc[idx_tr], y_law.iloc[idx_te]

    # Evaluate on hold-out split
    pipe_cong_eval = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1
        )),
    ])
    pipe_cong_eval.fit(X_tr, yc_tr)
    cong_auc = roc_auc_score(yc_te, pipe_cong_eval.predict_proba(X_te)[:, 1])

    pipe_law_eval = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1
        )),
    ])
    pipe_law_eval.fit(X_tr, yl_tr)
    law_auc = roc_auc_score(yl_te, pipe_law_eval.predict_proba(X_te)[:, 1])

    # Final models fitted on full training data
    pipe_cong = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1
        )),
    ])
    pipe_cong.fit(X, y_cong)

    pipe_law = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1
        )),
    ])
    pipe_law.fit(X, y_law)

    return {
        "congestion":     pipe_cong,
        "law_order":      pipe_law,
        "congestion_auc": float(cong_auc),
        "law_order_auc":  float(law_auc),
    }


def predict_risks(risk_models: dict, features: dict) -> dict:
    """Return congestion_prob and law_order_prob for a single event dict."""
    row = safe_df(pd.DataFrame([features]))
    X = row[_RISK_FEATURES]
    cong_prob = float(risk_models["congestion"].predict_proba(X)[0][1])
    law_prob  = float(risk_models["law_order"].predict_proba(X)[0][1])
    return {"congestion_prob": cong_prob, "law_order_prob": law_prob}
