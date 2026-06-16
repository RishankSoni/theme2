# src/model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, pairwise_distances

CAT_COLS = ["event_cause", "event_type", "corridor", "zone", "police_station", "hour_band"]
NUM_COLS = ["hour_of_day", "day_of_week", "requires_road_closure"]
ALL_FEATURE_COLS = CAT_COLS + NUM_COLS
TARGET_COL = "severity"


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
    ])


def _X(df: pd.DataFrame) -> pd.DataFrame:
    out: pd.DataFrame = df[ALL_FEATURE_COLS].copy()  # type: ignore[assignment]
    out["requires_road_closure"] = out["requires_road_closure"].astype(int)
    for col in CAT_COLS:
        col_s: pd.Series = out[col]  # type: ignore[assignment]
        out[col] = col_s.fillna("unknown").astype(str)
    return out


def train_model(train_df: pd.DataFrame) -> Pipeline:
    pipeline = _build_pipeline()
    pipeline.fit(_X(train_df), train_df[TARGET_COL])
    return pipeline


def evaluate_cv(train_df: pd.DataFrame, n_splits: int = 5) -> float:
    """Mean macro-F1 from stratified k-fold CV on the training set."""
    pipeline = _build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, _X(train_df), train_df[TARGET_COL],
        cv=cv, scoring="f1_macro"
    )
    return float(scores.mean())


def evaluate_test(pipeline: Pipeline, test_df: pd.DataFrame) -> float:
    """Macro-F1 on the held-out test set. Call exactly once."""
    y_pred = pipeline.predict(_X(test_df))
    return float(f1_score(test_df[TARGET_COL], y_pred, average="macro"))


def predict(pipeline: Pipeline, features: dict) -> tuple:
    """Returns (severity_class: str, confidence: dict[str, float])."""
    row = pd.DataFrame([features])
    row["requires_road_closure"] = row["requires_road_closure"].astype(int)
    for col in CAT_COLS:
        col_s: pd.Series = row[col]  # type: ignore[assignment]
        row[col] = col_s.fillna("unknown").astype(str)
    severity   = str(pipeline.predict(row[ALL_FEATURE_COLS])[0])
    proba      = pipeline.predict_proba(row[ALL_FEATURE_COLS])[0]
    confidence = {str(c): float(p) for c, p in zip(pipeline.classes_, proba)}
    return severity, confidence


def get_knn_neighbors(train_df: pd.DataFrame, query_features: dict, k: int = 5) -> pd.DataFrame:
    """Return k most similar historical events for the evidence panel."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    feature_df = _X(train_df)
    X_train = enc.fit_transform(feature_df)

    query_row = pd.DataFrame([query_features])
    query_row["requires_road_closure"] = query_row["requires_road_closure"].astype(int)
    for col in CAT_COLS:
        col_s: pd.Series = query_row[col]  # type: ignore[assignment]
        query_row[col] = col_s.fillna("unknown").astype(str)
    X_query = enc.transform(query_row[ALL_FEATURE_COLS])

    dists = pairwise_distances(X_query, X_train)[0]
    top_k = np.argsort(dists)[:k]
    result: pd.DataFrame = train_df.iloc[top_k][
        ["corridor", "start_datetime", TARGET_COL, "impact_score", "event_cause"]
    ].copy()
    result["distance"] = dists[top_k]
    return result.reset_index(drop=True)
