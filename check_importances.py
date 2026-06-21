import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.pipeline import load_raw, split_data
from src.baseline import (compute_window_counts, compute_corridor_baselines,
                           compute_excess_scores, compute_tertile_thresholds, label_severity)
from src.model import train_model, evaluate_cv, evaluate_test, _X, ALL_FEATURE_COLS

df = load_raw()
df['window_count'] = compute_window_counts(df)
train_df, val_df, test_df = split_data(df)
baselines = compute_corridor_baselines(train_df)
for s in [train_df, val_df, test_df]:
    s['impact_score'] = compute_excess_scores(s, baselines)
low_t, high_t = compute_tertile_thresholds(train_df)
for s in [train_df, val_df, test_df]:
    s['severity'] = label_severity(s, low_t, high_t)

old_params = {
    'n_estimators': 224, 'num_leaves': 200,
    'learning_rate': 0.2985879580529471, 'min_child_samples': 5,
    'reg_alpha': 3.016516532940732e-08, 'reg_lambda': 5.151065907260535e-08,
    'subsample': 0.6962164633886399, 'colsample_bytree': 0.7913373860606256,
}
pipeline = train_model(train_df, params=old_params)

# Feature importances
lgbm = pipeline.named_steps['lgbm']
cst  = pipeline.named_steps['corridor_stats']
X_t  = cst.transform(_X(train_df))
feat_names = list(X_t.columns)
imps = lgbm.feature_importances_
sorted_idx = np.argsort(imps)[::-1]
print('Feature importances (split count, descending):')
for i in sorted_idx:
    bar = '#' * (imps[i] // 5)
    print(f'  {feat_names[i]:<30} {imps[i]:>5}  {bar}')

auth_vc = train_df['authenticated'].value_counts().to_dict()
print(f'\nauthenticated value counts: {auth_vc}')
print(f'authenticated=1 rate: {train_df["authenticated"].mean()*100:.1f}%  (near-constant!)')

# ── Re-tune with new feature set ──────────────────────────────────────────────
print()
print('=' * 60)
print('Re-tuning hyperparameters for new feature set (50 trials)...')
print('=' * 60)

from src.tuner import tune_lgbm
best = tune_lgbm(train_df, n_trials=50)
print('Best params:', best)

pipeline2 = train_model(train_df, params=best)
cv2   = evaluate_cv(train_df, params=best)
test2 = evaluate_test(pipeline2, test_df)

print()
print(f'BASELINE (13 feat, old params)  cv=0.7147  test=0.7407')
print(f'NEW feat, old params            cv=0.7000  test=0.7099')
print(f'NEW feat, re-tuned ({len(ALL_FEATURE_COLS)} feat)    cv={cv2:.4f}  test={test2:.4f}')
print(f'Delta vs baseline               cv={cv2-0.7147:+.4f}  test={test2-0.7407:+.4f}')

# Per-class breakdown
from sklearn.metrics import classification_report
y_true = test_df['severity']
y_pred = pipeline2.predict(_X(test_df))
print()
print(classification_report(y_true, y_pred, target_names=['HIGH', 'LOW', 'MEDIUM']))
