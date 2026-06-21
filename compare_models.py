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
from sklearn.metrics import classification_report

df = load_raw()
df['window_count'] = compute_window_counts(df)
train_df, val_df, test_df = split_data(df)
baselines = compute_corridor_baselines(train_df)
for s in [train_df, val_df, test_df]:
    s['impact_score'] = compute_excess_scores(s, baselines)
low_t, high_t = compute_tertile_thresholds(train_df)
for s in [train_df, val_df, test_df]:
    s['severity'] = label_severity(s, low_t, high_t)

print(f'Feature set ({len(ALL_FEATURE_COLS)} raw + 4 corridor-engineered = {len(ALL_FEATURE_COLS)+4} total):')
print(f'  CAT: {[c for c in ALL_FEATURE_COLS if c in ["event_cause","event_type","corridor","zone","police_station","hour_band","priority","junction"]]}')
print(f'  NUM: {[c for c in ALL_FEATURE_COLS if c not in ["event_cause","event_type","corridor","zone","police_station","hour_band","priority","junction"]]}')
print(f'  NLP hit rates: desc_traffic_slow={train_df["desc_traffic_slow"].mean()*100:.1f}%  desc_breakdown={train_df["desc_breakdown"].mean()*100:.1f}%')
print()

# ── Test with old params first ────────────────────────────────────────────────
old_params = {
    'n_estimators': 224, 'num_leaves': 200,
    'learning_rate': 0.2985879580529471, 'min_child_samples': 5,
    'reg_alpha': 3.016516532940732e-08, 'reg_lambda': 5.151065907260535e-08,
    'subsample': 0.6962164633886399, 'colsample_bytree': 0.7913373860606256,
}
p_old = train_model(train_df, params=old_params)
cv_old  = evaluate_cv(train_df, params=old_params)
test_old = evaluate_test(p_old, test_df)
print(f'Pruned features, old params:  cv={cv_old:.4f}  test={test_old:.4f}  (delta test={test_old-0.7407:+.4f})')

# ── Re-tune with pruned feature set ──────────────────────────────────────────
print('Re-tuning (75 trials) with pruned feature set...')
from src.tuner import tune_lgbm
best = tune_lgbm(train_df, n_trials=75)
print('Best params:', best)

p_new = train_model(train_df, params=best)
cv_new  = evaluate_cv(train_df, params=best)
test_new = evaluate_test(p_new, test_df)

print()
print('=' * 65)
print(f'BASELINE (13 feat, old params)  cv=0.7147  test=0.7407')
print(f'Pruned feat, old params         cv={cv_old:.4f}  test={test_old:.4f}  (delta={test_old-0.7407:+.4f})')
print(f'Pruned feat, re-tuned           cv={cv_new:.4f}  test={test_new:.4f}  (delta={test_new-0.7407:+.4f})')
print('=' * 65)

print()
print('Per-class breakdown (re-tuned):')
y_true = test_df['severity']
y_pred = p_new.predict(_X(test_df))
print(classification_report(y_true, y_pred, target_names=['HIGH', 'LOW', 'MEDIUM']))

# Feature importances (re-tuned)
lgbm = p_new.named_steps['lgbm']
cst  = p_new.named_steps['corridor_stats']
X_t  = cst.transform(_X(train_df))
feat_names = list(X_t.columns)
imps = lgbm.feature_importances_
sorted_idx = np.argsort(imps)[::-1]
print('Feature importances (re-tuned, descending):')
for i in sorted_idx:
    print(f'  {feat_names[i]:<30} {imps[i]:>6}')

print()
print('Best params to paste into app.py:')
print(best)
