import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.pipeline import load_raw, split_data
from src.baseline import (compute_window_counts, compute_corridor_baselines,
                           compute_excess_scores, compute_tertile_thresholds, label_severity)
from src.model import train_model, _X

# ── Data prep ─────────────────────────────────────────────────────────────────
df = load_raw()
df['window_count'] = compute_window_counts(df)
train_df, val_df, test_df = split_data(df)
baselines = compute_corridor_baselines(train_df)
for s in [train_df, val_df, test_df]:
    s['impact_score'] = compute_excess_scores(s, baselines)
low_t, high_t = compute_tertile_thresholds(train_df)
for s in [train_df, val_df, test_df]:
    s['severity'] = label_severity(s, low_t, high_t)

best_params = {
    'n_estimators': 224, 'num_leaves': 200,
    'learning_rate': 0.2985879580529471, 'min_child_samples': 5,
    'reg_alpha': 3.016516532940732e-08, 'reg_lambda': 5.151065907260535e-08,
    'subsample': 0.6962164633886399, 'colsample_bytree': 0.7913373860606256,
}
pipeline = train_model(train_df, params=best_params)

# ── Probabilities on test set ─────────────────────────────────────────────────
CLASSES = ['HIGH', 'LOW', 'MEDIUM']
y_true  = test_df['severity'].values
y_prob  = pipeline.predict_proba(_X(test_df))   # shape (n, 3)
y_pred  = pipeline.predict(_X(test_df))

# Binarize for OvR ROC
y_bin = label_binarize(y_true, classes=CLASSES)  # (n, 3)

# ── Per-class ROC + AUC ───────────────────────────────────────────────────────
fpr, tpr, roc_auc = {}, {}, {}
for i, cls in enumerate(CLASSES):
    fpr[cls], tpr[cls], _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc[cls] = auc(fpr[cls], tpr[cls])

# Macro & weighted AUC
macro_auc    = roc_auc_score(y_bin, y_prob, average='macro',    multi_class='ovr')
weighted_auc = roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr')

# Micro-average (aggregate all classes)
fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
auc_micro = auc(fpr_micro, tpr_micro)

# ── Print results ─────────────────────────────────────────────────────────────
print('=' * 55)
print('ROC-AUC — Severity Classifier (one-vs-rest)')
print('=' * 55)
for cls in CLASSES:
    n_pos = (y_true == cls).sum()
    print(f'  {cls:<8}  AUC={roc_auc[cls]:.4f}  (n_positive={n_pos})')
print()
print(f'  Macro-average AUC    : {macro_auc:.4f}')
print(f'  Weighted-average AUC : {weighted_auc:.4f}')
print(f'  Micro-average AUC    : {auc_micro:.4f}')
print()
print('Reference: random classifier AUC = 0.5000')
print('Reference: perfect classifier AUC = 1.0000')
print()

# Per-class threshold analysis (Youden J = sensitivity + specificity - 1)
print('Optimal decision thresholds (Youden J statistic):')
for cls in CLASSES:
    fp, tp, thresholds = roc_curve(y_bin[:, CLASSES.index(cls)], y_prob[:, CLASSES.index(cls)])
    j_scores = tp - fp
    best_idx  = np.argmax(j_scores)
    best_thr  = thresholds[best_idx]
    best_sens = tp[best_idx]
    best_spec = 1 - fp[best_idx]
    print(f'  {cls:<8}  threshold={best_thr:.3f}  sensitivity={best_sens:.3f}  specificity={best_spec:.3f}  J={j_scores[best_idx]:.3f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {'HIGH': '#e74c3c', 'LOW': '#27ae60', 'MEDIUM': '#f39c12'}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC Curves — Traffic Severity Classifier (One-vs-Rest)', fontsize=13, fontweight='bold')

# Left: individual class curves
ax = axes[0]
for cls in CLASSES:
    ax.plot(fpr[cls], tpr[cls], color=COLORS[cls], lw=2,
            label=f'{cls}  (AUC={roc_auc[cls]:.3f})')
ax.plot(fpr_micro, tpr_micro, color='steelblue', lw=2, linestyle='--',
        label=f'Micro-avg  (AUC={auc_micro:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Per-Class ROC Curves')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])

# Right: zoom into top-left corner (high-performance region)
ax2 = axes[1]
for cls in CLASSES:
    ax2.plot(fpr[cls], tpr[cls], color=COLORS[cls], lw=2,
             label=f'{cls}  (AUC={roc_auc[cls]:.3f})')
ax2.plot(fpr_micro, tpr_micro, color='steelblue', lw=2, linestyle='--',
         label=f'Micro-avg  (AUC={auc_micro:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC — Zoomed (FPR 0-0.4)')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.01, 0.40])
ax2.set_ylim([0.40, 1.02])

plt.tight_layout()
plt.savefig('eda_report/roc_curves.png', dpi=150, bbox_inches='tight')
print()
print('Saved: eda_report/roc_curves.png')

# ── Precision-Recall AUC for completeness ─────────────────────────────────────
from sklearn.metrics import average_precision_score
print()
print('Precision-Recall AUC (handles class imbalance better):')
for i, cls in enumerate(CLASSES):
    ap = average_precision_score(y_bin[:, i], y_prob[:, i])
    prevalence = y_bin[:, i].mean()
    print(f'  {cls:<8}  AP={ap:.4f}  (prevalence={prevalence:.3f}, random_baseline={prevalence:.3f})')
macro_ap = average_precision_score(y_bin, y_prob, average='macro')
print(f'  Macro avg AP  : {macro_ap:.4f}')
