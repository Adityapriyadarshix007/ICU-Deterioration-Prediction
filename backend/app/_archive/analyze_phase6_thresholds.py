#!/usr/bin/env python3
"""
Compute recall/precision at multiple thresholds from Phase 6 predictions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score
)

sys.path.insert(0, str(Path(__file__).parent))
from config import TABLE_DIR

# Load predictions
df = pd.read_csv(TABLE_DIR / 'phase6_best_predictions.csv')
y_true = df['y_true'].values
y_prob = df['y_prob'].values

print("=" * 70)
print("THRESHOLD TRADE-OFF ANALYSIS")
print("=" * 70)
print(f"Test samples:  {len(y_true):,}")
print(f"Positives:     {y_true.sum():,} ({y_true.mean()*100:.1f}%)")
print()

# AUC and AUPRC
auc = roc_auc_score(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
baseline_auprc = y_true.mean()

print(f"AUC:    {auc:.4f}")
print(f"AUPRC:  {auprc:.4f}  (baseline = {baseline_auprc:.4f})")
print()

# Threshold sweep
print("=" * 70)
print(f"{'Thresh':>7} {'Recall':>8} {'Precision':>10} {'F1':>8} {'#Flagged':>10}")
print("=" * 70)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    n_flagged = int(y_pred.sum())
    if n_flagged == 0:
        print(f"{t:>7.2f} {'—':>8} {'—':>10} {'—':>8} {n_flagged:>10}")
        continue
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{t:>7.2f} {rec:>8.4f} {prec:>10.4f} {f1:>8.4f} {n_flagged:>10,}")

print("=" * 70)
print()

# What threshold achieves what recall?
print("=" * 70)
print("TARGET RECALL — What threshold is needed?")
print("=" * 70)

for target_recall in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    # Find the highest threshold that achieves at least this recall
    best_t = None
    for t in np.arange(0.99, 0.01, -0.01):
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        r = recall_score(y_true, y_pred)
        if r >= target_recall:
            best_t = t
            best_rec = r
            best_prec = precision_score(y_true, y_pred, zero_division=0)
            best_flagged = int(y_pred.sum())
            break
    if best_t is not None:
        print(f"  Recall ≥ {target_recall:.2f}:  "
              f"threshold={best_t:.2f}, "
              f"actual_recall={best_rec:.4f}, "
              f"precision={best_prec:.4f}, "
              f"flagged={best_flagged:,}")
    else:
        print(f"  Recall ≥ {target_recall:.2f}:  NOT ACHIEVABLE")

print("=" * 70)
print()

# Probability distribution by class
print("=" * 70)
print("PROBABILITY DISTRIBUTION BY CLASS")
print("=" * 70)

probs_neg = y_prob[y_true == 0]
probs_pos = y_prob[y_true == 1]

print(f"{'Percentile':>12} {'Stable (0)':>14} {'Progressor (1)':>16}")
print("-" * 70)
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"{p:>12} {np.percentile(probs_neg, p):>14.4f} "
          f"{np.percentile(probs_pos, p):>16.4f}")

print()
print(f"Mean:         {probs_neg.mean():>14.4f} {probs_pos.mean():>16.4f}")
print()
print("=" * 70)
