#!/usr/bin/env python3
"""
Full threshold comparison: Zero + LightGBM vs MICE + LightGBM.
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

# Load both prediction files
zero_df = pd.read_csv(TABLE_DIR / 'phase6_best_predictions.csv')
mice_df = pd.read_csv(TABLE_DIR / 'phase6_mice_predictions.csv')

y_true = zero_df['y_true'].values
z_prob = zero_df['y_prob'].values
m_prob = mice_df['y_prob'].values

# Sanity
assert np.array_equal(y_true, mice_df['y_true'].values), \
    "y_true mismatch between files!"

print("=" * 78)
print("ZERO + LightGBM  vs  MICE + LightGBM")
print("=" * 78)
print(f"Test samples: {len(y_true):,}")
print(f"Positives:    {y_true.sum():,} ({y_true.mean()*100:.1f}%)")
print()

# AUC / AUPRC comparison
z_auc = roc_auc_score(y_true, z_prob)
m_auc = roc_auc_score(y_true, m_prob)
z_auprc = average_precision_score(y_true, z_prob)
m_auprc = average_precision_score(y_true, m_prob)

print("-" * 78)
print(f"{'Metric':<20} {'ZERO':>15} {'MICE':>15} {'Δ(M-Z)':>15}")
print("-" * 78)
print(f"{'AUC':<20} {z_auc:>15.4f} {m_auc:>15.4f} {m_auc-z_auc:>+15.4f}")
print(f"{'AUPRC':<20} {z_auprc:>15.4f} {m_auprc:>15.4f} {m_auprc-z_auprc:>+15.4f}")
print()

# Threshold sweep
print("=" * 78)
print("THRESHOLD TRADE-OFF COMPARISON")
print("=" * 78)
print(f"{'Thresh':>7} | {'Z-Recall':>9} {'Z-Prec':>8} {'Z-F1':>8} | "
      f"{'M-Recall':>9} {'M-Prec':>8} {'M-F1':>8}")
print("-" * 78)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for t in thresholds:
    z_pred = (z_prob >= t).astype(int)
    m_pred = (m_prob >= t).astype(int)

    z_rec = recall_score(y_true, z_pred) if z_pred.sum() > 0 else 0
    m_rec = recall_score(y_true, m_pred) if m_pred.sum() > 0 else 0
    z_prec = precision_score(y_true, z_pred, zero_division=0)
    m_prec = precision_score(y_true, m_pred, zero_division=0)
    z_f1 = f1_score(y_true, z_pred, zero_division=0)
    m_f1 = f1_score(y_true, m_pred, zero_division=0)

    print(f"{t:>7.2f} | {z_rec:>9.4f} {z_prec:>8.4f} {z_f1:>8.4f} | "
          f"{m_rec:>9.4f} {m_prec:>8.4f} {m_f1:>8.4f}")

print("=" * 78)
print()

# F1-optimal threshold for each
print("=" * 78)
print("F1-OPTIMAL THRESHOLD")
print("=" * 78)

best_z_f1, best_z_t = 0, 0
best_m_f1, best_m_t = 0, 0

for t in np.arange(0.05, 0.95, 0.01):
    z_pred = (z_prob >= t).astype(int)
    m_pred = (m_prob >= t).astype(int)
    if z_pred.sum() > 0:
        z_f1 = f1_score(y_true, z_pred, zero_division=0)
        if z_f1 > best_z_f1:
            best_z_f1, best_z_t = z_f1, t
    if m_pred.sum() > 0:
        m_f1 = f1_score(y_true, m_pred, zero_division=0)
        if m_f1 > best_m_f1:
            best_m_f1, best_m_t = m_f1, t

print(f"ZERO:  best F1 = {best_z_f1:.4f} at threshold {best_z_t:.2f}")
print(f"MICE:  best F1 = {best_m_f1:.4f} at threshold {best_m_t:.2f}")
print()

# Recall-prioritized thresholds
print("=" * 78)
print("OPERATING POINTS (Recall-Prioritized)")
print("=" * 78)
print(f"{'Target Recall':>15} | {'Z-Thresh':>9} {'Z-Actual':>9} | "
      f"{'M-Thresh':>9} {'M-Actual':>9}")
print("-" * 78)

for target in [0.50, 0.60, 0.70]:
    z_best_t = None
    m_best_t = None
    for t in np.arange(0.95, 0.05, -0.01):
        z_pred = (z_prob >= t).astype(int)
        if z_pred.sum() > 0:
            r = recall_score(y_true, z_pred)
            if r >= target:
                z_best_t = t
                z_best_r = r
                break
    for t in np.arange(0.95, 0.05, -0.01):
        m_pred = (m_prob >= t).astype(int)
        if m_pred.sum() > 0:
            r = recall_score(y_true, m_pred)
            if r >= target:
                m_best_t = t
                m_best_r = r
                break

    z_str = f"{z_best_t:.2f}" if z_best_t else "N/A"
    m_str = f"{m_best_t:.2f}" if m_best_t else "N/A"
    z_a_str = f"{z_best_r:.4f}" if z_best_t else "N/A"
    m_a_str = f"{m_best_r:.4f}" if m_best_t else "N/A"
    print(f"{target:>15.2f} | {z_str:>9} {z_a_str:>9} | "
          f"{m_str:>9} {m_a_str:>9}")

print("=" * 78)
