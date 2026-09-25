#!/usr/bin/env python3
"""
Phase 8b: External validation of harmonized model on eICU.

Applies the frozen MIMIC scaler (scaler_params.csv) to eICU features
before prediction, since MIMIC features are z-scored and eICU features
are in raw clinical units.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, brier_score_loss, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))

from config import TABLE_DIR, MODEL_DIR


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y_true, y_prob)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'specificity': float(spec),
    }


def main():
    print("=" * 78)
    print("PHASE 8b: EXTERNAL VALIDATION OF HARMONIZED MODEL ON eICU")
    print("=" * 78)

    model = joblib.load(MODEL_DIR / 'phase6_harmonized_model.pkl')
    cfg = joblib.load(MODEL_DIR / 'phase6_harmonized_config.pkl')
    feature_cols = cfg['feature_cols']
    print(f"Model: {cfg['model']}")
    print(f"Features: {len(feature_cols)}")

    scaler = pd.read_csv(TABLE_DIR / 'scaler_params.csv')
    scaler_map = dict(zip(scaler['feature'], zip(scaler['mean'], scaler['scale'])))
    print(f"Loaded frozen scaler for {len(scaler_map)} features")

    eicu = pd.read_csv(TABLE_DIR / 'eicu_features_harmonized.csv')
    print(f"eICU harmonized: {eicu.shape}")

    missing_cols = [c for c in feature_cols if c not in eicu.columns]
    if missing_cols:
        print(f"WARNING: Missing {len(missing_cols)} cols; filling with 0")
        for c in missing_cols:
            eicu[c] = 0

    # Apply frozen scaler to continuous features
    print(f"\nApplying frozen MIMIC scaler...")
    eicu_scaled = eicu.copy()
    n_scaled = 0
    for c in feature_cols:
        if c in scaler_map and c in eicu_scaled.columns:
            m, s = scaler_map[c]
            if s > 1e-8:
                eicu_scaled[c] = (eicu_scaled[c] - m) / s
                n_scaled += 1
    print(f"Scaled {n_scaled} / {len(feature_cols)} features")

    # Sanity check on means
    print(f"\nSanity check (post-scale means should be closer to 0):")
    mimic = pd.read_csv(TABLE_DIR / 'X_train_harmonized.csv')
    for c in ['heart_rate_mean', 'creatinine_worst', 'gcs_total_min',
              'sodium_worst', 'glucose_worst']:
        if c in eicu_scaled.columns and c in mimic.columns:
            print(f"  {c:25s}  MIMIC={mimic[c].mean():7.2f}  "
                  f"eICU_scaled={eicu_scaled[c].mean():7.2f}")

    X = eicu_scaled[feature_cols].fillna(0).values
    y = eicu['outcome'].values

    print(f"\nX shape: {X.shape}")
    print(f"y positives: {y.sum():,} / {len(y):,} ({y.mean()*100:.2f}%)")

    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, y_prob)

    print(f"\neICU external validation metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    mimic_internal = pd.read_csv(
        TABLE_DIR / 'phase6_harmonized_test_metrics.csv'
    ).iloc[0]

    print(f"\nComparison (MIMIC harmonized vs eICU harmonized):")
    print(f"  {'metric':12s}  {'MIMIC':>10s}  {'eICU':>10s}  {'delta':>10s}")
    for k in ['auroc', 'auprc', 'recall', 'precision', 'f1', 'brier']:
        m = float(mimic_internal[k])
        e = metrics[k]
        print(f"  {k:12s}  {m:>10.4f}  {e:>10.4f}  {e-m:>+10.4f}")

    # Save
    pred_df = pd.DataFrame({
        'stay_id': eicu['stay_id'].values,
        'y_true': y,
        'y_prob': y_prob,
    })
    pred_df.to_csv(TABLE_DIR / 'phase8b_harmonized_predictions.csv', index=False)
    pd.DataFrame([metrics]).to_csv(
        TABLE_DIR / 'phase8b_harmonized_metrics.csv', index=False)

    report = []
    report.append("=" * 70)
    report.append("PHASE 8b: EXTERNAL VALIDATION (HARMONIZED MODEL)")
    report.append("=" * 70)
    report.append(f"Cohort: {len(y):,}")
    report.append(f"Prevalence: {y.mean()*100:.2f}%")
    report.append("")
    report.append("MIMIC test (harmonized model):")
    for k, v in mimic_internal.items():
        report.append(f"  {k}: {float(v):.4f}")
    report.append("")
    report.append("eICU external (harmonized model):")
    for k, v in metrics.items():
        report.append(f"  {k}: {v:.4f}")

    (TABLE_DIR / 'phase8b_harmonized_report.txt').write_text('\n'.join(report))

    print(f"\nSaved:")
    print(f"  phase8b_harmonized_predictions.csv")
    print(f"  phase8b_harmonized_metrics.csv")
    print(f"  phase8b_harmonized_report.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
