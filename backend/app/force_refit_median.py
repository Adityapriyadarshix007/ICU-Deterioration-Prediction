#!/usr/bin/env python3
"""
Force-refit an ALTERNATIVE model: median + LightGBM + with_masks.

Does NOT overwrite the primary Phase 6 artifacts. Writes to
outputs/models/phase6_alt/ instead.

Purpose: test whether switching from zero to median imputation
resolves the FiO2 missingness SHAP artifact observed in the primary
model (zero + LightGBM + with_masks).

Outputs:
    outputs/models/phase6_alt/phase6_alt_model.pkl
    outputs/models/phase6_alt/phase6_alt_config.pkl
    outputs/tables/phase6_alt_test_metrics.csv
    outputs/tables/phase6_alt_report.txt

Usage:
    python3 force_refit_median.py
"""

import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, MODEL_DIR, LOGS_DIR,
    RANDOM_STATE, N_FOLDS,
)

# Reuse Phase 6 building blocks
from phase6 import (
    load_raw_data, deduplicate_masks, build_model,
    fit_model_with_early_stopping, build_masks, fit_imputer,
    evaluate, setup_logging, EARLY_STOP_VAL_FRAC,
)


# ============================================================
# CONFIG: which combo to force-fit
# ============================================================
FORCE_IMPUTATION = 'median'
FORCE_MODEL = 'lightgbm'
FORCE_MASK_VARIANT = 'with_masks'

ALT_MODEL_DIR = MODEL_DIR / 'phase6_alt'
ALT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("FORCED REFIT: ALTERNATIVE MODEL")
    logger.info("=" * 70)
    logger.info(f"  Imputation:  {FORCE_IMPUTATION}")
    logger.info(f"  Model:       {FORCE_MODEL}")
    logger.info(f"  Mask variant:{FORCE_MASK_VARIANT}")
    logger.info(f"  Saving to:   {ALT_MODEL_DIR}")
    logger.info("=" * 70)

    t0 = datetime.now()

    # Load raw data
    X_train_raw, X_test_raw, y_train, y_test = load_raw_data(logger)

    # Deduplicate masks (same as primary)
    logger.info("\nMask deduplication...")
    kept_mask_cols = deduplicate_masks(X_train_raw, logger)

    # Fit imputer on full train
    logger.info(f"\nFitting imputer: {FORCE_IMPUTATION}")
    transform = fit_imputer(FORCE_IMPUTATION, X_train_raw)
    X_tr_full = transform(X_train_raw)
    X_te_full = transform(X_test_raw)

    # Add masks if requested
    if FORCE_MASK_VARIANT == 'with_masks':
        logger.info(f"Adding {len(kept_mask_cols)} masks")
        X_tr_full = pd.concat(
            [X_tr_full, build_masks(X_train_raw, kept_mask_cols)], axis=1)
        X_te_full = pd.concat(
            [X_te_full, build_masks(X_test_raw, kept_mask_cols)], axis=1)

    logger.info(f"Final train shape: {X_tr_full.shape}")
    logger.info(f"Final test shape:  {X_te_full.shape}")

    # Early-stopping split from train
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_tr_full, y_train, test_size=EARLY_STOP_VAL_FRAC,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    # Build + fit
    logger.info(f"\nBuilding model: {FORCE_MODEL}")
    model = build_model(FORCE_MODEL)
    model = fit_model_with_early_stopping(
        FORCE_MODEL, model, X_tr2, y_tr2, X_val2, y_val2,
    )

    # Evaluate on test
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate(model, X_te_full, y_test)
    logger.info(f"\nALT model test metrics:")
    for k, v in test_metrics.items():
        logger.info(f"    {k:12s}: {v:.4f}")

    # Load primary model's metrics for comparison
    primary_metrics_path = TABLE_DIR / 'phase6_test_metrics.csv'
    primary_metrics = None
    if primary_metrics_path.exists():
        primary_metrics = pd.read_csv(primary_metrics_path).iloc[0].to_dict()
        logger.info(f"\nPRIMARY model test metrics (for comparison):")
        for k in ['auroc', 'auprc', 'recall', 'precision', 'f1', 'brier']:
            if k in primary_metrics:
                logger.info(f"    {k:12s}: {float(primary_metrics[k]):.4f}")

    # Save model + config
    model_path = ALT_MODEL_DIR / 'phase6_alt_model.pkl'
    config_path = ALT_MODEL_DIR / 'phase6_alt_config.pkl'
    joblib.dump(model, model_path)
    joblib.dump({
        'imputation': FORCE_IMPUTATION,
        'model': FORCE_MODEL,
        'mask_variant': FORCE_MASK_VARIANT,
        'kept_mask_cols': kept_mask_cols,
        'feature_cols': list(X_train_raw.columns),
    }, config_path)
    logger.info(f"\nSaved: {model_path}")
    logger.info(f"Saved: {config_path}")

    # Save test metrics
    metrics_path = TABLE_DIR / 'phase6_alt_test_metrics.csv'
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Saved: {metrics_path}")

    # Comparison report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PHASE 6 ALT MODEL — FORCED REFIT REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now()}")
    report_lines.append("")
    report_lines.append("Combination:")
    report_lines.append(f"  Imputation:   {FORCE_IMPUTATION}")
    report_lines.append(f"  Model:        {FORCE_MODEL}")
    report_lines.append(f"  Mask variant: {FORCE_MASK_VARIANT}")
    report_lines.append(f"  n_features:   {X_tr_full.shape[1]}")
    report_lines.append("")
    report_lines.append("Purpose:")
    report_lines.append("  Test whether median imputation (instead of zero)")
    report_lines.append("  removes the FiO2 missingness SHAP artifact observed")
    report_lines.append("  in the primary model.")
    report_lines.append("")
    report_lines.append("ALT model test metrics:")
    for k, v in test_metrics.items():
        report_lines.append(f"  {k:12s}: {v:.4f}")
    report_lines.append("")
    if primary_metrics:
        report_lines.append("PRIMARY model test metrics:")
        for k in ['auroc', 'auprc', 'recall', 'precision', 'f1', 'brier']:
            if k in primary_metrics:
                report_lines.append(
                    f"  {k:12s}: {float(primary_metrics[k]):.4f}")
        report_lines.append("")
        report_lines.append("Delta (ALT - PRIMARY):")
        for k in ['auroc', 'auprc', 'recall', 'precision', 'f1', 'brier']:
            if k in primary_metrics and k in test_metrics:
                d = test_metrics[k] - float(primary_metrics[k])
                report_lines.append(f"  {k:12s}: {d:+.4f}")

    report_path = TABLE_DIR / 'phase6_alt_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Saved: {report_path}")

    elapsed = datetime.now() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed}")
    logger.info("=" * 70)
    print()
    print("=" * 70)
    print(" ALT MODEL READY")
    print("=" * 70)
    print(f"  Model:  {model_path}")
    print(f"  Config: {config_path}")
    print(f"  Metrics: {metrics_path}")
    print()
    print("Next:")
    print("  1. Run Phase 7 pointing at phase6_alt/ (see phase7_alt.py)")
    print("  2. Compare SHAP outputs")
    print("  3. Decide which model to lead with in the paper")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
