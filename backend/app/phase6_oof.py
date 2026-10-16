#!/usr/bin/env python3
"""
Standalone OOF generator for the Phase 6 best tree model.

Reads:
    outputs/tables/phase6_recommendation.json
    outputs/tables/X_train.csv

Writes:
    outputs/tables/phase6_oof_train_predictions.csv

Does NOT modify phase6.py or re-run the Phase 6 grid.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from config import TABLE_DIR, RANDOM_STATE, N_FOLDS
from phase6 import (
    load_raw_data, deduplicate_masks, fit_imputer, build_masks,
    build_model, fit_model_with_early_stopping, EARLY_STOP_VAL_FRAC,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()

    print("=" * 70)
    print(" PHASE 6 OOF GENERATOR (tree model only)")
    print("=" * 70)

    # ---- 1. Load best config ----
    rec_path = TABLE_DIR / 'phase6_recommendation.json'
    if not rec_path.exists():
        print(f"ERROR: {rec_path} not found. Run Phase 6 first.")
        return 1

    with open(rec_path) as f:
        rec = json.load(f)

    best_imp = rec['best_imputation']
    best_mdl = rec['best_model']
    best_variant = rec['best_mask_variant']
    use_masks = (best_variant == 'with_masks')

    print(f"\nBest config: {best_imp} + {best_mdl} + {best_variant}")
    print(f"CV AUPRC (from recommendation): {rec.get('best_auprc')}")
    print()

    # ---- 2. Load raw training data ----
    X_train_raw, _, y_train, _ = load_raw_data(logger)

    # ---- 3. Load stay_ids (aligned with X_train_raw) ----
    train_df = pd.read_csv(TABLE_DIR / 'X_train.csv', usecols=['stay_id'])
    stay_ids = train_df['stay_id'].values

    if len(stay_ids) != len(X_train_raw):
        logger.error(f"stay_id length mismatch: {len(stay_ids)} vs "
                     f"{len(X_train_raw)}")
        return 1

    # ---- 4. Deduplicate masks ----
    kept_mask_cols = deduplicate_masks(X_train_raw, logger)

    # ---- 5. Generate OOF predictions using same folds as Phase 6 ----
    logger.info("\nGenerating OOF predictions (5 folds)...")

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probs = np.full(len(X_train_raw), np.nan, dtype=float)

    for fold_idx, (tr_idx, val_idx) in enumerate(
            cv.split(X_train_raw, y_train)):
        logger.info(f"  Fold {fold_idx + 1}/{N_FOLDS}")

        X_tr_raw = X_train_raw.iloc[tr_idx].reset_index(drop=True)
        X_val_raw = X_train_raw.iloc[val_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]

        transform = fit_imputer(best_imp, X_tr_raw)
        X_tr = transform(X_tr_raw)
        X_val = transform(X_val_raw)

        if use_masks and kept_mask_cols:
            mask_tr = build_masks(X_tr_raw, kept_mask_cols)
            mask_val = build_masks(X_val_raw, kept_mask_cols)
            X_tr = pd.concat([X_tr, mask_tr], axis=1)
            X_val = pd.concat([X_val, mask_val], axis=1)

        X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
            X_tr, y_tr,
            test_size=EARLY_STOP_VAL_FRAC,
            stratify=y_tr,
            random_state=RANDOM_STATE,
        )

        model = build_model(best_mdl)
        model = fit_model_with_early_stopping(
            best_mdl, model, X_tr2, y_tr2, X_val2, y_val2,
        )

        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_auprc = average_precision_score(
            y_train[val_idx], oof_probs[val_idx]
        )
        logger.info(f"    Fold {fold_idx + 1} AUPRC: {fold_auprc:.4f}")

    # ---- 6. Verify ----
    if np.isnan(oof_probs).any():
        logger.error(f"{int(np.isnan(oof_probs).sum())} OOF predictions NaN")
        return 1

    # ---- 7. Save ----
    oof_df = pd.DataFrame({
        'stay_id': stay_ids,
        'y_true': y_train,
        'y_prob_oof': oof_probs,
    })
    out_path = TABLE_DIR / 'phase6_oof_train_predictions.csv'
    oof_df.to_csv(out_path, index=False)

    # ---- 8. Report ----
    oof_auprc = average_precision_score(y_train, oof_probs)
    oof_auroc = roc_auc_score(y_train, oof_probs)

    print("\n" + "=" * 70)
    print(" OOF COMPLETE (tree)")
    print("=" * 70)
    print(f"  n:          {len(oof_probs):,}")
    print(f"  OOF AUPRC:  {oof_auprc:.4f}  "
          f"(CV reference: {rec.get('best_auprc'):.4f})")
    print(f"  OOF AUROC:  {oof_auroc:.4f}  "
          f"(CV reference: {rec.get('best_auroc'):.4f})")
    print(f"  Saved:      {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
