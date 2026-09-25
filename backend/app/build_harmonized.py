#!/usr/bin/env python3
"""
Build harmonized MIMIC train/test sets using only features
that are available in both MIMIC and eICU.
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import TABLE_DIR, MODEL_DIR


def main():
    print("=" * 78)
    print("BUILD HARMONIZED MIMIC TRAIN/TEST")
    print("=" * 78)

    cfg = joblib.load(MODEL_DIR / 'phase6_final_config.pkl')
    mask_variant = cfg['mask_variant']
    print(f"Frozen config: imputation={cfg['imputation']}, "
          f"model={cfg['model']}, masks={mask_variant}")

    with open(TABLE_DIR / 'harmonized_features.txt') as f:
        common = [line.strip() for line in f if line.strip()]
    print(f"Harmonized features: {len(common)}")

    X_train = pd.read_csv(TABLE_DIR / 'X_train.csv')
    X_test = pd.read_csv(TABLE_DIR / 'X_test.csv')
    print(f"Original MIMIC train: {X_train.shape}, test: {X_test.shape}")

    if mask_variant == 'with_masks':
        mask_cols = [c for c in X_train.columns if c.endswith('_missing')]
        relevant_masks = [c for c in mask_cols
                          if c.replace('_missing', '') in common]
        print(f"Relevant masks for common features: {len(relevant_masks)}")
    else:
        relevant_masks = []
        print("Mask variant is 'without_masks'; no masks added")

    feature_cols = common + relevant_masks

    missing_in_train = [c for c in feature_cols if c not in X_train.columns]
    if missing_in_train:
        print(f"WARNING: Missing from X_train: {missing_in_train[:10]}")
        feature_cols = [c for c in feature_cols if c in X_train.columns]

    print(f"Final harmonized feature count: {len(feature_cols)}")

    keep_cols = ['stay_id'] + feature_cols + ['outcome']
    X_train_h = X_train[keep_cols].copy()
    X_test_h = X_test[keep_cols].copy()

    X_train_h.to_csv(TABLE_DIR / 'X_train_harmonized.csv', index=False)
    X_test_h.to_csv(TABLE_DIR / 'X_test_harmonized.csv', index=False)

    with open(TABLE_DIR / 'harmonized_feature_cols.txt', 'w') as f:
        for c in feature_cols:
            f.write(f'{c}\n')

    print(f"\nSaved:")
    print(f"  X_train_harmonized.csv  {X_train_h.shape}")
    print(f"  X_test_harmonized.csv   {X_test_h.shape}")
    print(f"  harmonized_feature_cols.txt  ({len(feature_cols)} cols)")

    print(f"\n  Train positives: {X_train_h['outcome'].sum():,} / {len(X_train_h):,} "
          f"({X_train_h['outcome'].mean()*100:.2f}%)")
    print(f"  Test positives:  {X_test_h['outcome'].sum():,} / {len(X_test_h):,} "
          f"({X_test_h['outcome'].mean()*100:.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
