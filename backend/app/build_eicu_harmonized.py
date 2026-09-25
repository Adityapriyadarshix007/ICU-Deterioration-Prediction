#!/usr/bin/env python3
"""
Build eICU harmonized feature set with the same column order as MIMIC.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import TABLE_DIR


def main():
    print("=" * 78)
    print("BUILD eICU HARMONIZED FEATURES")
    print("=" * 78)

    with open(TABLE_DIR / 'harmonized_feature_cols.txt') as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    print(f"Expected columns (from MIMIC): {len(feature_cols)}")

    eicu = pd.read_csv(TABLE_DIR / 'eicu_features_withmasks.csv')
    cohort = pd.read_csv(TABLE_DIR / 'eicu_cohort_sofa.csv')
    print(f"eICU with masks: {eicu.shape}")

    eicu = eicu.merge(cohort[['stay_id', 'outcome']], on='stay_id', how='inner')
    print(f"After merge with labels: {eicu.shape}")

    missing = [c for c in feature_cols if c not in eicu.columns]
    if missing:
        print(f"WARNING: Missing {len(missing)} columns; filling with 0:")
        for c in missing[:10]:
            print(f"     {c}")
        for c in missing:
            eicu[c] = 0

    keep = ['stay_id'] + feature_cols + ['outcome']
    eicu_h = eicu[keep].copy()

    out = TABLE_DIR / 'eicu_features_harmonized.csv'
    eicu_h.to_csv(out, index=False)

    print(f"\nSaved: {out}")
    print(f"   Shape: {eicu_h.shape}")
    print(f"   Positives: {eicu_h['outcome'].sum():,} / {len(eicu_h):,} "
          f"({eicu_h['outcome'].mean()*100:.2f}%)")

    mimic_h = pd.read_csv(TABLE_DIR / 'X_train_harmonized.csv', nrows=5)
    mimic_cols = set(mimic_h.columns)
    eicu_cols = set(eicu_h.columns)
    print(f"\n  Columns only in MIMIC: {sorted(mimic_cols - eicu_cols)[:5]}")
    print(f"  Columns only in eICU:  {sorted(eicu_cols - mimic_cols)[:5]}")


if __name__ == "__main__":
    main()
