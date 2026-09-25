#!/usr/bin/env python3
"""
Add _missing columns to eICU features, mirroring MIMIC's convention.
For each value column, add <col>_missing = 1 if NaN, 0 otherwise.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import TABLE_DIR


def main():
    print("=" * 78)
    print("ADD MASKS TO eICU FEATURES")
    print("=" * 78)

    mimic_h = pd.read_csv(TABLE_DIR / 'X_train_harmonized.csv', nrows=5)
    mimic_mask_cols = [c for c in mimic_h.columns if c.endswith('_missing')]
    print(f"MIMIC harmonized mask columns: {len(mimic_mask_cols)}")

    eicu = pd.read_csv(TABLE_DIR / 'eicu_features_raw.csv')
    print(f"eICU features shape: {eicu.shape}")

    added = 0
    for mc in mimic_mask_cols:
        base = mc.replace('_missing', '')
        if base in eicu.columns and mc not in eicu.columns:
            eicu[mc] = eicu[base].isna().astype(int)
            added += 1

    print(f"Masks added: {added}")

    print("\nSanity check (should sum to 100):")
    for col in ['creatinine_worst', 'lactate_last', 'gcs_total_min',
                'sodium_worst', 'glucose_worst']:
        mc = f'{col}_missing'
        if mc in eicu.columns and col in eicu.columns:
            m = eicu[mc].mean() * 100
            n = eicu[col].notna().mean() * 100
            print(f"  {col:20s}  missing={m:5.1f}%  notna={n:5.1f}%  sum={m+n:5.1f}%")

    # Sanity check on creatinine distribution
    if 'creatinine_worst' in eicu.columns:
        c = eicu['creatinine_worst'].dropna()
        print(f"\ncreatinine_worst (should be clean now):")
        print(f"  n={len(c):,}, mean={c.mean():.2f}, p50={c.median():.2f}, "
              f"p99={c.quantile(0.99):.2f}, max={c.max():.2f}")

    out = TABLE_DIR / 'eicu_features_withmasks.csv'
    eicu.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"   Shape: {eicu.shape}")


if __name__ == "__main__":
    main()
