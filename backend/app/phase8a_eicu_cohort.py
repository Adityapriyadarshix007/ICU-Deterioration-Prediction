#!/usr/bin/env python3
"""
Phase 8a Step 3: Build eICU cohort (mirrors MIMIC Phase 2).

IMPORTANT: This eICU extract does NOT have `unitadmitoffset`.
Instead, `unitdischargeoffset` is minutes from ICU admission (= ICU LOS).
`hospitaladmitoffset` is signed minutes between hospital admit and ICU
admit (negative = hospital admit happened before ICU admit).

Requirements mirrored:
  - Age >= 18 (with '> 89' -> 91)
  - ICU LOS >= 24 h
  - First ICU stay per patient (by uniquepid)
  - Exclude transfers IN (unitadmitsource)
  - Exclude transfers OUT (unitdischargelocation)
  - Exclude deaths <= 18 h from ICU admit

Output:
  outputs/tables/eicu_cohort.csv
  outputs/tables/eicu_cohort_summary.txt

Usage:
    python3 phase8a_eicu_cohort.py
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EICU_DB_NAME, TABLE_DIR,
    MIN_ICU_STAY_HOURS, EICU_MIN_LOS_MINUTES,
    PREDICTION_END_HOUR,
    EICU_EXCLUDE_TRANSFERS,
    EICU_TRANSFER_IN_SOURCES,
    EICU_TRANSFER_OUT_LOCATIONS,
)


def parse_age(age_str):
    """eICU stores age as text. '> 89' -> 91, else int, else NaN."""
    if pd.isna(age_str):
        return np.nan
    s = str(age_str).strip()
    if s == '> 89' or s == '>89':
        return 91
    try:
        return float(s)
    except ValueError:
        return np.nan


def main():
    print("=" * 78)
    print("PHASE 8a — STEP 3: BUILD eICU COHORT")
    print("=" * 78)

    db_path = Path(EICU_DB_NAME)
    if not db_path.exists():
        print(f"❌ {db_path} not found. Run phase8a_setup_eicu_db.py first.")
        return 1

    conn = duckdb.connect(str(db_path), read_only=True)

    print("\nStep 1: Loading patient table...")
    df = conn.execute("""
        SELECT
            patientunitstayid,
            uniquepid,
            patienthealthsystemstayid,
            gender,
            age,
            admissionweight,
            admissionheight,
            unitadmitsource,
            unitdischargeoffset,
            unitdischargestatus,
            unitdischargelocation,
            unittype,
            hospitaladmitoffset,
            hospitaldischargeoffset,
            hospitaldischargestatus,
            hospitalid
        FROM patient
    """).fetchdf()
    print(f"  Loaded: {len(df):,} rows")
    conn.close()

    exclusions = {}

    # ---------- Parse age ----------
    df['age_numeric'] = df['age'].apply(parse_age)

    # ---------- Compute ICU LOS ----------
    # In this eICU extract, unitdischargeoffset = minutes from ICU admission.
    # hospitaladmitoffset is signed minutes between hospital and ICU admission.
    df['icu_los_minutes'] = df['unitdischargeoffset']
    df['icu_los_hours'] = df['icu_los_minutes'] / 60.0

    # ---------- Age filter ----------
    before = len(df)
    df = df[df['age_numeric'] >= 18]
    exclusions['age_lt_18'] = before - len(df)
    print(f"\n  Excluded {exclusions['age_lt_18']:,} age < 18")

    # ---------- LOS filter ----------
    before = len(df)
    df = df[df['icu_los_minutes'] >= EICU_MIN_LOS_MINUTES]
    exclusions['stay_lt_min_hours'] = before - len(df)
    print(f"  Excluded {exclusions['stay_lt_min_hours']:,} "
          f"ICU LOS < {MIN_ICU_STAY_HOURS}h")

    # ---------- Valid offsets ----------
    before = len(df)
    df = df[
        df['unitdischargeoffset'].notna() &
        (df['unitdischargeoffset'] > 0)
    ]
    exclusions['invalid_offsets'] = before - len(df)
    print(f"  Excluded {exclusions['invalid_offsets']:,} invalid offsets")

    # ---------- Requirement 1 & 3: transfers ----------
    if EICU_EXCLUDE_TRANSFERS:
        before = len(df)
        df = df[~df['unitadmitsource'].isin(EICU_TRANSFER_IN_SOURCES)]
        exclusions['req1_transfer_in'] = before - len(df)
        print(f"  [Req 1] Excluded {exclusions['req1_transfer_in']:,} "
              f"transfers IN ({EICU_TRANSFER_IN_SOURCES})")

        before = len(df)
        df = df[~df['unitdischargelocation'].isin(EICU_TRANSFER_OUT_LOCATIONS)]
        exclusions['req3_transfer_out'] = before - len(df)
        print(f"  [Req 3] Excluded {exclusions['req3_transfer_out']:,} "
              f"transfers OUT ({EICU_TRANSFER_OUT_LOCATIONS})")

        n_ward = int((df['unitadmitsource'] == 'Floor').sum())
        print(f"  [Req 2] KEPT {n_ward:,} ward-to-ICU patients")

    # ---------- First stay per patient ----------
    before = len(df)
    df = df.sort_values(['uniquepid', 'unitdischargeoffset'])
    df = df.drop_duplicates('uniquepid', keep='first').reset_index(drop=True)
    exclusions['not_first_stay'] = before - len(df)
    print(f"  Excluded {exclusions['not_first_stay']:,} non-first stays")

    # ---------- Early death exclusion ----------
    before = len(df)
    death_mask = (
        (df['unitdischargestatus'] == 'Expired') &
        (df['icu_los_minutes'] <= PREDICTION_END_HOUR * 60)
    )
    df = df[~death_mask].copy()
    exclusions['early_death'] = before - len(df)
    print(f"  Excluded {exclusions['early_death']:,} "
          f"deaths ≤ {PREDICTION_END_HOUR}h")

    # ---------- Final cohort ----------
    print(f"\n  KEPT {len(df):,} direct ICU admissions")
    print(f"\n  Composition (top 8 admit sources):")
    kept_locs = df['unitadmitsource'].value_counts().head(8).to_dict()
    for loc, cnt in kept_locs.items():
        print(f"    {loc}: {cnt:,}")

    # ---------- Save ----------
    out_csv = TABLE_DIR / "eicu_cohort.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved: {out_csv}")
    print(f"   Rows: {len(df):,}")
    print(f"   Unique patients: {df['uniquepid'].nunique():,}")

    out_txt = TABLE_DIR / "eicu_cohort_summary.txt"
    with open(out_txt, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("eICU COHORT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total patients:     {len(df):,}\n")
        f.write(f"Unique uniquepid:   {df['uniquepid'].nunique():,}\n")
        f.write(f"Unique hospitals:   {df['hospitalid'].nunique()}\n\n")
        f.write("Exclusion counts:\n")
        for k, v in exclusions.items():
            f.write(f"  {k}: {v:,}\n")
        f.write("\nICU LOS distribution (hours):\n")
        f.write(f"  min:  {df['icu_los_hours'].min():.1f}\n")
        f.write(f"  p25:  {df['icu_los_hours'].quantile(0.25):.1f}\n")
        f.write(f"  med:  {df['icu_los_hours'].median():.1f}\n")
        f.write(f"  p75:  {df['icu_los_hours'].quantile(0.75):.1f}\n")
        f.write(f"  max:  {df['icu_los_hours'].max():.1f}\n")
        f.write("\nNOTE: unitdischargeoffset = ICU LOS in minutes (this extract).\n")
    print(f"✅ Saved: {out_txt}")
    print("\nNext: python3 phase8a_eicu_extract.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
