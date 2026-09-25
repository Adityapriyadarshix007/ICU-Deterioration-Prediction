#!/usr/bin/env python3
"""
Phase 8a Step 4: Extract eICU features (mirrors MIMIC Phase 3).

Queries eICU tables for the 0-6h window, aggregates to per-stay features.
Excludes non-blood lab specimens (urinary, CSF, body fluid) to prevent
contamination of blood-based features.

Output:
  outputs/tables/eicu_features_raw.csv

Usage:
    python3 phase8a_eicu_extract.py
"""

import sys
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EICU_DB_NAME, TABLE_DIR,
    EICU_FEATURE_WINDOW_MIN,
    EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS,
    EICU_VASO_PATTERNS,
    EICU_GCS_SOURCE, EICU_GCS_LABELS,
)


def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.search(r'-?\d+\.?\d*', s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return np.nan
    return np.nan


def parse_lab_name(labname, patterns, exclusions):
    """Return canonical feature name if labname matches patterns AND
    does not contain any exclusion substring."""
    name = str(labname).lower().strip()

    # Exclusion check first
    for excl in exclusions:
        if excl in name:
            return None

    for canonical, pats in patterns.items():
        for p in pats:
            if p in name:
                return canonical
    return None


def parse_vaso_name(drugname, patterns):
    name = str(drugname).lower().strip()
    for canonical, pats in patterns.items():
        for p in pats:
            if p in name:
                return canonical
    return None


def build_lab_where(patterns, exclusions):
    """SQL fragment: (matches any lab pattern) AND (no exclusion substring)."""
    all_p = [p.lower() for ps in patterns.values() for p in ps]
    include = " OR ".join(f"LOWER(labname) LIKE '%{p}%'" for p in all_p)
    exclude = " AND ".join(
        f"LOWER(labname) NOT LIKE '%{e}%'" for e in exclusions
    )
    return f"({include}) AND ({exclude})"


def main():
    print("=" * 78)
    print("PHASE 8a — STEP 4: EXTRACT eICU FEATURES (with lab exclusions)")
    print("=" * 78)

    cohort_path = TABLE_DIR / "eicu_cohort.csv"
    if not cohort_path.exists():
        print(f"ERROR: {cohort_path} not found.")
        return 1

    cohort = pd.read_csv(cohort_path)
    stay_ids = cohort['patientunitstayid'].tolist()
    print(f"\nLoaded cohort: {len(cohort):,} stays")

    conn = duckdb.connect(str(EICU_DB_NAME), read_only=True)
    ids = ",".join(map(str, stay_ids))

    # ---------- Vitals ----------
    print("\nQuerying vitals...")
    vitals = conn.execute(f"""
        SELECT patientunitstayid, observationoffset,
               heartrate, respiration, sao2, systemicmean,
               systemicsystolic, systemicdiastolic, temperature
        FROM vitalPeriodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset >= 0
          AND observationoffset < {EICU_FEATURE_WINDOW_MIN}
    """).fetchdf()
    print(f"  {len(vitals):,} rows")

    # ---------- Labs (with exclusions) ----------
    print("Querying labs (excluding urinary/CSF/fluid)...")
    lab_where = build_lab_where(EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS)
    labs = conn.execute(f"""
        SELECT patientunitstayid, labresultoffset, labname, labresult
        FROM lab
        WHERE patientunitstayid IN ({ids})
          AND labresultoffset >= 0
          AND labresultoffset < {EICU_FEATURE_WINDOW_MIN}
          AND labresult IS NOT NULL
          AND ({lab_where})
    """).fetchdf()
    print(f"  {len(labs):,} rows")

    # ---------- Vasopressors ----------
    print("Querying vasopressors...")
    all_vaso = [p.lower() for ps in EICU_VASO_PATTERNS.values() for p in ps]
    vaso_where = " OR ".join(f"LOWER(drugname) LIKE '%{p}%'" for p in all_vaso)
    vaso = conn.execute(f"""
        SELECT patientunitstayid, infusionoffset, drugname, drugrate, infusionrate
        FROM infusionDrug
        WHERE patientunitstayid IN ({ids})
          AND infusionoffset >= 0
          AND infusionoffset < {EICU_FEATURE_WINDOW_MIN}
          AND ({vaso_where})
    """).fetchdf()
    print(f"  {len(vaso):,} rows")

    # ---------- GCS ----------
    print("Querying GCS...")
    all_gcs = [v.lower() for vs in EICU_GCS_LABELS.values() for v in vs]
    gcs_where = " OR ".join(f"LOWER(nursingchartcelltypevalname) LIKE '%{v}%'" for v in all_gcs)
    gcs = conn.execute(f"""
        SELECT patientunitstayid, nursingchartoffset,
               nursingchartcelltypevalname, nursingchartvalue
        FROM {EICU_GCS_SOURCE}
        WHERE patientunitstayid IN ({ids})
          AND nursingchartoffset >= 0
          AND nursingchartoffset < {EICU_FEATURE_WINDOW_MIN}
          AND ({gcs_where})
    """).fetchdf()
    print(f"  {len(gcs):,} rows")

    # ---------- FiO2 ----------
    print("Querying FiO2...")
    fio2 = conn.execute(f"""
        SELECT patientunitstayid, respchartoffset, respchartvalue
        FROM respiratoryCharting
        WHERE patientunitstayid IN ({ids})
          AND respchartoffset >= 0
          AND respchartoffset < {EICU_FEATURE_WINDOW_MIN}
          AND LOWER(respchartvaluelabel) LIKE '%fio2%'
    """).fetchdf()
    print(f"  {len(fio2):,} rows")

    conn.close()

    # ---------- Aggregate ----------
    print("\nAggregating...")

    # Vitals
    vitals_agg = vitals.groupby('patientunitstayid').agg(
        heart_rate_mean=('heartrate', 'mean'),
        heart_rate_min=('heartrate', 'min'),
        heart_rate_max=('heartrate', 'max'),
        respiratory_rate_mean=('respiration', 'mean'),
        spo2_mean=('sao2', 'mean'),
        spo2_min=('sao2', 'min'),
        map_mean=('systemicmean', 'mean'),
        map_min=('systemicmean', 'min'),
        sbp_mean=('systemicsystolic', 'mean'),
        dbp_mean=('systemicdiastolic', 'mean'),
        temperature_mean=('temperature', 'mean'),
    ).reset_index().rename(columns={'patientunitstayid': 'stay_id'})

    # Labs (with exclusion-aware parsing)
    labs = labs.copy()
    labs['canonical'] = labs['labname'].apply(
        lambda x: parse_lab_name(x, EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS)
    )
    labs = labs.dropna(subset=['canonical'])
    lab_rows = []
    MIN_LABS = {'platelets', 'albumin', 'hemoglobin', 'hematocrit'}
    for (sid, lab), g in labs.groupby(['patientunitstayid', 'canonical']):
        v = g['labresult'].dropna()
        if len(v) == 0:
            continue
        agg = 'min' if lab in MIN_LABS else 'max'
        lab_rows.append({
            'stay_id': sid,
            f'{lab}_worst': float(getattr(v, agg)()),
            f'{lab}_last': float(v.iloc[-1]),
        })
    labs_agg = pd.DataFrame(lab_rows).groupby('stay_id').first().reset_index() \
        if lab_rows else pd.DataFrame({'stay_id': stay_ids})

    # Vasopressors
    vaso = vaso.copy()
    vaso['canonical'] = vaso['drugname'].apply(
        lambda x: parse_vaso_name(x, EICU_VASO_PATTERNS))
    vaso = vaso.dropna(subset=['canonical'])
    vaso['rate_num'] = vaso['drugrate'].apply(parse_numeric)
    vaso_rows = []
    for sid in stay_ids:
        g = vaso[vaso['patientunitstayid'] == sid]
        row = {'stay_id': sid}
        for name in EICU_VASO_PATTERNS.keys():
            sub = g[g['canonical'] == name]
            row[f'{name}_any'] = int(len(sub) > 0)
            row[f'{name}_max_rate'] = (
                float(sub['rate_num'].max())
                if len(sub) > 0 and sub['rate_num'].notna().any() else 0.0
            )
        vaso_rows.append(row)
    vaso_agg = pd.DataFrame(vaso_rows)

    # GCS (only 'GCS Total' exists in eICU)
    gcs = gcs.copy()
    gcs['value_num'] = gcs['nursingchartvalue'].apply(parse_numeric)
    gcs = gcs.dropna(subset=['value_num'])
    gcs_agg = gcs.groupby('patientunitstayid').agg(
        gcs_total_min=('value_num', 'min'),
    ).reset_index().rename(columns={'patientunitstayid': 'stay_id'})

    # FiO2
    fio2 = fio2.copy()
    fio2['value_num'] = fio2['respchartvalue'].apply(parse_numeric)
    fio2 = fio2.dropna(subset=['value_num'])
    fio2['fio2_norm'] = fio2['value_num'].apply(lambda v: v / 100.0 if v > 1.0 else v)
    fio2_agg = fio2.groupby('patientunitstayid').agg(
        fio2_max=('fio2_norm', 'max'),
        fio2_min=('fio2_norm', 'min'),
    ).reset_index().rename(columns={'patientunitstayid': 'stay_id'})

    # ---------- Merge ----------
    print("Merging...")
    features = cohort[['patientunitstayid', 'uniquepid', 'age_numeric',
                        'gender', 'icu_los_hours']].copy()
    features = features.rename(columns={'patientunitstayid': 'stay_id'})
    features['age'] = features['age_numeric']
    features['gender_male'] = (features['gender'] == 'Male').astype(int)
    features = features.drop(columns=['gender', 'age_numeric'])

    for agg in [vitals_agg, labs_agg, vaso_agg, gcs_agg, fio2_agg]:
        features = features.merge(agg, on='stay_id', how='left')

    # ---------- Feature unit corrections ----------
    # eICU aggregates from ~200 hospitals with different reporting units.
    # Small numbers of extreme outliers dominate downstream z-scoring,
    # so apply plausible-range caps.

    # Vasopressors: cap at clinical max (mixed units in drugrate)
    for col, cap in [('norepinephrine_max_rate', 5.0),
                     ('epinephrine_max_rate', 5.0),
                     ('dopamine_max_rate', 100.0),
                     ('dobutamine_max_rate', 50.0)]:
        if col in features.columns:
            mask = features[col] > cap
            n = int(mask.sum())
            if n > 0:
                print(f"  Capping {col}: {n} values > {cap} set to 0")
                features.loc[mask, col] = 0.0

    # Hemoglobin: some sites report in g/L (÷10 gives g/dL)
    for col in ['hemoglobin_last', 'hemoglobin_worst']:
        if col in features.columns:
            mask = features[col] > 25.0
            n = int(mask.sum())
            if n > 0:
                print(f"  Fixing {col}: {n} values > 25 (÷10, g/L → g/dL)")
                features.loc[mask, col] = features.loc[mask, col] / 10.0

    # FiO2: must be [0.21, 1.0]. Values > 1 are percentages.
    for col in ['fio2_max', 'fio2_min']:
        if col in features.columns:
            mask_pct = (features[col] > 1.0) & (features[col] <= 100.0)
            n_pct = int(mask_pct.sum())
            if n_pct > 0:
                print(f"  Fixing {col}: {n_pct} values in (1, 100] (÷100, % → fraction)")
                features.loc[mask_pct, col] = features.loc[mask_pct, col] / 100.0
            # Values < 0.21 or > 100 are implausible
            mask_bad = (features[col] < 0.21) | (features[col] > 100.0)
            n_bad = int(mask_bad.sum())
            if n_bad > 0:
                print(f"  Nulling {col}: {n_bad} values outside [0.21, 100]")
                features.loc[mask_bad, col] = np.nan

    print(f"\nRaw features shape: {features.shape}")
    out = TABLE_DIR / "eicu_features_raw.csv"
    features.to_csv(out, index=False)
    print(f"Saved: {out}")

    # Sanity: check creatinine distribution
    if 'creatinine_worst' in features.columns:
        c = features['creatinine_worst'].dropna()
        print(f"\ncreatinine_worst distribution (after exclusion):")
        print(f"  n={len(c):,}, mean={c.mean():.2f}, "
              f"p50={c.median():.2f}, p99={c.quantile(0.99):.2f}, max={c.max():.2f}")

    print("\nNext: python3 phase8a_eicu_sofa.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
