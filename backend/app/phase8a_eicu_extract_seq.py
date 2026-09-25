#!/usr/bin/env python3
"""
Phase 8a Step 6: Extract eICU sequences for DL model.

Mirrors MIMIC Phase 3b:
  - 12 bins x 30 minutes over [0, 6h)
  - 20 features + 20 masks = 40 channels
  - Apply FROZEN MIMIC mean/std + winsorization thresholds

Uses the same lab exclusions as phase8a_eicu_extract.py to prevent
contamination by urinary/CSF/body-fluid analytes.

Output:
  outputs/sequences/sequences_eicu.npz
"""

import sys
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EICU_DB_NAME, TABLE_DIR, SEQUENCE_DIR,
    EICU_FEATURE_WINDOW_MIN,
    EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS,
    EICU_VASO_PATTERNS,
    EICU_GCS_SOURCE, EICU_GCS_LABELS,
)

N_BINS = 12
BIN_MINUTES = 30
N_FEATURES = 20

FEATURE_NAMES = [
    'heart_rate', 'respiratory_rate', 'spo2', 'map', 'gcs_total',
    'fio2', 'pao2',
    'creatinine', 'lactate', 'bilirubin', 'platelets', 'wbc',
    'hemoglobin', 'sodium', 'potassium', 'bun', 'glucose',
    'norepinephrine', 'epinephrine', 'dopamine',
]


def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    m = re.search(r'-?\d+\.?\d*', str(val).strip())
    if m:
        try:
            return float(m.group())
        except ValueError:
            return np.nan
    return np.nan


def lab_is_excluded(labname, exclusions):
    name = str(labname).lower()
    for e in exclusions:
        if e in name:
            return True
    return False


def main():
    print("=" * 78)
    print("PHASE 8a — STEP 6: EXTRACT eICU SEQUENCES (with lab exclusions)")
    print("=" * 78)

    cohort_path = TABLE_DIR / "eicu_cohort_sofa.csv"
    if not cohort_path.exists():
        print(f"ERROR: {cohort_path} not found.")
        return 1

    cohort = pd.read_csv(cohort_path)
    stay_ids = cohort['stay_id'].tolist()
    print(f"\nCohort: {len(cohort):,} stays")

    stats_path = TABLE_DIR / "phase3b_feature_stats.csv"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found.")
        return 1
    stats = pd.read_csv(stats_path)
    print(f"Loaded frozen MIMIC feature stats: {len(stats)} features")

    stats_feats = set(stats['feature'].tolist())
    missing_stats = [f for f in FEATURE_NAMES if f not in stats_feats]
    if missing_stats:
        print(f"WARNING: Missing from MIMIC stats: {missing_stats}")

    conn = duckdb.connect(str(EICU_DB_NAME), read_only=True)
    ids = ",".join(map(str, stay_ids))

    # ---------- Vitals ----------
    print("\nQuerying vitals...")
    vitals = conn.execute(f"""
        SELECT patientunitstayid AS stay_id, observationoffset,
               heartrate AS heart_rate,
               respiration AS respiratory_rate,
               sao2 AS spo2,
               systemicmean AS map
        FROM vitalPeriodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset >= 0
          AND observationoffset < {EICU_FEATURE_WINDOW_MIN}
    """).fetchdf()
    print(f"  {len(vitals):,} rows")

    # ---------- Labs (with SQL-level exclusion) ----------
    print("Querying labs (excluding urinary/CSF/fluid)...")
    all_p = [p.lower() for ps in EICU_LAB_PATTERNS.values() for p in ps]
    lab_include = " OR ".join(f"LOWER(labname) LIKE '%{p}%'" for p in all_p)
    lab_exclude = " AND ".join(
        f"LOWER(labname) NOT LIKE '%{e}%'" for e in EICU_LAB_EXCLUSIONS)
    labs = conn.execute(f"""
        SELECT patientunitstayid AS stay_id, labresultoffset AS offset,
               labname, labresult AS value
        FROM lab
        WHERE patientunitstayid IN ({ids})
          AND labresultoffset >= 0
          AND labresultoffset < {EICU_FEATURE_WINDOW_MIN}
          AND labresult IS NOT NULL
          AND ({lab_include})
          AND ({lab_exclude})
    """).fetchdf()
    print(f"  {len(labs):,} rows")

    # ---------- Vasopressors ----------
    print("Querying vasopressors...")
    all_v = [p.lower() for ps in EICU_VASO_PATTERNS.values() for p in ps]
    vaso_where = " OR ".join(f"LOWER(drugname) LIKE '%{p}%'" for p in all_v)
    vaso = conn.execute(f"""
        SELECT patientunitstayid AS stay_id, infusionoffset AS offset,
               drugname, drugrate AS value
        FROM infusionDrug
        WHERE patientunitstayid IN ({ids})
          AND infusionoffset >= 0
          AND infusionoffset < {EICU_FEATURE_WINDOW_MIN}
          AND ({vaso_where})
    """).fetchdf()
    print(f"  {len(vaso):,} rows")

    # ---------- GCS ----------
    print("Querying GCS...")
    all_g = [v.lower() for vs in EICU_GCS_LABELS.values() for v in vs]
    gcs_where = " OR ".join(
        f"LOWER(nursingchartcelltypevalname) LIKE '%{v}%'" for v in all_g)
    gcs = conn.execute(f"""
        SELECT patientunitstayid AS stay_id, nursingchartoffset AS offset,
               nursingchartvalue AS value
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
        SELECT patientunitstayid AS stay_id, respchartoffset AS offset,
               respchartvalue AS value
        FROM respiratoryCharting
        WHERE patientunitstayid IN ({ids})
          AND respchartoffset >= 0
          AND respchartoffset < {EICU_FEATURE_WINDOW_MIN}
          AND LOWER(respchartvaluelabel) LIKE '%fio2%'
    """).fetchdf()
    print(f"  {len(fio2):,} rows")

    conn.close()

    # ---------- Build tensors with mean-in-bin ----------
    print("\nBuilding tensors...")
    sid_to_row = {sid: i for i, sid in enumerate(stay_ids)}
    sums = np.zeros((len(stay_ids), N_BINS, N_FEATURES), dtype=np.float64)
    counts = np.zeros((len(stay_ids), N_BINS, N_FEATURES), dtype=np.float64)
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def _accumulate(sid, offset, feat, val):
        if sid not in sid_to_row or feat not in feat_idx:
            return
        if pd.isna(val):
            return
        i = sid_to_row[sid]
        j = min(int(offset // BIN_MINUTES), N_BINS - 1)
        k = feat_idx[feat]
        sums[i, j, k] += float(val)
        counts[i, j, k] += 1.0

    # Vitals
    for _, r in vitals.iterrows():
        for col in ['heart_rate', 'respiratory_rate', 'spo2', 'map']:
            if pd.notna(r[col]):
                _accumulate(r['stay_id'], r['observationoffset'], col, r[col])

    # Labs (with Python-level exclusion as double check)
    for _, r in labs.iterrows():
        name = str(r['labname']).lower()
        if lab_is_excluded(name, EICU_LAB_EXCLUSIONS):
            continue
        for canonical, pats in EICU_LAB_PATTERNS.items():
            if canonical not in feat_idx:
                continue
            if any(p in name for p in pats):
                _accumulate(r['stay_id'], r['offset'], canonical, r['value'])
                break

    # Vasopressors — presence-only (value = 1.0)
    for _, r in vaso.iterrows():
        name = str(r['drugname']).lower()
        for canonical, pats in EICU_VASO_PATTERNS.items():
            if canonical not in feat_idx:
                continue
            if any(p in name for p in pats):
                _accumulate(r['stay_id'], r['offset'], canonical, 1.0)
                break

    # GCS (only 'GCS Total' exists in eICU)
    for _, r in gcs.iterrows():
        val = parse_numeric(r['value'])
        if not pd.isna(val):
            _accumulate(r['stay_id'], r['offset'], 'gcs_total', val)

    # FiO2 (normalize to fraction; reject implausible)
    for _, r in fio2.iterrows():
        val = parse_numeric(r['value'])
        if not pd.isna(val):
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            # Reject implausible values
            if val < 0.21 or val > 1.0:
                continue
            _accumulate(r['stay_id'], r['offset'], 'fio2', val)

    masks = (counts > 0).astype(np.float32)
    values = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0).astype(np.float32)

    # ---------- Apply FROZEN MIMIC winsor + z-score ----------
    print("\nApplying frozen MIMIC winsorization + z-score...")
    for _, row in stats.iterrows():
        feat = row['feature']
        if feat not in feat_idx:
            continue
        k = feat_idx[feat]
        m = masks[:, :, k] == 1
        if m.sum() == 0:
            continue
        lo, hi = float(row['winsor_lo']), float(row['winsor_hi'])
        values[:, :, k][m] = np.clip(values[:, :, k][m], lo, hi)
        mean, std = float(row['mean']), float(row['std'])
        if std > 1e-6:
            values[:, :, k][m] = (values[:, :, k][m] - mean) / std

    X = np.concatenate([values, masks], axis=2)
    y = cohort['outcome'].values.astype(np.float32)
    stay_ids_arr = np.array(stay_ids)

    print(f"\n  X shape: {X.shape}")
    print(f"  y: {int(y.sum())} pos / {len(y)} ({y.mean()*100:.2f}%)")
    print(f"  Values min: {values.min():.2f}, max: {values.max():.2f}")
    print(f"  NaN in values: {np.isnan(values).any()}")
    print(f"  Inf in values: {np.isinf(values).any()}")
    print(f"  Masks unique: {np.unique(masks)}")

    print(f"\n  Feature coverage (fraction of bins with mask=1):")
    for k, feat in enumerate(FEATURE_NAMES):
        cov = masks[:, :, k].mean() * 100
        print(f"    {feat:20s}: {cov:5.2f}%")

    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SEQUENCE_DIR / "sequences_eicu.npz"
    np.savez_compressed(out_path, X=X.astype(np.float32), y=y,
                        stay_ids=stay_ids_arr)

    print(f"\nSaved: {out_path}")
    print("\nNext: python3 add_eicu_masks.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
