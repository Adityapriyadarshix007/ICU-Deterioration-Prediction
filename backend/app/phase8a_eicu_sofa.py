#!/usr/bin/env python3
"""
Phase 8a Step 5: Compute SOFA labels for eICU cohort.

Mirrors MIMIC Phase 2 SOFA logic. Uses the same lab exclusions as the
feature extraction step, so no urinary/CSF labs contaminate the score.

Output:
  outputs/tables/eicu_cohort_sofa.csv
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
    SOFA_CHANGE_THRESHOLD,
    EICU_FEATURE_WINDOW_MIN, EICU_OUTCOME_START_MIN, EICU_OUTCOME_END_MIN,
    EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS,
    EICU_VASO_PATTERNS,
    EICU_GCS_SOURCE, EICU_GCS_LABELS,
)


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


def parse_lab_name(labname, patterns, exclusions):
    name = str(labname).lower().strip()
    for excl in exclusions:
        if excl in name:
            return None
    for canonical, pats in patterns.items():
        for p in pats:
            if p in name:
                return canonical
    return None


def build_lab_where(patterns, exclusions):
    all_p = [p.lower() for ps in patterns.values() for p in ps]
    include = " OR ".join(f"LOWER(labname) LIKE '%{p}%'" for p in all_p)
    exclude = " AND ".join(
        f"LOWER(labname) NOT LIKE '%{e}%'" for e in exclusions
    )
    return f"({include}) AND ({exclude})"


def score_cv(map_min, vaso_rates):
    scores = []
    if vaso_rates.get('norepinephrine', 0) > 0:
        scores.append(3 if vaso_rates['norepinephrine'] <= 0.1 else 4)
    if vaso_rates.get('epinephrine', 0) > 0:
        scores.append(3 if vaso_rates['epinephrine'] <= 0.1 else 4)
    if vaso_rates.get('dopamine', 0) > 0:
        d = vaso_rates['dopamine']
        scores.append(4 if d > 15 else 3 if d > 5 else 2)
    if vaso_rates.get('dobutamine', 0) > 0:
        scores.append(2)
    if scores:
        return max(scores)
    if pd.notna(map_min) and map_min < 70:
        return 1
    return 0


def score_resp(fio2_max, spo2_min):
    if pd.isna(fio2_max) or fio2_max <= 0:
        return 0
    if pd.notna(spo2_min):
        sf = spo2_min / fio2_max
        if sf >= 235: return 0
        if sf >= 200: return 1
        if sf >= 150: return 2
        if sf >= 100: return 3
        return 4
    return 0


def score_renal(creat_max):
    if pd.isna(creat_max): return 0
    v = creat_max
    if v < 1.2: return 0
    if v < 2.0: return 1
    if v < 3.5: return 2
    if v < 5.0: return 3
    return 4


def score_liver(bili_max):
    if pd.isna(bili_max): return 0
    v = bili_max
    if v < 1.2: return 0
    if v < 2.0: return 1
    if v < 6.0: return 2
    if v < 12.0: return 3
    return 4


def score_coag(plt_min):
    if pd.isna(plt_min): return 0
    v = plt_min
    if v >= 150: return 0
    if v >= 100: return 1
    if v >= 50: return 2
    if v >= 20: return 3
    return 4


def score_neuro(gcs_min):
    if pd.isna(gcs_min): return 0
    v = gcs_min
    if v >= 15: return 0
    if v >= 13: return 1
    if v >= 10: return 2
    if v >= 6: return 3
    return 4


def extract_window(conn, stay_ids, offset_min, offset_max, tag):
    print(f"\n  Extracting {tag} window [{offset_min}, {offset_max})...")
    ids = ",".join(map(str, stay_ids))

    # Labs (with exclusions)
    lab_where = build_lab_where(EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS)
    labs = conn.execute(f"""
        SELECT patientunitstayid, labname, labresult
        FROM lab
        WHERE patientunitstayid IN ({ids})
          AND labresultoffset >= {offset_min}
          AND labresultoffset < {offset_max}
          AND labresult IS NOT NULL
          AND ({lab_where})
    """).fetchdf()

    # Vitals
    vitals = conn.execute(f"""
        SELECT patientunitstayid, systemicmean AS map, sao2 AS spo2
        FROM vitalPeriodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset >= {offset_min}
          AND observationoffset < {offset_max}
    """).fetchdf()

    # FiO2
    fio2 = conn.execute(f"""
        SELECT patientunitstayid, respchartvalue
        FROM respiratoryCharting
        WHERE patientunitstayid IN ({ids})
          AND respchartoffset >= {offset_min}
          AND respchartoffset < {offset_max}
          AND LOWER(respchartvaluelabel) LIKE '%fio2%'
    """).fetchdf()
    if not fio2.empty:
        fio2['val'] = fio2['respchartvalue'].apply(parse_numeric)
        fio2['val'] = fio2['val'].apply(
            lambda v: v/100 if pd.notna(v) and v > 1 else v)
        fio2 = fio2.groupby('patientunitstayid')['val'].max().reset_index()
        fio2.columns = ['patientunitstayid', 'fio2_max']
    else:
        fio2 = pd.DataFrame(columns=['patientunitstayid', 'fio2_max'])

    # Vasopressors
    all_v = [p.lower() for ps in EICU_VASO_PATTERNS.values() for p in ps]
    vaso_where = " OR ".join(f"LOWER(drugname) LIKE '%{p}%'" for p in all_v)
    vaso = conn.execute(f"""
        SELECT patientunitstayid, drugname, drugrate
        FROM infusionDrug
        WHERE patientunitstayid IN ({ids})
          AND infusionoffset >= {offset_min}
          AND infusionoffset < {offset_max}
          AND ({vaso_where})
    """).fetchdf()

    # GCS
    all_g = [v.lower() for vs in EICU_GCS_LABELS.values() for v in vs]
    gcs_where = " OR ".join(
        f"LOWER(nursingchartcelltypevalname) LIKE '%{v}%'" for v in all_g)
    gcs = conn.execute(f"""
        SELECT patientunitstayid, nursingchartcelltypevalname, nursingchartvalue
        FROM {EICU_GCS_SOURCE}
        WHERE patientunitstayid IN ({ids})
          AND nursingchartoffset >= {offset_min}
          AND nursingchartoffset < {offset_max}
          AND ({gcs_where})
    """).fetchdf()

    print(f"    labs={len(labs):,}, vitals={len(vitals):,}, "
          f"fio2={len(fio2):,}, vaso={len(vaso):,}, gcs={len(gcs):,}")

    return {'labs': labs, 'vitals': vitals, 'fio2': fio2,
            'vaso': vaso, 'gcs': gcs}


def compute_sofa(data, stay_ids):
    rows = []
    for sid in stay_ids:
        # Labs (with exclusions)
        lab_sub = data['labs'][data['labs']['patientunitstayid'] == sid].copy()
        lab_sub['canonical'] = lab_sub['labname'].apply(
            lambda x: parse_lab_name(x, EICU_LAB_PATTERNS, EICU_LAB_EXCLUSIONS))
        creat = lab_sub.loc[
            lab_sub['canonical'] == 'creatinine', 'labresult'].max()
        bili = lab_sub.loc[
            lab_sub['canonical'] == 'bilirubin', 'labresult'].max()
        plt = lab_sub.loc[
            lab_sub['canonical'] == 'platelets', 'labresult'].min()

        # Vitals
        vs = data['vitals'][data['vitals']['patientunitstayid'] == sid]
        map_min = vs['map'].min() if len(vs) > 0 else np.nan
        spo2_min = vs['spo2'].min() if len(vs) > 0 else np.nan

        # FiO2
        f = data['fio2'][data['fio2']['patientunitstayid'] == sid]
        fio2_max = f['fio2_max'].iloc[0] if len(f) > 0 else np.nan

        # Vasopressors
        v = data['vaso'][data['vaso']['patientunitstayid'] == sid]
        vaso_rates = {}
        for name, pats in EICU_VASO_PATTERNS.items():
            sub = v[v['drugname'].str.lower().str.contains(pats[0], na=False)]
            if len(sub) > 0:
                rates = sub['drugrate'].apply(parse_numeric).dropna()
                vaso_rates[name] = float(rates.max()) if len(rates) > 0 else 0.0
            else:
                vaso_rates[name] = 0.0

        # GCS
        g = data['gcs'][data['gcs']['patientunitstayid'] == sid].copy()
        gcs_min = np.nan
        if len(g) > 0:
            g['val'] = g['nursingchartvalue'].apply(parse_numeric)
            g = g.dropna(subset=['val'])
            if len(g) > 0:
                gcs_min = float(g['val'].min())

        cv = score_cv(map_min, vaso_rates)
        resp = score_resp(fio2_max, spo2_min)
        renal = score_renal(creat)
        liver = score_liver(bili)
        coag = score_coag(plt)
        neuro = score_neuro(gcs_min)

        rows.append({
            'stay_id': sid,
            'sofa_cv': cv, 'sofa_resp': resp, 'sofa_renal': renal,
            'sofa_liver': liver, 'sofa_coag': coag, 'sofa_neuro': neuro,
            'sofa_total': cv + resp + renal + liver + coag + neuro,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("PHASE 8a — STEP 5: COMPUTE SOFA LABELS (with lab exclusions)")
    print("=" * 78)

    cohort_path = TABLE_DIR / "eicu_cohort.csv"
    if not cohort_path.exists():
        print(f"ERROR: {cohort_path} not found.")
        return 1

    cohort = pd.read_csv(cohort_path)
    stay_ids = cohort['patientunitstayid'].tolist()
    print(f"\nCohort: {len(cohort):,} stays")

    conn = duckdb.connect(str(EICU_DB_NAME), read_only=True)

    print("\n--- FEATURE WINDOW ---")
    feat_data = extract_window(conn, stay_ids, 0, EICU_FEATURE_WINDOW_MIN, 'feature')

    print("\n--- OUTCOME WINDOW ---")
    out_data = extract_window(conn, stay_ids,
                              EICU_OUTCOME_START_MIN, EICU_OUTCOME_END_MIN, 'outcome')

    conn.close()

    print("\nComputing feature SOFA...")
    sofa_feat = compute_sofa(feat_data, stay_ids)
    sofa_feat = sofa_feat.rename(columns={
        'sofa_total': 'sofa_feature',
        'sofa_cv': 'sofa_feature_cv',
        'sofa_resp': 'sofa_feature_resp',
        'sofa_renal': 'sofa_feature_renal',
        'sofa_liver': 'sofa_feature_liver',
        'sofa_coag': 'sofa_feature_coag',
        'sofa_neuro': 'sofa_feature_neuro',
    })

    print("Computing outcome SOFA...")
    sofa_out = compute_sofa(out_data, stay_ids)
    sofa_out = sofa_out.rename(columns={
        'sofa_total': 'sofa_outcome',
        'sofa_cv': 'sofa_outcome_cv',
        'sofa_resp': 'sofa_outcome_resp',
        'sofa_renal': 'sofa_outcome_renal',
        'sofa_liver': 'sofa_outcome_liver',
        'sofa_coag': 'sofa_outcome_coag',
        'sofa_neuro': 'sofa_outcome_neuro',
    })

    print("\nMerging...")
    merged = cohort.rename(columns={'patientunitstayid': 'stay_id'})
    merged = merged.merge(sofa_feat, on='stay_id', how='left')
    merged = merged.merge(sofa_out, on='stay_id', how='left')

    sofa_cols = [c for c in merged.columns if c.startswith('sofa_')]
    for c in sofa_cols:
        merged[c] = merged[c].fillna(0)

    merged['sofa_change'] = merged['sofa_outcome'] - merged['sofa_feature']
    merged['outcome'] = (merged['sofa_change'] >= SOFA_CHANGE_THRESHOLD).astype(int)

    prev = merged['outcome'].mean() * 100
    print(f"\nProgression rate: {prev:.2f}%")
    print(f"Feature SOFA: {merged['sofa_feature'].mean():.2f} ± {merged['sofa_feature'].std():.2f}")
    print(f"Outcome SOFA: {merged['sofa_outcome'].mean():.2f} ± {merged['sofa_outcome'].std():.2f}")
    print(f"Delta SOFA: {merged['sofa_change'].mean():.2f} ± {merged['sofa_change'].std():.2f}")

    out = TABLE_DIR / "eicu_cohort_sofa.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("\nNext: python3 phase8a_eicu_extract_seq.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
