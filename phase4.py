# ============================================================
# PHASE 4: FEATURE ENGINEERING + TARGET CREATION
# ============================================================
# This file creates the ML-ready dataset with:
# 1. 8 Core Features: HR, SBP, DBP, Creatinine, GCS, Lactate, Urine, FiO2
# 2. Delta Features (rate of change)
# 3. Target Variable (deterioration label)
# ============================================================

import duckdb
import pandas as pd
import numpy as np
import os

BASE = "data"
ICU = f"{BASE}/icu"
HOSP = f"{BASE}/hosp"
os.makedirs("outputs/tables", exist_ok=True)

con = duckdb.connect("mods.duckdb")

print("="*60)
print("PHASE 4: FEATURE ENGINEERING")
print("="*60)

# ============================================================
# STEP 1: Extract 8 Core Features from Chartevents
# ============================================================
print("\n[1] Extracting 8 core features...")

# MIMIC-IV Item IDs for our 8 features
# Source: https://mimic.mit.edu/docs/iv/modules/icu/chartevents/
ITEM_IDS = {
    'heart_rate': 220045,      # HR
    'sbp': 220179,             # Systolic BP
    'dbp': 220180,             # Diastolic BP
    'gcs': 220739,             # Glasgow Coma Scale (Brain)
    'lactate': 225664,         # Lactate
    'urine_output': 226559,    # Urine Output (mL)
    'fio2': 223835,            # FiO2 (Oxygen)
    'creatinine': 50912        # Creatinine (Kidney) - from labevents
}

con.execute("DROP TABLE IF EXISTS features_raw;")

con.execute(f"""
CREATE TABLE features_raw AS
SELECT 
    ce.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (ce.charttime - i.intime)) / 3600) AS icu_hour,
    ce.itemid,
    ce.valuenum
FROM chartevents ce
JOIN icustays i ON ce.stay_id = i.stay_id
WHERE ce.itemid IN ({', '.join(map(str, ITEM_IDS.values()))})
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum > 0
  AND ce.charttime BETWEEN i.intime AND i.outtime
""")

print(f"✅ Raw features extracted")

# ============================================================
# STEP 2: Pivot to Wide Format (One row per stay_id + icu_hour)
# ============================================================
print("\n[2] Pivoting to wide format...")

# Map itemid to feature names
con.execute("DROP TABLE IF EXISTS itemid_map;")
con.execute("""
CREATE TEMP TABLE itemid_map AS
SELECT * FROM (VALUES
    (220045, 'heart_rate'),
    (220179, 'sbp'),
    (220180, 'dbp'),
    (220739, 'gcs'),
    (225664, 'lactate'),
    (226559, 'urine_output'),
    (223835, 'fio2')
) AS t(itemid, feature_name);
""")

con.execute("DROP TABLE IF EXISTS features_wide;")

con.execute("""
CREATE TABLE features_wide AS
SELECT 
    fr.stay_id,
    fr.icu_hour,
    MAX(CASE WHEN im.feature_name = 'heart_rate' THEN fr.valuenum END) AS heart_rate,
    MAX(CASE WHEN im.feature_name = 'sbp' THEN fr.valuenum END) AS sbp,
    MAX(CASE WHEN im.feature_name = 'dbp' THEN fr.valuenum END) AS dbp,
    MAX(CASE WHEN im.feature_name = 'gcs' THEN fr.valuenum END) AS gcs,
    MAX(CASE WHEN im.feature_name = 'lactate' THEN fr.valuenum END) AS lactate,
    MAX(CASE WHEN im.feature_name = 'urine_output' THEN fr.valuenum END) AS urine_output,
    MAX(CASE WHEN im.feature_name = 'fio2' THEN fr.valuenum END) AS fio2
FROM features_raw fr
JOIN itemid_map im ON fr.itemid = im.itemid
GROUP BY fr.stay_id, fr.icu_hour
ORDER BY fr.stay_id, fr.icu_hour;
""")

print("✅ Wide format created")

# ============================================================
# STEP 3: Add Creatinine from Labevents
# ============================================================
print("\n[3] Adding Creatinine from labevents...")

con.execute("DROP TABLE IF EXISTS creatinine_lab;")

con.execute(f"""
CREATE TABLE creatinine_lab AS
SELECT 
    i.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600) AS icu_hour,
    AVG(le.valuenum) AS creatinine
FROM read_csv_auto('{HOSP}/labevents.csv.gz') le
JOIN icustays i ON le.hadm_id = i.hadm_id
WHERE le.itemid = 50912  -- Creatinine
  AND le.valuenum IS NOT NULL
  AND le.valuenum > 0
  AND le.charttime BETWEEN i.intime AND i.outtime
GROUP BY i.stay_id, FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600);
""")

# Merge creatinine into features_wide
con.execute("DROP TABLE IF EXISTS features_wide_with_creat;")

con.execute("""
CREATE TABLE features_wide_with_creat AS
SELECT 
    fw.stay_id,
    fw.icu_hour,
    fw.heart_rate,
    fw.sbp,
    fw.dbp,
    fw.gcs,
    fw.lactate,
    fw.urine_output,
    fw.fio2,
    cl.creatinine
FROM features_wide fw
LEFT JOIN creatinine_lab cl 
    ON fw.stay_id = cl.stay_id 
    AND fw.icu_hour = cl.icu_hour
ORDER BY fw.stay_id, fw.icu_hour;
""")

print("✅ Creatinine added")

# ============================================================
# STEP 4: Calculate Delta Features (Rate of Change)
# ============================================================
print("\n[4] Calculating delta features...")

# Load data into pandas for delta calculation
df = con.execute("""
SELECT * FROM features_wide_with_creat
ORDER BY stay_id, icu_hour;
""").fetchdf()

print(f"✅ Loaded {len(df)} rows, {df['stay_id'].nunique()} unique ICU stays")

# Define feature columns (excluding stay_id and icu_hour)
feature_cols = ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'urine_output', 'fio2', 'creatinine']

# Calculate delta features (change from hour 0 to hour 6 for each patient)
print("\n[4a] Calculating delta features (Hour 6 - Hour 0)...")

# Get first 6 hours of data per patient
df_first_6h = df[df['icu_hour'] <= 6].copy()

# Pivot to get hour 0 and hour 6 values
df_hour0 = df_first_6h[df_first_6h['icu_hour'] == 0].copy()
df_hour6 = df_first_6h[df_first_6h['icu_hour'] == 6].copy()

# Rename columns for merging
df_hour0 = df_hour0.rename(columns={col: f'{col}_hour0' for col in feature_cols})
df_hour6 = df_hour6.rename(columns={col: f'{col}_hour6' for col in feature_cols})

# Merge on stay_id
df_delta = df_hour0[['stay_id'] + [f'{col}_hour0' for col in feature_cols]].merge(
    df_hour6[['stay_id'] + [f'{col}_hour6' for col in feature_cols]],
    on='stay_id',
    how='inner'
)

# Calculate deltas
for col in feature_cols:
    df_delta[f'{col}_delta'] = df_delta[f'{col}_hour6'] - df_delta[f'{col}_hour0']

print(f"✅ Delta features calculated for {len(df_delta)} patients")

# ============================================================
# STEP 5: Create Target Variable (Deterioration Label)
# ============================================================
print("\n[5] Creating target variable (deterioration label)...")

# Load SOFA scores from Phase 3
df_sofa = con.execute("""
SELECT stay_id, icu_hour, total_sofa_norm
FROM sofa_phase3_norm
ORDER BY stay_id, icu_hour;
""").fetchdf()

print(f"✅ Loaded SOFA scores for {df_sofa['stay_id'].nunique()} unique stays")

# Calculate SOFA change from hour 6 to hour 18
df_sofa_hour6 = df_sofa[df_sofa['icu_hour'] == 6].copy()
df_sofa_hour18 = df_sofa[df_sofa['icu_hour'] == 18].copy()

df_sofa_hour6 = df_sofa_hour6.rename(columns={'total_sofa_norm': 'sofa_hour6'})
df_sofa_hour18 = df_sofa_hour18.rename(columns={'total_sofa_norm': 'sofa_hour18'})

df_sofa_merge = df_sofa_hour6[['stay_id', 'sofa_hour6']].merge(
    df_sofa_hour18[['stay_id', 'sofa_hour18']],
    on='stay_id',
    how='inner'
)

# Target: 1 if SOFA increased from hour 6 to hour 18 (deterioration)
df_sofa_merge['sofa_change'] = df_sofa_merge['sofa_hour18'] - df_sofa_merge['sofa_hour6']
df_sofa_merge['deteriorated'] = (df_sofa_merge['sofa_change'] > 0).astype(int)

print(f"✅ Target variable created")
print(f"   - Deteriorated (1): {df_sofa_merge['deteriorated'].sum()} patients")
print(f"   - Stable (0): {len(df_sofa_merge) - df_sofa_merge['deteriorated'].sum()} patients")

# ============================================================
# STEP 6: Merge Everything into Final Dataset
# ============================================================
print("\n[6] Creating final dataset...")

# Merge delta features with target
df_final = df_delta.merge(
    df_sofa_merge[['stay_id', 'sofa_hour6', 'sofa_hour18', 'sofa_change', 'deteriorated']],
    on='stay_id',
    how='inner'
)

print(f"✅ Final dataset: {len(df_final)} rows, {len(df_final.columns)} columns")

# ============================================================
# STEP 7: Save Dataset
# ============================================================
print("\n[7] Saving datasets...")

# Save full dataset
df_final.to_csv("outputs/tables/ml_dataset_full.csv", index=False)
print(f"✅ Saved: outputs/tables/ml_dataset_full.csv")

# Save feature columns only (for imputation comparison)
feature_cols_final = [f'{col}_hour0' for col in feature_cols] + \
                     [f'{col}_hour6' for col in feature_cols] + \
                     [f'{col}_delta' for col in feature_cols]

X_df = df_final[feature_cols_final]
y_df = df_final[['stay_id', 'deteriorated']]

X_df.to_csv("outputs/tables/X_features.csv", index=False)
y_df.to_csv("outputs/tables/y_target.csv", index=False)
print(f"✅ Saved: outputs/tables/X_features.csv ({len(X_df.columns)} features)")
print(f"✅ Saved: outputs/tables/y_target.csv")

# ============================================================
# STEP 8: Summary Statistics
# ============================================================
print("\n" + "="*60)
print("PHASE 4 COMPLETE - SUMMARY")
print("="*60)
print(f"Total patients: {len(df_final)}")
print(f"Features per patient: {len(feature_cols_final)}")
print(f"Class balance:")
print(f"  - Stable (0): {(df_final['deteriorated'] == 0).sum()} ({100*(df_final['deteriorated'] == 0).sum()/len(df_final):.1f}%)")
print(f"  - Deteriorated (1): {(df_final['deteriorated'] == 1).sum()} ({100*(df_final['deteriorated'] == 1).sum()/len(df_final):.1f}%)")

print("\n✅ Phase 4 completed successfully!")