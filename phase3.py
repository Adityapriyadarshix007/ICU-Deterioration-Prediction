# ============================================================
# Phase 3: Cardiovascular + Multi-organ SOFA (MIMIC-IV)
# ============================================================

import duckdb
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Paths (adjust to your setup)
# -----------------------------
BASE = "data"
ICU = f"{BASE}/icu"
HOSP = f"{BASE}/hosp"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Connect DuckDB
# -----------------------------
con = duckdb.connect("mods.duckdb")
print("✅ Connected to DuckDB")

# -----------------------------
# Load ICU tables as views
# -----------------------------
con.execute(f"""
CREATE OR REPLACE VIEW icustays AS
SELECT * FROM read_csv_auto('{ICU}/icustays.csv.gz');
""")

con.execute(f"""
CREATE OR REPLACE VIEW chartevents AS
SELECT * FROM read_csv_auto('{ICU}/chartevents.csv.gz');
""")

con.execute(f"""
CREATE OR REPLACE VIEW inputevents AS
SELECT * FROM read_csv_auto('{ICU}/inputevents.csv.gz');
""")

con.execute(f"""
CREATE OR REPLACE VIEW d_items AS
SELECT * FROM read_csv_auto('{ICU}/d_items.csv.gz');
""")

print("✅ Tables loaded")

# ============================================================
# Phase 2: Hourly organ measurements - FIXED TO INCLUDE ALL ORGANS
# ============================================================
print("\nCreating hourly organ measurements...")

con.execute("DROP TABLE IF EXISTS sofa_phase2_hourly;")

# Use CORRECT MIMIC-IV lab item IDs
creat_itemid = 50912      # Creatinine (serum) - CORRECT ID
bili_itemid = 50885       # Bilirubin, total - CORRECT ID  
platelet_itemid = 51265   # Platelet Count - CORRECT ID

print(f"Using CORRECT item IDs: Creatinine={creat_itemid}, Bilirubin={bili_itemid}, Platelets={platelet_itemid}")

# Now create the sofa_phase2_hourly table
con.execute(f"""
CREATE TABLE sofa_phase2_hourly AS
WITH hourly_metrics AS (
    -- Cardiovascular: MAP
    SELECT
        ce.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (ce.charttime - i.intime)) / 3600) AS icu_hour,
        'cardiovascular' AS organ,
        AVG(ce.valuenum) AS avg_value
    FROM chartevents ce
    JOIN icustays i ON ce.stay_id = i.stay_id
    WHERE ce.itemid = 220045  -- MAP
      AND ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN i.intime AND i.outtime
    GROUP BY ce.stay_id, FLOOR(EXTRACT(EPOCH FROM (ce.charttime - i.intime)) / 3600)
    
    UNION ALL
    
    -- Renal: Creatinine - LOAD FROM HOSPITAL DATA
    SELECT
        i.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600) AS icu_hour,
        'renal' AS organ,
        AVG(le.valuenum) AS avg_value
    FROM read_csv_auto('{HOSP}/labevents.csv.gz') le
    JOIN icustays i ON le.hadm_id = i.hadm_id
    WHERE le.itemid = {creat_itemid}
      AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN i.intime AND i.outtime
    GROUP BY i.stay_id, FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600)
    
    UNION ALL
    
    -- Liver: Bilirubin - LOAD FROM HOSPITAL DATA
    SELECT
        i.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600) AS icu_hour,
        'liver' AS organ,
        AVG(le.valuenum) AS avg_value
    FROM read_csv_auto('{HOSP}/labevents.csv.gz') le
    JOIN icustays i ON le.hadm_id = i.hadm_id
    WHERE le.itemid = {bili_itemid}
      AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN i.intime AND i.outtime
    GROUP BY i.stay_id, FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600)
    
    UNION ALL
    
    -- Coagulation: Platelets - LOAD FROM HOSPITAL DATA
    SELECT
        i.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600) AS icu_hour,
        'coagulation' AS organ,
        AVG(le.valuenum) AS avg_value
    FROM read_csv_auto('{HOSP}/labevents.csv.gz') le
    JOIN icustays i ON le.hadm_id = i.hadm_id
    WHERE le.itemid = {platelet_itemid}
      AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN i.intime AND i.outtime
    GROUP BY i.stay_id, FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime)) / 3600)
)
SELECT * FROM hourly_metrics;
""")

# Check what data we have
count_by_organ = con.execute("""
    SELECT organ, COUNT(*) as count, 
           MIN(avg_value) as min_val, 
           MAX(avg_value) as max_val,
           AVG(avg_value) as avg_val
    FROM sofa_phase2_hourly
    GROUP BY organ
    ORDER BY organ;
""").fetchdf()

print("\nData in sofa_phase2_hourly:")
print(count_by_organ)

# ============================================================
# Check all inputevents that could be vasopressors
# ============================================================
print("\nChecking available vasopressors...")

vasopressors = con.execute("""
SELECT DISTINCT ie.itemid, di.label
FROM inputevents ie
JOIN d_items di
  ON ie.itemid = di.itemid
WHERE LOWER(di.label) LIKE '%norepinephrine%'
   OR LOWER(di.label) LIKE '%epinephrine%'
   OR LOWER(di.label) LIKE '%dopamine%'
   OR LOWER(di.label) LIKE '%dobutamine%'
ORDER BY ie.itemid;
""").fetchall()

print(f"Found {len(vasopressors)} vasopressor types:")
for itemid, label in vasopressors:
    print(f"  - ItemID {itemid}: {label}")

# ============================================================
# Create vasopressor tables
# ============================================================
con.execute("DROP TABLE IF EXISTS sofa_pressor_raw;")

con.execute("""
CREATE TABLE sofa_pressor_raw AS
SELECT
    i.stay_id,
    ie.starttime AS charttime,
    ie.itemid,
    ie.rate
FROM inputevents ie
JOIN icustays i
  ON ie.stay_id = i.stay_id
WHERE ie.itemid IN (221906, 221289, 221662, 221653)
  AND ie.rate IS NOT NULL
  AND ie.starttime BETWEEN i.intime AND i.outtime;
""")

con.execute("DROP TABLE IF EXISTS sofa_pressor_ne;")

con.execute("""
CREATE TABLE sofa_pressor_ne AS
SELECT
    stay_id,
    charttime,
    CASE WHEN itemid = 221906 THEN rate ELSE 0 END AS norepinephrine,
    CASE WHEN itemid = 221289 THEN rate ELSE 0 END AS epinephrine,
    CASE WHEN itemid = 221662 THEN rate ELSE 0 END AS dopamine,
    CASE WHEN itemid = 221653 THEN 1 ELSE 0 END AS dobutamine
FROM sofa_pressor_raw;
""")

con.execute("DROP TABLE IF EXISTS sofa_pressor_hourly;")

con.execute("""
CREATE TABLE sofa_pressor_hourly AS
SELECT
    p.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (p.charttime - i.intime)) / 3600) AS icu_hour,
    MAX(p.norepinephrine) AS max_ne,
    MAX(p.epinephrine) AS max_epi,
    MAX(p.dopamine) AS max_dopa,
    MAX(p.dobutamine) AS has_dobutamine
FROM sofa_pressor_ne p
JOIN icustays i
  ON p.stay_id = i.stay_id
GROUP BY p.stay_id, icu_hour;
""")

con.execute("DROP TABLE IF EXISTS sofa_pressor_early6h;")

con.execute("""
CREATE TABLE sofa_pressor_early6h AS
SELECT
    i.stay_id,
    MAX(p.norepinephrine) AS max_ne_6h,
    MAX(p.epinephrine) AS max_epi_6h,
    MAX(p.dopamine) AS max_dopa_6h,
    MAX(p.dobutamine) AS any_dobutamine_6h
FROM sofa_pressor_ne p
JOIN icustays i
  ON p.stay_id = i.stay_id
WHERE p.charttime BETWEEN i.intime
                      AND i.intime + INTERVAL '6' HOUR
GROUP BY i.stay_id;
""")

con.execute("DROP TABLE IF EXISTS sofa_cardio_phase3;")

con.execute("""
CREATE TABLE sofa_cardio_phase3 AS
SELECT
    m.stay_id,
    m.icu_hour,
    m.avg_value AS map_value,
    COALESCE(v.max_ne,0) AS max_ne,
    COALESCE(v.max_epi,0) AS max_epi,
    COALESCE(v.max_dopa,0) AS max_dopa,
    COALESCE(v.has_dobutamine,0) AS has_dobutamine
FROM sofa_phase2_hourly m
LEFT JOIN sofa_pressor_hourly v
  ON m.stay_id = v.stay_id
 AND m.icu_hour = v.icu_hour
WHERE m.organ = 'cardiovascular';
""")

con.execute("DROP TABLE IF EXISTS sofa_cardio_score;")

con.execute("""
CREATE TABLE sofa_cardio_score AS
SELECT
    stay_id,
    icu_hour,
    'cardiovascular' AS organ,
    CASE
        WHEN max_ne > 0.1 OR max_epi > 0.1 OR max_dopa > 15 THEN 4
        WHEN max_ne > 0 OR max_epi > 0 OR max_dopa > 5 THEN 3
        WHEN has_dobutamine = 1 OR max_dopa > 0 THEN 2
        WHEN map_value < 70 THEN 1
        ELSE 0
    END AS sofa_score
FROM sofa_cardio_phase3
ORDER BY stay_id, icu_hour;
""")

con.execute("DROP TABLE IF EXISTS sofa_noncardio_scores;")

con.execute("""
CREATE TABLE sofa_noncardio_scores AS
SELECT
    stay_id,
    icu_hour,
    organ,
    CASE
        WHEN organ='renal' THEN
            CASE
                WHEN avg_value < 1.2 THEN 0
                WHEN avg_value < 2.0 THEN 1
                WHEN avg_value < 3.5 THEN 2
                WHEN avg_value < 5.0 THEN 3
                ELSE 4
            END
        WHEN organ='liver' THEN
            CASE
                WHEN avg_value < 1.2 THEN 0
                WHEN avg_value < 2.0 THEN 1
                WHEN avg_value < 6.0 THEN 2
                WHEN avg_value < 12.0 THEN 3
                ELSE 4
            END
        WHEN organ='coagulation' THEN
            CASE
                WHEN avg_value >= 150 THEN 0
                WHEN avg_value >= 100 THEN 1
                WHEN avg_value >= 50 THEN 2
                WHEN avg_value >= 20 THEN 3
                ELSE 4
            END
    END AS sofa_score
FROM sofa_phase2_hourly
WHERE organ <> 'cardiovascular';
""")

con.execute("DROP TABLE IF EXISTS sofa_phase3_scores;")

con.execute("""
CREATE TABLE sofa_phase3_scores AS
SELECT * FROM sofa_cardio_score
UNION ALL
SELECT * FROM sofa_noncardio_scores;
""")

con.execute("DROP TABLE IF EXISTS sofa_phase3_total;")

con.execute("""
CREATE TABLE sofa_phase3_total AS
SELECT
    stay_id,
    icu_hour,
    SUM(sofa_score) AS total_sofa,
    COUNT(sofa_score) AS organs_present
FROM sofa_phase3_scores
GROUP BY stay_id, icu_hour
ORDER BY stay_id, icu_hour;
""")

con.execute("DROP TABLE IF EXISTS sofa_early_features_6h;")

con.execute("""
CREATE TABLE sofa_early_features_6h AS
WITH cv_features AS (
    SELECT
        i.stay_id,
        MIN(m.avg_value) AS min_map_6h,
        p.max_ne_6h,
        p.any_dobutamine_6h
    FROM icustays i
    LEFT JOIN sofa_phase2_hourly m
      ON i.stay_id = m.stay_id
     AND m.organ = 'cardiovascular'
     AND m.icu_hour < 6
    LEFT JOIN sofa_pressor_early6h p
      ON i.stay_id = p.stay_id
    GROUP BY i.stay_id, p.max_ne_6h, p.any_dobutamine_6h
),
noncv_features AS (
    SELECT
        stay_id,
        MAX(CASE 
            WHEN organ='renal' THEN 
                CASE
                    WHEN avg_value < 1.2 THEN 0
                    WHEN avg_value < 2.0 THEN 1
                    WHEN avg_value < 3.5 THEN 2
                    WHEN avg_value < 5.0 THEN 3
                    ELSE 4
                END
            END) AS renal_sofa_6h,
        MAX(CASE 
            WHEN organ='liver' THEN
                CASE
                    WHEN avg_value < 1.2 THEN 0
                    WHEN avg_value < 2.0 THEN 1
                    WHEN avg_value < 6.0 THEN 2
                    WHEN avg_value < 12.0 THEN 3
                    ELSE 4
                END
            END) AS liver_sofa_6h,
        MAX(CASE 
            WHEN organ='coagulation' THEN
                CASE
                    WHEN avg_value >= 150 THEN 0
                    WHEN avg_value >= 100 THEN 1
                    WHEN avg_value >= 50 THEN 2
                    WHEN avg_value >= 20 THEN 3
                    ELSE 4
                END
            END) AS coag_sofa_6h
    FROM sofa_phase2_hourly
    WHERE organ IN ('renal','liver','coagulation') 
      AND icu_hour < 6
    GROUP BY stay_id
)
SELECT
    cv.stay_id,
    cv.min_map_6h,
    cv.max_ne_6h,
    cv.any_dobutamine_6h,
    ncv.renal_sofa_6h,
    ncv.liver_sofa_6h,
    ncv.coag_sofa_6h
FROM cv_features cv
LEFT JOIN noncv_features ncv
  ON cv.stay_id = ncv.stay_id;
""")

# ============================================================
# Run diagnostic queries
# ============================================================
print("\nRunning diagnostic queries...")

# Total rows and unique ICU stays
result = con.execute("""
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT stay_id) AS unique_stays
FROM sofa_early_features_6h;
""").fetchall()
print(f"Total rows: {result[0][0]}, Unique stays: {result[0][1]}")

# Count NULLs per column
result = con.execute("""
SELECT
    SUM(CASE WHEN min_map_6h IS NULL THEN 1 ELSE 0 END) AS null_min_map,
    SUM(CASE WHEN max_ne_6h IS NULL THEN 1 ELSE 0 END) AS null_max_ne,
    SUM(CASE WHEN any_dobutamine_6h IS NULL THEN 1 ELSE 0 END) AS null_dobutamine,
    SUM(CASE WHEN renal_sofa_6h IS NULL THEN 1 ELSE 0 END) AS null_renal,
    SUM(CASE WHEN liver_sofa_6h IS NULL THEN 1 ELSE 0 END) AS null_liver,
    SUM(CASE WHEN coag_sofa_6h IS NULL THEN 1 ELSE 0 END) AS null_coag
FROM sofa_early_features_6h;
""").fetchall()
print(f"NULL counts - MAP: {result[0][0]}, NE: {result[0][1]}, Dobutamine: {result[0][2]}, Renal: {result[0][3]}, Liver: {result[0][4]}, Coag: {result[0][5]}")

# Check SOFA scores
noncardio_stats = con.execute("""
    SELECT organ, COUNT(*) as count,
           MIN(sofa_score) as min_score,
           MAX(sofa_score) as max_score,
           AVG(sofa_score) as avg_score
    FROM sofa_noncardio_scores
    GROUP BY organ
    ORDER BY organ;
""").fetchdf()

print("\n✅ Non-cardiovascular SOFA statistics (ALL ORGANS SHOULD HAVE DATA):")
print(noncardio_stats)

# ============================================================
# Create normalized SOFA table
# ============================================================
con.execute("DROP TABLE IF EXISTS sofa_phase3_norm;")

con.execute("""
CREATE TABLE sofa_phase3_norm AS
SELECT
    t.stay_id,
    t.icu_hour,
    COALESCE(MAX(CASE WHEN s.organ='cardiovascular' THEN s.sofa_score END)/4.0, 0) AS sofa_cardio_norm,
    COALESCE(MAX(CASE WHEN s.organ='renal' THEN s.sofa_score END)/4.0, 0) AS sofa_renal_norm,
    COALESCE(MAX(CASE WHEN s.organ='liver' THEN s.sofa_score END)/4.0, 0) AS sofa_liver_norm,
    COALESCE(MAX(CASE WHEN s.organ='coagulation' THEN s.sofa_score END)/4.0, 0) AS sofa_coag_norm,
    COALESCE(SUM(s.sofa_score)/16.0, 0) AS total_sofa_norm
FROM sofa_phase3_total t
LEFT JOIN sofa_phase3_scores s
  ON t.stay_id = s.stay_id AND t.icu_hour = s.icu_hour
GROUP BY t.stay_id, t.icu_hour
ORDER BY t.stay_id, t.icu_hour;
""")

# Check normalized values
result = con.execute("""
SELECT MIN(sofa_cardio_norm), MAX(sofa_cardio_norm),
       MIN(sofa_renal_norm), MAX(sofa_renal_norm),
       MIN(sofa_liver_norm), MAX(sofa_liver_norm),
       MIN(sofa_coag_norm), MAX(sofa_coag_norm),
       MIN(total_sofa_norm), MAX(total_sofa_norm)
FROM sofa_phase3_norm;
""").fetchall()
print("\n✅ Normalized SOFA ranges:")
print(f"Cardiovascular: {result[0][0]:.4f} to {result[0][1]:.4f}")
print(f"Renal: {result[0][2]:.4f} to {result[0][3]:.4f}")
print(f"Liver: {result[0][4]:.4f} to {result[0][5]:.4f}")
print(f"Coagulation: {result[0][6]:.4f} to {result[0][7]:.4f}")
print(f"Total SOFA: {result[0][8]:.4f} to {result[0][9]:.4f}")

# ============================================================
# Visualization (REAL DATA ONLY – NO SYNTHETIC JITTER)
# ============================================================

print("\nCreating visualizations...")

# Load normalized SOFA
df_norm = con.execute("""
SELECT stay_id, icu_hour,
       sofa_cardio_norm, sofa_renal_norm,
       sofa_liver_norm, sofa_coag_norm, total_sofa_norm
FROM sofa_phase3_norm
ORDER BY stay_id, icu_hour;
""").fetchdf()

print(f"✅ Loaded {len(df_norm)} records, {df_norm['stay_id'].nunique()} unique ICU stays")

# Fill missing organ values with 0 (clinical assumption: no evidence = no dysfunction)
organs = ["sofa_cardio_norm", "sofa_renal_norm", "sofa_liver_norm", "sofa_coag_norm"]
df_norm[organs] = df_norm[organs].fillna(0)

# ---------------------------
# GRAPH 1: First 6h distributions
# ---------------------------
df_early = df_norm[df_norm['icu_hour'] < 6].copy()
if len(df_early) == 0:
    print("⚠️ No data in first 6 hours, using first 24 hours instead")
    df_early = df_norm[df_norm['icu_hour'] < 24].copy()

colors = ['red', 'blue', 'green', 'purple']
organ_names = ["Cardiovascular", "Renal", "Liver", "Coagulation"]

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Distribution of Normalized SOFA Scores (First 6 ICU Hours)',
              fontsize=16, fontweight='bold')

for ax, col, color, organ_name in zip(axes1.flat, organs, colors, organ_names):
    if df_early[col].nunique() > 1:
        sns.histplot(df_early[col], bins=15, kde=True, color=color, ax=ax)
        mean_val = df_early[col].mean()
        ax.axvline(mean_val, color=color, linestyle='--',
                   linewidth=2, label=f"Mean: {mean_val:.3f}")
    else:
        const_val = df_early[col].iloc[0]
        ax.hist(df_early[col], bins=1, color=color, alpha=0.7)
        ax.set_title(f"{organ_name} (Constant value: {const_val:.2f})")

    ax.set_title(organ_name)
    ax.set_xlabel('Normalized SOFA Score')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{FIG_DIR}/sofa_distribution_6h.png", dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------
# GRAPH 2: Patient-specific SOFA trajectories
# ---------------------------

# Pick patient with highest SOFA variability
patient_var = (
    df_norm.groupby('stay_id')
    .agg(hour_count=('total_sofa_norm', 'count'),
         std_dev=('total_sofa_norm', 'std'))
    .reset_index()
)

valid_patients = patient_var[
    (patient_var['hour_count'] >= 24) &
    (patient_var['std_dev'] > 0)
]

if len(valid_patients) > 0:
    sample_stay = valid_patients.loc[
        valid_patients['std_dev'].idxmax(), 'stay_id'
    ]
    df_one = df_norm[df_norm['stay_id'] == sample_stay].copy()
    print(f"Selected stay {sample_stay} with {len(df_one)} ICU hours")
else:
    sample_stay = None
    df_one = df_norm.groupby('icu_hour').mean().reset_index()
    print("No highly variable stay found — using population average")

# Melt for plotting
df_melt = df_one.melt(
    id_vars=['icu_hour'],
    value_vars=organs + ['total_sofa_norm'],
    var_name='organ', value_name='sofa_value'
)

df_melt['organ'] = df_melt['organ'].replace({
    'sofa_cardio_norm': 'Cardiovascular',
    'sofa_renal_norm': 'Renal',
    'sofa_liver_norm': 'Liver',
    'sofa_coag_norm': 'Coagulation',
    'total_sofa_norm': 'Total SOFA'
})

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(
    f"SOFA Trajectories {'for ICU Stay ' + str(sample_stay) if sample_stay else '(Average Across Patients)'}",
    fontsize=16, fontweight='bold'
)

# Subplot 1: Organ-wise trajectories (REAL DATA ONLY)
ax1 = axes2[0]
color_map = {
    'Cardiovascular': 'red',
    'Renal': 'blue',
    'Liver': 'green',
    'Coagulation': 'purple'
}

for organ in organ_names:
    od = df_melt[df_melt['organ'] == organ]
    if len(od) > 0:
        ax1.plot(
            od['icu_hour'], od['sofa_value'],
            marker='o', linewidth=2,
            color=color_map[organ], label=organ
        )

ax1.set_title('Individual Organ SOFA Scores')
ax1.set_xlabel('ICU Hour')
ax1.set_ylabel('Normalized SOFA Score')
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Subplot 2: Total SOFA
ax2 = axes2[1]
ax2.plot(
    df_one['icu_hour'], df_one['total_sofa_norm'],
    marker='s', linewidth=3,
    color='black', label='Total SOFA'
)

if len(df_one) > 5:
    df_one['total_smooth'] = df_one['total_sofa_norm'].rolling(
        window=min(5, len(df_one) // 4),
        center=True, min_periods=1
    ).mean()
    ax2.plot(
        df_one['icu_hour'], df_one['total_smooth'],
        linestyle='--', linewidth=2,
        color='orange', label='Smoothed Trend'
    )

ax2.set_title('Total SOFA Score')
ax2.set_xlabel('ICU Hour')
ax2.set_ylabel('Normalized SOFA Score')
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
if sample_stay:
    plt.savefig(f"{FIG_DIR}/sofa_trajectory_{sample_stay}.png",
                dpi=150, bbox_inches='tight')
else:
    plt.savefig(f"{FIG_DIR}/average_sofa_trajectory.png",
                dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization complete (REAL DATA ONLY)")
print(f"✅ Figures saved to: {FIG_DIR}")