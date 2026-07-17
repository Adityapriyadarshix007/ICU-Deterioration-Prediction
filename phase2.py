import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================
# Setup
# ===============================
os.makedirs("figures", exist_ok=True)
con = duckdb.connect(database=":memory:")
BASE = "data"

# ===============================
# Load tables
# ===============================
con.execute(f"""
CREATE VIEW icustays AS
SELECT stay_id, hadm_id, intime, outtime
FROM read_csv_auto('{BASE}/icu/icustays.csv.gz');
""")

con.execute(f"""
CREATE VIEW labevents AS
SELECT hadm_id, itemid, charttime, valuenum
FROM read_csv_auto('{BASE}/hosp/labevents.csv.gz');
""")

con.execute(f"""
CREATE VIEW chartevents AS
SELECT stay_id, itemid, charttime, valuenum
FROM read_csv_auto('{BASE}/icu/chartevents.csv.gz');
""")

# ===============================
# SOFA item mapping
# ===============================
con.execute("""
CREATE TABLE sofa_item_map (
  organ TEXT,
  variable TEXT,
  itemid INTEGER,
  source TEXT
);
""")

con.execute("""
INSERT INTO sofa_item_map VALUES
('renal', 'creatinine', 50912, 'labevents'),
('liver', 'bilirubin', 50885, 'labevents'),
('coagulation', 'platelets', 51265, 'labevents'),
('cardiovascular', 'MAP', 220052, 'chartevents'),
('cardiovascular', 'MAP', 225312, 'chartevents');
""")

# ===============================
# LABS → hourly average
# ===============================
con.execute("""
CREATE TABLE sofa_labs_hourly AS
SELECT
    i.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (le.charttime - i.intime))/3600) AS icu_hour,
    m.organ,
    AVG(le.valuenum) AS avg_value
FROM labevents le
JOIN icustays i ON le.hadm_id = i.hadm_id
JOIN sofa_item_map m ON le.itemid = m.itemid
WHERE m.source='labevents'
  AND le.valuenum IS NOT NULL
  AND le.charttime BETWEEN i.intime AND i.outtime
GROUP BY i.stay_id, icu_hour, m.organ;
""")

# ===============================
# CARDIO → worst MAP per hour
# ===============================
con.execute("""
CREATE TABLE sofa_cardio_hourly AS
SELECT
    i.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (ce.charttime - i.intime))/3600) AS icu_hour,
    'cardiovascular' AS organ,
    MIN(ce.valuenum) AS avg_value
FROM chartevents ce
JOIN icustays i ON ce.stay_id = i.stay_id
WHERE ce.itemid IN (220052, 225312)
  AND ce.valuenum BETWEEN 20 AND 200
  AND ce.charttime BETWEEN i.intime AND i.outtime
GROUP BY i.stay_id, icu_hour;
""")

# ===============================
# Combine all organs
# ===============================
con.execute("""
CREATE TABLE sofa_hourly AS
SELECT * FROM sofa_labs_hourly
UNION ALL
SELECT * FROM sofa_cardio_hourly;
""")

# ===============================
# Convert to pandas
# ===============================
df = con.execute("""
SELECT stay_id, icu_hour, organ, avg_value
FROM sofa_hourly
ORDER BY stay_id, icu_hour;
""").fetchdf()

print("\nRows per organ:")
print(df.groupby("organ").size())

# ===============================
# Pick a stay with all 4 organs
# ===============================
sample_stay = (
    df.groupby("stay_id")["organ"]
      .nunique()
      .loc[lambda x: x == 4]
      .index[0]
)

df_one = df[df["stay_id"] == sample_stay]

# ===============================
# Plot + SAVE + DISPLAY
# ===============================
sns.set_style("whitegrid")

g = sns.FacetGrid(
    df_one,
    col="organ",
    sharey=False,
    height=3,
    aspect=1.8
)

g.map_dataframe(
    sns.lineplot,
    x="icu_hour",
    y="avg_value",
    marker="o"
)

g.set_axis_labels("ICU Hour", "Value")
g.fig.suptitle(
    f"Organ Trajectories for ICU Stay {sample_stay}",
    y=1.05
)

plt.tight_layout()

# ✅ SAVE TO VS CODE FOLDER
output_path = f"figures/organ_trajectories_stay_{sample_stay}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

# ✅ DISPLAY
plt.show()

print(f"\n✅ Figure saved at: {output_path}")
print("✅ Phase 2 completed successfully.")