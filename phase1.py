import duckdb
import pandas as pd

# --------------------------------------------------
# 1. Connect to DuckDB (persistent database)
# --------------------------------------------------
con = duckdb.connect("mods.duckdb")

# --------------------------------------------------
# 2. Core tables (HOSP + ICU)
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW patients AS
SELECT *
FROM read_csv_auto('data/hosp/patients.csv.gz');
""")

con.execute("""
CREATE OR REPLACE VIEW admissions AS
SELECT *
FROM read_csv_auto('data/hosp/admissions.csv.gz');
""")

con.execute("""
CREATE OR REPLACE VIEW icustays AS
SELECT *
FROM read_csv_auto('data/icu/icustays.csv.gz');
""")

# --------------------------------------------------
# 3. ICU Cohort: stays >= 24 hours
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW icu_cohort AS
SELECT *
FROM icustays
WHERE (outtime - intime) >= INTERVAL '24 hours';
""")

# --------------------------------------------------
# 4. ICU length of stay summary
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW icu_los_summary AS
SELECT
    MIN(EXTRACT(EPOCH FROM (outtime - intime)) / 3600) AS min_hours,
    MAX(EXTRACT(EPOCH FROM (outtime - intime)) / 3600) AS max_hours,
    AVG(EXTRACT(EPOCH FROM (outtime - intime)) / 3600) AS avg_hours
FROM icu_cohort;
""")

# --------------------------------------------------
# 5. ICU chart events (Heart Rate)
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW d_items AS
SELECT *
FROM read_csv_auto('data/icu/d_items.csv.gz');
""")

con.execute("""
CREATE OR REPLACE VIEW chartevents AS
SELECT *
FROM read_csv_auto('data/icu/chartevents.csv.gz');
""")

# Heart Rate (itemid = 220045)
con.execute("""
CREATE OR REPLACE VIEW hr_events AS
SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS heart_rate
FROM chartevents ce
JOIN icu_cohort ic
  ON ce.stay_id = ic.stay_id
WHERE ce.itemid = 220045
  AND ce.valuenum IS NOT NULL;
""")

con.execute("""
CREATE OR REPLACE VIEW hr_with_hour AS
SELECT
    h.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (h.charttime - ic.intime)) / 3600) AS icu_hour,
    h.heart_rate
FROM hr_events h
JOIN icu_cohort ic
  ON h.stay_id = ic.stay_id
WHERE h.charttime >= ic.intime;
""")

con.execute("""
CREATE OR REPLACE VIEW hr_hourly AS
SELECT
    stay_id,
    icu_hour,
    AVG(heart_rate) AS mean_hr
FROM hr_with_hour
GROUP BY stay_id, icu_hour;
""")

# --------------------------------------------------
# 6. Laboratory events
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW labevents AS
SELECT *
FROM read_csv_auto('data/hosp/labevents.csv.gz');
""")

con.execute("""
CREATE OR REPLACE VIEW d_labitems AS
SELECT *
FROM read_csv_auto('data/hosp/d_labitems.csv.gz');
""")

# Relevant labs for SOFA
# Renal: Creatinine
# Liver: Bilirubin
# Coagulation: Platelets
con.execute("""
CREATE OR REPLACE VIEW labs_filtered AS
SELECT
    le.subject_id,
    ic.stay_id,
    le.itemid,
    le.charttime,
    le.valuenum
FROM labevents le
JOIN icu_cohort ic
  ON le.hadm_id = ic.hadm_id
WHERE le.valuenum IS NOT NULL
  AND le.itemid IN (
      50912, 51067, 51081, 51082,   -- creatinine
      50885, 51464,               -- bilirubin
      51265                       -- platelets
  );
""")

# --------------------------------------------------
# 7. Map labs to organ systems
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW labs_mapped AS
SELECT *,
    CASE
        WHEN itemid IN (50912, 51067, 51081, 51082) THEN 'renal'
        WHEN itemid IN (50885, 51464) THEN 'liver'
        WHEN itemid = 51265 THEN 'coagulation'
    END AS organ
FROM labs_filtered;
""")

# --------------------------------------------------
# 8. Convert lab times to ICU hour
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW labs_with_hour AS
SELECT
    lm.subject_id,
    lm.stay_id,
    lm.organ,
    lm.valuenum AS lab_value,
    FLOOR(EXTRACT(EPOCH FROM (lm.charttime - ic.intime)) / 3600) AS icu_hour
FROM labs_mapped lm
JOIN icu_cohort ic
  ON lm.stay_id = ic.stay_id
WHERE lm.charttime >= ic.intime;
""")

# --------------------------------------------------
# 9. SOFA scoring (renal, liver, coagulation)
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW sofa_scores AS
SELECT
    stay_id,
    icu_hour,
    organ,
    CASE
        WHEN organ = 'renal' THEN
            CASE
                WHEN lab_value < 1.2 THEN 0
                WHEN lab_value BETWEEN 1.2 AND 1.9 THEN 1
                WHEN lab_value BETWEEN 2.0 AND 3.4 THEN 2
                WHEN lab_value BETWEEN 3.5 AND 4.9 THEN 3
                ELSE 4
            END
        WHEN organ = 'liver' THEN
            CASE
                WHEN lab_value < 1.2 THEN 0
                WHEN lab_value BETWEEN 1.2 AND 1.9 THEN 1
                WHEN lab_value BETWEEN 2.0 AND 5.9 THEN 2
                WHEN lab_value BETWEEN 6.0 AND 11.9 THEN 3
                ELSE 4
            END
        WHEN organ = 'coagulation' THEN
            CASE
                WHEN lab_value >= 150 THEN 0
                WHEN lab_value BETWEEN 100 AND 149 THEN 1
                WHEN lab_value BETWEEN 50 AND 99 THEN 2
                WHEN lab_value BETWEEN 20 AND 49 THEN 3
                ELSE 4
            END
    END AS sofa_score
FROM labs_with_hour;
""")

# --------------------------------------------------
# 10. Hourly SOFA aggregation
# --------------------------------------------------

con.execute("""
CREATE OR REPLACE VIEW sofa_hourly AS
SELECT
    stay_id,
    icu_hour,
    MAX(CASE WHEN organ='renal' THEN sofa_score END) AS sofa_renal,
    MAX(CASE WHEN organ='liver' THEN sofa_score END) AS sofa_liver,
    MAX(CASE WHEN organ='coagulation' THEN sofa_score END) AS sofa_coag
FROM sofa_scores
GROUP BY stay_id, icu_hour
ORDER BY stay_id, icu_hour;
""")

# --------------------------------------------------
# 11. Final sanity check
# --------------------------------------------------

print("ICU stays:", con.execute("SELECT COUNT(*) FROM icu_cohort").fetchone()[0])
print("HR rows:", con.execute("SELECT COUNT(*) FROM hr_hourly").fetchone()[0])
print("SOFA rows:", con.execute("SELECT COUNT(*) FROM sofa_hourly").fetchone()[0])

print("\nSample SOFA rows:")
print(con.execute("SELECT * FROM sofa_hourly LIMIT 5").fetchdf())
# Check number of ICU stays
icu_count = con.execute("SELECT COUNT(*) FROM icu_cohort").fetchone()[0]
print("Number of ICU stays:", icu_count)

# Check number of heart rate events
hr_count = con.execute("SELECT COUNT(*) FROM hr_hourly").fetchone()[0]
print("Number of HR rows:", hr_count)

# Check number of SOFA rows
sofa_count = con.execute("SELECT COUNT(*) FROM sofa_hourly").fetchone()[0]
print("Number of SOFA rows:", sofa_count)

# Show a sample of SOFA scores
sofa_sample = con.execute("SELECT * FROM sofa_hourly LIMIT 5").fetchdf()
print("\nSample SOFA rows:")
print(sofa_sample)
