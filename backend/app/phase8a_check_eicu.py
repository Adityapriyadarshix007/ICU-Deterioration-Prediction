#!/usr/bin/env python3
"""
Phase 8a Step 2: Verify eICU schema before running extraction.

Prints:
  - Row counts per table
  - Column names for key tables
  - Distinct values for categorical fields we filter on
  - Sample lab names, drug names, GCS labels

Usage:
    python3 phase8a_check_eicu.py
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import EICU_DB_NAME, TABLE_DIR


def _df(conn, sql):
    return conn.execute(sql).fetchdf()


def main():
    print("=" * 78)
    print("PHASE 8a — STEP 2: VERIFY eICU SCHEMA")
    print("=" * 78)

    db_path = Path(EICU_DB_NAME)
    if not db_path.exists():
        print(f"❌ {db_path} not found. Run phase8a_setup_eicu_db.py first.")
        return 1

    conn = duckdb.connect(str(db_path), read_only=True)

    # ---------- 1. patient ----------
    print("\n" + "=" * 78)
    print("PATIENT TABLE")
    print("=" * 78)

    n = conn.execute("SELECT COUNT(*) FROM patient").fetchone()[0]
    print(f"Rows: {n:,}")

    print("\nUnique patients (uniquepid):")
    print(conn.execute(
        "SELECT COUNT(DISTINCT uniquepid) FROM patient"
    ).fetchone()[0])

    print("\n=== unitadmitsource ===")
    print(_df(conn, """
        SELECT unitadmitsource, COUNT(*) AS n
        FROM patient
        GROUP BY unitadmitsource
        ORDER BY n DESC
    """).to_string(index=False))

    print("\n=== unitdischargelocation ===")
    print(_df(conn, """
        SELECT unitdischargelocation, COUNT(*) AS n
        FROM patient
        GROUP BY unitdischargelocation
        ORDER BY n DESC
    """).to_string(index=False))

    print("\n=== unitdischargestatus ===")
    print(_df(conn, """
        SELECT unitdischargestatus, COUNT(*) AS n
        FROM patient
        GROUP BY unitdischargestatus
        ORDER BY n DESC
    """).to_string(index=False))

    print("\n=== age (top 15 values) ===")
    print(_df(conn, """
        SELECT age, COUNT(*) AS n
        FROM patient
        WHERE age IS NOT NULL
        GROUP BY age
        ORDER BY n DESC
        LIMIT 15
    """).to_string(index=False))

    # ---------- 2. vitalPeriodic ----------
    print("\n" + "=" * 78)
    print("VITALPERIODIC TABLE")
    print("=" * 78)
    print(_df(conn, "DESCRIBE vitalPeriodic").to_string(index=False))

    # ---------- 3. lab names ----------
    print("\n" + "=" * 78)
    print("LAB TABLE — distinct labname values (top 60)")
    print("=" * 78)
    print(_df(conn, """
        SELECT labname, COUNT(*) AS n
        FROM lab
        GROUP BY labname
        ORDER BY n DESC
        LIMIT 60
    """).to_string(index=False))

    # ---------- 4. drug names ----------
    print("\n" + "=" * 78)
    print("INFUSIONDRUG TABLE — vasopressor-related drug names")
    print("=" * 78)
    print(_df(conn, """
        SELECT drugname, COUNT(*) AS n
        FROM infusionDrug
        WHERE LOWER(drugname) LIKE '%norepinephrine%'
           OR LOWER(drugname) LIKE '%epinephrine%'
           OR LOWER(drugname) LIKE '%dopamine%'
           OR LOWER(drugname) LIKE '%dobutamine%'
           OR LOWER(drugname) LIKE '%levophed%'
           OR LOWER(drugname) LIKE '%adrenaline%'
        GROUP BY drugname
        ORDER BY n DESC
    """).to_string(index=False))

    print("\n=== infusionDrug schema ===")
    print(_df(conn, "DESCRIBE infusionDrug").to_string(index=False))

    # ---------- 5. GCS in nurseCharting ----------
    print("\n" + "=" * 78)
    print("NURSECHARTING TABLE — GCS labels")
    print("=" * 78)
    print(_df(conn, """
        SELECT nursingchartcelltypevalname, COUNT(*) AS n
        FROM nurseCharting
        WHERE LOWER(nursingchartcelltypevalname) LIKE '%gcs%'
           OR LOWER(nursingchartcelltypevalname) LIKE '%glasgow%'
        GROUP BY nursingchartcelltypevalname
        ORDER BY n DESC
        LIMIT 30
    """).to_string(index=False))

    # ---------- 6. FiO2 in respiratoryCharting ----------
    print("\n" + "=" * 78)
    print("RESPIRATORYCHARTING TABLE — FiO2 labels")
    print("=" * 78)
    print(_df(conn, """
        SELECT respchartvaluelabel, COUNT(*) AS n
        FROM respiratoryCharting
        WHERE LOWER(respchartvaluelabel) LIKE '%fio2%'
           OR LOWER(respchartvaluelabel) LIKE '%o2 conc%'
        GROUP BY respchartvaluelabel
        ORDER BY n DESC
        LIMIT 30
    """).to_string(index=False))

    # ---------- 7. Save full audit ----------
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "phase8a_eicu_schema_audit.txt"
    with open(out, "w") as f:
        f.write("eICU schema audit\n")
        f.write("=" * 60 + "\n")
        for t in ["patient", "vitalPeriodic", "vitalAperiodic",
                  "lab", "infusionDrug", "respiratoryCharting",
                  "nurseCharting"]:
            try:
                schema = _df(conn, f"DESCRIBE {t}")
                f.write(f"\n{t}\n{'-'*60}\n")
                f.write(schema.to_string(index=False))
                f.write("\n")
            except Exception as e:
                f.write(f"\n{t}: ERROR {e}\n")
    print(f"\n✅ Full audit saved: {out}")

    conn.close()
    print("\nNext: python3 phase8a_eicu_cohort.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
