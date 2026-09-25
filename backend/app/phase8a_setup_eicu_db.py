#!/usr/bin/env python3
"""
Phase 8a Step 1: Create DuckDB views over eICU-CRD CSVs.

Creates data/eicu.duckdb with lazy views over the gzipped CSVs.
No data is copied. Views are read on demand.

Usage:
    python3 phase8a_setup_eicu_db.py
"""

import sys
import duckdb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import EICU_DIR, EICU_DB_NAME, EICU_TABLES


def main():
    print("=" * 78)
    print("PHASE 8a — STEP 1: SETUP eICU DUCKDB")
    print("=" * 78)

    if not EICU_DIR.exists():
        print(f"❌ eICU directory not found: {EICU_DIR}")
        print(f"   Expected gzipped CSVs at: {EICU_DIR}/*.csv.gz")
        return 1

    eicu_db_path = Path(EICU_DB_NAME)
    eicu_db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 eICU data directory: {EICU_DIR}")
    print(f"📦 Target DB file:      {eicu_db_path}")

    conn = duckdb.connect(str(eicu_db_path))
    conn.execute("PRAGMA threads=4")
    conn.execute("PRAGMA memory_limit='8GB'")

    created = 0
    missing = []

    for table in EICU_TABLES:
        csv_path = EICU_DIR / f"{table}.csv.gz"
        if not csv_path.exists():
            print(f"  ⚠️  Missing: {csv_path.name}")
            missing.append(table)
            continue

        try:
            conn.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT * FROM read_csv_auto(
                    '{csv_path}',
                    header=True,
                    compression='gzip',
                    sample_size=-1
                )
            """)
            n_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  ✅ {table:25s}  {n_rows:>12,} rows")
            created += 1
        except Exception as e:
            print(f"  ❌ {table}: {e}")
            missing.append(table)

    print()
    print(f"✅ Created {created}/{len(EICU_TABLES)} views")
    if missing:
        print(f"⚠️  Missing/failed: {missing}")

    conn.close()
    print(f"\n✅ Saved: {eicu_db_path}")
    print("\nNext: python3 phase8a_check_eicu.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
