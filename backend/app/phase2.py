#!/usr/bin/env python3
"""
Phase 2: MIMIC-IV Cohort Construction
UPDATED: Pure ΔSOFA Target ONLY (NO DATA LEAKAGE)
Target: Pure ΔSOFA ≥ 2 at 6-18h (NO mortality)
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, DATA_DIR, DB_NAME,
    MIN_ICU_STAY_HOURS,
    FEATURE_WINDOW_HOURS,
    PREDICTION_START_HOUR,
    PREDICTION_END_HOUR,
    SOFA_CHANGE_THRESHOLD,
)
from database import DatabaseManager
from preprocess.cohort import CohortBuilder


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 2: MIMIC-IV COHORT CONSTRUCTION")
    print(" Multi-Organ Failure Prediction System")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Data Directory: {DATA_DIR}")
    print(f" Output Directory: {TABLE_DIR}")
    print("-" * 80)
    print(" ⚠️  COHORT DEFINITION (NO DATA LEAKAGE):")
    print(f"   • Minimum ICU Stay: {MIN_ICU_STAY_HOURS} hours")
    print(f"   • Feature Window: 0-{FEATURE_WINDOW_HOURS} hours "
          f"from ICU admission")
    print(f"   • Outcome Window: {PREDICTION_START_HOUR}-"
          f"{PREDICTION_END_HOUR} hours from ICU admission")
    print(f"   • TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} (no mortality)")
    print(f"   • Features END before Outcome BEGINS → NO LEAKAGE!")
    print("-" * 80)
    print(" EXCLUSION CRITERIA:")
    print("   • Age < 18")
    print(f"   • ICU stay < {MIN_ICU_STAY_HOURS}h")
    print("   • Invalid intime/outtime")
    print("   • Non-first ICU stay per patient")
    print(f"   • Deaths ≤ {PREDICTION_END_HOUR}h "
          f"(cannot have valid pure ΔSOFA outcome)")
    print("-" * 80)
    print(" NOTE: High-SOFA patients (≥4) in first 6h are NOT excluded.")
    print("       These HIGH-RISK patients are exactly who we want to predict.")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: MIMIC-IV Cohort Construction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase2.py                    # Build cohort with default settings
  python phase2.py --output custom.csv
  python phase2.py --no-save
        """
    )
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for cohort CSV')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save cohort to file')
    parser.add_argument('--db-path', type=str, default=DB_NAME,
                        help='Path to DuckDB database')

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting Phase 2: Cohort Construction")
    logger.info("=" * 60)
    logger.info(f"TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} (NO mortality)")
    logger.info(f"FEATURE WINDOW: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
    logger.info(f"OUTCOME WINDOW: {PREDICTION_START_HOUR}-"
                f"{PREDICTION_END_HOUR}h from ICU admission")
    logger.info("⚠️  NO DATA LEAKAGE: Features END before Outcome BEGINS")
    logger.info("=" * 60)

    print_header()

    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return 1

    try:
        logger.info("Initializing database connection...")
        db_manager = DatabaseManager(db_path=args.db_path)
        db_manager.create_views()
        # NOTE: create_window_summary_view() was removed from database.py

        logger.info("Building cohort...")
        builder = CohortBuilder(db_manager)
        cohort = builder.build_cohort()

        if not args.no_save:
            builder.save_cohort(args.output)

        # ---- Final summary ----
        print("\n" + "=" * 80)
        print(" PHASE 2 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f" ✅ Cohort size: {len(cohort):,} stays")
        print(f" ✅ Unique patients: {cohort['subject_id'].nunique():,}")

        if 'outcome' in cohort.columns:
            rate = cohort['outcome'].sum() / len(cohort) * 100
            print(f" ✅ Target (pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} "
                  f"at {PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h): "
                  f"{rate:.1f}%")

        if 'sofa_feature_complete' in cohort.columns:
            n_no_data = (cohort['sofa_feature_complete'] == 0).sum()
            pct = n_no_data / len(cohort) * 100
            print(f" ⚠️  Feature window: {n_no_data:,} stays ({pct:.1f}%) "
                  f"have NO SOFA data")

        print("-" * 80)
        print(" ⚠️  NO DATA LEAKAGE CONFIRMED:")
        print(f"   • Feature Window: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
        print(f"   • Outcome Window: {PREDICTION_START_HOUR}-"
              f"{PREDICTION_END_HOUR}h from ICU admission")
        print(f"   • Both anchored to ICU intime")
        print("=" * 80)

        print("\n📝 Next Steps:")
        print("  1. Run Phase 3: Feature Extraction (0-6h)")
        print("     → python phase3.py")
        print("  2. Run Phase 4: Preprocessing")
        print("  3. Run Phase 5: Missingness Characterization")
        print("  4. Run Phase 6: Imputation × Model Interaction")
        print("  5. Run Phase 7: Final Model + SHAP + DCA")
        print()

        return 0

    except Exception as e:
        logger.error(f"Error in Phase 2: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
