"""
Phase 1: Data Access and Environment Setup
MIMIC-IV Multi-Organ Failure Prediction System

UPDATED: 0-6h Features → 6-18h Outcome (NO DATA LEAKAGE)
TARGET: Pure ΔSOFA ≥ 2 (NO mortality)

FIXES APPLIED:
- Wire --skip-views and --data-dir flags into setup_database()
- Random sample (not head(100)) for validation using RANDOM_STATE
- Add package versions to environment_info.json
- Guard against --memory + --skip-views combination
- Move `import traceback` to top of file
- validate_directories returns True after creating missing dirs
- --validate-only simplified: no DB instantiation needed
"""

import os
import sys
import argparse
import logging
import json
import platform
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR,
    DATA_DIR, DB_NAME, LOG_LEVEL, LOG_FILE,
    MIN_ICU_STAY_HOURS,
    FEATURE_WINDOW_HOURS,
    PREDICTION_START_HOUR,
    PREDICTION_END_HOUR,
    SOFA_CHANGE_THRESHOLD,
    TIME_SERIES_IMPUTATION_STRATEGIES,
    TABULAR_IMPUTATION_STRATEGIES,
    MODELS_TO_TRAIN,
    RANDOM_STATE,
)
try:
    from app.database import DatabaseManager
except ImportError:
    from database import DatabaseManager
try:
    from app.data_loader import DataLoader
except ImportError:
    from data_loader import DataLoader


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 1: DATA ACCESS AND ENVIRONMENT SETUP")
    print(" MIMIC-IV Multi-Organ Failure Prediction System")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Working Directory: {os.getcwd()}")
    print(f" Data Directory: {DATA_DIR}")
    print(f" Output Directory: {OUTPUT_DIR}")
    print("-" * 80)
    print(" ⚠️  WINDOW CONFIGURATION (NO DATA LEAKAGE):")
    print(f"   • Minimum ICU Stay: {MIN_ICU_STAY_HOURS} hours")
    print(f"   • Feature Window: 0-{FEATURE_WINDOW_HOURS} hours from ICU admission")
    print(f"   • Outcome Window: {PREDICTION_START_HOUR}-{PREDICTION_END_HOUR} "
          f"hours from ICU admission")
    print(f"   • TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} (no mortality)")
    print(f"   • Features END before Outcome BEGINS → NO LEAKAGE!")
    print("=" * 80)
    print()


# ============================================================
# DIRECTORY VALIDATION
# ============================================================

def validate_directories():
    print("📁 Validating directory structure...")

    required_dirs = [
        DATA_DIR,
        DATA_DIR / "hosp",
        DATA_DIR / "icu",
        OUTPUT_DIR,
        FIG_DIR,
        TABLE_DIR,
        MODEL_DIR,
    ]

    all_exist = True
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"  ❌ Missing directory: {dir_path}")
            all_exist = False
        else:
            print(f"  ✅ Found directory: {dir_path}")

    if not all_exist:
        print("\n⚠️ Some directories are missing. Creating them...")
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        print("✅ All directories created")
        return True  # FIXED: now correctly returns True after creation

    return all_exist


# ============================================================
# DATABASE SETUP
# ============================================================

def setup_database(use_memory: bool = False,
                   data_dir: str = None,
                   skip_views: bool = False):
    """Set up DuckDB connection and views."""
    print("\n🗄️ Setting up database...")

    db_manager = DatabaseManager(db_path=DB_NAME, memory=use_memory)

    if not skip_views:
        db_manager.create_views(data_dir=data_dir)
    else:
        print("  ℹ️  Skipping view creation (--skip-views)")

    print("\n📊 Table counts:")
    counts = db_manager.get_table_counts()
    for table, count in counts.items():
        if count > 0:
            print(f"  ✅ {table}: {count:,} rows")
        else:
            print(f"  ❌ {table}: {count} rows (empty or missing)")

    return db_manager


# ============================================================
# DATA VALIDATION
# ============================================================

def load_and_validate_data(db_manager):
    print("\n📊 Loading and validating data...")

    data_loader = DataLoader(db_manager)

    print("\n🔍 Validating data files:")
    data_loader.validate_data_presence()

    print("\n📈 Checking data quality:")

    try:
        stays = data_loader.get_icu_stays()
        print(f"\n  ICU Stays: {len(stays):,} records")
        print(f"    - Unique patients: {stays['subject_id'].nunique():,}")
        print(f"    - Unique admissions: {stays['hadm_id'].nunique():,}")
        print(f"    - Age range: {stays['anchor_age'].min():.0f} - "
              f"{stays['anchor_age'].max():.0f} years")
        print(f"    - Gender: Male={stays[stays['gender'] == 'M'].shape[0]:,}, "
              f"Female={stays[stays['gender'] == 'F'].shape[0]:,}")

        stay_hours = stays['los'] * 24
        print(f"    - ICU stay range: {stay_hours.min():.0f} - "
              f"{stay_hours.max():.0f} hours")
        print(f"    - Patients with ≥{MIN_ICU_STAY_HOURS}h: "
              f"{(stay_hours >= MIN_ICU_STAY_HOURS).sum():,}")

        # FIXED: random sample, not head(100)
        sample_size = min(100, len(stays))
        sample_stays = stays.sample(n=sample_size, random_state=RANDOM_STATE)
        sample_stay_ids = sample_stays['stay_id'].tolist()
        sample_hadm_ids = sample_stays['hadm_id'].tolist()

        print(f"\n  Using random sample of {sample_size} stays for loader tests")

        # Vital signs
        vitals = data_loader.get_vital_signs(sample_stay_ids)
        print(f"\n  Vital signs (sample): {len(vitals):,} records")
        if len(vitals) > 0:
            print(f"    - Unique stays with vitals: {vitals['stay_id'].nunique():,}")

        # Labs
        labs = data_loader.get_all_labs(sample_hadm_ids)
        print(f"\n  Labs (sample): {len(labs):,} records")
        if len(labs) > 0:
            print(f"    - Unique admissions with labs: "
                  f"{labs['hadm_id'].nunique():,}")

        # Vasopressors
        vaso = data_loader.get_vasopressors(sample_stay_ids)
        print(f"\n  Vasopressor events (sample): {len(vaso):,} records")
        if len(vaso) > 0:
            print(f"    - Unique stays with vasopressors: "
                  f"{vaso['stay_id'].nunique():,}")

        print("\n  Testing FEATURE WINDOW methods (0-6h from ICU admission)...")
        feature_vitals = data_loader.get_feature_window_chart_events(sample_stay_ids)
        print(f"    - Feature vitals: {len(feature_vitals)} records")

        feature_labs = data_loader.get_feature_window_labs(sample_stay_ids)
        print(f"    - Feature labs: {len(feature_labs)} records")

        feature_vaso = data_loader.get_feature_window_vasopressors(sample_stay_ids)
        print(f"    - Feature vasopressors: {len(feature_vaso)} records")

        print("\n  Testing OUTCOME WINDOW methods (6-18h from ICU admission)...")
        outcome_vitals = data_loader.get_outcome_window_chart_events(sample_stay_ids)
        print(f"    - Outcome vitals: {len(outcome_vitals)} records")

        outcome_labs = data_loader.get_outcome_window_labs(sample_stay_ids)
        print(f"    - Outcome labs: {len(outcome_labs)} records")

        outcome_vaso = data_loader.get_outcome_window_vasopressors(sample_stay_ids)
        print(f"    - Outcome vasopressors: {len(outcome_vaso)} records")

        print("\n  Validating window data availability...")
        data_loader.validate_window_data(sample_stay_ids)

        return data_loader

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        traceback.print_exc()
        return None


# ============================================================
# ENVIRONMENT INFO
# ============================================================

def _get_package_versions() -> dict:
    """Collect versions of key packages for reproducibility."""
    versions = {}

    try:
        import duckdb
        versions['duckdb'] = duckdb.__version__
    except Exception:
        versions['duckdb'] = None

    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except Exception:
        versions['pandas'] = None

    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except Exception:
        versions['numpy'] = None

    try:
        import sklearn
        versions['scikit_learn'] = sklearn.__version__
    except Exception:
        versions['scikit_learn'] = None

    try:
        import catboost
        versions['catboost'] = catboost.__version__
    except Exception:
        versions['catboost'] = None

    try:
        import xgboost
        versions['xgboost'] = xgboost.__version__
    except Exception:
        versions['xgboost'] = None

    try:
        import lightgbm
        versions['lightgbm'] = lightgbm.__version__
    except Exception:
        versions['lightgbm'] = None

    return versions


def save_environment_info():
    info = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'data_directory': str(DATA_DIR),
        'output_directory': str(OUTPUT_DIR),
        'mimic_version': '3.1',
        'eicu_version': '2.0',
        'target': 'Pure ΔSOFA ≥ 2 (no mortality)',
        'window_configuration': {
            'min_icu_stay_hours': MIN_ICU_STAY_HOURS,
            'feature_window_hours': FEATURE_WINDOW_HOURS,
            'feature_window_description':
                f'0-{FEATURE_WINDOW_HOURS}h from ICU admission (for prediction)',
            'prediction_start_hour': PREDICTION_START_HOUR,
            'prediction_end_hour': PREDICTION_END_HOUR,
            'outcome_window_description':
                f'{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h from ICU '
                f'admission (what we predict)',
            'data_leakage_prevention': 'Features END before Outcome BEGINS',
            'reference_point': 'ICU intime (both windows)',
        },
        'imputation_strategies': {
            'time_series': TIME_SERIES_IMPUTATION_STRATEGIES,
            'tabular': TABULAR_IMPUTATION_STRATEGIES,
        },
        'models': MODELS_TO_TRAIN,
        'package_versions': _get_package_versions(),
    }

    info_file = OUTPUT_DIR / 'environment_info.json'
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n💾 Environment info saved to: {info_file}")
    return info


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: MIMIC-IV Data Setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Full setup
  python main.py --memory                 # In-memory database
  python main.py --validate-only          # Only validate data
  python main.py --data-dir /custom/path  # Custom data directory
  python main.py --skip-views             # Use existing views
        """
    )
    parser.add_argument('--memory', action='store_true',
                        help='Use in-memory database (faster but requires more RAM)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate data file presence, then exit')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Path to data directory')
    parser.add_argument('--skip-views', action='store_true',
                        help='Skip creating database views (use existing)')

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting Phase 1: Data Access and Environment Setup")

    print_header()
    validate_directories()

    # Guard: --memory + --skip-views would leave an empty DB with no views
    if args.memory and args.skip_views:
        print("❌ --skip-views is incompatible with --memory "
              "(in-memory DB starts empty)")
        return 1

    # Validate-only: just file presence, no DB needed
    if args.validate_only:
        print("\n🔍 Validation mode: checking data file presence only")
        # Create a lightweight loader purely for file checks
        db_manager = DatabaseManager(db_path=':memory:', memory=True)
        data_loader = DataLoader(db_manager)
        results = data_loader.validate_data_presence()
        if all(results.values()):
            print("\n✅ All required data files present")
            return 0
        else:
            print("\n❌ Some required data files are missing")
            return 1

    # Full setup
    db_manager = setup_database(
        use_memory=args.memory,
        data_dir=args.data_dir,
        skip_views=args.skip_views,
    )
    data_loader = load_and_validate_data(db_manager)

    if data_loader is None:
        logger.error("Failed to load data. Exiting.")
        return 1

    save_environment_info()

    print("\n" + "=" * 80)
    print(" PHASE 1 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f" ✅ Database: {DB_NAME}")
    print(f" ✅ Data loaded and validated")
    print(f" ✅ Output directory: {OUTPUT_DIR}")
    print(f" 📄 Log file: {LOG_FILE}")
    print("-" * 80)
    print(" ⚠️  WINDOW CONFIGURATION (NO DATA LEAKAGE):")
    print(f"   • Minimum ICU Stay: {MIN_ICU_STAY_HOURS} hours")
    print(f"   • Feature Window: 0-{FEATURE_WINDOW_HOURS} hours "
          f"from ICU admission")
    print(f"   • Outcome Window: {PREDICTION_START_HOUR}-"
          f"{PREDICTION_END_HOUR} hours from ICU admission")
    print(f"   • TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} (no mortality)")
    print(f"   • Features END before Outcome BEGINS → NO LEAKAGE!")
    print("=" * 80)

    print("\n📝 Next Steps:")
    print("  1. Run Phase 2: Cohort Construction")
    print("     → python phase2.py")
    print("  2. Run Phase 3: Feature Extraction (0-6h)")
    print("  3. Run Phase 4: Preprocessing")
    print("  4. Run Phase 5: Missingness Characterization")
    print("  5. Run Phase 6: Imputation × Model Interaction")
    print("  6. Run Phase 7: Final Model + SHAP + DCA")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
