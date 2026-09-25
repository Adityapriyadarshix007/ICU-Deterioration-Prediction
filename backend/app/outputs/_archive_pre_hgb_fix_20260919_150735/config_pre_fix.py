"""
Configuration and constants for MIMIC-IV Multi-Organ Failure Prediction
UPDATED: Pure ΔSOFA Target (Organ Dysfunction Progression ONLY)
FIXED: Corrected wrong item IDs (verified against MIMIC-IV v3.1)
"""

import os
from pathlib import Path
from datetime import datetime

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = Path(__file__).parent.absolute()      # backend/app
PROJECT_ROOT = BASE_DIR.parent                   # backend

DATA_DIR = PROJECT_ROOT / "data"                 # backend/data
HOSP_DIR = DATA_DIR / "hosp"
ICU_DIR = DATA_DIR / "icu"

OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
PREPROCESS_DIR = BASE_DIR / "preprocess"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, PREPROCESS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# Database Versions
# ============================================================
MIMIC_VERSION = "3.1"
EICU_VERSION = "2.0"
DB_NAME = "mods.duckdb"

# ============================================================
# MIMIC-IV Table Names
# ============================================================
ICU_TABLES = [
    "icustays",
    "chartevents",
    "datetimeevents",
    "inputevents",
    "outputevents",
    "procedureevents",
    "d_items"
]

HOSP_TABLES = [
    "admissions",
    "patients",
    "labevents",
    "diagnoses_icd",
    "procedures_icd",
    "drgcodes",
    "transfers",
    "d_labitems",
    "d_icd_diagnoses",
    "d_icd_procedures"
]

# ============================================================
# COHORT DEFINITION WINDOWS
# ============================================================
MIN_ICU_STAY_HOURS = 24
FEATURE_WINDOW_HOURS = 6
PREDICTION_START_HOUR = 6
PREDICTION_END_HOUR = 18

# ============================================================
# SOFA Item IDs - Cardiovascular (FIXED)
# ============================================================
HR_ITEMID = 220045
SBP_ITEMIDS = [220050, 220179]   # arterial first, NIBP fallback
DBP_ITEMIDS = [220051, 220180]
MAP_ITEMIDS = [220052, 220181]

VASOPRESSORS = {
    'norepinephrine': 221906,   # unchanged
    'epinephrine': 221289,      # unchanged
    'dopamine': 221662,         # unchanged
    'dobutamine': 221653,       # unchanged
    'vasopressin': 222315,      # FIXED: was 221749 (phenylephrine)
    'phenylephrine': 221749,    # unchanged
}

# ============================================================
# SOFA Item IDs - Lab Values (FIXED)
# ============================================================
LAB_ITEM_IDS = {
    'creatinine': 50912,           # unchanged
    'bilirubin': 50885,            # unchanged
    'platelets': 51265,            # unchanged
    'bun': 51006,                  # FIXED: was 50971 (potassium)
    'lactate': 50813,              # FIXED: was 50983 (sodium)
    'wbc': 51301,                  # unchanged
    'hemoglobin': 50822,           # unchanged
    'hematocrit': 51221,           # unchanged
    'sodium': 50983,               # unchanged
    'potassium': 50971,            # unchanged
    'chloride': 50902,             # FIXED: was 50995 (thyroxine)
    'bicarbonate': 50882,          # unchanged
    'glucose': 50931,              # unchanged
    'albumin': 50862,              # unchanged
    'alt': 50861,                  # unchanged
    'ast': 50878,                  # unchanged
    'alkaline_phosphatase': 50863, # unchanged
    'inr': 51237,                  # unchanged
    'ptt': 51275,                  # FIXED: was 51273 (protein S)
    'pt': 51274,                   # unchanged
}

SOFA_LAB_ITEM_IDS = {
    k: LAB_ITEM_IDS[k]
    for k in ['creatinine', 'bilirubin', 'platelets']
}

# ============================================================
# SOFA Item IDs - Respiratory (FIXED)
# ============================================================
RESPIRATORY_ITEM_IDS = {
    'spo2': 220277,        # unchanged
    'pao2': 220224,        # FIXED: was 220235 (PaCO2)
    'fio2': 223835,        # FIXED: was 220211
    'rr': 220210,          # unchanged
}

SOFA_RESPIRATORY_ITEM_IDS = {
    k: RESPIRATORY_ITEM_IDS[k]
    for k in ['pao2', 'fio2']
}

# ============================================================
# SOFA Item IDs - Neurological (GCS - FIXED)
# ============================================================
GCS_ITEM_IDS = {
    'gcs_verbal': 223900,   # unchanged
    'gcs_motor': 223901,    # unchanged
    'gcs_eyes': 220739,     # FIXED: was 223902 (Speech)
    'gcs_total': None,    # FIXED: was 220739 but for wrong item (Eye Opening)
                            # NOTE: gcs_total is computed in code, not queried directly
}

# ============================================================
# Organ Systems
# ============================================================
ORGAN_SYSTEMS = [
    'cardiovascular',
    'respiratory',
    'renal',
    'liver',
    'coagulation',
    'neurological'
]

ORGAN_COLORS = {
    'cardiovascular': '#FF6B6B',
    'respiratory': '#4ECDC4',
    'renal': '#45B7D1',
    'liver': '#96CEB4',
    'coagulation': '#FFEAA7',
    'neurological': '#DDA0DD'
}

# ============================================================
# Feature Engineering Parameters
# ============================================================
MAX_MISSING_RATE = 0.60
OUTLIER_LOW = 0.005
OUTLIER_HIGH = 0.995

AGGREGATION_METHODS = [
    'min', 'max', 'mean', 'median', 'std', 'last', 'count',
    'slope', 'delta', 'range', 'time_since_last_measure',
]

# ============================================================
# OUTCOME DEFINITION (Pure ΔSOFA ONLY)
# ============================================================
SOFA_CHANGE_THRESHOLD = 2

OUTCOME_DEFINITION = {
    'sofa_increase': SOFA_CHANGE_THRESHOLD,
    'includes_death': False,
    'time_window_start': PREDICTION_START_HOUR,
    'time_window_end': PREDICTION_END_HOUR,
    'target_type': 'organ_dysfunction_progression',
    'target_description': f'ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} points at '
                          f'{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR} hours'
}

# ============================================================
# IMPUTATION STRATEGIES (6 strategies)
# ============================================================
TIME_SERIES_IMPUTATION_STRATEGIES = [
    'none',      # baseline: aggregate over observed values only
    'locf',      # forward-fill within 0-6h window
    'linear',    # linear interpolation within 0-6h window
]

TABULAR_IMPUTATION_STRATEGIES = [
    'zero',
    'median',
    'knn',
    'ice',
]

TIME_SERIES_IMPUTATION_PARAMS = {
    'none':   {},
    'locf':   {'window_end': FEATURE_WINDOW_HOURS},
    'linear': {'window_end': FEATURE_WINDOW_HOURS},
}

TABULAR_IMPUTATION_PARAMS = {
    'zero':   {'value': 0},
    'median': {'strategy': 'median'},
    'knn':    {'n_neighbors': 5, 'weights': 'uniform',
               'metric': 'nan_euclidean'},
    'ice':    {'max_iter': 10, 'random_state': 42},
}

# ============================================================
# Model Training Parameters
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.15
N_FOLDS = 5

EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_METRIC = 'PRAUC'   # aligned with recall priority

CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.01,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',           # CatBoost supports PRAUC
    'custom_metric': ['PRAUC', 'Recall', 'Precision', 'F1', 'AUC'],
    'random_seed': RANDOM_STATE,
    'od_type': 'Iter',
    'od_wait': EARLY_STOPPING_ROUNDS,
    'verbose': False,
}

# Constructor: no early_stopping_rounds, no eval_metric='PRAUC' (XGB doesn't have it)
# Pass eval_metric in .fit() via custom_metric
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',           # XGBoost uses 'aucpr' for AUPRC
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
}

LGBM_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'num_leaves': 31,
    'objective': 'binary',
    'metric': 'average_precision',    # LightGBM uses this for AUPRC
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
}

MODELS_TO_TRAIN = [
    'catboost',
    'xgboost',
    'lightgbm'
]

# ============================================================
# Performance Metrics
# ============================================================
PRIMARY_METRIC = 'recall'
SECONDARY_METRICS = [
    'auroc',
    'auprc',
    'precision',
    'f1_score',
    'accuracy',
    'brier_score',
    'specificity',
    'npv'
]

# ============================================================
# SHAP and DCA Parameters
# ============================================================
SHAP_SAMPLE_SIZE = 1000
DCA_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# ============================================================
# Logging
# ============================================================
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ============================================================
# External Validation - eICU-CRD Mapping
# ============================================================
EICU_MAPPING = {
    'heart_rate': 'heartrate',
    'respiratory_rate': 'respiratoryrate',
    'spo2': 'sao2',
    'map': 'systemicmean',
    'sbp': 'systemicsystolic',
    'dbp': 'systemicdiastolic',
    'temperature': 'temperature',
    'creatinine': 'creatinine',
    'bilirubin': 'bilirubin',
    'platelets': 'platelets',
    'bun': 'bun',
    'lactate': 'lactate',
    'wbc': 'wbc',
    'hemoglobin': 'hemoglobin',
    'hematocrit': 'hematocrit',
    'sodium': 'sodium',
    'potassium': 'potassium',
    'chloride': 'chloride',
    'bicarbonate': 'bicarbonate',
    'glucose': 'glucose',
    'albumin': 'albumin',
    'alt': 'alt',
    'ast': 'ast',
    'inr': 'inr',
    'gcs_total': 'gcs',
    'norepinephrine': 'norepinephrine',
    'epinephrine': 'epinephrine',
    'dopamine': 'dopamine',
    'dobutamine': 'dobutamine',
}

REQUIRED_FEATURES_FOR_EXTERNAL = [
    'heart_rate', 'respiratory_rate', 'map', 'temperature',
    'creatinine', 'bilirubin', 'platelets', 'bun', 'lactate',
    'gcs_total', 'norepinephrine', 'epinephrine', 'dopamine'
]

# ============================================================
# Visualization Settings
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE = 'viridis'
