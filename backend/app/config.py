"""
Configuration and constants for MIMIC-IV Multi-Organ Failure Prediction
UPDATED: Pure ΔSOFA Target (Organ Dysfunction Progression ONLY)
VERIFIED: MIMIC-IV v3.1 item IDs and admission/discharge strings
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
SEQUENCE_DIR = OUTPUT_DIR / "sequences"
PREPROCESS_DIR = BASE_DIR / "preprocess"
LOGS_DIR = BASE_DIR / "logs"

# ============================================================
# Directory Creation
# ============================================================
for dir_path in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR,
                 SEQUENCE_DIR, PREPROCESS_DIR, LOGS_DIR]:
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
    "d_items",
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
    "d_icd_procedures",
]

# ============================================================
# COHORT DEFINITION WINDOWS
# ============================================================
MIN_ICU_STAY_HOURS = 24
FEATURE_WINDOW_HOURS = 6
PREDICTION_START_HOUR = 6
PREDICTION_END_HOUR = 18

# ============================================================
# COHORT EXCLUSION: Requirements 1 & 3
# ============================================================
# Requirement 1: Exclude inter-hospital transfers IN.
#   Patients arriving from another hospital or facility arrive with a
#   shifted illness window; their "first 6 hours" in our ICU is not the
#   first 6 hours of their acute illness. Transfer rates differ between
#   MIMIC and eICU, so including them would confound external validation.
#
# Requirement 3: Exclude transfers OUT to another acute setting.
#   Patients discharged to another acute hospital or facility have a
#   different clinical trajectory than patients discharged to lower-
#   acuity settings (home, SNF, rehab, hospice).
#
# Requirement 2 (ward-to-ICU patients from FLOOR) is NOT excluded.
#   Rationale: ICU admission is still the point at which ICU-level
#   organ dysfunction is assessed, consistent with standard SOFA
#   progression literature. Documented as a caveat.

EXCLUDE_INTERHOSPITAL_TRANSFERS = True

# VERIFIED against MIMIC-IV v3.1 admissions table on 2026-09-22:
#   TRANSFER FROM HOSPITAL:                        56,227 admissions
#   TRANSFER FROM SKILLED NURSING FACILITY:         6,317 admissions
MIMIC_TRANSFER_IN_LOCATIONS = [
    'TRANSFER FROM HOSPITAL',
    'TRANSFER FROM SKILLED NURSING FACILITY',
]

# VERIFIED against MIMIC-IV v3.1 admissions table on 2026-09-22:
#   OTHER FACILITY:    1,592 discharges
#   ACUTE HOSPITAL:    2,334 discharges
MIMIC_TRANSFER_OUT_LOCATIONS = [
    'OTHER FACILITY',
    'ACUTE HOSPITAL',
]

# ============================================================
# SOFA Item IDs - Cardiovascular
# ============================================================
HR_ITEMID = 220045
SBP_ITEMIDS = [220050, 220179]   # arterial first, NIBP fallback
DBP_ITEMIDS = [220051, 220180]
MAP_ITEMIDS = [220052, 220181]

VASOPRESSORS = {
    'norepinephrine': 221906,
    'epinephrine': 221289,
    'dopamine': 221662,
    'dobutamine': 221653,
    'vasopressin': 222315,
    'phenylephrine': 221749,
}

# ============================================================
# SOFA Item IDs - Lab Values
# ============================================================
LAB_ITEM_IDS = {
    'creatinine': 50912,
    'bilirubin': 50885,
    'platelets': 51265,
    'bun': 51006,
    'lactate': 50813,
    'wbc': 51301,
    'hemoglobin': 51222,
    'hematocrit': 51221,
    'sodium': 50983,
    'potassium': 50971,
    'chloride': 50902,
    'bicarbonate': 50882,
    'glucose': 50931,
    'albumin': 50862,
    'alt': 50861,
    'ast': 50878,
    'alkaline_phosphatase': 50863,
    'inr': 51237,
    'ptt': 51275,
    'pt': 51274,
}

SOFA_LAB_ITEM_IDS = {
    k: LAB_ITEM_IDS[k]
    for k in ['creatinine', 'bilirubin', 'platelets']
}

# ============================================================
# SOFA Item IDs - Respiratory
# ============================================================
RESPIRATORY_ITEM_IDS = {
    'spo2': 220277,
    'pao2': 220224,
    'fio2': 223835,
    'rr': 220210,
}

SOFA_RESPIRATORY_ITEM_IDS = {
    k: RESPIRATORY_ITEM_IDS[k]
    for k in ['pao2', 'fio2']
}

# ============================================================
# SOFA Item IDs - Neurological (GCS)
# ============================================================
GCS_ITEM_IDS = {
    'gcs_verbal': 223900,
    'gcs_motor': 223901,
    'gcs_eyes': 220739,
    'gcs_total': None,   # computed in code, not queried directly
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
    'neurological',
]

ORGAN_COLORS = {
    'cardiovascular': '#FF6B6B',
    'respiratory': '#4ECDC4',
    'renal': '#45B7D1',
    'liver': '#96CEB4',
    'coagulation': '#FFEAA7',
    'neurological': '#DDA0DD',
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
# OUTCOME DEFINITION (Pure ΔSOFA)
# ============================================================
SOFA_CHANGE_THRESHOLD = 2

OUTCOME_DEFINITION = {
    'sofa_increase': SOFA_CHANGE_THRESHOLD,
    'includes_death': False,
    'time_window_start': PREDICTION_START_HOUR,
    'time_window_end': PREDICTION_END_HOUR,
    'target_type': 'organ_dysfunction_progression',
    'target_description': f'ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} points at '
                          f'{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR} hours',
}

# ============================================================
# IMPUTATION STRATEGIES
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
EARLY_STOPPING_METRIC = 'PRAUC'

CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.01,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'custom_metric': ['PRAUC', 'Recall', 'Precision', 'F1', 'AUC'],
    'random_seed': RANDOM_STATE,
    'od_type': 'Iter',
    'od_wait': EARLY_STOPPING_ROUNDS,
    'verbose': False,
}

XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
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
    'metric': 'average_precision',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
}

# ============================================================
# Deep Learning Parameters (Phase 6b — CNN-LSTM)
# ============================================================
DEEP_LEARNING_PARAMS = {
    'sequence_length': 12,
    'n_features': 20,
    'n_channels': 40,

    'conv_filters': [64, 32],
    'conv_kernel_size': 2,
    'conv_activation': 'relu',
    'use_batch_norm': True,

    'lstm_units': 128,
    'lstm_return_sequences': True,

    'attention_units': 64,

    'dense_units': 32,
    'dense_activation': 'relu',
    'dropout_rate': 0.3,

    'output_activation': 'none',

    'loss': 'bce_with_logits',
    'optimizer': 'adam',
    'gradient_clip': 1.0,

    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'early_stopping_metric': 'auprc',
    'pos_weight': 4.3,
    'n_folds': 5,
    'random_state': 42,

    'num_workers': 0,
    'shuffle': True,
    'pin_memory': False,

    'save_best_only': True,
    'checkpoint_dir': 'outputs/models/phase6b',
    'log_every_n_epochs': 5,
    'verbose': False,

    'deterministic': True,
    'seed': 42,

    'framework': 'pytorch',
    'device': 'auto',
}

MODELS_TO_TRAIN = [
    'catboost',
    'xgboost',
    'lightgbm',
    'cnn_lstm',
]

# ============================================================
# Performance Metrics
# ============================================================
PRIMARY_METRIC = 'recall'
SELECTION_METRIC = 'auprc'

SECONDARY_METRICS = [
    'auroc',
    'auprc',
    'precision',
    'f1_score',
    'accuracy',
    'brier_score',
    'specificity',
    'npv',
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
# eICU PATHS AND CONFIGURATION
# ============================================================
EICU_DIR = DATA_DIR / "eicu"
EICU_DB_NAME = "eicu.duckdb"

EICU_TABLES = [
    "patient",
    "vitalPeriodic",
    "vitalAperiodic",
    "lab",
    "infusionDrug",
    "respiratoryCare",
    "respiratoryCharting",
    "nurseCharting",
    "intakeOutput",
    "hospital",
    "apachePatientResult",
]

# ============================================================
# eICU COHORT EXCLUSION (mirrors MIMIC Req 1 & 3)
# ============================================================
EICU_EXCLUDE_TRANSFERS = True

# unitadmitsource values indicating transfer IN (Req 1)
EICU_TRANSFER_IN_SOURCES = [
    'ICU',
    'Other ICU',
    'Other Hospital',
]

# unitdischargelocation values indicating transfer OUT (Req 3)
EICU_TRANSFER_OUT_LOCATIONS = [
    'ICU',
    'Other Hospital',
]

# Ward-to-ICU equivalent: eICU admits from 'Floor' are kept (Req 2)
EICU_WARD_SOURCE = 'Floor'

# ============================================================
# eICU TIME WINDOWS (minutes from ICU admit)
# ============================================================
EICU_FEATURE_WINDOW_MIN = FEATURE_WINDOW_HOURS * 60          # 360
EICU_OUTCOME_START_MIN = PREDICTION_START_HOUR * 60          # 360
EICU_OUTCOME_END_MIN = PREDICTION_END_HOUR * 60              # 1080
EICU_MIN_LOS_MINUTES = MIN_ICU_STAY_HOURS * 60               # 1440

# ============================================================
# eICU LAB NAME MAPPINGS
# ============================================================
# eICU stores labs as strings in `labname`. These are case-insensitive
# substrings. Verify against phase8a_check_eicu.py output before running.
EICU_LAB_PATTERNS = {
    'creatinine': ['creatinine'],
    'bilirubin': ['total bilirubin', 'bilirubin, total', 'bilirubin'],
    'platelets': ['platelets', 'platelet count'],
    'bun': ['bun', 'blood urea nitrogen', 'urea nitrogen'],
    'lactate': ['lactate'],
    'wbc': ['wbc', 'white blood cell'],
    'hemoglobin': ['hemoglobin', 'hgb'],
    'hematocrit': ['hematocrit', 'hct'],
    'sodium': ['sodium'],
    'potassium': ['potassium'],
    'chloride': ['chloride'],
    'bicarbonate': ['bicarbonate', 'hco3'],
    'glucose': ['glucose'],
    'albumin': ['albumin'],
    'alt': ['alt', 'sgpt'],
    'ast': ['ast', 'sgot'],
    'inr': ['inr'],
    'pao2': ['pao2'],
}

# Lab names to EXCLUDE from matching (contain other analytes)
# Applied globally: any lab name containing these substrings is skipped
EICU_LAB_EXCLUSIONS = [
    'urinary',
    'urine',
    'csf',
    'body fluid',
    'whole blood',
    'fluid',
    'dialysate',
    'drain',
    'stool',
    'sputum',
    'csf',
    'serum-free',
]

# ============================================================
# eICU VASOPRESSOR DRUG NAME PATTERNS
# ============================================================
EICU_VASO_PATTERNS = {
    'norepinephrine': ['norepinephrine', 'levophed'],
    'epinephrine': ['epinephrine', 'adrenaline'],
    'dopamine': ['dopamine'],
    'dobutamine': ['dobutamine'],
}

# ============================================================
# eICU GCS SOURCE
# ============================================================
EICU_GCS_SOURCE = 'nurseCharting'
# VERIFIED 2026-09-22 against your eICU-CRD v2.0:
# eICU nurseCharting has ONLY 'GCS Total'. No component-level GCS.
EICU_GCS_LABELS = {
    'gcs_total': ['gcs total', 'gcs'],
}

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
    'gcs_total', 'norepinephrine', 'epinephrine', 'dopamine',
]

# ============================================================
# Visualization Settings
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE = 'viridis'
