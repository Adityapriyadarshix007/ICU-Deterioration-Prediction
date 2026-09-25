# config.py
"""
Configuration and constants for MIMIC-IV Multi-Organ Failure Prediction
UPDATED: Pure ΔSOFA Target (Organ Dysfunction Progression ONLY)
"""

import os
from pathlib import Path
from datetime import datetime

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
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
#
# TIMELINE:
# |--- 0-6h (FEATURES) ---|--- 6-18h (OUTCOME) ---|
#
# FEATURES: Extract from 0-6 hours ONLY
# OUTCOME: Assess ΔSOFA at 6-18 hours
# NO DATA LEAKAGE: Features END before Outcome BEGINS
# ============================================================

# Minimum ICU stay (24 hours ensures we have complete data)
MIN_ICU_STAY_HOURS = 24

# FEATURE EXTRACTION WINDOW: Use ONLY first 6 hours
FEATURE_WINDOW_HOURS = 6

# OUTCOME WINDOW: Assess deterioration at 6-18 hours
PREDICTION_START_HOUR = 6
PREDICTION_END_HOUR = 18

# ============================================================
# SOFA Item IDs - Cardiovascular
# ============================================================
MAP_ITEMID = 220045
HR_ITEMID = 220180
SBP_ITEMID = 220179
DBP_ITEMID = 220180  # Note: DBP uses same itemid as HR in MIMIC-IV

# Vasopressors with their item IDs
VASOPRESSORS = {
    'norepinephrine': 221906,
    'epinephrine': 221289,
    'dopamine': 221662,
    'dobutamine': 221653,
    'vasopressin': 221749,
    'phenylephrine': 221749,  # Note: Same itemid as vasopressin
}

# ============================================================
# SOFA Item IDs - Lab Values (from hosp/labevents)
# ============================================================
LAB_ITEM_IDS = {
    'creatinine': 50912,
    'bilirubin': 50885,
    'platelets': 51265,
    'bun': 50971,
    'lactate': 50983,
    'wbc': 51301,
    'hemoglobin': 50822,
    'hematocrit': 51221,
    'sodium': 50983,
    'potassium': 50971,
    'chloride': 50995,
    'bicarbonate': 50882,
    'glucose': 50931,
    'albumin': 50862,
    'alt': 50861,
    'ast': 50878,
    'alkaline_phosphatase': 50863,
    'inr': 51237,
    'ptt': 51273,
    'pt': 51274,
}

# SOFA-specific lab item IDs (for SOFA score calculation)
SOFA_LAB_ITEM_IDS = {
    'creatinine': 50912,
    'bilirubin': 50885,
    'platelets': 51265,
}

# ============================================================
# SOFA Item IDs - Respiratory (from chartevents)
# ============================================================
RESPIRATORY_ITEM_IDS = {
    'spo2': 220277,
    'pao2': 220235,
    'fio2': 220211,
    'rr': 220210,
}

# ============================================================
# SOFA Item IDs - Neurological (GCS - from chartevents)
# ============================================================
GCS_ITEM_IDS = {
    'gcs_verbal': 223900,
    'gcs_motor': 223901,
    'gcs_eyes': 223902,
    'gcs_total': 220739,
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
MAX_MISSING_RATE = 0.60  # Drop features with >60% missing
OUTLIER_LOW = 0.005      # Winsorize at 0.5th percentile
OUTLIER_HIGH = 0.995     # Winsorize at 99.5th percentile

# Aggregation methods for time-series features
AGGREGATION_METHODS = ['min', 'max', 'mean', 'median', 'std', 'last', 'count']

# ============================================================
# OUTCOME DEFINITION (Pure ΔSOFA ONLY)
# ============================================================
SOFA_CHANGE_THRESHOLD = 2

OUTCOME_DEFINITION = {
    'sofa_increase': 2,
    'includes_death': False,
    'time_window_start': 6,
    'time_window_end': 18,
    'target_type': 'organ_dysfunction_progression',
    'target_description': 'ΔSOFA ≥ 2 points at 6-18 hours'
}

# ============================================================
# IMPUTATION STRATEGIES (6 strategies)
# ============================================================
IMPUTATION_STRATEGIES = [
    'zero',
    'median',
    'locf',
    'linear',
    'knn',
    'mice'
]

IMPUTATION_PARAMS = {
    'zero': {
        'value': 0,
        'description': 'Replace missing values with 0'
    },
    'median': {
        'strategy': 'median',
        'description': 'Replace missing values with column median'
    },
    'locf': {
        'method': 'ffill',
        'limit': None,
        'description': 'Forward fill (Last Observation Carried Forward)'
    },
    'linear': {
        'method': 'linear',
        'limit_direction': 'both',
        'description': 'Linear interpolation between known points'
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'metric': 'nan_euclidean',
        'description': 'K-Nearest Neighbors imputation'
    },
    'mice': {
        'n_imputations': 5,
        'max_iter': 10,
        'random_state': 42,
        'description': 'Multiple Imputation by Chained Equations'
    }
}

# ============================================================
# Model Training Parameters
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.15
N_FOLDS = 5

# CatBoost Parameters
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.01,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'Recall',
    'custom_metric': ['Recall', 'Precision', 'F1', 'AUC'],
    'random_seed': RANDOM_STATE,
    'early_stopping_rounds': 50,
    'verbose': False
}

# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'early_stopping_rounds': 50
}

# LightGBM Parameters
LGBM_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'num_leaves': 31,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'early_stopping_rounds': 50
}

# Deep Learning (CNN-LSTM+Attention) Parameters
DEEP_LEARNING_PARAMS = {
    'sequence_length': 6,
    'n_features': 42,
    'conv_filters': [64, 32],
    'conv_kernel_size': 2,
    'lstm_units': 128,
    'attention_units': 64,
    'dense_units': 32,
    'dropout_rate': 0.3,
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'patience': 10
}

# Models to train
MODELS_TO_TRAIN = [
    'catboost',
    'xgboost',
    'lightgbm',
    'cnn_lstm_attention'
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

# Required features for external validation
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

# ============================================================
# Clinical Reference Ranges (for outlier detection)
# ============================================================
CLINICAL_RANGES = {
    'heart_rate': (30, 200),
    'respiratory_rate': (5, 50),
    'spo2': (50, 100),
    'map': (40, 160),
    'sbp': (60, 220),
    'dbp': (30, 140),
    'temperature': (32, 42),
    'creatinine': (0.1, 15),
    'bilirubin': (0.1, 50),
    'platelets': (1, 1000),
    'bun': (1, 200),
    'lactate': (0.1, 20),
    'wbc': (0.1, 100),
    'hemoglobin': (3, 20),
    'hematocrit': (10, 70),
    'sodium': (100, 180),
    'potassium': (1, 10),
    'chloride': (60, 130),
    'bicarbonate': (5, 50),
    'glucose': (20, 800),
    'albumin': (0.1, 6),
    'alt': (1, 1000),
    'ast': (1, 1000),
    'inr': (0.5, 10),
}

# ============================================================
# SOFA Score Reference Tables (for calculation)
# ============================================================
SOFA_SCORE_TABLES = {
    'cardiovascular': {
        'map': {70: 1, 0: 0},
        'vasopressors': {'dopamine': 5, 'epinephrine': 0.1, 'norepinephrine': 0.1}
    },
    'respiratory': {
        'pao2_fio2': {400: 0, 300: 1, 200: 2, 100: 3, 0: 4}
    },
    'renal': {
        'creatinine': {1.2: 0, 2.0: 1, 3.5: 2, 5.0: 3, 0: 4}
    },
    'liver': {
        'bilirubin': {1.2: 0, 2.0: 1, 6.0: 2, 12.0: 3, 0: 4}
    },
    'coagulation': {
        'platelets': {150: 0, 100: 1, 50: 2, 20: 3, 0: 4}
    },
    'neurological': {
        'gcs': {15: 0, 13: 1, 10: 2, 6: 3, 3: 4}
    }
}

print(f"✅ Configuration loaded from: {Path(__file__).name}")
print(f"📁 Base Directory: {BASE_DIR}")
print(f"📁 Data Directory: {DATA_DIR}")
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"📁 Models Directory: {MODEL_DIR}")
print(f"📁 Tables Directory: {TABLE_DIR}")
print(f"📁 Figures Directory: {FIG_DIR}")
print(f"📁 Logs Directory: {LOGS_DIR}")
print(f"\n📊 MIMIC Version: {MIMIC_VERSION}")
print(f"📊 eICU Version: {EICU_VERSION}")
print(f"\n🎯 Target: {OUTCOME_DEFINITION['target_description']}")
print(f"📈 Imputation Strategies: {len(IMPUTATION_STRATEGIES)}")
print(f"🤖 Models: {len(MODELS_TO_TRAIN)}")