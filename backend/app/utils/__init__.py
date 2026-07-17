"""
Utility modules for the ICU Deterioration Prediction API.

Modules:
- preprocessing: Data preprocessing and feature engineering
- validation: Input validation utilities
"""

from app.utils.preprocessing import (
    preprocess_features,
    normalize_data,
    impute_missing,
    extract_features_from_dict,
    get_feature_summary,
    FEATURE_NAMES,
    NORM_STATS
)

from app.utils.validation import (
    validate_patient_data,
    validate_prediction_input,
    validate_feature_range,
    validate_patient_id,
    is_within_range,
    get_feature_ranges,
    validate_batch,
    CLINICAL_RANGES,
    REQUIRED_FEATURES
)

__all__ = [
    # Preprocessing
    "preprocess_features",
    "normalize_data",
    "impute_missing",
    "extract_features_from_dict",
    "get_feature_summary",
    "FEATURE_NAMES",
    "NORM_STATS",
    
    # Validation
    "validate_patient_data",
    "validate_prediction_input",
    "validate_feature_range",
    "validate_patient_id",
    "is_within_range",
    "get_feature_ranges",
    "validate_batch",
    "CLINICAL_RANGES",
    "REQUIRED_FEATURES"
]
