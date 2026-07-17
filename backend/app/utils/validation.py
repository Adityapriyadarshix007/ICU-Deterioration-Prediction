"""
Input validation utilities for the ICU prediction API.
Validates patient data, feature ranges, and prediction inputs.
"""

from typing import Dict, Any, Tuple, Optional, List
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Clinical reference ranges for features
CLINICAL_RANGES = {
    'heart_rate': {'min': 30, 'max': 220, 'unit': 'bpm'},
    'sbp': {'min': 50, 'max': 250, 'unit': 'mmHg'},
    'dbp': {'min': 20, 'max': 150, 'unit': 'mmHg'},
    'gcs': {'min': 3, 'max': 15, 'unit': 'score'},
    'lactate': {'min': 0, 'max': 20, 'unit': 'mmol/L'},
    'urine_output': {'min': 0, 'max': 500, 'unit': 'mL'},
    'fio2': {'min': 21, 'max': 100, 'unit': '%'},
    'creatinine': {'min': 0.1, 'max': 15, 'unit': 'mg/dL'}
}

# Required features for prediction
REQUIRED_FEATURES = [
    'patient_id',
    'heart_rate',
    'sbp',
    'dbp',
    'gcs',
    'lactate',
    'urine_output',
    'fio2',
    'creatinine'
]


def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate patient data dictionary.
    
    Args:
        data: Patient data dictionary
    
    Returns:
        Tuple[bool, Optional[str], Dict]: (is_valid, error_message, cleaned_data)
    """
    errors = []
    cleaned_data = {}
    
    # Check if data is empty
    if not data:
        return False, "Patient data is empty", {}
    
    # Check required fields
    for field in REQUIRED_FEATURES:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        else:
            cleaned_data[field] = data[field]
    
    # Validate patient_id
    patient_id = cleaned_data.get('patient_id', '')
    if not patient_id:
        errors.append("Patient ID cannot be empty")
    elif not isinstance(patient_id, str):
        errors.append("Patient ID must be a string")
    
    # Validate each feature
    for feature in REQUIRED_FEATURES:
        if feature == 'patient_id':
            continue
            
        value = cleaned_data.get(feature)
        if value is None:
            errors.append(f"Feature '{feature}' is missing or None")
            continue
        
        try:
            value = float(value)
            cleaned_data[feature] = value
            
            # Check range
            is_valid, range_error = validate_feature_range(feature, value)
            if not is_valid:
                errors.append(f"Feature '{feature}': {range_error}")
                
        except (ValueError, TypeError):
            errors.append(f"Feature '{feature}' must be a number, got: {value}")
    
    if errors:
        return False, "; ".join(errors), cleaned_data
    
    return True, None, cleaned_data


def validate_feature_range(feature: str, value: float) -> Tuple[bool, Optional[str]]:
    """
    Validate that a feature value is within clinical ranges.
    
    Args:
        feature: Feature name
        value: Feature value
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if feature not in CLINICAL_RANGES:
        return True, None
    
    ranges = CLINICAL_RANGES[feature]
    min_val = ranges.get('min')
    max_val = ranges.get('max')
    unit = ranges.get('unit', '')
    
    if min_val is not None and value < min_val:
        return False, f"Value {value} below minimum {min_val}{unit}"
    
    if max_val is not None and value > max_val:
        return False, f"Value {value} above maximum {max_val}{unit}"
    
    return True, None


def validate_prediction_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate prediction input data.
    
    Args:
        data: Input data dictionary
    
    Returns:
        Tuple[bool, Optional[str], Dict]: (is_valid, error_message, cleaned_data)
    """
    # First, validate as patient data
    is_valid, error, cleaned = validate_patient_data(data)
    if not is_valid:
        return False, error, cleaned
    
    # Additional checks for prediction
    # Check that patient has been in ICU for at least 1 hour (not applicable for mock)
    
    # Check for extreme combinations that might indicate data entry errors
    heart_rate = cleaned.get('heart_rate', 0)
    sbp = cleaned.get('sbp', 0)
    dbp = cleaned.get('dbp', 0)
    
    # Check if SBP > DBP (clinical expectation)
    if sbp > 0 and dbp > 0 and sbp <= dbp:
        # Not necessarily invalid, but suspicious
        logger.warning(f"Suspicious blood pressure: SBP={sbp}, DBP={dbp}")
    
    # Check if heart rate is suspiciously low with normal blood pressure
    if heart_rate < 40 and sbp > 100:
        logger.warning(f"Suspicious: Low heart rate ({heart_rate}) with normal BP ({sbp})")
    
    return True, None, cleaned


def validate_patient_id(patient_id: str) -> bool:
    """
    Validate patient ID format.
    
    Args:
        patient_id: Patient ID string
    
    Returns:
        bool: True if valid
    """
    if not patient_id:
        return False
    
    # Allow alphanumeric, dash, underscore
    pattern = r'^[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, patient_id))


def get_feature_ranges() -> Dict[str, Dict]:
    """
    Get clinical reference ranges for all features.
    
    Returns:
        Dict[str, Dict]: Feature ranges
    """
    return CLINICAL_RANGES


def is_within_range(feature: str, value: float, tolerance: float = 0.1) -> bool:
    """
    Check if a value is within the clinical range with tolerance.
    
    Args:
        feature: Feature name
        value: Value to check
        tolerance: Tolerance factor (0.1 = 10%)
    
    Returns:
        bool: True if within range
    """
    if feature not in CLINICAL_RANGES:
        return True
    
    ranges = CLINICAL_RANGES[feature]
    min_val = ranges.get('min')
    max_val = ranges.get('max')
    
    if min_val is not None:
        min_val = min_val * (1 - tolerance)
    if max_val is not None:
        max_val = max_val * (1 + tolerance)
    
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    
    return True


def validate_batch(data_list: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
    """
    Validate a batch of patient data.
    
    Args:
        data_list: List of patient data dictionaries
    
    Returns:
        Tuple[List[Dict], List[str]]: (valid_data, error_messages)
    """
    valid_data = []
    errors = []
    
    for i, data in enumerate(data_list):
        is_valid, error, cleaned = validate_patient_data(data)
        if is_valid:
            valid_data.append(cleaned)
        else:
            errors.append(f"Item {i}: {error}")
    
    return valid_data, errors
