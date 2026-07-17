"""
Data preprocessing utilities for the ICU prediction model.
Handles feature extraction, normalization, and imputation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)

# Feature names in order
FEATURE_NAMES = [
    'heart_rate',
    'sbp', 
    'dbp',
    'gcs',
    'lactate',
    'urine_output',
    'fio2',
    'creatinine'
]

# Normalization statistics (mean, std) - from training data
# These would be computed from your training data in production
NORM_STATS = {
    'heart_rate': {'mean': 85.0, 'std': 20.0},
    'sbp': {'mean': 120.0, 'std': 25.0},
    'dbp': {'mean': 70.0, 'std': 15.0},
    'gcs': {'mean': 12.0, 'std': 3.0},
    'lactate': {'mean': 2.0, 'std': 1.5},
    'urine_output': {'mean': 50.0, 'std': 30.0},
    'fio2': {'mean': 35.0, 'std': 15.0},
    'creatinine': {'mean': 1.0, 'std': 0.8}
}


def preprocess_features(
    data: Dict[str, Any],
    normalize: bool = True,
    impute: bool = True
) -> np.ndarray:
    """
    Preprocess patient features for model prediction.
    
    Args:
        data: Dictionary containing patient features
        normalize: Whether to normalize features
        impute: Whether to impute missing values
    
    Returns:
        np.ndarray: Preprocessed feature array of shape (1, 1, 8)
    """
    try:
        # Extract features in correct order
        features = []
        for feature in FEATURE_NAMES:
            value = data.get(feature, None)
            features.append(value)
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Impute missing values
        if impute:
            features_array = impute_missing(features_array)
        
        # Normalize
        if normalize:
            features_array = normalize_data(features_array)
        
        # Reshape for LSTM: (batch_size, timesteps, features)
        # We use 1 timestep for single prediction
        features_array = features_array.reshape(1, 1, -1)
        
        return features_array
        
    except Exception as e:
        logger.error(f"Error in preprocess_features: {e}")
        raise ValueError(f"Failed to preprocess features: {e}")


def normalize_data(
    features: np.ndarray,
    stats: Optional[Dict[str, Dict[str, float]]] = None
) -> np.ndarray:
    """
    Normalize features using mean and standard deviation.
    
    Args:
        features: Feature array
        stats: Normalization statistics (mean, std) for each feature
    
    Returns:
        np.ndarray: Normalized features
    """
    if stats is None:
        stats = NORM_STATS
    
    normalized = np.zeros_like(features, dtype=np.float32)
    
    for i, feature_name in enumerate(FEATURE_NAMES):
        if i < len(features):
            mean = stats.get(feature_name, {}).get('mean', 0)
            std = stats.get(feature_name, {}).get('std', 1)
            
            if std == 0:
                std = 1
            
            normalized[i] = (features[i] - mean) / std
    
    # Clip extreme values to prevent instability
    normalized = np.clip(normalized, -10, 10)
    
    return normalized


def impute_missing(
    features: np.ndarray,
    strategy: str = 'mean'
) -> np.ndarray:
    """
    Impute missing values in feature array.
    
    Args:
        features: Feature array with potential NaN values
        strategy: Imputation strategy ('mean', 'median', 'zero')
    
    Returns:
        np.ndarray: Feature array with missing values imputed
    """
    imputed = features.copy()
    
    # Get column means (excluding NaN)
    if strategy == 'mean':
        col_mean = np.nanmean(features, axis=0)
        for i in range(len(imputed)):
            if np.isnan(imputed[i]):
                imputed[i] = col_mean[i] if not np.isnan(col_mean[i]) else 0.0
                
    elif strategy == 'median':
        col_median = np.nanmedian(features, axis=0)
        for i in range(len(imputed)):
            if np.isnan(imputed[i]):
                imputed[i] = col_median[i] if not np.isnan(col_median[i]) else 0.0
                
    elif strategy == 'zero':
        imputed = np.nan_to_num(imputed, nan=0.0)
    
    return imputed


def extract_features_from_dict(
    data: Dict[str, Any],
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Extract features from a dictionary with validation.
    
    Args:
        data: Input dictionary
        feature_names: List of feature names to extract
    
    Returns:
        Dict[str, float]: Extracted features
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    extracted = {}
    for feature in feature_names:
        value = data.get(feature, None)
        
        # Convert to float if possible
        if value is not None:
            try:
                extracted[feature] = float(value)
            except (ValueError, TypeError):
                extracted[feature] = None
        else:
            extracted[feature] = None
    
    return extracted


def get_feature_summary(features: np.ndarray) -> Dict[str, Any]:
    """
    Get summary statistics for features.
    
    Args:
        features: Feature array
    
    Returns:
        Dict[str, Any]: Summary statistics
    """
    return {
        'shape': features.shape,
        'min': float(np.min(features)),
        'max': float(np.max(features)),
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'has_nan': bool(np.isnan(features).any()),
        'has_inf': bool(np.isinf(features).any())
    }


def save_preprocessed_features(
    features: np.ndarray,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save preprocessed features to file.
    
    Args:
        features: Feature array
        filepath: Path to save file
        metadata: Optional metadata to include
    """
    data = {
        'features': features.tolist(),
        'metadata': metadata or {},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_preprocessed_features(filepath: str) -> Dict[str, Any]:
    """
    Load preprocessed features from file.
    
    Args:
        filepath: Path to file
    
    Returns:
        Dict[str, Any]: Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data
