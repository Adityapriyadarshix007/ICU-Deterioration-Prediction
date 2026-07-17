"""
Advanced Feature Engineering for ICU Time-Series Data
Computes temporal statistics, trends, and derived features
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Extract rich temporal features from ICU time-series data"""
    
    def __init__(self):
        self.feature_functions = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'std': np.std,
            'slope': self._compute_slope,
            'first_diff': self._first_difference,
            'last_value': self._last_value,
            'coefficient_variation': self._compute_cv,
            'moving_avg_3': self._moving_average_3,
            'moving_avg_6': self._moving_average_6,
            'percentile_25': lambda x: np.percentile(x, 25),
            'percentile_75': lambda x: np.percentile(x, 75),
            'skewness': stats.skew,
            'kurtosis': stats.kurtosis,
        }
        
        self.variables = [
            'heart_rate', 'sbp', 'dbp', 'map', 'spo2',
            'gcs', 'lactate', 'creatinine', 'fio2',
            'urine_output', 'respiratory_rate', 'temperature'
        ]
    
    def _compute_slope(self, series: np.ndarray) -> float:
        """Compute linear slope of the series"""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    def _first_difference(self, series: np.ndarray) -> float:
        """Compute first difference (change from first to last)"""
        if len(series) < 2:
            return 0.0
        return series[-1] - series[0]
    
    def _last_value(self, series: np.ndarray) -> float:
        """Get the last non-null value"""
        valid = series[~np.isnan(series)]
        return valid[-1] if len(valid) > 0 else 0.0
    
    def _compute_cv(self, series: np.ndarray) -> float:
        """Coefficient of variation"""
        mean = np.mean(series)
        std = np.std(series)
        return std / mean if mean != 0 else 0.0
    
    def _moving_average_3(self, series: np.ndarray) -> float:
        """3-point moving average"""
        if len(series) < 3:
            return np.mean(series) if len(series) > 0 else 0.0
        return np.mean(series[-3:])
    
    def _moving_average_6(self, series: np.ndarray) -> float:
        """6-point moving average"""
        if len(series) < 6:
            return np.mean(series) if len(series) > 0 else 0.0
        return np.mean(series[-6:])
    
    def extract_features(self, patient_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Extract all features for a patient
        """
        features = {}
        
        for var_name, series in patient_data.items():
            if var_name not in self.variables:
                continue
                
            series_array = np.array(series)
            series_array = series_array[~np.isnan(series_array)]
            
            if len(series_array) == 0:
                continue
                
            for feat_name, func in self.feature_functions.items():
                try:
                    value = func(series_array)
                    if isinstance(value, np.ndarray):
                        value = value.item() if value.size == 1 else value.mean()
                    features[f"{var_name}_{feat_name}"] = float(value)
                except:
                    features[f"{var_name}_{feat_name}"] = 0.0
        
        return features
    
    def create_time_windows(self, data: pd.DataFrame, window_hours: List[int] = [2, 4, 6, 12]) -> pd.DataFrame:
        """
        Create features over multiple time windows
        """
        window_features = {}
        
        for window in window_hours:
            window_data = data[data['icu_hour'] <= window].copy()
            for var in self.variables:
                if var in window_data.columns:
                    series = window_data[var].dropna().values
                    if len(series) > 0:
                        window_features[f"{var}_window_{window}_mean"] = np.mean(series)
                        window_features[f"{var}_window_{window}_std"] = np.std(series)
                        window_features[f"{var}_window_{window}_slope"] = self._compute_slope(series)
        
        return pd.DataFrame([window_features])
    
    def get_delta_features(self, data: pd.DataFrame, hours: List[int] = [0, 2, 4, 6]) -> Dict[str, float]:
        """
        Compute delta features between different time points
        """
        delta_features = {}
        
        for var in self.variables:
            if var not in data.columns:
                continue
                
            for i, h1 in enumerate(hours):
                for h2 in hours[i+1:]:
                    val1 = data[data['icu_hour'] == h1][var].values
                    val2 = data[data['icu_hour'] == h2][var].values
                    
                    if len(val1) > 0 and len(val2) > 0:
                        delta_features[f"{var}_delta_{h1}_{h2}"] = val2[0] - val1[0]
                        delta_features[f"{var}_delta_pct_{h1}_{h2}"] = ((val2[0] - val1[0]) / (val1[0] + 1e-6)) * 100
        
        return delta_features

# Singleton instance
feature_engineer = FeatureEngineer()
