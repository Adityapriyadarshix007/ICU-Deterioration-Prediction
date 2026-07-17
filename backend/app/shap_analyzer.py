"""
SHAP Explainability Module for ICU Predictions
Provides model interpretability at both global and local levels
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """SHAP-based model explainability"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, background_data: np.ndarray, method='kernel'):
        """Create SHAP explainer"""
        if method == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        elif method == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model, background_data)
        
        return self.explainer
    
    def compute_shap_values(self, data: np.ndarray):
        """Compute SHAP values for the given data"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        self.shap_values = self.explainer.shap_values(data)
        return self.shap_values
    
    def plot_summary(self, save_path: str = None):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_waterfall(self, instance_idx: int, save_path: str = None):
        """Create SHAP waterfall plot for a single instance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=self.shap_values[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_top_features(self, n: int = 20) -> Dict[str, float]:
        """Get top N important features"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        importance = np.abs(self.shap_values).mean(axis=0)
        feature_importance = dict(zip(self.feature_names, importance))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:n])
    
    def explain_prediction(self, instance: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction
        Returns: Top contributing features and their impact
        """
        if self.shap_values is None:
            self.compute_shap_values(instance)
        
        shap_vals = self.shap_values[instance_idx]
        feature_impacts = dict(zip(self.feature_names, shap_vals))
        
        # Sort by absolute impact
        sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 5 contributors
        top_contributors = []
        for feature, impact in sorted_impacts[:5]:
            direction = 'increases' if impact > 0 else 'decreases'
            top_contributors.append({
                'feature': feature,
                'impact': float(impact),
                'direction': direction,
                'value': instance[instance_idx][self.feature_names.index(feature)]
            })
        
        return {
            'top_contributors': top_contributors,
            'prediction': float(self.model.predict(instance)[0]),
            'feature_values': dict(zip(self.feature_names, instance[instance_idx]))
        }
    
    def plot_force(self, instance_idx: int, save_path: str = None):
        """Create SHAP force plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
