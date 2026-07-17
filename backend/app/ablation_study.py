"""
Ablation Study Module
Systematically remove components to measure their impact
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class AblationStudy:
    """Conduct ablation studies on model components"""
    
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.results = {}
        
    def run_ablation(self, components: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study by removing each component
        components: List of component names to remove
        """
        results = {}
        
        # Baseline (full model)
        baseline_metrics = self._evaluate_model(self.X_train, self.X_test)
        results['Full Model'] = baseline_metrics
        
        for component in components:
            logger.info(f"Removing component: {component}")
            
            if component == 'Attention':
                X_train_abl, X_test_abl = self._remove_attention()
            elif component == 'Imputation':
                X_train_abl, X_test_abl = self._remove_imputation()
            elif component == 'Delta Features':
                X_train_abl, X_test_abl = self._remove_delta_features()
            elif component == 'Temporal Features':
                X_train_abl, X_test_abl = self._remove_temporal_features()
            else:
                X_train_abl, X_test_abl = self.X_train, self.X_test
            
            metrics = self._evaluate_model(X_train_abl, X_test_abl)
            results[f'Without {component}'] = metrics
            
        self.results = results
        return results
    
    def _evaluate_model(self, X_train, X_test) -> Dict[str, float]:
        """Train and evaluate model on given data"""
        # This should be implemented based on your model training pipeline
        # For now, returns placeholder metrics
        return {
            'AUC-ROC': 0.80,
            'Accuracy': 0.75,
            'F1-Score': 0.70
        }
    
    def _remove_attention(self):
        """Remove attention mechanism from the model"""
        # Implementation depends on your model architecture
        # For now, return original data
        return self.X_train, self.X_test
    
    def _remove_imputation(self):
        """Use no imputation (drop missing values)"""
        # Implementation depends on your data pipeline
        return self.X_train, self.X_test
    
    def _remove_delta_features(self):
        """Remove delta (change) features"""
        if self.feature_names:
            # Keep only features without '_delta' in name
            keep_indices = [i for i, name in enumerate(self.feature_names) if '_delta' not in name]
            X_train_abl = self.X_train[:, keep_indices]
            X_test_abl = self.X_test[:, keep_indices]
            return X_train_abl, X_test_abl
        return self.X_train, self.X_test
    
    def _remove_temporal_features(self):
        """Remove all temporal features (keep only static features)"""
        if self.feature_names:
            # Keep only features with 'hour0' in name (static)
            keep_indices = [i for i, name in enumerate(self.feature_names) if '_hour0' in name]
            X_train_abl = self.X_train[:, keep_indices]
            X_test_abl = self.X_test[:, keep_indices]
            return X_train_abl, X_test_abl
        return self.X_train, self.X_test
    
    def get_impact_summary(self) -> pd.DataFrame:
        """Get summary of ablation impacts"""
        import pandas as pd
        
        if not self.results:
            return pd.DataFrame()
        
        baseline = self.results['Full Model']
        summary = []
        
        for component, metrics in self.results.items():
            if component == 'Full Model':
                continue
            
            impact = {
                'Component': component,
                'AUC-ROC Drop': baseline['AUC-ROC'] - metrics['AUC-ROC'],
                'Accuracy Drop': baseline['Accuracy'] - metrics['Accuracy'],
                'F1-Score Drop': baseline['F1-Score'] - metrics['F1-Score']
            }
            summary.append(impact)
        
        return pd.DataFrame(summary)
    
    def get_critical_components(self, threshold: float = 0.05) -> List[str]:
        """Identify components that are critical (drop > threshold)"""
        critical = []
        impacts = self.get_impact_summary()
        
        for _, row in impacts.iterrows():
            if row['AUC-ROC Drop'] > threshold:
                critical.append(row['Component'])
        
        return critical
