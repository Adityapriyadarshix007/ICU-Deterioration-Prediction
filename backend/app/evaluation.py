"""
Model Evaluation Module with Calibration and Statistical Testing
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix, classification_report
)
from scipy import stats
from sklearn.utils import resample
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with calibration and statistics"""
    
    def __init__(self, y_true, y_pred, y_proba):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(self.y_pred == self.y_true)
        metrics['precision'] = self._precision()
        metrics['recall'] = self._recall()
        metrics['f1'] = self._f1()
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba)
        except:
            metrics['roc_auc'] = 0.5
        
        # Brier Score (calibration)
        metrics['brier_score'] = brier_score_loss(self.y_true, self.y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0][0])
        metrics['false_positives'] = int(cm[0][1])
        metrics['false_negatives'] = int(cm[1][0])
        metrics['true_positives'] = int(cm[1][1])
        
        return metrics
    
    def _precision(self) -> float:
        tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        fp = np.sum((self.y_pred == 1) & (self.y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _recall(self) -> float:
        tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        fn = np.sum((self.y_pred == 0) & (self.y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _f1(self) -> float:
        p = self._precision()
        r = self._recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def bootstrap_ci(self, n_bootstrap: int = 1000, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals"""
        metrics = {}
        n = len(self.y_true)
        
        for _ in range(n_bootstrap):
            idx = resample(range(n), n_samples=n)
            y_true_b = self.y_true[idx]
            y_proba_b = self.y_proba[idx]
            y_pred_b = (y_proba_b > 0.5).astype(int)
            
            try:
                auc = roc_auc_score(y_true_b, y_proba_b)
                metrics.setdefault('roc_auc', []).append(auc)
            except:
                pass
            
            brier = brier_score_loss(y_true_b, y_proba_b)
            metrics.setdefault('brier_score', []).append(brier)
        
        cis = {}
        for metric_name, values in metrics.items():
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            cis[metric_name] = (lower, upper)
        
        return cis
    
    def calibration_curve(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve"""
        return calibration_curve(self.y_true, self.y_proba, n_bins=n_bins)
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, save_path: str = None):
        """Plot calibration curve"""
        prob_true, prob_pred = self.calibration_curve()
        brier = brier_score_loss(self.y_true, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'Brier = {brier:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curve(self, save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
