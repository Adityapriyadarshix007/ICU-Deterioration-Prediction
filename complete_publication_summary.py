"""
Complete Publication Summary
All information in one place
"""

import pandas as pd
import os

print("="*70)
print("📊 COMPLETE PUBLICATION SUMMARY")
print("="*70)

# Load metrics
metrics = pd.read_csv('outputs/tables/final_consistent_metrics.csv')
metrics_dict = dict(zip(metrics['Metric'], metrics['Value']))

print("\n📌 FINAL METRICS (USE THESE EVERYWHERE)")
print("-"*50)
print(f"   AUC-ROC:        {metrics_dict['AUC-ROC']:.4f}")
print(f"   PR-AUC:         {metrics_dict['PR-AUC']:.4f}")
print(f"   Sensitivity:    {metrics_dict['Sensitivity']:.4f}")
print(f"   Specificity:    {metrics_dict['Specificity']:.4f}")
print(f"   PPV:            {metrics_dict['PPV']:.4f}")
print(f"   NPV:            {metrics_dict['NPV']:.4f}")
print(f"   F1-Score:       {metrics_dict['F1-Score']:.4f}")
print(f"   Accuracy:       {metrics_dict['Accuracy']:.4f}")
print(f"   MCC:            {metrics_dict['MCC']:.4f}")
print(f"   Brier Score:    {metrics_dict['Brier Score']:.4f}")
print(f"   Optimal Threshold: {metrics_dict['Optimal Threshold']:.3f}")

print("\n📌 KEY FINDINGS")
print("-"*50)
print("   Best Model: CatBoost")
print(f"   Primary AUC: {metrics_dict['AUC-ROC']:.4f}")
print("   Most Important Feature: Heart Rate at Hour 6 (SHAP: 0.574)")
print("   Clinical Comparison: ML outperforms qSOFA, NEWS2, MEWS")
print("   Calibration: Brier Score 0.2049")
print("   Robustness: 5-fold CV AUC 0.6987 ± 0.0053")
print("   Class Prevalence: 13.3%")

print("\n📌 DATASET SUMMARY")
print("-"*50)
print("   Database: MIMIC-IV v3.1")
print("   ICU Stays: 57,515")
print("   Features: 15 SHAP-selected")
print("   Train/Test Split: 80/20")

print("\n📌 NOVELTY STATEMENT")
print("-"*50)
print("   1. Systematic evaluation of ML models with SHAP-based feature selection")
print("   2. Clinical comparison against qSOFA, NEWS2, MEWS")
print("   3. Comprehensive evaluation: calibration, DCA, subgroup analysis")
print("   4. Reproducible pipeline with consistent metrics")
print("   5. Identification of Heart Rate at Hour 6 as key predictor")

print("\n✅ Complete publication summary ready!")
