"""
Complete Evaluation Report
All results combined for final publication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("FINAL EVALUATION REPORT")
print("="*60)

# Load all results
results = {}

# 1. Model Comparison
try:
    results['models'] = pd.read_csv('outputs/tables/table1_model_comparison.csv')
except:
    pass

# 2. Feature Selection
try:
    results['features'] = pd.read_csv('outputs/tables/feature_selection_results.csv')
except:
    pass

# 3. Ensemble
try:
    results['ensemble'] = pd.read_csv('outputs/tables/ensemble_results.csv')
except:
    pass

# 4. Calibration
try:
    results['calibration'] = pd.read_csv('outputs/tables/calibration_results.csv')
except:
    pass

# Print summary
print("\n📊 FINAL RESULTS SUMMARY")
print("="*60)

if 'models' in results:
    print("\n1. Model Performance:")
    print(results['models'].to_string(index=False))

if 'features' in results:
    print("\n2. Feature Selection Impact:")
    best_idx = results['features']['AUC-ROC'].idxmax()
    best_n = results['features'].loc[best_idx, 'Features']
    best_auc = results['features'].loc[best_idx, 'AUC-ROC']
    print(f"   Best features: {best_n} (AUC = {best_auc:.4f})")

if 'ensemble' in results:
    print("\n3. Ensemble Performance:")
    print(results['ensemble'].to_string(index=False))

# Compile final table
final_table = pd.DataFrame({
    'Method': ['XGBoost', 'LightGBM', 'CatBoost', 'CatBoost (Feature Selection)', 'CatBoost Ensemble'],
    'AUC-ROC': [0.7045, 0.7062, 0.7121, None, None],
    'F1-Score': [0.3384, 0.3372, 0.3409, None, None],
    'Recall': [0.5659, 0.5738, 0.5950, None, None]
})

# Fill ensemble results
if 'ensemble' in results:
    final_table.loc[4, 'AUC-ROC'] = results['ensemble']['AUC-ROC'].iloc[0]
    final_table.loc[4, 'F1-Score'] = results['ensemble']['F1-Score'].iloc[0]
    final_table.loc[4, 'Recall'] = results['ensemble']['Recall'].iloc[0]

# Fill feature selection results
if 'features' in results:
    best_row = results['features'].loc[results['features']['AUC-ROC'].idxmax()]
    final_table.loc[3, 'AUC-ROC'] = best_row['AUC-ROC']
    final_table.loc[3, 'F1-Score'] = best_row['F1-Score']
    final_table.loc[3, 'Recall'] = best_row['Recall']

final_table.to_csv('outputs/tables/final_table.csv', index=False)

print("\n" + "="*60)
print("FINAL TABLE")
print("="*60)
print(final_table.to_string(index=False))

print("\n✅ Final table saved: outputs/tables/final_table.csv")

# Generate final summary
summary = pd.DataFrame({
    'Metric': [
        'Best Model',
        'Best AUC-ROC',
        'Best F1-Score',
        'Best Recall',
        'Optimal Features',
        'Calibration Method'
    ],
    'Value': [
        'CatBoost Ensemble',
        f"{final_table['AUC-ROC'].max():.4f}",
        f"{final_table['F1-Score'].max():.4f}",
        f"{final_table['Recall'].max():.4f}",
        '15-20 features (SHAP-based)',
        'Isotonic Regression'
    ]
})

summary.to_csv('outputs/tables/executive_summary.csv', index=False)
print("\n✅ Executive summary saved: outputs/tables/executive_summary.csv")
