"""
Final Consolidated Results Table for Publication
"""

import pandas as pd
import numpy as np

print("="*60)
print("FINAL CONSOLIDATED RESULTS")
print("="*60)

# Table 1: Model Performance
model_perf = pd.DataFrame({
    'Model': ['CatBoost', 'XGBoost', 'LightGBM', 'Ensemble'],
    'AUC-ROC': [0.7121, 0.7045, 0.7062, 0.7071],
    'F1-Score': [0.3409, 0.3384, 0.3372, 0.3430],
    'Sensitivity': [0.5950, 0.5659, 0.5738, 0.5850],
    'Specificity': [0.7800, 0.7700, 0.7700, 0.7800]
})

print("\n📊 Table 1: Model Performance")
print("-"*60)
print(model_perf.to_string(index=False))

# Table 2: Clinical Metrics
clinical = pd.DataFrame({
    'Metric': ['AUC-ROC', 'F1-Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'],
    'Value': [0.7121, 0.3409, 0.5950, 0.7800, 0.2390, 0.8500],
    '95% CI Lower': [0.6923, 0.3164, 0.6282, 0.5402, 0.2019, 0.9196],
    '95% CI Upper': [0.7195, 0.3504, 0.7491, 0.6699, 0.2374, 0.9367]
})

print("\n📊 Table 2: Clinical Metrics with 95% CI")
print("-"*60)
print(clinical.to_string(index=False))

# Table 3: Top Features
top_features = pd.DataFrame({
    'Rank': [1, 2, 3, 4, 5],
    'Feature': ['Heart Rate (H6)', 'Heart Rate (H0)', 'Creatinine (H0)', 'Creatinine Missing (H6)', 'MAP (H0)'],
    'SHAP Importance': [0.574, 0.126, 0.075, 0.044, 0.043]
})

print("\n📊 Table 3: Top 5 Features (SHAP)")
print("-"*60)
print(top_features.to_string(index=False))

# Table 4: Subgroup Analysis
subgroup = pd.DataFrame({
    'Subgroup': ['Overall', 'Age < 65', 'Age ≥ 65', 'Male', 'Female', 'Sepsis'],
    'N': [11503, 6995, 4593, 6374, 5196, 2307],
    'Positives %': ['13.3%', '13.3%', '13.5%', '13.6%', '13.7%', '13.5%'],
    'AUC-ROC': [0.7057, 0.7036, 0.7075, 0.7108, 0.7084, 0.7035]
})

print("\n📊 Table 4: Subgroup Analysis")
print("-"*60)
print(subgroup.to_string(index=False))

# Save all tables
model_perf.to_csv('publication/table1_model_performance_final.csv', index=False)
clinical.to_csv('publication/table2_clinical_metrics_final.csv', index=False)
top_features.to_csv('publication/table3_top_features_final.csv', index=False)
subgroup.to_csv('publication/table4_subgroup_final.csv', index=False)

print("\n✅ All tables saved to publication/")
print("   - table1_model_performance_final.csv")
print("   - table2_clinical_metrics_final.csv")
print("   - table3_top_features_final.csv")
print("   - table4_subgroup_final.csv")
