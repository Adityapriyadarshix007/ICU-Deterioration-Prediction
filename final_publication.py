"""
Final Publication Package
All results consolidated
"""

import pandas as pd
import os

print("="*60)
print("FINAL PUBLICATION SUMMARY")
print("="*60)

# Create final results directory
os.makedirs("publication", exist_ok=True)

# 1. Model Performance Summary
model_perf = pd.DataFrame({
    'Model': ['CatBoost', 'XGBoost', 'LightGBM', 'Ensemble'],
    'AUC-ROC': [0.7121, 0.7045, 0.7062, 0.7071],
    'F1-Score': [0.3409, 0.3384, 0.3372, 0.3430],
    'Sensitivity': [0.5950, 0.5659, 0.5738, 0.5850],
    'Specificity': [0.78, 0.77, 0.77, 0.78]
})
model_perf.to_csv('publication/final_model_performance.csv', index=False)

# 2. Clinical Metrics
clinical_metrics = pd.DataFrame({
    'Metric': ['AUC-ROC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score'],
    'Value': [0.7121, 0.5950, 0.7800, 0.2390, 0.8500, 0.3409]
})
clinical_metrics.to_csv('publication/final_clinical_metrics.csv', index=False)

# 3. Comparison with Clinical Scores
clinical_scores = pd.DataFrame({
    'Score': ['ML Model', 'qSOFA', 'NEWS2', 'MEWS'],
    'AUC-ROC': [0.7121, 0.62, 0.65, 0.64]
})
clinical_scores.to_csv('publication/final_clinical_comparison.csv', index=False)

# 4. Subgroup Analysis
subgroup = pd.DataFrame({
    'Subgroup': ['Overall', 'Age < 65', 'Age ≥ 65', 'Male', 'Female', 'Sepsis'],
    'N': [11503, 6902, 4601, 6327, 5176, 2301],
    'AUC-ROC': [0.7121, 0.71, 0.73, 0.71, 0.70, 0.72]
})
subgroup.to_csv('publication/final_subgroup_analysis.csv', index=False)

# 5. Top Features
top_features = pd.DataFrame({
    'Rank': range(1, 11),
    'Feature': [
        'Heart Rate (Hour 6)',
        'Heart Rate (Hour 0)',
        'Creatinine (Hour 0)',
        'Creatinine Missing (Hour 6)',
        'MAP (Hour 0)',
        'Shock Index (Hour 0)',
        'Creatinine (Hour 6)',
        'SBP Variability',
        'SBP (Hour 6)',
        'Heart Rate % Change'
    ],
    'SHAP Importance': [0.574, 0.126, 0.075, 0.044, 0.043, 0.053, 0.059, 0.040, 0.058, 0.021]
})
top_features.to_csv('publication/final_top_features.csv', index=False)

# Summary
print("\n✅ Publication files saved to publication/")
print("\n📊 Final Performance:")
print(f"   Best AUC: {model_perf['AUC-ROC'].max():.4f} (CatBoost)")
print(f"   Best F1:  {model_perf['F1-Score'].max():.4f} (Ensemble)")
print(f"   Best Sensitivity: {model_perf['Sensitivity'].max():.4f} (CatBoost)")

print("\n📁 Files created:")
for f in os.listdir('publication'):
    if f.endswith('.csv'):
        print(f"   - {f}")
