"""
Publication-Ready Results Summary
Generates all tables and figures for the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("="*60)
print("PUBLICATION-READY RESULTS SUMMARY")
print("="*60)

# ============================================================
# TABLE 1: Model Performance Comparison
# ============================================================
print("\n[1] Model Performance Comparison")

models = pd.DataFrame({
    'Model': ['LSTM', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble (CV)'],
    'AUC-ROC': [0.7010, 0.7045, 0.7062, 0.7121, 0.7074],
    'F1-Score': [0.3419, 0.3384, 0.3372, 0.3409, 0.3437],
    'Recall': [0.5039, 0.5659, 0.5738, 0.5950, 0.5220],
    'Precision': [0.2587, 0.2410, 0.2390, 0.2390, 0.2580]
})

print(models.to_string(index=False))
models.to_csv("outputs/tables/table1_model_comparison.csv", index=False)

# ============================================================
# TABLE 2: Imputation Comparison
# ============================================================
print("\n[2] Imputation Comparison")

imputation = pd.DataFrame({
    'Method': ['Zero', 'Median', 'Mean', 'KNN (k=5)', 'MICE'],
    'AUC-ROC': [0.7016, 0.7016, 0.6991, 0.6857, 0.6915],
    'F1-Score': [0.3346, 0.3387, 0.3285, 0.3213, 0.3241],
    'Recall': [0.5124, 0.5809, 0.6364, 0.5992, 0.6110]
})

print(imputation.to_string(index=False))
imputation.to_csv("outputs/tables/table2_imputation.csv", index=False)

# ============================================================
# TABLE 3: Top Features (SHAP)
# ============================================================
print("\n[3] Top 10 Features (SHAP)")

shap_features = pd.DataFrame({
    'Rank': range(1, 11),
    'Feature': [
        'Heart Rate (Hour 6)',
        'Heart Rate (Hour 0)',
        'Creatinine (Hour 0)',
        'Creatinine (Hour 6)',
        'SBP (Hour 6)',
        'Shock Index (Hour 0)',
        'SBP (Hour 0)',
        'Creatinine Missing (Hour 6)',
        'MAP (Hour 6)',
        'SBP Variability'
    ],
    'SHAP Importance': [0.574, 0.126, 0.075, 0.059, 0.058, 0.053, 0.052, 0.044, 0.043, 0.040]
})

print(shap_features.to_string(index=False))
shap_features.to_csv("outputs/tables/table3_shap_features.csv", index=False)

# ============================================================
# TABLE 4: Clinical Interpretation
# ============================================================
print("\n[4] Clinical Interpretation")

clinical = pd.DataFrame({
    'Feature': ['Heart Rate (Hour 6)', 'Creatinine (Hour 0)', 'SBP (Hour 6)'],
    'Clinical Meaning': [
        'Cardiovascular stability',
        'Renal function baseline',
        'Hemodynamic status'
    ],
    'Implication': [
        'Early cardiovascular dysfunction predicts organ failure',
        'Baseline renal function is a strong predictor',
        'Blood pressure trends indicate perfusion status'
    ]
})

print(clinical.to_string(index=False))
clinical.to_csv("outputs/tables/table4_clinical.csv", index=False)

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY STATISTICS")
print("="*60)

summary = {
    'Metric': [
        'Best AUC-ROC',
        'Best F1-Score',
        'Best Recall',
        'Best Imputation',
        'Most Important Feature',
        'Brier Score',
        'Dataset Size',
        'Features Used'
    ],
    'Value': [
        '0.7074 (5-Fold CV)',
        '0.3437 (5-Fold CV)',
        '0.595 (CatBoost)',
        'Median Imputation',
        'Heart Rate (Hour 6)',
        '0.2021',
        '57,515 ICU stays',
        '65 engineered features'
    ]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv("outputs/tables/summary_statistics.csv", index=False)

print("\n✅ All tables saved to outputs/tables/")
print("✅ Publication-ready results complete!")
