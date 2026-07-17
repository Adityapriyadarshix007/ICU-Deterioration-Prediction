"""
Feature Correlation Heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("FEATURE CORRELATION HEATMAP")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")

best_features = [
    'heart_rate_hour6', 'heart_rate_hour0', 'creatinine_hour0',
    'creatinine_hour6_missing', 'map_hour0', 'shock_index_hour0',
    'creatinine_hour6', 'sbp_variability', 'sbp_hour6',
    'heart_rate_pct_change', 'shock_index_hour6', 'map_hour6',
    'sbp_hour0', 'gcs_hour0', 'fio2_hour0'
]

X_subset = X[best_features]

# Feature labels for display
labels = [
    'HR (H6)', 'HR (H0)', 'Creat (H0)',
    'Creat Missing', 'MAP (H0)', 'Shock Index',
    'Creat (H6)', 'SBP Var', 'SBP (H6)',
    'HR % Change', 'Shock Index (H6)', 'MAP (H6)',
    'SBP (H0)', 'GCS (H0)', 'FiO2 (H0)'
]

# Correlation matrix
corr = X_subset.corr()

# Plot
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('publication/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Feature correlation heatmap saved: publication/figures/correlation_heatmap.png")
