"""
SHAP Dependence Plot for Top Features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SHAP DEPENDENCE PLOTS")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

best_features = [
    'heart_rate_hour6', 'heart_rate_hour0', 'creatinine_hour0',
    'creatinine_hour6_missing', 'map_hour0', 'shock_index_hour0',
    'creatinine_hour6', 'sbp_variability', 'sbp_hour6',
    'heart_rate_pct_change', 'shock_index_hour6', 'map_hour6',
    'sbp_hour0', 'gcs_hour0', 'fio2_hour0'
]

X_subset = X[best_features]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_subset.values, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False
)
model.fit(X_train, y_train)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Dependence plot for top 3 features
top_features = ['heart_rate_hour6', 'creatinine_hour0', 'map_hour0']
feature_labels = ['Heart Rate (Hour 6)', 'Creatinine (Hour 0)', 'MAP (Hour 0)']

for feat, label in zip(top_features, feature_labels):
    idx = best_features.index(feat)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(idx, shap_values, X_test, feature_names=best_features,
                         show=False, interaction_index=None)
    plt.title(f'SHAP Dependence Plot: {label}', fontsize=14)
    plt.xlabel(label, fontsize=12)
    plt.ylabel('SHAP Value', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'publication/figures/shap_dependence_{feat}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP Dependence plot for {label} saved")

print("✅ All SHAP dependence plots saved")
