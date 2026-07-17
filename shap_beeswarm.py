"""
SHAP Beeswarm Summary Plot
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
print("SHAP BEESWARM SUMMARY PLOT")
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

# Feature names
feature_names = [f.replace('_hour', ' H') for f in best_features]
feature_names = [f.replace('_', ' ').title() for f in feature_names]

# Summary plot (beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Feature Importance (Beeswarm)', fontsize=14)
plt.tight_layout()
plt.savefig('publication/figures/shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ SHAP Beeswarm plot saved: publication/figures/shap_beeswarm.png")
