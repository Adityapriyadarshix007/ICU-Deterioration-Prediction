"""
Calibration Plot with Perfect Calibration Line
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CALIBRATION PLOT")
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

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')
brier = brier_score_loss(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=10,
         color='blue', label=f'Model (Brier = {brier:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Observed Frequency', fontsize=12)
plt.title('Calibration Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.6)
plt.ylim(0, 0.6)
plt.tight_layout()
plt.savefig('publication/figures/calibration_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Calibration plot saved: publication/figures/calibration_plot.png")
print(f"   Brier Score: {brier:.4f}")
