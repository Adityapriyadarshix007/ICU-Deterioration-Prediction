"""
Calibration Analysis for Clinical Models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CALIBRATION ANALYSIS")
print("="*60)

# Load enhanced data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Data Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

print("✅ Model trained")

# Get predictions
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)

# Brier Score
brier = brier_score_loss(y_test, y_proba)
print(f"\n📊 Brier Score: {brier:.4f}")

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

# Plot calibration curve
plt.figure(figsize=(10, 8))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'Model (Brier={brier:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/calibration_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/plots/calibration_curve.png")

# Reliability Diagram
plt.figure(figsize=(10, 8))
plt.bar(prob_pred, prob_true - prob_pred, width=0.08, alpha=0.7, color='steelblue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Predicted Probability')
plt.ylabel('Calibration Error (True - Predicted)')
plt.title('Reliability Diagram')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/reliability_diagram.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/plots/reliability_diagram.png")

# Save results
results = pd.DataFrame({
    'Metric': ['Brier Score'],
    'Value': [brier]
})
results.to_csv('outputs/tables/calibration_results.csv', index=False)
print("\n✅ Calibration analysis complete!")
