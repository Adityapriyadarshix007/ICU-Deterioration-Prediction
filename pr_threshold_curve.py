"""
Precision-Recall Threshold Curve
Shows Precision, Recall, and F1 vs Threshold
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PRECISION-RECALL THRESHOLD CURVE")
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

# Compute metrics across thresholds
thresholds = np.linspace(0.1, 0.9, 100)
precisions = []
recalls = []
f1_scores = []

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Optimal threshold (Youden's J)
optimal_threshold = 0.459

# Plot
plt.figure(figsize=(10, 8))
plt.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
plt.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
plt.plot(thresholds, f1_scores, 'r-', linewidth=2, label='F1-Score')

# Mark optimal threshold
plt.axvline(x=optimal_threshold, color='k', linestyle='--', linewidth=1.5,
            label=f'Optimal Threshold (0.459)')
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Annotations
plt.annotate(f'F1 = {f1_scores[np.argmin(np.abs(thresholds - optimal_threshold))]:.3f}',
             xy=(optimal_threshold, f1_scores[np.argmin(np.abs(thresholds - optimal_threshold))]),
             xytext=(optimal_threshold + 0.05, f1_scores[np.argmin(np.abs(thresholds - optimal_threshold))] + 0.05),
             fontsize=10, fontweight='bold')

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision, Recall, and F1 vs Threshold', fontsize=14)
plt.legend(loc='center right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0.1, 0.9)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('publication/figures/pr_threshold_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Precision-Recall threshold curve saved: publication/figures/pr_threshold_curve.png")
print(f"   Optimal threshold: {optimal_threshold:.3f}")
