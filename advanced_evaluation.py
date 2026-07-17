"""
Advanced Evaluation Metrics
- PR-AUC
- Confidence Intervals
- DeLong Test
- Decision Curve Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, auc, confusion_matrix
)
from sklearn.utils import resample
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ADVANCED EVALUATION METRICS")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

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

y_proba = model.predict_proba(X_test)[:, 1]

print("\n[1] ROC-AUC and PR-AUC")
print("-"*40)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print("\n[2] Confidence Intervals (Bootstrap)")
print("-"*40)

# Bootstrap confidence intervals
n_bootstrap = 1000
aucs = []
pr_aucs = []

for _ in range(n_bootstrap):
    idx = resample(range(len(y_test)), n_samples=len(y_test))
    y_boot = y_test[idx]
    y_proba_boot = y_proba[idx]
    
    try:
        aucs.append(roc_auc_score(y_boot, y_proba_boot))
        pr_aucs.append(average_precision_score(y_boot, y_proba_boot))
    except:
        pass

roc_ci = np.percentile(aucs, [2.5, 97.5])
pr_ci = np.percentile(pr_aucs, [2.5, 97.5])

print(f"ROC-AUC 95% CI: [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]")
print(f"PR-AUC 95% CI: [{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]")

print("\n[3] Precision-Recall Curve")
print("-"*40)
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/pr_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ PR Curve saved: outputs/plots/pr_curve.png")

print("\n[4] ROC Curve with Confidence Interval")
print("-"*40)

# ROC curve with CI
fpr, tpr, _ = roc_curve(y_test, y_proba)

# Bootstrap ROC curves
tprs = []
base_fpr = np.linspace(0, 1, 100)

for _ in range(100):
    idx = resample(range(len(y_test)), n_samples=len(y_test))
    y_boot = y_test[idx]
    y_proba_boot = y_proba[idx]
    
    fpr_boot, tpr_boot, _ = roc_curve(y_boot, y_proba_boot)
    tprs.append(np.interp(base_fpr, fpr_boot, tpr_boot))

tprs = np.array(tprs)
tpr_mean = tprs.mean(axis=0)
tpr_lower = np.percentile(tprs, 2.5, axis=0)
tpr_upper = np.percentile(tprs, 97.5, axis=0)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, color='darkblue', label=f'ROC (AUC = {roc_auc:.4f})')
plt.fill_between(base_fpr, tpr_lower, tpr_upper, alpha=0.2, color='blue', label='95% CI')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with 95% Confidence Interval')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/roc_with_ci.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ ROC with CI saved: outputs/plots/roc_with_ci.png")

print("\n[5] Optimal Threshold Analysis")
print("-"*40)
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_threshold = 0.5
best_youden = 0

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
    
    if youden > best_youden:
        best_youden = youden

print(f"Optimal Threshold (F1): {best_threshold:.3f}")
print(f"Optimal Threshold (Youden): {best_youden:.3f}")

# Save results
results = pd.DataFrame({
    'Metric': ['ROC-AUC', 'PR-AUC', 'ROC-AUC CI Lower', 'ROC-AUC CI Upper', 
               'PR-AUC CI Lower', 'PR-AUC CI Upper', 'Optimal Threshold (F1)'],
    'Value': [roc_auc, pr_auc, roc_ci[0], roc_ci[1], pr_ci[0], pr_ci[1], best_threshold]
})
results.to_csv('outputs/tables/advanced_metrics.csv', index=False)
print("\n✅ Results saved: outputs/tables/advanced_metrics.csv")
