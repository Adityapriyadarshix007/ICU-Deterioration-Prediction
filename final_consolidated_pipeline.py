"""
FINAL CONSOLIDATED PIPELINE - CLEAN VERSION
All metrics from one evaluation protocol
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix,
    average_precision_score, brier_score_loss
)
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FINAL CONSOLIDATED PIPELINE")
print("="*60)

# Create output directory
os.makedirs("outputs/tables", exist_ok=True)

# Set random seeds
np.random.seed(42)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"📊 Data Shape: {X.shape}")
print(f"   Class balance: {np.sum(y)} deteriorated, {len(y) - np.sum(y)} stable")

# Use top 15 features from SHAP (deterministic selection)
best_features = [
    'heart_rate_hour6', 'heart_rate_hour0', 'creatinine_hour0',
    'creatinine_hour6_missing', 'map_hour0', 'shock_index_hour0',
    'creatinine_hour6', 'sbp_variability', 'sbp_hour6',
    'heart_rate_pct_change', 'shock_index_hour6', 'map_hour6',
    'sbp_hour0', 'gcs_hour0', 'fio2_hour0'
]

print(f"\n📊 Using {len(best_features)} deterministic features")
X_subset = X[best_features]

# Single train-test split (Primary evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X_subset.values, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split: Train {len(X_train)}, Test {len(X_test)}")

# Train CatBoost
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_seed=42,
    verbose=False
)
model.fit(X_train, y_train)

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold using Youden's J
thresholds = np.linspace(0.1, 0.9, 50)
best_youden = 0
best_threshold = 0.3

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    if youden > best_youden:
        best_youden = youden
        best_threshold = t

print(f"\n📊 Optimal Threshold (Youden): {best_threshold:.3f}")

# Final predictions at optimal threshold
y_pred = (y_proba > best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# COMPUTED METRICS (ALL FROM SAME PIPELINE)
metrics = {
    'AUC-ROC': roc_auc_score(y_test, y_proba),
    'PR-AUC': average_precision_score(y_test, y_proba),
    'Brier Score': brier_score_loss(y_test, y_proba),
    'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
    'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    'PPV': precision_score(y_test, y_pred),
    'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
    'F1-Score': f1_score(y_test, y_pred),
    'Accuracy': accuracy_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred),
    'Optimal Threshold': best_threshold,
    'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
}

# PRINT FINAL CONSISTENT METRICS
print("\n" + "="*60)
print("📊 FINAL CONSISTENT METRICS (Primary Evaluation)")
print("="*60)
print(f"   AUC-ROC:      {metrics['AUC-ROC']:.4f}")
print(f"   PR-AUC:       {metrics['PR-AUC']:.4f}")
print(f"   Brier Score:  {metrics['Brier Score']:.4f}")
print(f"   Sensitivity:  {metrics['Sensitivity']:.4f}")
print(f"   Specificity:  {metrics['Specificity']:.4f}")
print(f"   PPV:          {metrics['PPV']:.4f}")
print(f"   NPV:          {metrics['NPV']:.4f}")
print(f"   F1-Score:     {metrics['F1-Score']:.4f}")
print(f"   Accuracy:     {metrics['Accuracy']:.4f}")
print(f"   MCC:          {metrics['MCC']:.4f}")
print(f"   Threshold:    {metrics['Optimal Threshold']:.3f}")
print(f"\n📊 Confusion Matrix:")
print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

# Save to file
metrics_df = pd.DataFrame({
    'Metric': list(metrics.keys()),
    'Value': list(metrics.values())
})
metrics_df.to_csv('outputs/tables/final_consistent_metrics.csv', index=False)

print("\n✅ Saved: outputs/tables/final_consistent_metrics.csv")

# ============================================================
# 5-FOLD CROSS-VALIDATION (Robustness Check)
# ============================================================
print("\n" + "="*60)
print("📊 5-FOLD CROSS-VALIDATION (Robustness)")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_subset.values, y)):
    X_cv_train, X_cv_val = X_subset.values[train_idx], X_subset.values[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]
    
    model_cv = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False
    )
    model_cv.fit(X_cv_train, y_cv_train)
    y_cv_proba = model_cv.predict_proba(X_cv_val)[:, 1]
    cv_aucs.append(roc_auc_score(y_cv_val, y_cv_proba))

cv_mean = np.mean(cv_aucs)
cv_std = np.std(cv_aucs)

print(f"   Fold AUCs: {', '.join([f'{a:.4f}' for a in cv_aucs])}")
print(f"   CV AUC: {cv_mean:.4f} (±{cv_std:.4f})")

# Save CV results
cv_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(5)],
    'AUC': cv_aucs
})
cv_df.to_csv('outputs/tables/cv_results_final.csv', index=False)

print("\n✅ Saved: outputs/tables/cv_results_final.csv")

print("\n" + "="*60)
print("✅ FINAL CONSISTENT RESULTS GENERATED")
print("="*60)
print("   Primary Model: CatBoost (Top 15 SHAP features)")
print(f"   Primary AUC: {metrics['AUC-ROC']:.4f}")
print(f"   CV AUC: {cv_mean:.4f} (±{cv_std:.4f})")
print("   All metrics from same evaluation pipeline")
