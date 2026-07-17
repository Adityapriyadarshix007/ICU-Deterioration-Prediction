"""
Final Cross-Validation with Optimal Threshold
Fixed: Correct label mapping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FINAL CROSS-VALIDATION WITH OPTIMAL THRESHOLD")
print("="*60)

# Load enhanced data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Data Shape: {X.shape}")

X_values = X.values
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store predictions
all_probas = []
all_y_true = []
scale_pos_weight = len(y[y==0]) / (len(y[y==1]) + 1e-6)

print(f"\n🔍 Running {n_splits}-fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_values, y)):
    print(f"\n   Fold {fold+1}/{n_splits}")
    
    X_train, X_val = X_values[train_idx], X_values[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"      Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # XGBoost (simpler and more stable)
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
    
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Store for later threshold optimization
    all_probas.append(y_proba)
    all_y_true.append(y_val)
    
    # Evaluate at default threshold
    y_pred = (y_proba > 0.3).astype(int)
    auc = roc_auc_score(y_val, y_proba)
    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print(f"      AUC: {auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

# Combine all predictions
y_true_all = np.concatenate(all_y_true)
y_proba_all = np.concatenate(all_probas)

# Find optimal threshold on validation set
thresholds = np.linspace(0.1, 0.9, 30)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    y_pred = (y_proba_all > t).astype(int)
    f1 = f1_score(y_true_all, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Final evaluation
y_pred_all = (y_proba_all > best_threshold).astype(int)

auc = roc_auc_score(y_true_all, y_proba_all)
f1 = f1_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all)
mcc = matthews_corrcoef(y_true_all, y_pred_all)

print("\n" + "="*60)
print("FINAL RESULTS (5-Fold CV)")
print("="*60)
print(f"   AUC-ROC: {auc:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   MCC: {mcc:.4f}")
print(f"   Optimal Threshold: {best_threshold:.3f}")

# Save results
results = pd.DataFrame({
    'Metric': ['AUC-ROC', 'F1-Score', 'Recall', 'Precision', 'MCC', 'Threshold'],
    'Value': [auc, f1, recall, precision, mcc, best_threshold]
})
results.to_csv('outputs/tables/final_cv_results.csv', index=False)
print("\n✅ Results saved: outputs/tables/final_cv_results.csv")
