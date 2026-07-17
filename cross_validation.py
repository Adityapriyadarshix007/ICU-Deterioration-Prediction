"""
5-Fold Stratified Cross-Validation with Ensemble
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*60)

# Load data
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
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
    
    # Train CatBoost
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    cat_proba = cat_model.predict_proba(X_val)[:, 1]
    
    # Ensemble
    ensemble_proba = 0.3 * xgb_proba + 0.3 * lgb_proba + 0.4 * cat_proba
    
    all_probas.append(ensemble_proba)
    all_y_true.append(y_val)
    
    auc = roc_auc_score(y_val, ensemble_proba)
    print(f"      Fold AUC: {auc:.4f}")

# Combine all predictions
y_true_all = np.concatenate(all_y_true)
y_proba_all = np.concatenate(all_probas)

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 20)
best_f1 = 0
best_th = 0.5
for t in thresholds:
    y_pred = (y_proba_all > t).astype(int)
    f1 = f1_score(y_true_all, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = t

y_pred_all = (y_proba_all > best_th).astype(int)

auc = roc_auc_score(y_true_all, y_proba_all)
f1 = f1_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)
print(f"   AUC-ROC: {auc:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   Optimal Threshold: {best_th:.3f}")

# Save results
results_df = pd.DataFrame({
    'Metric': ['AUC-ROC', 'F1-Score', 'Recall', 'Threshold'],
    'Value': [auc, f1, recall, best_th]
})
results_df.to_csv("outputs/tables/cv_results.csv", index=False)
print("\n✅ Results saved: outputs/tables/cv_results.csv")
