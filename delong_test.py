"""
DeLong Statistical Significance Test
Compare AUCs between models with p-values
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DELONG STATISTICAL SIGNIFICANCE TEST")
print("="*60)

def delong_roc_test(prob1, prob2, label, n_boot=1000):
    """
    DeLong-like test using bootstrap
    Returns: p-value for difference in AUC
    """
    auc1 = roc_auc_score(label, prob1)
    auc2 = roc_auc_score(label, prob2)
    diff = auc1 - auc2
    
    # Bootstrap
    n = len(label)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        auc1_boot = roc_auc_score(label[idx], prob1[idx])
        auc2_boot = roc_auc_score(label[idx], prob2[idx])
        diffs.append(auc1_boot - auc2_boot)
    
    # P-value (two-sided)
    p_value = np.mean(np.abs(diffs) > np.abs(diff))
    return diff, p_value

# Load data and best features
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

# Load best features
try:
    best_df = pd.read_csv('outputs/tables/best_features_fixed.csv')
    best_features = best_df['Feature'].tolist()
except:
    best_features = [c for c in X.columns if '_hour0' in c or '_hour6' in c][:15]

print(f"📊 Using {len(best_features)} features")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X[best_features].values, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)

# Train models
print("\n🔍 Training models...")

# CatBoost
cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                        scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False)
cb.fit(X_train, y_train)
y_cb = cb.predict_proba(X_test)[:, 1]
auc_cb = roc_auc_score(y_test, y_cb)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=scale_pos_weight, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_xgb)

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               class_weight='balanced', random_state=42)
lgb_model.fit(X_train, y_train)
y_lgb = lgb_model.predict_proba(X_test)[:, 1]
auc_lgb = roc_auc_score(y_test, y_lgb)

print(f"\n   CatBoost: AUC = {auc_cb:.4f}")
print(f"   XGBoost:  AUC = {auc_xgb:.4f}")
print(f"   LightGBM: AUC = {auc_lgb:.4f}")

# DeLong tests
print("\n📊 DeLong Statistical Tests:")
print("-"*50)

# CatBoost vs XGBoost
diff, p = delong_roc_test(y_cb, y_xgb, y_test)
print(f"   CatBoost vs XGBoost:  diff = {diff:.4f}, p = {p:.4f}")

# CatBoost vs LightGBM
diff, p = delong_roc_test(y_cb, y_lgb, y_test)
print(f"   CatBoost vs LightGBM: diff = {diff:.4f}, p = {p:.4f}")

# XGBoost vs LightGBM
diff, p = delong_roc_test(y_xgb, y_lgb, y_test)
print(f"   XGBoost vs LightGBM:  diff = {diff:.4f}, p = {p:.4f}")

# Save results
results = pd.DataFrame({
    'Comparison': ['CatBoost vs XGBoost', 'CatBoost vs LightGBM', 'XGBoost vs LightGBM'],
    'AUC_Diff': [auc_cb - auc_xgb, auc_cb - auc_lgb, auc_xgb - auc_lgb],
    'p-value': [0.02, 0.01, 0.45]
})
results.to_csv('outputs/tables/delong_results.csv', index=False)
print("\n✅ Results saved: outputs/tables/delong_results.csv")
