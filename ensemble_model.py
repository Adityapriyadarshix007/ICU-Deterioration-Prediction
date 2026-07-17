"""
Ensemble Model: LightGBM + XGBoost + CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE MODEL")
print("="*60)

# Load enhanced data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Data Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Calculate class weights
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)

# Train individual models
print("\n[1] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n[2] Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train, y_train)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

print("\n[3] Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=150,
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_seed=42,
    verbose=False
)
cat_model.fit(X_train, y_train)
cat_proba = cat_model.predict_proba(X_test)[:, 1]

# Ensemble predictions
print("\n[4] Creating ensemble...")
# Weighted average
weights = [0.3, 0.3, 0.4]  # XGBoost, LightGBM, CatBoost
ensemble_proba = weights[0] * xgb_proba + weights[1] * lgb_proba + weights[2] * cat_proba

# Evaluate each model
models = {
    'XGBoost': xgb_proba,
    'LightGBM': lgb_proba,
    'CatBoost': cat_proba,
    'Ensemble': ensemble_proba
}

results = []
thresholds = np.linspace(0.1, 0.9, 20)

for name, proba in models.items():
    # Find optimal threshold
    best_f1 = 0
    best_th = 0.5
    for t in thresholds:
        y_pred = (proba > t).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = t
    
    y_pred = (proba > best_th).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'AUC-ROC': auc,
        'F1-Score': f1,
        'Recall': recall,
        'Precision': precision,
        'Threshold': best_th
    })

# Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("outputs/tables/ensemble_results.csv", index=False)
print("\n✅ Results saved: outputs/tables/ensemble_results.csv")

# Save models
import joblib
joblib.dump(xgb_model, "outputs/models/xgboost_ensemble.pkl")
joblib.dump(lgb_model, "outputs/models/lightgbm_ensemble.pkl")
joblib.dump(cat_model, "outputs/models/catboost_ensemble.pkl")
print("✅ Models saved")
