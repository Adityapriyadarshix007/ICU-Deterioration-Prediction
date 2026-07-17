"""
CatBoost with Optuna Hyperparameter Optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import optuna
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CATBOOST WITH OPTUNA TUNING")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Data Shape: {X.shape}")
print(f"   Class balance: {np.sum(y)} deteriorated, {len(y) - np.sum(y)} stable")

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Calculate class weight
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)

def objective(trial):
    """Optuna objective function"""
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'scale_pos_weight': scale_pos_weight,
        'random_seed': 42,
        'verbose': False
    }
    
    model = CatBoostClassifier(**params)
    
    # 3-fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_proba))
    
    return np.mean(auc_scores)

print("\n🔍 Running Optuna optimization...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n✅ Best parameters: {study.best_params}")
print(f"✅ Best CV AUC: {study.best_value:.4f}")

# Train final model with best parameters
best_params = study.best_params
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_seed'] = 42
best_params['verbose'] = False

model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)

# Evaluate
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)

auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n📊 Final Performance:")
print(f"   AUC-ROC: {auc:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Recall: {recall:.4f}")

# Save model
import joblib
joblib.dump(model, "outputs/models/catboost_tuned.pkl")
print("\n✅ Model saved: outputs/models/catboost_tuned.pkl")
