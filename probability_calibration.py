"""
Probability Calibration with Isotonic Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PROBABILITY CALIBRATION")
print("="*60)

# Load best features
try:
    best_features = pd.read_csv('outputs/tables/best_features.csv', header=None)[0].tolist()
    print(f"\n📊 Using {len(best_features)} best features")
except:
    best_features = None

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
if best_features:
    X = X[best_features]
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"📊 Data Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)

# Base model
base_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_seed=42,
    verbose=False
)
base_model.fit(X_train, y_train)
y_proba_base = base_model.predict_proba(X_test)[:, 1]

# Calibrated models
print("\n🔍 Training calibrated models...")

for method in ['sigmoid', 'isotonic']:
    calibrated = CalibratedClassifierCV(
        base_model,
        method=method,
        cv=5
    )
    calibrated.fit(X_train, y_train)
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    
    auc_base = roc_auc_score(y_test, y_proba_base)
    auc_cal = roc_auc_score(y_test, y_proba_cal)
    brier_base = brier_score_loss(y_test, y_proba_base)
    brier_cal = brier_score_loss(y_test, y_proba_cal)
    
    print(f"\n📊 {method.capitalize()} Calibration:")
    print(f"   AUC: {auc_base:.4f} → {auc_cal:.4f}")
    print(f"   Brier: {brier_base:.4f} → {brier_cal:.4f}")

# Save calibrated model
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

import joblib
joblib.dump(calibrated_model, 'outputs/models/catboost_calibrated.pkl')
print("\n✅ Calibrated model saved: outputs/models/catboost_calibrated.pkl")
