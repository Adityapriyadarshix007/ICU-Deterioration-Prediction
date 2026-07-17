"""
Hyperparameter Table and Runtime Analysis
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

print("="*60)
print("HYPERPARAMETERS & RUNTIME ANALYSIS")
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

# Hyperparameters
hyperparameters = pd.DataFrame({
    'Parameter': [
        'iterations',
        'depth',
        'learning_rate',
        'scale_pos_weight',
        'loss_function',
        'random_seed',
        'l2_leaf_reg',
        'border_count'
    ],
    'Value': [
        300,
        6,
        0.1,
        len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6),
        'Logloss',
        42,
        3.0,
        128
    ]
})

print("\n📊 Hyperparameters:")
print("-"*40)
print(hyperparameters.to_string(index=False))

# Runtime analysis
print("\n📊 Runtime Analysis:")
print("-"*40)

# Training time
start = time.time()
model = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6),
    random_seed=42, verbose=False
)
model.fit(X_train, y_train)
train_time = time.time() - start

# Prediction time
start = time.time()
y_proba = model.predict_proba(X_test[:100])[:, 1]
pred_time = time.time() - start
pred_per_patient = pred_time / 100 * 1000  # ms per patient

print(f"   Training Time: {train_time:.2f} seconds")
print(f"   Prediction Time (100 patients): {pred_time:.4f} seconds")
print(f"   Prediction Time per patient: {pred_per_patient:.2f} ms")

# Model size
import os
model.save_model('outputs/models/catboost_final_model.cbm')
model_size = os.path.getsize('outputs/models/catboost_final_model.cbm') / (1024 * 1024)

print(f"   Model Size: {model_size:.2f} MB")

# Save hyperparameters
hyperparameters.to_csv('publication/hyperparameters.csv', index=False)

print("\n✅ Hyperparameters and runtime analysis complete")
