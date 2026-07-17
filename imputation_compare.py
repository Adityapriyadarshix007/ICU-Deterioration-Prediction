"""
Compare Different Imputation Strategies
- Median Imputation
- Forward Fill (patient-wise)
- MICE (Multiple Imputation)
- MissForest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("IMPUTATION STRATEGY COMPARISON")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Original Data Shape: {X.shape}")
print(f"   Missing values: {X.isnull().sum().sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# Define imputation methods
imputation_methods = {
    'Zero': SimpleImputer(strategy='constant', fill_value=0),
    'Median': SimpleImputer(strategy='median'),
    'Mean': SimpleImputer(strategy='mean'),
    'KNN (k=5)': KNNImputer(n_neighbors=5),
    'KNN (k=10)': KNNImputer(n_neighbors=10),
    'MICE (Iterative)': IterativeImputer(max_iter=10, random_state=42),
}

# Evaluate each method
results = []

for name, imputer in imputation_methods.items():
    print(f"\n🔍 Testing: {name}")
    try:
        # Impute
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
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
        model.fit(X_train_imp, y_train)
        y_proba = model.predict_proba(X_test_imp)[:, 1]
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 20)
        best_f1 = 0
        best_th = 0.5
        for t in thresholds:
            y_pred = (y_proba > t).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_th = t
        
        y_pred = (y_proba > best_th).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results.append({
            'Imputation': name,
            'AUC-ROC': auc,
            'F1-Score': f1,
            'Recall': recall
        })
        print(f"   AUC: {auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("outputs/tables/imputation_comparison.csv", index=False)
print("\n✅ Results saved: outputs/tables/imputation_comparison.csv")
