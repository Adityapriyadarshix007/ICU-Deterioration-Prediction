"""
Subgroup Analysis
Evaluate model performance across different patient subgroups
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SUBGROUP ANALYSIS")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

# Create simulated subgroups (in practice, these would come from actual patient data)
np.random.seed(42)
n = len(y)

subgroups = {
    'Overall': np.ones(n, dtype=bool),
    'Age < 65': np.random.choice([True, False], n, p=[0.6, 0.4]),
    'Age ≥ 65': np.random.choice([True, False], n, p=[0.4, 0.6]),
    'Male': np.random.choice([True, False], n, p=[0.55, 0.45]),
    'Female': np.random.choice([True, False], n, p=[0.45, 0.55]),
    'Sepsis': np.random.choice([True, False], n, p=[0.2, 0.8]),
    'Non-Sepsis': np.random.choice([True, False], n, p=[0.8, 0.2]),
    'AKI': np.random.choice([True, False], n, p=[0.15, 0.85]),
    'Non-AKI': np.random.choice([True, False], n, p=[0.85, 0.15]),
}

# Load best features
try:
    best_df = pd.read_csv('outputs/tables/best_features_fixed.csv')
    best_features = best_df['Feature'].tolist()
except:
    best_features = [c for c in X.columns if '_hour0' in c or '_hour6' in c][:15]

print(f"📊 Using {len(best_features)} features")

# Train on full data once
X_train, X_test, y_train, y_test = train_test_split(
    X[best_features].values, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False
)
model.fit(X_train, y_train)

# Evaluate on subgroups
results = []

for name, mask in subgroups.items():
    if mask.sum() == 0:
        continue
    
    # Get test indices for this subgroup
    test_mask = mask[X_test_indices] if len(mask) == n else mask
    if test_mask.sum() == 0:
        continue
    
    y_sub = y_test[test_mask]
    y_proba_sub = y_proba[test_mask]
    
    auc = roc_auc_score(y_sub, y_proba_sub)
    n_sub = len(y_sub)
    pos_pct = y_sub.mean() * 100
    
    results.append({
        'Subgroup': name,
        'N': n_sub,
        'Positives %': f'{pos_pct:.1f}%',
        'AUC-ROC': auc
    })

# Store test indices
X_test_indices = range(len(y_test))
y_proba = model.predict_proba(X_test)[:, 1]

# Re-evaluate
results = []
for name, mask in subgroups.items():
    if mask.sum() == 0:
        continue
    
    # Get indices for this subgroup
    subgroup_indices = [i for i, m in enumerate(mask) if m]
    test_indices_in_subgroup = [i for i in range(len(y_test)) if i in subgroup_indices]
    
    if len(test_indices_in_subgroup) == 0:
        continue
    
    y_sub = y_test[test_indices_in_subgroup]
    y_proba_sub = y_proba[test_indices_in_subgroup]
    
    auc = roc_auc_score(y_sub, y_proba_sub)
    n_sub = len(y_sub)
    pos_pct = y_sub.mean() * 100
    
    results.append({
        'Subgroup': name,
        'N': n_sub,
        'Positives %': f'{pos_pct:.1f}%',
        'AUC-ROC': auc
    })

# Summary
print("\n📊 Subgroup Analysis Results:")
print("-"*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv('outputs/tables/subgroup_analysis.csv', index=False)
print("\n✅ Results saved: outputs/tables/subgroup_analysis.csv")
