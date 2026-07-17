"""
Ablation Study: Impact of Each Component
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ABLATION STUDY")
print("="*60)

# Load data
X_full = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"📊 Full Features: {X_full.shape[1]}")

# Define feature groups
feature_groups = {
    'Baseline (8 vitals)': [c for c in X_full.columns if any(x in c for x in 
                          ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 
                           'urine_output', 'fio2', 'creatinine']) and 
                          '_hour' in c and 'missing' not in c],
    '+ Hour 6 Features': [c for c in X_full.columns if '_hour6' in c],
    '+ Delta Features': [c for c in X_full.columns if '_delta' in c],
    '+ Derived Features': [c for c in X_full.columns if any(x in c for x in 
                          ['shock_index', 'map', 'pulse_pressure', 'slope', 
                           'pct_change', 'variability', 'missing', 'risk_score'])],
    '+ Clinical Scores': [c for c in X_full.columns if any(x in c for x in 
                          ['qSOFA', 'MEWS', 'NEWS2'])]
}

# Keep only existing columns
feature_groups = {k: [c for c in v if c in X_full.columns] for k, v in feature_groups.items()}

results = []

for name, cols in feature_groups.items():
    if not cols:
        continue
    
    print(f"\n🔍 Testing: {name}")
    print(f"   Features: {len(cols)}")
    
    X_subset = X_full[cols].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.2, random_state=42, stratify=y
    )
    
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
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.3).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results.append({
        'Feature Set': name,
        'Num Features': len(cols),
        'AUC-ROC': auc,
        'F1-Score': f1,
        'Recall': recall
    })

print("\n" + "="*60)
print("ABLATION STUDY RESULTS")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv('outputs/tables/ablation_results.csv', index=False)
print("\n✅ Results saved: outputs/tables/ablation_results.csv")
