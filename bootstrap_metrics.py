"""
Bootstrap Confidence Intervals for All Metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix
)
from catboost import CatBoostClassifier
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("="*60)

# Load data
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

# Train model
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold (Youden's J)
thresholds = np.linspace(0.1, 0.9, 50)
best_youden = 0
best_threshold = 0.3

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    if youden > best_youden:
        best_youden = youden
        best_threshold = t

print(f"📊 Optimal Threshold: {best_threshold:.3f}")

y_pred = (y_proba > best_threshold).astype(int)

# Point estimates
metrics = {
    'AUC-ROC': roc_auc_score(y_test, y_proba),
    'F1-Score': f1_score(y_test, y_pred),
    'Sensitivity': recall_score(y_test, y_pred),
    'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    'PPV': precision_score(y_test, y_pred),
    'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
    'Accuracy': accuracy_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# Bootstrap
n_bootstrap = 1000
n = len(y_test)
boot_results = {key: [] for key in metrics.keys()}

for _ in range(n_bootstrap):
    idx = resample(range(n), n_samples=n)
    y_boot = y_test[idx]
    y_proba_boot = y_proba[idx]
    
    # Optimal threshold for bootstrap sample
    best_t = 0.3
    best_j = 0
    for t in thresholds:
        y_pred_boot = (y_proba_boot > t).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_boot, y_pred_boot).ravel()
        sens_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
        j = sens_b + spec_b - 1
        if j > best_j:
            best_j = j
            best_t = t
    
    y_pred_boot = (y_proba_boot > best_t).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_boot, y_pred_boot).ravel()
    
    boot_results['AUC-ROC'].append(roc_auc_score(y_boot, y_proba_boot))
    boot_results['F1-Score'].append(f1_score(y_boot, y_pred_boot))
    boot_results['Sensitivity'].append(recall_score(y_boot, y_pred_boot))
    boot_results['Specificity'].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
    boot_results['PPV'].append(precision_score(y_boot, y_pred_boot))
    boot_results['NPV'].append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0)
    boot_results['Accuracy'].append(accuracy_score(y_boot, y_pred_boot))
    boot_results['MCC'].append(matthews_corrcoef(y_boot, y_pred_boot))

# Calculate confidence intervals
results = []
for key, value in metrics.items():
    ci_lower = np.percentile(boot_results[key], 2.5)
    ci_upper = np.percentile(boot_results[key], 97.5)
    results.append({
        'Metric': key,
        'Value': value,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper
    })

# Print results
print("\n📊 Bootstrap Confidence Intervals:")
print("-"*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save
results_df.to_csv('outputs/tables/bootstrap_metrics.csv', index=False)
print("\n✅ Results saved: outputs/tables/bootstrap_metrics.csv")

# Format for publication
print("\n📊 Publication-Ready Table:")
print("-"*60)
for _, row in results_df.iterrows():
    print(f"   {row['Metric']}: {row['Value']:.4f} (95% CI: {row['CI Lower']:.4f}–{row['CI Upper']:.4f})")
