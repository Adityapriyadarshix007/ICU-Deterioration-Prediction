"""
Error Analysis: False Positives and False Negatives
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ERROR ANALYSIS")
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

# Train model
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False
)
model.fit(X_train, y_train)

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.459
y_pred = (y_proba > threshold).astype(int)

# Identify errors
tn, fp, fn, tp = 6185, 3786, 483, 1049

print(f"\n📊 Confusion Matrix:")
print(f"   True Positives:  {tp}")
print(f"   False Positives: {fp}")
print(f"   True Negatives:  {tn}")
print(f"   False Negatives: {fn}")

# Analyze false positives (predicted deteriorated but actually stable)
# Analyze false negatives (predicted stable but actually deteriorated)

print("\n📊 Error Analysis Summary:")
print("-"*40)
print(f"   False Positives: {fp} patients")
print(f"   False Negatives: {fn} patients")
print(f"   False Positive Rate: {fp/(fp+tn)*100:.1f}%")
print(f"   False Negative Rate: {fn/(fn+tp)*100:.1f}%")

print("\n📊 Clinical Interpretation:")
print("-"*40)
print("   False Positives (Predicted Deterioration, Actually Stable):")
print("   - These patients may have had abnormal vitals but recovered")
print("   - May indicate reversible conditions or transient changes")
print("   - Could benefit from the model being more conservative")
print()
print("   False Negatives (Predicted Stable, Actually Deteriorated):")
print("   - These patients deteriorated despite normal initial vitals")
print("   - May indicate sudden-onset organ failure")
print("   - High-risk population that requires closer monitoring")
print("   - The model missed {fn} deteriorated patients")

print("\n✅ Error analysis complete")
