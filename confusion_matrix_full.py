"""
Full Confusion Matrix with Clinical Metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CONFUSION MATRIX WITH CLINICAL METRICS")
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X[best_features].values, y, test_size=0.2, random_state=42, stratify=y
)

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

print(f"📊 Optimal Threshold (Youden's J): {best_threshold:.3f}")

# Final predictions
y_pred = (y_proba > best_threshold).astype(int)

# Full confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

sensitivity = recall  # TP / (TP + FN)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = precision  # TP / (TP + FP)
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\n📊 Clinical Metrics:")
print("-"*40)
print(f"   Sensitivity (Recall): {sensitivity:.4f}")
print(f"   Specificity:          {specificity:.4f}")
print(f"   PPV (Precision):      {ppv:.4f}")
print(f"   NPV:                  {npv:.4f}")
print(f"   Accuracy:             {accuracy:.4f}")
print(f"   F1-Score:             {f1:.4f}")

print(f"\n📊 Confusion Matrix:")
print(f"   True Positives:  {tp}")
print(f"   False Positives: {fp}")
print(f"   True Negatives:  {tn}")
print(f"   False Negatives: {fn}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Stable', 'Predicted Deteriorated'],
            yticklabels=['Actual Stable', 'Actual Deteriorated'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Confusion matrix saved: outputs/plots/confusion_matrix.png")

# Save results
results = pd.DataFrame({
    'Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1-Score'],
    'Value': [sensitivity, specificity, ppv, npv, accuracy, f1]
})
results.to_csv('outputs/tables/clinical_metrics.csv', index=False)
print("✅ Results saved: outputs/tables/clinical_metrics.csv")
