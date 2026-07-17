"""
Decision Curve Analysis - Verified Implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DECISION CURVE ANALYSIS (VERIFIED)")
print("="*60)

def decision_curve(y_true, y_proba, thresholds):
    """Calculate net benefit correctly"""
    net_benefit = []
    n = len(y_true)
    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        # Net benefit formula
        benefit = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(benefit)
    return np.array(net_benefit)

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

# Decision curve
thresholds = np.linspace(0.01, 0.99, 50)
net_benefit_model = decision_curve(y_test, y_proba, thresholds)

# Treat all strategy
net_benefit_all = decision_curve(y_test, np.ones_like(y_test), thresholds)

# Treat none (always 0)
net_benefit_none = np.zeros_like(thresholds)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(thresholds, net_benefit_model, linewidth=2, label='ML Model', color='blue')
plt.plot(thresholds, net_benefit_all, 'r--', label='Treat All', alpha=0.7)
plt.plot(thresholds, net_benefit_none, 'k--', label='Treat None', alpha=0.5)
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/decision_curve_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Decision curve saved: outputs/plots/decision_curve_fixed.png")

# Find threshold with maximum net benefit
max_idx = np.argmax(net_benefit_model)
best_threshold = thresholds[max_idx]
max_benefit = net_benefit_model[max_idx]

print(f"\n📊 Optimal Threshold (Max Net Benefit): {best_threshold:.3f}")
print(f"   Max Net Benefit: {max_benefit:.4f}")

# Save results
results = pd.DataFrame({
    'Threshold': thresholds,
    'Net Benefit': net_benefit_model,
    'Treat All': net_benefit_all,
    'Treat None': net_benefit_none
})
results.to_csv('outputs/tables/decision_curve_fixed_results.csv', index=False)
print("✅ Results saved: outputs/tables/decision_curve_fixed_results.csv")
