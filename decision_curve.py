"""
Decision Curve Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DECISION CURVE ANALYSIS")
print("="*60)

def decision_curve(y_true, y_proba, thresholds):
    """Calculate net benefit for decision curve"""
    net_benefit = []
    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        benefit = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(benefit)
    return np.array(net_benefit)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"📊 Data: {X.shape}")

# Load best features correctly
try:
    best_df = pd.read_csv('outputs/tables/best_features_fixed.csv')
    best_features = best_df['Feature'].tolist()
    print(f"✅ Using {len(best_features)} best features")
except:
    best_features = [c for c in X.columns if '_hour0' in c or '_hour6' in c][:15]
    print(f"⚠️ Using fallback features: {len(best_features)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X[best_features].values, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_seed=42,
    verbose=False
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Decision curve
thresholds = np.linspace(0.01, 0.99, 50)
net_benefit = decision_curve(y_test, y_proba, thresholds)

# Perfect model (for reference)
perfect_proba = y_test.astype(float)
perfect_benefit = decision_curve(y_test, perfect_proba, thresholds)

# Treat all (baseline)
treat_all_benefit = decision_curve(y_test, np.ones_like(y_test), thresholds)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(thresholds, net_benefit, linewidth=2, label='ML Model', color='blue')
plt.plot(thresholds, perfect_benefit, 'k--', label='Perfect Model', alpha=0.5)
plt.plot(thresholds, treat_all_benefit, 'r--', label='Treat All', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.02, 0.15)
plt.tight_layout()
plt.savefig('outputs/plots/decision_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Decision curve saved: outputs/plots/decision_curve.png")

# Calculate clinical utility at threshold 0.3
t = 0.3
y_pred = (y_proba > t).astype(int)
tp = np.sum((y_pred == 1) & (y_test == 1))
fp = np.sum((y_pred == 1) & (y_test == 0))
n = len(y_test)

benefit = (tp / n) - (fp / n) * (t / (1 - t))
print(f"\n📊 Clinical Utility at threshold 0.3:")
print(f"   Net Benefit: {benefit:.4f}")
print(f"   True Positives: {tp}")
print(f"   False Positives: {fp}")

# Save results
results = pd.DataFrame({
    'Threshold': thresholds,
    'Net Benefit': net_benefit
})
results.to_csv('outputs/tables/decision_curve_results.csv', index=False)
print("✅ Results saved: outputs/tables/decision_curve_results.csv")
