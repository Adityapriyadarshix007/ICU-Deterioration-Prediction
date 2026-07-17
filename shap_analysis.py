"""
SHAP Analysis for Model Interpretability
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*60)

# Load enhanced data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Data Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

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
model.fit(X_train, y_train)

print("✅ Model trained")

# SHAP Analysis
print("\n🔍 Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature names
feature_names = X.columns.tolist()

# 1. Summary Plot
print("\n📊 Generating SHAP Summary Plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Feature Importance Summary')
plt.tight_layout()
plt.savefig('outputs/plots/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/plots/shap_summary.png")

# 2. Top 20 Features
print("\n📊 Top 20 Features:")
importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))
feature_importance.to_csv('outputs/tables/shap_top20_features.csv', index=False)

# 3. Bar Plot of Top 10
plt.figure(figsize=(10, 8))
top10 = feature_importance.head(10)
plt.barh(top10['Feature'], top10['Importance'], color='steelblue')
plt.xlabel('Mean SHAP Value')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/plots/shap_top10.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/plots/shap_top10.png")

# 4. Waterfall plot for sample patient
print("\n📊 Sample Waterfall Plot...")
patient_idx = 0
plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[patient_idx],
        base_values=explainer.expected_value,
        data=X_test[patient_idx],
        feature_names=feature_names
    ),
    show=False
)
plt.title(f'SHAP Waterfall Plot - Sample Patient (Predicted Risk: {model.predict_proba(X_test[patient_idx].reshape(1,-1))[0,1]:.3f})')
plt.tight_layout()
plt.savefig('outputs/plots/shap_waterfall_sample.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/plots/shap_waterfall_sample.png")

print("\n✅ SHAP Analysis Complete!")
