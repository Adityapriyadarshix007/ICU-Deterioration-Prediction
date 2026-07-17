"""
SHAP-Based Feature Selection
Remove noisy features to improve performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from catboost import CatBoostClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SHAP-BASED FEATURE SELECTION")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"\n📊 Full Data: {X.shape[1]} features")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# Train CatBoost
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
full_auc = roc_auc_score(y_test, y_proba)

print(f"\nFull Model AUC: {full_auc:.4f}")

# Get SHAP values
print("\n🔍 Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Get feature importance
importance = np.abs(shap_values).mean(axis=0)
feature_names = X.columns.tolist()
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# Test different feature counts
feature_counts = [5, 10, 15, 20, 25, 30, 40, 50, 60]
results = []

for n in feature_counts:
    top_features = feature_importance.head(n)['Feature'].tolist()
    X_subset = X[top_features]
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_subset.values, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_s = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False
    )
    model_s.fit(X_train_s, y_train_s)
    y_proba_s = model_s.predict_proba(X_test_s)[:, 1]
    
    auc = roc_auc_score(y_test_s, y_proba_s)
    y_pred = (y_proba_s > 0.3).astype(int)
    f1 = f1_score(y_test_s, y_pred)
    recall = recall_score(y_test_s, y_pred)
    
    results.append({
        'Features': n,
        'AUC-ROC': auc,
        'F1-Score': f1,
        'Recall': recall,
        'Selected': ', '.join(top_features[:5]) + '...'
    })
    
    print(f"   Top {n} features: AUC = {auc:.4f}, F1 = {f1:.4f}")

# Summary
print("\n" + "="*60)
print("FEATURE SELECTION RESULTS")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('outputs/tables/feature_selection_results.csv', index=False)

# Save best feature set
best_n = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Features']
best_features = feature_importance.head(best_n)['Feature'].tolist()
pd.Series(best_features).to_csv('outputs/tables/best_features.csv', index=False)

print(f"\n✅ Best feature count: {best_n}")
print(f"✅ Best AUC: {results_df['AUC-ROC'].max():.4f}")
print("✅ Results saved: outputs/tables/feature_selection_results.csv")
print("✅ Best features saved: outputs/tables/best_features.csv")
