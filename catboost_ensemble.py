"""
CatBoost Ensemble with Bagging
Multiple seeds for improved stability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CATBOOST ENSEMBLE WITH BAGGING")
print("="*60)

# Load best features
try:
    best_features = pd.read_csv('outputs/tables/best_features.csv', header=None)[0].tolist()
    print(f"\n📊 Using {len(best_features)} best features")
except:
    print("\n⚠️ Best features not found, using all features")
    best_features = None

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
if best_features:
    X = X[best_features]
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"📊 Data Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)

# Train multiple CatBoost models with different seeds
seeds = [0, 11, 42, 77, 101]
models = []
probas = []

print("\n🔍 Training ensemble...")

for seed in seeds:
    print(f"   Seed {seed}...")
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_seed=seed,
        verbose=False
    )
    model.fit(X_train, y_train)
    models.append(model)
    proba = model.predict_proba(X_test)[:, 1]
    probas.append(proba)

# Average predictions
ensemble_proba = np.mean(probas, axis=0)
ensemble_pred = (ensemble_proba > 0.3).astype(int)

# Evaluate
auc = roc_auc_score(y_test, ensemble_proba)
f1 = f1_score(y_test, ensemble_pred)
recall = recall_score(y_test, ensemble_pred)

print("\n" + "="*60)
print("ENSEMBLE RESULTS")
print("="*60)
print(f"   AUC-ROC: {auc:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Recall: {recall:.4f}")

# Individual model performance
print("\n📊 Individual Model Performance:")
for i, (seed, proba) in enumerate(zip(seeds, probas)):
    auc_i = roc_auc_score(y_test, proba)
    print(f"   Seed {seed}: AUC = {auc_i:.4f}")

# Save ensemble model
import joblib
joblib.dump(models, 'outputs/models/catboost_ensemble.pkl')
print("\n✅ Ensemble saved: outputs/models/catboost_ensemble.pkl")

# Save results
results = pd.DataFrame({
    'Model': ['CatBoost Ensemble'],
    'AUC-ROC': [auc],
    'F1-Score': [f1],
    'Recall': [recall],
    'Seeds': [seeds]
})
results.to_csv('outputs/tables/ensemble_results.csv', index=False)
print("✅ Results saved: outputs/tables/ensemble_results.csv")
