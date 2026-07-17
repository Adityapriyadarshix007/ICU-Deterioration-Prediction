# ============================================================
# PHASE 5: XGBOOST BASELINE + IMPUTATION COMPARISON (FIXED)
# ============================================================
# This file:
# 1. Loads the dataset
# 2. Applies 3 imputation strategies (Forward Fill, Linear, KNN)
# 3. Trains XGBoost on each
# 4. Compares performance
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("="*60)
print("PHASE 5: XGBOOST BASELINE + IMPUTATION COMPARISON")
print("="*60)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[1] Loading data...")

X_raw = pd.read_csv("outputs/tables/X_features.csv")
y_raw = pd.read_csv("outputs/tables/y_target.csv")

print(f"✅ Loaded X: {X_raw.shape}, y: {y_raw.shape}")

# Check for columns that are all NaN
all_nan_cols = X_raw.columns[X_raw.isnull().all()].tolist()
print(f"   Columns with all NaN: {all_nan_cols}")

# Drop columns that are all NaN (they provide no information)
X_clean = X_raw.drop(columns=all_nan_cols)
print(f"   Cleaned X shape: {X_clean.shape}")

# ============================================================
# STEP 2: Define Imputation Functions
# ============================================================
print("\n[2] Defining imputation strategies...")

def forward_fill_impute(df):
    """Forward fill: carry last value forward"""
    return df.ffill(axis=0).bfill(axis=0)

def linear_interpolate_impute(df):
    """Linear interpolation between known points"""
    return df.interpolate(method='linear', axis=0, limit_direction='both')

def knn_impute(df, n_neighbors=5):
    """K-Nearest Neighbors imputation"""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    # KNN returns numpy array with same shape
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns)

# ============================================================
# STEP 3: Apply All 3 Imputation Methods
# ============================================================
print("\n[3] Applying imputation methods...")

# Store all imputed datasets
imputed_data = {}

for method_name, method_func in [
    ('Forward Fill', forward_fill_impute),
    ('Linear Interpolate', linear_interpolate_impute),
    ('KNN', knn_impute)
]:
    print(f"   - Applying {method_name}...")
    X_imputed = method_func(X_clean.copy())
    imputed_data[method_name] = X_imputed
    null_count = X_imputed.isnull().sum().sum()
    print(f"     ✅ Complete. Null values remaining: {null_count}")

# ============================================================
# STEP 4: Train XGBoost on Each Imputed Dataset
# ============================================================
print("\n[4] Training XGBoost models...")

y = y_raw['deteriorated'].values
results = {}

for method_name, X_imputed in imputed_data.items():
    print(f"\n   --- {method_name} ---")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights for imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results[method_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'model': model,
        'scaler': scaler,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"   Accuracy: {results[method_name]['accuracy']:.4f}")
    print(f"   F1-Score: {results[method_name]['f1']:.4f}")
    print(f"   AUC-ROC:  {results[method_name]['auc_roc']:.4f}")

# ============================================================
# STEP 5: Compare Results
# ============================================================
print("\n[5] Comparing imputation methods...")

comparison_df = pd.DataFrame({
    method: {
        'Accuracy': results[method]['accuracy'],
        'F1-Score': results[method]['f1'],
        'AUC-ROC': results[method]['auc_roc']
    }
    for method in results.keys()
}).T

print("\n📊 Performance Comparison:")
print(comparison_df.round(4))

comparison_df.to_csv("outputs/tables/xgboost_comparison.csv")
print(f"✅ Saved: outputs/tables/xgboost_comparison.csv")

# ============================================================
# STEP 6: Visualize Comparison
# ============================================================
print("\n[6] Creating comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart comparison
comparison_df[['Accuracy', 'F1-Score', 'AUC-ROC']].plot(
    kind='bar', ax=axes[0], colormap='viridis', edgecolor='black'
)
axes[0].set_title('XGBoost Performance by Imputation Method')
axes[0].set_xlabel('Imputation Method')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.3f', fontsize=8)

# Plot 2: ROC Curves
from sklearn.metrics import roc_curve
for method_name in results.keys():
    fpr, tpr, _ = roc_curve(results[method_name]['y_test'], results[method_name]['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{method_name} (AUC = {results[method_name]['auc_roc']:.3f})", linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
axes[1].set_title('ROC Curves by Imputation Method')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/plots/xgboost_imputation_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Saved: outputs/plots/xgboost_imputation_comparison.png")

# ============================================================
# STEP 7: Feature Importance
# ============================================================
print("\n[7] Analyzing feature importance...")

# Use KNN imputation model for feature importance
best_method = comparison_df['AUC-ROC'].idxmax()
best_model = results[best_method]['model']

feature_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 Top 10 Features ({best_method} imputation):")
print(feature_importance.head(10))

feature_importance.to_csv("outputs/tables/feature_importance.csv", index=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='steelblue', edgecolor='black')
plt.xlabel('Importance')
plt.title(f'Top 10 Features ({best_method} Imputation)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Saved: outputs/plots/feature_importance.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("PHASE 5 COMPLETE - SUMMARY")
print("="*60)
print(f"✅ Best imputation method: {best_method}")
print(f"✅ Best AUC-ROC: {comparison_df.loc[best_method, 'AUC-ROC']:.4f}")
print(f"✅ Best F1-Score: {comparison_df.loc[best_method, 'F1-Score']:.4f}")

print("\n✅ Phase 5 completed successfully!")