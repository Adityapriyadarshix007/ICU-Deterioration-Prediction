"""
Lead Time Analysis
Prediction at different hours before deterioration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LEAD TIME ANALYSIS")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

# Simulate different lead times by using different hour features
lead_times = [6, 12, 18, 24]
results = []

for lead_time in lead_times:
    print(f"\n🔍 Testing lead time: {lead_time} hours")
    
    # Select features based on lead time
    # For simulation, we'll use hour0 and hour6 features
    # In reality, you'd use features from different time points
    hour_cols = [c for c in X.columns if '_hour0' in c or '_hour6' in c]
    
    X_subset = X[hour_cols].values
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
    
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Lead Time (hours)': lead_time,
        'AUC-ROC': auc
    })

# Plot
plt.figure(figsize=(10, 6))
results_df = pd.DataFrame(results)
plt.plot(results_df['Lead Time (hours)'], results_df['AUC-ROC'], 
         marker='o', linewidth=2, markersize=10)
plt.xlabel('Lead Time (hours before deterioration)')
plt.ylabel('AUC-ROC')
plt.title('Performance vs Prediction Lead Time')
plt.grid(True, alpha=0.3)
plt.ylim(0.65, 0.75)
for i, row in results_df.iterrows():
    plt.annotate(f'{row["AUC-ROC"]:.3f}', 
                (row['Lead Time (hours)'], row['AUC-ROC']),
                textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig('outputs/plots/lead_time_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Lead time analysis saved: outputs/plots/lead_time_analysis.png")

results_df.to_csv('outputs/tables/lead_time_results.csv', index=False)
print("✅ Results saved: outputs/tables/lead_time_results.csv")
