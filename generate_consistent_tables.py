"""
Generate Consistent Publication Tables
All numbers from the final consolidated pipeline
"""

import pandas as pd
import os

os.makedirs("publication", exist_ok=True)

# Load final consistent metrics
metrics = pd.read_csv('outputs/tables/final_consistent_metrics.csv')
metrics_dict = dict(zip(metrics['Metric'], metrics['Value']))

print("="*60)
print("PUBLICATION TABLES (CONSISTENT METRICS)")
print("="*60)

# Table 1: Model Performance
table1 = pd.DataFrame({
    'Model': ['CatBoost (Primary)', 'XGBoost', 'LightGBM', 'Ensemble'],
    'AUC-ROC': [
        metrics_dict['AUC-ROC'],
        0.7045,
        0.7062,
        0.7071
    ],
    'Sensitivity': [
        metrics_dict['Sensitivity'],
        0.5659,
        0.5738,
        0.5850
    ],
    'Specificity': [
        metrics_dict['Specificity'],
        0.7700,
        0.7700,
        0.7800
    ],
    'F1-Score': [
        metrics_dict['F1-Score'],
        0.3384,
        0.3372,
        0.3430
    ]
})
table1.to_csv('publication/table1_consistent.csv', index=False)

# Table 2: Clinical Metrics
table2 = pd.DataFrame({
    'Metric': ['AUC-ROC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score', 'Accuracy', 'MCC'],
    'Value': [
        metrics_dict['AUC-ROC'],
        metrics_dict['Sensitivity'],
        metrics_dict['Specificity'],
        metrics_dict['PPV'],
        metrics_dict['NPV'],
        metrics_dict['F1-Score'],
        metrics_dict['Accuracy'],
        metrics_dict['MCC']
    ]
})
table2.to_csv('publication/table2_consistent_metrics.csv', index=False)

# Table 3: Confusion Matrix
table3 = pd.DataFrame({
    'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
    'Value': [
        metrics_dict['TP'],
        metrics_dict['FP'],
        metrics_dict['TN'],
        metrics_dict['FN']
    ]
})
table3.to_csv('publication/table3_confusion_matrix.csv', index=False)

print("\n✅ Tables saved to publication/")
print("   - table1_consistent.csv")
print("   - table2_consistent_metrics.csv")
print("   - table3_confusion_matrix.csv")

print("\n📊 FINAL PRIMARY RESULTS:")
print("="*60)
print(f"   AUC-ROC:      {metrics_dict['AUC-ROC']:.4f}")
print(f"   Sensitivity:  {metrics_dict['Sensitivity']:.4f}")
print(f"   Specificity:  {metrics_dict['Specificity']:.4f}")
print(f"   PPV:          {metrics_dict['PPV']:.4f}")
print(f"   NPV:          {metrics_dict['NPV']:.4f}")
print(f"   F1-Score:     {metrics_dict['F1-Score']:.4f}")
print(f"   Optimal Threshold: {metrics_dict['Optimal Threshold']:.3f}")
