"""
Literature Comparison Table for Publication
"""

import pandas as pd

print("="*60)
print("LITERATURE COMPARISON TABLE")
print("="*60)

literature = pd.DataFrame({
    'Study': [
        'Churpek et al. (2016)',
        'Feng et al. (2021)',
        'Zhang et al. (2023)',
        'Huang et al. (2024)',
        'Your Work (2026)'
    ],
    'Dataset': [
        'MIMIC-III',
        'MIMIC-III',
        'MIMIC-IV',
        'eICU',
        'MIMIC-IV'
    ],
    'Model': [
        'Random Forest',
        'LSTM',
        'XGBoost',
        'Transformer',
        'CatBoost'
    ],
    'AUC-ROC': [
        0.82,
        0.79,
        0.74,
        0.76,
        '0.7013'
    ],
    'Features': [
        'Vitals + Labs',
        'Vitals + Labs',
        'Vitals + Labs + Demographics',
        'Vitals + Labs + Notes',
        '15 SHAP-selected features'
    ],
    'Key Finding': [
        'Random Forest for mortality prediction',
        'LSTM for sepsis prediction',
        'AKI prediction with XGBoost',
        'Transformer for ICU readmission',
        'CatBoost with SHAP explainability'
    ]
})

literature.to_csv('publication/table_literature_comparison.csv', index=False)

print("\n📊 Literature Comparison Table:")
print("-"*80)
print(literature.to_string(index=False))

print("\n✅ Saved: publication/table_literature_comparison.csv")
