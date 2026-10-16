#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import TABLE_DIR, FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

tree = pd.read_csv(TABLE_DIR / 'phase7_threshold_source.csv').iloc[0]
dl = pd.read_csv(TABLE_DIR / 'phase7b_threshold_source.csv').iloc[0]

metric_cols = ['test_recall', 'test_precision', 'test_specificity', 'test_f1']
metric_labels = ['Recall', 'Precision', 'Specificity', 'F1']

tree_vals = [float(tree[m]) for m in metric_cols]
dl_vals = [float(dl[m]) for m in metric_cols]

print(f"Tree threshold: {float(tree['threshold']):.3f}")
print(f"DL   threshold: {float(dl['threshold']):.3f}")
print(f"Tree metrics: {tree_vals}")
print(f"DL   metrics: {dl_vals}")

x = np.arange(len(metric_cols))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bars_tree = ax.bar(x - width/2, tree_vals, width, label='LightGBM (tree)',
                    color='steelblue', edgecolor='black', linewidth=0.7)
bars_dl = ax.bar(x + width/2, dl_vals, width, label='CNN-LSTM',
                  color='crimson', edgecolor='black', linewidth=0.7)

for bars in (bars_tree, bars_dl):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel('Metric value', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Test-Set Discrimination at OOF-Selected Thresholds', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)
ax.text(0.02, 0.98,
        f"Tree threshold = {float(tree['threshold']):.2f}   |   "
        f"DL threshold = {float(dl['threshold']):.2f}",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_discrimination.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig5_discrimination.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIG_DIR / 'fig5_discrimination.pdf'}")
print(f"Saved: {FIG_DIR / 'fig5_discrimination.png'}")
