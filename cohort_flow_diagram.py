"""
Cohort Flow Diagram for Publication
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 10))

# Define boxes
boxes = [
    {"text": "MIMIC-IV v3.1\n(All ICU Admissions)", "y": 0.90, "color": "#2E86AB"},
    {"text": "Adult ICU Patients\n(Age ≥ 18)", "y": 0.78, "color": "#3D85C6"},
    {"text": "ICU Stay ≥ 24 hours", "y": 0.66, "color": "#5B9BD5"},
    {"text": "Has SOFA measurements", "y": 0.54, "color": "#7BAFD4"},
    {"text": "No missing outcome labels", "y": 0.42, "color": "#9ECAE1"},
    {"text": "FINAL COHORT\nN = 57,515 ICU stays", "y": 0.28, "color": "#D4A373"},
    {"text": "Training Set\n80% (46,012 stays)", "y": 0.16, "color": "#52B788"},
    {"text": "Test Set\n20% (11,503 stays)", "y": 0.06, "color": "#40916C"},
]

# Draw boxes
for box in boxes:
    rect = FancyBboxPatch(
        (0.3, box["y"] - 0.04), 0.4, 0.07,
        boxstyle="round,pad=0.02",
        facecolor=box["color"],
        edgecolor="black",
        linewidth=1.5,
        transform=fig.transFigure
    )
    ax.add_patch(rect)
    
    # Add text
    ax.text(0.5, box["y"], box["text"],
            transform=fig.transFigure,
            ha='center', va='center',
            fontsize=10,
            fontweight='bold',
            color='white' if box["color"] != "#9ECAE1" and box["color"] != "#D4A373" else 'black')

# Draw arrows
for i in range(len(boxes) - 1):
    ax.annotate('', xy=(0.5, boxes[i+1]["y"] + 0.03),
                xytext=(0.5, boxes[i]["y"] - 0.04),
                xycoords='figure fraction',
                textcoords='figure fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Title
ax.text(0.5, 0.98, "Cohort Selection Flow Diagram",
        transform=fig.transFigure,
        ha='center', va='center',
        fontsize=14,
        fontweight='bold')

# Remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('publication/figures/cohort_flow_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Cohort flow diagram saved: publication/figures/cohort_flow_diagram.png")
