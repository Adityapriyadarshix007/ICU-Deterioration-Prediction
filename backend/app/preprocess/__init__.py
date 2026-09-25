"""
Preprocessing module for MIMIC-IV cohort construction and feature extraction
UPDATED: Pure ΔSOFA Target (Organ Dysfunction Progression)
NO DATA LEAKAGE: 0-6h Features → 6-18h Outcome
"""

from .cohort import CohortBuilder

__all__ = ['CohortBuilder']

# Module metadata
__version__ = '1.0.0'
__author__ = 'Aditya Priyadarshi'
__description__ = 'MIMIC-IV cohort construction for organ dysfunction progression prediction'
