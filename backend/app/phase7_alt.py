#!/usr/bin/env python3
"""
Phase 7 ALT: Run Phase 7 analysis on the alternative model
(median + LightGBM + with_masks) without overwriting primary outputs.

Reads from:   outputs/models/phase6_alt/
Writes to:    outputs/tables/phase7_alt_*.csv
              outputs/figures/phase7_alt_*.png

Usage:
    python3 phase7_alt.py
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ------------------------------------------------------------
# Patch model paths BEFORE importing phase7
# ------------------------------------------------------------
import config
ALT_DIR = config.MODEL_DIR / 'phase6_alt'

# Overwrite the config's MODEL_DIR reference used by phase7
# by monkey-patching at the module level.
import phase7

# Patch the specific constants that phase7 reads
phase7.MODEL_DIR = ALT_DIR

# Patch output paths: rename phase7_* to phase7_alt_*
ALT_PREFIX = 'phase7_alt_'
_original_savefig = None

import matplotlib.pyplot as plt
_orig_savefig = plt.savefig


def _patched_savefig(fname, *args, **kwargs):
    """Prefix any phase7_ figure with phase7_alt_."""
    fname = Path(str(fname))
    if fname.name.startswith('phase7_') and not fname.name.startswith('phase7_alt_'):
        new_name = ALT_PREFIX + fname.name[len('phase7_'):]
        fname = fname.with_name(new_name)
    return _orig_savefig(fname, *args, **kwargs)


plt.savefig = _patched_savefig


# ------------------------------------------------------------
# Also patch DataFrame.to_csv calls inside phase7
# ------------------------------------------------------------
_orig_to_csv = None


def _patch_module_outputs():
    """
    Patch phase7 module to write to phase7_alt_* prefixes.
    This wraps phase7's functions with an output-renaming shim.
    """
    import pandas as pd
    original_to_csv = pd.DataFrame.to_csv

    def _patched_to_csv(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None:
            p = Path(str(path_or_buf))
            if p.name.startswith('phase7_') and not p.name.startswith('phase7_alt_'):
                p = p.with_name(ALT_PREFIX + p.name[len('phase7_'):])
                path_or_buf = str(p)
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    pd.DataFrame.to_csv = _patched_to_csv


# ------------------------------------------------------------
# Also patch the report file name
# ------------------------------------------------------------
def _patch_report_path():
    """phase7.save_report writes to TABLE_DIR/phase7_report.txt.
    Patch it by wrapping save_report."""
    original_save_report = phase7.save_report

    def _wrapped_save_report(*args, **kwargs):
        # Redirect by pointing at a modified TABLE_DIR is harder;
        # simply call the original and then rename.
        original_save_report(*args, **kwargs)
        src = phase7.TABLE_DIR / 'phase7_report.txt'
        dst = phase7.TABLE_DIR / 'phase7_alt_report.txt'
        if src.exists():
            shutil.move(str(src), str(dst))

    phase7.save_report = _wrapped_save_report


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("=" * 78)
    print(" PHASE 7 ALT: SHAP on median+LightGBM+with_masks model")
    print("=" * 78)
    print(f" Reading model from: {ALT_DIR}")
    print(f" Output files prefixed with: {ALT_PREFIX}")
    print("=" * 78)

    # Verify alt model exists
    model_path = ALT_DIR / 'phase6_alt_model.pkl'
    config_path = ALT_DIR / 'phase6_alt_config.pkl'
    if not model_path.exists() or not config_path.exists():
        print(f"❌ Missing alt model artifacts in {ALT_DIR}")
        print(f"   Run force_refit_median.py first.")
        return 1

    # Patch phase7's constants
    phase7.MODEL_DIR = ALT_DIR
    # Patch the filename constants used inside phase7.main()
    # phase7.py uses MODEL_DIR / 'phase6_final_model.pkl'
    # We need it to use 'phase6_alt_model.pkl' instead.
    # The cleanest way: monkey-patch the specific load call.

    original_joblib_load = phase7.joblib.load

    def _patched_load(path, *args, **kwargs):
        p = Path(str(path))
        if p.name == 'phase6_final_model.pkl':
            p = ALT_DIR / 'phase6_alt_model.pkl'
        elif p.name == 'phase6_final_config.pkl':
            p = ALT_DIR / 'phase6_alt_config.pkl'
        return original_joblib_load(str(p), *args, **kwargs)

    phase7.joblib.load = _patched_load

    # Patch output prefixes
    _patch_module_outputs()
    _patch_report_path()

    # Run Phase 7's main
    rc = phase7.main()

    print()
    print("=" * 78)
    print(" PHASE 7 ALT COMPLETED")
    print("=" * 78)
    print(f" Outputs: outputs/tables/phase7_alt_*.csv")
    print(f" Figures: outputs/figures/phase7_alt_*.png")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
