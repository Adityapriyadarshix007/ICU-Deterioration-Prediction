#!/usr/bin/env python3
"""
Recover missing serving artifacts:
  - winsor_lo.json / winsor_hi.json  (Phase 4 train quantiles 0.5/99.5)
  - medians.json                     (Phase 6 SimpleImputer('median'))
  - serving_pipeline.json            (self-contained serving spec)

Uses features_0_6h_none.csv + phase6_final_config.pkl + scaler_params.csv.
Deterministic (RANDOM_STATE=42, TEST_SIZE=0.15). No retraining.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).parent
TABLE_DIR = APP_DIR / "outputs" / "tables"
MODEL_DIR = APP_DIR / "outputs" / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.15
OUTLIER_LOW = 0.005
OUTLIER_HIGH = 0.995


def main():
    print("=" * 70)
    print(" RECOVERING SERVING ARTIFACTS")
    print("=" * 70)

    # ---- load inputs ----
    cfg = joblib.load(MODEL_DIR / "phase6_final_config.pkl")
    feature_cols = list(cfg["feature_cols"])
    mask_cols = list(cfg["kept_mask_cols"])

    scaler_df = pd.read_csv(TABLE_DIR / "scaler_params.csv")
    scaler_map = dict(zip(
        scaler_df["feature"],
        zip(scaler_df["mean"], scaler_df["scale"]),
    ))

    feats = pd.read_csv(TABLE_DIR / "features_0_6h_none.csv")
    print(f"Loaded features_0_6h_none.csv: {feats.shape}")

    if "outcome" not in feats.columns:
        raise RuntimeError("features file has no 'outcome' column")

    # ---- verify all 120 base features exist ----
    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        alt = {}
        for m in missing:
            cand = m.replace(" ", "_")
            if cand in feats.columns:
                alt[m] = cand
        still_missing = [m for m in missing if m not in alt]
        if still_missing:
            raise RuntimeError(f"Base features not found in raw file: {still_missing}")
        feats = feats.rename(columns={v: k for k, v in alt.items()})
        print(f"Renamed {len(alt)} underscore-variant columns to config form")

    # ---- reproduce Phase 4 train/test split ----
    X_all = feats[feature_cols].copy()
    y_all = feats["outcome"].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    print(f"Train: {len(X_train_raw):,}  Test: {len(X_test_raw):,}")

    # ---- winsorization bounds from TRAIN ----
    print("\nComputing winsorization bounds from train split...")
    winsor_lo = {}
    winsor_hi = {}
    for c in feature_cols:
        s = X_train_raw[c]
        if s.isnull().all() or s.nunique() <= 2:
            winsor_lo[c] = None
            winsor_hi[c] = None
        else:
            winsor_lo[c] = float(s.quantile(OUTLIER_LOW))
            winsor_hi[c] = float(s.quantile(OUTLIER_HIGH))

    # ---- apply winsorization (train + test) ----
    X_tr_w = X_train_raw.copy()
    X_te_w = X_test_raw.copy()
    for c in feature_cols:
        lo, hi = winsor_lo[c], winsor_hi[c]
        if lo is None:
            continue
        X_tr_w[c] = X_tr_w[c].clip(lo, hi)
        X_te_w[c] = X_te_w[c].clip(lo, hi)
    print("Winsorization applied to train and test")

    # ---- medians from winsorized TRAIN ----
    print("\nComputing medians from winsorized train split...")
    medians = {}
    for c in feature_cols:
        m = X_tr_w[c].median()
        if pd.isna(m):
            m = 0.0
        medians[c] = float(m)

    # ---- verify by reconstructing Phase 4's X_train ----
    print("\nVerifying reconstruction against X_train.csv...")
    X_tr_std = X_tr_w.copy()
    for c in feature_cols:
        if c in scaler_map:
            mean, scale = scaler_map[c]
            X_tr_std[c] = (X_tr_std[c] - mean) / scale

    masks = X_tr_w.isnull().astype(int)
    masks.columns = [f"{c}_missing" for c in masks.columns]

    X_reference = pd.read_csv(TABLE_DIR / "X_train.csv")
    stay_ids_tr = feats.loc[X_train_raw.index, "stay_id"].values

    ref_indexed = X_reference.set_index("stay_id")
    ref_base = ref_indexed.loc[stay_ids_tr, feature_cols].reset_index(drop=True)
    ref_masks = ref_indexed.loc[stay_ids_tr, mask_cols].reset_index(drop=True)

    diff_base = np.abs(X_tr_std.values - ref_base.values)
    diff_masks = np.abs(masks[mask_cols].values - ref_masks.values)

    print(f"  Base cols max abs diff : {np.nanmax(diff_base):.2e}")
    print(f"  Base cols mean abs diff: {np.nanmean(diff_base):.2e}")
    print(f"  Mask cols max abs diff : {np.nanmax(diff_masks):.2e}")
    def _nan_frac(df):
        num = df.select_dtypes(include="number").values
        return float(np.isnan(num).mean()) if num.size else 0.0
    print(f"  Base NaN fraction (ours): {_nan_frac(X_tr_std):.4f}")
    print(f"  Base NaN fraction (ref) : {_nan_frac(ref_base):.4f}")

    if np.nanmax(diff_base) < 1e-4 and np.nanmax(diff_masks) < 1e-9:
        print("  OK - reconstruction matches X_train.csv")
    else:
        print("  WARNING - reconstruction differs; inspect before deploying")

    # ---- persist ----
    json.dump(winsor_lo, open(MODEL_DIR / "winsor_lo.json", "w"), indent=2)
    json.dump(winsor_hi, open(MODEL_DIR / "winsor_hi.json", "w"), indent=2)
    json.dump(medians, open(MODEL_DIR / "medians.json", "w"), indent=2)
    print(f"\nWrote:")
    print(f"  {MODEL_DIR}/winsor_lo.json ({len(winsor_lo)} entries)")
    print(f"  {MODEL_DIR}/winsor_hi.json ({len(winsor_hi)} entries)")
    print(f"  {MODEL_DIR}/medians.json   ({len(medians)} entries)")

    pipeline = {
        "feature_cols": feature_cols,
        "mask_cols": mask_cols,
        "n_total": len(feature_cols) + len(mask_cols),
        "winsor_lo": winsor_lo,
        "winsor_hi": winsor_hi,
        "medians": medians,
        "scaler": {k: {"mean": v[0], "scale": v[1]}
                   for k, v in scaler_map.items()},
        "mask_suffix": "_missing",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "outlier_low": OUTLIER_LOW,
        "outlier_high": OUTLIER_HIGH,
    }
    json.dump(pipeline, open(MODEL_DIR / "serving_pipeline.json", "w"), indent=2)
    print(f"  {MODEL_DIR}/serving_pipeline.json (self-contained)")
    print("\n" + "=" * 70)
    print(" RECOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
