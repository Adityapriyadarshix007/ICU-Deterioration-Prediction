#!/usr/bin/env python3
"""
Phase 6b Experiments: Mask ablation + LR/simplification variant.

Runs 3 CNN-LSTM variants sequentially, evaluates each with true 5-fold CV,
and reports test metrics for each. Uses CV AUPRC (never test) to rank.

Variants:
    A) baseline    — current config
    B) no_masks    — masks zeroed out (input channel 20-39 set to 0)
    C) simplified  — LR=3e-4, LSTM=64, dropout=0.5
"""

import sys
import time
import json
import logging
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, LOGS_DIR, SEQUENCE_DIR,
    DEEP_LEARNING_PARAMS as P_BASE,
    RANDOM_STATE,
)

# Import the training + evaluation components from phase6b
from phase6b_cnn_lstm import (
    get_device, check_mps_bce, set_seeds, setup_logging,
    SequenceDataset, CNNLSTMAttention,
    train_model, compute_metrics,
    step1_cv, summarize_cv, step2_ensemble, step3_final_model,
)

from torch.utils.data import DataLoader


# ============================================================
# VARIANTS
# ============================================================

def make_variant(name):
    """Return (params_dict, transform_fn) for the named variant."""
    params = deepcopy(P_BASE)

    if name == "baseline":
        return params, None

    if name == "no_masks":
        # Zero out mask channels (indices 20:40 in the last axis)
        def transform(X):
            X = X.copy()
            X[:, :, params['n_features']:] = 0.0
            return X
        return params, transform

    if name == "simplified":
        params['learning_rate'] = 3e-4
        params['lstm_units'] = 64
        params['dropout_rate'] = 0.5
        return params, None

    raise ValueError(f"Unknown variant: {name}")


# ============================================================
# RUN ONE VARIANT
# ============================================================

def run_variant(name, X_train_raw, y_train, X_test_raw, y_test,
                device, logger):
    """Run the 3-step protocol for one variant."""

    params, transform = make_variant(name)

    # Monkey-patch the imported module's P to point at variant params
    import phase6b_cnn_lstm as p6b
    p6b.P = params

    logger.info(f"\n{'#'*70}")
    logger.info(f"# VARIANT: {name}")
    logger.info(f"# Params: LR={params['learning_rate']}, "
                f"LSTM={params['lstm_units']}, "
                f"dropout={params['dropout_rate']}, "
                f"masks_zeroed={name == 'no_masks'}")
    logger.info(f"{'#'*70}")

    # Apply transform
    if transform is not None:
        X_train = transform(X_train_raw)
        X_test = transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    # --- Step 1: CV ---
    set_seeds(params['seed'], deterministic=params.get('deterministic', True))

    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True,
                          random_state=RANDOM_STATE)

    fold_metrics = []
    fold_models_state = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_fold = X_train[tr_idx]
        y_tr_fold = y_train[tr_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
            X_tr_fold, y_tr_fold, test_size=0.15,
            stratify=y_tr_fold, random_state=RANDOM_STATE,
        )

        result = train_model(
            X_tr2, y_tr2, X_val2, y_val2,
            X_eval=X_val_fold, y_eval=y_val_fold,
            fixed_epochs=None, device=device, logger=None,
            tag=f"{name}_fold{fold_idx}",
        )

        fm = compute_metrics(result['eval_labels'], result['eval_probs'])
        fm['fold'] = fold_idx
        fm['best_epoch'] = result['best_epoch']
        fold_metrics.append(fm)
        fold_models_state.append(result['best_state'])

        logger.info(f"  [{name}] fold {fold_idx+1}: "
                    f"AUPRC={fm['auprc']:.4f}  AUROC={fm['auroc']:.4f}  "
                    f"best_epoch={result['best_epoch']}")

    cv_df = pd.DataFrame(fold_metrics)
    cv_summary = {
        'cv_auprc_mean': float(cv_df['auprc'].mean()),
        'cv_auprc_std': float(cv_df['auprc'].std()),
        'cv_auroc_mean': float(cv_df['auroc'].mean()),
        'cv_auroc_std': float(cv_df['auroc'].std()),
        'cv_recall_mean': float(cv_df['recall'].mean()),
    }

    # --- Step 2: Ensemble on test (secondary) ---
    all_probs = []
    for state in fold_models_state:
        m = CNNLSTMAttention(params).to(device)
        m.load_state_dict({k: v.to(device) for k, v in state.items()})
        test_ds = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(
            test_ds, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], pin_memory=params['pin_memory'],
        )
        m.eval()
        probs_list = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(device)
                probs_list.append(torch.sigmoid(m(Xb)).cpu().numpy())
        all_probs.append(np.concatenate(probs_list))

    ensemble_probs = np.stack(all_probs, axis=0).mean(axis=0)
    ensemble_metrics = compute_metrics(y_test, ensemble_probs)

    # --- Step 3: Single final model (primary) ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    final_result = train_model(
        X_tr, y_tr, X_val, y_val,
        X_eval=X_test, y_eval=y_test,
        fixed_epochs=None, device=device, logger=None,
        tag=f"{name}_final",
    )
    single_metrics = compute_metrics(final_result['eval_labels'],
                                     final_result['eval_probs'])

    logger.info(f"  [{name}] Step 2 ensemble: AUPRC={ensemble_metrics['auprc']:.4f}")
    logger.info(f"  [{name}] Step 3 single:   AUPRC={single_metrics['auprc']:.4f}")

    return {
        'variant': name,
        **cv_summary,
        'ensemble_test_auprc': ensemble_metrics['auprc'],
        'ensemble_test_auroc': ensemble_metrics['auroc'],
        'single_test_auprc': single_metrics['auprc'],
        'single_test_auroc': single_metrics['auroc'],
        'single_test_recall': single_metrics['recall'],
        'single_test_precision': single_metrics['precision'],
        'single_test_f1': single_metrics['f1'],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info(" PHASE 6b: 3-VARIANT EXPERIMENTS")
    logger.info("=" * 80)

    t0 = time.time()

    # Load sequences once
    train_data = np.load(SEQUENCE_DIR / 'sequences_train.npz')
    test_data = np.load(SEQUENCE_DIR / 'sequences_test.npz')

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test:  {X_test.shape}")

    # Device
    device = get_device()
    logger.info(f"Device: {device}")
    if device.type == 'mps' and not check_mps_bce():
        logger.warning("MPS BCE failed → falling back to CPU")
        device = torch.device('cpu')

    # Run variants
    variants = ['baseline', 'no_masks', 'simplified']
    results = []
    for name in variants:
        try:
            res = run_variant(name, X_train, y_train, X_test, y_test,
                              device, logger)
            results.append(res)
        except Exception as e:
            logger.error(f"Variant {name} FAILED: {e}", exc_info=True)

    # Save results
    out_df = pd.DataFrame(results)
    out_path = TABLE_DIR / 'phase6b_experiments_results.csv'
    out_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(" EXPERIMENT SUMMARY (ranked by CV AUPRC)")
    logger.info("=" * 80)
    ranked = out_df.sort_values('cv_auprc_mean', ascending=False)
    logger.info(ranked[['variant', 'cv_auprc_mean', 'cv_auprc_std',
                        'single_test_auprc', 'single_test_auroc',
                        'single_test_recall']].to_string(index=False))

    # Winner (by CV AUPRC — never test!)
    winner = ranked.iloc[0]['variant']
    logger.info(f"\n WINNER (by CV AUPRC): {winner}")
    logger.info(f" Test AUPRC: {ranked.iloc[0]['single_test_auprc']:.4f}")
    logger.info(f" Test AUROC: {ranked.iloc[0]['single_test_auroc']:.4f}")

    # Compare to tree
    tree_path = TABLE_DIR / 'phase6_test_metrics.csv'
    if tree_path.exists():
        tree = pd.read_csv(tree_path).iloc[0].to_dict()
        logger.info(f"\n TREE BASELINE: AUROC={tree['auroc']:.4f}  "
                    f"AUPRC={tree['auprc']:.4f}  Recall={tree['recall']:.4f}")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
