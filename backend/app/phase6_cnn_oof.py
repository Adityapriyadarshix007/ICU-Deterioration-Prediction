#!/usr/bin/env python3
"""
Standalone OOF generator for the CNN-LSTM (Phase 6b).
Reuses saved fold checkpoints — no retraining.

Reads:
    outputs/models/phase6b/cnn_lstm_fold{0-4}.pt
    outputs/sequences/sequences_train.npz
    outputs/tables/X_train.csv   (for stay_id alignment check)

Writes:
    outputs/tables/phase6b_oof_train_predictions.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, SEQUENCE_DIR, RANDOM_STATE,
    DEEP_LEARNING_PARAMS as P,
)
from phase6b_cnn_lstm import (
    get_device, set_seeds, SequenceDataset, CNNLSTMAttention,
    _predict, check_mps_bce,
)


def main():
    print("=" * 70)
    print(" PHASE 6b OOF GENERATOR (CNN-LSTM, reuses checkpoints)")
    print("=" * 70)

    set_seeds(P['seed'], deterministic=P.get('deterministic', True))

    # ---- 1. Load training sequences ----
    train_path = SEQUENCE_DIR / 'sequences_train.npz'
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run Phase 3b first.")
        return 1

    train_data = np.load(train_path)
    X_train = train_data['X']
    y_train = train_data['y']
    seq_stay_ids = train_data['stay_ids']

    print(f"\nX_train: {X_train.shape}")
    print(f"y_train: {int(y_train.sum())} pos / {len(y_train)}")

    # ---- 2. Sanity check: sequences align with X_train.csv ----
    tab_ids = pd.read_csv(TABLE_DIR / 'X_train.csv',
                          usecols=['stay_id'])['stay_id'].values
    if len(tab_ids) != len(seq_stay_ids):
        print(f"WARNING: X_train.csv has {len(tab_ids)} rows, "
              f"sequences have {len(seq_stay_ids)}.")
    else:
        match = (tab_ids == seq_stay_ids).all()
        print(f"Row-order match with X_train.csv: {match}")
        if not match:
            print("  (Set match: "
                  f"{set(tab_ids) == set(seq_stay_ids)})")

    # ---- 3. Device ----
    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == 'mps' and not check_mps_bce():
        print("  MPS BCE failed -> using CPU")
        device = torch.device('cpu')

    # ---- 4. Same 5-fold splits as Phase 6b CV ----
    skf = StratifiedKFold(
        n_splits=P['n_folds'],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probs = np.full(len(X_train), np.nan, dtype=float)
    ckpt_dir = Path(P['checkpoint_dir'])

    # ---- 5. For each fold: load checkpoint, predict on val fold ----
    for fold_idx, (_, val_idx) in enumerate(skf.split(X_train, y_train)):
        ckpt_path = ckpt_dir / f'cnn_lstm_fold{fold_idx}.pt'
        if not ckpt_path.exists():
            print(f"ERROR: Missing checkpoint {ckpt_path}")
            print("       Re-run Phase 6b first.")
            return 1

        ckpt = torch.load(ckpt_path, map_location=device,
                          weights_only=False)

        model = CNNLSTMAttention(P).to(device)
        model.load_state_dict({
            k: v.to(device) for k, v in ckpt['model_state_dict'].items()
        })
        model.eval()

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        ds = SequenceDataset(X_val, y_val)
        loader = DataLoader(
            ds,
            batch_size=P['batch_size'],
            shuffle=False,
            num_workers=P['num_workers'],
            pin_memory=P['pin_memory'],
        )

        probs, _ = _predict(model, loader, device)
        oof_probs[val_idx] = probs

        fold_auprc = average_precision_score(y_val, probs)
        print(f"  Fold {fold_idx + 1}/{P['n_folds']}: "
              f"AUPRC={fold_auprc:.4f}  "
              f"(best_epoch={ckpt.get('best_epoch', '?')})")

    # ---- 6. Verify ----
    if np.isnan(oof_probs).any():
        print(f"ERROR: {int(np.isnan(oof_probs).sum())} OOF NaN")
        return 1

    # ---- 7. Save ----
    oof_df = pd.DataFrame({
        'stay_id': seq_stay_ids,
        'y_true': y_train,
        'y_prob_oof': oof_probs,
    })
    out_path = TABLE_DIR / 'phase6b_oof_train_predictions.csv'
    oof_df.to_csv(out_path, index=False)

    # ---- 8. Report ----
    oof_auprc = average_precision_score(y_train, oof_probs)
    oof_auroc = roc_auc_score(y_train, oof_probs)

    print("\n" + "=" * 70)
    print(" OOF COMPLETE (CNN-LSTM)")
    print("=" * 70)
    print(f"  n:          {len(oof_probs):,}")
    print(f"  OOF AUPRC:  {oof_auprc:.4f}")
    print(f"  OOF AUROC:  {oof_auroc:.4f}")
    print(f"  Saved:      {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
