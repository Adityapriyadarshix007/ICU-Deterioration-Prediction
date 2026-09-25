#!/usr/bin/env python3
"""
Phase 6b: CNN-LSTM Training and Evaluation (RIGOROUS)

PURPOSE
-------
Train a 1D-CNN + LSTM + Attention model on 0-6h sequence data to predict
ΔSOFA ≥ 2 at 6-18h. Compare against the tabular tree-model baseline
from Phase 6 with a MATCHING evaluation protocol.

EVALUATION PROTOCOL
-------------------
Step 1: 5-fold CV (grid evaluation)
    - StratifiedKFold splits train into 5 × (80% train / 20% val).
    - Each fold: train on 80% (with internal 15% for early stopping),
      evaluate on the SKF val fold (true CV).
    - Reports mean ± std of CV metrics (matches Phase 6 Tiers 1/2).

Step 2: Ensemble evaluation (secondary)
    - Average test-set predictions across the 5 CV models.

Step 3: Single final model (primary comparison)
    - Train ONE model on 100% of train, using its own internal 15%
      validation split for early stopping.
    - NO fixed-epoch constraint.
    - Evaluate once on the test set.
    - THIS is the number that directly compares to Phase 6 Tier 4.

Step 4: Comparison
    - Tree single-model test metrics (from Phase 6)
    - DL single-model test metrics (from Step 3) ← primary
    - DL CV metrics (from Step 1)                ← secondary
    - DL ensemble test metrics (from Step 2)     ← secondary

ARCHITECTURE
------------
Input: (batch, T=12, C=40) — 12 timesteps × (20 values + 20 masks)
  Conv1D(64, k=2) → BN → ReLU
  Conv1D(32, k=2) → BN → ReLU
  LSTM(128, return_sequences=True)
  Attention(64)
  Dense(32, ReLU) → Dropout(0.3)
  Dense(1) → logits

OUTPUTS
-------
    outputs/models/phase6b/cnn_lstm_fold{0-4}.pt
    outputs/models/phase6b/cnn_lstm_final.pt
    outputs/tables/phase6b_cv_results.csv
    outputs/tables/phase6b_cv_summary.csv
    outputs/tables/phase6b_test_metrics_single.csv
    outputs/tables/phase6b_test_metrics_ensemble.csv
    outputs/tables/phase6b_test_predictions.csv
    outputs/tables/phase6b_report.txt
    outputs/figures/phase6b_training_curves.png
    outputs/figures/phase6b_comparison.png
"""

import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, accuracy_score, brier_score_loss,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, FIG_DIR, LOGS_DIR, MODEL_DIR, SEQUENCE_DIR,
    DEEP_LEARNING_PARAMS as P,
    RANDOM_STATE, N_FOLDS,
    FEATURE_WINDOW_HOURS, PREDICTION_START_HOUR,
    PREDICTION_END_HOUR, SOFA_CHANGE_THRESHOLD,
)


# ============================================================
# DEVICE + REPRODUCIBILITY
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def check_mps_bce():
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return True
    try:
        device = torch.device('mps')
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([P['pos_weight']], device=device)
        )
        logits = torch.randn(4, device=device)
        labels = torch.tensor([0., 1., 0., 1.], device=device)
        _ = criterion(logits, labels)
        return True
    except Exception:
        return False


def set_seeds(seed: int, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if deterministic and hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Could not enable deterministic mode: {e}"
            )


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase6b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 6b: CNN-LSTM (RIGOROUS EVALUATION)")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    print(f" Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
          f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    print(f" Sequence: T={P['sequence_length']} × C={P['n_channels']}")
    print(f" Step 1: 5-fold CV (evaluated on SKF val fold)")
    print(f" Step 2: Ensemble of 5 fold models (test set) [secondary]")
    print(f" Step 3: Single final model on 100% train (test set) [PRIMARY]")
    print(f" Batch size: {P['batch_size']}, LR: {P['learning_rate']}")
    print(f" Device: {get_device()}")
    print("=" * 80)
    print()


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        scores = self.v(torch.tanh(self.W(x)))
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class CNNLSTMAttention(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        n_channels = cfg['n_channels']
        conv_filters = cfg['conv_filters']
        kernel = cfg['conv_kernel_size']
        lstm_units = cfg['lstm_units']
        attn_units = cfg['attention_units']
        dense_units = cfg['dense_units']
        dropout = cfg['dropout_rate']

        self.conv1 = nn.Conv1d(n_channels, conv_filters[0], kernel,
                                padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(conv_filters[0])
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel,
                                padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(conv_filters[1])

        self.lstm = nn.LSTM(
            input_size=conv_filters[1],
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=False,
        )
        self.attention = AttentionPooling(lstm_units, attn_units)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        h = self.dropout(F.relu(self.fc1(attn_out)))
        logits = self.fc2(h).squeeze(-1)
        return logits


# ============================================================
# TRAINING / EVALUATION HELPERS
# ============================================================

def _build_loaders(X_tr, y_tr, X_val, y_val, X_eval=None, y_eval=None):
    loaders = {}
    train_ds = SequenceDataset(X_tr, y_tr)
    val_ds = SequenceDataset(X_val, y_val)
    loaders['train'] = DataLoader(
        train_ds, batch_size=P['batch_size'], shuffle=P['shuffle'],
        num_workers=P['num_workers'], pin_memory=P['pin_memory'],
    )
    loaders['val'] = DataLoader(
        val_ds, batch_size=P['batch_size'], shuffle=False,
        num_workers=P['num_workers'], pin_memory=P['pin_memory'],
    )
    if X_eval is not None:
        eval_ds = SequenceDataset(X_eval, y_eval)
        loaders['eval'] = DataLoader(
            eval_ds, batch_size=P['batch_size'], shuffle=False,
            num_workers=P['num_workers'], pin_memory=P['pin_memory'],
        )
    return loaders


def _train_one_epoch(model, loader, optimizer, criterion, device,
                     gradient_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if gradient_clip and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                       labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y_true, y_prob)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'specificity': float(spec),
        'npv': float(npv),
    }


# ============================================================
# CORE TRAINING LOOP
# ============================================================

def train_model(X_tr, y_tr, X_val, y_val,
                X_eval=None, y_eval=None,
                fixed_epochs=None, device='cpu', logger=None,
                tag="model"):
    """
    Train a model. Early stopping on val AUPRC (unless fixed_epochs given).
    Returns dict with model, best_state, best_epoch, best_val_auprc,
    history, and (optionally) eval predictions.
    """
    loaders = _build_loaders(X_tr, y_tr, X_val, y_val, X_eval, y_eval)

    model = CNNLSTMAttention(P).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([P['pos_weight']], device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=P['learning_rate'],
        weight_decay=P['weight_decay'],
    )

    best_val_auprc = -1.0
    best_epoch = -1
    best_state = None
    patience = 0
    history = {'epoch': [], 'train_loss': [], 'val_auprc': [], 'val_auroc': []}

    n_epochs = fixed_epochs if fixed_epochs is not None else P['epochs']

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_one_epoch(
            model, loaders['train'], optimizer, criterion,
            device, P['gradient_clip'],
        )
        val_probs, val_labels = _predict(model, loaders['val'], device)
        val_auprc = average_precision_score(val_labels, val_probs)
        val_auroc = roc_auc_score(val_labels, val_probs)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_auprc'].append(val_auprc)
        history['val_auroc'].append(val_auroc)

        improved = val_auprc > best_val_auprc
        if improved:
            best_val_auprc = val_auprc
            best_epoch = epoch
            patience = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1

        if fixed_epochs is None and patience >= P['early_stopping_patience']:
            if logger:
                logger.info(f"    [{tag}] Early stopping at epoch {epoch} "
                            f"(best={best_epoch}, AUPRC={best_val_auprc:.4f})")
            break

    # Defensive: ensure best_state was set
    if best_state is None:
        raise RuntimeError(f"[{tag}] No training epochs completed — "
                            f"best_state is None")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    out = {
        'model': model,
        'best_state': best_state,
        'best_epoch': best_epoch,
        'best_val_auprc': best_val_auprc,
        'history': history,
    }

    if X_eval is not None:
        probs, labels = _predict(model, loaders['eval'], device)
        out['eval_probs'] = probs
        out['eval_labels'] = labels

    return out


# ============================================================
# STEP 1: TRUE CV EVALUATION
# ============================================================

def step1_cv(X_train, y_train, device, logger):
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: TRUE 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=P['n_folds'], shuffle=True,
                          random_state=RANDOM_STATE)

    fold_results = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"\n  Fold {fold_idx+1}/{P['n_folds']}")

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
            fixed_epochs=None, device=device, logger=logger,
            tag=f"fold{fold_idx}",
        )

        fold_metrics = compute_metrics(result['eval_labels'],
                                        result['eval_probs'])
        fold_metrics['fold'] = fold_idx
        fold_metrics['best_epoch'] = result['best_epoch']
        fold_metrics['best_val_auprc'] = result['best_val_auprc']
        fold_results.append({
            'fold_idx': fold_idx,
            'metrics': fold_metrics,
            'result': result,
        })

        logger.info(f"    Fold {fold_idx+1} CV metrics:")
        for k in ['recall', 'auroc', 'auprc', 'precision', 'f1']:
            logger.info(f"      {k:12s}: {fold_metrics[k]:.4f}")

        ckpt_dir = Path(P['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'fold': fold_idx,
            'model_state_dict': result['best_state'],
            'best_epoch': result['best_epoch'],
            'best_val_auprc': result['best_val_auprc'],
            'cv_metrics': fold_metrics,
        }, ckpt_dir / f"cnn_lstm_fold{fold_idx}.pt")

    return fold_results


def summarize_cv(fold_results):
    rows = [fr['metrics'] for fr in fold_results]
    cv_df = pd.DataFrame(rows)

    metric_cols = [c for c in cv_df.columns
                    if c not in ('fold', 'best_epoch', 'best_val_auprc')]

    summary = {}
    for m in metric_cols:
        summary[m] = {
            'mean': float(cv_df[m].mean()),
            'std': float(cv_df[m].std()),
        }
    return cv_df, summary


# ============================================================
# STEP 2: ENSEMBLE ON TEST
# ============================================================

def step2_ensemble(fold_results, X_test, y_test, device, logger):
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: ENSEMBLE OF FOLD MODELS (test set) [secondary]")
    logger.info("=" * 60)

    all_probs = []
    for fr in fold_results:
        fold_idx = fr['fold_idx']
        model = CNNLSTMAttention(P).to(device)
        model.load_state_dict({
            k: v.to(device) for k, v in fr['result']['best_state'].items()
        })

        test_ds = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(
            test_ds, batch_size=P['batch_size'], shuffle=False,
            num_workers=P['num_workers'], pin_memory=P['pin_memory'],
        )
        probs, labels = _predict(model, test_loader, device)
        all_probs.append(probs)
        logger.info(f"  Fold {fold_idx+1} test AUPRC: "
                    f"{average_precision_score(labels, probs):.4f}")

    mean_probs = np.stack(all_probs, axis=0).mean(axis=0)
    ensemble_metrics = compute_metrics(y_test, mean_probs)

    logger.info(f"\n  Ensemble (5-model averaged) test metrics:")
    for k, v in ensemble_metrics.items():
        logger.info(f"    {k:12s}: {v:.4f}")

    return mean_probs, ensemble_metrics


# ============================================================
# STEP 3: SINGLE FINAL MODEL
# ============================================================

def step3_final_model(X_train, y_train, X_test, y_test, device, logger):
    """
    Train ONE model on 100% of train, using its own 15% internal val
    split for early stopping. NO fixed epoch constraint.

    This is the primary comparison model for Phase 6's tree model.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: SINGLE FINAL MODEL (100% train → test) [PRIMARY]")
    logger.info("=" * 60)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    logger.info(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  "
                f"Test: {len(X_test):,}")
    logger.info(f"  Training with early stopping (no fixed-epoch constraint)")

    result = train_model(
        X_tr, y_tr, X_val, y_val,
        X_eval=X_test, y_eval=y_test,
        fixed_epochs=None,       # ← USE EARLY STOPPING
        device=device, logger=logger, tag="final",
    )

    logger.info(f"  Final model best epoch: {result['best_epoch']} "
                f"(AUPRC={result['best_val_auprc']:.4f})")

    test_metrics = compute_metrics(result['eval_labels'],
                                    result['eval_probs'])
    logger.info(f"\n  Single-model test metrics:")
    for k, v in test_metrics.items():
        logger.info(f"    {k:12s}: {v:.4f}")

    ckpt_dir = Path(P['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': result['best_state'],
        'best_epoch': result['best_epoch'],
        'test_metrics': test_metrics,
    }, ckpt_dir / 'cnn_lstm_final.pt')

    return result, test_metrics


# ============================================================
# FIGURES
# ============================================================

def make_training_curves(fold_results, final_history, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fr in fold_results:
        h = fr['result']['history']
        axes[0].plot(h['epoch'], h['train_loss'],
                     label=f"Fold {fr['fold_idx']+1}", alpha=0.7)
        axes[1].plot(h['epoch'], h['val_auprc'],
                     label=f"Fold {fr['fold_idx']+1}", alpha=0.7)

    axes[0].plot(final_history['epoch'], final_history['train_loss'],
                 label='Final', color='black', linewidth=2)
    axes[1].plot(final_history['epoch'], final_history['val_auprc'],
                 label='Final', color='black', linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train loss')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val AUPRC')
    axes[1].set_title('Validation AUPRC')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'phase6b_training_curves.png'
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"  Figure: {out}")


def make_comparison_figure(tree_metrics, dl_single, dl_ensemble, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics_to_show = ['recall', 'auroc', 'auprc', 'precision', 'f1']
    tree_vals = [tree_metrics.get(m, 0) for m in metrics_to_show]
    single_vals = [dl_single.get(m, 0) for m in metrics_to_show]
    ensemble_vals = [dl_ensemble.get(m, 0) for m in metrics_to_show]

    x = np.arange(len(metrics_to_show))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, tree_vals, width,
           label='Tree (zero+LightGBM+masks)', color='steelblue')
    ax.bar(x, single_vals, width,
           label='CNN-LSTM (single) [primary]', color='crimson')
    ax.bar(x + width, ensemble_vals, width,
           label='CNN-LSTM (5-ensemble) [secondary]', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_show)
    ax.set_ylabel('Value')
    ax.set_title('Tree vs. CNN-LSTM: Test Set Comparison')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    for i, (t, s, e) in enumerate(zip(tree_vals, single_vals, ensemble_vals)):
        ax.text(i - width, t + 0.005, f'{t:.3f}', ha='center', fontsize=7)
        ax.text(i, s + 0.005, f'{s:.3f}', ha='center', fontsize=7)
        ax.text(i + width, e + 0.005, f'{e:.3f}', ha='center', fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / 'phase6b_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"  Figure: {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger = setup_logging()
    print_header()

    t0 = time.time()

    set_seeds(P['seed'], deterministic=P.get('deterministic', True))
    logger.info(f"  Seeds set (seed={P['seed']}, "
                f"deterministic={P.get('deterministic')})")

    # Load sequences
    logger.info("\n" + "=" * 60)
    logger.info("LOADING SEQUENCES FROM PHASE 3b")
    logger.info("=" * 60)

    train_path = SEQUENCE_DIR / 'sequences_train.npz'
    test_path = SEQUENCE_DIR / 'sequences_test.npz'

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Missing sequence files in {SEQUENCE_DIR}")
        return 1

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']
    stay_ids_test = test_data['stay_ids']

    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test:  {X_test.shape}")
    logger.info(f"  y_train: {int(y_train.sum())} pos / {len(y_train)}")
    logger.info(f"  y_test:  {int(y_test.sum())} pos / {len(y_test)}")

    assert X_train.shape[1] == P['sequence_length']
    assert X_train.shape[2] == P['n_channels']

    # Device
    device = get_device()
    logger.info(f"\n  Device: {device}")
    if device.type == 'mps':
        logger.info("  Checking MPS support for BCEWithLogitsLoss(pos_weight)...")
        if not check_mps_bce():
            logger.warning("  ⚠️  Falling back to CPU")
            device = torch.device('cpu')
        else:
            logger.info("  ✅ MPS supports BCEWithLogitsLoss")

    # STEP 1
    fold_results = step1_cv(X_train, y_train, device, logger)
    cv_df, cv_summary = summarize_cv(fold_results)

    cv_df.to_csv(TABLE_DIR / 'phase6b_cv_results.csv', index=False)
    cv_summary_rows = [{'metric': m, 'mean': v['mean'], 'std': v['std']}
                       for m, v in cv_summary.items()]
    pd.DataFrame(cv_summary_rows).to_csv(
        TABLE_DIR / 'phase6b_cv_summary.csv', index=False)

    logger.info("\n  CV SUMMARY (mean ± std across 5 folds):")
    for m, v in cv_summary.items():
        logger.info(f"    {m:12s}: {v['mean']:.4f} ± {v['std']:.4f}")

    best_epochs = [fr['metrics']['best_epoch'] for fr in fold_results]
    logger.info(f"\n  Best epochs across folds: {best_epochs}")

    # STEP 2
    ensemble_probs, ensemble_metrics = step2_ensemble(
        fold_results, X_test, y_test, device, logger
    )
    pd.DataFrame([ensemble_metrics]).to_csv(
        TABLE_DIR / 'phase6b_test_metrics_ensemble.csv', index=False)

    # STEP 3
    final_result, single_metrics = step3_final_model(
        X_train, y_train, X_test, y_test,
        device, logger,
    )
    pd.DataFrame([single_metrics]).to_csv(
        TABLE_DIR / 'phase6b_test_metrics_single.csv', index=False)

    # Predictions
    pred_df = pd.DataFrame({
        'stay_id': stay_ids_test,
        'y_true': y_test,
        'y_prob_cnn_lstm_single': final_result['eval_probs'],
        'y_prob_cnn_lstm_ensemble': ensemble_probs,
    })
    pred_df.to_csv(TABLE_DIR / 'phase6b_test_predictions.csv', index=False)
    logger.info(f"\n  Saved: phase6b_test_predictions.csv")

    # Figures
    make_training_curves(fold_results, final_result['history'], logger)

    tree_metrics_path = TABLE_DIR / 'phase6_test_metrics.csv'
    tree_metrics = {}
    if tree_metrics_path.exists():
        tree_metrics = pd.read_csv(tree_metrics_path).iloc[0].to_dict()
        logger.info(f"  Loaded tree metrics from phase6_test_metrics.csv")

    make_comparison_figure(tree_metrics, single_metrics,
                           ensemble_metrics, logger)

    # Report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PHASE 6b: CNN-LSTM REPORT (RIGOROUS EVALUATION)")
    report_lines.append("=" * 70)
    report_lines.append(f"Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    report_lines.append(f"Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
                        f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    report_lines.append(f"Sequence: T={P['sequence_length']} × C={P['n_channels']}")
    report_lines.append(f"Device: {device}")
    report_lines.append(f"Seed: {P['seed']}, deterministic={P.get('deterministic')}")
    report_lines.append("")

    report_lines.append("EVALUATION PROTOCOL:")
    report_lines.append("  Step 1: 5-fold CV (evaluated on SKF val fold)       [secondary]")
    report_lines.append("  Step 2: Ensemble of 5 fold models on test set       [secondary]")
    report_lines.append("  Step 3: Single model on 100% train → test set       [PRIMARY]")
    report_lines.append("")
    report_lines.append("  PRIMARY COMPARISON: Step 3 (single DL model) vs. tree model")
    report_lines.append("")

    report_lines.append("ARCHITECTURE:")
    report_lines.append(f"  Conv1D: {P['conv_filters']}, kernel={P['conv_kernel_size']}")
    report_lines.append(f"  LSTM: {P['lstm_units']} units")
    report_lines.append(f"  Attention: {P['attention_units']} units")
    report_lines.append(f"  Dense: {P['dense_units']} units, dropout={P['dropout_rate']}")
    report_lines.append("")

    report_lines.append("STEP 1 — TRUE CV METRICS (mean ± std across 5 folds):")
    for m, v in cv_summary.items():
        report_lines.append(f"  {m:12s}: {v['mean']:.4f} ± {v['std']:.4f}")
    report_lines.append("")

    report_lines.append("STEP 2 — ENSEMBLE TEST METRICS (5-model averaged):")
    for k, v in ensemble_metrics.items():
        report_lines.append(f"  {k:12s}: {v:.4f}")
    report_lines.append("")

    report_lines.append("STEP 3 — SINGLE FINAL MODEL TEST METRICS [PRIMARY]:")
    for k, v in single_metrics.items():
        report_lines.append(f"  {k:12s}: {v:.4f}")
    report_lines.append("")

    if tree_metrics:
        report_lines.append("STEP 4 — COMPARISON (single DL vs. tree):")
        report_lines.append(f"  {'metric':12s}  {'tree':>10s}  {'dl':>10s}  {'Δ':>10s}")
        for m in ['recall', 'auroc', 'auprc', 'precision', 'f1', 'brier']:
            t = tree_metrics.get(m)
            d = single_metrics.get(m)
            if t is not None and d is not None:
                report_lines.append(f"  {m:12s}  {t:>10.4f}  {d:>10.4f}  "
                                    f"{d - t:>+10.4f}")
        report_lines.append("")

    report_lines.append("Per-fold CV details:")
    report_lines.append(cv_df.to_string(index=False))

    report_path = TABLE_DIR / 'phase6b_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"  Saved: {report_path}")

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(" PHASE 6b COMPLETED")
    print("=" * 80)
    print(f" Elapsed: {elapsed/60:.1f} min")
    print("-" * 80)
    print(" STEP 1 — TRUE CV (mean ± std):")
    for m in ['recall', 'auroc', 'auprc', 'precision', 'f1']:
        v = cv_summary.get(m, {'mean': 0, 'std': 0})
        print(f"   {m:12s}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("-" * 80)
    print(" STEP 3 — SINGLE MODEL TEST [PRIMARY]:")
    for m in ['recall', 'auroc', 'auprc', 'precision', 'f1']:
        print(f"   {m:12s}: {single_metrics.get(m, 0):.4f}")
    print("-" * 80)
    print(" STEP 2 — ENSEMBLE TEST [secondary]:")
    for m in ['recall', 'auroc', 'auprc', 'precision', 'f1']:
        print(f"   {m:12s}: {ensemble_metrics.get(m, 0):.4f}")
    print("-" * 80)
    if tree_metrics:
        print(" TREE vs. DL (single model, primary comparison):")
        for m in ['recall', 'auroc', 'auprc']:
            t = tree_metrics.get(m, 0)
            d = single_metrics.get(m, 0)
            winner = "DL" if d > t else "Tree"
            print(f"   {m:12s}: tree={t:.4f}  dl={d:.4f}  → {winner}")
    print("-" * 80)
    print(" Outputs:")
    print(f"   {P['checkpoint_dir']}/cnn_lstm_fold*.pt")
    print(f"   {P['checkpoint_dir']}/cnn_lstm_final.pt")
    print(f"   {TABLE_DIR}/phase6b_cv_results.csv")
    print(f"   {TABLE_DIR}/phase6b_cv_summary.csv")
    print(f"   {TABLE_DIR}/phase6b_test_metrics_single.csv")
    print(f"   {TABLE_DIR}/phase6b_test_metrics_ensemble.csv")
    print(f"   {TABLE_DIR}/phase6b_test_predictions.csv")
    print(f"   {TABLE_DIR}/phase6b_report.txt")
    print(f"   {FIG_DIR}/phase6b_training_curves.png")
    print(f"   {FIG_DIR}/phase6b_comparison.png")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
