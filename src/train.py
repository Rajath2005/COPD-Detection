"""
train.py — Training script for COPD-Detection models
=====================================================
Trains either the Baseline CNN (binary) or the Hybrid CNN-LSTM (4-class).
Handles class imbalance with weighted cross-entropy loss.
Saves best model checkpoint by validation loss.

Usage (PowerShell):
    # Train baseline CNN (binary)
    python src/train.py --model baseline_cnn --num-classes 2

    # Train hybrid CNN-LSTM (4-class severity)
    python src/train.py --model cnn_lstm --num-classes 4

    # Resume interrupted training
    python src/train.py --model cnn_lstm --resume models/cnn_lstm/last_model.pth

Author: VCET COPD-Detection Team
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

# ── Make src/models importable ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.baseline_cnn import build_baseline_cnn
from models.cnn_lstm      import build_cnn_lstm

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════

class COPDDataset(Dataset):
    """
    PyTorch Dataset wrapping the pre-extracted numpy feature arrays.

    For the baseline CNN  (num_classes=2):
        Labels 0 (Normal) stay 0; labels 1,2,3 (any pathology) → 1
    For the CNN-LSTM      (num_classes=4):
        Labels used as-is (0=Normal, 1=Crackle, 2=Wheeze, 3=Both)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 num_classes: int = 4, single_channel: bool = False):
        self.X = torch.from_numpy(X).float()
        raw_y = y.copy()

        if num_classes == 2:
            # Binary: Normal=0, any pathology=1
            raw_y = (raw_y > 0).astype(np.int64)

        self.y = torch.from_numpy(raw_y).long()
        self.single_channel = single_channel

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                    # (3, 128, 128)
        if self.single_channel:
            x = x[0:1]                    # take only Mel channel → (1, 128, 128)
        return x, self.y[idx]


# ══════════════════════════════════════════════
# Class-balanced sampler
# ══════════════════════════════════════════════

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that upsamples minority classes
    so every class is seen equally often per epoch.
    """
    classes      = np.unique(labels)
    class_weights = compute_class_weight("balanced", classes=classes, y=labels)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True,
    )
    return sampler


# ══════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════

def compute_class_weights_tensor(labels: np.ndarray,
                                 num_classes: int,
                                 device: torch.device) -> torch.Tensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits      = model(X_batch)
        loss        = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ══════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════

def train(args):
    # ── Reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load feature arrays ───────────────────────────────────────────────
    feat_dir = Path(args.features_dir)
    log.info(f"Loading features from {feat_dir} ...")

    X_train = np.load(feat_dir / "X_train.npy")
    y_train = np.load(feat_dir / "y_train.npy")
    X_val   = np.load(feat_dir / "X_val.npy")
    y_val   = np.load(feat_dir / "y_val.npy")

    log.info(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

    # ── Determine if baseline CNN uses single channel ─────────────────────
    single_ch = (args.model == "baseline_cnn")

    # ── Binary label conversion (for display) ─────────────────────────────
    if args.num_classes == 2:
        y_train_disp = (y_train > 0).astype(np.int64)
    else:
        y_train_disp = y_train

    # ── Datasets & loaders ───────────────────────────────────────────────
    train_ds = COPDDataset(X_train, y_train,
                           num_classes=args.num_classes,
                           single_channel=single_ch)
    val_ds   = COPDDataset(X_val,   y_val,
                           num_classes=args.num_classes,
                           single_channel=single_ch)

    sampler     = make_weighted_sampler(y_train_disp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    if args.model == "baseline_cnn":
        model = build_baseline_cnn(num_classes=args.num_classes)
    elif args.model == "cnn_lstm":
        model = build_cnn_lstm(
            num_classes   = args.num_classes,
            lstm_hidden   = args.lstm_hidden,
            lstm_layers   = args.lstm_layers,
            bidirectional = True,
            dropout       = 0.4,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    # ── Loss with class weights ───────────────────────────────────────────
    class_weights = compute_class_weights_tensor(y_train_disp, args.num_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    log.info(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        log.info(f"Resumed from epoch {ckpt['epoch']}  (best val loss: {best_val_loss:.4f})")

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training CSV log ──────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    # ══════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ══════════════════════════════════════════════════════════════════════
    patience_counter = 0
    log.info(f"Starting training: {args.model}  |  {args.epochs} epochs  |  device: {device}")
    log.info("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        # Scheduler step
        if args.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        elapsed = time.time() - t0

        log.info(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:5.1f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:5.1f}%  |  "
            f"LR: {current_lr:.2e}  |  {elapsed:.1f}s"
        )

        # Save CSV log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f},{current_lr:.2e}\n")

        # Save last checkpoint (always)
        last_ckpt = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "best_val_loss":  best_val_loss,
            "args":           vars(args),
        }
        torch.save(last_ckpt, out_dir / "last_model.pth")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(last_ckpt, out_dir / "best_model.pth")
            log.info(f"  ✅ Best model saved  (val loss: {best_val_loss:.4f}  acc: {val_acc*100:.1f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            log.info(f"Early stopping triggered at epoch {epoch} "
                     f"(no improvement for {args.early_stop_patience} epochs)")
            break

    log.info("-" * 72)
    log.info(f"Training complete.  Best val loss: {best_val_loss:.4f}")
    log.info(f"Models saved to: {out_dir}")
    log.info(f"Now run:  python src/evaluate.py --model-path {out_dir}/best_model.pth "
             f"--model-type {args.model} --num-classes {args.num_classes}")


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train COPD detection models")

    parser.add_argument("--model",         type=str,   default="baseline_cnn",
                        choices=["baseline_cnn", "cnn_lstm"])
    parser.add_argument("--features-dir",  type=str,   default="data/features")
    parser.add_argument("--num-classes",   type=int,   default=2)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--lr-scheduler",  type=str,   default="plateau",
                        choices=["plateau", "cosine"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lstm-hidden",   type=int,   default=128)
    parser.add_argument("--lstm-layers",   type=int,   default=2)
    parser.add_argument("--out-dir",       type=str,   default="models/baseline_cnn")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint .pth to resume training")
    parser.add_argument("--seed",          type=int,   default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("COPD-Detection — Training")
    log.info(f"  Model         : {args.model}")
    log.info(f"  Classes       : {args.num_classes}")
    log.info(f"  Epochs        : {args.epochs}")
    log.info(f"  Batch size    : {args.batch_size}")
    log.info(f"  Learning rate : {args.lr}")
    log.info(f"  LR scheduler  : {args.lr_scheduler}")
    log.info(f"  Output dir    : {args.out_dir}")
    log.info("")

    train(args)