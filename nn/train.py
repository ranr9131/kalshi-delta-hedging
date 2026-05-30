"""
Train the time-series win predictor on the dataset built by build_dataset.py.

Time-based train/val split (no shuffling — would leak future into past).
Best model by val log-loss is saved to nn/checkpoints/best.pt.

Usage:
    python3 nn/build_dataset.py
    python3 nn/train.py
"""

import os
import sys
import json
import time
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
# Output filenames are picked at runtime based on --multi flag.

VAL_FRAC    = 0.20
BATCH_SIZE  = 128
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EPOCHS      = 60
PATIENCE    = 8
SEED        = 0

# Single-mode decision minute. Multi-mode trains with a random decision
# minute in [MULTI_MIN_DEC, MULTI_MAX_DEC] per example, so one model can
# predict from any partial sequence in that range.
DECISION_MINUTE = 10
MULTI_MIN_DEC   = 4
MULTI_MAX_DEC   = 13


class TruncatedDataset(Dataset):
    """Yields (X, mask, y) with X/mask truncated to a decision minute.

    If `random_truncation=True`, pick a random minute in
    [MULTI_MIN_DEC, MULTI_MAX_DEC] per __getitem__. Otherwise, fixed at
    DECISION_MINUTE (legacy behavior).
    """

    def __init__(self, X, mask, y, random_truncation: bool):
        self.X = X.copy(); self.mask = mask.copy(); self.y = y
        self.random = random_truncation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx].copy()
        m = self.mask[idx].copy()
        if self.random:
            dec = random.randint(MULTI_MIN_DEC, MULTI_MAX_DEC)
        else:
            dec = DECISION_MINUTE
        X[dec + 1:, :] = 0.0
        m[dec + 1:]    = False
        return (
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(m),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_dataset(multi: bool):
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts)
    X, mask, y, ts = X[order], mask[order], y[order], ts[order]

    n = len(y)
    n_val = int(n * VAL_FRAC); n_train = n - n_val
    print(f"  Total: {n}  Train: {n_train}  Val: {n_val}")
    if multi:
        print(f"  Random truncation per example in T+[{MULTI_MIN_DEC}..{MULTI_MAX_DEC}]")
    else:
        print(f"  Fixed truncation at T+{DECISION_MINUTE}")

    # Compute normalization stats on train slice with EARLIEST truncation
    # (most missing data) to be conservative — features past MULTI_MIN_DEC
    # may be zeroed in some examples.
    if multi:
        Xn = X[:n_train].copy()
        Xn[:, MULTI_MIN_DEC + 1:, :] = 0.0
    else:
        Xn = X[:n_train].copy()
        Xn[:, DECISION_MINUTE + 1:, :] = 0.0
    flat = Xn.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32) + 1e-6
    print(f"  Feature means: {mean}")
    print(f"  Feature stds:  {std}")

    # Apply normalization to raw (untruncated) data; the Dataset class will
    # apply per-example truncation at __getitem__.
    Xnorm = ((X - mean) / std).astype(np.float32)

    train_ds = TruncatedDataset(Xnorm[:n_train], mask[:n_train],
                                 y[:n_train], random_truncation=multi)
    val_ds   = TruncatedDataset(Xnorm[n_train:], mask[n_train:],
                                 y[n_train:], random_truncation=False)
    return train_ds, val_ds, mean, std


def run_epoch(model, loader, criterion, device, opt=None):
    train = opt is not None
    model.train(train)
    total_loss = 0.0; total_n = 0; correct = 0
    for X, mk, y in loader:
        X, mk, y = X.to(device), mk.to(device), y.to(device)
        if train: opt.zero_grad()
        logits = model(X, mk)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item() * len(y)
        total_n    += len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
    return total_loss / total_n, correct / total_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multi", action="store_true",
                   help=f"Random truncation in T+[{MULTI_MIN_DEC}..{MULTI_MAX_DEC}] "
                        f"per example. Writes best_multi.pt instead of best.pt.")
    args = p.parse_args()

    ckpt_path  = os.path.join(CKPT_DIR, "best_multi.pt" if args.multi else "best.pt")
    stats_path = os.path.join(CKPT_DIR, "train_stats_multi.json" if args.multi else "train_stats.json")

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = pick_device()
    print(f"Device: {device}")
    print(f"Mode:   {'multi (random truncation)' if args.multi else 'single (fixed T+10)'}")
    print(f"Output: {ckpt_path}")

    train_ds, val_ds, mean, std = load_dataset(args.multi)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Sample one batch to find feature dim
    sample_X, _, _ = next(iter(train_loader))
    model = TSWinPredictor(n_features=sample_X.shape[-1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf"); best_epoch = -1; patience = 0
    history = []
    print(f"{'epoch':>5} {'tr_loss':>9} {'tr_acc':>8} {'va_loss':>9} {'va_acc':>8} {'time':>6}")
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "va_loss": va_loss, "va_acc": va_acc})
        print(f"{epoch:5d} {tr_loss:9.4f} {tr_acc:8.4f} {va_loss:9.4f} {va_acc:8.4f} "
              f"{time.time()-t0:5.1f}s")
        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss; best_epoch = epoch; patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "n_features": sample_X.shape[-1],
                "feature_mean": mean.tolist(),
                "feature_std":  std.tolist(),
                "epoch": epoch,
                "val_loss": va_loss,
                "multi": args.multi,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stop at epoch {epoch} (best {best_epoch})")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"Best checkpoint: {ckpt_path}")
    with open(stats_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "n_params": n_params,
            "n_train": len(train_ds), "n_val": len(val_ds),
            "multi": args.multi,
            "feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
