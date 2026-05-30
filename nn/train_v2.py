"""
v2 trainer — bigger transformer, 14-feature dataset, stronger regularization,
longer training. Output checkpoint: nn/checkpoints/best_v2.pt.

Architecture: 4 layers, d_model=48, n_heads=6, ff=128, dropout=0.2
Params: ~55k (vs v1's 17k)

Usage:
    python3 nn/build_dataset_v2.py    # build dataset_v2.npz
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 nn/train_v2.py
"""

import os, sys, json, time, math, random, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset_v2.npz")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
CKPT_PATH  = os.path.join(CKPT_DIR, "best_v2.pt")
STATS_PATH = os.path.join(CKPT_DIR, "train_stats_v2.json")

VAL_FRAC    = 0.20
BATCH_SIZE  = 128
LR          = 1e-3
WEIGHT_DECAY= 5e-4
EPOCHS      = 80
PATIENCE    = 12
SEED        = 0

# Architecture (bigger than v1)
D_MODEL     = 48
N_HEADS     = 6
N_LAYERS    = 4
DIM_FF      = 128
DROPOUT     = 0.2

# Random truncation range — matches v1 multi-mode
MIN_DEC     = 4
MAX_DEC     = 13


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class TruncDataset(Dataset):
    def __init__(self, X, mask, y, random_trunc=True, fixed_dec=10):
        self.X, self.mask, self.y = X, mask, y
        self.random_trunc = random_trunc
        self.fixed_dec = fixed_dec
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        X = self.X[idx].copy(); m = self.mask[idx].copy()
        dec = random.randint(MIN_DEC, MAX_DEC) if self.random_trunc else self.fixed_dec
        X[dec+1:, :] = 0.0
        m[dec+1:]    = False
        return (torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(m),
                torch.tensor(self.y[idx], dtype=torch.float32))


def load_dataset():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y); n_val = int(n * VAL_FRAC); n_train = n - n_val
    print(f"  Total: {n}  Train: {n_train}  Val: {n_val}")

    # Stats computed with earliest truncation (most missing data) — conservative
    Xn = X[:n_train].copy()
    Xn[:, MIN_DEC+1:, :] = 0.0
    flat = Xn.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32) + 1e-6
    print(f"  Feature means: {[f'{m:+.3f}' for m in mean]}")
    print(f"  Feature stds:  {[f'{s:.3f}' for s in std]}")

    Xnorm = ((X - mean) / std).astype(np.float32)
    train_ds = TruncDataset(Xnorm[:n_train], mask[:n_train], y[:n_train], random_trunc=True)
    val_ds   = TruncDataset(Xnorm[n_train:], mask[n_train:], y[n_train:],
                            random_trunc=False, fixed_dec=10)
    return train_ds, val_ds, mean, std


def run_epoch(model, loader, criterion, device, opt=None):
    train = opt is not None
    model.train(train)
    total = 0.0; n_tot = 0; correct = 0
    for X, mk, y in loader:
        X, mk, y = X.to(device), mk.to(device), y.to(device)
        if train: opt.zero_grad()
        logits = model(X, mk)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * len(y); n_tot += len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
    return total / n_tot, correct / n_tot


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    train_ds, val_ds, mean, std = load_dataset()
    n_features = train_ds.X.shape[-1]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = TSWinPredictor(
        n_features=n_features, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FF, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model v2: {n_params:,} parameters ({n_features} features)\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    best_val = float("inf"); best_epoch = -1; patience = 0
    history = []
    print(f"{'epoch':>5} {'tr_loss':>9} {'tr_acc':>8} {'va_loss':>9} {'va_acc':>8} {'time':>6}")
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, crit, device, opt)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, crit, device)
        history.append({"epoch":epoch,"tr_loss":tr_loss,"tr_acc":tr_acc,
                        "va_loss":va_loss,"va_acc":va_acc})
        print(f"{epoch:5d} {tr_loss:9.4f} {tr_acc:8.4f} {va_loss:9.4f} {va_acc:8.4f} "
              f"{time.time()-t0:5.1f}s")
        if va_loss < best_val - 1e-5:
            best_val = va_loss; best_epoch = epoch; patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "n_features": n_features,
                "d_model": D_MODEL, "n_heads": N_HEADS,
                "n_layers": N_LAYERS, "dim_feedforward": DIM_FF,
                "dropout": DROPOUT,
                "feature_mean": mean.tolist(), "feature_std": std.tolist(),
                "epoch": epoch, "val_loss": va_loss,
            }, CKPT_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stop at epoch {epoch} (best {best_epoch})")
                break

    print(f"\nBest val_loss: {best_val:.4f}  (epoch {best_epoch})")
    print(f"Best checkpoint: {CKPT_PATH}")
    with open(STATS_PATH, "w") as f:
        json.dump({
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "n_params": n_params, "n_features": n_features,
            "feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
