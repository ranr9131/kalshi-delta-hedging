"""
Train v2_small: 14 features, v1-sized architecture (~18k params).
Output checkpoint: nn/checkpoints/best_v2_small.pt
"""
import os, sys, json, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset_v2.npz")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
CKPT_PATH  = os.path.join(CKPT_DIR, "best_v2_small.pt")
STATS_PATH = os.path.join(CKPT_DIR, "train_stats_v2_small.json")

VAL_FRAC   = 0.20
BATCH_SIZE = 128
LR         = 1e-3
WD         = 1e-4
EPOCHS     = 60
PATIENCE   = 10
SEED       = 0

# v1-sized architecture (proven best in walkforward)
D_MODEL  = 32
N_HEADS  = 4
N_LAYERS = 2
DIM_FF   = 64
DROPOUT  = 0.1

MIN_DEC = 4
MAX_DEC = 13


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
        correct += ((torch.sigmoid(logits) > 0.5).float() == y).sum().item()
    return total / n_tot, correct / n_tot


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y); n_val = int(n * VAL_FRAC); n_train = n - n_val
    n_features = X.shape[-1]
    print(f"  Total: {n}  Train: {n_train}  Val: {n_val}  features={n_features}")

    Xn = X[:n_train].copy(); Xn[:, MIN_DEC+1:, :] = 0.0
    flat = Xn.reshape(-1, n_features)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32) + 1e-6
    Xnorm = ((X - mean) / std).astype(np.float32)

    train_ds = TruncDataset(Xnorm[:n_train], mask[:n_train], y[:n_train], random_trunc=True)
    val_ds   = TruncDataset(Xnorm[n_train:], mask[n_train:], y[n_train:],
                            random_trunc=False, fixed_dec=10)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = TSWinPredictor(
        n_features=n_features, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FF, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model v2_small: {n_params:,} parameters\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()

    best_val = float("inf"); best_epoch = -1; patience = 0
    print(f"{'epoch':>5} {'tr_loss':>9} {'tr_acc':>8} {'va_loss':>9} {'va_acc':>8} {'time':>6}")
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, crit, device, opt)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, crit, device)
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
    print(f"Saved: {CKPT_PATH}")


if __name__ == "__main__":
    main()
