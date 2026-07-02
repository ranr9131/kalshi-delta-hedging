"""
Walkforward A/B: does anchoring the transformer on fair_price_model_v3 help?

Compares, on identical chronological folds (same arch/seed/sim as
walkforward_v2_small.py):

  NN14 — baseline transformer, first 14 features (= current v2_small)
  NN16 — same transformer + v3_fair + v3_minus_mid features
  V3   — v3 fair price alone (feature 14), no NN

Reports per fold: Brier at m=10 (vs market mid reference) and the trading sim
(acted windows, win%, ROI). Reads nn/data/dataset_v2p.npz (build with
augment_dataset_v3.py).
"""

import os, sys, csv, math, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset_v2p.npz")
SEED       = 0

BATCH_SIZE = 128
LR         = 1e-3
WD         = 1e-4
EPOCHS     = 40
PATIENCE   = 8
MIN_DEC    = 4
MAX_DEC    = 13

D_MODEL  = 32
N_HEADS  = 4
N_LAYERS = 2
DIM_FF   = 64
DROPOUT  = 0.1

MINUTES_LATE = list(range(10, 14))
BASE_STAKE   = 100.0
MIN_BET      = 5.0
SLIP_C       = 4
MIN_EDGE_C   = 10
RH_TRIGGER   = 10.0
MAX_HEDGE_F  = 0.80
MAX_FILL     = 0.97
MAX_LEGS     = 2
FEE_KEEP     = 0.93
SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0


def f_btc(p):  return SIG_MAX / (1.0 + math.exp(-SIG_K * (p - SIG_C)))
def g_misp(m): return MISPR_MAX / (1.0 + math.exp(-MISPR_K * m))
def time_decay(minute):
    if minute < 7:  return 0.4
    if minute < 10: return 0.8
    return 1.2


class TruncDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X, self.mask, self.y = X, mask, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        X = self.X[idx].copy(); m = self.mask[idx].copy()
        dec = random.randint(MIN_DEC, MAX_DEC)
        X[dec+1:, :] = 0.0
        m[dec+1:]    = False
        return (torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(m),
                torch.tensor(self.y[idx], dtype=torch.float32))


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def train_nn(X_train, mask_train, y_train, n_features):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = pick_device()
    Xn = X_train.copy(); Xn[:, MIN_DEC+1:, :] = 0.0
    flat = Xn.reshape(-1, n_features)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32) + 1e-6
    Xnorm = ((X_train - mean) / std).astype(np.float32)
    ds = TruncDataset(Xnorm, mask_train, y_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = TSWinPredictor(
        n_features=n_features, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FF, dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    best_loss = float("inf"); best_state = None; patience = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0; n = 0
        for X, mk, y in loader:
            X, mk, y = X.to(device), mk.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X, mk)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y); n += len(y)
        epoch_loss = total / n
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE: break
    model.load_state_dict(best_state)
    model = model.to("cpu"); model.eval()
    return model, mean, std


def predict_minutes(model, mean, std, Xte, mte):
    """p_yes[i, m] for each test window at each decision minute."""
    Xn = ((Xte - mean) / std).astype(np.float32)
    p = np.zeros((len(Xte), 15), dtype=np.float32)
    with torch.no_grad():
        for m in MINUTES_LATE:
            Xt = Xn.copy(); mt = mte.copy()
            Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p[:, m] = torch.sigmoid(logits).numpy()
    return p


def simulate_window(X, mask, y, fair_fn):
    resolved_yes = (y == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c = no_c = 0.0
    legs = 0
    for m in MINUTES_LATE:
        if legs >= MAX_LEGS: break
        if not mask[m]: continue
        ret = float(X[m, 0]); kal = float(X[m, 4])
        if not (0.01 < kal < 0.99) or ret == 0: continue
        up = ret > 0; pct = abs(ret) * 100
        slip = SLIP_C / 100.0
        yes_fill = min(MAX_FILL, kal + slip)
        no_fill  = min(MAX_FILL, (1 - kal) + slip)
        fair_y = fair_fn(m)
        fair = fair_y if up else (1 - fair_y)
        mispr = fair - (yes_fill if up else no_fill)
        if mispr * 100 < MIN_EDGE_C:
            target = 0.0
        else:
            target = BASE_STAKE * f_btc(pct) * g_misp(mispr) * time_decay(m)
        if up:
            gap = max(0.0, target - yes_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and yes_fill < MAX_FILL:
                yes_bets.append((gap, yes_fill))
                yes_exp += gap; yes_c += gap / yes_fill; legs += 1
        else:
            gap = max(0.0, target - no_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and no_fill < MAX_FILL:
                no_bets.append((gap, no_fill))
                no_exp += gap; no_c += gap / no_fill; legs += 1
        if m >= 10 and legs < MAX_LEGS:
            if up:
                if no_exp >= RH_TRIGGER and no_c > 0 and yes_fill <= MAX_HEDGE_F:
                    hedge = no_c * yes_fill
                    if hedge >= MIN_BET:
                        yes_bets.append((hedge, yes_fill))
                        yes_exp += hedge; yes_c += hedge / yes_fill; legs += 1
            else:
                if yes_exp >= RH_TRIGGER and yes_c > 0 and no_fill <= MAX_HEDGE_F:
                    hedge = yes_c * no_fill
                    if hedge >= MIN_BET:
                        no_bets.append((hedge, no_fill))
                        no_exp += hedge; no_c += hedge / no_fill; legs += 1
    pnl_yes = sum((1-fp)*(s/fp)*FEE_KEEP if resolved_yes else -s for s, fp in yes_bets)
    pnl_no  = sum((1-fp)*(s/fp)*FEE_KEEP if not resolved_yes else -s for s, fp in no_bets)
    return {"pnl": pnl_yes + pnl_no, "wagered": yes_exp + no_exp,
            "n_bets": len(yes_bets) + len(no_bets)}


def run_sim(Xte, mte, yte, fair_for):
    pnl = wag = 0.0; acted = wins = 0
    for i in range(len(yte)):
        r = simulate_window(Xte[i], mte[i], yte[i], lambda m, ii=i: fair_for(ii, m))
        pnl += r["pnl"]; wag += r["wagered"]
        if r["wagered"] > 0:
            acted += 1
            if r["pnl"] > 0: wins += 1
    return {"acted": acted, "win": wins/acted*100 if acted else 0,
            "wag": wag, "pnl": pnl, "roi": pnl/wag*100 if wag else 0}


def brier_at(p_arr, yte, mte, m=10):
    sel = mte[:, m]
    if sel.sum() == 0: return float("nan")
    return float(np.mean((p_arr[sel] - yte[sel]) ** 2))


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y)
    print(f"Dataset v2p: {n} windows | features={X.shape[-1]} (14 base + v3_fair + v3_minus_mid)")
    print(f"  Date range: {datetime.fromtimestamp(int(ts[0])).date()} -> "
          f"{datetime.fromtimestamp(int(ts[-1])).date()}\n")

    fold_starts = [int(n * f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size   = int(n * 0.15)
    folds = []
    for i, s in enumerate(fold_starts):
        e = min(s + fold_size, n)
        if s <= 100 or e - s < 100: continue
        folds.append((f"F{i+1}", s, e))

    agg = {}
    for label, s, e in folds:
        print(f"Fold {label} (test {datetime.fromtimestamp(int(ts[s])).date()} -> "
              f"{datetime.fromtimestamp(int(ts[e-1])).date()}, n={e-s}):")
        Xte, mte, yte = X[s:e], mask[s:e], y[s:e]

        variants = {}
        t0 = time.time()
        m14, mean14, std14 = train_nn(X[:s, :, :14], mask[:s], y[:s], 14)
        p14 = predict_minutes(m14, mean14, std14, Xte[:, :, :14], mte)
        variants["NN14"] = p14
        m16, mean16, std16 = train_nn(X[:s], mask[:s], y[:s], 16)
        p16 = predict_minutes(m16, mean16, std16, Xte, mte)
        variants["NN16"] = p16
        print(f"  trained both models in {time.time()-t0:.0f}s", flush=True)

        # v3 alone: feature 14 per minute
        pv3 = np.zeros((len(yte), 15), dtype=np.float32)
        for m in MINUTES_LATE:
            pv3[:, m] = Xte[:, m, 14]
        variants["V3"] = pv3

        b_mid = brier_at(Xte[:, 10, 4], yte, mte)
        print(f"  Brier@10:  mid={b_mid:.4f}  " +
              "  ".join(f"{k}={brier_at(v[:, 10], yte, mte):.4f}"
                        for k, v in variants.items()))

        for name, p_arr in variants.items():
            r = run_sim(Xte, mte, yte, lambda i, m, pa=p_arr: float(pa[i, m]))
            print(f"  {name:>4s}: acted={r['acted']:>4d} win%={r['win']:>5.1f} "
                  f"wagered=${r['wag']:>9.0f} P&L=${r['pnl']:>+9.0f} ROI={r['roi']:>+7.2f}%",
                  flush=True)
            a = agg.setdefault(name, {"pnl": 0.0, "wag": 0.0, "acted": 0,
                                      "briers": []})
            a["pnl"] += r["pnl"]; a["wag"] += r["wag"]; a["acted"] += r["acted"]
            a["briers"].append(brier_at(p_arr[:, 10], yte, mte))
        print()

    print("━━━ Summary (all folds) ━━━")
    print(f"  {'variant':>7s} | {'Brier@10':>9s} {'ROI':>9s} {'P&L':>10s} {'acted':>6s}")
    for name, a in agg.items():
        roi = a["pnl"]/a["wag"]*100 if a["wag"] else 0
        print(f"  {name:>7s} | {np.nanmean(a['briers']):>9.4f} {roi:>+8.2f}% "
              f"${a['pnl']:>+9.0f} {a['acted']:>6d}")


if __name__ == "__main__":
    main()
