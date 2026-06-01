"""
v2_small edge-threshold sweep. Trains v2_small once per fold, then evaluates
the multi-bet strategy at multiple edge filter values to find the sweet spot
between ROI per dollar and total daily $ profit.
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
DATA_PATH  = os.path.join(ROOT, "data", "dataset_v2.npz")
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
RH_TRIGGER   = 10.0
MAX_HEDGE_F  = 0.80
MAX_FILL     = 0.97
MAX_LEGS     = 2
FEE_KEEP     = 0.93
SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0

EDGES = [1, 2, 3, 5, 7, 10, 15]


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
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE: break
    model.load_state_dict(best_state)
    model = model.to("cpu"); model.eval()
    return model, mean, std


def simulate_window(X, mask, y, p_yes_row, min_edge_c):
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
        p_yes = float(p_yes_row[m])
        fair = p_yes if up else (1 - p_yes)
        mispr = fair - (yes_fill if up else no_fill)
        if mispr * 100 < min_edge_c:
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
    pnl_yes = sum((1-fp)*(s/fp)*FEE_KEEP if resolved_yes else -s for s,fp in yes_bets)
    pnl_no  = sum((1-fp)*(s/fp)*FEE_KEEP if not resolved_yes else -s for s,fp in no_bets)
    return {"pnl": pnl_yes + pnl_no, "wagered": yes_exp + no_exp}


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y); n_features = X.shape[-1]
    print(f"Dataset: {n} windows | features={n_features}\n", flush=True)

    fold_starts = [int(n * f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size   = int(n * 0.15)
    folds = []
    for i, s in enumerate(fold_starts):
        e = min(s + fold_size, n)
        if s <= 100 or e - s < 100: continue
        folds.append((f"F{i+1}", s, e))

    all_results = {(label, ec): {"acted":0,"wins":0,"wag":0.0,"pnl":0.0}
                   for label, _, _ in folds for ec in EDGES}

    for label, s, e in folds:
        print(f"━━━ Fold {label} (test size {e-s}) ━━━", flush=True)
        Xtr, mtr, ytr = X[:s], mask[:s], y[:s]
        Xte, mte, yte = X[s:e], mask[s:e], y[s:e]
        t0 = time.time()
        model, mean, std = train_nn(Xtr, mtr, ytr, n_features)
        print(f"  trained in {time.time()-t0:.1f}s", flush=True)
        Xn = ((Xte - mean) / std).astype(np.float32)
        p_yes = np.zeros((len(yte), 15), dtype=np.float32)
        with torch.no_grad():
            for m in MINUTES_LATE:
                Xt = Xn.copy(); mt = mte.copy()
                Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
                logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
                p_yes[:, m] = torch.sigmoid(logits).numpy()
        for ec in EDGES:
            for i in range(len(yte)):
                r = simulate_window(Xte[i], mte[i], yte[i], p_yes[i], ec)
                if r["wagered"] > 0:
                    all_results[(label, ec)]["acted"] += 1
                    if r["pnl"] > 0: all_results[(label, ec)]["wins"] += 1
                    all_results[(label, ec)]["wag"] += r["wagered"]
                all_results[(label, ec)]["pnl"] += r["pnl"]
        print(f"  {'edge':>4s}  {'acted':>5s}  {'win%':>5s}  {'wagered':>10s}  {'P&L':>10s}  {'ROI':>7s}", flush=True)
        for ec in EDGES:
            r = all_results[(label, ec)]
            roi = r["pnl"]/r["wag"]*100 if r["wag"] else 0
            wp  = r["wins"]/r["acted"]*100 if r["acted"] else 0
            print(f"  {ec:>3d}c  {r['acted']:>5d}  {wp:>4.1f}%  ${r['wag']:>8.0f}  ${r['pnl']:>+8.0f}  {roi:>+6.2f}%", flush=True)
        print(flush=True)

    print(f"━━━ Aggregate across all folds ━━━", flush=True)
    print(f"  {'edge':>4s}  {'bets':>5s}  {'win%':>5s}  {'wagered':>10s}  {'P&L':>10s}  {'ROI':>7s}", flush=True)
    for ec in EDGES:
        acted = sum(all_results[(lbl, ec)]["acted"] for lbl,_,_ in folds)
        wins  = sum(all_results[(lbl, ec)]["wins"]  for lbl,_,_ in folds)
        wag   = sum(all_results[(lbl, ec)]["wag"]   for lbl,_,_ in folds)
        pnl   = sum(all_results[(lbl, ec)]["pnl"]   for lbl,_,_ in folds)
        roi = pnl/wag*100 if wag else 0
        wp  = wins/acted*100 if acted else 0
        print(f"  {ec:>3d}c  {acted:>5d}  {wp:>4.1f}%  ${wag:>8.0f}  ${pnl:>+8.0f}  {roi:>+6.2f}%", flush=True)


if __name__ == "__main__":
    main()
