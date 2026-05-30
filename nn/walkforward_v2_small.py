"""
v2_small walkforward: 14 features (same as v2) but v1-sized architecture
(d_model=32, n_layers=2, ~17k params) to isolate whether the bigger model
was hurting.

If v2_small >= v2: bigger model was overfitting; stick with v1 architecture.
If v2_small < v2: bigger model helps; keep it.
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
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")
SEED       = 0

BATCH_SIZE = 128
LR         = 1e-3
WD         = 1e-4    # weaker reg since smaller model
EPOCHS     = 40
PATIENCE   = 8
MIN_DEC    = 4
MAX_DEC    = 13

# v1-sized architecture
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


_BUCKETS = [(0.000,0.05),(0.050,0.10),(0.100,0.20),(0.200,0.50),(0.500,float("inf"))]
_LABELS  = ["0.00-0.05%","0.05-0.10%","0.10-0.20%","0.20-0.50%","0.50%+"]
_FAIR_BY_MIN = {1:0.582,2:0.617,3:0.636,4:0.670,5:0.698,6:0.728,7:0.751,
                8:0.759,9:0.783,10:0.798,11:0.806,12:0.815,13:0.826,14:0.704}
_TABLE = {}; _MIN_N = 30


def _bucket(p):
    for i,(lo,hi) in enumerate(_BUCKETS):
        if lo <= p < hi: return i
    return len(_BUCKETS) - 1


def _load_table():
    if _TABLE: return
    l2i = {l:i for i,l in enumerate(_LABELS)}
    with open(TABLE_PATH, newline="") as f:
        for r in csv.DictReader(f):
            m = int(r["minute"]); bi = l2i.get(r["bucket"])
            if bi is None: continue
            _TABLE[(m, bi)] = (float(r["win_rate"]), int(r["n"]))


def fair_2d(minute, pct):
    _load_table()
    e = _TABLE.get((minute, _bucket(pct)))
    if e and e[1] >= _MIN_N: return e[0]
    return _FAIR_BY_MIN.get(minute, 0.7)


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
    n_params = sum(p.numel() for p in model.parameters())
    return model, mean, std, n_params


def simulate_window(X, mask, y, fair_fn):
    resolved_yes = (y == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c = no_c = 0.0
    legs = 0
    for m in MINUTES_LATE:
        if legs >= MAX_LEGS: break
        if not mask[m]: continue
        ret = float(X[m, 0]); kal = float(X[m, 4])  # v2: feature 4 = yes_close
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
    pnl_yes = sum((1-fp)*(s/fp)*FEE_KEEP if resolved_yes else -s for s,fp in yes_bets)
    pnl_no  = sum((1-fp)*(s/fp)*FEE_KEEP if not resolved_yes else -s for s,fp in no_bets)
    return {"pnl": pnl_yes + pnl_no, "wagered": yes_exp + no_exp,
            "n_bets": len(yes_bets) + len(no_bets)}


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y); n_features = X.shape[-1]
    print(f"Dataset v2 (small arch): {n} windows | features={n_features}")
    print(f"  Date range: {datetime.fromtimestamp(int(ts[0])).date()} -> "
          f"{datetime.fromtimestamp(int(ts[-1])).date()}\n")

    fold_starts = [int(n * f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size   = int(n * 0.15)
    folds = []
    for i, s in enumerate(fold_starts):
        e = min(s + fold_size, n)
        if s <= 100 or e - s < 100: continue
        folds.append((f"F{i+1}", s, e))

    rows = []
    for label, s, e in folds:
        print(f"Fold {label}:")
        Xtr, mtr, ytr = X[:s], mask[:s], y[:s]
        Xte, mte, yte = X[s:e], mask[s:e], y[s:e]
        t0 = time.time()
        model, mean, std, np_count = train_nn(Xtr, mtr, ytr, n_features)
        print(f"  trained v1-arch model ({np_count:,} params) in {time.time()-t0:.1f}s", flush=True)
        Xn = ((Xte - mean) / std).astype(np.float32)
        p_yes = np.zeros((len(yte), 15), dtype=np.float32)
        with torch.no_grad():
            for m in MINUTES_LATE:
                Xt = Xn.copy(); mt = mte.copy()
                Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
                logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
                p_yes[:, m] = torch.sigmoid(logits).numpy()
        # NN eval
        pnl = wag = 0.0; acted = wins = 0
        for i in range(len(yte)):
            ff = lambda m, ii=i: float(p_yes[ii, m])
            r = simulate_window(Xte[i], mte[i], yte[i], ff)
            pnl += r["pnl"]; wag += r["wagered"]
            if r["wagered"] > 0:
                acted += 1
                if r["pnl"] > 0: wins += 1
        roi = pnl/wag*100 if wag else 0
        wr = wins/acted*100 if acted else 0
        print(f"    NN_v2_small: acted={acted:>4d} win%={wr:>5.1f} "
              f"wagered=${wag:>9.0f} P&L=${pnl:>+9.0f} ROI={roi:>+7.2f}%", flush=True)
        rows.append({"fold":label,"name":"NN_v2_small","acted":acted,"win":wr,
                     "wag":wag,"pnl":pnl,"roi":roi})

    print(f"\n━━━ Summary v2_small (14 feats, v1 arch) ━━━")
    print(f"  {'fold':>4s} | {'ROI':>9s} {'P&L':>10s} {'bets':>6s}")
    for r in rows:
        print(f"  {r['fold']:>4s} | {r['roi']:>+8.2f}% ${r['pnl']:>+8.0f} {r['acted']:>6d}")
    total_pnl = sum(r["pnl"] for r in rows)
    total_wag = sum(r["wag"] for r in rows)
    total_bets = sum(r["acted"] for r in rows)
    print(f"  {'all':>4s} | {total_pnl/total_wag*100:>+8.2f}% ${total_pnl:>+8.0f} {total_bets:>6d}")


if __name__ == "__main__":
    main()
