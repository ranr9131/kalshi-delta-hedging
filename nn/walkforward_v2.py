"""
Walk-forward validation for v2 NN (14 features, bigger architecture).

Same fold structure as walkforward.py but uses dataset_v2.npz and trains a
larger model per fold. Compares NN v2 vs 2D table across 4 chronological folds.
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

# v2 training config (matches train_v2.py)
BATCH_SIZE = 128
LR         = 1e-3
WD         = 5e-4
EPOCHS     = 40
PATIENCE   = 8
MIN_DEC    = 4
MAX_DEC    = 13

# Architecture v2
D_MODEL  = 48
N_HEADS  = 6
N_LAYERS = 4
DIM_FF   = 128
DROPOUT  = 0.2

# Backtest config (matches walkforward.py)
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


# 2D table
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
    return model, mean, std


def simulate_window(X, mask, y, fair_fn, minutes, rh_minute, max_legs, min_edge_c):
    resolved_yes = (y == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c = no_c = 0.0
    legs = 0
    for m in minutes:
        if legs >= max_legs: break
        if not mask[m]: continue
        ret = float(X[m, 0]); kal = float(X[m, 4])   # v2: feature 4 is yes_close
        if not (0.01 < kal < 0.99) or ret == 0: continue
        up = ret > 0; pct = abs(ret) * 100
        slip = SLIP_C / 100.0
        yes_fill = min(MAX_FILL, kal + slip)
        no_fill  = min(MAX_FILL, (1 - kal) + slip)
        fair_y = fair_fn(m)
        fair = fair_y if up else (1 - fair_y)
        mispr = fair - (yes_fill if up else no_fill)
        if mispr * 100 < min_edge_c:
            target = 0.0
        else:
            target = BASE_STAKE * f_btc(pct) * g_misp(mispr) * time_decay(m)
        if up:
            gap = max(0.0, target - yes_exp)
            if gap >= MIN_BET and legs < max_legs and yes_fill < MAX_FILL:
                yes_bets.append((gap, yes_fill))
                yes_exp += gap; yes_c += gap / yes_fill; legs += 1
        else:
            gap = max(0.0, target - no_exp)
            if gap >= MIN_BET and legs < max_legs and no_fill < MAX_FILL:
                no_bets.append((gap, no_fill))
                no_exp += gap; no_c += gap / no_fill; legs += 1
        if rh_minute is not None and m >= rh_minute and legs < max_legs:
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


def run_fold(X_tr, mk_tr, y_tr, X_te, mk_te, y_te, label, n_features):
    print(f"  [fold {label}] training v2 NN on {len(y_tr)} windows…", flush=True)
    t0 = time.time()
    model, mean, std = train_nn(X_tr, mk_tr, y_tr, n_features)
    print(f"    trained in {time.time()-t0:.1f}s", flush=True)

    Xn = ((X_te - mean) / std).astype(np.float32)
    p_yes = np.zeros((len(y_te), 15), dtype=np.float32)
    with torch.no_grad():
        for m in MINUTES_LATE:
            Xt = Xn.copy(); mt = mk_te.copy()
            Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p_yes[:, m] = torch.sigmoid(logits).numpy()

    def evalst(name, factory):
        pnl = wag = 0.0; acted = wins = 0
        for i in range(len(y_te)):
            ff = factory(i)
            r = simulate_window(X_te[i], mk_te[i], y_te[i], ff,
                                MINUTES_LATE, 10, MAX_LEGS, MIN_EDGE_C)
            pnl += r["pnl"]; wag += r["wagered"]
            if r["wagered"] > 0:
                acted += 1
                if r["pnl"] > 0: wins += 1
        roi = pnl/wag*100 if wag else 0
        wr = wins/acted*100 if acted else 0
        return {"name": name, "acted": acted, "win": wr, "wag": wag, "pnl": pnl, "roi": roi}

    nn_r = evalst("NN_v2", lambda i: lambda m, ii=i: float(p_yes[ii, m]))
    def tb_factory(i):
        def fn(m):
            ret = float(X_te[i, m, 0]); pct = abs(ret) * 100
            wr = fair_2d(m, pct)
            return wr if ret > 0 else (1 - wr)
        return fn
    tb_r = evalst("2D", tb_factory)
    return nn_r, tb_r


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y); n_features = X.shape[-1]
    print(f"Dataset v2: {n} windows | features={n_features}")
    print(f"  Date range: {datetime.fromtimestamp(int(ts[0])).date()} -> "
          f"{datetime.fromtimestamp(int(ts[-1])).date()}\n")

    fold_starts = [int(n * f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size   = int(n * 0.15)
    folds = []
    for i, s in enumerate(fold_starts):
        e = min(s + fold_size, n)
        if s <= 100 or e - s < 100: continue
        folds.append((f"F{i+1}", s, e))
    print(f"Folds:")
    for label, s, e in folds:
        d0 = datetime.fromtimestamp(int(ts[s])).date()
        d1 = datetime.fromtimestamp(int(ts[e-1])).date()
        print(f"  {label}: train={s} | test={e-s}  ({d0} → {d1})")
    print()

    rows = []
    for label, s, e in folds:
        print(f"Fold {label}:")
        nn_r, tb_r = run_fold(X[:s], mask[:s], y[:s], X[s:e], mask[s:e], y[s:e],
                              label, n_features)
        for r in (nn_r, tb_r):
            print(f"    {r['name']:>5s}: acted={r['acted']:>4d} win%={r['win']:>5.1f} "
                  f"wagered=${r['wag']:>9.0f} P&L=${r['pnl']:>+9.0f} ROI={r['roi']:>+7.2f}%")
            r["fold"] = label
            rows.append(r)

    print(f"\n━━━ Summary v2 ━━━")
    print(f"  {'fold':>4s} | {'NN v2 ROI':>9s} {'NN v2 P&L':>10s} {'bets':>6s} | "
          f"{'2D ROI':>8s} {'2D P&L':>10s} {'bets':>6s}")
    for label, s, e in folds:
        nn_r = next(r for r in rows if r["fold"]==label and r["name"]=="NN_v2")
        tb_r = next(r for r in rows if r["fold"]==label and r["name"]=="2D")
        print(f"  {label:>4s} | {nn_r['roi']:>+8.2f}% ${nn_r['pnl']:>+8.0f} {nn_r['acted']:>6d} | "
              f"{tb_r['roi']:>+7.2f}% ${tb_r['pnl']:>+8.0f} {tb_r['acted']:>6d}")
    nn_pnl = sum(r["pnl"] for r in rows if r["name"]=="NN_v2")
    nn_wag = sum(r["wag"] for r in rows if r["name"]=="NN_v2")
    tb_pnl = sum(r["pnl"] for r in rows if r["name"]=="2D")
    tb_wag = sum(r["wag"] for r in rows if r["name"]=="2D")
    print(f"  {'all':>4s} | {nn_pnl/nn_wag*100:>+8.2f}% ${nn_pnl:>+8.0f} "
          f"{sum(r['acted'] for r in rows if r['name']=='NN_v2'):>6d} | "
          f"{tb_pnl/tb_wag*100:>+7.2f}% ${tb_pnl:>+8.0f} "
          f"{sum(r['acted'] for r in rows if r['name']=='2D'):>6d}")


if __name__ == "__main__":
    main()
