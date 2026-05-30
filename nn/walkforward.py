"""
Walk-forward validation: train NN on growing chronological window, evaluate on
the next ~14 days. Repeat across the 90-day dataset. Compare NN vs 2D table at
each fold to see if NN edge is robust or regime-specific.

For each fold:
  1. Train NN on all data BEFORE the fold's test period.
  2. Run multi-bet DH backtest on the test period using the freshly-trained NN.
  3. Same backtest with 2D table for comparison.
  4. Record ROI, win rate, bets.

Folds are sized to give ~1000-1500 test windows each (about 2 weeks of data).
"""

import os
import sys
import csv
import math
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")
SEED       = 0

# Training config (must match train.py --multi defaults)
BATCH_SIZE = 128
LR         = 1e-3
WD         = 1e-4
EPOCHS     = 30          # shorter than full train; early-stopping cuts further
PATIENCE   = 6
MIN_DEC    = 4
MAX_DEC    = 13

# Backtest config (matches simulate_dh.py canonical, scaled to $100 base stake)
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


# ── 2D table ──────────────────────────────────────────────────────────────────
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


# ── Training ─────────────────────────────────────────────────────────────────
class TruncDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X = X; self.mask = mask; self.y = y
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
    """Train a fresh NN on the given train slice; return model + feat stats."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = pick_device()
    # Use earliest truncation for feature stats to be conservative
    Xn = X_train.copy(); Xn[:, MIN_DEC+1:, :] = 0.0
    flat = Xn.reshape(-1, n_features)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32) + 1e-6
    Xnorm = ((X_train - mean) / std).astype(np.float32)
    ds = TruncDataset(Xnorm, mask_train, y_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = TSWinPredictor(n_features=n_features).to(device)
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
            best_loss = epoch_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model = model.to("cpu")
    model.eval()
    return model, mean, std


# ── Backtest core (mirrors backtest_late.py, multi-bet target mode) ──────────
def simulate_window(X, mask, y, fair_fn, minutes, rh_minute, max_legs, min_edge_c):
    resolved_yes = (y == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c = no_c = 0.0
    legs = 0
    for m in minutes:
        if legs >= max_legs: break
        if not mask[m]: continue
        ret = float(X[m, 0]); kal = float(X[m, 2])
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
        # RH overlay
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
    return {"pnl": pnl_yes + pnl_no, "wagered": yes_exp + no_exp, "n_bets": len(yes_bets) + len(no_bets)}


def run_fold(X_train, mask_train, y_train,
             X_test, mask_test, y_test, ts_test,
             fold_label, n_features):
    """Train on train slice, eval both NN and 2D table on test slice."""
    print(f"  [fold {fold_label}] training NN on {len(y_train)} windows…", flush=True)
    t0 = time.time()
    model, mean, std = train_nn(X_train, mask_train, y_train, n_features)
    print(f"    trained in {time.time()-t0:.1f}s", flush=True)

    # Pre-compute NN P(YES) for each test window at each minute T+10..T+13
    Xn = ((X_test - mean) / std).astype(np.float32)
    p_yes = np.zeros((len(y_test), 15), dtype=np.float32)
    with torch.no_grad():
        for m in MINUTES_LATE:
            Xt = Xn.copy(); mt = mask_test.copy()
            Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p_yes[:, m] = torch.sigmoid(logits).numpy()

    # Run both strategies on test
    def eval_strat(name, fair_factory):
        pnl = wag = 0.0; acted = wins = 0
        for i in range(len(y_test)):
            ff = fair_factory(i)
            r = simulate_window(X_test[i], mask_test[i], y_test[i],
                                ff, MINUTES_LATE, 10, MAX_LEGS, MIN_EDGE_C)
            pnl += r["pnl"]; wag += r["wagered"]
            if r["wagered"] > 0:
                acted += 1
                if r["pnl"] > 0: wins += 1
        roi = pnl/wag*100 if wag else 0
        wr  = wins/acted*100 if acted else 0
        return {"name": name, "n_test": len(y_test), "acted": acted,
                "win_pct": wr, "wagered": wag, "pnl": pnl, "roi": roi}

    nn_results = eval_strat("NN", lambda i: lambda m, ii=i: float(p_yes[ii, m]))
    def tb_factory(i):
        def fn(m):
            ret = float(X_test[i, m, 0]); pct = abs(ret) * 100
            wr = fair_2d(m, pct)
            return wr if ret > 0 else (1 - wr)
        return fn
    tb_results = eval_strat("2D", tb_factory)
    return nn_results, tb_results


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y); n_features = X.shape[-1]
    print(f"Dataset: {n} windows | features={n_features}")
    print(f"  Date range: {datetime.fromtimestamp(int(ts[0])).date()} -> "
          f"{datetime.fromtimestamp(int(ts[-1])).date()}\n")

    # Define 4 walk-forward folds — each test ~1500 windows (~16 days)
    fold_starts = [int(n * f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size   = int(n * 0.15)
    folds = []
    for i, start in enumerate(fold_starts):
        end = min(start + fold_size, n)
        if start <= 100 or end - start < 100: continue
        folds.append((f"F{i+1}", start, end))
    print(f"Walk-forward folds (train growing, test fixed-size next chunk):")
    for label, s, e in folds:
        d_start = datetime.fromtimestamp(int(ts[s])).date()
        d_end   = datetime.fromtimestamp(int(ts[e-1])).date()
        print(f"  {label}: train={s} windows | test={e-s} windows  ({d_start} → {d_end})")
    print()

    rows = []
    for label, start, end in folds:
        X_tr, mk_tr, y_tr = X[:start], mask[:start], y[:start]
        X_te, mk_te, y_te = X[start:end], mask[start:end], y[start:end]
        ts_te = ts[start:end]
        print(f"Fold {label}:")
        nn_r, tb_r = run_fold(X_tr, mk_tr, y_tr, X_te, mk_te, y_te, ts_te, label, n_features)
        # Print and store
        for r in (nn_r, tb_r):
            print(f"    {r['name']:>3s}: acted={r['acted']:>4d} "
                  f"win%={r['win_pct']:>5.1f} wagered=${r['wagered']:>9.0f} "
                  f"P&L=${r['pnl']:>+9.0f} ROI={r['roi']:>+7.2f}%")
            r["fold"] = label
            rows.append(r)

    print(f"\n━━━ Summary across {len(folds)} folds ━━━")
    print(f"  {'fold':>4s} | {'NN ROI':>8s} {'NN P&L':>10s} {'NN bets':>8s} | "
          f"{'2D ROI':>8s} {'2D P&L':>10s} {'2D bets':>8s}")
    for label, start, end in folds:
        nn_r = next(r for r in rows if r["fold"]==label and r["name"]=="NN")
        tb_r = next(r for r in rows if r["fold"]==label and r["name"]=="2D")
        print(f"  {label:>4s} | {nn_r['roi']:>+7.2f}% ${nn_r['pnl']:>+8.0f} {nn_r['acted']:>8d} | "
              f"{tb_r['roi']:>+7.2f}% ${tb_r['pnl']:>+8.0f} {tb_r['acted']:>8d}")
    # Aggregated
    nn_pnl = sum(r["pnl"] for r in rows if r["name"]=="NN")
    nn_wag = sum(r["wagered"] for r in rows if r["name"]=="NN")
    tb_pnl = sum(r["pnl"] for r in rows if r["name"]=="2D")
    tb_wag = sum(r["wagered"] for r in rows if r["name"]=="2D")
    print(f"  {'all':>4s} | {nn_pnl/nn_wag*100:>+7.2f}% ${nn_pnl:>+8.0f} {sum(r['acted'] for r in rows if r['name']=='NN'):>8d} | "
          f"{tb_pnl/tb_wag*100:>+7.2f}% ${tb_pnl:>+8.0f} {sum(r['acted'] for r in rows if r['name']=='2D'):>8d}")


if __name__ == "__main__":
    main()
