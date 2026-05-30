"""
Single-bet backtest swept across decision minutes T+4..T+13 and min-edge
thresholds. Uses the multi-minute NN model. Verifies the corrected P&L
formula throughout.

For each (minute, edge) cell: run single-bet strategy at that minute, with
that edge threshold, on the held-out val set. Compare against 2D table at
the same (minute, edge) for context.

Strategy template per window:
  1. At decision minute T+m, compute BTC direction (up/down) from open.
  2. Fair price from NN = P(directional side wins).
  3. Mispricing = fair - fill (fill = ask + 4¢).
  4. If mispricing*100 >= MIN_EDGE: place bet sized by sigmoid × sigmoid.
  5. Otherwise skip.
"""

import os
import sys
import csv
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best_multi.pt")
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")

VAL_FRAC   = 0.20
MINUTES    = list(range(4, 14))
EDGES      = [1, 2, 5, 10]
BASE_STAKE = 100.0   # match simulate_dh.py STAKE
SLIP_C     = 4
FEE_KEEP   = 0.93
MAX_FILL   = 0.97

SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0


def f_btc(p):  return SIG_MAX / (1 + math.exp(-SIG_K * (p - SIG_C)))
def g_misp(m): return MISPR_MAX / (1 + math.exp(-MISPR_K * m))


# ── 2D table ──────────────────────────────────────────────────────────────────
_BUCKETS = [(0.000,0.05),(0.050,0.10),(0.100,0.20),(0.200,0.50),(0.500,float("inf"))]
_LABELS  = ["0.00-0.05%","0.05-0.10%","0.10-0.20%","0.20-0.50%","0.50%+"]
_FAIR_BY_MIN = {1:0.582,2:0.617,3:0.636,4:0.670,5:0.698,6:0.728,7:0.751,
                8:0.759,9:0.783,10:0.798,11:0.806,12:0.815,13:0.826,14:0.704}
_TABLE = {}; _MIN_N = 30


def _load_table():
    l2i = {l:i for i,l in enumerate(_LABELS)}
    with open(TABLE_PATH, newline="") as f:
        for r in csv.DictReader(f):
            m = int(r["minute"]); bi = l2i.get(r["bucket"])
            if bi is None: continue
            _TABLE[(m, bi)] = (float(r["win_rate"]), int(r["n"]))


def _bucket(pct):
    for i, (lo, hi) in enumerate(_BUCKETS):
        if lo <= pct < hi: return i
    return len(_BUCKETS) - 1


def fair_2d(minute, pct):
    e = _TABLE.get((minute, _bucket(pct)))
    if e and e[1] >= _MIN_N:
        return e[0]
    return _FAIR_BY_MIN.get(minute, 0.7)


# ── Strategy evaluation ───────────────────────────────────────────────────────
def single_bet(fair_for_yes, kal_t, direction_up, pct_abs, resolved_yes, min_edge_c):
    """Return (pnl, wagered). fair_for_yes = P(YES wins)."""
    slip = SLIP_C / 100.0
    fill = min(MAX_FILL, (kal_t if direction_up else 1.0 - kal_t) + slip)
    fair = fair_for_yes if direction_up else (1.0 - fair_for_yes)
    mispr = fair - fill
    if mispr * 100 < min_edge_c:
        return 0.0, 0.0
    stake = BASE_STAKE * f_btc(pct_abs) * g_misp(mispr)
    if stake < 1.0:
        return 0.0, 0.0
    contracts = stake / fill
    side_won = (direction_up and resolved_yes) or (not direction_up and not resolved_yes)
    pnl = contracts * (1 - fill) * FEE_KEEP if side_won else -stake
    return pnl, stake


def main():
    # Load val set
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n_val = int(len(y) * VAL_FRAC); n_train = len(y) - n_val
    Xva, mva, yva = X[n_train:], mask[n_train:], y[n_train:]
    n = len(yva)
    print(f"Val windows: {n}")

    _load_table()

    # NN predictions per (window, decision minute)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"],  dtype=np.float32)
    model = TSWinPredictor(n_features=7); model.load_state_dict(ckpt["model_state"]); model.eval()

    Xn = ((Xva - mean) / std).astype(np.float32)
    p_yes_nn = np.zeros((n, 15), dtype=np.float32)
    for m in MINUTES:
        Xt = Xn.copy(); mt = mva.copy()
        Xt[:, m+1:, :] = 0.0
        mt[:, m+1:]    = False
        with torch.no_grad():
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p_yes_nn[:, m] = torch.sigmoid(logits).numpy()
    print(f"NN predictions ready for T+4..T+13\n")

    # Run for each (minute, edge)
    for source_name, fair_fn in [
        ("NN",       lambda i, m: float(p_yes_nn[i, m])),
        ("2D table", lambda i, m: fair_2d(m, abs(float(Xva[i, m, 0])) * 100)
                                   if float(Xva[i, m, 0]) > 0
                                   else 1.0 - fair_2d(m, abs(float(Xva[i, m, 0])) * 100)),
    ]:
        print(f"━━━ {source_name} single-bet, by decision minute and edge ━━━")
        print(f"  {'min':>3s} | " + " | ".join(
            f"{'edge='+str(e)+'c':>22s}" for e in EDGES))
        print(f"  {'':>3s} | " + " | ".join(
            f"{'bets':>5s} {'win%':>5s} {'ROI':>9s}" for _ in EDGES))
        print(f"  {'-'*3} | " + " | ".join("-" * 22 for _ in EDGES))
        for mnt in MINUTES:
            row = []
            for ec in EDGES:
                pnl = wag = 0.0
                n_b = n_w = 0
                for i in range(n):
                    if not mva[i, mnt]: continue
                    ret = float(Xva[i, mnt, 0]); kal = float(Xva[i, mnt, 2])
                    if not (0.01 < kal < 0.99) or ret == 0: continue
                    up = ret > 0; pct = abs(ret) * 100.0
                    fair_y = fair_fn(i, mnt)
                    p, w = single_bet(fair_y, kal, up, pct, bool(yva[i] == 1), ec)
                    pnl += p; wag += w
                    if w > 0:
                        n_b += 1
                        if p > 0: n_w += 1
                roi = pnl/wag*100 if wag else 0
                wrt = n_w/n_b*100 if n_b else 0
                row.append(f"{n_b:>5d} {wrt:>4.0f}% {roi:>+7.1f}%")
            print(f"  T+{mnt:>2d} | " + " | ".join(row))
        print()


if __name__ == "__main__":
    main()
