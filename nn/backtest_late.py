"""
Variant sweep: full DH multi-bet logic (target mode, RH overlay, leg cap)
but restricted to a configurable minute range. Compares NN multi-min model
vs 2D table at the same configs.

The intent: skip the low-ROI early minutes that dragged down the wide
T+4..T+13 multi-bet result.
"""

import os, sys, csv, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best_multi.pt")
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")

VAL_FRAC = 0.20
BASE_STAKE = 100.0
MIN_BET    = 5.0
SLIP_C     = 4
RH_TRIGGER = 10.0
MAX_HEDGE_F = 0.80
MAX_FILL    = 0.97
MAX_LEGS    = 2
EARLY_SKIP_M = 5
EARLY_SKIP_P = 0.05
FEE_KEEP    = 0.93

SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0


def f_btc(p):  return SIG_MAX / (1 + math.exp(-SIG_K * (p - SIG_C)))
def g_misp(m): return MISPR_MAX / (1 + math.exp(-MISPR_K * m))
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
    for i, (lo, hi) in enumerate(_BUCKETS):
        if lo <= p < hi: return i
    return len(_BUCKETS) - 1


def _load_table():
    l2i = {l:i for i,l in enumerate(_LABELS)}
    with open(TABLE_PATH, newline="") as f:
        for r in csv.DictReader(f):
            m = int(r["minute"]); bi = l2i.get(r["bucket"])
            if bi is None: continue
            _TABLE[(m, bi)] = (float(r["win_rate"]), int(r["n"]))


def fair_2d(minute, pct):
    e = _TABLE.get((minute, _bucket(pct)))
    if e and e[1] >= _MIN_N: return e[0]
    return _FAIR_BY_MIN.get(minute, 0.7)


def simulate_one(X_raw, mask, y_label, fair_fn, minutes, min_edge_c, rh_minute):
    resolved_yes = (y_label == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c   = no_c   = 0.0
    legs = 0

    for m in minutes:
        if legs >= MAX_LEGS: break
        if not mask[m]: continue
        ret = float(X_raw[m, 0]); kal = float(X_raw[m, 2])
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
        if m <= EARLY_SKIP_M and pct < EARLY_SKIP_P:
            target = 0.0

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

        if rh_minute is not None and m >= rh_minute and legs < MAX_LEGS:
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

    pnl_yes = sum((1 - fp) * (s / fp) * FEE_KEEP if resolved_yes else -s
                  for s, fp in yes_bets)
    pnl_no  = sum((1 - fp) * (s / fp) * FEE_KEEP if not resolved_yes else -s
                  for s, fp in no_bets)
    return {
        "pnl": pnl_yes + pnl_no,
        "wagered": yes_exp + no_exp,
        "n_bets": len(yes_bets) + len(no_bets),
    }


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n_val = int(len(y) * VAL_FRAC); n_train = len(y) - n_val
    Xva, mva, yva = X[n_train:], mask[n_train:], y[n_train:]
    n = len(yva)

    _load_table()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"], dtype=np.float32)
    model = TSWinPredictor(n_features=7); model.load_state_dict(ckpt["model_state"]); model.eval()

    Xn = ((Xva - mean) / std).astype(np.float32)
    p_yes = np.zeros((n, 15), dtype=np.float32)
    for m in range(4, 14):
        Xt = Xn.copy(); mt = mva.copy()
        Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
        with torch.no_grad():
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p_yes[:, m] = torch.sigmoid(logits).numpy()

    # Configs to test
    configs = [
        ("T+4-13",  list(range(4, 14)), 10),     # baseline canonical
        ("T+10-13", list(range(10, 14)), None),  # late only, no RH
        ("T+10-13", list(range(10, 14)), 10),    # late only, with RH
        ("T+11-13", list(range(11, 14)), 11),    # later only
        ("T+10-13", list(range(10, 14)), 10),    # repeat for edges
    ]
    edges = [1, 5, 10]

    print(f"Val windows: {n}\n")
    for source in ["NN", "2D table"]:
        print(f"━━━ {source} ━━━")
        print(f"  {'minutes':>10s}  {'rh':>4s}  {'edge':>5s} | {'acted':>5s}  {'win%':>5s}  "
              f"{'wagered':>10s}  {'P&L':>10s}  {'ROI':>8s}  {'bets/win':>8s}")
        seen = set()
        for label, minutes, rh in configs:
            for edge in edges:
                key = (label, rh, edge)
                if key in seen: continue
                seen.add(key)
                pnl = wag = 0.0; acted = wins = 0; total_bets = 0
                for i in range(n):
                    if source == "NN":
                        fair_fn = lambda mm, ii=i: float(p_yes[ii, mm])
                    else:
                        def fair_fn(mm, ii=i):
                            ret = float(Xva[ii, mm, 0]); pct = abs(ret) * 100
                            wr = fair_2d(mm, pct)
                            return wr if ret > 0 else (1 - wr)
                    r = simulate_one(Xva[i], mva[i], yva[i], fair_fn, minutes, edge, rh)
                    pnl += r["pnl"]; wag += r["wagered"]; total_bets += r["n_bets"]
                    if r["wagered"] > 0:
                        acted += 1
                        if r["pnl"] > 0: wins += 1
                roi = pnl/wag*100 if wag else 0
                wr  = wins/acted*100 if acted else 0
                bw  = total_bets/acted if acted else 0
                rh_str = f"T+{rh}" if rh else "off"
                print(f"  {label:>10s}  {rh_str:>4s}  {edge:>3d}c | {acted:>5d}  "
                      f"{wr:>4.0f}%  ${wag:>8.0f}  ${pnl:>+8.0f}  {roi:>+7.2f}%  {bw:>7.2f}")
        print()


if __name__ == "__main__":
    main()
