"""
Backtest: NN fair-price vs 2D-table fair-price, head-to-head on the same
held-out val set, using a simple single-bet-at-T+10 strategy.

This avoids touching simulate_dh.py / the live trader. The strategy:

    At T+10 of each market:
      1. Look at BTC direction from open.
      2. Get a "fair price" for the directional side (from NN or 2D table).
      3. If (fair - fill_price) >= MIN_EDGE_CENTS / 100, place a bet of size
         BASE_STAKE × sigmoid(|move|) × sigmoid(mispricing).
      4. Compute P&L: win = stake × (1 - fill) / fill × 0.93,
                      lose = -stake.

The 2D table path uses the same lookup the trader uses today. The NN path
uses the trained model from checkpoints/best.pt.

Run after: build_dataset.py + train.py.
"""

import os
import sys
import csv
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best.pt")
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")

VAL_FRAC         = 0.20
DECISION_MINUTE  = 10
BASE_STAKE       = 100.0   # match simulate_dh.py STAKE for direct ROI comparison
SLIP_CENTS       = 4
MIN_EDGE_CENTS   = 1
FEE_KEEP         = 0.93
MAX_FILL         = 0.97    # match live MAX_FILL_PRICE
# Sigmoid params (same as simulate_dh.py)
SIG_CENTER       = 0.10
SIG_K            = 20.0
SIG_MAX          = 3.0
MISPR_K          = 8.0
MISPR_MAX        = 2.0


def sigmoid_btc(pct_abs):
    """Smooth scaling on |BTC move|, 0..SIG_MAX."""
    x = SIG_K * (pct_abs - SIG_CENTER)
    return SIG_MAX / (1.0 + math.exp(-x))


def sigmoid_mispr(mispr):
    """1×at zero edge, up to MISPR_MAX at high edge, near 0 at very negative."""
    x = MISPR_K * mispr
    # symmetric around 1.0
    return 1.0 + (MISPR_MAX - 1.0) / (1.0 + math.exp(-x)) - (MISPR_MAX - 1.0) / 2


_2D_BUCKETS = [
    (0.000, 0.05), (0.050, 0.10), (0.100, 0.20),
    (0.200, 0.50), (0.500, float("inf")),
]
_2D_BUCKET_LABELS = [
    "0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+",
]


def bucket_for(pct):
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= pct < hi:
            return i
    return len(_2D_BUCKETS) - 1


def load_2d_table():
    table = {}
    label_to_idx = {lbl: i for i, lbl in enumerate(_2D_BUCKET_LABELS)}
    with open(TABLE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            m = int(row["minute"]); bi = label_to_idx.get(row["bucket"])
            if bi is None: continue
            table[(m, bi)] = float(row["win_rate"])
    return table


def trade(fair_dir, kalshi_yes_at_t10, direction_up, pct_abs, resolved_yes):
    """Return (pnl, wagered, side_str). Single bet on directional side.

    fair_dir  = P(directional side wins) — from NN or 2D table
    """
    slip = SLIP_CENTS / 100.0
    fill_yes = min(MAX_FILL, kalshi_yes_at_t10 + slip)
    fill_no  = min(MAX_FILL, (1.0 - kalshi_yes_at_t10) + slip)
    if direction_up:
        side = "yes"
        fill = fill_yes
    else:
        side = "no"
        fill = fill_no
    mispr = fair_dir - fill
    if mispr * 100 < MIN_EDGE_CENTS:
        return 0.0, 0.0, "skip"
    f = sigmoid_btc(pct_abs)
    g = sigmoid_mispr(mispr)
    stake = BASE_STAKE * f * g
    if stake < 1.0:
        return 0.0, 0.0, "tiny"
    contracts = stake / fill
    won = (side == "yes" and resolved_yes) or (side == "no" and not resolved_yes)
    if won:
        pnl = contracts * (1.0 - fill) * FEE_KEEP
    else:
        pnl = -contracts * fill
    return pnl, stake, side


def main():
    # Load val data + NN
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts)
    X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n_val = int(len(y) * VAL_FRAC); n_train = len(y) - n_val
    Xva, mva, yva = X[n_train:], mask[n_train:], y[n_train:]
    print(f"Val markets: {len(yva)}")

    # Truncate at T+DECISION_MINUTE for NN
    Xva_nn  = Xva.copy(); mva_nn = mva.copy()
    Xva_nn[:, DECISION_MINUTE+1:, :] = 0.0
    mva_nn[:, DECISION_MINUTE+1:]    = False

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"], dtype=np.float32)
    model = TSWinPredictor(n_features=7)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    Xn = ((Xva_nn - mean) / std).astype(np.float32)
    with torch.no_grad():
        p_yes_nn = torch.sigmoid(model(torch.from_numpy(Xn), torch.from_numpy(mva_nn))).numpy()

    table = load_2d_table()

    # Run both strategies
    pnl_nn = 0.0; wag_nn = 0.0; n_bets_nn = 0; wins_nn = 0
    pnl_tb = 0.0; wag_tb = 0.0; n_bets_tb = 0; wins_tb = 0
    n_skip_no_data = 0

    for i in range(len(yva)):
        if not mva[i, DECISION_MINUTE]:
            n_skip_no_data += 1
            continue
        ret_t = float(Xva[i, DECISION_MINUTE, 0])
        kal_t = float(Xva[i, DECISION_MINUTE, 2])
        if not (0.01 < kal_t < 0.99):
            continue
        direction_up = ret_t > 0
        pct_abs      = abs(ret_t) * 100
        resolved_yes = bool(yva[i] == 1)

        # NN: P(directional wins)
        nn_pyes = float(p_yes_nn[i])
        fair_nn = nn_pyes if direction_up else (1.0 - nn_pyes)
        pnl, wag, side = trade(fair_nn, kal_t, direction_up, pct_abs, resolved_yes)
        pnl_nn += pnl; wag_nn += wag
        if wag > 0:
            n_bets_nn += 1
            if pnl > 0: wins_nn += 1

        # 2D table: same idea, fair = P(direction wins)
        bi = bucket_for(pct_abs)
        win_rate = table.get((DECISION_MINUTE, bi), 0.5)
        fair_tb = win_rate
        pnl, wag, side = trade(fair_tb, kal_t, direction_up, pct_abs, resolved_yes)
        pnl_tb += pnl; wag_tb += wag
        if wag > 0:
            n_bets_tb += 1
            if pnl > 0: wins_tb += 1

    print(f"\n── Single-bet-at-T+{DECISION_MINUTE} strategy on val set ──")
    print(f"  Skipped (no data at T+{DECISION_MINUTE}): {n_skip_no_data}")
    print(f"\n  {'source':<12s}  {'bets':>5s}  {'win%':>6s}  {'wagered':>10s}  {'P&L':>10s}  {'ROI':>8s}")
    for label, p, w, n, wn in [
        ("2D table",  pnl_tb, wag_tb, n_bets_tb, wins_tb),
        ("NN",        pnl_nn, wag_nn, n_bets_nn, wins_nn),
    ]:
        roi = p/w*100 if w else 0
        wr  = wn/n*100 if n else 0
        print(f"  {label:<12s}  {n:>5d}  {wr:>5.1f}%  ${w:>8.0f}  ${p:>+8.0f}  {roi:>+7.2f}%")

    print(f"\nAt $10 live stake (scale by 0.1):")
    print(f"  NN P&L:       ${pnl_nn*0.1:+.2f} on ${wag_nn*0.1:.2f} wagered")
    print(f"  2D table P&L: ${pnl_tb*0.1:+.2f} on ${wag_tb*0.1:.2f} wagered")


if __name__ == "__main__":
    main()
