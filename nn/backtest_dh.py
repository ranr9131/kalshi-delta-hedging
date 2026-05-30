"""
DH-style backtest: NN multi-minute fair-price vs 2D-table fair-price, both run
through the SAME multi-bet target-mode logic the live trader uses.

Strategy template (mirrors simulate_dh.py canonical config):
  - Target mode (bet the gap to computed target each minute)
  - Minutes T+4 through T+13
  - Slippage: 4¢ above mid
  - Min edge: 1¢
  - Time-decay multipliers: ×0.4 / ×0.8 / ×1.2
  - Reversal hedge at T+10, trigger $10 wrong-side exposure, cap 0.80
  - Per-window leg cap: 2 (matches live MAX_LEGS_PER_WINDOW)
  - Early skip: minute ≤ 5 and |move| < 0.05%

The 2D-table baseline replicates exactly what simulate_dh.py would do on the
same windows. The NN path replaces the (minute, magnitude bucket) lookup with
NN inference at each decision minute.

Runs only on the held-out val set (last 20% of windows by time) — the same
markets the NN never saw during training.
"""

import os
import sys
import csv
import math
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best_multi.pt")
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")

# ── Strategy config (must match simulate_dh.py canonical) ─────────────────────
VAL_FRAC     = 0.20
MINUTES      = list(range(4, 14))          # T+4..T+13
BASE_STAKE   = 100.0                       # $100 to match simulate_dh.py STAKE
MIN_BET      = 5.0
SLIP_C       = 4                           # cents above mid
MIN_EDGE_C   = 1
RH_MIN       = 10
RH_TRIGGER   = 10.0
MAX_HEDGE_F  = 0.80
MAX_FILL     = 0.97
MAX_LEGS     = 2
EARLY_SKIP_M = 5
EARLY_SKIP_P = 0.05
FEE_RATE     = 0.07
FEE_KEEP     = 1.0 - FEE_RATE

SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0


def sigmoid_btc(p):  return SIG_MAX / (1.0 + math.exp(-SIG_K * (p - SIG_C)))
def sigmoid_misp(m): return MISPR_MAX / (1.0 + math.exp(-MISPR_K * m))


def time_decay(minute):
    if minute < 7:   return 0.4
    if minute < 10:  return 0.8
    return 1.2


# ── 2D table lookup ───────────────────────────────────────────────────────────
_2D_BUCKETS = [(0.000,0.05),(0.050,0.10),(0.100,0.20),(0.200,0.50),(0.500,float("inf"))]
_2D_LABELS  = ["0.00-0.05%","0.05-0.10%","0.10-0.20%","0.20-0.50%","0.50%+"]
FAIR_PRICE_BY_MINUTE = {
    1:0.582,2:0.617,3:0.636,4:0.670,5:0.698,6:0.728,7:0.751,8:0.759,
    9:0.783,10:0.798,11:0.806,12:0.815,13:0.826,14:0.704,
}
_TABLE = {}; _MIN_N = 30


def _bucket(pct):
    for i,(lo,hi) in enumerate(_2D_BUCKETS):
        if lo <= pct < hi: return i
    return len(_2D_BUCKETS) - 1


def _load_table():
    l2i = {l:i for i,l in enumerate(_2D_LABELS)}
    with open(TABLE_PATH, newline="") as f:
        for r in csv.DictReader(f):
            m = int(r["minute"]); bi = l2i.get(r["bucket"])
            if bi is None: continue
            _TABLE[(m,bi)] = (float(r["win_rate"]), int(r["n"]))


def fair_2d(minute, pct):
    e = _TABLE.get((minute, _bucket(pct)))
    if e and e[1] >= _MIN_N:
        return e[0]
    return FAIR_PRICE_BY_MINUTE.get(minute, 0.7)


# ── Simulation core ───────────────────────────────────────────────────────────
def simulate_one(X_raw, mask, y_label, fair_fn_yes):
    """Run DH target mode on one window.

    X_raw    : (15, 7) feature tensor (unnormalized)
    mask     : (15,) bool — True where minute had data
    y_label  : 1.0 if YES wins
    fair_fn_yes(minute) → P(YES wins) given data through that minute

    Returns dict with totals.
    """
    yes_bets = []   # list of (stake, fill_price) — store actual fill
    no_bets  = []
    yes_exp = no_exp = 0.0
    yes_c   = no_c   = 0.0
    legs = 0

    for m in MINUTES:
        if legs >= MAX_LEGS:
            break
        if not mask[m]:
            continue

        ret_t = float(X_raw[m, 0])    # cumulative BTC return from T0
        kal_t = float(X_raw[m, 2])    # kalshi yes mid
        if not (0.01 < kal_t < 0.99): continue
        if ret_t == 0: continue

        direction_up = ret_t > 0
        pct_abs      = abs(ret_t) * 100.0

        slip = SLIP_C / 100.0
        yes_fill = min(MAX_FILL, kal_t + slip)
        no_fill  = min(MAX_FILL, (1.0 - kal_t) + slip)

        # Fair price (probability the directional side wins)
        p_yes = fair_fn_yes(m)
        fair  = p_yes if direction_up else (1.0 - p_yes)

        if direction_up:
            mispr = fair - yes_fill
            fill_use = yes_fill
        else:
            mispr = fair - no_fill
            fill_use = no_fill

        if mispr * 100 < MIN_EDGE_C:
            target = 0.0
        else:
            f = sigmoid_btc(pct_abs)
            g = sigmoid_misp(mispr)
            td = time_decay(m)
            target = BASE_STAKE * f * g * td

        # Early-skip overlay (mirror simulate_dh.py)
        if m <= EARLY_SKIP_M and pct_abs < EARLY_SKIP_P:
            target = 0.0

        # Target-mode gap
        if direction_up:
            gap = max(0.0, target - yes_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and fill_use < MAX_FILL:
                yes_bets.append((gap, yes_fill))
                yes_exp += gap
                yes_c   += gap / yes_fill
                legs    += 1
        else:
            gap = max(0.0, target - no_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and fill_use < MAX_FILL:
                no_bets.append((gap, no_fill))
                no_exp += gap
                no_c   += gap / no_fill
                legs    += 1

        # Reversal-hedge at T+10+
        if m >= RH_MIN and legs < MAX_LEGS:
            if direction_up:
                # Wrong-side: NO; hedge with YES if fill OK
                if no_exp >= RH_TRIGGER and no_c > 0 and yes_fill <= MAX_HEDGE_F:
                    hedge = no_c * yes_fill
                    if hedge >= MIN_BET:
                        yes_bets.append((hedge, yes_fill))
                        yes_exp += hedge
                        yes_c   += hedge / yes_fill
                        legs    += 1
            else:
                if yes_exp >= RH_TRIGGER and yes_c > 0 and no_fill <= MAX_HEDGE_F:
                    hedge = yes_c * no_fill
                    if hedge >= MIN_BET:
                        no_bets.append((hedge, no_fill))
                        no_exp += hedge
                        no_c   += hedge / no_fill
                        legs    += 1

    # P&L — fp stored is the actual fill price for each side.
    resolved_yes = (y_label == 1.0)
    pnl_yes = sum((1 - fp) * (s / fp) * FEE_KEEP if resolved_yes else -s
                  for s, fp in yes_bets)
    pnl_no  = sum((1 - fp) * (s / fp) * FEE_KEEP if not resolved_yes else -s
                  for s, fp in no_bets)
    total_pnl = pnl_yes + pnl_no
    total_wag = yes_exp + no_exp
    return {
        "n_yes": len(yes_bets), "n_no": len(no_bets),
        "yes_exp": yes_exp, "no_exp": no_exp,
        "pnl": total_pnl, "wagered": total_wag,
        "won": total_pnl > 0,
    }


def main():
    # Load val set
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n_val = int(len(y) * VAL_FRAC); n_train = len(y) - n_val
    Xva, mva, yva, tsva = X[n_train:], mask[n_train:], y[n_train:], ts[n_train:]
    print(f"Val windows: {len(yva)}")
    print(f"  date range: {datetime.fromtimestamp(int(tsva.min())).date()} -> "
          f"{datetime.fromtimestamp(int(tsva.max())).date()}")

    _load_table()
    print(f"  2D table loaded ({len(_TABLE)} cells)")

    # Load multi-minute NN
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"], dtype=np.float32)
    model = TSWinPredictor(n_features=7); model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"  NN multi-min loaded: {CKPT_PATH}")

    # Pre-compute NN P(YES) for all (market, decision_minute) cells.
    # Shape: (n_val, n_minutes)
    Xn = ((Xva - mean) / std).astype(np.float32)
    p_yes_nn = np.zeros((len(yva), 15), dtype=np.float32)
    for m in MINUTES:
        Xt = Xn.copy(); mt = mva.copy()
        Xt[:, m+1:, :] = 0.0
        mt[:, m+1:]    = False
        with torch.no_grad():
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p_yes_nn[:, m] = torch.sigmoid(logits).numpy()
    print(f"  NN predictions pre-computed for T+{MINUTES[0]}..T+{MINUTES[-1]}\n")

    # Run both strategies on the same val windows
    def run_strategy(name, fair_at_minute_fn_factory):
        total_pnl = total_wag = 0.0
        n_acted = n_wins = 0
        n_loss = 0
        per_window = []
        for i in range(len(yva)):
            fair_fn = fair_at_minute_fn_factory(i)
            r = simulate_one(Xva[i], mva[i], yva[i], fair_fn)
            total_pnl += r["pnl"]
            total_wag += r["wagered"]
            if r["wagered"] > 0:
                n_acted += 1
                if r["won"]: n_wins += 1
                else:        n_loss += 1
            per_window.append(r["pnl"])
        roi = total_pnl/total_wag*100 if total_wag else 0
        wr  = n_wins/n_acted*100 if n_acted else 0
        print(f"  {name:<10s}  acted={n_acted:>4d}  win%={wr:>5.1f}  "
              f"wagered=${total_wag:>8.0f}  P&L=${total_pnl:>+8.0f}  ROI={roi:>+6.2f}%")
        return total_pnl, total_wag, n_acted, n_wins, per_window

    # NN fair fn factory
    def nn_factory(idx):
        return lambda m: float(p_yes_nn[idx, m])

    # 2D table fair fn factory — fair = P(directional wins), and our sim wants P(YES)
    def table_factory(idx):
        def f(m):
            ret = float(Xva[idx, m, 0])
            pct = abs(ret) * 100
            wr = fair_2d(m, pct)
            # 2D table returns P(directional wins). Convert to P(YES wins).
            return wr if ret > 0 else (1.0 - wr)
        return f

    print(f"Strategy results (target mode, multi-bet, RH=10, leg cap=2):\n")
    print(f"  {'source':<10s}  {'acted':>5s}  {'win%':>5s}  {'wagered':>10s}  {'P&L':>10s}  {'ROI':>8s}")
    tb_pnl, tb_wag, tb_acted, tb_wins, tb_p = run_strategy("2D table",   table_factory)
    nn_pnl, nn_wag, nn_acted, nn_wins, nn_p = run_strategy("NN multi",   nn_factory)

    # Per-window pairwise comparison
    same   = sum(1 for a,b in zip(nn_p, tb_p) if abs(a-b) < 0.01)
    nn_won = sum(1 for a,b in zip(nn_p, tb_p) if a > b + 0.01)
    tb_won = sum(1 for a,b in zip(nn_p, tb_p) if b > a + 0.01)
    print(f"\nPer-window head-to-head:")
    print(f"  Same (±$0.01):           {same}")
    print(f"  NN > 2D by >$0.01:       {nn_won}")
    print(f"  2D > NN by >$0.01:       {tb_won}")
    print(f"  Avg per-window: NN ${np.mean(nn_p):+.2f}  vs  2D ${np.mean(tb_p):+.2f}")


if __name__ == "__main__":
    main()
