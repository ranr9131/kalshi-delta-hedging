"""
Does NN14's paper edge survive REALISTIC execution?

The live taker log showed real fills land ~5c worse than the backtest's
'mid + 4c' assumption. This re-runs the NN14 walkforward, training each fold
ONCE, then re-simulating at a sweep of slippage levels (and a matching minimum
edge) to find the break-even point.

Reuses train/predict from sharpe_eval.py so the model is identical.
"""
import os, sys, math
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sharpe_eval as se   # train_nn14, predict_minutes, constants, data path

BASE_STAKE = se.BASE_STAKE; MIN_BET = se.MIN_BET
MAX_FILL = se.MAX_FILL; MAX_LEGS = se.MAX_LEGS; FEE_KEEP = se.FEE_KEEP
RH_TRIGGER = se.RH_TRIGGER; MAX_HEDGE_F = se.MAX_HEDGE_F
MINUTES_LATE = se.MINUTES_LATE
f_btc = se.f_btc; g_misp = se.g_misp; time_decay = se.time_decay


def simulate(X, mask, y, p_yes, slip_c, min_edge_c):
    """Same engine as sharpe_eval.simulate_window but slip & min-edge parametrized.
    Also returns acted-trade (pred_fair, outcome) pairs for Brier."""
    resolved_yes = (y == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = yes_c = no_c = 0.0
    legs = 0
    briers = []
    slip = slip_c / 100.0
    for m in MINUTES_LATE:
        if legs >= MAX_LEGS: break
        if not mask[m]: continue
        ret = float(X[m, 0]); kal = float(X[m, 4])
        if not (0.01 < kal < 0.99) or ret == 0: continue
        up = ret > 0; pct = abs(ret) * 100
        yes_fill = min(MAX_FILL, kal + slip)
        no_fill = min(MAX_FILL, (1 - kal) + slip)
        fair_y = float(p_yes[m])
        fair = fair_y if up else (1 - fair_y)
        mispr = fair - (yes_fill if up else no_fill)
        if mispr * 100 < min_edge_c:
            target = 0.0
        else:
            target = BASE_STAKE * f_btc(pct) * g_misp(mispr) * time_decay(m)
        if up:
            gap = max(0.0, target - yes_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and yes_fill < MAX_FILL:
                yes_bets.append((gap, yes_fill)); yes_exp += gap
                yes_c += gap / yes_fill; legs += 1
                briers.append((fair_y, 1.0 if resolved_yes else 0.0))
        else:
            gap = max(0.0, target - no_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and no_fill < MAX_FILL:
                no_bets.append((gap, no_fill)); no_exp += gap
                no_c += gap / no_fill; legs += 1
                briers.append((fair_y, 1.0 if resolved_yes else 0.0))
        if m >= 10 and legs < MAX_LEGS:
            if up and no_exp >= RH_TRIGGER and no_c > 0 and yes_fill <= MAX_HEDGE_F:
                h = no_c * yes_fill
                if h >= MIN_BET:
                    yes_bets.append((h, yes_fill)); yes_exp += h; yes_c += h/yes_fill; legs += 1
            elif (not up) and yes_exp >= RH_TRIGGER and yes_c > 0 and no_fill <= MAX_HEDGE_F:
                h = yes_c * no_fill
                if h >= MIN_BET:
                    no_bets.append((h, no_fill)); no_exp += h; no_c += h/no_fill; legs += 1
    pnl_yes = sum((1-fp)*(s/fp)*FEE_KEEP if resolved_yes else -s for s, fp in yes_bets)
    pnl_no = sum((1-fp)*(s/fp)*FEE_KEEP if not resolved_yes else -s for s, fp in no_bets)
    return pnl_yes + pnl_no, yes_exp + no_exp, briers


def main():
    z = np.load(se.DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y)
    print(f"Dataset: {n} windows, "
          f"{datetime.fromtimestamp(int(ts[0]),tz=timezone.utc).date()} -> "
          f"{datetime.fromtimestamp(int(ts[-1]),tz=timezone.utc).date()}")

    fold_starts = [int(n*f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size = int(n*0.15)
    folds = [(s, min(s+fold_size, n)) for s in fold_starts if s > 100 and min(s+fold_size, n)-s >= 100]

    # train each fold once, cache predictions
    cached = []
    for s, e in folds:
        print(f"  training fold test n={e-s} ...", flush=True)
        model, mean, std = se.train_nn14(X[:s], mask[:s], y[:s])
        p14 = se.predict_minutes(model, mean, std, X[s:e], mask[s:e])
        cached.append((s, e, p14))

    # NOTE: slip_c is TOTAL fill penalty vs mid-ish kalshi_close.
    #   4c  = original optimistic backtest
    #   9c  = empirically measured real taker cost (mid+4c was ~5c too good)
    sweep = [(4, 10), (6, 10), (8, 12), (9, 12), (10, 14), (12, 16)]
    print("\nslip_c  min_edge  acted   ROI%      PnL        Brier@acted")
    print("-" * 62)
    for slip_c, min_edge in sweep:
        tot_pnl = tot_wag = 0.0; acted = 0; briers = []
        for s, e, p14 in cached:
            for i in range(e - s):
                pnl, wag, br = simulate(X[s+i], mask[s+i], y[s+i], p14[i], slip_c, min_edge)
                if wag > 0:
                    acted += 1; tot_pnl += pnl; tot_wag += wag; briers.extend(br)
        roi = tot_pnl/tot_wag*100 if tot_wag else 0
        if briers:
            bp = np.array([b[0] for b in briers]); bo = np.array([b[1] for b in briers])
            brier = float(np.mean((bp - bo)**2))
        else:
            brier = float("nan")
        tag = "  <- original" if slip_c == 4 else ("  <- measured real" if slip_c == 9 else "")
        print(f"{slip_c:>4}c   {min_edge:>5}c   {acted:>5}  {roi:>+7.1f}  ${tot_pnl:>+10,.0f}   {brier:.4f}{tag}")


if __name__ == "__main__":
    main()
