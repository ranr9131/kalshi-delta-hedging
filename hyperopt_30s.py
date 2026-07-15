"""
Hyperparameter sweep for the 30-second DH strategy.

Re-uses simulate_dh.py's per-window logic but iterates over (SIGMOID_K,
SIGMOID_CENTER, SIGMOID_MAX_MULT, MISPRICING_K, MISPRICING_MAX) combinations.
Reports each combo's total PnL, ROI, win rate, and tail metrics.

Usage:
    python hyperopt_30s.py                # default coarse grid
    python hyperopt_30s.py --quick        # tiny grid for sanity
    python hyperopt_30s.py --minutes 5-13 # match live trader minutes

Notes:
- Uses cached BTC + Kalshi data (data/cache/). Run once, fast after first load.
- Reports the TARGET-mode strategy (matches live trader). Additive ignored.
- Sweep space is intentionally coarse; refine after picking a region.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

import simulate_dh
import btc_data
import kalshi_client
from config import CACHE_DIR, DATA_DAYS, KALSHI_SERIES, BTC_SYMBOL, LOGS_DIR


def patch_params(sigmoid_k, sigmoid_center, sigmoid_max, misp_k, misp_max):
    """Override simulate_dh module-level params for this run."""
    simulate_dh.SIGMOID_K        = sigmoid_k
    simulate_dh.SIGMOID_CENTER   = sigmoid_center
    simulate_dh.SIGMOID_MAX_MULT = sigmoid_max
    simulate_dh.MISPRICING_K     = misp_k
    simulate_dh.MISPRICING_MAX   = misp_max


def run_sim(markets, btc_prices, dh_minutes, use_2d):
    """Run target-mode simulation across all markets, return aggregated metrics."""
    target_rows = []
    for m in markets:
        _, tgt = simulate_dh.simulate_market_dh(
            m, btc_prices, dh_minutes=dh_minutes,
            dynamic_fair_price=False, dead_zone=0.0, fair_price_2d=use_2d,
        )
        if tgt is not None:
            target_rows.append(tgt)

    if not target_rows:
        return None

    df = pd.DataFrame(target_rows)
    total_pnl = df["total_pnl"].sum()
    total_wagered = df["total_wagered"].sum()
    n = len(df)
    n_wins = (df["total_pnl"] > 0).sum()
    win_rate = n_wins / n if n else 0
    roi = total_pnl / total_wagered * 100 if total_wagered else 0
    worst = df["total_pnl"].min()
    p5 = df["total_pnl"].quantile(0.05)
    avg_wagered = df["total_wagered"].mean()
    sharpe_like = df["total_pnl"].mean() / df["total_pnl"].std() if df["total_pnl"].std() > 0 else 0

    return {
        "n_windows": n,
        "total_pnl": round(total_pnl, 2),
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round(roi, 3),
        "win_rate": round(win_rate, 4),
        "worst_window": round(worst, 2),
        "p5_window": round(p5, 2),
        "avg_wagered": round(avg_wagered, 2),
        "sharpe_like": round(sharpe_like, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", default="4-13",
                   help="Minute range, e.g. '4-13' (matches live DH window)")
    p.add_argument("--quick", action="store_true", help="Tiny grid for testing")
    p.add_argument("--use-2d", action="store_true", default=True,
                   help="Use 2D empirical fair-price table (default on)")
    p.add_argument("--out", default=os.path.join(LOGS_DIR, "hyperopt_30s_results.csv"))
    args = p.parse_args()

    start_min, end_min = map(int, args.minutes.split("-"))
    dh_minutes = list(range(start_min, end_min + 1))

    # Load 2D table once
    if args.use_2d:
        csv_2d = os.path.join(LOGS_DIR, "minute_analysis_2d.csv")
        if not os.path.exists(csv_2d):
            print(f"ERROR: {csv_2d} not found. Run analyze_minutes_2d.py first.")
            sys.exit(1)
        simulate_dh._load_2d_table(csv_2d)
        print(f"Loaded 2D fair price table: {len(simulate_dh._FAIR_PRICE_2D)} cells")

    print(f"Loading market list (last {DATA_DAYS} days)...")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    markets = [m for m in markets if m.get("result") in ("yes", "no")]
    print(f"  {len(markets)} settled markets")

    print("Loading BTC prices...")
    from datetime import datetime
    timestamps = [datetime.fromisoformat(m["open_time"].replace("Z","+00:00")).timestamp()
                  for m in markets if m.get("open_time")]
    range_start = int(min(timestamps)) - 600
    range_end   = int(max(timestamps)) + 1800
    btc_prices = btc_data.fetch_btc_prices(range_start, range_end)
    print(f"  {len(btc_prices)} price points")

    # ── Define sweep space ────────────────────────────────────────────────────
    if args.quick:
        sweep = list(itertools.product(
            [10, 20],          # sigmoid_k
            [0.05, 0.10],      # sigmoid_center
            [3.0],             # sigmoid_max
            [8.0],             # misp_k
            [2.0],             # misp_max
        ))
    else:
        sweep = list(itertools.product(
            [10, 15, 20, 30],          # sigmoid_k:        steepness of f(BTC %)
            [0.05, 0.08, 0.10, 0.15],  # sigmoid_center:   inflection of f
            [2.0, 3.0, 4.0],            # sigmoid_max:      max multiplier
            [4.0, 8.0, 12.0, 16.0],     # misp_k:           steepness of g(mispricing)
            [1.5, 2.0, 2.5],            # misp_max:         max mispricing multiplier
        ))

    print(f"\nSweeping {len(sweep)} parameter combinations over {len(markets)} markets...")
    print(f"DH minutes: T+{start_min}..T+{end_min}\n")

    results = []
    baseline = None
    for i, (sk, sc, sm, mk, mm) in enumerate(sweep, 1):
        patch_params(sk, sc, sm, mk, mm)
        r = run_sim(markets, btc_prices, dh_minutes, args.use_2d)
        if r is None:
            continue
        r.update({
            "sigmoid_k": sk, "sigmoid_center": sc, "sigmoid_max": sm,
            "mispricing_k": mk, "mispricing_max": mm,
        })
        results.append(r)
        # Track baseline (current production params)
        if (sk, sc, sm, mk, mm) == (20.0, 0.10, 3.0, 8.0, 2.0):
            baseline = r
        print(f"[{i:>3}/{len(sweep)}] k={sk:>4} ctr={sc:.2f} max={sm:.1f} "
              f"mk={mk:>4.1f} mm={mm:.1f}  →  pnl=${r['total_pnl']:>8.2f}  "
              f"roi={r['roi_pct']:>5.2f}%  worst=${r['worst_window']:>7.2f}")

    df = pd.DataFrame(results)
    df = df.sort_values("total_pnl", ascending=False)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n=== Top 15 by total PnL ===")
    print(df.head(15).to_string(index=False))

    if baseline:
        print(f"\n=== Current production params (k=20, ctr=0.10, max=3.0, mk=8, mm=2) ===")
        print(f"  pnl=${baseline['total_pnl']:.2f}  roi={baseline['roi_pct']:.2f}%  "
              f"worst=${baseline['worst_window']:.2f}  p5=${baseline['p5_window']:.2f}")
        best = df.iloc[0]
        print(f"\n=== Best params ===")
        print(f"  k={best['sigmoid_k']} ctr={best['sigmoid_center']} max={best['sigmoid_max']} "
              f"mk={best['mispricing_k']} mm={best['mispricing_max']}")
        print(f"  pnl=${best['total_pnl']:.2f}  roi={best['roi_pct']:.2f}%  "
              f"worst=${best['worst_window']:.2f}  Δ vs baseline: ${best['total_pnl']-baseline['total_pnl']:+.2f}")

    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
