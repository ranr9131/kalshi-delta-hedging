"""
hyperopt.py

Grid search over key hyperparameters for the DH strategy.
Base config: 15-sec + wr-f + no-neg-mis (variant E).

Phase 1: WR_CENTER x min_pct  (WR_K=20, SIGMOID_MAX=3.0 fixed)
Phase 2: WR_K x SIGMOID_MAX   (best center+min_pct from phase 1 fixed)

Usage:
  python hyperopt.py
"""

import itertools
import math
import os

from config import CACHE_DIR, DATA_DAYS
from collect_15s_data import _parse_ts
from analyze_coin import fetch_markets, fetch_prices, build_2d_table
from simulate_15s import (
    simulate_one, summarise, build_2d_table_15s,
    WR_FALLBACK, FILL_SIM_BUFFER,
)

SERIES = "KXBTC15M"
SYMBOL = "BTC-USD"


def make_f_wr(center, k, max_val):
    def f(pct, wr):
        if wr is None:
            wr = WR_FALLBACK
        return max_val / (1 + math.exp(-k * (wr - center)))
    return f


def run(markets_15s, prices_1min, table_1min, table_15s,
        center, k, max_val, min_pct):
    f_fn = make_f_wr(center, k, max_val)
    res  = simulate_one(
        markets_15s, prices_1min, table_1min, table_15s,
        f_fn, use_15s=True, skip_neg_mis=True,
        fill_buf=FILL_SIM_BUFFER, min_pct=min_pct,
    )
    row = summarise("", res)
    if row:
        row.update(center=center, k=k, max_val=max_val, min_pct=min_pct)
    return row


def main():
    markets = fetch_markets(SERIES, DATA_DAYS)
    markets_15s = [
        m for m in markets
        if m.get("result") in ("yes", "no")
        and os.path.exists(os.path.join(CACHE_DIR, f"trades_15s_{m.get('ticker','')}.json"))
    ]
    print(f"Markets with 15s data: {len(markets_15s)}")

    timestamps   = [_parse_ts(m["open_time"]) for m in markets_15s if m.get("open_time")]
    prices       = fetch_prices(SYMBOL, min(timestamps) - 600, max(timestamps) + 1800)
    prices_1min  = {int(k): v for k, v in prices.items()}
    table_1min   = build_2d_table(markets_15s, prices, SERIES)
    table_15s    = build_2d_table_15s(markets_15s, prices_1min, SERIES)
    print("Data loaded.\n")

    all_results = []

    # ── Phase 1: WR_CENTER × min_pct ────────────────────────────────────────
    wr_centers = [0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725]
    min_pcts   = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]

    print(f"Phase 1: WR_CENTER × min_pct  (k=20, max=3.0)  —  {len(wr_centers)*len(min_pcts)} variants")
    print(f"  {'CTR':>6}  {'min%':>6}  {'ROI':>8}  {'Win%':>6}  {'Bets/w':>7}")
    for center, min_pct in itertools.product(wr_centers, min_pcts):
        row = run(markets_15s, prices_1min, table_1min, table_15s,
                  center=center, k=20, max_val=3.0, min_pct=min_pct)
        if row:
            all_results.append(row)
            marker = " *" if (center == 0.65 and min_pct == 0.0) else ""
            print(f"  {center:.3f}  {min_pct:.2f}%  {row['roi']:>+7.2f}%  {row['win_rate']:>5.1%}  {row['avg_bets']:>6.1f}{marker}")

    best_p1  = max(all_results, key=lambda r: r["roi"])
    best_ctr = best_p1["center"]
    best_mp  = best_p1["min_pct"]
    print(f"\nBest phase 1: center={best_ctr:.3f}  min_pct={best_mp:.2f}%  ROI={best_p1['roi']:+.2f}%\n")

    # ── Phase 2: WR_K × SIGMOID_MAX ─────────────────────────────────────────
    wr_ks    = [10, 15, 20, 25, 30, 40]
    sig_maxs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print(f"Phase 2: WR_K × SIGMOID_MAX  (center={best_ctr:.3f}, min%={best_mp:.2f})  —  {len(wr_ks)*len(sig_maxs)} variants")
    print(f"  {'K':>4}  {'MAX':>5}  {'ROI':>8}  {'Win%':>6}  {'Bets/w':>7}")
    for k, max_val in itertools.product(wr_ks, sig_maxs):
        row = run(markets_15s, prices_1min, table_1min, table_15s,
                  center=best_ctr, k=k, max_val=max_val, min_pct=best_mp)
        if row:
            all_results.append(row)
            marker = " *" if (k == 20 and max_val == 3.0) else ""
            print(f"  {k:>4}  {max_val:>5.1f}  {row['roi']:>+7.2f}%  {row['win_rate']:>5.1%}  {row['avg_bets']:>6.1f}{marker}")

    # ── Summary ─────────────────────────────────────────────────────────────
    all_results.sort(key=lambda r: -r["roi"])
    baseline = next(
        (r for r in all_results if r["center"] == 0.65 and r["min_pct"] == 0.0
         and r["k"] == 20 and r["max_val"] == 3.0), None
    )

    print(f"\n{'='*80}")
    print(f"  Top 15 configs  (base config E: center=0.650  min%=0.00  k=20  max=3.0)")
    if baseline:
        print(f"  Baseline ROI: {baseline['roi']:+.2f}%  bets/w {baseline['avg_bets']:.1f}")
    print(f"\n  {'CTR':>6}  {'min%':>6}  {'K':>4}  {'MAX':>5}  {'ROI':>8}  {'vs base':>8}  {'Win%':>6}  {'Bets/w':>7}  {'P&L':>12}")
    print("-" * 80)
    base_roi = baseline["roi"] if baseline else 0.0
    for r in all_results[:15]:
        marker = " <--" if r == all_results[0] else ""
        print(
            f"  {r['center']:.3f}  {r['min_pct']:.2f}%  {r['k']:>4}  {r['max_val']:>5.1f}"
            f"  {r['roi']:>+7.2f}%  {r['roi']-base_roi:>+7.2f}pp"
            f"  {r['win_rate']:>5.1%}  {r['avg_bets']:>6.1f}  ${r['pnl']:>+11,.0f}{marker}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
