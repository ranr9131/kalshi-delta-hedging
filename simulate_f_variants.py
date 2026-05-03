"""
simulate_f_variants.py

Compares different f-function designs:
  Option 2 — steeper magnitude sigmoid (K=20,30,40,60)
  Option 3 — win-rate-based f: f(win_rate_2d) instead of f(magnitude)

Usage:
  python simulate_f_variants.py              # BTC default
  python simulate_f_variants.py --series KXSOL15M --symbol SOL-USD
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from config import CACHE_DIR, LOGS_DIR, DATA_DAYS, STAKE, FEE_RATE
from analyze_coin import (
    fetch_markets, fetch_prices, fetch_candles,
    lookup_price, get_price_at, bucket_idx,
    build_2d_table, sig_mis,
    _2D_MIN_N, DH_MINUTES, MIN_BET,
    _2D_BUCKET_LABELS, _2D_BUCKETS,
)

# ── f-function variants ───────────────────────────────────────────────────────

SIGMOID_MAX_MULT = 3.0
BASELINE_CENTER  = 0.10
BASELINE_K       = 20.0

WR_FALLBACK = 0.698  # global win-rate fallback when 2D table has no data


def make_mag_f(k):
    """Option 2: magnitude sigmoid with steepness k."""
    def f(pct, wr=None):
        return SIGMOID_MAX_MULT / (1 + np.exp(-k * (pct - BASELINE_CENTER)))
    f.__name__ = f"mag_K{k}"
    return f


def make_wr_f(center, k=20.0):
    """Option 3: win-rate sigmoid — f depends on 2D win rate, not magnitude."""
    def f(pct, wr):
        if wr is None:
            wr = WR_FALLBACK
        return SIGMOID_MAX_MULT / (1 + np.exp(-k * (wr - center)))
    f.__name__ = f"wr_C{center:.2f}_K{k}"
    return f


# All variants to test
VARIANTS = [
    ("baseline  (mag K=20)",         make_mag_f(20)),
    ("opt2-mild (mag K=30)",          make_mag_f(30)),
    ("opt2-med  (mag K=40)",          make_mag_f(40)),
    ("opt2-steep(mag K=60)",          make_mag_f(60)),
    ("opt3-lo   (wr C=0.65)",         make_wr_f(0.65)),
    ("opt3-mid  (wr C=0.70)",         make_wr_f(0.70)),
    ("opt3-hi   (wr C=0.75)",         make_wr_f(0.75)),
    ("opt3-vhi  (wr C=0.80)",         make_wr_f(0.80)),
]


# ── f-value preview table ─────────────────────────────────────────────────────

def print_f_preview(variants):
    # Show f values at key (pct, win_rate) pairs
    sample_pts = [
        (0.003, 0.572),   # tiny move, bucket 0 win rate
        (0.010, 0.587),   # still tiny
        (0.050, 0.670),   # bucket 1 low
        (0.075, 0.722),   # bucket 1 high
        (0.150, 0.860),   # bucket 2
        (0.300, 0.930),   # bucket 3
        (0.600, 0.985),   # bucket 4
    ]
    labels = [f"{p:.3f}% (wr={w:.2f})" for p, w in sample_pts]
    col_w  = 10

    print("f-function preview (how much each variant bets per unit):")
    print(f"  {'Move / win-rate':<22}", end="")
    for lbl, _ in variants:
        print(f"  {lbl[:col_w]:>{col_w}}", end="")
    print()
    print("  " + "-" * (22 + (col_w + 2) * len(variants)))

    for pct, wr in sample_pts:
        row_lbl = f"{pct:.3f}% (wr={wr:.2f})"
        print(f"  {row_lbl:<22}", end="")
        for _, f_fn in variants:
            val = f_fn(pct, wr)
            print(f"  {val:>{col_w}.3f}", end="")
        print()
    print()


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(markets, prices, series, table_2d, f_fn):
    results = []

    for market in markets:
        open_iso = market.get("open_time", "")
        result   = market.get("result", "")
        if not open_iso or result not in ("yes", "no"):
            continue

        open_dt  = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
        t0       = int(open_dt.timestamp())
        resolved = result == "yes"
        coin_t0  = lookup_price(prices, t0)
        if coin_t0 is None:
            continue

        candles = fetch_candles(series, market.get("ticker", ""),
                                open_iso, market.get("close_time", ""))
        if not candles:
            continue

        k_t0 = candles[0]["yes_open"] if candles else None
        if k_t0 is None or not (0.01 < k_t0 < 0.99):
            continue

        yes_exp, no_exp = 0.0, 0.0
        yes_bets, no_bets = [], []

        for minute in DH_MINUTES:
            t      = t0 + minute * 60
            coin_t = lookup_price(prices, t)
            k_yes  = get_price_at(candles, t)
            if coin_t is None or k_yes is None or not (0.01 < k_yes < 0.99):
                continue

            direction_up = coin_t > coin_t0
            pct  = abs(coin_t - coin_t0) / coin_t0 * 100
            bi   = bucket_idx(pct)
            cell = table_2d.get((minute, bi))
            wr   = (cell["wins"] / cell["n"]) if (cell and cell["n"] >= _2D_MIN_N) else None
            fair = wr if wr is not None else WR_FALLBACK

            f = f_fn(pct, wr)

            if direction_up:
                mis      = fair - k_yes
                g        = sig_mis(mis)
                tgt_yes  = STAKE * f * g
                tgt_no   = 0.0
            else:
                mis      = fair - (1.0 - k_yes)
                g        = sig_mis(mis)
                tgt_yes  = 0.0
                tgt_no   = STAKE * f * g

            gap_yes = max(0.0, tgt_yes - yes_exp)
            gap_no  = max(0.0, tgt_no  - no_exp)

            if gap_yes >= MIN_BET:
                yes_bets.append((gap_yes, k_yes))
                yes_exp += gap_yes
            if gap_no >= MIN_BET:
                no_bets.append((gap_no, k_yes))
                no_exp += gap_no

        def _yes_pnl(s, p): return s*(1-p)/p*(1-FEE_RATE) if resolved else -s
        def _no_pnl(s, p):
            np_ = 1 - p
            return s*(1-np_)/np_*(1-FEE_RATE) if not resolved else -s

        pnl_yes = sum(_yes_pnl(s, p) for s, p in yes_bets)
        pnl_no  = sum(_no_pnl(s, p)  for s, p in no_bets)
        total_w = yes_exp + no_exp

        if total_w == 0:
            continue

        results.append({
            "pnl":      pnl_yes + pnl_no,
            "wagered":  total_w,
            "n_bets":   len(yes_bets) + len(no_bets),
            "win":      int((pnl_yes + pnl_no) > 0),
        })

    return results


def summarise(label, results):
    if not results:
        return None
    pnl      = sum(r["pnl"]     for r in results)
    wagered  = sum(r["wagered"] for r in results)
    win_rate = sum(r["win"]     for r in results) / len(results)
    roi      = pnl / wagered * 100
    avg_bets = sum(r["n_bets"]  for r in results) / len(results)
    return dict(label=label, windows=len(results), win_rate=win_rate,
                pnl=pnl, wagered=wagered, roi=roi, avg_bets=avg_bets)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--symbol", default="BTC-USD")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    markets = fetch_markets(args.series, DATA_DAYS)
    print(f"Markets: {len(markets)}\n")

    timestamps = [int(datetime.fromisoformat(m["open_time"].replace("Z","+00:00")).timestamp())
                  for m in markets if m.get("open_time")]
    prices = fetch_prices(args.symbol, min(timestamps)-600, max(timestamps)+1800)
    print(f"Prices:  {len(prices)} minute-prices\n")

    print("Building 2D table...")
    table_2d = build_2d_table(markets, prices, args.series)
    print("Done.\n")

    print_f_preview(VARIANTS)

    rows = []
    for label, f_fn in VARIANTS:
        print(f"  Simulating: {label} ...", end="\r")
        res = simulate(markets, prices, args.series, table_2d, f_fn)
        row = summarise(label, res)
        rows.append(row)
        print(f"  {label:<35}  ROI {row['roi']:+.2f}%  win {row['win_rate']:.1%}  bets/w {row['avg_bets']:.1f}  P&L ${row['pnl']:+,.0f}")

    # Summary comparison
    base = rows[0]
    print(f"\n{'='*75}")
    print(f"{'Variant':<35}  {'ROI':>8}  {'vs base':>8}  {'Win%':>6}  {'Bets/w':>7}  {'P&L':>12}")
    print("-" * 75)
    for r in rows:
        delta = r["roi"] - base["roi"]
        marker = " <--" if r["roi"] == max(x["roi"] for x in rows) else ""
        print(
            f"  {r['label']:<33}  {r['roi']:>+7.2f}%  {delta:>+7.2f}pp"
            f"  {r['win_rate']:>5.1%}  {r['avg_bets']:>6.1f}  ${r['pnl']:>+11,.0f}{marker}"
        )
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
