"""
Question 1: does 15-minute continuation survive magnitude weighting and costs?

Strategy under test: at minute M of each 15-minute window (aligned :00/:15/:30/:45),
if BTC has moved from the window-open price, take a 1x position in the move's
direction; exit at window close. P&L is proportional to the move (bps), unlike
the Kalshi binary which pays fixed $1 on direction alone.

No fitted parameters — a fixed rule swept over (entry minute, entry-move filter),
so the whole history is effectively out-of-sample. Costs modeled as a flat
round-trip in bps (default 5 = taker in + taker out on a decent perp venue,
spread included; 11-minute holds make funding negligible).

Usage: python3 btc_direct/continuation_backtest.py [--cost-bps 5]
"""

import argparse
import glob
import json
import os
import statistics

CACHE_GLOB = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "btc_cb_*.json")
WINDOW_SEC = 900


def load_prices():
    prices = {}
    for path in sorted(glob.glob(CACHE_GLOB)):
        with open(path) as f:
            day = json.load(f)
        for ts, px in day.items():
            prices[int(ts)] = float(px)
    return prices


def run(prices, entry_minute, min_entry_move_bps, cost_bps):
    """Returns list of net-bps P&L, one per trade."""
    ts_min, ts_max = min(prices), max(prices)
    start = (ts_min // WINDOW_SEC + 1) * WINDOW_SEC
    gross_list = []
    t = start
    while t + WINDOW_SEC <= ts_max:
        p0 = prices.get(t)
        pe = prices.get(t + entry_minute * 60)
        px = prices.get(t + WINDOW_SEC)
        t += WINDOW_SEC
        if p0 is None or pe is None or px is None:
            continue
        entry_move_bps = (pe - p0) / p0 * 1e4
        if abs(entry_move_bps) < min_entry_move_bps or entry_move_bps == 0:
            continue
        direction = 1 if entry_move_bps > 0 else -1
        gross = direction * (px - pe) / pe * 1e4
        gross_list.append(gross)
    return gross_list


def summarize(gross_list, cost_bps):
    if not gross_list:
        return None
    net = [g - cost_bps for g in gross_list]
    n = len(net)
    hit = sum(1 for g in gross_list if g > 0) / n
    return {
        "n": n,
        "hit_gross": hit,
        "mean_gross_bps": statistics.fmean(gross_list),
        "mean_net_bps": statistics.fmean(net),
        "total_net_bps": sum(net),
        "stdev_bps": statistics.stdev(gross_list) if n > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=5.0, help="round-trip cost in bps")
    args = ap.parse_args()

    prices = load_prices()
    days = (max(prices) - min(prices)) / 86400
    print(f"Loaded {len(prices):,} 1-min closes spanning {days:.0f} days "
          f"({min(prices)} .. {max(prices)})")
    print(f"Round-trip cost: {args.cost_bps} bps\n")

    header = (f"{'entry_min':>9} {'min_move':>8} {'n':>6} {'hit%':>6} "
              f"{'gross_bps':>9} {'net_bps':>8} {'total_net':>10} {'net$/day/10k':>12}")
    print(header)
    print("-" * len(header))
    for entry_minute in (2, 4, 6, 8, 10):
        for min_move in (0.0, 1.0, 2.0, 5.0, 10.0):
            s = summarize(run(prices, entry_minute, min_move, args.cost_bps), args.cost_bps)
            if s is None or s["n"] < 30:
                continue
            per_day = s["total_net_bps"] / days
            print(f"{entry_minute:>9} {min_move:>8.1f} {s['n']:>6} {s['hit_gross']*100:>6.1f} "
                  f"{s['mean_gross_bps']:>9.3f} {s['mean_net_bps']:>8.3f} "
                  f"{s['total_net_bps']:>10.0f} {per_day/1e4*10000:>12.2f}")

    print("\nhit% = share of trades where the continuation direction was gross-profitable")
    print("net$/day/10k = average daily P&L trading $10,000 notional per signal")


if __name__ == "__main__":
    main()
