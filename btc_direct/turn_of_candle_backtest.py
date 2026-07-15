"""
Test the turn-of-the-candle anomaly (research candidate #1) on our own data.

Published claim (verified 3-0 in deep research, PMC10015199): BTC 1-min returns
are positive (~+0.58 bp/min) in minutes 0/15/30/45 of each hour and negative on
average elsewhere, through Aug 2022. Question: does it replicate on our
Feb-Jul 2026 Coinbase data, and does it survive costs?

Usage: python3 btc_direct/turn_of_candle_backtest.py
"""

import glob
import json
import math
import os
import statistics

CACHE_GLOB = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "btc_cb_*.json")


def main():
    prices = {}
    for path in sorted(glob.glob(CACHE_GLOB)):
        with open(path) as f:
            for ts, px in json.load(f).items():
                prices[int(ts)] = float(px)
    days = (max(prices) - min(prices)) / 86400
    print(f"{len(prices):,} 1-min closes, {days:.0f} days")

    turn, other = [], []
    for ts, px in prices.items():
        prev = prices.get(ts - 60)
        if prev is None:
            continue
        r_bps = (px - prev) / prev * 1e4
        minute = (ts // 60) % 60
        # return over the candle-turn minute: close[m-1] -> close[m], m in {0,15,30,45}
        (turn if minute % 15 == 0 else other).append(r_bps)

    for name, xs in (("turn (0/15/30/45)", turn), ("other minutes", other)):
        m = statistics.fmean(xs)
        sd = statistics.stdev(xs)
        t = m / (sd / math.sqrt(len(xs)))
        print(f"{name:>18}: n={len(xs):,} mean={m:+.4f} bp/min  t={t:+.2f}")

    diff = statistics.fmean(turn) - statistics.fmean(other)
    print(f"\nturn-minus-other spread: {diff:+.4f} bp/min")
    per_day = statistics.fmean(turn) * 96  # 96 turn-minutes per day, 1 unit long each
    print(f"gross if long every turn-minute: {per_day:+.2f} bp/day before costs")
    print("(round-trip taker cost 5 bp/trade x 96 trades/day = -480 bp/day; "
          "maker/zero-fee execution is mandatory for any tradable version)")


if __name__ == "__main__":
    main()
