"""
Expected slippage vs order size, from the passive book log (book_logger.py).

For each depth snapshot we walk the real ladder to find the VWAP to buy N
contracts, then measure how far that is from:
  - TOUCH : the best price (paying the top of book once)
  - MID   : (best yes_bid + best yes_ask)/2 mapped to the side -- what the
            backtest assumes.

This is the LIQUIDITY / book-walk component of slippage, measured with zero
trades. The remaining (execution/latency/adverse-selection) piece needs real
fills -- see the trader's actual_fill column + reconcile_fills.py.

Usage:
    python depth_slippage.py [book_log.jsonl]
"""

import json
import os
import statistics as st
import sys

SIZES = [1, 5, 10, 25, 50, 100, 200]


def vwap(ladder, n):
    """VWAP (dollars) to buy n contracts walking `ladder` [[price,size],...];
    returns (vwap, filled, exhausted)."""
    remaining, cost, got = float(n), 0.0, 0.0
    for price, size in ladder:
        take = min(remaining, float(size))
        cost += take * price
        got += take
        remaining -= take
        if remaining <= 1e-9:
            break
    return ((cost / got) if got > 0 else None, got, remaining > 1e-9)


def side_mid(rec, side):
    yb, ya = rec.get("yes_bid"), rec.get("yes_ask")
    if yb is None or ya is None:
        return None
    mid = (yb + ya) / 2
    return mid if side == "yes" else (1 - mid)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "book_log.jsonl")
    if not os.path.exists(path):
        print(f"No book log at {path}. Run book_logger.py first.")
        return

    snaps = [json.loads(ln) for ln in open(path) if ln.strip()]
    print(f"Loaded {len(snaps)} snapshots from {os.path.basename(path)}\n")

    for side, key in (("yes", "yes_asks"), ("no", "no_asks")):
        print(f"{'='*74}\n  BUY {side.upper()} — expected slippage by size (cents/contract)\n{'='*74}")
        print(f"  {'size':>5} {'n':>6} {'vs TOUCH med':>13} {'vs MID med':>11} {'vs MID p90':>11} {'can''t fill':>10}")
        for n in SIZES:
            tvs, mvs, exhausted, usable = [], [], 0, 0
            for rec in snaps:
                ladder = rec.get(key) or []
                if not ladder:
                    continue
                vw, filled, ex = vwap(ladder, n)
                if vw is None:
                    continue
                usable += 1
                if ex:
                    exhausted += 1
                touch = ladder[0][0]
                tvs.append((vw - touch) * 100)
                mid = side_mid(rec, side)
                if mid is not None:
                    mvs.append((vw - mid) * 100)
            if not tvs:
                continue
            tmed = sorted(tvs)[len(tvs) // 2]
            mmed = sorted(mvs)[len(mvs) // 2] if mvs else float("nan")
            mp90 = sorted(mvs)[min(len(mvs) - 1, int(0.9 * len(mvs)))] if mvs else float("nan")
            pct_ex = 100 * exhausted / usable if usable else 0
            print(f"  {n:>5} {usable:>6} {tmed:>+12.2f}c {mmed:>+10.2f}c {mp90:>+10.2f}c {pct_ex:>9.1f}%")
        print()

    print("Reading: 'vs MID' is the tax your backtest ignores; 'can't fill' is how")
    print("often the book was too thin to absorb that size in a single sweep.")


if __name__ == "__main__":
    main()
