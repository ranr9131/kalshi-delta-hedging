"""
Slippage measurement from real Kalshi fills.

For every logged fill we compare the ACTUAL fill_price against two references:
  - MID   : kalshi_yes_mid (yes) or 1-mid (no) -- what the backtest ASSUMES.
  - TOUCH : yes_ask (yes) or 1-yes_bid (no)    -- the expected marketable price
            if you just cross the spread once.

slippage = fill_price - reference   (in cents; POSITIVE = you paid MORE = worse).

  vs MID   = half-spread + book-walk + adverse move  (the total tax the
             backtest ignores).
  vs TOUCH = book-walk + adverse move only  (execution quality beyond simply
             paying the spread).

Real fills = order_result == 'ok'. 'paper' rows are the simulator's own fill
model (shown separately for contrast). Weighted stats weight by contract count.
"""

import csv
import glob
import os
import statistics as st

LOG_GLOB = os.path.join(os.path.dirname(__file__), "live", "trade_log.*.csv")


def refs(row):
    """Return (mid_ref, touch_ref) = the price you'd expect to pay for bet_side."""
    yb, ya = float(row["yes_bid"]), float(row["yes_ask"])
    mid = float(row["kalshi_yes_mid"])
    if row["bet_side"] == "yes":
        return mid, ya
    else:  # buying NO: price = 1 - yes
        return 1 - mid, 1 - yb


def load():
    real, paper = [], []
    for path in sorted(glob.glob(LOG_GLOB)):
        with open(path) as f:
            for row in csv.DictReader(f):
                res = (row.get("order_result") or "").strip()
                # Prefer the ACTUAL executed price when the newer schema recorded
                # it (backfilled by reconcile_fills.py); else fall back to the
                # intended fill_price. Older logs only have the intended price.
                actual = (row.get("actual_fill") or "").strip()
                try:
                    fill = float(actual) if actual else float(row["fill_price"])
                    cnt = float(row["count"])
                    mid_ref, touch_ref = refs(row)
                except (ValueError, KeyError, TypeError):
                    continue
                if not (0 < fill < 1) or cnt <= 0:
                    continue
                book_exp = (row.get("book_expected_fill") or "").strip()
                rec = {
                    "src": os.path.basename(path), "side": row["bet_side"],
                    "fill": fill, "cnt": cnt, "mid": mid_ref, "touch": touch_ref,
                    "is_actual": bool(actual),
                    "spread": float(row["yes_ask"]) - float(row["yes_bid"]),
                    "slip_mid": (fill - mid_ref) * 100,      # cents
                    "slip_touch": (fill - touch_ref) * 100,  # cents
                    # true book-walk: actual executed vs what the live book promised
                    "slip_book": (fill - float(book_exp)) * 100 if (actual and book_exp) else None,
                }
                if res == "ok":
                    real.append(rec)
                elif res == "paper":
                    paper.append(rec)
    return real, paper


def wmean(recs, key):
    tot = sum(r["cnt"] for r in recs)
    return sum(r[key] * r["cnt"] for r in recs) / tot if tot else 0.0


def summarize(recs, title):
    print(f"\n{'='*70}\n  {title}   (n={len(recs)} fills, {int(sum(r['cnt'] for r in recs))} contracts)\n{'='*70}")
    if not recs:
        print("  (no fills)")
        return
    for key, label in (("slip_mid", "vs MID  (backtest assumption)"),
                       ("slip_touch", "vs TOUCH (execution quality)")):
        vals = sorted(r[key] for r in recs)
        n = len(vals)
        median = vals[n // 2]
        mean = st.mean(vals)
        w = wmean(recs, key)
        p90 = vals[min(n - 1, int(0.90 * n))]
        worst = vals[-1]
        print(f"\n  {label}:")
        print(f"    mean {mean:+.2f}c | median {median:+.2f}c | contract-wtd {w:+.2f}c "
              f"| p90 {p90:+.2f}c | worst {worst:+.2f}c")
    avg_spread = wmean(recs, "spread") if False else st.mean(r["spread"] for r in recs) * 100
    print(f"\n  avg quoted spread: {avg_spread:.2f}c")
    # breakdown by side
    for side in ("yes", "no"):
        s = [r for r in recs if r["side"] == side]
        if s:
            print(f"    {side:>3}: n={len(s):>3}  vs-mid median {sorted(r['slip_mid'] for r in s)[len(s)//2]:+.2f}c"
                  f"  vs-touch median {sorted(r['slip_touch'] for r in s)[len(s)//2]:+.2f}c")


def main():
    real, paper = load()
    summarize(real, "REAL FILLS (order_result=ok)")
    summarize(paper, "PAPER FILLS (simulator's own model, for contrast)")

    if real:
        w_mid = wmean(real, "slip_mid")
        print(f"\n{'='*70}\n  BOTTOM LINE (real fills, contract-weighted)\n{'='*70}")
        print(f"  Every contract cost ~{w_mid:+.2f}c MORE than the backtest's mid-price assumption.")
        print(f"  Recall the measured edge was ~+4c/contract. Slippage is a COST, so net edge: "
              f"~{4 - w_mid:+.2f}c/contract.")
        print("  NOTE: real-fill sample is small; treat as directional, not precise.")


if __name__ == "__main__":
    main()
