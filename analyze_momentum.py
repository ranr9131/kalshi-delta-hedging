"""
Momentum analysis: does BTC velocity within a window predict outcomes
beyond what magnitude alone captures?

Key questions:
  1. When BTC direction and Kalshi direction disagree, who is right?
  2. Within the same magnitude bucket, does momentum (toward vs away from floor)
     change the win rate significantly?
  3. Does Kalshi mid direction predict outcomes better than BTC position alone?
"""

import os, sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
import kalshi_client, btc_data
from config import DATA_DAYS, LOGS_DIR

markets    = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
timestamps = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
              for m in markets if m.get("open_time")]
btc_prices = btc_data.fetch_btc_prices(min(timestamps) - 600, max(timestamps) + 1800)

MINUTES = list(range(4, 14))
_2D_BUCKETS = [
    (0.000, 0.05), (0.050, 0.10), (0.100, 0.20), (0.200, 0.50), (0.500, float("inf")),
]
_2D_LABELS = ["0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]

def bucket(pct):
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= pct < hi:
            return i
    return len(_2D_BUCKETS) - 1

# ── Data collection ───────────────────────────────────────────────────────────
# [minute][scenario] -> list of signal_correct (bool)
agree    = defaultdict(lambda: defaultdict(list))  # kalshi agrees with BTC position
momentum = defaultdict(lambda: defaultdict(list))  # momentum toward vs away, per bucket

processed = 0
for market in markets:
    open_iso  = market.get("open_time", "")
    result    = market.get("result", "")
    if not open_iso or result not in ("yes", "no"):
        continue
    t0       = int(datetime.fromisoformat(open_iso.replace("Z", "+00:00")).timestamp())
    resolved = result == "yes"
    btc_t0   = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None:
        continue

    candles = kalshi_client.fetch_candlesticks(
        market["ticker"], open_iso, market.get("close_time", "")
    )
    if not candles:
        continue

    for minute in MINUTES:
        t      = t0 + minute * 60
        t_prev = t - 60
        btc_t  = btc_data.lookup(btc_prices, t)
        btc_p  = btc_data.lookup(btc_prices, t_prev)
        k_yes  = kalshi_client.get_yes_price_at(candles, t)
        if btc_t is None or k_yes is None or not (0.01 < k_yes < 0.99):
            continue

        above_floor  = btc_t > btc_t0
        kal_yes_side = k_yes > 0.5
        signal_correct = (above_floor == resolved)

        # Kalshi agreement
        kal_agrees = above_floor == kal_yes_side
        sc = "agree" if kal_agrees else "disagree"
        agree[minute][sc].append(signal_correct)

        # Directional breakdown for disagree case
        if not kal_agrees:
            if above_floor:
                agree[minute]["btc_up_kal_no"].append(signal_correct)
            else:
                agree[minute]["btc_dn_kal_yes"].append(signal_correct)

        # Momentum: is BTC moving toward or away from floor?
        if btc_p is not None:
            toward = (above_floor and btc_t < btc_p) or (not above_floor and btc_t > btc_p)
            pct    = abs(btc_t - btc_t0) / btc_t0 * 100
            bi     = bucket(pct)
            label  = _2D_LABELS[bi]
            mom    = "toward" if toward else "away"
            momentum[minute][f"{label}|{mom}"].append(signal_correct)

    processed += 1

print(f"Processed {processed} windows\n")

# ── Table 1: Kalshi agreement ─────────────────────────────────────────────────
print("=" * 65)
print("TABLE 1: DOES KALSHI DIRECTION AGREE WITH BTC POSITION?")
print("=" * 65)

# Aggregate across all minutes
all_agree    = []
all_disagree = []
all_btc_up_kal_no  = []
all_btc_dn_kal_yes = []
for minute in MINUTES:
    all_agree.extend(agree[minute]["agree"])
    all_disagree.extend(agree[minute]["disagree"])
    all_btc_up_kal_no.extend(agree[minute]["btc_up_kal_no"])
    all_btc_dn_kal_yes.extend(agree[minute]["btc_dn_kal_yes"])

def wr(lst): return sum(lst) / len(lst) if lst else float("nan")
def fmt(lst): return f"{wr(lst):.1%}  (n={len(lst):,})"

print(f"\n  Kalshi AGREES with BTC direction:    {fmt(all_agree)}")
print(f"  Kalshi DISAGREES with BTC direction: {fmt(all_disagree)}")
print(f"    of which BTC up, Kalshi says NO:   {fmt(all_btc_up_kal_no)}")
print(f"    of which BTC dn, Kalshi says YES:  {fmt(all_btc_dn_kal_yes)}")

print(f"\n  By minute (disagree cases only):")
print(f"  {'Minute':<8}  {'BTC^ Kal_':>9}  {'n':>5}  {'BTC_ Kal^':>9}  {'n':>5}")
print("  " + "-" * 42)
for m in MINUTES:
    u = agree[m]["btc_up_kal_no"]
    d = agree[m]["btc_dn_kal_yes"]
    r1 = f"{wr(u):.1%}" if u else " n/a "
    r2 = f"{wr(d):.1%}" if d else " n/a "
    print(f"  T+{m:<6}  {r1:>9}  {len(u):>5}  {r2:>9}  {len(d):>5}")

# ── Table 2: Momentum within each magnitude bucket ────────────────────────────
print(f"\n{'=' * 65}")
print("TABLE 2: MOMENTUM EFFECT WITHIN MAGNITUDE BUCKETS")
print("Win rate when signal is correct = direction holds to T+15")
print("'away'  = BTC moving further from floor (confirming move)")
print("'toward'= BTC pulling back toward floor (counter-momentum)")
print("=" * 65)

for bi, label in enumerate(_2D_LABELS):
    away_all   = []
    toward_all = []
    for m in MINUTES:
        away_all.extend(momentum[m][f"{label}|away"])
        toward_all.extend(momentum[m][f"{label}|toward"])
    if not away_all and not toward_all:
        continue
    diff = wr(away_all) - wr(toward_all)
    print(f"\n  Bucket {label}:")
    print(f"    Moving away from floor (momentum):  {fmt(away_all)}")
    print(f"    Pulling back toward floor:           {fmt(toward_all)}")
    print(f"    Difference:                          {diff:+.1%}")

# ── Table 3: Minute-by-minute momentum for key buckets ───────────────────────
print(f"\n{'=' * 65}")
print("TABLE 3: MOMENTUM BY MINUTE (0.10-0.20% and 0.20-0.50% buckets)")
print("=" * 65)

for label in ["0.10-0.20%", "0.20-0.50%"]:
    print(f"\n  Bucket {label}:")
    print(f"  {'Minute':<8}  {'away win%':>10}  {'n':>5}  {'toward win%':>12}  {'n':>5}  {'diff':>6}")
    print("  " + "-" * 50)
    for m in MINUTES:
        away   = momentum[m][f"{label}|away"]
        toward = momentum[m][f"{label}|toward"]
        if not away or not toward:
            continue
        diff = wr(away) - wr(toward)
        print(f"  T+{m:<6}  {wr(away):>10.1%}  {len(away):>5}  {wr(toward):>12.1%}  {len(toward):>5}  {diff:>+6.1%}")
