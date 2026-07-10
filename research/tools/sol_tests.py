"""
Strategies 8 & 9 on KXSOL15M (30 days, executable fills, exact fee, no lookahead).

  S8  cross-asset lead-lag — buy the SOL contract side matching BTC's move
      since the SOL window opened (BTC as the signal, SOL book as the venue).
  S9  SOL continuation     — same-asset continuation taker (is the SOL book
      less efficiently priced than BTC's?). Plus NO-side bias check.

BTC minute prices come from the repo cache (Coinbase, corrected lookup);
SOL minute prices from sol_cache. Both lag-corrected: price at t = close of
the candle keyed t-60 (Coinbase candles are start-keyed).
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "sol_cache")
sys.path.insert(0, "/Users/leolee/Desktop/kalshi-delta-hedging")
import btc_data  # corrected lookup


def fee(c):
    return 0.07 * c * (1.0 - c)


def load_spot(prefix, folder):
    px = {}
    for f in glob.glob(os.path.join(folder, f"{prefix}_*.json")):
        px.update(json.load(open(f)))
    return px


def cluster_se(bw):
    vals = [sum(v) / len(v) for v in bw.values()]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    return (sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 / len(vals) ** 0.5


def quotes_at(candles, t):
    row = None
    for c in candles:
        if c["ts"] <= t:
            row = c
        else:
            break
    return row


def main():
    sol_spot = load_spot("sol", CACHE)
    btc = load_spot("btc_cb", "/Users/leolee/Desktop/kalshi-delta-hedging/data/cache")
    markets = [m for m in json.load(open(os.path.join(CACHE, "markets.json")))
               if m.get("result") in ("yes", "no") and m.get("open_time")]
    print(f"SOL markets: {len(markets)}  eth spot: {len(sol_spot):,}  btc spot: {len(btc):,}")

    obs = []
    miss = 0
    for mi, m in enumerate(markets):
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        ry = m["result"] == "yes"
        e0 = btc_data.lookup(sol_spot, t0)
        b0 = btc_data.lookup(btc, t0)
        cp = os.path.join(CACHE, f"c_{m['ticker']}.json")
        if not os.path.exists(cp) or e0 is None:
            miss += 1
            continue
        candles = json.load(open(cp))
        if not candles:
            miss += 1
            continue
        for minute in range(1, 14):
            t = t0 + minute * 60
            et = btc_data.lookup(sol_spot, t)
            bt = btc_data.lookup(btc, t) if b0 else None
            row = quotes_at(candles, t)
            if et is None or row is None:
                continue
            if not (0.01 < row["yes"] < 0.99) or row["ask"] is None or row["bid"] is None:
                continue
            obs.append({
                "mi": mi, "minute": minute,
                "eth_up": et > e0, "eth_mv": abs(et - e0) / e0 * 100,
                "btc_up": (bt > b0) if (bt and b0) else None,
                "btc_mv": (abs(bt - b0) / b0 * 100) if (bt and b0) else None,
                "yes_won": ry, "yes_exec": row["ask"], "no_exec": 1.0 - row["bid"],
            })
    print(f"obs: {len(obs):,}  (skipped {miss} markets)")

    def pnl(c, won):
        return (1 - c - fee(c)) if won else (-c - fee(c))

    def report(name, sel, side_fn):
        bw = defaultdict(list)
        for o in sel:
            side_up = side_fn(o)
            if side_up is None:
                continue
            c = o["yes_exec"] if side_up else o["no_exec"]
            if not (0.03 < c < 0.97):
                continue
            bw[o["mi"]].append(pnl(c, o["yes_won"] == side_up))
        n = sum(len(v) for v in bw.values())
        if n < 100:
            print(f"  {name}: n={n} (too few)")
            return
        e = sum(x for v in bw.values() for x in v) / n
        print(f"  {name}: n={n:>6} edge={e*100:+6.2f}c ±{cluster_se(bw)*100:.2f}")

    print("\n── S9: SOL same-asset continuation taker")
    report("all obs           ", obs, lambda o: o["eth_up"])
    report("|eth mv| >= 0.10% ", [o for o in obs if o["eth_mv"] >= 0.10], lambda o: o["eth_up"])
    report("|eth mv| >= 0.25% ", [o for o in obs if o["eth_mv"] >= 0.25], lambda o: o["eth_up"])
    print("  NO-side bias check:")
    report("buy NO always     ", obs, lambda o: False)
    report("buy YES always    ", obs, lambda o: True)

    print("\n── S8: cross-asset — trade SOL book on BTC's move")
    bobs = [o for o in obs if o["btc_up"] is not None]
    report("btc dir, all      ", bobs, lambda o: o["btc_up"])
    report("|btc mv| >= 0.10% ", [o for o in bobs if o["btc_mv"] >= 0.10], lambda o: o["btc_up"])
    report("|btc mv| >= 0.25% ", [o for o in bobs if o["btc_mv"] >= 0.25], lambda o: o["btc_up"])
    print("  divergence: BTC moved >=0.15%, SOL's own move < 0.05% (SOL lagging)")
    report("btc-lead divergence",
           [o for o in bobs if o["btc_mv"] >= 0.15 and o["eth_mv"] < 0.05],
           lambda o: o["btc_up"])


if __name__ == "__main__":
    main()
