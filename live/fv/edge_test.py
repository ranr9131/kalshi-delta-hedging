"""
Edge test: does v3's fair value beat the MARKET MID out of sample?

Uses real order-book snapshots (mm_shadow_snapshots.csv: spot, strike, mins_left,
best bid/ask, and the existing model's fair_c) for KXXRPD (XRP hourly threshold),
joined to true settled outcomes from the Kalshi API.

Three questions:
  1. Accuracy: Brier of market-mid vs v3 vs v2(fair_c) against the real outcome.
  2. Does the market beat v3, or v3 beat the market?
  3. THE edge test (per the selection-effect finding): when v3 DISAGREES with the
     mid, does the outcome move in v3's direction beyond the mid?  We regress
     outcome ~ logit(mid) + (logit(v3) - logit(mid)) and inspect the disagreement
     coefficient: >0 & significant = real incremental info; ~0 = model noise.
  Plus a binned table of mean(outcome - mid) by signed edge, and a fee-aware
  take-v3's-side P&L proxy.

Usage: python3 edge_test.py --snapshots ../mm_shadow_snapshots.csv
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fair_price_model_v3 as v3
from fair_price_model_v3 import _logit, FALLBACK_SIGMA_PER_MIN

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
FLOOR, CEIL = 0.25, 5.0  # sigma clamp mults (mirror the model)


def settled_result(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}", timeout=10)
        m = r.json().get("market", {}) if r.ok else {}
        res = (m.get("result") or "").lower()
        cache[ticker] = res if res in ("yes", "no") else None
    except Exception:
        cache[ticker] = None
    return cache[ticker]


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def logloss(pairs):
    s = 0.0
    for p, y in pairs:
        p = min(max(p, 1e-9), 1 - 1e-9)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(pairs) if pairs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mm_shadow_snapshots.csv"))
    ap.add_argument("--series", default="KXXRPD")
    ap.add_argument("--asset", default="XRP")
    ap.add_argument("--max-ml", type=float, default=120.0)
    ap.add_argument("--fee", type=float, default=1.0, help="round-trip fee cents at p~0.5 for P&L proxy")
    args = ap.parse_args()

    # pass 1: spot time series + collect rows
    spot_ts = {}     # int(ts) -> spot
    rows = []
    with open(args.snapshots) as f:
        for r in csv.DictReader(f):
            try:
                ts = float(r["ts"]); spot = float(r["spot"])
            except (ValueError, KeyError):
                continue
            if spot > 0:
                spot_ts[int(ts)] = spot
            if not r["ticker"].startswith(args.series):
                continue
            try:
                ml = float(r["mins_left"]); strike = float(r["strike"])
                bb = float(r["bb_c"]); ba = float(r["ba_c"])
                fair_c = float(r["fair_c"]) if r["fair_c"] else None
            except (ValueError, KeyError):
                continue
            if not (0 < ml <= args.max_ml) or strike <= 0:
                continue
            if not (0 < bb <= 99 and 0 < ba <= 99 and ba >= bb):
                continue
            rows.append({"ts": ts, "ticker": r["ticker"], "spot": spot,
                         "strike": strike, "ml": ml, "bb": bb, "ba": ba, "fair_c": fair_c})
    print(f"loaded {len(rows)} {args.series} snapshot rows; {len(spot_ts)} spot ticks")

    # minute-bucketed spot for realized sigma
    mins = {}
    for ts, sp in spot_ts.items():
        mins[ts // 60] = sp  # last spot in each minute wins (dict order ~ish)
    def sigma_at(ts):
        m0 = int(ts // 60)
        seq = [mins[m] for m in range(m0 - 10, m0 + 1) if m in mins]
        fb = FALLBACK_SIGMA_PER_MIN.get(args.asset, 0.002)
        if len(seq) >= 4:
            rets = [math.log(seq[i]/seq[i-1]) for i in range(1, len(seq)) if seq[i] > 0 and seq[i-1] > 0]
            if len(rets) >= 2:
                mu = sum(rets)/len(rets); var = sum((x-mu)**2 for x in rets)/(len(rets)-1)
                return max(fb*FLOOR, min(fb*CEIL, math.sqrt(var)))
        return fb

    # outcomes
    cache = {}
    tickers = sorted(set(r["ticker"] for r in rows))
    print(f"fetching {len(tickers)} settlements...")
    for i, t in enumerate(tickers):
        settled_result(t, cache)
        if i % 20 == 0:
            time.sleep(0.05)
    n_set = sum(1 for t in tickers if cache.get(t))
    print(f"  {n_set}/{len(tickers)} settled\n")

    # build prediction triples
    mid_pairs, v3_pairs, v2_pairs = [], [], []
    edge_rows = []  # (mid, v3, outcome, ba, bb)
    for r in rows:
        res = cache.get(r["ticker"])
        if res is None:
            continue
        y = 1 if res == "yes" else 0
        mid = (r["bb"] + r["ba"]) / 200.0
        sig = sigma_at(r["ts"])
        pv3 = v3.fair_p(r["spot"], r["ml"], args.asset, floor_strike=r["strike"],
                        strike_type="greater", sigma_per_min=sig)
        mid_pairs.append((mid, y)); v3_pairs.append((pv3, y))
        if r["fair_c"] is not None:
            v2_pairs.append((r["fair_c"] / 100.0, y))
        edge_rows.append((mid, pv3, y, r["ba"]/100.0, r["bb"]/100.0))

    band = lambda prs: [(p, y) for p, y in prs if 0.05 < p < 0.95]
    print("== ACCURACY vs true outcome (all / band-by-mid) ==")
    for name, prs in [("market_mid", mid_pairs), ("v3", v3_pairs), ("v2(fair_c)", v2_pairs)]:
        if not prs:
            continue
        b = [(p, y) for p, y in prs if 0.05 < (mid_pairs[0][0] if False else p) < 0.95]
        print(f"  {name:>12}: n={len(prs):>6} brier={brier(prs):.4f} ll={logloss(prs):.4f}  "
              f"| band n={len(band(prs)):>6} brier={brier(band(prs)):.4f}")

    # restrict edge analysis to band-by-mid (the tradeable zone)
    eb = [e for e in edge_rows if 0.05 < e[0] < 0.95]
    print(f"\n== EDGE: outcome vs signed edge e = v3 - mid  (band, n={len(eb)}) ==")
    print(f"  {'edge bin':>14} {'n':>6} {'mean_mid':>9} {'mean_v3':>9} {'realized':>9} {'real-mid':>9}")
    bins = [(-1, -0.10), (-0.10, -0.05), (-0.05, -0.02), (-0.02, 0.02),
            (0.02, 0.05), (0.05, 0.10), (0.10, 1)]
    for lo, hi in bins:
        sub = [e for e in eb if lo <= (e[1]-e[0]) < hi]
        if not sub:
            continue
        mm = sum(e[0] for e in sub)/len(sub); mv = sum(e[1] for e in sub)/len(sub)
        rz = sum(e[2] for e in sub)/len(sub)
        print(f"  {f'[{lo:+.2f},{hi:+.2f})':>14} {len(sub):>6} {mm:>9.3f} {mv:>9.3f} {rz:>9.3f} {rz-mm:>+9.3f}")

    # incremental-info regression: outcome ~ a + b*logit(mid) + c*(logit(v3)-logit(mid))
    import math as _m
    X = [(_logit(m), _logit(v) - _logit(m)) for m, v, y, _, _ in eb]
    Y = [y for _, _, y, _, _ in eb]
    a, b, c = 0.0, 1.0, 0.0
    n = len(X)
    if n > 50:
        for _ in range(3000):
            ga = gb = gc = 0.0
            for (x1, x2), y in zip(X, Y):
                z = a + b*x1 + c*x2
                p = 1/(1+_m.exp(-z)) if z > -30 else 0.0
                d = p - y
                ga += d; gb += d*x1; gc += d*x2
            a -= 0.05*ga/n; b -= 0.05*gb/n; c -= 0.05*gc/n
        print(f"\n  incremental-info logistic fit:  b(mid)={b:+.3f}  c(v3-mid)={c:+.3f}")
        print("  c>0 & sizable => v3 adds info beyond the mid; c~0 => model noise (no edge).")

    # fee-aware take-v3's-side P&L proxy (cross the spread)
    fee = args.fee / 100.0
    for thr in (0.03, 0.05, 0.08):
        pnl = 0.0; ntr = 0
        for m, v, y, ask, bid in eb:
            if v - m > thr:        # v3 thinks YES underpriced -> buy YES at ask
                pnl += (y - ask) - fee; ntr += 1
            elif m - v > thr:      # buy NO at (1-bid)
                pnl += ((1 - y) - (1 - bid)) - fee; ntr += 1
        if ntr:
            print(f"  take-v3 side @edge>{thr:.2f}: {ntr:>5} trades, P&L/trade = {pnl/ntr*100:+.2f}¢ (fee {args.fee}¢)")


if __name__ == "__main__":
    main()
