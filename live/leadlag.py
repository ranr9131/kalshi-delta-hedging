"""
Lead-lag / latency-edge test: does crypto spot PREDICT the Kalshi mid a few
seconds out (an exploitable lag), and is that predictable move > the spread?

This is the decisive test of "buy the inefficiency, sell as it normalizes." We
reconstruct the Kalshi YES mid over time from the recorded orderbook deltas,
align it to the recorded crypto spot, and measure:

  At time t we have ALREADY observed the spot move over [t-W, t]  (r = past spot
  return).  Does the Kalshi mid then move over [t, t+h]  (future mid change)?
    * slope ~ 0   -> mid already moved with spot = efficient, NO edge.
    * slope > 0   -> mid lags spot, catching up = exploitable latency edge.
  We size the predictable future move in CENTS and compare to the spread+fees.

Only near-money (0.15<mid<0.85), two-sided samples count (where prob actually
responds to spot).  Focus: tight markets (BTC/ETH 15M) — the only place a small
catch-up could beat the cost.

Usage (on EC2):
  python3.11 leadlag.py --dir ~/kalshi-delta-hedging/recordings/2026-06-07 \
      --series KXBTC15M --product BTC-USD
"""
from __future__ import annotations
import argparse
import bisect
import glob
import gzip
import json
import math
import os
from collections import defaultdict


def lines(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for ln in f:
            yield ln


def load_spot(files, product):
    pts = []
    for p in files:
        for ln in lines(p):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("product_id") == product and d.get("price"):
                try:
                    pts.append((int(d["_t"]), float(d["price"])))
                except Exception:
                    pass
    pts.sort()
    return pts


def make_lookup(pts):
    ts = [t for t, _ in pts]
    vs = [v for _, v in pts]

    def at(t, tol_ms=4000):
        i = bisect.bisect_right(ts, t) - 1   # last sample <= t
        if i < 0:
            return None
        if t - ts[i] > tol_ms:
            return None
        return vs[i]
    return at


def build_mids(files, series_prefix):
    """Return {market: [(t_ms, mid_prob, spread_c), ...]} from orderbook deltas."""
    books = defaultdict(lambda: {"yes": {}, "no": {}})
    mids = defaultdict(list)
    for p in sorted(files):
        for ln in lines(p):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            typ = d.get("type")
            msg = d.get("msg", {})
            mk = msg.get("market_ticker", "")
            if not mk.startswith(series_prefix):
                continue
            if typ == "orderbook_snapshot":
                bk = {"yes": {}, "no": {}}
                for side in ("yes", "no"):
                    for lvl in msg.get(side, []) or []:
                        try:
                            pc = int(round(float(lvl[0]) * (100 if float(lvl[0]) <= 1.5 else 1)))
                            bk[side][pc] = float(lvl[1])
                        except Exception:
                            pass
                books[mk] = bk
            elif typ == "orderbook_delta":
                side = msg.get("side")
                if side not in ("yes", "no"):
                    continue
                try:
                    pc = int(round(float(msg["price_dollars"]) * 100))
                    sz = books[mk][side].get(pc, 0.0) + float(msg["delta_fp"])
                except Exception:
                    continue
                if sz <= 0:
                    books[mk][side].pop(pc, None)
                else:
                    books[mk][side][pc] = sz
                yes, no = books[mk]["yes"], books[mk]["no"]
                if yes and no:
                    yb = max(yes); ya = 100 - max(no)
                    if 0 < yb < ya < 100:
                        mids[mk].append((int(d["_t"]), (yb + ya) / 200.0, ya - yb))
    return mids


def ols(xs, ys):
    n = len(xs)
    if n < 5:
        return 0.0, 0.0, 0
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-30:
        return 0.0, sy / n, n
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    return a, b, n


def ols2(x1, x2, y):
    """y = b1*x1 + b2*x2 + c  (normal equations).  Returns (b1, b2)."""
    n = len(y)
    if n < 8:
        return 0.0, 0.0
    s1 = sum(x1); s2 = sum(x2); sy = sum(y)
    s11 = sum(a * a for a in x1); s22 = sum(a * a for a in x2)
    s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * b for a, b in zip(x1, y)); s2y = sum(a * b for a, b in zip(x2, y))
    # center to drop intercept
    m1 = s1 / n; m2 = s2 / n; my = sy / n
    c11 = s11 - n * m1 * m1; c22 = s22 - n * m2 * m2; c12 = s12 - n * m1 * m2
    c1y = s1y - n * m1 * my; c2y = s2y - n * m2 * my
    det = c11 * c22 - c12 * c12
    if abs(det) < 1e-30:
        return 0.0, 0.0
    b1 = (c22 * c1y - c12 * c2y) / det
    b2 = (c11 * c2y - c12 * c1y) / det
    return b1, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--series", default="KXBTC15M")
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--past-w", type=int, default=2000, help="past spot window ms")
    ap.add_argument("--horizons", default="1000,2000,5000")
    ap.add_argument("--sample-ms", type=int, default=500)
    ap.add_argument("--mid-lo", type=float, default=0.15)
    ap.add_argument("--mid-hi", type=float, default=0.85)
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    kfiles = glob.glob(os.path.join(args.dir, "kalshi.jsonl*"))
    cfiles = glob.glob(os.path.join(args.dir, "crypto.jsonl*"))
    print(f"kalshi files: {len(kfiles)}  crypto files: {len(cfiles)}")
    spot = make_lookup(load_spot(cfiles, args.product))
    mids = build_mids(kfiles, args.series)
    print(f"{args.series}: {len(mids)} markets reconstructed")

    # collected samples: per horizon, (past_spot_return, future_mid_change_cents)
    samp = {h: [] for h in horizons}
    contemp = []   # (past spot return, contemporaneous mid change over [t-W,t]) cents
    spreads = []
    for mk, series in mids.items():
        if len(series) < 20:
            continue
        ts = [t for t, _, _ in series]
        midv = [m for _, m, _ in series]
        sprd = [s for _, _, s in series]

        def mid_at(t, tol=1500):
            i = bisect.bisect_right(ts, t) - 1
            if i < 0 or t - ts[i] > tol:
                return None
            return midv[i], sprd[i]

        t0, t1 = ts[0], ts[-1]
        t = t0 + args.past_w
        while t <= t1 - max(horizons):
            mt = mid_at(t)
            if mt and args.mid_lo < mt[0] < args.mid_hi:
                sp_now = spot(t); sp_past = spot(t - args.past_w)
                m_past = mid_at(t - args.past_w)
                if sp_now and sp_past and m_past:
                    r = math.log(sp_now / sp_past)            # past spot return
                    spreads.append(mt[1])
                    contemp.append((r, (mt[0] - m_past[0]) * 100.0))
                    for h in horizons:
                        mh = mid_at(t + h)
                        sp_fut = spot(t + h)
                        if mh and sp_fut:
                            rf = math.log(sp_fut / sp_now)   # FUTURE spot return
                            samp[h].append((r, (mh[0] - mt[0]) * 100.0, rf))
            t += args.sample_ms

    spreads.sort()
    med_spread = spreads[len(spreads) // 2] if spreads else float("nan")
    print(f"\nnear-money samples: {len(contemp)}   median spread: {med_spread:.1f}c")
    typ_move = 0.0010   # a 0.10% spot move, for sizing
    # contemporaneous (already moved with spot)
    a, b, n = ols([r for r, _ in contemp], [d for _, d in contemp])
    print(f"\nCONTEMPORANEOUS  mid change over [t-{args.past_w}ms, t] vs past spot return:")
    print(f"  slope={a*typ_move:+.2f}c per 0.10% spot move  (how much mid ALREADY moved)")

    print(f"\nPREDICTIVE (naive)  future mid change vs ALREADY-OBSERVED past spot return:")
    print(f"  {'horizon':>8} {'n':>7} {'slope_c/0.1%':>13}")
    for h in horizons:
        a, b, n = ols([s[0] for s in samp[h]], [s[1] for s in samp[h]])
        print(f"  {h:>6}ms {n:>7} {a*typ_move:>+12.2f}c")

    print(f"\nLAG vs MOMENTUM  (future mid ~ b1*PAST_spot + b2*FUTURE_spot):")
    print(f"  b1 = TRUE Kalshi lag (mid catching up, controls for new spot moves)")
    print(f"  b2 = mid's contemporaneous response to NEW spot (momentum lives here)")
    print(f"  {'horizon':>8} {'b1_lag_c/0.1%':>14} {'b2_resp_c/0.1%':>15} {'lag vs spread':>15}")
    for h in horizons:
        b1, b2 = ols2([s[0] for s in samp[h]], [s[2] for s in samp[h]], [s[1] for s in samp[h]])
        lag = b1 * typ_move
        verdict = "REAL LAG?" if abs(lag) > med_spread else "< spread (no)"
        print(f"  {h:>6}ms {lag:>+13.2f}c {b2*typ_move:>+14.2f}c {verdict:>15}")

    print(f"\nEVENT STUDY (big past moves |r|>0.03%), avg FUTURE mid follow-through (cents):")
    for h in horizons:
        big = [s for s in samp[h] if abs(s[0]) > 0.0003]
        if not big:
            print(f"  +{h}ms: (no big moves in sample)"); continue
        aligned = [s[1] * (1 if s[0] > 0 else -1) for s in big]
        avg = sum(aligned) / len(aligned)
        print(f"  +{h}ms: n={len(big):>5}  avg follow-through={avg:+.2f}c (spread {med_spread:.1f}c)")


if __name__ == "__main__":
    main()
