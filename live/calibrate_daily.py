"""
Horizon-bucketed calibration for daily/hourly crypto markets (KX{coin}D).

WHY: fair_price_model_v2's Platt calibration (calibration_v2.json) was fit on
15M markets — horizons of 0.5–14 min. The daily/hourly markets (KXXRPD etc.)
trade at much longer horizons, where the raw log-normal model is mis-calibrated
(observed: model 65¢ vs market 82¢). This refits Platt *per horizon bucket* from
settled daily-market outcomes.

SCOPE: we calibrate the 5min–2h band, where fills actually occur (near expiry).
We do NOT try to calibrate multi-day horizons — nobody actively MM-quotes a
6-day market, and √t-scaled intraday σ is meaningless out there.

PIPELINE (one script):
  1. Pull settled KX{coin}D markets over --days.
  2. Group strikes by close_time → fetch 1-min Coinbase klines ONCE per ladder.
  3. For each (strike, horizon) sample: spot@t, realized σ from trailing 10min,
     raw model p = fair_p_yes_raw_v2, paired with the settled outcome.
  4. Fit Platt (a,b) per (asset, horizon-bucket); print calibration tables.
  5. Write calibration_daily.json: {asset: {bucket_label: {a,b,n,...}}}.

The model consumes it via fair_price_model_v2.fair_p_yes_daily() (see wiring).

CLI:
  python3 calibrate_daily.py --series KXXRPD --days 14
  python3 calibrate_daily.py --series KXXRPD,KXDOGED,KXSOLD --days 21 --out calibration_daily.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_price_model_v2 import fair_p_yes_raw_v2, FALLBACK_SIGMA_PER_MIN, SIGMA_FLOOR_MULT, SIGMA_CEIL_MULT

KALSHI_BASE  = "https://api.elections.kalshi.com"
COINBASE     = "https://api.exchange.coinbase.com"

COIN_PRODUCT = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "DOGE": "DOGE-USD", "BNB": "BNB-USD", "ADA": "ADA-USD", "LTC": "LTC-USD",
}
COIN_SYMS = sorted(COIN_PRODUCT.keys(), key=len, reverse=True)

# Horizons (minutes before close) to sample. Capped at 120 so all klines for a
# market fit one Coinbase request (+10min trailing for σ → 130 < 300 candle cap).
SAMPLE_MINS = [5, 10, 15, 20, 30, 45, 60, 90, 120]
# Horizon buckets for the Platt fit (minutes-to-close).
BUCKETS = [(0, 12), (12, 25), (25, 50), (50, 100), (100, 1e9)]


def bucket_label(mins: float) -> str | None:
    for lo, hi in BUCKETS:
        if lo <= mins < hi:
            return f"{lo}-{int(hi) if hi < 1e8 else 'inf'}m"
    return None


def coin_of(ticker: str):
    body = ticker.split("-")[0].upper()
    body = body[2:] if body.startswith("KX") else body
    for s in COIN_SYMS:
        if body.startswith(s):
            return s
    return None


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def fit_platt(pairs, iters=500, lr=0.15):
    if len(pairs) < 2:
        return 0.0, 1.0, {"n": len(pairs)}
    xs = [logit(p) for p, _ in pairs]
    ys = [float(y) for _, y in pairs]
    a, b, n = 0.0, 1.0, len(xs)
    for _ in range(iters):
        ps = [sigmoid(a + b * x) for x in xs]
        ga = sum(p - y for p, y in zip(ps, ys)) / n
        gb = sum((p - y) * x for p, y, x in zip(ps, ys, xs)) / n
        a -= lr * ga; b -= lr * gb
    ll  = sum(-(y*math.log(max(p,1e-9)) + (1-y)*math.log(max(1-p,1e-9))) for p,y in zip(ps,ys))/n
    ll0 = sum(-(y*math.log(max(p,1e-9)) + (1-y)*math.log(max(1-p,1e-9))) for p,y in pairs)/n
    br  = sum((p-y)**2 for p,y in zip(ps,ys))/n
    br0 = sum((p-y)**2 for p,y in pairs)/n
    return a, b, {"n": n, "log_loss": round(ll,4), "log_loss_raw": round(ll0,4),
                  "brier": round(br,4), "brier_raw": round(br0,4)}


def fetch_settled(series: str, days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out, cursor, page = [], None, 0
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/trade-api/v2/markets", params=params, timeout=20)
        if not r.ok:
            print(f"  [{series}] page {page}: HTTP {r.status_code}"); break
        body = r.json(); batch = body.get("markets", [])
        if not batch:
            break
        stop = False
        for mk in batch:
            ci = mk.get("close_time")
            if not ci:
                continue
            try:
                cdt = datetime.fromisoformat(ci.replace("Z", "+00:00"))
            except Exception:
                continue
            if cdt < cutoff:
                stop = True; continue
            out.append(mk)
        cursor = body.get("cursor"); page += 1
        if not cursor or stop:
            break
        time.sleep(0.12)
    print(f"  [{series}] {len(out)} settled markets over {days}d")
    return out


def fetch_klines(product: str, start_ms: int, end_ms: int):
    """1-min candles ascending [t_ms, o,h,l,c,v]. Single request (<300 candles)."""
    try:
        r = requests.get(f"{COINBASE}/products/{product}/candles",
                         params={"granularity": 60,
                                 "start": datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).isoformat(),
                                 "end":   datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).isoformat()},
                         timeout=15)
        if not r.ok:
            return []
        rows = r.json()
    except Exception:
        return []
    out = []
    for row in reversed(rows):
        try:
            t, lo, hi, op, cl, vol = row
            out.append([int(t)*1000, float(op), float(hi), float(lo), float(cl), float(vol)])
        except Exception:
            continue
    return out


def price_at(kl, t_ms):
    if not kl or t_ms < kl[0][0] or t_ms > kl[-1][0] + 60_000:
        return None
    lo, hi = 0, len(kl)-1
    while lo < hi:
        mid = (lo+hi)//2
        if kl[mid][0] + 60_000 <= t_ms:
            lo = mid+1
        else:
            hi = mid
    return float(kl[lo][4])


def sigma_at(kl, t_ms, asset, window_min=10):
    """Realized per-min σ from the 10 min of closes ending at t_ms. Clamped
    around the asset fallback, mirroring fair_price_model_v2."""
    fb = FALLBACK_SIGMA_PER_MIN.get(asset, 0.0015)
    closes = [k[4] for k in kl if t_ms - window_min*60_000 <= k[0] <= t_ms]
    if len(closes) >= 4:
        rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))
                if closes[i] > 0 and closes[i-1] > 0]
        if len(rets) >= 2:
            mean = sum(rets)/len(rets)
            var = sum((x-mean)**2 for x in rets)/(len(rets)-1)
            s = math.sqrt(var)
            return max(fb*SIGMA_FLOOR_MULT, min(fb*SIGMA_CEIL_MULT, s))
    return fb


def cal_table(pairs, n_bins=10):
    bins = defaultdict(list)
    for p, y in pairs:
        bins[min(n_bins-1, int(p*n_bins))].append((p, y))
    print(f"    {'bin':>9} {'n':>5} {'pred':>6} {'realized':>9} {'err':>7}")
    for b in range(n_bins):
        it = bins[b]
        if not it:
            continue
        mp = sum(p for p,_ in it)/len(it)
        rw = sum(y for _,y in it)/len(it)
        print(f"    {b*10:>3}-{(b+1)*10:>3}% {len(it):>5} {mp:>6.3f} {rw:>9.3f} {rw-mp:>+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXXRPD")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "calibration_daily.json"))
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--dump-csv", default=None)
    ap.add_argument("--from-csv", default=None,
                    help="load samples from a previous --dump-csv (skip fetch)")
    ap.add_argument("--fit-band", nargs=2, type=float, default=[0.03, 0.97],
                    help="only fit Platt on samples whose raw_p is in this band "
                         "(excludes trivial extreme strikes that drown the fit)")
    args = ap.parse_args()

    # asset -> list[(mins_to_close, raw_p, outcome)]
    samples = defaultdict(list)
    if args.from_csv:
        with open(args.from_csv) as fh:
            for r in csv.DictReader(fh):
                samples[r["asset"]].append((float(r["mins_to_close"]),
                                            float(r["raw_p"]), int(r["outcome"])))
        print(f"  loaded {sum(len(v) for v in samples.values())} samples from {args.from_csv}")
    for series in ([] if args.from_csv else args.series.split(",")):
        series = series.strip()
        asset = coin_of(series)
        product = COIN_PRODUCT.get(asset)
        if not product:
            print(f"  skip {series}: no product mapping"); continue
        mkts = fetch_settled(series, args.days)
        # group by close_time so we fetch klines once per ladder
        by_close = defaultdict(list)
        for mk in mkts:
            res = (mk.get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            try:
                strike = float(mk.get("floor_strike") or mk.get("cap_strike") or 0)
            except Exception:
                continue
            if strike <= 0:
                continue
            by_close[mk["close_time"]].append((strike, 1 if res == "yes" else 0))
        print(f"  [{series}] {len(by_close)} distinct close-times; fetching klines...")
        for ci, strikes in by_close.items():
            try:
                close_dt = datetime.fromisoformat(ci.replace("Z", "+00:00"))
            except Exception:
                continue
            close_ms = int(close_dt.timestamp()*1000)
            kl = fetch_klines(product, close_ms - (max(SAMPLE_MINS)+12)*60_000, close_ms + 60_000)
            if not kl:
                continue
            for m in SAMPLE_MINS:
                t_ms = close_ms - m*60_000
                px = price_at(kl, t_ms)
                if px is None:
                    continue
                sig = sigma_at(kl, t_ms, asset)
                for strike, outcome in strikes:
                    raw = fair_p_yes_raw_v2(px, strike, m, asset, sigma_per_min=sig)
                    samples[asset].append((m, raw, outcome))
            time.sleep(0.08)

    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["asset","mins_to_close","raw_p","outcome"])
            for a, rows in samples.items():
                for m, raw, o in rows:
                    w.writerow([a, m, round(raw,5), o])
        print(f"  dumped raw samples → {args.dump_csv}")

    # Fit per (asset, bucket) + report
    out_cal = {}
    for asset, rows in samples.items():
        print(f"\n=== {asset}: {len(rows)} samples ===")
        print("  RAW calibration (uncorrected model):")
        cal_table([(r, o) for _, r, o in rows])
        out_cal.setdefault(asset, {})
        lo_b, hi_b = args.fit_band
        print(f"  BAND-RESTRICTED fit ({lo_b}<raw<{hi_b}), tradeable zone only:")
        cal_table([(r, o) for _, r, o in rows if lo_b < r < hi_b])
        by_bucket = defaultdict(list)
        for m, raw, o in rows:
            if not (lo_b < raw < hi_b):
                continue
            bl = bucket_label(m)
            if bl:
                by_bucket[bl].append((raw, o))
        for bl in sorted(by_bucket, key=lambda x: BUCKETS.index(next(b for b in BUCKETS if f"{b[0]}-{int(b[1]) if b[1]<1e8 else 'inf'}m" == bl))):
            pairs = by_bucket[bl]
            a, b, info = fit_platt(pairs)
            flag = "" if info["n"] >= args.min_n else "  (LOW N — not emitted)"
            print(f"  bucket {bl:>9}: n={info['n']:>4}  a={a:+.3f} b={b:+.3f}  "
                  f"brier {info.get('brier_raw','?')}→{info.get('brier','?')}{flag}")
            if info["n"] >= args.min_n:
                out_cal[asset][bl] = {"a": round(a,5), "b": round(b,5), **info}

    with open(args.out, "w") as fh:
        json.dump(out_cal, fh, indent=2)
    print(f"\nWrote {args.out}")
    print("Buckets emitted:", {a: list(v.keys()) for a, v in out_cal.items()})


if __name__ == "__main__":
    main()
