"""
(a) Last-minute validation corpus from Binance 1-second klines.

The main corpus.csv tops out at 1-minute resolution (Coinbase klines), so the
tau<1min behavior of the model is only validated by extrapolation.  This builds a
FINE-horizon corpus for the 15M markets (BTC/ETH/SOL) from Binance 1s data:

  * reconstruct the true settlement object = mean of the 60 one-second closes in
    (close-60s, close]  (proxy for the BRTI 60s-average),
  * sample spot at fine horizons tau in {5s,10s,20s,30s,45s, 1,1.5,2,3 min},
  * pair with the real outcome + exp_val (true BRTI settle from Kalshi).

Then it VALIDATES the model's last-minute variance law
    Var[ln(settle) - ln(spot_t)] = sigma^2 (tau^3 + (1-tau)^3)/3 + sigma_b^2   (tau<1)
against the empirical per-horizon residual variance, and reports binary
Brier/log-loss of the OLD (tau^3/3 only) vs NEW (tau^3+(1-tau)^3)/3 variance law.

Usage:
  python3 build_corpus_1s.py --days 7 --assets BTC,ETH,SOL --out corpus_1s.csv
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_corpus import fetch_settled, FALLBACK_SIGMA, SIGMA_FLOOR_MULT, SIGMA_CEIL_MULT
from fair_price_model_v3 import _ybar_variance, _sf_unit, AVG_WINDOW_MIN

BINANCE_SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
SERIES_15M = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M"}
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
os.makedirs(CACHE, exist_ok=True)

# fine horizons in MINUTES (sub-minute first, then >=1 for continuity)
HORIZONS = [5/60, 10/60, 20/60, 30/60, 45/60, 1.0, 1.5, 2.0, 3.0]


def load_1s_closes(asset: str, date_str: str):
    """Return {ts_s: close_price} for one UTC day from Binance Vision 1s klines.
    Cached as a compact JSON of second->close."""
    cache_path = os.path.join(CACHE, f"binance_1sclose_{asset}_{date_str.replace('-','')}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception:
            pass
    sym = BINANCE_SYM[asset]
    url = f"https://data.binance.vision/data/spot/daily/klines/{sym}/1s/{sym}-1s-{date_str}.zip"
    try:
        r = requests.get(url, timeout=180)
        if r.status_code == 404:
            print(f"    {asset} {date_str}: 404"); return {}
        r.raise_for_status()
    except Exception as e:
        print(f"    {asset} {date_str}: err {e}"); return {}
    closes = {}
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            with zf.open(name) as fh:
                for row in csv.reader(io.TextIOWrapper(fh, encoding="utf-8")):
                    if not row or not row[0].isdigit():
                        continue
                    ts = int(row[0])
                    while ts > 100_000_000_000:   # normalize us/ms -> seconds
                        ts //= 1000
                    closes[ts] = float(row[4])
    with open(cache_path, "w") as f:
        json.dump(closes, f)
    print(f"    {asset} {date_str}: {len(closes)} 1s closes")
    return closes


def nearest(closes, ts_s, tol=5):
    """Closest 1s close to ts_s within +/- tol seconds."""
    if ts_s in closes:
        return closes[ts_s]
    for d in range(1, tol + 1):
        if ts_s - d in closes:
            return closes[ts_s - d]
        if ts_s + d in closes:
            return closes[ts_s + d]
    return None


def true_avg(closes, close_s):
    """Mean of the 60 one-second closes in (close-60, close]."""
    vals = [closes[s] for s in range(close_s - 59, close_s + 1) if s in closes]
    return sum(vals) / len(vals) if len(vals) >= 30 else None


def sigma_at(closes, t_s, asset, window_min=10):
    """Per-minute sigma from 1-min closes over the trailing 10 min (mirrors the
    main corpus / live estimator), clamped around the asset fallback."""
    fb = FALLBACK_SIGMA.get(asset, 0.0015)
    mins = []
    for k in range(window_min, 0, -1):
        px = nearest(closes, t_s - k * 60, tol=3)
        if px:
            mins.append(px)
    px0 = nearest(closes, t_s, tol=3)
    if px0:
        mins.append(px0)
    if len(mins) >= 4:
        rets = [math.log(mins[i] / mins[i-1]) for i in range(1, len(mins))
                if mins[i] > 0 and mins[i-1] > 0]
        if len(rets) >= 2:
            mean = sum(rets) / len(rets)
            var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
            return max(fb * SIGMA_FLOOR_MULT, min(fb * SIGMA_CEIL_MULT, math.sqrt(var)))
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_1s.csv"))
    args = ap.parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    FIELDS = ["asset", "close_time", "mins_left", "spot_t", "sigma_t", "exp_val",
              "true_avg_1s", "partial_avg_1s", "basis_1s", "floor_strike",
              "strike_type", "outcome"]
    fout = open(args.out, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    w.writeheader()
    n_rows = 0

    for asset in assets:
        mkts = fetch_settled(SERIES_15M[asset], args.days)
        # group settled markets by UTC date of close_time
        by_date = defaultdict(list)
        for mk in mkts:
            res = (mk.get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            try:
                ev = float(mk.get("expiration_value"))
                cdt = datetime.fromisoformat(mk["close_time"].replace("Z", "+00:00"))
            except (TypeError, ValueError, KeyError):
                continue
            by_date[cdt.strftime("%Y-%m-%d")].append({
                "close_time": mk["close_time"], "close_s": int(cdt.timestamp()),
                "exp_val": ev, "floor_strike": mk.get("floor_strike"),
                "strike_type": mk.get("strike_type") or "greater_or_equal",
                "outcome": 1 if res == "yes" else 0,
            })
        print(f"[{asset}] {sum(len(v) for v in by_date.values())} settled 15M over {len(by_date)} days")
        for date_str in sorted(by_date):
            closes = load_1s_closes(asset, date_str)
            if not closes:
                continue
            for it in by_date[date_str]:
                cs = it["close_s"]
                tavg = true_avg(closes, cs)
                if tavg is None or tavg <= 0:
                    continue
                for tau in HORIZONS:
                    t_s = cs - int(round(tau * 60))
                    spot = nearest(closes, t_s, tol=3)
                    if spot is None or spot <= 0:
                        continue
                    sig = sigma_at(closes, t_s, asset)
                    # running average over the elapsed part of the settlement
                    # window [cs-59, t_s] (only meaningful for tau < 1 min)
                    pvals = [closes[s] for s in range(cs - 59, t_s + 1) if s in closes]
                    pavg = sum(pvals) / len(pvals) if len(pvals) >= 3 else ""
                    w.writerow({
                        "asset": asset, "close_time": it["close_time"],
                        "mins_left": round(tau, 5), "spot_t": round(spot, 6),
                        "sigma_t": round(sig, 8), "exp_val": it["exp_val"],
                        "true_avg_1s": round(tavg, 6),
                        "partial_avg_1s": round(pavg, 6) if pavg != "" else "",
                        "basis_1s": round(math.log(it["exp_val"] / tavg), 8),
                        "floor_strike": it["floor_strike"], "strike_type": it["strike_type"],
                        "outcome": it["outcome"],
                    })
                    n_rows += 1
            fout.flush()
    fout.close()
    print(f"\nWrote {n_rows} fine-horizon rows -> {args.out}")


if __name__ == "__main__":
    main()
