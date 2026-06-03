"""
Model calibration backtest for fair_p_yes.

Pulls historical settled Kalshi 15M crypto markets, then replays
`fair_p_yes` against Binance public 1-minute klines for several time points
per market.  Aggregates (predicted, actual) tuples into deciles to produce
a calibration table:

  predicted bin    n     mean_pred    realized_win_rate    error
  0-10%           ###    0.052        0.030                -0.022
  ...
  90-100%         ###    0.948        0.960                +0.012

Well-calibrated → realized matches predicted within ±0.05 per bin.
Mis-calibrated → systematic gap reveals which probability regions the model
overestimates / underestimates.

Usage:
  python backtest_calibration.py [--days 7] [--series KXSOL15M] [--samples 1,5,10]
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_price_model import fair_p_yes

KALSHI_BASE = "https://api.elections.kalshi.com"
COINBASE_BASE = "https://api.exchange.coinbase.com"

SERIES_TO_PRODUCT = {
    "KXBTC15M": "BTC-USD",
    "KXETH15M": "ETH-USD",
    "KXSOL15M": "SOL-USD",
    "KXXRP15M": "XRP-USD",
}

# Sample times = how many *minutes before close* to evaluate fair_p
DEFAULT_SAMPLES = [13, 10, 7, 5, 3, 1]


# ── Kalshi: pull settled markets ───────────────────────────────────────────

def fetch_settled_markets(series: str, days: int) -> List[dict]:
    """Pull every settled 15M market for `series` whose close was within the
    last `days` days.  Returns list of market dicts.

    Kalshi's `/markets` endpoint paginates by cursor.  status=settled
    returns historicals in reverse-chronological order."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    cursor = None
    page = 0
    while True:
        params = {
            "series_ticker": series,
            "status": "settled",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/trade-api/v2/markets",
                         params=params, timeout=20)
        if not r.ok:
            print(f"  [{series}] page {page}: HTTP {r.status_code}", flush=True)
            break
        body = r.json()
        batch = body.get("markets", [])
        if not batch:
            break
        # Stop once we're past the cutoff
        oldest_in_batch = batch[-1].get("close_time")
        for mk in batch:
            close_iso = mk.get("close_time")
            if not close_iso:
                continue
            try:
                close_dt = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
            except Exception:
                continue
            if close_dt < cutoff:
                continue
            out.append(mk)
        cursor = body.get("cursor")
        page += 1
        if not cursor:
            break
        if oldest_in_batch:
            try:
                oldest_dt = datetime.fromisoformat(oldest_in_batch.replace("Z","+00:00"))
                if oldest_dt < cutoff:
                    break
            except Exception:
                pass
        time.sleep(0.15)
    print(f"  [{series}] pulled {len(out)} settled markets over {days}d", flush=True)
    return out


# ── Coinbase Exchange: pull 1-minute candles ──────────────────────────────
# Returns ascending-by-time [[open_ms, open, high, low, close, volume], ...]
# Coinbase max 300 candles per request → for a 16-min window we get all in
# one call.  No auth.  Allows US AWS.

_PRICE_CACHE: Dict[Tuple[str, int, int], List[list]] = {}

def fetch_candles_1m(product: str, start_ms: int, end_ms: int) -> List[list]:
    key = (product, start_ms, end_ms)
    if key in _PRICE_CACHE:
        return _PRICE_CACHE[key]
    start_iso = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
    end_iso   = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()
    try:
        r = requests.get(
            f"{COINBASE_BASE}/products/{product}/candles",
            params={"granularity": 60, "start": start_iso, "end": end_iso},
            timeout=15,
        )
        if not r.ok:
            if r.status_code != 429:
                print(f"    coinbase err {r.status_code}: {r.text[:120]}",
                      flush=True)
            return []
        rows = r.json()
    except Exception as e:
        print(f"    coinbase exc: {e}", flush=True)
        return []
    # Coinbase returns DESCENDING [time_sec, low, high, open, close, volume].
    # Normalise to ASCENDING [time_ms, open, high, low, close, volume].
    out = []
    for row in reversed(rows):
        try:
            t_sec, lo_, hi_, op, cl, vol = row
        except Exception:
            continue
        out.append([int(t_sec) * 1000, float(op), float(hi_), float(lo_),
                    float(cl), float(vol)])
    _PRICE_CACHE[key] = out
    return out


def price_at(klines: List[list], target_ms: int) -> float | None:
    """Return the close price of the kline whose [open, open+60s) contains
    target_ms.  Returns None if target_ms is outside the kline range."""
    if not klines:
        return None
    if target_ms < klines[0][0] or target_ms > klines[-1][0] + 60_000:
        return None
    lo, hi = 0, len(klines) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if klines[mid][0] + 60_000 <= target_ms:
            lo = mid + 1
        else:
            hi = mid
    return float(klines[lo][4])  # close


# ── Per-market sampling ────────────────────────────────────────────────────

def sample_market(mk: dict, samples_min: List[int], asset: str,
                  product: str) -> List[Tuple[float, int]]:
    """For one settled market, return list of (fair_p_yes, actual_outcome)
    at each requested minute-before-close.  outcome is 1 if YES won."""
    ticker = mk.get("ticker", "")
    close_iso = mk.get("close_time")
    result = (mk.get("result") or "").lower()
    if not close_iso or result not in ("yes", "no"):
        return []
    try:
        strike = float(mk.get("floor_strike") or mk.get("cap_strike") or 0)
    except Exception:
        return []
    if strike <= 0:
        return []
    try:
        close_dt = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    except Exception:
        return []
    close_ms = int(close_dt.timestamp() * 1000)
    start_ms = close_ms - 16 * 60_000  # 16 min before close, with buffer
    klines = fetch_candles_1m(product, start_ms, close_ms + 60_000)
    if not klines:
        return []
    actual = 1 if result == "yes" else 0
    out = []
    for m in samples_min:
        t_ms = close_ms - m * 60_000
        px = price_at(klines, t_ms)
        if px is None:
            continue
        fpy = fair_p_yes(px, strike, m, asset)
        out.append((fpy, actual))
    return out


# ── Calibration table ──────────────────────────────────────────────────────

def calibration_table(pairs: List[Tuple[float, int]],
                      n_bins: int = 10) -> List[dict]:
    bins: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for p, y in pairs:
        b = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[b].append((p, y))
    rows = []
    for b in range(n_bins):
        items = bins[b]
        if not items:
            rows.append({"bin": f"{b*10}-{(b+1)*10}%", "n": 0,
                         "mean_pred": None, "realized": None, "error": None})
            continue
        n = len(items)
        mp = sum(p for p, _ in items) / n
        rw = sum(y for _, y in items) / n
        rows.append({"bin": f"{b*10}-{(b+1)*10}%", "n": n,
                     "mean_pred": round(mp, 4),
                     "realized": round(rw, 4),
                     "error": round(rw - mp, 4)})
    return rows


# ── Brier / log-loss ───────────────────────────────────────────────────────

def brier(pairs):
    if not pairs: return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def logloss(pairs):
    import math
    if not pairs: return None
    s = 0.0
    for p, y in pairs:
        p = min(max(p, 1e-6), 1 - 1e-6)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(pairs)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--series", default="KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M")
    ap.add_argument("--samples", default=",".join(map(str, DEFAULT_SAMPLES)),
                    help="comma-separated minutes-before-close to evaluate")
    ap.add_argument("--output", default="calibration_results.csv")
    args = ap.parse_args()

    samples_min = [int(s) for s in args.samples.split(",")]
    series_list = args.series.split(",")

    print(f"Backtest range: last {args.days} days  series={series_list}  "
          f"sample-times (min before close)={samples_min}\n", flush=True)

    all_pairs: List[Tuple[float, int]] = []
    all_triples: List[Tuple[str, int, float, int]] = []  # (asset, t_to_close_min, fair_p, outcome)
    by_series: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    by_minutes: Dict[int, List[Tuple[float, int]]] = defaultdict(list)

    for series in series_list:
        if series not in SERIES_TO_PRODUCT:
            print(f"skip {series}: no Coinbase product", flush=True)
            continue
        asset = series[2:5]  # "BTC", "ETH", ...
        product = SERIES_TO_PRODUCT[series]
        markets = fetch_settled_markets(series, args.days)
        sampled = 0
        for i, mk in enumerate(markets):
            pairs = sample_market(mk, samples_min, asset, product)
            for k, (fpy, y) in enumerate(pairs):
                all_pairs.append((fpy, y))
                by_series[series].append((fpy, y))
                m = samples_min[k] if k < len(samples_min) else -1
                by_minutes[m].append((fpy, y))
                all_triples.append((asset, m, fpy, y))
            sampled += len(pairs)
            if (i + 1) % 50 == 0:
                print(f"  [{series}] processed {i+1}/{len(markets)} markets, "
                      f"{sampled} samples", flush=True)
        print(f"  [{series}] DONE: {sampled} (pred, outcome) pairs\n", flush=True)

    if not all_pairs:
        print("no data collected.  exiting.", flush=True)
        return

    # ── Overall table ────────────────────────────────────────────────────
    print("=" * 70)
    print("OVERALL CALIBRATION")
    print("=" * 70)
    print(f"{'bin':>10}  {'n':>5}  {'mean_pred':>10}  "
          f"{'realized':>10}  {'error':>10}")
    rows = calibration_table(all_pairs)
    for r in rows:
        if r["n"] == 0:
            print(f"{r['bin']:>10}  {0:>5}  {'-':>10}  {'-':>10}  {'-':>10}")
            continue
        print(f"{r['bin']:>10}  {r['n']:>5}  {r['mean_pred']:>10.4f}  "
              f"{r['realized']:>10.4f}  {r['error']:>+10.4f}")
    print(f"\ntotal pairs:  {len(all_pairs)}")
    print(f"Brier score:  {brier(all_pairs):.5f}  (lower is better; 0.25 = coin flip)")
    print(f"Log loss:     {logloss(all_pairs):.5f}  (lower is better; 0.693 = coin flip)")

    # ── Per-asset ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SERIES")
    print("=" * 70)
    for s, pairs in by_series.items():
        if not pairs: continue
        print(f"\n[{s}]  n={len(pairs)}  Brier={brier(pairs):.5f}")
        rows = calibration_table(pairs)
        for r in rows:
            if r["n"] == 0: continue
            print(f"  {r['bin']:>10}  n={r['n']:>4}  "
                  f"mean_pred={r['mean_pred']:.3f}  "
                  f"realized={r['realized']:.3f}  err={r['error']:+.3f}")

    # ── Per-minutes-before-close ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER MINUTES-BEFORE-CLOSE")
    print("=" * 70)
    for m in sorted(by_minutes.keys()):
        pairs = by_minutes[m]
        if not pairs: continue
        print(f"\n[t-{m}min]  n={len(pairs)}  Brier={brier(pairs):.5f}")
        rows = calibration_table(pairs)
        for r in rows:
            if r["n"] == 0: continue
            print(f"  {r['bin']:>10}  n={r['n']:>4}  "
                  f"mean_pred={r['mean_pred']:.3f}  "
                  f"realized={r['realized']:.3f}  err={r['error']:+.3f}")

    # ── Save raw triples to CSV for follow-up analysis ───────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset", "minutes_to_close", "fair_p", "outcome"])
        for a, m, p, y in all_triples:
            w.writerow([a, m, f"{p:.6f}", y])
    print(f"\nraw triples saved → {out_path}  ({len(all_triples)} rows)", flush=True)


if __name__ == "__main__":
    main()
