"""
Phase 0 — ground-truth corpus + basis measurement for the v3 fair-value model.

Pulls SETTLED Kalshi crypto markets (15M up/down + hourly threshold/range) and,
for each, records the raw facts needed to fit/score any fair-value model:

  asset, family, close_time, ticker, strike_type, floor_strike, cap_strike,
  mins_left, spot_t, sigma_t, exp_val, coinbase_close, basis, outcome

KEY INSIGHT (verified against the live API 2026-06-08):
  * BOTH families settle on the SAME engine: the 60-second simple average of the
    CF Benchmarks Real-Time Index over the final minute. `expiration_value` IS
    that realized 60s-avg — exact ground truth for outcome AND basis.
  * The 15M up/down markets carry a `floor_strike` that equals the PRIOR window's
    `expiration_value` (the locked start-of-window 60s-avg). So a 15M market is
    just  "is settle >= <locked prior settle>"  — same shape as a threshold
    market. We store both uniformly and let the model layer handle strike_type.

`basis = expiration_value (BRTI 60s-avg) - coinbase_close (spot at close)` is the
single largest unmodeled error in v2 (which prices off Coinbase spot). We measure
its distribution here so v3 can correct it.

Usage:
  python3 build_corpus.py --days 7 \
      --series KXBTC15M,KXETH15M,KXSOL15M,KXBTCD,KXETHD,KXSOLD,KXXRPD,KXDOGED \
      --out corpus.csv --basis-out basis_report.json
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

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com"

# asset -> coinbase product
PRODUCT = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "DOGE": "DOGE-USD", "BNB": "BNB-USD", "ADA": "ADA-USD", "LTC": "LTC-USD",
}
SYMS = sorted(PRODUCT.keys(), key=len, reverse=True)

# Per-asset fallback per-minute sigma (mirrors fair_price_model_v2).
FALLBACK_SIGMA = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0025, "XRP": 0.0020, "HYPE": 0.0030,
    "BNB": 0.0018, "TON": 0.0025, "DOGE": 0.0030, "ADA": 0.0020, "LTC": 0.0022,
}
SIGMA_FLOOR_MULT, SIGMA_CEIL_MULT = 0.25, 5.0

# Horizons (minutes before close) to sample, per family.
HORIZONS_15M = [1, 2, 3, 4, 5, 7, 9, 11, 13]
HORIZONS_HOURLY = [1, 2, 3, 5, 8, 12, 18, 25, 35, 45, 55]


def family_of(series: str) -> str:
    return "15m" if series.upper().endswith("15M") else "hourly"


def coin_of(series: str):
    body = series.upper()
    body = body[2:] if body.startswith("KX") else body
    for s in SYMS:
        if body.startswith(s):
            return s
    return None


def fetch_settled(series: str, days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out, cursor, page = [], None, 0
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(f"{KALSHI}/markets", params=params, timeout=25)
        except Exception as e:
            print(f"  [{series}] page {page} err {e}"); break
        if not r.ok:
            print(f"  [{series}] page {page} HTTP {r.status_code}"); break
        body = r.json()
        batch = body.get("markets", [])
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
                stop = True
                continue
            out.append(mk)
        cursor = body.get("cursor")
        page += 1
        if not cursor or stop:
            break
        time.sleep(0.1)
    print(f"  [{series}] {len(out)} settled over {days}d ({page} pages)")
    return out


def fetch_klines(product: str, start_ms: int, end_ms: int):
    """1-min candles ascending [t_ms, open, high, low, close, vol]."""
    try:
        r = requests.get(
            f"{COINBASE}/products/{product}/candles",
            params={"granularity": 60,
                    "start": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
                    "end": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()},
            timeout=20)
        if not r.ok:
            return []
        rows = r.json()
    except Exception:
        return []
    out = []
    for row in reversed(rows):
        try:
            t, lo, hi, op, cl, vol = row
            out.append([int(t) * 1000, float(op), float(hi), float(lo), float(cl), float(vol)])
        except Exception:
            continue
    return out


def _candle_containing(kl, t_ms):
    """Index of the 1-min candle [start, start+60s) that contains t_ms."""
    if not kl or t_ms < kl[0][0] or t_ms > kl[-1][0] + 60_000:
        return None
    lo, hi = 0, len(kl) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if kl[mid][0] + 60_000 <= t_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def price_end_at(kl, t_ms):
    """Price AT t_ms = close of the candle ENDING at t_ms (start = t_ms-60s).
    This is the correct 1-min proxy for the instantaneous price the live model
    sees at t_ms (NOT the candle starting at t_ms, which closes 60s later)."""
    i = _candle_containing(kl, t_ms - 1)   # candle [t-60s, t) contains t-1
    return float(kl[i][4]) if i is not None else None


def ohlc4_window(kl, t_ms):
    """OHLC4 of the 60s candle ENDING at t_ms — a proxy for the 60s-average
    over [t-60s, t] (the settlement window).  Used to estimate the PURE venue
    index basis (avg-vs-avg), separating it from endpoint-vs-avg timing noise."""
    i = _candle_containing(kl, t_ms - 1)
    if i is None:
        return None
    _, op, hi, lo, cl, _ = kl[i]
    return (op + hi + lo + cl) / 4.0


def sigma_at(kl, t_ms, asset, window_min=10):
    fb = FALLBACK_SIGMA.get(asset, 0.0015)
    # closes of candles whose END is in (t_ms - window, t_ms]  (start in
    # [t-(w+1)min, t-1min]) — i.e. prices at the w minutes ending at t_ms.
    closes = [k[4] for k in kl
              if t_ms - (window_min + 1) * 60_000 <= k[0] <= t_ms - 60_000]
    if len(closes) >= 4:
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
                if closes[i] > 0 and closes[i - 1] > 0]
        if len(rets) >= 2:
            mean = sum(rets) / len(rets)
            var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
            s = math.sqrt(var)
            return max(fb * SIGMA_FLOOR_MULT, min(fb * SIGMA_CEIL_MULT, s))
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--series", default="KXBTC15M,KXETH15M,KXSOL15M,KXBTCD,KXETHD,KXSOLD,KXXRPD,KXDOGED")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.csv"))
    ap.add_argument("--basis-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "basis_report.json"))
    args = ap.parse_args()

    FIELDS = ["asset", "family", "close_time", "ticker", "strike_type",
              "floor_strike", "cap_strike", "mins_left", "spot_t", "sigma_t",
              "exp_val", "coinbase_close", "coinbase_avg", "basis", "basis_idx",
              "outcome"]
    fout = open(args.out, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    w.writeheader()

    # basis samples: asset -> list of (basis_endpoint, basis_idx_avg, family)
    basis_samples = defaultdict(list)
    n_rows = 0

    for series in [s.strip() for s in args.series.split(",") if s.strip()]:
        asset = coin_of(series)
        product = PRODUCT.get(asset)
        fam = family_of(series)
        if not product:
            print(f"  skip {series}: no product"); continue
        mkts = fetch_settled(series, args.days)

        by_close = defaultdict(list)
        for mk in mkts:
            res = (mk.get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            try:
                ev = float(mk.get("expiration_value"))
            except (TypeError, ValueError):
                continue
            by_close[mk["close_time"]].append({
                "ticker": mk["ticker"],
                "strike_type": mk.get("strike_type") or "",
                "floor_strike": mk.get("floor_strike"),
                "cap_strike": mk.get("cap_strike"),
                "exp_val": ev,
                "outcome": 1 if res == "yes" else 0,
            })

        horizons = HORIZONS_15M if fam == "15m" else HORIZONS_HOURLY
        print(f"  [{series}] {len(by_close)} close-times; fetching klines...")
        for ci, items in by_close.items():
            try:
                close_dt = datetime.fromisoformat(ci.replace("Z", "+00:00"))
            except Exception:
                continue
            close_ms = int(close_dt.timestamp() * 1000)
            kl = fetch_klines(product, close_ms - (max(horizons) + 12) * 60_000, close_ms + 120_000)
            if not kl:
                continue
            cb_close = price_end_at(kl, close_ms)   # Coinbase price AT close
            cb_avg = ohlc4_window(kl, close_ms)     # Coinbase 60s-avg proxy
            ev = items[0]["exp_val"]  # same for the whole ladder
            if cb_close and cb_close > 0:
                b_end = math.log(ev / cb_close)
                b_idx = math.log(ev / cb_avg) if cb_avg and cb_avg > 0 else b_end
                basis_samples[asset].append((b_end, b_idx, fam))
            for m in horizons:
                t_ms = close_ms - m * 60_000
                px = price_end_at(kl, t_ms)
                if px is None:
                    continue
                sig = sigma_at(kl, t_ms, asset)
                for it in items:
                    w.writerow({
                        "asset": asset, "family": fam, "close_time": ci,
                        "ticker": it["ticker"], "strike_type": it["strike_type"],
                        "floor_strike": it["floor_strike"], "cap_strike": it["cap_strike"],
                        "mins_left": m, "spot_t": round(px, 6), "sigma_t": round(sig, 8),
                        "exp_val": it["exp_val"],
                        "coinbase_close": round(cb_close, 6) if cb_close else "",
                        "coinbase_avg": round(cb_avg, 6) if cb_avg else "",
                        "basis": round(math.log(it["exp_val"] / cb_close), 8) if cb_close else "",
                        "basis_idx": round(math.log(it["exp_val"] / cb_avg), 8) if (cb_avg and cb_avg > 0) else "",
                        "outcome": it["outcome"],
                    })
                    n_rows += 1
            time.sleep(0.07)
        fout.flush()

    fout.close()
    print(f"\nWrote {n_rows} sample rows -> {args.out}")

    # ── basis report ────────────────────────────────────────────────────────
    def stats(vals):
        n = len(vals)
        if n == 0:
            return None
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / max(1, n - 1)
        sd = math.sqrt(var)
        sv = sorted(vals)
        def q(p): return sv[min(n - 1, int(p * n))]
        return {"n": n, "mean_bps": round(mu * 1e4, 3), "std_bps": round(sd * 1e4, 3),
                "p05_bps": round(q(0.05) * 1e4, 3), "p50_bps": round(q(0.50) * 1e4, 3),
                "p95_bps": round(q(0.95) * 1e4, 3), "abs_mean_bps": round(sum(abs(v) for v in vals) / n * 1e4, 3)}

    report = {}
    print("\n=== BASIS in bps.  endpoint = log(settle / Coinbase price@close);")
    print("    idx = log(settle / Coinbase 60s-avg proxy)  [≈ pure venue basis] ===")
    print(f"{'asset':>6} {'fam':>7} {'n':>5} | {'end_mean':>9} {'end_std':>9} | {'idx_mean':>9} {'idx_std':>9}")
    for asset in sorted(basis_samples):
        for fam in ("15m", "hourly", "all"):
            ev = [b for b, bi, f in basis_samples[asset] if fam == "all" or f == fam]
            iv = [bi for b, bi, f in basis_samples[asset] if fam == "all" or f == fam]
            st, sti = stats(ev), stats(iv)
            if not st:
                continue
            report.setdefault(asset, {})[fam] = {"endpoint": st, "index": sti}
            print(f"{asset:>6} {fam:>7} {st['n']:>5} | {st['mean_bps']:>+9.2f} {st['std_bps']:>9.2f} | "
                  f"{sti['mean_bps']:>+9.2f} {sti['std_bps']:>9.2f}")

    with open(args.basis_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote basis report -> {args.basis_out}")


if __name__ == "__main__":
    main()
