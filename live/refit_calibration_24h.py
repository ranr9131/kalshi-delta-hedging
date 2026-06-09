"""
Refit Platt calibration from JUST the last 24 hours of data.

Pulls from V2 paper + live LEO + live FRIEND snipes (all use V2 pricing),
joins with settlements, de-dupes by ticker (so a market with 20 snipes
counts once), and fits per-asset (a, b) parameters via logistic regression
in logit-of-raw-p space.

Writes to calibration_v2_24h.json (NOT calibration_v2.json), so you can
compare before deciding to swap.

Usage:
  python3 refit_calibration_24h.py                # last 24h
  python3 refit_calibration_24h.py --hours 12     # last 12h
  python3 refit_calibration_24h.py --hours 48     # last 48h
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

# Use the same raw model the sniper uses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_price_model import fair_p_yes_raw

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(BASE, "snipes_v2.csv"),
    os.path.join(BASE, "snipes_leo.csv"),
    os.path.join(BASE, "snipes_friend.csv"),
    os.path.join(BASE, "snipes.csv"),          # V1 paper (uses same raw model)
]
SETTLE_PATH = os.path.join(BASE, "settlements.csv")
EXISTING_CAL = os.path.join(BASE, "calibration_v2.json")
OUT_PATH = os.path.join(BASE, "calibration_v2_24h.json")

ASSETS = ("BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "DOGE")


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit_platt(pairs: List[Tuple[float, int]]):
    """Find (a, b) maximizing log-likelihood for sigmoid(a + b × logit(raw_p))."""
    if not pairs:
        return 0.0, 1.0, {"n": 0, "log_loss": 0.0, "brier": 0.0}

    # Pre-compute logits and outcomes
    xs = [logit(p) for p, _ in pairs]
    ys = [y for _, y in pairs]

    # Simple Newton-Raphson on (a, b) for logistic regression
    a, b = 0.0, 1.0
    for _ in range(80):
        # Gradient + Hessian
        g_a = g_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for x, y in zip(xs, ys):
            z = a + b * x
            p = sigmoid(z)
            w = p * (1 - p)
            err = y - p
            g_a += err
            g_b += err * x
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        # Solve Hessian × delta = gradient (no minus because we ADD to maximize LL)
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (-h_ab * g_a + h_aa * g_b) / det
        a += da
        b += db
        if abs(da) < 1e-6 and abs(db) < 1e-6:
            break

    # Metrics
    ll = 0.0
    brier = 0.0
    for x, y in zip(xs, ys):
        p = sigmoid(a + b * x)
        p = min(max(p, 1e-12), 1 - 1e-12)
        ll  -= y * math.log(p) + (1 - y) * math.log(1 - p)
        brier += (p - y) ** 2
    n = len(pairs)
    return a, b, {"n": n, "log_loss": ll / n, "brier": brier / n}


def baseline(pairs):
    """Metrics for trivial 'always predict mean' baseline."""
    if not pairs:
        return {"log_loss": 0.0, "brier": 0.0}
    y_mean = sum(y for _, y in pairs) / len(pairs)
    y_mean = min(max(y_mean, 1e-12), 1 - 1e-12)
    ll = brier = 0.0
    for _, y in pairs:
        ll -= y * math.log(y_mean) + (1 - y) * math.log(1 - y_mean)
        brier += (y_mean - y) ** 2
    return {"log_loss": ll / len(pairs), "brier": brier / len(pairs)}


def load_settlements():
    out = {}
    if not os.path.exists(SETTLE_PATH):
        return out
    with open(SETTLE_PATH) as f:
        for r in csv.DictReader(f):
            t = (r.get("ticker") or "").strip()
            res = (r.get("result") or "").strip().lower()
            if t and res in ("yes", "no"):
                out[t] = res
    return out


def within_hours(ts_iso, hours):
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t) <= timedelta(hours=hours)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()

    settlements = load_settlements()
    print(f"settlements: {len(settlements)} known tickers")
    print(f"window: last {args.hours:.0f} hours\n")

    # Build pairs, de-duped by ticker across all sources
    seen_tickers = set()
    by_asset: dict = defaultdict(list)
    n_in_window = 0
    n_skipped_dupe = 0
    n_skipped_no_settle = 0

    for path in SOURCES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if not within_hours(r.get("ts_iso", ""), args.hours):
                    continue
                n_in_window += 1
                ticker = r.get("ticker", "")
                if ticker in seen_tickers:
                    n_skipped_dupe += 1
                    continue
                res = settlements.get(ticker)
                if res not in ("yes", "no"):
                    n_skipped_no_settle += 1
                    continue
                seen_tickers.add(ticker)
                try:
                    price  = float(r["crypto_price"])
                    strike = float(r["strike"])
                    mins   = float(r["minutes_left"])
                    asset  = r["asset"].upper()
                except Exception:
                    continue
                if mins <= 0 or price <= 0 or strike <= 0:
                    continue
                raw_p = fair_p_yes_raw(price, strike, mins, asset)
                y = 1 if res == "yes" else 0
                by_asset[asset].append((raw_p, y))

    print(f"in window:         {n_in_window} rows across all sources")
    print(f"deduped by ticker: {n_skipped_dupe}")
    print(f"unsettled:         {n_skipped_no_settle}")
    print(f"usable pairs:      {sum(len(v) for v in by_asset.values())}")
    print()

    # Load existing calibration for comparison
    existing = {}
    if os.path.exists(EXISTING_CAL):
        existing = json.load(open(EXISTING_CAL))

    # Fit each asset
    out = {}
    print(f"{'asset':<5}  {'n':>4}  {'OLD a':>7}  {'OLD b':>7}  "
          f"{'NEW a':>7}  {'NEW b':>7}  {'brier':>6}  {'baseline':>8}  diff")
    for asset in ASSETS:
        pairs = by_asset.get(asset, [])
        if len(pairs) < args.min_n:
            print(f"{asset:<5}  {len(pairs):>4}  -- below min-n={args.min_n} --")
            continue
        old = existing.get(asset, {})
        old_a = old.get("a", 0.0)
        old_b = old.get("b", 1.0)

        a, b, info = fit_platt(pairs)
        base = baseline(pairs)

        # Tag whether fit is meaningfully better than baseline
        diff_brier = base["brier"] - info["brier"]
        marker = "BETTER" if diff_brier > 0.005 else ("same" if diff_brier > -0.005 else "WORSE")

        out[asset] = {
            "a": a, "b": b, **info,
            "baseline_log_loss": base["log_loss"],
            "baseline_brier":    base["brier"],
        }

        print(f"{asset:<5}  {info['n']:>4}  "
              f"{old_a:>+7.3f}  {old_b:>+7.3f}  "
              f"{a:>+7.3f}  {b:>+7.3f}  "
              f"{info['brier']:>6.3f}  {base['brier']:>8.3f}  {marker}")

    # Global fit
    all_pairs = []
    for p in by_asset.values(): all_pairs.extend(p)
    if all_pairs:
        a_g, b_g, info_g = fit_platt(all_pairs)
        base_g = baseline(all_pairs)
        out["_global"] = {"a": a_g, "b": b_g, **info_g,
                          "baseline_log_loss": base_g["log_loss"],
                          "baseline_brier": base_g["brier"]}
        old_g = existing.get("_global", {})
        print(f"{'_glb':<5}  {info_g['n']:>4}  "
              f"{old_g.get('a',0):>+7.3f}  {old_g.get('b',1):>+7.3f}  "
              f"{a_g:>+7.3f}  {b_g:>+7.3f}  "
              f"{info_g['brier']:>6.3f}  {base_g['brier']:>8.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {OUT_PATH}")
    print("(NOT swapped in — review and run `cp calibration_v2_24h.json calibration_v2.json` to activate)")


if __name__ == "__main__":
    main()
