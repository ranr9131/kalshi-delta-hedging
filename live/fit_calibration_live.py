"""
Fit Platt scaling from LIVE snipes.csv + settlements.csv.

Why this matters: the original calibration was fit on historical Kalshi
markets sampled at fixed minute intervals — a reasonable proxy but not the
exact conditions the sniper actually sees.  Each snipes.csv row, by
contrast, captures the EXACT (price, strike, mins_left, asset) at the
moment a fire was considered, which is more representative.

Output: calibration_v2.json  (same schema as the v1 calibration.json)

The v2 sniper loads this file via fair_price_model_v2.py.  Re-run this
script periodically (e.g. daily) as more data accumulates.

CLI:
    python fit_calibration_live.py              # uses default paths
    python fit_calibration_live.py --min-n 30   # require >= 30 pairs per asset
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_price_model import fair_p_yes_raw


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def fit_platt(pairs: List[Tuple[float, int]],
              iters: int = 400, lr: float = 0.1) -> Tuple[float, float, dict]:
    """Logistic regression with one feature (logit of model_p) + bias."""
    if not pairs:
        return 0.0, 1.0, {"n": 0}
    xs = [logit(p) for p, _ in pairs]
    ys = [float(y) for _, y in pairs]
    a, b = 0.0, 1.0
    n = len(xs)
    for _ in range(iters):
        zs = [a + b * x for x in xs]
        ps = [sigmoid(z) for z in zs]
        ga = sum(p - y for p, y in zip(ps, ys)) / n
        gb = sum((p - y) * x for p, y, x in zip(ps, ys, xs)) / n
        a -= lr * ga
        b -= lr * gb
    ll = sum(-(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
             for p, y in zip(ps, ys)) / n
    br = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    return a, b, {"n": n, "log_loss": ll, "brier": br}


def baseline_metrics(pairs: List[Tuple[float, int]]) -> dict:
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    ll = sum(-(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
             for p, y in pairs) / n
    br = sum((p - y) ** 2 for p, y in pairs) / n
    return {"n": n, "log_loss": ll, "brier": br}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--snipes",  default=os.path.join(here, "snipes.csv"))
    ap.add_argument("--settle",  default=os.path.join(here, "settlements.csv"))
    ap.add_argument("--out",     default=os.path.join(here, "calibration_v2.json"))
    ap.add_argument("--min-n",   type=int, default=20,
                    help="minimum pairs required per asset to emit a fit")
    ap.add_argument("--merge",   action="store_true",
                    help="merge with existing --out file; keep existing entry "
                         "if its n >= new fit's n (preserves curated patches)")
    args = ap.parse_args()

    if not os.path.exists(args.snipes) or not os.path.exists(args.settle):
        print(f"missing {args.snipes} or {args.settle}", file=sys.stderr)
        sys.exit(1)

    snipes = list(csv.DictReader(open(args.snipes)))
    settlements = {r["ticker"]: r for r in csv.DictReader(open(args.settle))
                   if r.get("ticker")}

    # Build (asset, raw_p, yes_outcome) triples.  De-dup by ticker so each
    # market contributes ONCE — otherwise markets with many snipes get
    # weighted absurdly heavy.
    seen_tickers = set()
    by_asset: dict[str, List[Tuple[float, int]]] = defaultdict(list)
    skipped_no_settlement = 0
    skipped_bad_result = 0
    skipped_dupe = 0

    for r in snipes:
        ticker = r.get("ticker")
        if not ticker:
            continue
        if ticker in seen_tickers:
            skipped_dupe += 1
            continue
        s = settlements.get(ticker)
        if not s:
            skipped_no_settlement += 1
            continue
        result = (s.get("result") or "").lower()
        if result not in ("yes", "no"):
            skipped_bad_result += 1
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

        raw_p_yes = fair_p_yes_raw(price, strike, mins, asset)
        y = 1 if result == "yes" else 0
        by_asset[asset].append((raw_p_yes, y))

    total_pairs = sum(len(p) for p in by_asset.values())
    print(f"loaded {len(snipes)} snipes, {len(settlements)} settlements")
    print(f"built {total_pairs} (raw_p, outcome) pairs across {len(by_asset)} assets")
    print(f"  skipped_no_settlement={skipped_no_settlement}  "
          f"skipped_bad_result={skipped_bad_result}  dupes={skipped_dupe}")
    print()

    out = {}
    print(f"{'asset':>5}  {'n':>5}  {'a':>8}  {'b':>8}  "
          f"{'base_ll':>9}  {'fit_ll':>9}  {'base_brier':>11}  {'fit_brier':>10}")
    for asset in ("BTC", "ETH", "SOL", "XRP"):
        pairs = by_asset.get(asset, [])
        if len(pairs) < args.min_n:
            print(f"{asset:>5}  {len(pairs):>5}  -- below min-n={args.min_n}, skipping --")
            continue
        base = baseline_metrics(pairs)
        a, b, info = fit_platt(pairs)
        out[asset] = {
            "a": a, "b": b, **info,
            "baseline_log_loss": base["log_loss"],
            "baseline_brier":    base["brier"],
        }
        print(f"{asset:>5}  {info['n']:>5}  "
              f"{a:>+8.4f}  {b:>+8.4f}  "
              f"{base['log_loss']:>9.4f}  {info['log_loss']:>9.4f}  "
              f"{base['brier']:>11.5f}  {info['brier']:>10.5f}")

    # Global fallback fit using all pairs
    all_pairs: List[Tuple[float, int]] = []
    for p in by_asset.values():
        all_pairs.extend(p)
    if all_pairs:
        a_g, b_g, info_g = fit_platt(all_pairs)
        out["_global"] = {"a": a_g, "b": b_g, **info_g}
        print(f"\nglobal: n={info_g['n']}  a={a_g:+.4f}  b={b_g:+.4f}  "
              f"log_loss={info_g['log_loss']:.5f}")

    if not out:
        print("no assets met min-n threshold; not writing output", file=sys.stderr)
        sys.exit(1)

    # Merge mode: load existing, keep any entry where existing_n >= new_n.
    # This prevents accidentally clobbering a curated fit (e.g. the ETH
    # patch where we copied V1's historical fit n=2736 because the live
    # V2 fit n=30 was unreliable) with a smaller live re-fit.
    if args.merge and os.path.exists(args.out):
        try:
            existing = json.load(open(args.out))
        except Exception:
            existing = {}
        kept = []
        for asset, new_entry in list(out.items()):
            old = existing.get(asset)
            if not old:
                continue
            old_n = old.get("n", 0)
            new_n = new_entry.get("n", 0)
            if old_n >= new_n:
                out[asset] = old
                kept.append(f"{asset}(old n={old_n} >= new n={new_n})")
        # Bring forward any existing entries not in the new fit
        for asset, old_entry in existing.items():
            if asset not in out:
                out[asset] = old_entry
                kept.append(f"{asset}(preserved, not refit)")
        if kept:
            print("\nmerge preserved: " + ", ".join(kept))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
