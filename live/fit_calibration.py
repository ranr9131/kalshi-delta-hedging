"""
Fit Platt scaling (sigmoid in logit space) per asset to correct the
log-normal model's overconfidence in the longshot zone.

Input:  calibration_results.csv with columns (asset, minutes_to_close, fair_p, outcome)
Output: calibration.json with per-asset (a, b) such that
            corrected_p = sigmoid(a + b * logit(model_p))

The fit minimises log-loss via gradient descent (closed-form NR also works,
but we want zero deps beyond stdlib + numpy if available).  If sklearn is
available we use that; otherwise we fall back to a simple Newton-Raphson on
binary logistic regression.
"""
from __future__ import annotations
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import List, Tuple


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_platt(pairs: List[Tuple[float, int]], iters: int = 200,
              lr: float = 0.1) -> Tuple[float, float, dict]:
    """Logistic regression with one feature (logit of model_p) + bias.
    Returns (a, b, info)."""
    if not pairs:
        return 0.0, 1.0, {"n": 0}
    xs = [logit(p) for p, _ in pairs]
    ys = [float(y)  for _, y in pairs]
    a = 0.0
    b = 1.0
    n = len(xs)
    for _ in range(iters):
        # Predictions
        zs = [a + b * x for x in xs]
        ps = [sigmoid(z) for z in zs]
        # Gradient: dL/da = sum(p - y) / n, dL/db = sum((p - y) * x) / n
        ga = sum(p - y for p, y in zip(ps, ys)) / n
        gb = sum((p - y) * x for p, y, x in zip(ps, ys, xs)) / n
        a -= lr * ga
        b -= lr * gb
    # Log loss + Brier on training set (sanity)
    ll = sum(-(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
             for p, y in zip(ps, ys)) / n
    br = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    return a, b, {"n": n, "log_loss": ll, "brier": br}


def baseline_metrics(pairs: List[Tuple[float, int]]) -> dict:
    if not pairs: return {"n": 0}
    n = len(pairs)
    ll = sum(-(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
             for p, y in pairs) / n
    br = sum((p - y) ** 2 for p, y in pairs) / n
    return {"n": n, "log_loss": ll, "brier": br}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "calibration_results.csv")
    out_path = os.path.join(here, "calibration.json")

    by_asset: dict[str, List[Tuple[float, int]]] = defaultdict(list)
    all_pairs: List[Tuple[float, int]] = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                p = float(r["fair_p"]); y = int(r["outcome"])
            except Exception:
                continue
            by_asset[r["asset"]].append((p, y))
            all_pairs.append((p, y))

    if not all_pairs:
        print(f"no data in {csv_path}", flush=True)
        return

    out = {}
    print(f"{'asset':>5}  {'n':>5}  "
          f"{'a':>8} {'b':>8}  "
          f"{'baseline_ll':>12} {'fitted_ll':>10}  "
          f"{'baseline_brier':>14} {'fitted_brier':>12}")
    for asset in ("BTC", "ETH", "SOL", "XRP"):
        pairs = by_asset.get(asset, [])
        if not pairs:
            print(f"{asset:>5}  no data")
            continue
        base = baseline_metrics(pairs)
        a, b, info = fit_platt(pairs)
        out[asset] = {"a": a, "b": b, **info,
                      "baseline_log_loss": base["log_loss"],
                      "baseline_brier":    base["brier"]}
        print(f"{asset:>5}  {info['n']:>5}  "
              f"{a:>+8.4f} {b:>+8.4f}  "
              f"{base['log_loss']:>12.5f} {info['log_loss']:>10.5f}  "
              f"{base['brier']:>14.5f} {info['brier']:>12.5f}")

    # Also fit a "global" mapping as a fallback for unknown assets.
    a_g, b_g, info_g = fit_platt(all_pairs)
    out["_global"] = {"a": a_g, "b": b_g, **info_g}
    print(f"\nglobal: a={a_g:+.4f} b={b_g:+.4f} n={info_g['n']} "
          f"log_loss={info_g['log_loss']:.5f}")

    # Show how the fit transforms sample probabilities
    print("\nExample correction (BTC fit):")
    btc = out.get("BTC") or out["_global"]
    a, b = btc["a"], btc["b"]
    print(f"  raw model_p  →  corrected_p")
    for raw in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        corrected = sigmoid(a + b * logit(raw))
        print(f"    {raw:.2f}        →  {corrected:.3f}")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
