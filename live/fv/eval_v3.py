"""
Phase 3 — honest out-of-sample evaluation: v2 vs v3-raw vs v3-calibrated.

Apples-to-apples: every model is fed the SAME inputs from the corpus
(spot_t, sigma_t, strike, strike_type, mins_left).  The only differences are the
model internals.  We score on the TEST split only (held out from the fit).

Metrics: Brier + log-loss, overall (all + tradeable band), and broken down by
family, horizon bucket, and moneyness.  Plus a reliability table for v3.

  v2        = fair_price_model_v2 raw log-normal (Gaussian, spot snapshot, no
              basis, no drift) — the current production model.
  v3_raw    = v3 physics, no Platt.
  v3_cal    = v3 physics + residual Platt (calibration_v3.json).

Usage:
  python3 eval_v3.py --corpus corpus.csv --test-frac 0.3
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fair_price_model_v3 as v3
from fair_price_model_v2 import fair_p_yes_raw_v2
from fit_v3 import load, time_split, bucket_label, brier, logloss


def p_v2(r):
    """v2-equivalent P(YES) on the same inputs, strike_type-aware."""
    tau, spot, sig, asset = r["mins_left"], r["spot_t"], r["sigma_t"], r["asset"]
    def above(K):
        if not K or K <= 0:
            return 1.0
        return fair_p_yes_raw_v2(spot, K, tau, asset, sigma_per_min=sig)
    st = (r["strike_type"] or "greater").lower()
    if st in ("greater", "greater_or_equal"):
        return above(r["floor_strike"])
    if st == "less":
        return 1.0 - above(r["cap_strike"])
    if st == "between":
        return min(max(above(r["floor_strike"]) - above(r["cap_strike"]), 0.0), 1.0)
    return 0.5


def p_v3(r, calibrated):
    return v3.fair_p(r["spot_t"], r["mins_left"], r["asset"],
                     floor_strike=r["floor_strike"], cap_strike=r["cap_strike"],
                     strike_type=r["strike_type"], sigma_per_min=r["sigma_t"],
                     calibrated=calibrated)


def score(rows, predfn, band=None):
    pairs = []
    for r in rows:
        p = predfn(r)
        if band and not (band[0] < p < band[1]):
            continue
        pairs.append((p, r["outcome"]))
    return pairs


def line(name, pairs):
    return f"  {name:>10}: n={len(pairs):>6}  brier={brier(pairs):.4f}  logloss={logloss(pairs):.4f}"


def reliability(rows, predfn, nbins=10):
    bins = defaultdict(list)
    for r in rows:
        p = predfn(r)
        bins[min(nbins - 1, int(p * nbins))].append((p, r["outcome"]))
    print(f"    {'bin':>9} {'n':>6} {'pred':>7} {'real':>7} {'err':>7}")
    for b in range(nbins):
        it = bins[b]
        if not it:
            continue
        mp = sum(p for p, _ in it) / len(it)
        rw = sum(y for _, y in it) / len(it)
        print(f"    {b*10:>3}-{b*10+10:<3}% {len(it):>6} {mp:>7.3f} {rw:>7.3f} {rw-mp:>+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.csv"))
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--band", nargs=2, type=float, default=[0.05, 0.95])
    args = ap.parse_args()

    rows = load(args.corpus)
    _, test, cut = time_split(rows, args.test_frac)
    print(f"TEST set: {len(test)} rows (close_time >= {cut})\n")
    band = tuple(args.band)

    models = [("v2", lambda r: p_v2(r)),
              ("v3_raw", lambda r: p_v3(r, False)),
              ("v3_cal", lambda r: p_v3(r, True))]

    def block(title, subset):
        if not subset:
            return
        print(f"== {title}  (n_all={len(subset)}) ==")
        for name, fn in models:
            # full
            full = [(fn(r), r["outcome"]) for r in subset]
            bnd = [(p, y) for p, y in full if band[0] < p < band[1]]
            print(f"  {name:>7}  ALL brier={brier(full):.4f} ll={logloss(full):.4f}  |  "
                  f"BAND n={len(bnd):>5} brier={brier(bnd):.4f} ll={logloss(bnd):.4f}")
        print()

    block("OVERALL", test)
    for fam in ("15m", "hourly"):
        block(f"FAMILY={fam}", [r for r in test if r["family"] == fam])
    print("== BY ASSET ==")
    for asset in sorted(set(r["asset"] for r in test)):
        sub = [r for r in test if r["asset"] == asset]
        out = []
        for name, fn in models:
            bnd = [(fn(r), r["outcome"]) for r in sub]
            bnd = [(p, y) for p, y in bnd if band[0] < p < band[1]]
            out.append(f"{name} {brier(bnd):.4f}/{logloss(bnd):.4f}")
        print(f"  {asset:>5} (n={len(sub):>5}): " + "  ".join(out))
    print()

    print("== BY HORIZON BUCKET (band brier / logloss) ==")
    for lo, hi in v3.__dict__.get("_dummy", []) or [(0, 12), (12, 25), (25, 50), (50, 100), (100, 1e9)]:
        sub = [r for r in test if lo <= r["mins_left"] < hi]
        if not sub:
            continue
        out = []
        for name, fn in models:
            bnd = [(fn(r), r["outcome"]) for r in sub]
            bnd = [(p, y) for p, y in bnd if band[0] < p < band[1]]
            out.append(f"{name} {brier(bnd):.4f}/{logloss(bnd):.4f}")
        lbl = f"{lo}-{int(hi) if hi < 1e8 else 'inf'}m"
        print(f"  {lbl:>9} (n={len(sub):>5}): " + "  ".join(out))
    print()

    print("== v3_cal reliability (TEST, all) ==")
    reliability(test, lambda r: p_v3(r, True))
    print("\n== v2 reliability (TEST, all) ==")
    reliability(test, lambda r: p_v2(r))


if __name__ == "__main__":
    main()
