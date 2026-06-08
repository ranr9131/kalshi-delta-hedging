"""
Validate the last-minute (tau<1) variance law against Binance 1s ground truth.

Checks two things on corpus_1s.csv:

1. VARIANCE LAW.  The within-venue residual  r = ln(true_avg_1s / spot_t)  should
   have  Var[r] = vol_mult^2 * sigma_t^2 * g(tau)  with
       g_new(tau) = (tau^3 + (1-tau)^3)/3     (v3, with the elapsed-window term)
       g_old(tau) =  tau^3 / 3                (the buggy version, future only)
   We print, per horizon, empirical Var[r] and the model's mean predicted var
   under each law, plus mean(z^2) (should be ~1 for the correct law).  The
   tau->0 floor should be ~sigma * sqrt(1/3).

2. BINARY CALIBRATION at tau<1.  Brier/log-loss of P(settle>=strike) under the
   OLD vs NEW variance law, predicting the real outcome from spot_t.  (Uses the
   calibrated 15m params: vol_mult, sigma_b, nu, sd_mult.)

Usage:  python3 validate_1s.py --corpus corpus_1s.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_price_model_v3 import _sf_unit, AVG_WINDOW_MIN

W = AVG_WINDOW_MIN


def g_new(tau):
    return (tau - 2*W/3) if tau >= W else (tau**3 + (W - tau)**3) / (3*W*W)


def g_old(tau):
    return (tau - 2*W/3) if tau >= W else (tau**3) / (3*W*W)


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "asset": r["asset"], "tau": float(r["mins_left"]),
                    "spot": float(r["spot_t"]), "sigma": float(r["sigma_t"]),
                    "exp_val": float(r["exp_val"]), "avg": float(r["true_avg_1s"]),
                    "partial": float(r["partial_avg_1s"]) if r.get("partial_avg_1s") else None,
                    "floor": float(r["floor_strike"]) if r["floor_strike"] else None,
                    "y": int(r["outcome"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def params_15m(asset):
    p = {"vol_mult": 1.0, "basis_std": 0.0002, "nu": 200.0, "sd_mult": 1.0, "basis_mean": 0.0}
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "calibration_v3.json")) as f:
            cal = json.load(f)
        fp = cal["assets"][asset]["by_family"]["15m"]
        p.update({k: fp[k] for k in p if k in fp})
    except Exception:
        pass
    return p


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def logloss(pairs):
    s = 0.0
    for p, y in pairs:
        p = min(max(p, 1e-9), 1 - 1e-9)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(pairs) if pairs else float("nan")


def p_above(spot, strike, sd, nu, mean_shift):
    if not strike or strike <= 0:
        return 1.0
    mean = math.log(spot) + mean_shift
    return _sf_unit((math.log(strike) - mean) / sd, nu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_1s.csv"))
    args = ap.parse_args()
    rows = load(args.corpus)
    print(f"loaded {len(rows)} fine-horizon rows; assets={sorted(set(r['asset'] for r in rows))}\n")

    print("== 1. VARIANCE LAW: within-venue residual r = ln(true_avg/spot) ==")
    print("   (mean(z^2) should be ~1 for the correct law; floor at tau->0 ~ sigma*sqrt(1/3))\n")
    for asset in sorted(set(r["asset"] for r in rows)):
        pa = params_15m(asset)
        vm = pa["vol_mult"]
        print(f"  --- {asset}  (vol_mult={vm}) ---")
        print(f"    {'tau':>7} {'n':>5} {'sd(spot)':>9} {'sd(part)':>9} {'z2_old':>8} {'z2_new':>8} {'z2_part':>8}")
        by = defaultdict(list)
        for r in rows:
            if r["asset"] == asset and r["avg"] > 0 and r["spot"] > 0:
                by[r["tau"]].append(r)
        for tau in sorted(by):
            sub = by[tau]
            rs = [math.log(r["avg"] / r["spot"]) for r in sub]
            mu = sum(rs) / len(rs)
            emp_var = sum((x - mu) ** 2 for x in rs) / max(1, len(rs) - 1)
            z2o = sum((math.log(r["avg"]/r["spot"]))**2 /
                      max(vm*vm*r["sigma"]**2*g_old(tau), 1e-18) for r in sub) / len(sub)
            z2n = sum((math.log(r["avg"]/r["spot"]))**2 /
                      max(vm*vm*r["sigma"]**2*g_new(tau), 1e-18) for r in sub) / len(sub)
            # partial: residual vs (1-tau)*ln(partial)+tau*ln(spot), var=sigma^2 tau^3/3
            psub = [r for r in sub if r["partial"] and r["partial"] > 0 and tau < 1.0]
            if psub:
                presid = [math.log(r["avg"]) - ((1-tau)*math.log(r["partial"]) + tau*math.log(r["spot"]))
                          for r in psub]
                pmu = sum(presid)/len(presid)
                p_sd = math.sqrt(sum((x-pmu)**2 for x in presid)/max(1, len(presid)-1))
                z2p = sum(pr**2 / max(vm*vm*r["sigma"]**2*(tau**3/3.0), 1e-18)
                          for pr, r in zip(presid, psub)) / len(psub)
                p_sd_s, z2p_s = f"{p_sd*1e4:>9.2f}", f"{z2p:>8.2f}"
            else:
                p_sd_s, z2p_s = f"{'-':>9}", f"{'-':>8}"
            print(f"    {tau:>7.3f} {len(sub):>5} {math.sqrt(emp_var)*1e4:>9.2f} {p_sd_s} {z2o:>8.2f} {z2n:>8.2f} {z2p_s}")
        print()

    print("== 2. BINARY CALIBRATION at tau<1 (predict outcome from spot) ==")
    print(f"  {'asset':>6} {'n':>5} | {'OLD ll':>8} | {'NEW ll':>8} | {'PARTIAL ll':>10}  (brier in parens)")
    A_old, A_new, A_part = [], [], []
    for asset in sorted(set(r["asset"] for r in rows)):
        pa = params_15m(asset)
        vm, sb, nu, sdm, bm = pa["vol_mult"], pa["basis_std"], pa["nu"], pa["sd_mult"], pa["basis_mean"]
        old_p, new_p, part_p = [], [], []
        for r in rows:
            if r["asset"] != asset or r["tau"] >= 1.0 or not r["floor"]:
                continue
            base = vm*vm*r["sigma"]**2
            sd_old = math.sqrt(max(base*g_old(r["tau"]) + sb*sb, 1e-18)) * sdm
            sd_new = math.sqrt(max(base*g_new(r["tau"]) + sb*sb, 1e-18)) * sdm
            old_p.append((p_above(r["spot"], r["floor"], sd_old, nu, bm), r["y"]))
            new_p.append((p_above(r["spot"], r["floor"], sd_new, nu, bm), r["y"]))
            # PARTIAL: reduced var (tau^3/3 only) + re-centered mean
            if r["partial"] and r["partial"] > 0:
                sd_p = math.sqrt(max(base*(r["tau"]**3/3.0) + sb*sb, 1e-18)) * sdm
                mean_shift = (1-r["tau"])*math.log(r["partial"]) + r["tau"]*math.log(r["spot"]) - math.log(r["spot"]) + bm
                part_p.append((p_above(r["spot"], r["floor"], sd_p, nu, mean_shift), r["y"]))
            else:
                part_p.append((new_p[-1][0], r["y"]))
        if not old_p:
            continue
        A_old += old_p; A_new += new_p; A_part += part_p
        print(f"  {asset:>6} {len(old_p):>5} | {logloss(old_p):>8.4f} | {logloss(new_p):>8.4f} | "
              f"{logloss(part_p):>10.4f}  ({brier(old_p):.4f}/{brier(new_p):.4f}/{brier(part_p):.4f})")
    print(f"  {'ALL':>6} {len(A_old):>5} | {logloss(A_old):>8.4f} | {logloss(A_new):>8.4f} | "
          f"{logloss(A_part):>10.4f}  ({brier(A_old):.4f}/{brier(A_new):.4f}/{brier(A_part):.4f})")


if __name__ == "__main__":
    main()
