"""
Phase 2 — fit calibration_v3.json from the ground-truth corpus.

Two-stage fit, time-split (train on earliest dates, hold out the latest):

STAGE A — the physics (terminal-distribution moments), per asset.
  Residual  r = ln(settle 60s-avg) - ln(spot_t)  is, under the v3 model,
      r ~ location  mu_b + drift(tau)
          scale     sqrt( (vol_mult*sigma_t)^2 * g(tau) + sigma_b^2 )
          shape     standardized Student-t(nu)
  where g(tau) = Var[Ybar]/sigma^2 is the 60s-average horizon factor.

  We recover the parameters directly from data (no model circularity):
    * mu_b      = median(r)                              (location / mean basis)
    * vol_mult^2, sigma_b^2  : least-squares regress (r-mu_b)^2 on
                  [ sigma_t^2 * g(tau) ,  1 ]            (slope, intercept >= 0)
    * nu        : from the excess kurtosis of the standardized residuals
                  (t-kurtosis = 6/(nu-4)  ->  nu = 4 + 6/kurt)
  Uses UNIQUE (close_time, mins_left) rows so strike duplication doesn't bias.

STAGE B — residual Platt on the binary outcome, per (asset, horizon bucket),
  restricted to the tradeable band, to mop up any leftover miscalibration.

Diagnostics for seasonality and vol term-structure are printed; they are only
emitted into the JSON when the signal is strong enough to beat shrinkage.

Usage:
  python3 fit_v3.py --corpus corpus.csv --out ../calibration_v3.json --test-frac 0.3
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
from fair_price_model_v3 import _ybar_variance, _sf_unit, _logit, _sigmoid, AVG_WINDOW_MIN

BUCKETS = [(0, 12), (12, 25), (25, 50), (50, 100), (100, 1e9)]


def bucket_label(m):
    for lo, hi in BUCKETS:
        if lo <= m < hi:
            return f"{lo}-{int(hi) if hi < 1e8 else 'inf'}m"
    return None


def g_tau(tau):
    """Horizon factor: Var[Ybar] = sigma^2 * g(tau)."""
    return _ybar_variance(1.0, tau)


# ── load ────────────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "asset": r["asset"], "family": r["family"],
                    "close_time": r["close_time"], "mins_left": float(r["mins_left"]),
                    "spot_t": float(r["spot_t"]), "sigma_t": float(r["sigma_t"]),
                    "exp_val": float(r["exp_val"]),
                    "floor_strike": float(r["floor_strike"]) if r["floor_strike"] else None,
                    "cap_strike": float(r["cap_strike"]) if r["cap_strike"] else None,
                    "strike_type": r["strike_type"],
                    "outcome": int(r["outcome"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def time_split(rows, test_frac):
    cts = sorted(set(r["close_time"] for r in rows))
    cut = cts[int(len(cts) * (1 - test_frac))] if cts else None
    train = [r for r in rows if r["close_time"] < cut]
    test = [r for r in rows if r["close_time"] >= cut]
    return train, test, cut


# ── moments / stats helpers ──────────────────────────────────────────────────
def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols2(xs, ys):
    """y = a*x + b least squares; return (a, b)."""
    return wls2(xs, ys, [1.0] * len(xs))


def wls2(xs, ys, ws):
    """Weighted y = a*x + b least squares; return (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    den = sw * sxx - sx * sx
    if abs(den) < 1e-40:
        return 0.0, sy / sw
    a = (sw * sxy - sx * sy) / den
    b = (sy - a * sx) / sw
    return a, b


# ── Stage A ──────────────────────────────────────────────────────────────────
def fit_stage_a(rows):
    """Return per-(asset,family) physics params and a diagnostics dict, each
    keyed by (asset, family)."""
    # unique (close_time, mins_left) per (asset, family)
    seen = set()
    by_key = defaultdict(list)
    for r in rows:
        key = (r["asset"], r["family"], r["close_time"], r["mins_left"])
        if key in seen:
            continue
        seen.add(key)
        if r["spot_t"] <= 0 or r["exp_val"] <= 0:
            continue
        rr = math.log(r["exp_val"] / r["spot_t"])
        by_key[(r["asset"], r["family"])].append((r["mins_left"], r["sigma_t"], rr))

    params, diag = {}, {}
    for (asset, family), recs in by_key.items():
        n = len(recs)
        if n < 30:
            continue
        rs = [rr for _, _, rr in recs]
        mu_b = median(rs)
        # PER-HORIZON method of moments: for each distinct tau, the empirical
        # residual variance Var_k ≈ vol_mult^2 * mean(sigma_t^2)*g(tau) + sigma_b^2.
        # Regressing across horizons (n-weighted) lets the near-expiry horizons
        # anchor the intercept sigma_b^2, instead of letting high-variance
        # long-horizon points dominate a raw squared-residual OLS.
        by_h = defaultdict(list)
        for tau, sig, rr in recs:
            by_h[tau].append((sig, rr))
        Xk, Yk, Wk = [], [], []
        for tau, items in by_h.items():
            if len(items) < 5:
                continue
            varv = sum((rr - mu_b) ** 2 for _, rr in items) / len(items)
            xbar = (sum(s * s for s, _ in items) / len(items)) * g_tau(tau)
            Xk.append(xbar); Yk.append(varv); Wk.append(len(items))
        slope, intercept = wls2(Xk, Yk, Wk)
        vol_mult2 = max(slope, 1e-6)
        sigma_b2 = max(intercept, (2.0e-4) ** 2)   # floor sigma_b at 2 bps
        # global unit-variance correction: scale total variance so mean(z^2)=1
        def zvar(vm2, sb2):
            tot = 0.0
            for tau, sig, rr in recs:
                v = vm2 * (sig * sig) * g_tau(tau) + sb2
                tot += (rr - mu_b) ** 2 / v
            return tot / len(recs)
        c = zvar(vol_mult2, sigma_b2)
        vol_mult2 *= c; sigma_b2 *= c            # now mean(z^2) == 1 by construction
        vol_mult = math.sqrt(vol_mult2)
        sigma_b = math.sqrt(sigma_b2)
        # standardized residuals -> kurtosis -> nu
        zs = []
        for tau, sig, rr in recs:
            var = vol_mult2 * (sig * sig) * g_tau(tau) + sigma_b2
            if var > 0:
                zs.append((rr - mu_b) / math.sqrt(var))
        m2 = sum(z * z for z in zs) / len(zs)
        m4 = sum(z ** 4 for z in zs) / len(zs)
        kurt_excess = m4 / (m2 * m2) - 3.0 if m2 > 0 else 0.0
        if kurt_excess > 0.05:
            nu = 4.0 + 6.0 / kurt_excess
        else:
            nu = 200.0
        nu = max(3.0, min(200.0, nu))
        params[(asset, family)] = {
            "basis_mean": round(mu_b, 8),
            "basis_std": round(sigma_b, 8),
            "vol_mult": round(vol_mult, 5),
            "nu": round(nu, 2),
            "drift_scale": 1.0,
        }
        diag[(asset, family)] = {"n": n, "mu_b_bps": round(mu_b * 1e4, 2),
                                 "sigma_b_bps": round(sigma_b * 1e4, 2),
                                 "vol_mult": round(vol_mult, 3), "nu": round(nu, 1),
                                 "z_var": round(m2, 3), "excess_kurt": round(kurt_excess, 2)}
    return params, diag


# ── model probability (raw, given stage-A params) ────────────────────────────
def model_p_raw(r, pa):
    tau = r["mins_left"]
    if tau <= 0:
        return 1.0 if (r["floor_strike"] and r["spot_t"] >= r["floor_strike"]) else 0.0
    sigma = pa["vol_mult"] * r["sigma_t"]
    var = sigma * sigma * g_tau(tau) + pa["basis_std"] ** 2
    sd = math.sqrt(max(var, 1e-12)) * pa.get("sd_mult", 1.0)
    tau_mid = max(0.0, tau - AVG_WINDOW_MIN / 2.0)
    drift = -0.5 * sigma * sigma * tau_mid * pa.get("drift_scale", 1.0)
    mean = math.log(r["spot_t"]) + pa["basis_mean"] + drift
    nu = pa["nu"]

    def p_above(K):
        if not K or K <= 0:
            return 1.0
        return _sf_unit((math.log(K) - mean) / sd, nu)

    st = (r["strike_type"] or "greater").lower()
    if st in ("greater", "greater_or_equal"):
        return p_above(r["floor_strike"])
    if st == "less":
        return 1.0 - p_above(r["cap_strike"])
    if st == "between":
        return min(max(p_above(r["floor_strike"]) - p_above(r["cap_strike"]), 0.0), 1.0)
    return 0.5


# ── Stage B: residual Platt per (asset, bucket) ──────────────────────────────
def refine_shape(train_rows, params, val_frac=0.25, band=(0.05, 0.95)):
    """Validation-gated selection of distribution shape (nu, sd_mult) per
    (asset, family), optimizing the ACTUAL band log-loss on a held-out slice of
    train — rather than trusting the moment-based nu.  This fixes assets that are
    near-Gaussian in the tradeable zone where a kurtosis-fit t over-fattens."""
    cts = sorted(set(r["close_time"] for r in train_rows))
    if len(cts) < 8:
        return params
    cut = cts[int(len(cts) * (1 - val_frac))]
    val = [r for r in train_rows if r["close_time"] >= cut]
    by = defaultdict(list)
    for r in val:
        by[(r["asset"], r["family"])].append(r)

    NU_GRID = [200.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 40.0]
    SD_GRID = [0.90, 0.95, 1.0, 1.05, 1.10, 1.20]
    for key, pa in params.items():
        rows = by.get(key, [])
        if len(rows) < 200:
            continue
        best, best_ll = None, float("inf")
        for nu in NU_GRID:
            for sdm in SD_GRID:
                cand = dict(pa); cand["nu"] = nu; cand["sd_mult"] = sdm
                pairs = []
                for r in rows:
                    p = model_p_raw(r, cand)
                    if band[0] < p < band[1]:
                        pairs.append((p, r["outcome"]))
                if len(pairs) < 50:
                    continue
                ll = logloss(pairs)
                if ll < best_ll:
                    best_ll, best = ll, (nu, sdm)
        if best:
            pa["nu"], pa["sd_mult"] = round(best[0], 2), round(best[1], 3)
    return params


def fit_platt(pairs, iters=800, lr=0.2, l2=0.02):
    """Platt (a,b) with ridge toward identity (a=0,b=1) to resist overfit."""
    if len(pairs) < 2:
        return 0.0, 1.0
    xs = [_logit(p) for p, _ in pairs]
    ys = [float(y) for _, y in pairs]
    a, b, n = 0.0, 1.0, len(xs)
    for _ in range(iters):
        ps = [_sigmoid(a + b * x) for x in xs]
        ga = sum(p - y for p, y in zip(ps, ys)) / n + l2 * a
        gb = sum((p - y) * x for p, y, x in zip(ps, ys, xs)) / n + l2 * (b - 1.0)
        a -= lr * ga; b -= lr * gb
    return a, b


def fit_stage_b(rows, params, band=(0.03, 0.97), min_n=80, min_gain=0.003):
    """Per (asset, family, bucket) regularized Platt, kept only if it improves
    train log-loss by >= min_gain (relative) — otherwise leave the physics raw."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pa = params.get((r["asset"], r["family"]))
        if not pa:
            continue
        p = model_p_raw(r, pa)
        if not (band[0] < p < band[1]):
            continue
        bl = bucket_label(r["mins_left"])
        if bl:
            by[(r["asset"], r["family"])][bl].append((p, r["outcome"]))
    out = {}
    for key, buckets in by.items():
        platt = {}
        for bl, pairs in buckets.items():
            if len(pairs) < min_n:
                continue
            a, b = fit_platt(pairs)
            ll_raw = logloss(pairs)
            ll_cal = logloss([(_sigmoid(a + b * _logit(p)), y) for p, y in pairs])
            if ll_raw - ll_cal >= min_gain * ll_raw:
                platt[bl] = {"a": round(a, 5), "b": round(b, 5), "n": len(pairs),
                             "ll_raw": round(ll_raw, 4), "ll_cal": round(ll_cal, 4)}
        if platt:
            out[key] = platt
    return out


# ── metrics ──────────────────────────────────────────────────────────────────
def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def logloss(pairs):
    s = 0.0
    for p, y in pairs:
        p = min(max(p, 1e-9), 1 - 1e-9)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(pairs) if pairs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_v3.json"))
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--with-platt", action="store_true",
                    help="also emit the Stage-B residual Platt. OFF by default: "
                         "it absorbs transient directional drift (helps DOGE in "
                         "the sample, hurts well-behaved assets out of sample). "
                         "The physics is the robust model.")
    args = ap.parse_args()

    rows = load(args.corpus)
    train, test, cut = time_split(rows, args.test_frac)
    print(f"loaded {len(rows)} rows; train={len(train)} test={len(test)} (cut at {cut})")

    params, diag = fit_stage_a(train)
    print("\n=== STAGE A (physics) per (asset, family) ===")
    print(f"{'asset':>6} {'fam':>7} {'n':>6} {'mu_b':>9} {'sig_b':>9} {'volmult':>8} {'nu':>7} {'kurt':>7}")
    for (a, fam) in sorted(diag):
        d = diag[(a, fam)]
        print(f"{a:>6} {fam:>7} {d['n']:>6} {d['mu_b_bps']:>+8.2f}b {d['sigma_b_bps']:>+8.2f}b "
              f"{d['vol_mult']:>8.3f} {d['nu']:>7.1f} {d['excess_kurt']:>7.2f}")

    params = refine_shape(train, params)
    print("\n=== shape refinement (validation-gated nu, sd_mult) ===")
    for (a, fam) in sorted(params):
        pa = params[(a, fam)]
        print(f"  {a:>5} {fam:>7}: nu={pa['nu']:>6}  sd_mult={pa.get('sd_mult', 1.0)}")

    platt = fit_stage_b(train, params) if args.with_platt else {}
    if args.with_platt:
        print("\n=== STAGE B (residual Platt) emitted buckets ===")
        for (a, fam) in sorted(platt):
            print(f"  {a} {fam}: " + ", ".join(f"{bl}(n={e['n']})" for bl, e in platt[(a, fam)].items()))
    else:
        print("\n(Stage-B Platt skipped — physics-only. Use --with-platt to add it.)")

    cal = {"_meta": {"source": "fit_v3.py", "train_cut": cut, "n_train": len(train)},
           "assets": {}}
    for (a, fam), pa in params.items():
        ent = cal["assets"].setdefault(a, {"by_family": {}})
        fam_params = dict(pa)
        if (a, fam) in platt:
            fam_params["platt"] = platt[(a, fam)]
        ent["by_family"][fam] = fam_params
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nWrote {args.out}  (assets: {sorted(cal['assets'])})")


if __name__ == "__main__":
    main()
