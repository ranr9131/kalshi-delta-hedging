"""
fair_price_model_v3 — pin-point fair value for Kalshi crypto markets.

Supersedes fair_price_model_v2.  Same spirit (log-normal binary option +
calibration) but fixes the structural errors that v2 had at the place you
actually trade — near expiry.

WHAT KALSHI ACTUALLY SETTLES ON (verified against the live API, 2026-06-08):
  EVERY crypto market — 15-min up/down AND hourly threshold/range — settles on
  the SAME engine: the 60-second SIMPLE AVERAGE of the CF Benchmarks Real-Time
  Index (BRTI for BTC, per-asset RTI otherwise) over the final minute before
  close.  `expiration_value` in the API is that realized 60s-average.

  The 15-min up/down market is NOT special: it carries a `floor_strike` equal to
  the PRIOR window's settle (the locked start-of-window 60s-avg), so it reduces
  to the same shape as a threshold market:  settle  {>= , > , < , between}  K.

FIVE THINGS v3 GETS RIGHT THAT v2 GOT WRONG
  1. Terminal object is a 60-SECOND AVERAGE, not a spot snapshot.  Under a
     diffusion with per-minute vol sigma, the variance of the average log-price
     Ybar over the final 1-min window is, in closed form:
         tau >= 1 min :  Var[Ybar] = sigma^2 * (tau - 2/3)
         tau <  1 min :  Var[Ybar] = sigma^2 * tau^3 / 3      (-> sigma^2/3 at 1)
     i.e. averaging shaves 2/3 of a minute off the effective horizon.  v2 used
     sigma^2 * tau and so was overconfident near expiry.

  2. SETTLEMENT INDEX != Coinbase spot.  We price off Coinbase but settlement is
     multi-venue BRTI.  We add (a) a small mean basis mu_b and (b) an additive
     index-basis variance sigma_b^2 (measured ~11 bps).  sigma_b^2 is the
     residual uncertainty at tau->0 (Coinbase != settlement index even at the
     last second) — it stops the model collapsing to 0/1 too fast.

  3. FAT TAILS.  Crypto log-returns are leptokurtic; Gaussian underprices the
     wings.  v3 uses a standardized Student-t (unit-variance) with calibrated
     degrees of freedom nu.  nu -> inf recovers the Gaussian.

  4. MARTINGALE DRIFT.  Includes the -sigma^2/2 log-drift over the (effective)
     horizon.  This nudges ATM "up" probabilities slightly below 0.5 and fixes
     the long-horizon YES over-prediction v2 showed empirically.

  5. CORRECT PRODUCT STRUCTURE / STRIKE TYPES.  Handles greater,
     greater_or_equal, less, between via floor_strike / cap_strike.

All of it is closed form and dependency-free (no numpy/scipy) so it runs inside
the live trader.  Parameters live in calibration_v3.json (hot-reloaded); sane
defaults let the model work before any calibration is fit.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

# ── shared realized-vol machinery: reuse v2's 1Hz history + estimator ────────
try:
    from fair_price_model_v2 import (
        record_price, effective_sigma_per_min, realized_sigma_per_min,
        FALLBACK_SIGMA_PER_MIN,
    )
except Exception:  # pragma: no cover - allow standalone import
    FALLBACK_SIGMA_PER_MIN = {
        "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0025, "XRP": 0.0020,
        "HYPE": 0.0030, "BNB": 0.0018, "TON": 0.0025, "DOGE": 0.0030,
        "ADA": 0.0020, "LTC": 0.0022,
    }
    _hist: Dict[str, Deque[Tuple[float, float]]] = {}
    _lock = threading.Lock()

    def record_price(asset: str, price: float) -> None:
        if not price or price <= 0:
            return
        a = asset.upper()
        with _lock:
            self_hist = _hist.setdefault(a, deque(maxlen=900))
            self_hist.append((time.time(), float(price)))

    def realized_sigma_per_min(asset: str):  # minimal fallback
        return None

    def effective_sigma_per_min(asset: str) -> float:
        return FALLBACK_SIGMA_PER_MIN.get(asset.upper(), 0.0015)


# Length of the settlement averaging window, in minutes (Kalshi = 60 seconds).
AVG_WINDOW_MIN = 1.0

# Access v2's 1Hz price history so we can compute the running partial average of
# the settlement window live (for the tau<1 sharpening).
try:
    from fair_price_model_v2 import _price_hist as _V2_HIST, _hist_lock as _V2_LOCK
except Exception:
    _V2_HIST, _V2_LOCK = None, None


def trailing_avg(asset: str, elapsed_sec: float, now: Optional[float] = None) -> Optional[float]:
    """Mean recorded price over the last `elapsed_sec` seconds (the elapsed part
    of the current settlement window).  Returns None if not enough samples.
    Requires record_price() to have been fed live ticks."""
    if _V2_HIST is None or elapsed_sec <= 0:
        return None
    a = asset.upper()
    now = now if now is not None else time.time()
    cutoff = now - elapsed_sec
    try:
        with _V2_LOCK:
            pts = [px for ts, px in _V2_HIST.get(a, ()) if ts >= cutoff]
    except Exception:
        return None
    if len(pts) < 3:
        return None
    return sum(pts) / len(pts)

# ── default parameters (overridden per-asset by calibration_v3.json) ─────────
# Measured basis = log(BRTI 60s-avg settle / Coinbase spot@close).  Defaults
# from a 7-day measurement; calibration refines per asset.
DEFAULT_BASIS_MEAN = {        # additive log shift (bps/1e4)
    "BTC": 0.00026, "ETH": 0.00026, "SOL": 0.00040, "XRP": 0.00050,
    "DOGE": 0.00050,
}
DEFAULT_BASIS_STD = {         # log-std of index basis (≈ residual at tau->0)
    "BTC": 0.00115, "ETH": 0.00120, "SOL": 0.00150, "XRP": 0.00123,
    "DOGE": 0.00150,
}
DEFAULT_NU = 8.0              # Student-t dof (fat tails); high = ~Gaussian
DEFAULT_DRIFT_SCALE = 1.0     # multiplier on the -sigma^2/2 martingale drift
DEFAULT_VOL_MULT = 1.0        # global per-asset vol scaler from calibration


# ════════════════════════════════════════════════════════════════════════════
#  Special functions (dependency-free)
# ════════════════════════════════════════════════════════════════════════════
def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, nu: float) -> float:
    """CDF of a Student-t with nu degrees of freedom."""
    if nu >= 250:                      # numerically Gaussian
        return _phi(t)
    x = nu / (nu + t * t)
    ib = 0.5 * _betai(nu / 2.0, 0.5, x)
    return 1.0 - ib if t > 0 else ib


def _sf_unit(z: float, nu: float) -> float:
    """P(U > z) where U is a UNIT-VARIANCE standardized Student-t (nu dof).
    nu -> inf gives the Gaussian survival function 1-Phi(z)."""
    if nu is None or nu >= 250 or nu <= 2.0:
        return 1.0 - _phi(z)           # guard: t-var undefined for nu<=2
    scale = math.sqrt(nu / (nu - 2.0))  # so that Var[U]=1
    return 1.0 - _student_t_cdf(z * scale, nu)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ════════════════════════════════════════════════════════════════════════════
#  Calibration (calibration_v3.json), hot-reloaded
# ════════════════════════════════════════════════════════════════════════════
_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "calibration_v3.json")
_CAL: dict = {}
_CAL_MTIME = 0.0
_cal_lock = threading.Lock()


def _maybe_reload():
    global _CAL, _CAL_MTIME
    try:
        if not os.path.exists(_CAL_PATH):
            return
        m = os.path.getmtime(_CAL_PATH)
        if m == _CAL_MTIME:
            return
        with open(_CAL_PATH) as f:
            data = json.load(f)
        with _cal_lock:
            _CAL = data
            _CAL_MTIME = m
    except Exception:
        pass


_maybe_reload()


def family_of(strike_type: str) -> str:
    """Kalshi 15M up/down markets use strike_type 'greater_or_equal'; the hourly
    threshold/range markets use 'greater'/'less'/'between'.  Clean discriminator."""
    return "15m" if (strike_type or "").lower() == "greater_or_equal" else "hourly"


def _params(asset: str, family: str = "hourly") -> dict:
    """Resolved parameter bundle: defaults <- _global <- asset <- asset.by_family.

    Calibration is family-specific because 15M up/down and hourly threshold
    markets have different structure; pooling them mis-calibrates both."""
    a = asset.upper()
    _maybe_reload()
    with _cal_lock:
        asset_entry = dict((_CAL.get("assets", {}) or {}).get(a, {}))
        glob = dict(_CAL.get("_global", {}) or {})
    fam_entry = dict((asset_entry.pop("by_family", {}) or {}).get(family, {}))
    p = {
        "basis_mean": DEFAULT_BASIS_MEAN.get(a, 0.0003),
        "basis_std": DEFAULT_BASIS_STD.get(a, 0.0013),
        "nu": DEFAULT_NU,
        "drift_scale": DEFAULT_DRIFT_SCALE,
        "vol_mult": DEFAULT_VOL_MULT,
        "platt": None,            # optional residual Platt by horizon bucket
        "seasonal": None,         # optional hour-of-week vol multiplier list[168]
        "ts_halflife_min": None,  # vol mean-reversion half-life (term structure)
        "ts_long_mult": 1.0,      # long-horizon vol multiplier target
    }
    p.update(glob)
    asset_entry.pop("platt", None)  # asset-level platt ignored; use family-level
    p.update(asset_entry)
    p.update(fam_entry)
    return p


# ════════════════════════════════════════════════════════════════════════════
#  Volatility forecast helpers
# ════════════════════════════════════════════════════════════════════════════
def _seasonal_mult(p: dict, ts_epoch: Optional[float]) -> float:
    """Hour-of-week multiplicative vol factor, if calibrated."""
    seas = p.get("seasonal")
    if not seas or ts_epoch is None:
        return 1.0
    try:
        # hour-of-week index 0..167, UTC.  (epoch 0 = Thursday.)
        how = int((ts_epoch // 3600) % 168)
        return float(seas[how])
    except Exception:
        return 1.0


def _term_structure_factor(p: dict, tau_min: float) -> float:
    """Mean-reversion term structure: scales the per-minute sigma so that the
    INTEGRATED variance over a long horizon does not just grow like tau (which
    over-states long-horizon vol because vol mean-reverts).

    We model instantaneous variance reverting from sigma^2 toward
    (ts_long_mult^2 * sigma^2) with half-life h.  The average variance multiplier
    over [0, tau] has closed form; we return its sqrt (a multiplier on sigma)."""
    h = p.get("ts_halflife_min")
    long_mult = float(p.get("ts_long_mult", 1.0))
    if not h or h <= 0 or tau_min <= 0 or abs(long_mult - 1.0) < 1e-6:
        return 1.0
    k = math.log(2.0) / float(h)           # reversion rate
    L = long_mult * long_mult              # long-run variance multiplier
    # avg over [0,tau] of  L + (1-L)*exp(-k t)  =  L + (1-L)*(1-exp(-k tau))/(k tau)
    avg_var_mult = L + (1.0 - L) * (1.0 - math.exp(-k * tau_min)) / (k * tau_min)
    return math.sqrt(max(avg_var_mult, 1e-9))


def _ybar_variance(sigma_per_min: float, tau_min: float) -> float:
    """Variance of  Ybar - X(t)  where Ybar is the average log-price over the
    final w-min settlement window and X(t) is the current spot, under a driftless
    diffusion with per-min vol sigma.  Closed form (window length w):

        tau >= w :  sigma^2 (tau - 2w/3)                 [window entirely future]
        tau <  w :  sigma^2 (tau^3 + (w-tau)^3) / (3 w^2) [t inside the window]

    For tau < w, t sits INSIDE the averaging window: the future part [t, T] and
    the already-elapsed part [T-w, t] each diffuse away from X(t) independently,
    contributing tau^3/3 and (w-tau)^3/3 respectively.  The elapsed term is the
    endpoint-vs-trailing-average noise; at tau->0 it gives a floor of sigma^2 w/3
    (~0.577 sigma), matching the measured ~6-8 bps last-minute residual.  Both
    branches agree at tau=w (sigma^2 w/3)."""
    s2 = sigma_per_min * sigma_per_min
    w = AVG_WINDOW_MIN
    if tau_min >= w:
        return s2 * (tau_min - 2.0 * w / 3.0)
    return s2 * (tau_min ** 3 + (w - tau_min) ** 3) / (3.0 * w * w)


# ════════════════════════════════════════════════════════════════════════════
#  Core: terminal settlement distribution (log space)
# ════════════════════════════════════════════════════════════════════════════
def terminal_log_moments(spot: float, mins_left: float, asset: str,
                         sigma_per_min: Optional[float] = None,
                         ts_epoch: Optional[float] = None,
                         family: str = "hourly",
                         partial_avg: Optional[float] = None
                         ) -> Tuple[float, float, float]:
    """Return (mean, sd, nu) of ln(settlement 60s-avg index) given current spot.

    partial_avg : if provided AND tau < AVG_WINDOW_MIN, this is the running
      average price over the ALREADY-ELAPSED part of the settlement window
      [T-w, t].  Using it both re-centers the mean to (1-tau)*ln(avg)+tau*ln(spot)
      and removes the elapsed-window variance term, leaving only sigma^2*tau^3/3.
      (Without it, the model assumes the elapsed part equals current spot.)
    """
    p = _params(asset, family)
    tau = max(0.0, float(mins_left))
    if sigma_per_min is None:
        sigma_per_min = effective_sigma_per_min(asset)
    sigma = sigma_per_min * float(p["vol_mult"])
    sigma *= _seasonal_mult(p, ts_epoch)
    sigma *= _term_structure_factor(p, tau)

    w = AVG_WINDOW_MIN
    use_partial = partial_avg is not None and partial_avg > 0 and tau < w
    if use_partial:
        # only the FUTURE part [t, T] (duration tau) is still random
        var_diff = sigma * sigma * (tau ** 3) / (3.0 * w * w)
        frac_obs = max(0.0, min(1.0, (w - tau) / w))
        base_log = frac_obs * math.log(partial_avg) + (1.0 - frac_obs) * math.log(max(spot, 1e-12))
        tau_mid = tau / 2.0                       # drift over the future half only
    else:
        var_diff = _ybar_variance(sigma, tau)
        base_log = math.log(max(spot, 1e-12))
        tau_mid = max(0.0, tau - w / 2.0)

    var_basis = float(p["basis_std"]) ** 2        # index basis (feed != settle idx)
    sd = math.sqrt(max(var_diff + var_basis, 1e-12)) * float(p.get("sd_mult", 1.0))
    drift = -0.5 * sigma * sigma * tau_mid * float(p["drift_scale"])
    mean = base_log + float(p["basis_mean"]) + drift
    return mean, sd, float(p["nu"])


def _p_above(mean: float, sd: float, nu: float, strike: float) -> float:
    """P(settlement > strike) with standardized Student-t innovations."""
    if strike <= 0:
        return 1.0
    z = (math.log(strike) - mean) / sd
    return _sf_unit(z, nu)


def _apply_platt(p_raw: float, p: dict, mins_left: float) -> float:
    """Optional residual Platt correction by horizon bucket."""
    platt = p.get("platt")
    if not platt:
        return p_raw
    BUCKETS = [(0, 12), (12, 25), (25, 50), (50, 100), (100, 1e9)]
    label = None
    for lo, hi in BUCKETS:
        if lo <= mins_left < hi:
            label = f"{lo}-{int(hi) if hi < 1e8 else 'inf'}m"
            break
    ent = platt.get(label) or platt.get("_all")
    if not ent:
        return p_raw
    try:
        return _sigmoid(float(ent["a"]) + float(ent["b"]) * _logit(p_raw))
    except Exception:
        return p_raw


# ════════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════════
def fair_p(spot: float, mins_left: float, asset: str = "BTC",
           floor_strike: Optional[float] = None,
           cap_strike: Optional[float] = None,
           strike_type: str = "greater",
           sigma_per_min: Optional[float] = None,
           ts_epoch: Optional[float] = None,
           calibrated: bool = True,
           partial_avg: Optional[float] = None,
           use_partial_history: bool = False) -> float:
    """P(YES) for a Kalshi crypto market.

    strike_type:
      "greater" / "greater_or_equal"  -> YES if settle (>=) floor_strike
      "less"                          -> YES if settle <  cap_strike
      "between"                       -> YES if floor_strike <= settle <= cap_strike

    For 15-min up/down markets pass floor_strike = the market's floor_strike
    (Kalshi already sets it to the locked start-of-window settle).

    Last-minute (tau < 1m) sharpening:
      partial_avg          : running average price over the elapsed part of the
                             settlement window [T-60s, t].  Pass it explicitly
                             (backtest) to use the partial-average model.
      use_partial_history  : if True and partial_avg is None, compute it live from
                             the recorded price history (call record_price()).
    """
    if mins_left <= 0:
        # already settled-ish: collapse to the indicator on current spot
        s = spot
        st = (strike_type or "greater").lower()
        if st in ("greater", "greater_or_equal"):
            return 1.0 if floor_strike is not None and s >= floor_strike else 0.0
        if st == "less":
            return 1.0 if cap_strike is not None and s < cap_strike else 0.0
        if st == "between":
            return 1.0 if (floor_strike is not None and cap_strike is not None
                           and floor_strike <= s <= cap_strike) else 0.0
        return 0.5
    if spot is None or spot <= 0:
        return 0.5

    st = (strike_type or "greater").lower()
    fam = family_of(st)
    if partial_avg is None and use_partial_history and mins_left < AVG_WINDOW_MIN:
        elapsed_sec = max(0.0, (AVG_WINDOW_MIN - mins_left)) * 60.0
        partial_avg = trailing_avg(asset, elapsed_sec)
    mean, sd, nu = terminal_log_moments(spot, mins_left, asset, sigma_per_min,
                                        ts_epoch, family=fam, partial_avg=partial_avg)

    if st in ("greater", "greater_or_equal"):
        raw = _p_above(mean, sd, nu, floor_strike) if floor_strike else 1.0
    elif st == "less":
        raw = 1.0 - _p_above(mean, sd, nu, cap_strike) if cap_strike else 0.0
    elif st == "between":
        if floor_strike and cap_strike:
            raw = _p_above(mean, sd, nu, floor_strike) - _p_above(mean, sd, nu, cap_strike)
        else:
            raw = 0.5
        raw = min(max(raw, 0.0), 1.0)
    else:
        raw = 0.5

    if not calibrated:
        return raw
    return _apply_platt(raw, _params(asset, fam), mins_left)


def fair_p_above(spot: float, strike: float, mins_left: float, asset: str = "BTC",
                 sigma_per_min: Optional[float] = None,
                 ts_epoch: Optional[float] = None,
                 calibrated: bool = True) -> float:
    """Convenience: P(settle >= strike).  Drop-in for v2's fair_p_yes."""
    return fair_p(spot, mins_left, asset, floor_strike=strike,
                  strike_type="greater_or_equal", sigma_per_min=sigma_per_min,
                  ts_epoch=ts_epoch, calibrated=calibrated)


def calibration_active() -> bool:
    return bool(_CAL)


if __name__ == "__main__":
    # smoke test: BTC near expiry vs far
    for ml in (0.25, 0.5, 1, 2, 5, 10, 30, 55):
        p_atm = fair_p_above(63000, 63000, ml, "BTC", calibrated=False)
        p_up = fair_p_above(63000, 63010, ml, "BTC", calibrated=False)
        m, sd, nu = terminal_log_moments(63000, ml, "BTC")
        print(f"  ml={ml:>5}m  sd={sd*1e4:7.1f}bps  P(>=ATM)={p_atm:.4f}  P(>=+$10)={p_up:.4f}")
    print("between test:", fair_p(63000, 30, "BTC", floor_strike=62900,
                                  cap_strike=63100, strike_type="between", calibrated=False))
    print("less test:", fair_p(63000, 30, "BTC", cap_strike=63100,
                               strike_type="less", calibrated=False))
