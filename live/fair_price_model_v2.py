"""
Sniper v2 fair-price model.

Two upgrades over v1 (fair_price_model.py):

  1. **Realized volatility** instead of a hardcoded per-asset σ.  Computed
     from the most recent 5-15 minutes of actual price moves, refreshed
     continuously.  Adapts to current regime — shrinks in calm periods,
     grows during volatile spikes.

  2. **Calibration loaded from calibration_v2.json** instead of v1's
     calibration.json.  v2's calibration is refit from LIVE snipe data
     (via fit_calibration_live.py), not historical reconstructions.

Otherwise the math is unchanged from v1:
    z = ln(K/P) / (σ × √minutes_left)
    raw_p = 1 − Φ(z)
    fair_p = sigmoid(a + b × logit(raw_p))   (per-asset Platt)
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from typing import Deque, Tuple, Dict, Optional


# Fallback σ if the realized-vol estimator doesn't have enough samples yet.
# Same numbers as v1 — only used at startup until the rolling window fills.
FALLBACK_SIGMA_PER_MIN = {
    "BTC":  0.0012, "ETH": 0.0015, "SOL":  0.0025, "XRP":  0.0020, "HYPE": 0.0030,
    "BNB":  0.0018, "TON": 0.0025, "DOGE": 0.0030, "ADA":  0.0020,
}

# Window over which we estimate realized vol from per-minute log returns.
REALIZED_VOL_WINDOW_MIN = float(os.environ.get("REALIZED_VOL_WINDOW_MIN", "10"))
MIN_VOL_SAMPLES         = int(os.environ.get("MIN_VOL_SAMPLES", "4"))

# Floor and ceiling on the realized σ to avoid pathological readings on
# either side (e.g. flatline → σ=0 → div-by-zero in z; or one big spike →
# σ blows up).  Multipliers around the fallback σ.
SIGMA_FLOOR_MULT = 0.25
SIGMA_CEIL_MULT  = 5.0


# ── Per-asset price history (1Hz buffer, ~15min depth) ────────────────────
_HIST_LEN = int(REALIZED_VOL_WINDOW_MIN * 60 + 30)
_price_hist: Dict[str, Deque[Tuple[float, float]]] = {
    a: deque(maxlen=_HIST_LEN) for a in FALLBACK_SIGMA_PER_MIN
}
_hist_lock = threading.Lock()


def record_price(asset: str, price: float) -> None:
    """Append a (ts, price) sample to the asset's rolling history.  Call
    this from the sniper's main tick loop (already does so for crypto WS)."""
    if price is None or price <= 0:
        return
    a = asset.upper()
    if a not in _price_hist:
        return
    with _hist_lock:
        _price_hist[a].append((time.time(), float(price)))


def realized_sigma_per_min(asset: str) -> Optional[float]:
    """Compute σ of 1-minute log returns over the last
    REALIZED_VOL_WINDOW_MIN minutes.  Returns None if not enough samples.

    Buckets the rolling price history into 1-minute bins (last price wins
    in each bin) then takes stdev of consecutive log returns."""
    a = asset.upper()
    with _hist_lock:
        snapshot = list(_price_hist.get(a, ()))
    if len(snapshot) < MIN_VOL_SAMPLES * 60:
        return None
    # Bucket by integer minute, keep last price in each bucket
    bins: Dict[int, float] = {}
    for ts, px in snapshot:
        bins[int(ts // 60)] = px
    minutes_sorted = sorted(bins.keys())
    if len(minutes_sorted) < MIN_VOL_SAMPLES:
        return None
    log_rets = []
    prev_price = bins[minutes_sorted[0]]
    for m in minutes_sorted[1:]:
        cur = bins[m]
        if prev_price > 0 and cur > 0:
            log_rets.append(math.log(cur / prev_price))
        prev_price = cur
    if len(log_rets) < 2:
        return None
    mean = sum(log_rets) / len(log_rets)
    var  = sum((r - mean) ** 2 for r in log_rets) / max(1, len(log_rets) - 1)
    return math.sqrt(var)


def effective_sigma_per_min(asset: str) -> float:
    """Return realized σ if available + sane, else fallback.  Always clamped
    to [floor × fallback, ceil × fallback] to avoid outliers."""
    fallback = FALLBACK_SIGMA_PER_MIN.get(asset.upper(), 0.0015)
    rv = realized_sigma_per_min(asset)
    sigma = rv if rv and rv > 0 else fallback
    sigma = max(fallback * SIGMA_FLOOR_MULT, min(fallback * SIGMA_CEIL_MULT, sigma))
    return sigma


# ── Calibration loading (calibration_v2.json) ──────────────────────────────

_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "calibration_v2.json")
_CAL: dict = {}
_CAL_MTIME = 0.0


def _maybe_reload_calibration():
    """Hot-reload calibration if the file mtime changed.  Lets us drop in
    a new calibration_v2.json (from fit_calibration_live.py) without
    restarting the sniper."""
    global _CAL, _CAL_MTIME
    try:
        if not os.path.exists(_CAL_PATH):
            _CAL = {}
            return
        m = os.path.getmtime(_CAL_PATH)
        if m == _CAL_MTIME:
            return
        with open(_CAL_PATH) as f:
            _CAL = json.load(f)
        _CAL_MTIME = m
    except Exception:
        pass


_maybe_reload_calibration()


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _apply_calibration(p_raw: float, asset: str) -> float:
    _maybe_reload_calibration()
    if not _CAL:
        return p_raw
    entry = _CAL.get(asset.upper()) or _CAL.get("_global")
    if not entry:
        return p_raw
    try:
        a = float(entry["a"]); b = float(entry["b"])
    except Exception:
        return p_raw
    return _sigmoid(a + b * _logit(p_raw))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fair_p_yes_raw_v2(current_price: float, strike: float,
                      minutes_left: float, asset: str = "BTC",
                      sigma_per_min: Optional[float] = None) -> float:
    """Uncalibrated log-normal output using realized σ (or override)."""
    if minutes_left <= 0:
        return 1.0 if current_price >= strike else 0.0
    if current_price <= 0 or strike <= 0:
        return 0.5
    sigma = sigma_per_min if sigma_per_min is not None else effective_sigma_per_min(asset)
    total_sigma = sigma * math.sqrt(minutes_left)
    z = math.log(strike / current_price) / total_sigma
    return 1.0 - _phi(z)


def fair_p_yes_v2(current_price: float, strike: float, minutes_left: float,
                  asset: str = "BTC",
                  sigma_per_min: Optional[float] = None) -> float:
    """P(YES wins), v2: realized σ + calibration_v2.json."""
    raw = fair_p_yes_raw_v2(current_price, strike, minutes_left, asset, sigma_per_min)
    return _apply_calibration(raw, asset)


# ── TOUCH / first-passage model (for "reach $X" / "dip to $X" markets) ──────
#
# fair_p_yes_v2 above is a TERMINAL model: P(price ≥ strike AT SETTLEMENT).
# That is correct for Up/Down and hourly-settlement markets (what the sniper
# trades), and MUST stay as-is.
#
# Touch markets ("Will BTC *reach* $80k in June?", "Will SOL *dip to* $40?")
# pay if the barrier is hit AT ANY POINT before expiry. The probability of
# *touching* a level is ~2× the probability of *finishing* beyond it, so a
# terminal model materially mis-prices them. These functions compute the
# one-touch (first-passage) probability under GBM, reusing the SAME realized-σ
# machinery as the terminal model.
#
# Direction is inferred from the barrier vs spot: barrier above → up-touch,
# barrier below → down-touch.
#
# Driftless closed form (drift_per_min=0):  P_touch = 2·Φ(-|ln(H/S)| / (σ√T))
# General form keeps a per-minute log-drift term (default 0 — over short
# crypto horizons drift is negligible and adds estimation noise).

def p_touch_v2(current_price: float, barrier: float, minutes_left: float,
               asset: str = "BTC", sigma_per_min: Optional[float] = None,
               drift_per_min: float = 0.0) -> float:
    """P(price touches `barrier` at any time within `minutes_left`).

    First-passage probability for arithmetic BM in log-price with per-minute
    vol σ and drift ν. Use for TOUCH markets, NOT settlement markets.
    Returns a value in [0, 1]."""
    if current_price <= 0 or barrier <= 0:
        return 0.5
    if minutes_left <= 0:
        # No time left: "touched" iff spot is already at/through the barrier.
        if barrier >= current_price:
            return 1.0 if current_price >= barrier else 0.0
        return 1.0 if current_price <= barrier else 0.0

    sigma = sigma_per_min if sigma_per_min is not None else effective_sigma_per_min(asset)
    if sigma <= 0:
        return 0.0
    T = float(minutes_left)
    total_sigma = sigma * math.sqrt(T)
    b = math.log(barrier / current_price)        # log-distance to barrier
    nu = float(drift_per_min)

    def _safe_exp_phi(expo: float, phi_arg: float) -> float:
        # exp(expo)·Φ(phi_arg) with overflow guard; the Φ factor → 0 for very
        # negative args, so capping the exponent never inflates a real value.
        try:
            return math.exp(min(expo, 50.0)) * _phi(phi_arg)
        except OverflowError:
            return 0.0

    if b >= 0:  # up-barrier
        p = (_phi((nu * T - b) / total_sigma)
             + _safe_exp_phi(2.0 * nu * b / (sigma * sigma), (-nu * T - b) / total_sigma))
    else:       # down-barrier
        p = (_phi((b - nu * T) / total_sigma)
             + _safe_exp_phi(2.0 * nu * b / (sigma * sigma), (b + nu * T) / total_sigma))

    return min(1.0, max(0.0, p))


def fair_p_no_touch_v2(current_price: float, barrier: float, minutes_left: float,
                       asset: str = "BTC", sigma_per_min: Optional[float] = None,
                       drift_per_min: float = 0.0) -> float:
    """Model 'No' value for a touch market = 1 − P(touch).

    This is the number to compare against the market's 'No' ask. Edge =
    fair_p_no_touch_v2(...) − market_no_ask. Uncalibrated by design: Phase-0
    shadow logging measures raw edge before any Platt fit. (A touch-specific
    calibration can be layered on later — do NOT reuse calibration_v2.json,
    which is fit to the terminal model.)"""
    return 1.0 - p_touch_v2(current_price, barrier, minutes_left, asset,
                            sigma_per_min, drift_per_min)


def touch_moneyness_z(current_price: float, barrier: float, minutes_left: float,
                      asset: str = "BTC",
                      sigma_per_min: Optional[float] = None) -> float:
    """Vol-units from spot to the barrier: |ln(barrier/spot)| / (σ√T).

    The far-OTM discipline gate: only quote the 'No' when z is large enough
    (e.g. ≥ 1.5–2) that the touch is near-impossible and σ-misestimation
    can't flip the trade. Returns +inf if no time left."""
    if current_price <= 0 or barrier <= 0:
        return 0.0
    if minutes_left <= 0:
        return float("inf")
    sigma = sigma_per_min if sigma_per_min is not None else effective_sigma_per_min(asset)
    if sigma <= 0:
        return float("inf")
    return abs(math.log(barrier / current_price)) / (sigma * math.sqrt(minutes_left))


def implied_sigma_from_touch(p_touch_target: float, current_price: float,
                             barrier: float, minutes_left: float,
                             drift_per_min: float = 0.0,
                             lo: float = 1e-6, hi: float = 0.05,
                             iters: int = 60) -> Optional[float]:
    """Invert p_touch_v2 for σ-per-minute: find σ s.t. the model's touch
    probability equals `p_touch_target` (the market's implied touch prob, =
    market 'Yes' mid). p_touch is monotincreasing in σ, so bisection is safe.

    Use this to read the market's OWN volatility off a liquid near-the-money
    strike, then price other strikes of the same expiry with it. Returns None
    if the target is unreachable within [lo, hi] (e.g. already-touched, or a
    target so extreme no σ in range fits)."""
    if not (0.0 < p_touch_target < 1.0):
        return None
    if current_price <= 0 or barrier <= 0 or minutes_left <= 0:
        return None

    def f(sig):
        return p_touch_v2(current_price, barrier, minutes_left, sigma_per_min=sig,
                          drift_per_min=drift_per_min)

    flo, fhi = f(lo), f(hi)
    if not (min(flo, fhi) <= p_touch_target <= max(flo, fhi)):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < p_touch_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibration_active() -> bool:
    return bool(_CAL)


if __name__ == "__main__":
    # quick smoke
    record_price("BTC", 67000)
    record_price("BTC", 67050)
    record_price("BTC", 66980)
    print("BTC effective σ:", effective_sigma_per_min("BTC"))
    print("BTC fair @ ATM 10min:", fair_p_yes_v2(67000, 67000, 10, "BTC"))
    print("BTC fair @ -0.1% 10min:", fair_p_yes_v2(67000, 67067, 10, "BTC"))
    print("calibration active:", calibration_active())

    # ── touch vs terminal: the ~2x gap ──
    S, K, mins, sig = 65000, 80000, 23 * 24 * 60, 0.0012  # BTC $65k→$80k, ~23d, σ_min
    term_yes = fair_p_yes_raw_v2(S, K, mins, "BTC", sigma_per_min=sig)
    touch_yes = p_touch_v2(S, K, mins, "BTC", sigma_per_min=sig)
    print(f"\nBTC $65k→$80k in 23d (σ/min={sig}):")
    print(f"  terminal P(yes/finish≥80k): {term_yes:.4f}")
    print(f"  touch    P(yes/ever hit  ): {touch_yes:.4f}   (~{touch_yes/max(term_yes,1e-9):.1f}x)")
    print(f"  model 'No' value (touch):   {fair_p_no_touch_v2(S,K,mins,'BTC',sigma_per_min=sig):.4f}")
    print(f"  moneyness z:                {touch_moneyness_z(S,K,mins,'BTC',sigma_per_min=sig):.2f} vol-units")
    # sanity: driftless touch == 2·Φ(-z)
    z = touch_moneyness_z(S, K, mins, "BTC", sigma_per_min=sig)
    print(f"  check 2·Φ(-z):              {2*_phi(-z):.4f}  (should match touch)")
