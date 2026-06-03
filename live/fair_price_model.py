"""Simple binary option pricing for 15-min Kalshi crypto markets.

Base model:
  For "Will price ≥ strike S at time T?", under log-normal random walk:
    z = ln(S / P) / (σ × √minutes_left)
    P_raw = 1 − Φ(z)

Calibration correction (Platt scaling):
  Per-asset (a, b) learned from historical (model_p, realized_outcome) pairs:
    P_corrected = sigmoid(a + b × logit(P_raw))

  If `calibration.json` exists alongside this file, we apply the correction
  automatically.  Otherwise we return raw model probability.
"""
import json
import math
import os


# Empirically tuned per-minute volatility (close-to-close std dev)
ASSET_VOL_PER_MIN = {
    "BTC": 0.0012,
    "ETH": 0.0015,
    "SOL": 0.0025,
    "XRP": 0.0020,
}


# ── Optional calibration correction ────────────────────────────────────────

_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "calibration.json")
_CAL: dict = {}
try:
    if os.path.exists(_CAL_PATH):
        with open(_CAL_PATH) as f:
            _CAL = json.load(f)
except Exception:
    _CAL = {}


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _apply_calibration(p_raw: float, asset: str) -> float:
    """Return P_corrected = sigmoid(a + b * logit(p_raw)) if we have a
    calibration entry for `asset`, else p_raw."""
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


def fair_p_yes_raw(current_price: float, strike: float, minutes_left: float,
                   asset: str = "BTC") -> float:
    """Uncalibrated log-normal model output (for diagnostics)."""
    if minutes_left <= 0:
        return 1.0 if current_price >= strike else 0.0
    if current_price <= 0 or strike <= 0:
        return 0.5
    sigma_per_min = ASSET_VOL_PER_MIN.get(asset.upper(), 0.0015)
    total_sigma = sigma_per_min * math.sqrt(minutes_left)
    z = math.log(strike / current_price) / total_sigma
    return 1.0 - _phi(z)


def fair_p_yes(current_price: float, strike: float, minutes_left: float,
               asset: str = "BTC") -> float:
    """P(YES wins).  Applies calibration correction if calibration.json
    exists; otherwise identical to the raw log-normal output."""
    raw = fair_p_yes_raw(current_price, strike, minutes_left, asset)
    return _apply_calibration(raw, asset)


def calibration_active() -> bool:
    return bool(_CAL)


def fair_p_no(current_price: float, strike: float, minutes_left: float,
              asset: str = "BTC") -> float:
    return 1.0 - fair_p_yes(current_price, strike, minutes_left, asset)


if __name__ == "__main__":
    # Sanity check
    # BTC at $50,000, strike $50,000, 10 min left
    print("BTC at strike, 10 min left:", fair_p_yes(50000, 50000, 10, "BTC"))
    # BTC at $49,500, strike $50,000, 10 min left (1% below)
    print("BTC 1% below strike, 10 min left:", fair_p_yes(49500, 50000, 10, "BTC"))
    # BTC at $49,500, strike $50,000, 2 min left
    print("BTC 1% below strike, 2 min left:", fair_p_yes(49500, 50000, 2, "BTC"))
    # BTC at $50,500, strike $50,000, 2 min left
    print("BTC 1% above strike, 2 min left:", fair_p_yes(50500, 50000, 2, "BTC"))
