"""
Daily/hourly horizon-bucketed calibration for KX{coin}D markets.

STANDALONE module — it imports the production model read-only
(fair_price_model_v2.fair_p_yes_raw_v2) and does NOT modify it, so the live
sniper that depends on fair_price_model_v2 is completely unaffected.

Produced by calibrate_daily.py → calibration_daily.json.
Schema: {asset: {bucket_label: {a, b, ...}}}.  Buckets MUST match
calibrate_daily.BUCKETS. Used for daily/hourly markets whose horizons are
outside the 15M calibration's design band.

  fair_p_yes_daily(spot, strike, minutes_left, asset) -> calibrated P(YES)
"""
from __future__ import annotations
import json
import math
import os
from typing import Optional

from fair_price_model_v2 import fair_p_yes_raw_v2

_BUCKETS = [(0, 12), (12, 25), (25, 50), (50, 100), (100, 1e9)]
_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "calibration_daily.json")
_CAL: dict = {}
_CAL_MTIME = 0.0


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _bucket_label(mins: float):
    for lo, hi in _BUCKETS:
        if lo <= mins < hi:
            return f"{lo}-{int(hi) if hi < 1e8 else 'inf'}m"
    return None


def _maybe_reload():
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


def fair_p_yes_daily(current_price: float, strike: float, minutes_left: float,
                     asset: str = "XRP",
                     sigma_per_min: Optional[float] = None) -> float:
    """P(YES wins) for daily/hourly markets: realized σ + horizon-bucketed
    Platt. Falls back to the raw model if no daily calibration exists for this
    (asset, bucket)."""
    raw = fair_p_yes_raw_v2(current_price, strike, minutes_left, asset, sigma_per_min)
    _maybe_reload()
    entry = _CAL.get(asset.upper())
    if not entry:
        return raw
    cal = entry.get(_bucket_label(minutes_left))
    if not cal:
        return raw
    try:
        return _sigmoid(float(cal["a"]) + float(cal["b"]) * _logit(raw))
    except Exception:
        return raw


def daily_calibration_active() -> bool:
    _maybe_reload()
    return bool(_CAL)
