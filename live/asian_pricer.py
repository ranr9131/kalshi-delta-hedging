"""
Asian-settlement fair value for Kalshi crypto binaries.

Settlement is the simple average of 60 once-per-second index prints in the
FINAL MINUTE of the window — an Asian digital, not a European one. In the last
~2 minutes the correct fair value depends on state the 2D table cannot see:
how much of the settlement average is already banked, and where it sits vs the
strike. This module maintains a 1-second price buffer and prices that directly.

Model: locally-arithmetic Brownian motion with per-second sigma estimated from
the trailing buffer. With a = close-60s, b = close, S_t = spot now:

  before averaging (t < a):
      mu  = S_t
      var = sigma1^2 * ((a - t) + 20)          # Var of mean of BM over [a,b]
  inside averaging (a <= t < b), r = b - t, f = (t - a)/60:
      mu  = f * banked_mean + (1 - f) * S_t
      var = sigma1^2 * r^3 / (3 * 3600)        # Var of remaining-average part

  P(settle > strike) = Phi((mu - strike) / sqrt(var))

Single-asset usage (module-level, unchanged — trader.py path):
    asian_pricer.start(get_price_fn)          # once, after the feed is live
    p = asian_pricer.p_up(strike, close_ts)   # None if not enough data

Multi-asset usage (one instance per asset — conv_maker path):
    ap = asian_pricer.AsianPricer()
    ap.start(get_price_fn)
    p = ap.p_up(strike, close_ts)
"""

import math
import threading
import time
from collections import deque

MIN_VOL_SAMPLES = 120      # need >= 2 min of returns before quoting
MIN_COVERAGE = 0.8         # required sample coverage of the banked interval
SIGMA_FLOOR = 1e-6         # $/sqrt(s); guards div-by-zero on flat feeds


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class AsianPricer:
    """1s sampler + Asian-digital pricer for one price feed."""

    def __init__(self, maxlen: int = 1800):
        self._buf: deque = deque(maxlen=maxlen)   # (ts, price) at ~1s cadence
        self._lock = threading.Lock()
        self._started = False

    def start(self, get_price_fn) -> None:
        """Start the 1s sampler thread (idempotent)."""
        if self._started:
            return
        self._started = True

        def _run():
            while True:
                try:
                    p = get_price_fn()
                    if p is not None:
                        with self._lock:
                            self._buf.append((time.time(), float(p)))
                except Exception:
                    pass
                time.sleep(1.0)

        threading.Thread(target=_run, daemon=True, name="asian-sampler").start()

    def _sigma1(self, samples) -> float | None:
        """Per-second dollar vol from up to the last 900 1s diffs."""
        diffs = []
        prev_ts, prev_p = None, None
        for ts, p in samples[-901:]:
            if prev_ts is not None:
                dt = ts - prev_ts
                if 0.5 <= dt <= 3.0:            # skip gaps from feed drops
                    diffs.append((p - prev_p) / math.sqrt(dt))
            prev_ts, prev_p = ts, p
        if len(diffs) < MIN_VOL_SAMPLES:
            return None
        m = sum(diffs) / len(diffs)
        var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
        return max(SIGMA_FLOOR, math.sqrt(var))

    def p_up(self, strike: float, close_ts: float,
             now: float | None = None) -> float | None:
        """P(settlement average > strike), or None if we can't price responsibly."""
        now = now if now is not None else time.time()
        a, b = close_ts - 60.0, close_ts
        if now >= b:
            return None
        with self._lock:
            samples = list(self._buf)
        if not samples:
            return None
        spot = samples[-1][1]
        if now - samples[-1][0] > 5.0:          # stale feed — refuse to price
            return None
        s1 = self._sigma1(samples)
        if s1 is None:
            return None

        if now < a:
            mu = spot
            var = (s1 ** 2) * ((a - now) + 20.0)
        else:
            banked = [p for ts, p in samples if a <= ts <= now]
            elapsed = now - a
            if elapsed >= 1.0 and len(banked) < MIN_COVERAGE * elapsed:
                return None                     # holes in the banked window
            f = min(1.0, elapsed / 60.0)
            banked_mean = (sum(banked) / len(banked)) if banked else spot
            r = b - now
            mu = f * banked_mean + (1.0 - f) * spot
            var = (s1 ** 2) * (r ** 3) / (3.0 * 3600.0)

        if var <= 0:
            return 1.0 if mu > strike else 0.0
        p = _phi((mu - strike) / math.sqrt(var))
        return min(0.999, max(0.001, p))

    def state(self, close_ts: float, now: float | None = None) -> dict:
        """Debug/telemetry: banked fraction + banked mean for logging."""
        now = now if now is not None else time.time()
        a = close_ts - 60.0
        with self._lock:
            banked = [p for ts, p in self._buf if a <= ts <= now]
            spot = self._buf[-1][1] if self._buf else None
        f = max(0.0, min(1.0, (now - a) / 60.0))
        return {
            "banked_frac": round(f, 3),
            "banked_mean": (sum(banked) / len(banked)) if banked else None,
            "spot": spot,
            "n_banked": len(banked),
        }


# ── module-level API (backward compatible: one default instance) ──────────────
_default = AsianPricer()


def start(get_price_fn) -> None:
    _default.start(get_price_fn)


def p_up(strike: float, close_ts: float, now: float | None = None) -> float | None:
    return _default.p_up(strike, close_ts, now)


def state(close_ts: float, now: float | None = None) -> dict:
    return _default.state(close_ts, now)
