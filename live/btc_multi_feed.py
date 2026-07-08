"""
Multi-source BTC feed that takes the freshest price across Coinbase + Binance perp,
plus rolling velocity (% move over last N seconds).

Used by the latency-arb mode. The classic edge: BTC just moved on Binance perp,
Kalshi's order book hasn't repriced yet — fire a bet at stale Kalshi prices.

The velocity buffer keeps prices for the last LOOKBACK_SECS and computes
per-second drift over that window.
"""

import threading
import time
from collections import deque

import btc_feed
import binance_feed

LOOKBACK_SECS = 5.0    # default velocity window (5s rolling)

_lock = threading.Lock()
_buffer: deque = deque(maxlen=200)   # (timestamp, price) pairs


def start():
    """Start both feeds + a thread that samples the freshest price every 250ms."""
    btc_feed.start()
    binance_feed.start()
    t = threading.Thread(target=_sample_loop, daemon=True, name="multi-feed-sampler")
    t.start()
    return t


def _sample_loop():
    """Poll both feeds 4×/s, push freshest valid price into buffer."""
    while True:
        try:
            cb_price = btc_feed.get_price()
            cb_age   = btc_feed.get_price_age()
            bn_price = binance_feed.get_price()
            bn_age   = binance_feed.get_price_age()

            best_price = None
            best_age   = float("inf")
            if cb_price is not None and cb_age < best_age:
                best_price = cb_price; best_age = cb_age
            if bn_price is not None and bn_age < best_age:
                best_price = bn_price; best_age = bn_age

            if best_price is not None and best_age < 5.0:
                with _lock:
                    _buffer.append((time.time(), best_price))
        except Exception:
            pass
        time.sleep(0.25)


def get_price() -> float | None:
    """Latest combined price (freshest across feeds)."""
    with _lock:
        return _buffer[-1][1] if _buffer else None


def get_price_age() -> float:
    """Seconds since the last buffer sample."""
    with _lock:
        if not _buffer: return float("inf")
        return time.time() - _buffer[-1][0]


def get_velocity(lookback_secs: float = LOOKBACK_SECS) -> float | None:
    """
    Velocity = (price_now - price_lookback_ago) / price_lookback_ago * 100.
    Returns % move (signed). None if buffer is too short.
    """
    with _lock:
        if len(_buffer) < 3: return None
        now_ts, now_price = _buffer[-1]
        target_ts = now_ts - lookback_secs
        # find the closest sample at or before target_ts
        prior_price = None
        for ts, p in reversed(_buffer):
            if ts <= target_ts:
                prior_price = p; break
        if prior_price is None or prior_price == 0:
            return None
        return (now_price - prior_price) / prior_price * 100.0


def get_velocity_with_meta(lookback_secs: float = LOOKBACK_SECS):
    """Return (velocity_pct, current_price, lookback_actual_secs) or None."""
    with _lock:
        if len(_buffer) < 3: return None
        now_ts, now_price = _buffer[-1]
        target_ts = now_ts - lookback_secs
        for ts, p in reversed(_buffer):
            if ts <= target_ts:
                actual_lb = now_ts - ts
                if p == 0: return None
                return ((now_price - p) / p * 100.0, now_price, actual_lb)
        return None
