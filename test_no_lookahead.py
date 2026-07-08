"""
Regression tests for the 60-second lookahead bug (discovered 2026-07-07).

Coinbase candles are keyed by their START timestamp; the close of the candle
keyed t is the price at t+60. The original btc_data.lookup(prices, t) returned
prices[t] — i.e. the price 60 seconds in the FUTURE of t — which leaked one
minute of future BTC movement into every backtest feature and manufactured a
phantom ~4c/contract "edge" (walk-forward "validated" it because the bug lives
in shared plumbing and travels with the methodology).

1. Synthetic tests: lookup(t) must return the close of the last COMPLETED
   candle (keyed t-60) and must never read a key >= t.
2. Offline integration test (uses data/cache if present): Kalshi settles on
   the BRTI average over [close-60, close]. The price knowable AT close must
   match settlement strictly better than the future candle's close does
   (observed ~1.7bps vs ~2.9bps median). If someone reintroduces the shift,
   this inverts and the test fails.

Run: python3 test_no_lookahead.py
"""

import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone

import btc_data
from config import CACHE_DIR


def test_lookup_returns_last_completed_candle():
    # candles keyed by START ts: close of [0,60) is "price at 60", etc.
    prices = {"0": 100.0, "60": 200.0, "120": 300.0}
    # At t=120 the last completed candle is [60,120) -> keyed 60 -> 200.
    assert btc_data.lookup(prices, 120) == 200.0, \
        f"lookup(120) leaked the future candle: {btc_data.lookup(prices, 120)}"
    # Mid-minute t=150: candle [120,180) is still open; price knowable is 200.
    assert btc_data.lookup(prices, 150) == 200.0
    # At t=60 only [0,60) has completed.
    assert btc_data.lookup(prices, 60) == 100.0


def test_lookup_gap_falls_back_to_past_only():
    prices = {"0": 100.0, "180": 400.0}
    # t=240: last completed candle [180,240) -> 400.
    assert btc_data.lookup(prices, 240) == 400.0
    # t=180: candle keyed 120 missing; must fall BACK to keyed 0 (=100),
    # never forward to keyed 180 (close at 240, i.e. the future).
    assert btc_data.lookup(prices, 180) == 100.0, \
        "gap fallback scanned forward into the future"
    # t=120: keyed-60 missing, falls back to keyed 0.
    assert btc_data.lookup(prices, 120) == 100.0


def test_settlement_pins_key_semantics():
    """Offline check against real Kalshi settlement prints (skips w/o cache)."""
    mk_path = os.path.join(CACHE_DIR, "markets_90d.json")
    if not os.path.exists(mk_path):
        print("  (skipped: no markets cache)")
        return
    markets = json.load(open(mk_path))
    prices = {}
    for f in glob.glob(os.path.join(CACHE_DIR, "btc_cb_*.json")):
        prices.update(json.load(open(f)))
    if not prices:
        print("  (skipped: no BTC cache)")
        return

    err_knowable, err_future = [], []
    for m in markets:
        ev, ci = m.get("expiration_value"), m.get("close_time")
        if not ev or not ci:
            continue
        try:
            ev = float(ev)
        except (TypeError, ValueError):
            continue
        if ev < 1000:   # corrupt records
            continue
        tc = int(datetime.fromisoformat(ci.replace("Z", "+00:00")).timestamp())
        knowable = btc_data.lookup(prices, tc)        # close of [tc-60, tc)
        future = prices.get(str((tc // 60) * 60))      # close of [tc, tc+60)
        if knowable is None or future is None:
            continue
        err_knowable.append(abs(knowable - ev) / ev * 1e4)
        err_future.append(abs(future - ev) / ev * 1e4)

    assert len(err_knowable) > 200, f"too few settlements to test ({len(err_knowable)})"
    med_k = statistics.median(err_knowable)
    med_f = statistics.median(err_future)
    print(f"  |knowable-at-close - settlement| median: {med_k:.2f} bps (n={len(err_knowable)})")
    print(f"  |future-candle     - settlement| median: {med_f:.2f} bps")
    # Settlement is the BRTI average over [close-60, close]; the price knowable
    # at close sits ~30s from its center, the future candle's close ~90s.
    # If lookup regresses to the future candle, these become equal -> fail.
    assert med_k < med_f * 0.85, (
        f"lookup(t) is NOT closer to settlement than the future candle "
        f"({med_k:.2f} vs {med_f:.2f} bps) — lookahead reintroduced?")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"{name} ...")
                fn()
                print("  PASS")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL: {e}")
    sys.exit(1 if failures else 0)
