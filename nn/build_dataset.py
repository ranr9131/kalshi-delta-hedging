"""
Build a tensor dataset from historical KXBTC15M markets for time-series NN
training. Reads cached BTC + Kalshi data — does NOT touch any live state, the
2D table, or the running trader.

Each window becomes a (T, F) tensor where T = 15 minutes (T+0..T+14) and F = 6
features per minute, plus a binary label (1 = YES won).

Features per minute:
    0: btc_return_from_t0   — cumulative BTC return since window open
    1: btc_return_1m        — last-minute BTC return
    2: kalshi_yes           — Kalshi yes mid price at that minute
    3: kalshi_yes_change    — change in Kalshi yes since open
    4: minute_norm          — t / 14 (positional info, also visible to transformer)
    5: hour_sin / hour_cos  — time-of-day cyclical encoding (one per dim)

Output saved to nn/data/dataset.npz with:
    X         (N, 15, 7)   float32
    mask      (N, 15)      bool — True where minute had valid data
    y         (N,)         float32 — 1 if YES won
    open_ts   (N,)         int64  — window open unix ts (for time-based split)
    tickers   (N,)         str    — for traceability
"""

import os
import sys
import math
import numpy as np
from datetime import datetime

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kalshi_client
import btc_data
from config import DATA_DAYS, CACHE_DIR

WINDOW_MINUTES = 15
N_FEATURES     = 7   # see header

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_window_features(market, btc_prices):
    """Return (X: (15, 7) float32, mask: (15,) bool, label: float, open_ts: int)
    or None on missing data."""
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")
    if not open_iso or result not in ("yes", "no"):
        return None
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    label = 1.0 if result == "yes" else 0.0

    btc_t0 = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None:
        return None
    candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, close_iso)
    if not candles:
        return None
    kalshi_t0 = candles[0].get("yes_open")
    if kalshi_t0 is None or not (0.01 < kalshi_t0 < 0.99):
        return None

    X = np.zeros((WINDOW_MINUTES, N_FEATURES), dtype=np.float32)
    mask = np.zeros(WINDOW_MINUTES, dtype=bool)
    last_btc = btc_t0
    hour = open_dt.hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    for m in range(WINDOW_MINUTES):
        t = t0 + m * 60
        btc_t = btc_data.lookup(btc_prices, t)
        kalshi_yes = kalshi_client.get_yes_price_at(candles, t)
        if btc_t is None or kalshi_yes is None or not (0.01 < kalshi_yes < 0.99):
            continue
        ret_t0 = (btc_t / btc_t0) - 1.0
        ret_1m = (btc_t / last_btc) - 1.0 if last_btc else 0.0
        last_btc = btc_t
        kalshi_change = kalshi_yes - kalshi_t0
        X[m, 0] = ret_t0
        X[m, 1] = ret_1m
        X[m, 2] = kalshi_yes
        X[m, 3] = kalshi_change
        X[m, 4] = m / float(WINDOW_MINUTES - 1)
        X[m, 5] = hour_sin
        X[m, 6] = hour_cos
        mask[m] = True

    if mask.sum() < 5:   # too few valid minutes — likely data gap
        return None
    return X, mask, label, t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Fetching {DATA_DAYS} days of settled markets ...")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    if not markets:
        print("No markets found.")
        return
    print(f"  {len(markets)} markets.")

    ts_list = []
    for mk in markets:
        try:
            dt = datetime.fromisoformat(mk["open_time"].replace("Z", "+00:00"))
            ts_list.append(int(dt.timestamp()))
        except Exception:
            pass
    btc_prices = btc_data.fetch_btc_prices(min(ts_list) - 600, max(ts_list) + 1800)
    print(f"  {len(btc_prices)} BTC minute prices loaded.")

    Xs, masks, ys, tss, tickers = [], [], [], [], []
    skipped = 0
    print(f"\nExtracting features per window...")
    for i, mk in enumerate(markets):
        r = build_window_features(mk, btc_prices)
        if r is None:
            skipped += 1
            continue
        X, mask, y, t0 = r
        Xs.append(X)
        masks.append(mask)
        ys.append(y)
        tss.append(t0)
        tickers.append(mk["ticker"])
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(markets)}]  kept={len(Xs)}  skipped={skipped}")

    X    = np.stack(Xs)
    mask = np.stack(masks)
    y    = np.array(ys, dtype=np.float32)
    ts   = np.array(tss, dtype=np.int64)
    tk   = np.array(tickers)

    print(f"\nFinal dataset:")
    print(f"  X shape:   {X.shape}")
    print(f"  y mean:    {y.mean():.4f}  (overall YES win rate)")
    print(f"  ts range:  {datetime.fromtimestamp(ts.min()).date()} -> "
          f"{datetime.fromtimestamp(ts.max()).date()}")

    out_path = os.path.join(OUT_DIR, "dataset.npz")
    np.savez_compressed(out_path, X=X, mask=mask, y=y, ts=ts, tickers=tk)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
