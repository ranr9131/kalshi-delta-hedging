"""
v2 dataset builder — 14 features per timestep including order-book + volume
fields recently added to kalshi_client. Output: nn/data/dataset_v2.npz.

Features per minute (in order):
    0: btc_return_t0        — cumulative BTC return since window open
    1: btc_return_1m        — last-minute BTC return
    2: btc_return_5m        — 5-minute BTC return (zero before T+5)
    3: btc_abs_max_so_far   — max abs return from open seen so far in window
    4: kalshi_yes_close     — mid-ish yes price (candle close)
    5: kalshi_yes_from_open — yes_close - kalshi_t0
    6: kalshi_yes_intramin  — yes_close - yes_open (intra-minute momentum)
    7: kalshi_yes_range     — (yes_high - yes_low) / max(yes_close, 0.05)
                              (intra-minute volatility, normalized)
    8: kalshi_spread        — yes_ask_close - yes_bid_close (book width, capped)
    9: kalshi_volume_log    — log1p(volume) / 10 (scaled trade volume)
   10: minute_norm          — m / 14
   11: hour_sin
   12: hour_cos
   13: dow_sin              — day-of-week sin
"""

import os
import sys
import math
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kalshi_client
import btc_data
from config import DATA_DAYS, CACHE_DIR

WINDOW_MINUTES = 15
N_FEATURES     = 14

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(OUT_DIR, "dataset_v2.npz")


def _candle_at(candles, target_ts):
    """Return the candle whose end_period_ts >= target_ts (i.e. covers the
    minute ending at target_ts), or None."""
    for c in candles:
        if c["ts"] >= target_ts:
            return c
    return None


def build_window(market, btc_prices):
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
    c0 = _candle_at(candles, t0)
    if c0 is None:
        return None
    kalshi_t0 = c0["yes_close"]
    if not (0.01 < kalshi_t0 < 0.99):
        return None

    X = np.zeros((WINDOW_MINUTES, N_FEATURES), dtype=np.float32)
    mask = np.zeros(WINDOW_MINUTES, dtype=bool)

    hour = open_dt.hour
    dow  = open_dt.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin  = math.sin(2 * math.pi * dow / 7)

    btc_history = []          # for 5min return + max_so_far
    last_btc = btc_t0
    abs_max  = 0.0

    for m in range(WINDOW_MINUTES):
        t = t0 + m * 60
        btc = btc_data.lookup(btc_prices, t)
        cand = _candle_at(candles, t)
        if btc is None or cand is None:
            continue
        yes_close = cand["yes_close"]
        if not (0.01 < yes_close < 0.99):
            continue

        btc_history.append(btc)
        ret_t0  = (btc / btc_t0) - 1.0
        ret_1m  = (btc / last_btc) - 1.0 if last_btc else 0.0
        last_btc = btc
        abs_max  = max(abs_max, abs(ret_t0))
        # 5-min return: btc now vs btc 5 min ago (0 if before m=5)
        ret_5m = 0.0
        if len(btc_history) >= 6:
            ret_5m = (btc / btc_history[-6]) - 1.0

        # Kalshi features
        yc = float(yes_close)
        yo = float(cand.get("yes_open",  yc))
        yh = float(cand.get("yes_high",  yc))
        yl = float(cand.get("yes_low",   yc))
        bc = float(cand.get("yes_bid_close", yc))
        ac = float(cand.get("yes_ask_close", yc))
        vol = float(cand.get("volume", 0.0))

        intramin = yc - yo
        rng_norm = (yh - yl) / max(yc, 0.05)
        spread   = max(0.0, min(0.20, ac - bc))   # cap spread to avoid wild values
        vol_log  = math.log1p(vol) / 10.0          # scale; log(50k)/10 ≈ 1.0

        X[m, 0]  = ret_t0
        X[m, 1]  = ret_1m
        X[m, 2]  = ret_5m
        X[m, 3]  = abs_max
        X[m, 4]  = yc
        X[m, 5]  = yc - kalshi_t0
        X[m, 6]  = intramin
        X[m, 7]  = rng_norm
        X[m, 8]  = spread
        X[m, 9]  = vol_log
        X[m, 10] = m / float(WINDOW_MINUTES - 1)
        X[m, 11] = hour_sin
        X[m, 12] = hour_cos
        X[m, 13] = dow_sin
        mask[m]  = True

    if mask.sum() < 5:
        return None
    return X, mask, label, t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Fetching {DATA_DAYS} days of settled markets…", flush=True)
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    if not markets:
        print("No markets.")
        return
    print(f"  {len(markets)} markets.", flush=True)

    ts_list = []
    for mk in markets:
        try:
            dt = datetime.fromisoformat(mk["open_time"].replace("Z", "+00:00"))
            ts_list.append(int(dt.timestamp()))
        except Exception:
            pass
    btc_prices = btc_data.fetch_btc_prices(min(ts_list) - 600, max(ts_list) + 1800)
    print(f"  {len(btc_prices)} BTC minute prices loaded.", flush=True)

    Xs, masks, ys, tss, tickers = [], [], [], [], []
    skipped = 0
    print(f"\nExtracting v2 features per window (re-fetches Kalshi candles)…", flush=True)
    for i, mk in enumerate(markets):
        r = build_window(mk, btc_prices)
        if r is None:
            skipped += 1
            continue
        X, mask, y, t0 = r
        Xs.append(X); masks.append(mask); ys.append(y); tss.append(t0); tickers.append(mk["ticker"])
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(markets)}]  kept={len(Xs)}  skipped={skipped}", flush=True)

    X    = np.stack(Xs)
    mask = np.stack(masks)
    y    = np.array(ys, dtype=np.float32)
    ts   = np.array(tss, dtype=np.int64)
    tk   = np.array(tickers)

    print(f"\nFinal v2 dataset:")
    print(f"  X shape:   {X.shape}")
    print(f"  y mean:    {y.mean():.4f}")
    print(f"  ts range:  {datetime.fromtimestamp(ts.min()).date()} -> "
          f"{datetime.fromtimestamp(ts.max()).date()}")

    np.savez_compressed(OUT_PATH, X=X, mask=mask, y=y, ts=ts, tickers=tk)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
