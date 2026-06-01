"""
ETH version of build_dataset_v2.py. Same 14-feature schema, same logic —
just hits KXETH15M Kalshi markets and ETH-USD Coinbase prices.

Output: nn/data/dataset_eth.npz
"""

import os
import sys
import math
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kalshi_client
import eth_data
from config import DATA_DAYS, CACHE_DIR

# Override series for ETH
KALSHI_SERIES_ETH = "KXETH15M"

WINDOW_MINUTES = 15
N_FEATURES     = 14

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(OUT_DIR, "dataset_eth.npz")


def _candle_at(candles, target_ts):
    for c in candles:
        if c["ts"] >= target_ts:
            return c
    return None


def fetch_eth_markets(days):
    """Fetch settled KXETH15M markets, similar to kalshi_client.fetch_settled_markets
    but for ETH. Uses paginated API."""
    import requests, time
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    min_close = int(time.time()) - days * 86400
    cursor = None
    out = []
    print(f"Fetching settled KXETH15M markets last {days}d...", flush=True)
    for page in range(1000):
        params = {"series_ticker": KALSHI_SERIES_ETH, "status": "settled",
                  "min_close_ts": min_close, "limit": 1000}
        if cursor: params["cursor"] = cursor
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  error page {page}: {e}", flush=True)
            time.sleep(5)
            continue
        data = r.json()
        ms = data.get("markets", [])
        if not ms: break
        out.extend(ms)
        cursor = data.get("cursor")
        if not cursor: break
        if page % 5 == 4:
            print(f"  fetched {len(out)} so far...", flush=True)
        time.sleep(0.3)
    print(f"  total: {len(out)} settled markets", flush=True)
    return out


def build_window(market, eth_prices):
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")
    if not open_iso or result not in ("yes", "no"):
        return None
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    label = 1.0 if result == "yes" else 0.0

    eth_t0 = eth_data.lookup(eth_prices, t0)
    if eth_t0 is None:
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

    eth_history = []
    last_eth = eth_t0
    abs_max  = 0.0

    for m in range(WINDOW_MINUTES):
        t = t0 + m * 60
        eth = eth_data.lookup(eth_prices, t)
        cand = _candle_at(candles, t)
        if eth is None or cand is None: continue
        yc = float(cand["yes_close"])
        if not (0.01 < yc < 0.99): continue
        eth_history.append(eth)
        ret_t0  = (eth / eth_t0) - 1.0
        ret_1m  = (eth / last_eth) - 1.0 if last_eth else 0.0
        last_eth = eth
        abs_max  = max(abs_max, abs(ret_t0))
        ret_5m = (eth / eth_history[-6]) - 1.0 if len(eth_history) >= 6 else 0.0
        yo = float(cand.get("yes_open", yc))
        yh = float(cand.get("yes_high", yc))
        yl = float(cand.get("yes_low",  yc))
        bc = float(cand.get("yes_bid_close", yc))
        ac = float(cand.get("yes_ask_close", yc))
        vol = float(cand.get("volume", 0.0))
        intramin = yc - yo
        rng_norm = (yh - yl) / max(yc, 0.05)
        spread   = max(0.0, min(0.20, ac - bc))
        vol_log  = math.log1p(vol) / 10.0
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
        mask[m] = True

    if mask.sum() < 5: return None
    return X, mask, label, t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    markets = fetch_eth_markets(DATA_DAYS)
    if not markets:
        print("No markets."); return

    ts_list = []
    for mk in markets:
        try:
            dt = datetime.fromisoformat(mk["open_time"].replace("Z","+00:00"))
            ts_list.append(int(dt.timestamp()))
        except Exception:
            pass
    eth_prices = eth_data.fetch_eth_prices(min(ts_list) - 600, max(ts_list) + 1800)
    print(f"  {len(eth_prices)} ETH minute prices loaded", flush=True)

    Xs, masks, ys, tss, tickers = [], [], [], [], []
    skipped = 0
    print(f"\nExtracting ETH features per window…", flush=True)
    for i, mk in enumerate(markets):
        r = build_window(mk, eth_prices)
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

    print(f"\nFinal ETH dataset:")
    print(f"  X shape:   {X.shape}")
    print(f"  y mean:    {y.mean():.4f}")
    print(f"  ts range:  {datetime.fromtimestamp(ts.min()).date()} -> "
          f"{datetime.fromtimestamp(ts.max()).date()}")

    np.savez_compressed(OUT_PATH, X=X, mask=mask, y=y, ts=ts, tickers=tk)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
