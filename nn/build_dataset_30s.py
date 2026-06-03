"""30-second granularity dataset builder.

Uses Binance 30s crypto candles (already cached) + Kalshi 1m candles.
Same 14-feature schema as v2 but at 30-second resolution.

Sequence length: 30 steps (15 minutes × 2 per min)

Output: nn/data/dataset_30s_{ASSET}.npz

Run: ASSET=BTC python build_dataset_30s.py
"""
import os
import sys
import math
import json
import glob
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kalshi_client
from config import DATA_DAYS, CACHE_DIR

ASSET = os.environ.get("ASSET", "BTC").upper()
SERIES = f"KX{ASSET}15M"

WINDOW_MINUTES = 15
STEPS_PER_MINUTE = 2  # 30-second resolution
WINDOW_STEPS = WINDOW_MINUTES * STEPS_PER_MINUTE  # 30 steps
N_FEATURES = 14
STEP_SECS = 30

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(OUT_DIR, f"dataset_30s_{ASSET.lower()}.npz")


def load_binance_30s(start_ts: int, end_ts: int) -> dict:
    """Load all Binance 30s candles in range. Returns {ts_secs: {open, high, low, close, volume}}."""
    start_day = int(start_ts // 86400) * 86400
    end_day   = int(end_ts // 86400) * 86400 + 86400
    out = {}
    cur = start_day
    while cur < end_day:
        day_str = datetime.utcfromtimestamp(cur).strftime("%Y%m%d")
        cache_path = os.path.join(CACHE_DIR, f"binance_30s_{ASSET}_{day_str}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                day_data = json.load(f)
            for ts_str, candle in day_data.items():
                out[int(ts_str)] = candle
        cur += 86400
    return out


def lookup_30s(buckets: dict, ts: int) -> dict | None:
    """Find the 30s bucket at or before ts."""
    bucket_ts = (ts // STEP_SECS) * STEP_SECS
    for offset in range(0, 300, STEP_SECS):  # search back up to 5 min
        if (bucket_ts - offset) in buckets:
            return buckets[bucket_ts - offset]
    return None


def _candle_at(candles, target_ts):
    """Kalshi 1m candle at or after target_ts."""
    for c in candles:
        if c["ts"] >= target_ts:
            return c
    return None


def fetch_markets(days):
    import requests, time
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    min_close = int(time.time()) - days * 86400
    cursor = None
    out = []
    print(f"Fetching settled {SERIES} markets last {days}d...", flush=True)
    for page in range(1000):
        params = {"series_ticker": SERIES, "status": "settled",
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


def build_window(market, asset_buckets):
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")
    if not open_iso or result not in ("yes", "no"):
        return None
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    label = 1.0 if result == "yes" else 0.0

    # Asset price at t0 (30s granularity)
    bucket_t0 = lookup_30s(asset_buckets, t0)
    if bucket_t0 is None:
        return None
    asset_t0 = bucket_t0["close"]

    # Kalshi candles (1m)
    candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, close_iso)
    if not candles:
        return None
    c0 = _candle_at(candles, t0)
    if c0 is None:
        return None
    kalshi_t0 = c0["yes_close"]
    if not (0.01 < kalshi_t0 < 0.99):
        return None

    X = np.zeros((WINDOW_STEPS, N_FEATURES), dtype=np.float32)
    mask = np.zeros(WINDOW_STEPS, dtype=bool)
    hour = open_dt.hour
    dow  = open_dt.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin  = math.sin(2 * math.pi * dow / 7)

    asset_history = []
    last_asset = asset_t0
    abs_max  = 0.0

    for step in range(WINDOW_STEPS):
        t = t0 + step * STEP_SECS
        bucket = lookup_30s(asset_buckets, t)
        # Kalshi candle: use the 1m candle for the current minute
        cand = _candle_at(candles, t)
        if bucket is None or cand is None: continue
        asset_now = bucket["close"]
        yc = float(cand["yes_close"])
        if not (0.01 < yc < 0.99): continue
        asset_history.append(asset_now)
        ret_t0  = (asset_now / asset_t0) - 1.0
        # 1m return at 30s granularity = compare to 2 steps back
        ret_1m  = (asset_now / asset_history[-3]) - 1.0 if len(asset_history) >= 3 else 0.0
        last_asset = asset_now
        abs_max  = max(abs_max, abs(ret_t0))
        # 5m return at 30s granularity = compare to 10 steps back
        ret_5m = (asset_now / asset_history[-11]) - 1.0 if len(asset_history) >= 11 else 0.0
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
        X[step, 0]  = ret_t0
        X[step, 1]  = ret_1m
        X[step, 2]  = ret_5m
        X[step, 3]  = abs_max
        X[step, 4]  = yc
        X[step, 5]  = yc - kalshi_t0
        X[step, 6]  = intramin
        X[step, 7]  = rng_norm
        X[step, 8]  = spread
        X[step, 9]  = vol_log
        X[step, 10] = step / float(WINDOW_STEPS - 1)
        X[step, 11] = hour_sin
        X[step, 12] = hour_cos
        X[step, 13] = dow_sin
        mask[step] = True

    if mask.sum() < 10: return None  # require at least 5 minutes of 30s data
    return X, mask, label, t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    markets = fetch_markets(DATA_DAYS)
    if not markets:
        print("No markets."); return

    # Load all binance 30s buckets in range
    ts_list = []
    for mk in markets:
        try:
            dt = datetime.fromisoformat(mk["open_time"].replace("Z","+00:00"))
            ts_list.append(int(dt.timestamp()))
        except Exception:
            pass
    asset_buckets = load_binance_30s(min(ts_list) - 600, max(ts_list) + 1800)
    print(f"  {len(asset_buckets)} {ASSET} 30s buckets loaded", flush=True)

    Xs, masks, ys, tss, tickers = [], [], [], [], []
    skipped = 0
    print(f"\nExtracting {ASSET} 30s features per window…", flush=True)
    for i, mk in enumerate(markets):
        r = build_window(mk, asset_buckets)
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

    print(f"\nFinal {ASSET} 30s dataset:")
    print(f"  X shape:   {X.shape}")
    print(f"  y mean:    {y.mean():.4f}")
    print(f"  kept:      {len(Xs)}")
    print(f"  skipped:   {skipped}")
    print(f"  ts range:  {datetime.utcfromtimestamp(ts.min()).date()} -> "
          f"{datetime.utcfromtimestamp(ts.max()).date()}")

    np.savez_compressed(OUT_PATH, X=X, mask=mask, y=y, ts=ts, tickers=tk)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
