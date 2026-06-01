"""
ETH/USD 1-minute price data from Coinbase. Mirrors btc_data.py exactly,
just hits the ETH-USD product instead of BTC-USD.

Cache stored per day as eth_cb_YYYYMMDD.json.
"""

import time
import json
import os
import requests
from datetime import datetime, timezone, timedelta

from config import CACHE_DIR

COINBASE_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
MAX_CANDLES = 300


def fetch_eth_prices(start_ts_sec, end_ts_sec):
    """Returns a dict mapping minute timestamps (unix seconds, as strings)
    to ETH close price. Fetches 1-min Coinbase candles, caches by day."""
    start_day = int(start_ts_sec // 86400) * 86400
    end_day   = int(end_ts_sec // 86400) * 86400 + 86400

    prices = {}
    current = start_day
    total_days = (end_day - start_day) // 86400
    day_num = 0

    while current < end_day:
        day_num += 1
        day_str = datetime.fromtimestamp(current, tz=timezone.utc).strftime("%Y%m%d")
        cache_path = os.path.join(CACHE_DIR, f"eth_cb_{day_str}.json")

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                day_prices = json.load(f)
            if not day_prices:
                os.remove(cache_path)
                day_prices = _fetch_day(current, day_str, day_num, total_days)
                if day_prices:
                    with open(cache_path, "w") as f:
                        json.dump(day_prices, f)
        else:
            day_prices = _fetch_day(current, day_str, day_num, total_days)
            if day_prices:
                with open(cache_path, "w") as f:
                    json.dump(day_prices, f)

        prices.update(day_prices)
        current += 86400

    return prices


def _fetch_day(day_start_ts, day_str, day_num, total_days):
    """Fetch one full day of 1-min Coinbase ETH-USD candles."""
    prices = {}
    print(f"  [{day_num}/{total_days}] {day_str}: ", end="", flush=True)
    chunk_start = day_start_ts
    # Cap day_end at "now" to avoid 400 errors for chunks extending into the future.
    # Coinbase rejects ranges past current time. Without this, the trader fetching
    # today's candles aborts the whole day when the final chunk straddles `now`.
    day_end = min(day_start_ts + 86400, int(time.time()))
    if day_end <= chunk_start:
        print("0 prices (day not yet started)", flush=True)
        return prices
    while chunk_start < day_end:
        chunk_end = min(chunk_start + MAX_CANDLES * 60, day_end)
        params = {
            "granularity": 60,
            "start": datetime.fromtimestamp(chunk_start, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   datetime.fromtimestamp(chunk_end,   tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        success = False
        for attempt in range(4):
            try:
                r = requests.get(COINBASE_URL, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
                for c in data:
                    ts = int(c[0])
                    close = float(c[4])
                    prices[str(ts)] = close
                success = True
                break
            except Exception as e:
                print(f"\n  Coinbase error (attempt {attempt+1}): {e}", end="")
                time.sleep(2 ** attempt)
        if not success:
            break
        chunk_start = chunk_end
        time.sleep(0.3)
    print(f"{len(prices)} prices", flush=True)
    return prices


# Alias so callers that historically used btc_data can call the same function
# name on eth_data without conditional logic at every call site.
fetch_btc_prices = fetch_eth_prices


def lookup(prices, ts):
    """Return the most recent ETH price at or before ts. Same as btc_data.lookup."""
    for offset in range(0, 600):
        key = str(int(ts) - offset)
        if key in prices:
            return prices[key]
    return None


if __name__ == "__main__":
    now = int(time.time())
    p = fetch_eth_prices(now - 3600, now)
    print(f"loaded {len(p)} prices")
