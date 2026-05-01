"""
Fetches BTC/USD 1-minute price data from Coinbase Exchange public API.
No API key required. Returns up to 300 candles per request.
Cache stored per day as btc_cb_YYYYMMDD.json.
"""

import time
import json
import os
import requests
from datetime import datetime, timezone, timedelta

from config import CACHE_DIR

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
MAX_CANDLES = 300   # Coinbase limit per request


def fetch_btc_prices(start_ts_sec, end_ts_sec):
    """
    Returns a dict mapping minute timestamps (unix seconds, as strings) to BTC close price.
    Fetches 1-minute Coinbase OHLCV candles and caches by day.
    """
    start_day = int(start_ts_sec // 86400) * 86400
    end_day   = int(end_ts_sec // 86400) * 86400 + 86400

    prices = {}
    current = start_day
    total_days = (end_day - start_day) // 86400
    day_num = 0

    while current < end_day:
        day_num += 1
        day_str = datetime.fromtimestamp(current, tz=timezone.utc).strftime("%Y%m%d")
        cache_path = os.path.join(CACHE_DIR, f"btc_cb_{day_str}.json")

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
    """Fetch one full day of 1-minute Coinbase OHLC candles (~5 requests)."""
    prices = {}
    # 1440 minutes per day, 300 per request → 5 chunks
    chunk_sec = MAX_CANDLES * 60   # 18,000 seconds = 300 minutes

    day_end_ts = day_start_ts + 86400
    chunk_start = day_start_ts

    while chunk_start < day_end_ts:
        chunk_end = min(chunk_start + chunk_sec, day_end_ts)
        start_iso = datetime.fromtimestamp(chunk_start, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        end_iso   = datetime.fromtimestamp(chunk_end,   tz=timezone.utc).isoformat().replace("+00:00", "Z")

        params = {"granularity": 60, "start": start_iso, "end": end_iso}

        for attempt in range(4):
            try:
                resp = requests.get(COINBASE_URL, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"  Coinbase rate limit, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                candles = resp.json()
                break
            except Exception as e:
                print(f"  Coinbase error (attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))
                continue
        else:
            chunk_start += chunk_sec
            continue

        # Format: [time, low, high, open, close, volume]
        for row in candles:
            ts = int(row[0])
            close_price = float(row[4])
            if day_start_ts <= ts < day_end_ts:
                prices[str(ts)] = close_price

        chunk_start += chunk_sec
        time.sleep(0.15)  # polite rate limiting

    if prices:
        print(f"  [{day_num}/{total_days}] {day_str}: {len(prices)} prices")
    return prices


def lookup(prices, ts_sec):
    """
    Look up BTC price at a specific unix second timestamp.
    Tries the exact minute boundary, then scans ±2 minutes.
    """
    minute_ts = (ts_sec // 60) * 60
    for offset in [0, 60, -60, 120, -120]:
        key = str(minute_ts + offset)
        if key in prices:
            return prices[key]
    return None
