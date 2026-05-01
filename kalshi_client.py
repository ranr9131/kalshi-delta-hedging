"""
Kalshi API client — fetches historical KXBTC15M markets and 1-minute candlesticks.
No authentication required for public market data.
"""

import time
import json
import os
import requests
from datetime import datetime, timezone, timedelta
from config import KALSHI_BASE_URL, KALSHI_SERIES, CACHE_DIR


def _get(path, params=None, retries=3):
    url = f"{KALSHI_BASE_URL}{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            print(f"  HTTP {resp.status_code} on {url}: {e}")
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                raise
        except requests.RequestException as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
            time.sleep(1)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def fetch_settled_markets(days=30):
    """
    Returns list of settled KXBTC15M markets from the past `days` days.
    One market per 15-min window. result="yes" means BTC closed above target (went up).
    Caches results to avoid repeat API calls.
    """
    cache_path = os.path.join(CACHE_DIR, f"markets_{days}d.json")
    if os.path.exists(cache_path):
        print(f"Loading markets from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    markets = []
    cursor = None

    print(f"Fetching settled {KALSHI_SERIES} markets (last {days} days)...")
    while True:
        params = {
            "series_ticker": KALSHI_SERIES,
            "status": "settled",
            "min_close_ts": cutoff_ts,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        data = _get("/markets", params=params)
        batch = data.get("markets", [])
        markets.extend(batch)

        cursor = data.get("cursor")
        print(f"  Fetched {len(batch)} markets (total: {len(markets)})")
        if not cursor or not batch:
            break
        time.sleep(0.3)

    with open(cache_path, "w") as f:
        json.dump(markets, f)
    print(f"Saved {len(markets)} markets to cache.")
    return markets


def fetch_candlesticks(ticker, open_time_iso, close_time_iso):
    """
    Fetches 1-minute candlesticks for a historical market.
    Returns a list of {ts, yes_open, yes_close} dicts sorted by time.
    Caches per-ticker to avoid repeat calls.
    """
    cache_path = os.path.join(CACHE_DIR, f"candles_{ticker}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    open_dt = datetime.fromisoformat(open_time_iso.replace("Z", "+00:00"))
    close_dt = datetime.fromisoformat(close_time_iso.replace("Z", "+00:00"))

    # Add 1-minute buffer on each side
    start_ts = int((open_dt - timedelta(minutes=1)).timestamp())
    end_ts = int((close_dt + timedelta(minutes=1)).timestamp())

    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1,
    }

    try:
        data = _get(f"/series/{KALSHI_SERIES}/markets/{ticker}/candlesticks", params=params)
    except Exception as e:
        print(f"  Could not fetch candlesticks for {ticker}: {e}")
        return []

    raw = data.get("candlesticks", [])
    result = []
    for c in raw:
        price = c.get("price", {})
        # API uses open_dollars/close_dollars suffix
        yes_open = price.get("open_dollars") or price.get("open")
        yes_close = price.get("close_dollars") or price.get("close")
        if yes_open is None or yes_close is None:
            continue
        result.append({
            "ts": c["end_period_ts"],
            "yes_open": float(yes_open),
            "yes_close": float(yes_close),
        })

    result.sort(key=lambda x: x["ts"])

    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


def get_yes_price_at(candles, target_ts):
    """
    Returns the most recent yes_close price at or before target_ts.
    Returns None if no candle exists before the target.
    """
    price = None
    for c in candles:
        if c["ts"] <= target_ts:
            price = c["yes_close"]
        else:
            break
    return price
