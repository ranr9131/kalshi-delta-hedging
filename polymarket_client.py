"""
Fetches Polymarket BTC 5-minute market resolutions by constructing slugs
directly from Kalshi market open timestamps.

Slug: btc-updown-5m-{end_unix_ts} where end = kalshi_open_ts + 300

Resolution: outcomePrices[0] > 0.5 means YES won (BTC went UP).
No authentication required.
"""

import time
import json
import os
import requests
from config import CACHE_DIR

GAMMA_BASE = "https://gamma-api.polymarket.com"
_slug_cache = {}   # in-memory: end_ts -> True/False/None


def _load_cache():
    cache_path = os.path.join(CACHE_DIR, "poly_resolutions.json")
    if os.path.exists(cache_path) and not _slug_cache:
        with open(cache_path) as f:
            raw = json.load(f)
        _slug_cache.update({int(k): v for k, v in raw.items()})


def _save_cache():
    cache_path = os.path.join(CACHE_DIR, "poly_resolutions.json")
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in _slug_cache.items()}, f)


def _fetch_slug(end_ts):
    """
    Fetch a single BTC 5-min market by its end timestamp.
    Returns True (BTC up), False (BTC down), or None (not found/unresolved).
    Caches result in memory and persists to disk.
    """
    if end_ts in _slug_cache:
        return _slug_cache[end_ts]

    slug = f"btc-updown-5m-{end_ts}"
    url = f"{GAMMA_BASE}/markets/slug/{slug}"

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 404:
                _slug_cache[end_ts] = None
                return None
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.HTTPError as e:
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            _slug_cache[end_ts] = None
            return None
        except Exception:
            time.sleep(1)
    else:
        _slug_cache[end_ts] = None
        return None

    # data could be a list or a single market dict
    market = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
    if not market:
        _slug_cache[end_ts] = None
        return None

    raw_prices = market.get("outcomePrices", [])
    if not raw_prices:
        _slug_cache[end_ts] = None
        return None

    # outcomePrices comes as a JSON-encoded string e.g. '["1", "0"]'
    try:
        if isinstance(raw_prices, str):
            import json as _json
            raw_prices = _json.loads(raw_prices)
        yes_price = float(raw_prices[0])
    except (TypeError, ValueError, IndexError):
        _slug_cache[end_ts] = None
        return None

    # After resolution: winning outcome = 1.0, losing = 0.0
    result = yes_price > 0.5
    _slug_cache[end_ts] = result
    return result


def prefetch_for_markets(kalshi_markets, batch_save_every=100):
    """
    Pre-fetches Polymarket 5-min resolutions for all Kalshi markets.
    Kalshi market open_ts + 300 = the Polymarket end_ts (slug suffix).
    Skips already-cached entries.
    """
    _load_cache()

    from datetime import datetime, timezone
    to_fetch = []
    for m in kalshi_markets:
        try:
            open_dt = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            t0 = int(open_dt.timestamp())
            end_ts = t0 + 300
            if end_ts not in _slug_cache:
                to_fetch.append(end_ts)
        except Exception:
            pass

    if not to_fetch:
        print(f"Polymarket: all {len(kalshi_markets)} signals already cached.")
        return

    print(f"Fetching {len(to_fetch)} Polymarket 5-min resolutions (skipping {len(kalshi_markets)-len(to_fetch)} cached)...")
    found = 0
    for i, end_ts in enumerate(to_fetch):
        result = _fetch_slug(end_ts)
        if result is not None:
            found += 1
        if (i + 1) % batch_save_every == 0:
            _save_cache()
            print(f"  [{i+1}/{len(to_fetch)}] found so far: {found}")
        time.sleep(0.12)

    _save_cache()
    print(f"Done. Found {found}/{len(to_fetch)} resolutions.")


def lookup_t5_signal(kalshi_open_ts):
    """
    Returns True (BTC up at T+5), False (BTC down), or None (missing).
    """
    _load_cache()
    end_ts = kalshi_open_ts + 300
    if end_ts in _slug_cache:
        return _slug_cache[end_ts]
    return _fetch_slug(end_ts)
