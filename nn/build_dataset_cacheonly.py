"""
Cache-only rebuild of dataset_v2.npz.

Same as build_dataset_v2.main(), but candle fetches that would hit the Kalshi
API (uncached windows) are skipped instead -- avoids HTTP 429 rate limiting on
the production endpoint. Result: every window whose 1-min candles are already in
data/cache/ is included; uncached windows are dropped.

This keeps the full Mar23->May25 history (cached) plus whatever post-May-25
windows the live bots already cached, so the retrain can be checked on recent
data without hammering the API.
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kalshi_client
from config import CACHE_DIR

_orig_fetch = kalshi_client.fetch_candlesticks

def _cache_only_fetch(ticker, open_time_iso, close_time_iso):
    cache_path = os.path.join(CACHE_DIR, f"candles_{ticker}.json")
    if os.path.exists(cache_path):
        return _orig_fetch(ticker, open_time_iso, close_time_iso)
    return []  # uncached -> skip API call, window gets dropped

kalshi_client.fetch_candlesticks = _cache_only_fetch

import build_dataset_v2
build_dataset_v2.main()
