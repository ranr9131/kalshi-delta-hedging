"""
Scan all open Kalshi markets for market-making opportunities.

Pulls every open market across all series, sorts by:
  - Spread (wider = more capturable edge per round trip)
  - Volume (more volume = more potential fills per hour)
  - Score = spread × sqrt(daily_volume) — favors wide AND active

Outputs the top 30 candidates so we can pick where to provide liquidity.

Run: python3 scan_markets.py
"""
import os
import sys
import math
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import kalshi_auth

_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))
API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
raw_pem     = env.get("KALSHI_PRIVATE_KEY", "")
PRIVATE_KEY = kalshi_auth.load_private_key(raw_pem) if raw_pem else None

BASE_URL = "https://api.elections.kalshi.com"


def fetch_all_open_markets():
    """Page through every open market."""
    out = []
    cursor = ""
    page = 0
    while True:
        params = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(f"{BASE_URL}/trade-api/v2/markets", params=params, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"page {page} error: {e}")
            break
        data = resp.json()
        markets = data.get("markets", [])
        if not markets:
            break
        out.extend(markets)
        cursor = data.get("cursor", "")
        page += 1
        if not cursor:
            break
        if page > 30:   # safety
            break
        time.sleep(0.1)
    return out


def score_market(m, require_volume=True):
    """Compute a MM-suitability score. Higher = better for MM."""
    try:
        bid = float(m.get("yes_bid_dollars", 0))   # in dollars (0.xx)
        ask = float(m.get("yes_ask_dollars", 0))
        vol_24h = float(m.get("volume_24h_fp", 0))
        vol_total = float(m.get("volume_fp", 0))
        liq = float(m.get("liquidity_dollars", 0))
        bid_size = float(m.get("yes_bid_size_fp", 0))
        ask_size = float(m.get("yes_ask_size_fp", 0))
    except Exception:
        return 0, {}

    if bid <= 0 or ask <= 0 or ask <= bid:
        return 0, {}

    # Convert to cents for cleaner display
    bid_c = round(bid * 100)
    ask_c = round(ask * 100)

    # Skip extreme prices — limited spread upside
    if bid_c < 5 or ask_c > 95:
        return 0, {}

    spread = ask_c - bid_c
    if spread < 2:   # need at least 2c room to MM
        return 0, {}

    # Need actual liquidity — skip markets with no real bid
    if bid_size < 1 or ask_size < 1:
        return 0, {}

    # Require non-zero recent volume — MM only works if there's flow to capture
    if require_volume and vol_24h < 10:
        return 0, {}

    # Score: spread × sqrt(daily volume). Favors both width and activity.
    score = spread * math.sqrt(max(vol_24h, 1))

    return score, {
        "ticker":   m.get("ticker"),
        "title":    m.get("title", "")[:55] if m.get("title") else "",
        "bid":      bid_c,
        "ask":      ask_c,
        "spread":   spread,
        "vol_24h":  vol_24h,
        "vol_total": vol_total,
        "liquidity": liq,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "close":    m.get("close_time", "")[:16],
    }


def main():
    print("Fetching all open Kalshi markets...")
    markets = fetch_all_open_markets()
    print(f"  {len(markets)} open markets across all series\n")

    scored = []
    for m in markets:
        s, info = score_market(m)
        if s > 0:
            scored.append((s, info))
    scored.sort(reverse=True, key=lambda x: x[0])

    print(f"Scored {len(scored)} markets meeting basic MM criteria (spread>=2c, prices 5-95c, valid book).\n")

    # Top 30
    print(f"{'#':>3}  {'score':>7}  {'spread':>6}  {'bid':>4}  {'ask':>4}  {'bid_sz':>6}  {'ask_sz':>6}  {'vol_24h':>8}  {'liq':>8}  {'ticker':<45}")
    print("-" * 120)
    for i, (s, info) in enumerate(scored[:30]):
        print(f"{i+1:>3}  {s:>7.0f}  {info['spread']:>5}c  "
              f"{info['bid']:>3}c  {info['ask']:>3}c  "
              f"{info['bid_size']:>6.0f}  {info['ask_size']:>6.0f}  "
              f"{info['vol_24h']:>8.0f}  {info['liquidity']:>8.0f}  "
              f"{info['ticker']:<45}")
        if info.get('title'):
            print(f"      {info['title']}")

    # Bucket by series prefix to show where the meat is
    print("\n\nBy series (top of scored list):")
    by_series: dict[str, list] = {}
    for s, info in scored:
        prefix = info["ticker"].split("-")[0]
        by_series.setdefault(prefix, []).append((s, info))
    series_summary = []
    for prefix, lst in by_series.items():
        total_score = sum(x[0] for x in lst)
        avg_spread = sum(x[1]["spread"] for x in lst) / len(lst)
        total_vol  = sum(x[1]["vol_24h"] for x in lst)
        series_summary.append((total_score, prefix, len(lst), avg_spread, total_vol))
    series_summary.sort(reverse=True)
    print(f"{'series':<20}  {'#mkts':>6}  {'avg_spread':>11}  {'sum_vol_24h':>12}  {'sum_score':>10}")
    print("-" * 75)
    for total_score, prefix, count, avg_spread, total_vol in series_summary[:15]:
        print(f"{prefix:<20}  {count:>6}  {avg_spread:>10.1f}c  {total_vol:>12,.0f}  {total_score:>10.0f}")


if __name__ == "__main__":
    main()
