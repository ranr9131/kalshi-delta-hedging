"""
test_price_feed.py

Compares BTC price sources side-by-side to check for systematic drift:
  1. Coinbase WebSocket (what the trader uses)
  2. Coinbase REST   (same exchange, different transport)
  3. Kalshi floor_strike (what determines the settlement cutoff)

At each window boundary, also computes the implied direction from each source
and flags disagreements.

Run for ~5 minutes to capture at least one window boundary.
Usage:
  python test_price_feed.py
"""

import time
from datetime import datetime, timezone

import requests

import btc_feed

COINBASE_REST_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLL_INTERVAL = 5   # seconds between each comparison row
WINDOW_SECS   = 15 * 60


def get_coinbase_rest() -> float | None:
    try:
        resp = requests.get(COINBASE_REST_URL, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as e:
        print(f"  [REST error] {e}")
        return None


def get_kalshi_floor() -> tuple[float | None, str]:
    try:
        resp = requests.get(
            KALSHI_MARKETS_URL,
            params={"series_ticker": "KXBTC15M", "status": "open"},
            timeout=5,
        )
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        if markets:
            m = markets[0]
            return float(m["floor_strike"]), m["ticker"]
        return None, ""
    except Exception as e:
        print(f"  [Kalshi error] {e}")
        return None, ""


def window_boundary(dt: datetime) -> datetime:
    b = (dt.minute // 15) * 15
    return dt.replace(minute=b, second=0, microsecond=0)


def main():
    print("Starting BTC feed WebSocket...")
    btc_feed.start()

    print("Waiting for first WS price...")
    for _ in range(15):
        if btc_feed.get_price() is not None:
            break
        time.sleep(1)
    else:
        print("ERROR: no WS price received in 15s")
        return

    print(f"\n{'Time':>12}  {'WS price':>12}  {'REST price':>12}  {'WS-REST':>9}  {'Kalshi floor':>14}  {'WS-floor':>10}  {'Ticker'}")
    print("-" * 110)

    last_ticker = ""
    last_boundary = None

    while True:
        now    = datetime.now(timezone.utc)
        ws     = btc_feed.get_price()
        ws_age = btc_feed.get_price_age()
        rest   = get_coinbase_rest()
        floor, ticker = get_kalshi_floor()

        boundary = window_boundary(now)
        new_window = (boundary != last_boundary)
        if new_window:
            last_boundary = boundary
            print(f"\n  -- Window boundary: {boundary.strftime('%H:%M')} UTC --")

        ws_rest_diff  = (ws - rest)   if (ws and rest)   else None
        ws_floor_diff = (ws - floor)  if (ws and floor)  else None

        ws_str    = f"${ws:,.2f}"     if ws    else "  N/A"
        rest_str  = f"${rest:,.2f}"   if rest  else "  N/A"
        floor_str = f"${floor:,.2f}"  if floor else "  N/A"
        diff1_str = f"{ws_rest_diff:+.2f}"  if ws_rest_diff  is not None else "   N/A"
        diff2_str = f"{ws_floor_diff:+.2f}" if ws_floor_diff is not None else "   N/A"

        flag = ""
        if ws_rest_diff is not None and abs(ws_rest_diff) > 5:
            flag += " !! WS/REST GAP"
        if ws_floor_diff is not None and abs(ws_floor_diff) > 20:
            flag += " !! LARGE FLOOR DIFF"
        if ticker != last_ticker and last_ticker:
            flag += " [new window]"
            last_ticker = ticker
        elif not last_ticker:
            last_ticker = ticker

        stale = f" (stale {ws_age:.0f}s)" if ws_age > 3 else ""

        print(
            f"{now.strftime('%H:%M:%S'):>12}  {ws_str:>12}{stale}  {rest_str:>12}"
            f"  {diff1_str:>9}  {floor_str:>14}  {diff2_str:>10}  {ticker}{flag}"
        )

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
