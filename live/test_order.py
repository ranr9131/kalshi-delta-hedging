"""
Live order test: place a tiny YES buy on the current KXBTC15M market,
then verify the actual fill price matches what the trader assumes.

Tests:
  1. Orders reach Kalshi and are accepted
  2. Actual fill price == yes_ask (simulation assumption)
  3. Order-to-fill latency

Usage:
  python test_order.py          # default $3 stake
  python test_order.py 5.00     # custom stake

WARNING: This places a REAL order using your API key. Small stake only.
"""

import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import dotenv_values

import kalshi_auth
import kalshi_feed
import kalshi_trade
from kalshi_trade import FILL_BUFFER_CENTS

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
env = dotenv_values(_env_path)

API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
raw_pem     = env.get("KALSHI_PRIVATE_KEY", "")
if not API_KEY_ID or not raw_pem:
    print("ERROR: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY missing from .env")
    sys.exit(1)

private_key = kalshi_auth.load_private_key(raw_pem)
BASE_URL    = "https://api.elections.kalshi.com"
STAKE       = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0


def _get(path: str, params: dict = None) -> dict:
    headers = kalshi_auth.make_auth_headers(private_key, API_KEY_ID, "GET", path)
    resp    = requests.get(BASE_URL + path, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_order_status(order_id: str) -> dict:
    return _get(f"/trade-api/v2/portfolio/orders/{order_id}").get("order", {})


def get_recent_fills(limit: int = 10) -> list:
    return _get("/trade-api/v2/portfolio/fills", {"limit": limit}).get("fills", [])


def get_balance() -> float:
    cents = _get("/trade-api/v2/portfolio/balance").get("balance", 0)
    return cents / 100.0


def main():
    # ── Fetch market metadata via REST, prices via WebSocket ──────────────────
    print("Fetching current KXBTC15M market...")
    market = kalshi_trade.get_open_market()
    if market is None:
        print("ERROR: No open KXBTC15M market right now. Try again in a moment.")
        sys.exit(1)

    ticker     = market["ticker"]
    floor      = float(market["floor_strike"])
    close_time = market.get("close_time", "")

    # Start WebSocket and wait up to 3s for live prices
    kalshi_feed.start(private_key, API_KEY_ID, ticker)
    for _ in range(30):
        if kalshi_feed.get_bid() is not None:
            break
        time.sleep(0.1)

    if kalshi_feed.get_bid() is not None:
        yes_bid = kalshi_feed.get_bid()
        yes_ask = kalshi_feed.get_ask()
        print(f"Using WebSocket prices (age={kalshi_feed.get_age():.1f}s)")
    else:
        yes_bid = float(market["yes_bid_dollars"])
        yes_ask = float(market["yes_ask_dollars"])
        print("WebSocket not ready, using REST prices (may be stale)")

    spread        = round(yes_ask - yes_bid, 4)
    expected_fill = yes_ask + FILL_BUFFER_CENTS / 100
    count         = max(1, round(STAKE / expected_fill))

    balance_before = get_balance()

    print(f"\nMarket:        {ticker}")
    print(f"Floor strike:  ${floor:,.2f}")
    print(f"Close time:    {close_time}")
    rest_bid = float(market["yes_bid_dollars"])
    rest_ask = float(market["yes_ask_dollars"])
    print(f"Bid / Ask:     {yes_bid:.3f} / {yes_ask:.3f}  (spread={spread:.3f})  REST was {rest_bid:.3f}/{rest_ask:.3f}")
    print(f"Balance:       ${balance_before:.2f}")
    print(f"\nOrder:         BUY YES  ${STAKE:.2f} -> {count} contracts @ {expected_fill:.3f} ({expected_fill*100:.1f}c limit = ask {yes_ask:.3f} + {FILL_BUFFER_CENTS}c buffer)")

    # ── Place order ───────────────────────────────────────────────────────────
    t_submit = time.time()
    try:
        resp = kalshi_trade.place_order(private_key, API_KEY_ID, ticker, "yes", market, STAKE)
    except Exception as e:
        print(f"\nERROR placing order: {e}")
        sys.exit(1)
    t_accepted = time.time()
    submit_ms  = (t_accepted - t_submit) * 1000

    order      = resp.get("order", resp)
    order_id   = order.get("order_id", "unknown")
    status     = order.get("status", "unknown")
    order_price_cents = order.get("yes_price")

    print(f"\nOrder accepted in {submit_ms:.0f}ms")
    print(f"  order_id:    {order_id}")
    print(f"  status:      {status}")
    if order_price_cents is not None:
        print(f"  yes_price:   {order_price_cents}c  ({order_price_cents/100:.3f})")

    # ── Poll for fill ─────────────────────────────────────────────────────────
    print("\nPolling for fill (up to 15s)...")
    fill_found   = None
    t_fill       = None

    for attempt in range(15):
        time.sleep(1)
        fills = get_recent_fills(limit=10)
        for f in fills:
            fill_ts_str = f.get("created_time", "")
            try:
                fill_ts = datetime.fromisoformat(fill_ts_str.replace("Z", "+00:00")).timestamp()
            except Exception:
                fill_ts = 0
            if f.get("order_id") == order_id or (f.get("ticker") == ticker and fill_ts > t_submit - 5):
                fill_found = f
                t_fill     = time.time()
                break
        if fill_found:
            break
        print(f"  {attempt+1:2d}s: no fill yet...")

    fill_latency = (t_fill - t_submit) if t_fill else None

    # ── Order status ──────────────────────────────────────────────────────────
    print("\nRaw order response:")
    try:
        o = get_order_status(order_id)
        for k, v in o.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  Could not fetch order: {e}")
        o = {}

    if fill_found:
        print("\nRaw fill record:")
        for k, v in fill_found.items():
            print(f"  {k}: {v}")

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"EXPECTED fill price:  {yes_ask:.4f}  ({yes_ask*100:.1f}c/contract)")

    if fill_found:
        actual_price  = float(fill_found.get("yes_price_dollars") or 0)
        actual_count  = float(fill_found.get("count_fp") or fill_found.get("count") or 0)
        slippage      = actual_price - yes_ask
        slippage_pct  = slippage / yes_ask * 100

        print(f"ACTUAL fill price:    {actual_price:.4f}  ({actual_price*100:.1f}c/contract)")
        print(f"Slippage:             {slippage:+.4f}  ({slippage*100:+.2f}c,  {slippage_pct:+.2f}%)")
        print(f"Contracts filled:     {actual_count}")
        print(f"Submit latency:       {submit_ms:.0f}ms")
        print(f"Fill latency:         {fill_latency:.1f}s")

        if abs(slippage) < 0.005:
            print("\nFill matches expected — simulation assumptions are valid.")
        else:
            print(f"\nWARNING: {slippage*100:+.2f}c slippage. Simulation may underestimate cost.")
    else:
        print("Fill record not found in API within 15s.")
        print("(Order may still have executed — check raw order response above.)")

    balance_after = get_balance()
    print(f"\nBalance before: ${balance_before:.2f}")
    print(f"Balance after:  ${balance_after:.2f}  (delta=${balance_after - balance_before:+.2f})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
