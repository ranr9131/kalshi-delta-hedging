"""
Quick portfolio snapshot using your Kalshi API key.
Run: python check_portfolio.py

Shows: balance, open positions, last 20 fills.
"""

import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import dotenv_values

import kalshi_auth

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
env = dotenv_values(_env_path)

API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
raw_pem     = env.get("KALSHI_PRIVATE_KEY", "")
if not API_KEY_ID or not raw_pem:
    print("ERROR: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY missing from .env")
    sys.exit(1)

private_key = kalshi_auth.load_private_key(raw_pem)
BASE_URL    = "https://api.elections.kalshi.com"


def _get(path: str, params: dict = None) -> dict:
    headers = kalshi_auth.make_auth_headers(private_key, API_KEY_ID, "GET", path)
    resp    = requests.get(BASE_URL + path, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def show_balance():
    data           = _get("/trade-api/v2/portfolio/balance")
    balance_cents  = data.get("balance", 0)
    print(f"Balance: ${balance_cents / 100:.2f}")


def show_positions():
    data      = _get("/trade-api/v2/portfolio/positions", {"limit": 50})
    positions = data.get("market_positions", [])
    open_pos  = [p for p in positions if p.get("total_traded", 0) > 0]
    if not open_pos:
        print("Positions: none")
        return
    print(f"\nPositions ({len(open_pos)}):")
    for p in open_pos:
        ticker    = p.get("ticker", "?")
        yes_held  = p.get("position", 0)
        resting   = p.get("resting_orders_count", 0)
        value     = p.get("market_exposure", 0) / 100
        print(f"  {ticker}  held={yes_held:+.2f}  value=${value:.2f}  resting_orders={resting}")


def show_fills():
    data  = _get("/trade-api/v2/portfolio/fills", {"limit": 20})
    fills = data.get("fills", [])
    if not fills:
        print("\nFills: none")
        return
    print(f"\nLast {len(fills)} fills:")
    for f in fills:
        ts      = f.get("created_time", "")
        ticker  = f.get("ticker", "?")
        side    = f.get("side", "?")
        count   = f.get("count", 0)
        price   = f.get("yes_price", 0) / 100
        action  = f.get("action", "?")
        try:
            dt  = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts  = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            pass
        print(f"  {ts}  {ticker}  {action} {side}  {count} contracts @ ${price:.3f}")


if __name__ == "__main__":
    show_balance()
    show_positions()
    show_fills()
