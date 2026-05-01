"""
Kalshi market fetching and order placement.
Field names confirmed from live API: yes_bid_dollars, yes_ask_dollars, no_ask_dollars.
"""

import uuid
import requests
from kalshi_auth import make_auth_headers

BASE_URL = "https://api.elections.kalshi.com"
SERIES   = "KXBTC15M"


def get_open_market() -> dict | None:
    """
    Fetch the currently open KXBTC15M market. No auth required.
    Returns the market dict or None if no open market found.

    Relevant fields:
      ticker           - e.g. "KXBTC15M-26APR302130-30"
      yes_bid_dollars  - e.g. "0.7400"
      yes_ask_dollars  - e.g. "0.7600"
      no_ask_dollars   - e.g. "0.2600"
      floor_strike     - BTC target price (float)
      open_time        - ISO string
      close_time       - ISO string
    """
    resp = requests.get(
        f"{BASE_URL}/trade-api/v2/markets",
        params={"series_ticker": SERIES, "status": "open"},
        timeout=10,
    )
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    return markets[0] if markets else None


def place_order(
    private_key,
    api_key_id: str,
    ticker: str,
    side: str,
    market: dict,
    stake_dollars: float,
) -> dict:
    """
    Place a limit buy order for Yes or No.

    side: "yes" or "no"
    market: the dict returned by get_open_market()
    stake_dollars: dollar amount to risk

    Pricing for immediate fill:
      Yes buy: yes_price = yes_ask (we cross the ask)
      No buy:  yes_price = yes_bid (crossing the ask for No = 1 - yes_bid)

    count: fractional contracts supported (fractional_trading_enabled=true).
    """
    if side == "yes":
        yes_price_dollars = float(market["yes_ask_dollars"])
        cost_per_contract = yes_price_dollars
    else:
        yes_price_dollars = float(market["yes_bid_dollars"])
        cost_per_contract = 1.0 - yes_price_dollars   # No price = no_ask

    yes_price_cents = round(yes_price_dollars * 100)
    count = round(stake_dollars / cost_per_contract, 2)   # fractional ok
    count = max(0.01, count)

    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker":          ticker,
        "action":          "buy",
        "side":            side,
        "count":           count,
        "type":            "limit",
        "yes_price":       yes_price_cents,
        "client_order_id": str(uuid.uuid4()),
    }

    headers = make_auth_headers(private_key, api_key_id, "POST", path)
    resp = requests.post(BASE_URL + path, json=body, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_market_result(ticker: str) -> str | None:
    """
    Return 'yes' or 'no' if market is finalized, else None. No auth required.
    Poll this after close_time until it returns non-None.
    """
    resp = requests.get(
        f"{BASE_URL}/trade-api/v2/markets/{ticker}",
        timeout=10,
    )
    resp.raise_for_status()
    market = resp.json().get("market", {})
    if market.get("status") == "finalized":
        return market.get("result")  # "yes" or "no"
    return None


def get_balance(private_key, api_key_id: str) -> float | None:
    """Return available balance in dollars, or None on error."""
    path = "/trade-api/v2/portfolio/balance"
    headers = make_auth_headers(private_key, api_key_id, "GET", path)
    try:
        resp = requests.get(BASE_URL + path, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        balance_cents = data.get("balance", 0)
        return balance_cents / 100.0
    except Exception:
        return None
