"""
Kalshi market fetching and order placement.
Field names confirmed from live API: yes_bid_dollars, yes_ask_dollars, no_ask_dollars.
"""

import uuid
import requests
from kalshi_auth import make_auth_headers

BASE_URL = "https://api.elections.kalshi.com"
SERIES   = "KXBTC15M"

FILL_BUFFER_CENTS = 2  # absorbs ~300ms price movement between WS read and order landing


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
        yes_price_cents   = round(float(market["yes_ask_dollars"]) * 100) + FILL_BUFFER_CENTS
        cost_per_contract = yes_price_cents / 100.0
    else:
        yes_price_cents   = round(float(market["yes_bid_dollars"]) * 100) - FILL_BUFFER_CENTS
        cost_per_contract = 1.0 - yes_price_cents / 100.0

    yes_price_cents = max(1, min(99, yes_price_cents))
    count = max(1, round(stake_dollars / cost_per_contract))

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
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason}: {resp.text}", response=resp
        )
    return resp.json()


def cancel_order(private_key, api_key_id: str, order_id: str) -> bool:
    """Cancel a resting order. Returns True if cancelled successfully."""
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    headers = make_auth_headers(private_key, api_key_id, "DELETE", path)
    try:
        resp = requests.delete(BASE_URL + path, headers=headers, timeout=10)
        return resp.ok
    except Exception:
        return False


def get_order_status(private_key, api_key_id: str, order_id: str) -> dict:
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    headers = make_auth_headers(private_key, api_key_id, "GET", path)
    resp = requests.get(BASE_URL + path, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("order", {})


def get_order_filled_stake(private_key, api_key_id: str, order_id: str) -> float:
    """
    Return total dollars actually filled for `order_id`. Used to recover
    partial fills that landed on Kalshi between place and cancel.
    Returns 0.0 on any error (caller should not block on this).
    """
    path = "/trade-api/v2/portfolio/fills"
    headers = make_auth_headers(private_key, api_key_id, "GET", path)
    try:
        resp = requests.get(
            BASE_URL + path,
            params={"order_id": order_id, "limit": 100},
            headers=headers, timeout=10,
        )
        resp.raise_for_status()
        fills = resp.json().get("fills", [])
        total = 0.0
        for f in fills:
            # Defensive: in case the order_id query param is ignored, filter locally.
            fill_oid = f.get("order_id")
            if fill_oid and fill_oid != order_id:
                continue
            count       = float(f.get("count", 0))
            price_cents = float(f.get("yes_price", 0))
            side        = f.get("side", "yes")
            per_contract = price_cents / 100.0 if side == "yes" else (100.0 - price_cents) / 100.0
            total += count * per_contract
        return total
    except Exception:
        return 0.0


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
