"""
Kalshi market fetching and order placement.
Field names confirmed from live API: yes_bid_dollars, yes_ask_dollars, no_ask_dollars.

A module-level `requests.Session()` is used so that TLS handshakes are reused
across calls.  Measured savings on AWS↔Kalshi: ~30ms per signed call (69ms →
39ms median).  Critical for the sniper, where every millisecond between
detecting a stale level and submitting the IOC matters.
"""

import os
import uuid
import requests
from kalshi_auth import make_auth_headers

BASE_URL = "https://api.elections.kalshi.com"
SERIES   = "KXBTC15M"

# V2 order endpoint (V1 POST /trade-api/v2/portfolio/orders returns 410
# deprecated_v1_order_endpoint since ~2026-05-06).
V2_ORDERS_PATH = "/trade-api/v2/portfolio/events/orders"


def _normalize_v2_order_response(raw: dict, requested_count: float) -> dict:
    """Map a V2 order response onto the V1-ish {"order": {...}} shape callers
    expect (order_id + status), while passing through the V2 extras
    (average_fill_price, average_fee_paid, fill_count, remaining_count).

    V2 has no "resting" status string; derive it: remaining > 0 -> resting,
    fills > 0 -> executed, else canceled.
    """
    o = dict(raw.get("order", raw))
    fill_count = float(o.get("fill_count") or 0)
    remaining  = float(o.get("remaining_count") or 0)
    if "status" not in o:
        if remaining > 1e-9:
            o["status"] = "resting"
        elif fill_count > 1e-9:
            o["status"] = "executed"
        else:
            o["status"] = "canceled"
    o.setdefault("requested_count", requested_count)
    return {"order": o, "v2_raw": raw}

# Module-level pooled session.  Reused across all HTTP calls in this file.
# urllib3 keeps the underlying TCP/TLS connection alive between calls.
_session = requests.Session()


def warmup_session():
    """Issue a no-op unauthenticated GET so the TCP+TLS handshake completes
    before the first real order.  Call this at sniper boot to remove cold-
    start lag from the first fire."""
    try:
        _session.get(f"{BASE_URL}/trade-api/v2/markets",
                     params={"series_ticker": SERIES, "status": "open", "limit": 1},
                     timeout=10)
        return True
    except Exception:
        return False

# Buffer added to the limit price to absorb price movement between quote read
# and order landing. History assumed ~300ms latency; measured 2026-07-02 on the
# AWS box: 27ms order round-trip, and book depth shows 0.0c walk at our sizes —
# so 5c is ~10x oversized and erases most positive-edge decisions. Env-
# overridable per instance (FILL_BUFFER_CENTS=1) for shadow A/B testing.
FILL_BUFFER_CENTS = int(os.environ.get("FILL_BUFFER_CENTS", "5"))
# History: 2c → 5c (raised after rested-and-cancelled failures on fast markets)
# → 3c (lowered 2026-05-22 after sim showed 5c buffer is the dominant friction
# costing ~16pp ROI. 3c is a compromise: half the spread cost vs 5c, while still
# absorbing typical 1-3c price movements between read and order landing). On
# faster moves we now rely on the chase-retry path in place_order_with_retry.
# → 5c (raised 2026-06-01 after observing 15-35% order failure rate live across
# all 4 traders, especially XRP at 35%. Sim's projected -16pp ROI cost is the
# THEORETICAL ceiling but doesn't capture the lost edge from bets that never
# execute at all. 5c → expected fill rate ~95%+ across all assets.)


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
    resp = _session.get(
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
    extra_buffer_cents: int = 0,
    ioc: bool = False,  # DISABLED: expiration_ts=now+3 caused stacked resting orders
) -> dict:
    """
    Place a limit buy order for Yes or No.

    side: "yes" or "no"
    market: the dict returned by get_open_market()
    stake_dollars: dollar amount to risk
    extra_buffer_cents: additional cents added to FILL_BUFFER_CENTS, for chase
        retries (widen the limit price each attempt).
    ioc: if True, submits with time_in_force=immediate_or_cancel (first-class
        in the V2 API) — Kalshi fills what it can immediately at the limit
        price and cancels any remainder instead of leaving it on the book.
        Eliminates the "rested and cancelled" failure mode where price moved
        between read and submit.

    Pricing for immediate fill:
      Yes buy: yes_price = yes_ask (we cross the ask)
      No buy:  yes_price = yes_bid (crossing the ask for No = 1 - yes_bid)

    count: fractional contracts supported (fractional_trading_enabled=true).
    """
    total_buffer = FILL_BUFFER_CENTS + extra_buffer_cents
    if side == "yes":
        yes_price_cents   = round(float(market["yes_ask_dollars"]) * 100) + total_buffer
        cost_per_contract = yes_price_cents / 100.0
    else:
        yes_price_cents   = round(float(market["yes_bid_dollars"]) * 100) - total_buffer
        cost_per_contract = 1.0 - yes_price_cents / 100.0

    yes_price_cents = max(1, min(99, yes_price_cents))
    count = max(1, round(stake_dollars / cost_per_contract))

    # ── V2 order API (V1 POST /portfolio/orders was killed ~2026-05: 410
    # deprecated_v1_order_endpoint). V2 is a single YES book quoted in dollars:
    #   buy YES at p        -> side "bid", price p
    #   buy NO  at (1 - p)  -> side "ask", price p   (selling yes IS buying no)
    # Both cases use the same yes-denominated price we already computed above.
    # IOC is now first-class via time_in_force (the old expiration_ts hack is
    # gone). count/price must be fixed-point strings.
    path = V2_ORDERS_PATH
    body = {
        "ticker":          ticker,
        "side":            "bid" if side == "yes" else "ask",
        "count":           f"{count:.2f}",
        "price":           f"{yes_price_cents / 100.0:.4f}",
        "time_in_force":   "immediate_or_cancel" if ioc else "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": str(uuid.uuid4()),
    }

    headers = make_auth_headers(private_key, api_key_id, "POST", path)
    resp = _session.post(BASE_URL + path, json=body, headers=headers, timeout=10)
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason}: {resp.text}", response=resp
        )
    return _normalize_v2_order_response(resp.json(), count)


def cancel_order(private_key, api_key_id: str, order_id: str) -> bool:
    """Cancel a resting order. Returns True if cancelled successfully.
    Tries the V2 path first; falls back to the legacy path just in case."""
    for path in (f"{V2_ORDERS_PATH}/{order_id}",
                 f"/trade-api/v2/portfolio/orders/{order_id}"):
        headers = make_auth_headers(private_key, api_key_id, "DELETE", path)
        try:
            resp = _session.delete(BASE_URL + path, headers=headers, timeout=10)
            if resp.ok:
                return True
        except Exception:
            pass
    return False


def get_order_status(private_key, api_key_id: str, order_id: str) -> dict:
    """V2 first (orders are created there now), legacy path as fallback."""
    last_exc = None
    for path in (f"{V2_ORDERS_PATH}/{order_id}",
                 f"/trade-api/v2/portfolio/orders/{order_id}"):
        headers = make_auth_headers(private_key, api_key_id, "GET", path)
        try:
            resp = _session.get(BASE_URL + path, headers=headers, timeout=10)
            if resp.ok:
                j = resp.json()
                return j.get("order", j)
            last_exc = requests.HTTPError(f"{resp.status_code} {resp.reason}")
        except Exception as e:
            last_exc = e
    raise last_exc


def get_order_filled_stake(private_key, api_key_id: str, order_id: str) -> float:
    """
    Return total dollars actually filled for `order_id`. Used to recover
    partial fills that landed on Kalshi between place and cancel.
    Returns 0.0 on any error (caller should not block on this).
    """
    path = "/trade-api/v2/portfolio/fills"
    headers = make_auth_headers(private_key, api_key_id, "GET", path)
    try:
        resp = _session.get(
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
    resp = _session.get(
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
        resp = _session.get(BASE_URL + path, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        balance_cents = data.get("balance", 0)
        return balance_cents / 100.0
    except Exception:
        return None


def get_order_fills(private_key, api_key_id: str, order_id: str) -> dict:
    """
    Return the ACTUAL executed fill for `order_id` — ground truth for slippage.
    {filled_count, vwap, n_fills} where vwap is the size-weighted average price
    per contract in dollars, from the bet side's perspective. filled_count=0 if
    nothing filled or on error (caller should not block on this).
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
        total_cost = 0.0
        total_count = 0.0
        n = 0
        for f in fills:
            fill_oid = f.get("order_id")
            if fill_oid and fill_oid != order_id:
                continue
            count       = float(f.get("count", 0))
            price_cents = float(f.get("yes_price", 0))
            side        = f.get("side", "yes")
            per_contract = price_cents / 100.0 if side == "yes" else (100.0 - price_cents) / 100.0
            total_cost += count * per_contract
            total_count += count
            n += 1
        vwap = (total_cost / total_count) if total_count > 0 else None
        return {"filled_count": total_count, "vwap": vwap, "n_fills": n}
    except Exception:
        return {"filled_count": 0.0, "vwap": None, "n_fills": 0}
