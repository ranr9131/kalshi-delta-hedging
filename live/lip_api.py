"""
Rate-limited Kalshi REST client for the LIP farmer.

Reuses kalshi_auth for signing. Separate from kalshi_trade.py because LIP
needs raw resting limit orders at explicit prices (kalshi_trade.place_order
is a taker-style helper that prices off the touch with a fill buffer).

Two token buckets: public (unauthenticated reads) and auth (signed calls),
so a book-scan sweep can never starve order placement.
"""

import logging
import threading
import time
import uuid

import requests

from kalshi_auth import make_auth_headers
import lip_config as cfg

BASE_URL = "https://api.elections.kalshi.com"

log = logging.getLogger("lip_api")

_session = requests.Session()


class TokenBucket:
    def __init__(self, rps: float, burst: float = None):
        self.rate = rps
        self.capacity = burst if burst is not None else max(1.0, rps)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


_public_bucket = TokenBucket(cfg.PUBLIC_RPS)
_auth_bucket = TokenBucket(cfg.AUTH_RPS)
_write_bucket = TokenBucket(cfg.WRITE_RPS, burst=1.0)


def _write(method: str, path: str, body, private_key, api_key_id,
           retries: int = 3):
    """Signed POST/DELETE with strict write pacing and 429 backoff.
    Verified live 2026-06-10: bursting 2 POSTs inside a second gets 429."""
    for attempt in range(retries + 1):
        _write_bucket.take()
        headers = make_auth_headers(private_key, api_key_id, method, path)
        if method == "POST":
            resp = _session.post(BASE_URL + path, json=body, headers=headers, timeout=15)
        else:
            resp = _session.delete(BASE_URL + path, headers=headers, timeout=15)
        if resp.status_code == 429 and attempt < retries:
            time.sleep(2.0 * (attempt + 1))
            continue
        return resp
    return resp


def _get(path: str, params: dict = None, auth: tuple = None, retries: int = 2):
    """GET with rate limiting and one retry on transient failure."""
    for attempt in range(retries + 1):
        try:
            if auth:
                _auth_bucket.take()
                private_key, api_key_id = auth
                headers = make_auth_headers(private_key, api_key_id, "GET", path)
            else:
                _public_bucket.take()
                headers = None
            resp = _session.get(BASE_URL + path, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt >= retries:
                raise
            log.warning("GET %s failed (%s), retrying", path, e)
            time.sleep(1.0 + attempt)


# --------------------------------------------------------------------------
# Public endpoints
# --------------------------------------------------------------------------

def get_liquidity_programs(status: str = "active") -> list:
    """All liquidity incentive programs. Public, paginated."""
    out, cursor = [], None
    while True:
        params = {"status": status, "type": "liquidity", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = _get("/trade-api/v2/incentive_programs", params)
        out.extend(data.get("incentive_programs") or [])
        cursor = data.get("next_cursor")
        if not cursor:
            return out


def get_orderbook(ticker: str) -> dict:
    return _get(f"/trade-api/v2/markets/{ticker}/orderbook")


def get_markets_by_tickers(tickers: list) -> dict:
    """Batch market lookup -> {ticker: market_dict}. Chunks of 50."""
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        data = _get("/trade-api/v2/markets",
                    params={"tickers": ",".join(chunk), "limit": len(chunk)})
        for m in data.get("markets") or []:
            out[m["ticker"]] = m
    return out


# --------------------------------------------------------------------------
# Authenticated endpoints
# --------------------------------------------------------------------------

def place_resting_bid(private_key, api_key_id, ticker: str, side: str,
                      price_cents: int, count: int) -> dict:
    """
    Rest a buy limit order at an explicit price. side: "yes"|"no".
    A buy-yes at p is a yes bid at p; a buy-no at q is a no bid at q
    (equivalently a yes ask at 100-q). post_only guards against ever
    crossing if the book moved between our read and the order landing.
    Returns the order dict (with order_id).
    """
    price_cents = max(1, min(98, int(price_cents)))
    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": int(count),
        "type": "limit",
        "client_order_id": f"lip-{uuid.uuid4()}",
        "post_only": True,
    }
    if side == "yes":
        body["yes_price"] = price_cents
    else:
        body["no_price"] = price_cents
    resp = _write("POST", path, body, private_key, api_key_id)
    if resp.status_code == 400 and "post_only" in resp.text:
        # API tier without post_only support: retry without it (price caps
        # in lip_scoring already keep us far from crossing)
        body.pop("post_only")
        resp = _write("POST", path, body, private_key, api_key_id)
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code}: {resp.text}", response=resp)
    return resp.json().get("order") or resp.json()


def cancel_order(private_key, api_key_id, order_id: str) -> bool:
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    try:
        resp = _write("DELETE", path, None, private_key, api_key_id)
        if resp.ok:
            return True
        # already filled/cancelled is fine for our purposes
        if resp.status_code in (404, 409):
            return True
        log.warning("cancel %s -> %s %s", order_id, resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        log.warning("cancel %s failed: %s", order_id, e)
        return False


def get_resting_orders(private_key, api_key_id) -> list:
    """All resting orders (paginated)."""
    out, cursor = [], None
    while True:
        params = {"status": "resting", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get("/trade-api/v2/portfolio/orders", params,
                    auth=(private_key, api_key_id))
        out.extend(data.get("orders") or [])
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            return out


def get_fills(private_key, api_key_id, min_ts: int = None) -> list:
    out, cursor = [], None
    while True:
        params = {"limit": 200}
        if min_ts:
            params["min_ts"] = int(min_ts)
        if cursor:
            params["cursor"] = cursor
        data = _get("/trade-api/v2/portfolio/fills", params,
                    auth=(private_key, api_key_id))
        out.extend(data.get("fills") or [])
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            return out


def get_balance(private_key, api_key_id) -> float:
    data = _get("/trade-api/v2/portfolio/balance", auth=(private_key, api_key_id))
    return data.get("balance", 0) / 100.0
