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
        if resp.status_code == 410 or "deprecated" in (resp.text or "")[:200]:
            raise RuntimeError(f"DEPRECATED endpoint {path}: {resp.status_code} "
                               f"{resp.text[:200]}")
        return resp
    return resp


def _get(path: str, params: dict = None, auth: tuple = None, retries: int = 4):
    """GET with rate limiting and retries; never return an implicit None."""
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
            if resp.status_code == 429 and attempt < retries:
                time.sleep(3.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt >= retries:
                raise
            log.warning("GET %s failed (%s), retrying", path, e)
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"GET {path}: retries exhausted")


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

_post_only_supported = True

def place_resting_bid(private_key, api_key_id, ticker: str, side: str,
                      price_cents: int, count: int,
                      expiration_ts: int = None) -> dict:
    """
    Rest a bid via the V2 single-YES-book API. A no bid at q is represented
    as a YES ask at 100-q. The LIP side is encoded in client_order_id so the
    order listing can reconstruct it.
    """
    global _post_only_supported
    price_cents = max(1, min(98, int(price_cents)))
    if side == "yes":
        yes_price_c, v2_side = price_cents, "bid"
    else:
        yes_price_c, v2_side = 100 - price_cents, "ask"
    path = "/trade-api/v2/portfolio/events/orders"
    body = {
        "ticker": ticker,
        "side": v2_side,
        "count": f"{int(count)}.00",
        "price": f"{yes_price_c / 100.0:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": f"lip-{side}-{uuid.uuid4()}",
        "cancel_order_on_pause": True,
    }
    if _post_only_supported:
        body["post_only"] = True
    if expiration_ts is not None:
        body["expiration_time"] = int(expiration_ts)
    resp = _write("POST", path, body, private_key, api_key_id)
    if resp.status_code == 400 and "post_only" in (resp.text or ""):
        _post_only_supported = False
        log.warning("post_only NOT supported on V2 - proceeding without it")
        body.pop("post_only", None)
        resp = _write("POST", path, body, private_key, api_key_id)
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code}: {resp.text}", response=resp)
    o = resp.json().get("order") or resp.json()
    if not o.get("order_id"):
        raise RuntimeError(f"order accepted but no order_id in response: {o}")
    return o


def cancel_order(private_key, api_key_id, order_id: str) -> bool:
    for path in (f"/trade-api/v2/portfolio/events/orders/{order_id}",
                 f"/trade-api/v2/portfolio/orders/{order_id}"):
        try:
            resp = _write("DELETE", path, None, private_key, api_key_id)
            if resp.ok or resp.status_code in (404, 409):
                return True
            log.warning("cancel %s via %s -> %s %s", order_id, path,
                        resp.status_code, resp.text[:150])
        except Exception as e:
            log.warning("cancel %s via %s failed: %s", order_id, path, e)
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


def get_balance(private_key, api_key_id):
    data = _get("/trade-api/v2/portfolio/balance", auth=(private_key, api_key_id))
    for key, scale in (("balance", 100.0), ("balance_dollars", 1.0)):
        value = data.get(key)
        if value is not None:
            try:
                return float(value) / scale
            except (TypeError, ValueError):
                continue
    log.warning("balance response unparseable: %s", str(data)[:200])
    return None
