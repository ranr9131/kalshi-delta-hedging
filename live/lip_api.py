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
            # endpoint retired by Kalshi — this must be LOUD, not a warn-loop:
            # if the place path dies our revenue is fiction; if the cancel
            # path dies our brakes are gone
            raise RuntimeError(f"DEPRECATED endpoint {path}: {resp.status_code} "
                               f"{resp.text[:200]}")
        return resp
    return resp


def _get(path: str, params: dict = None, auth: tuple = None, retries: int = 4):
    """GET with rate limiting and retries. Raises on final failure — NEVER
    returns None (a 429 on the last attempt used to fall off the loop end and
    return None, crashing callers that expect a dict; bug found 7/14)."""
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

_post_only_supported = True   # cleared at runtime if the API rejects it


def place_resting_bid(private_key, api_key_id, ticker: str, side: str,
                      price_cents: int, count: int) -> dict:
    """
    Rest a buy limit order at an explicit price. side: "yes"|"no".

    V2 single-YES-book semantics (legacy /portfolio/orders 410'd — verified
    live 2026-07-14): buy YES at p -> side "bid" price p; buy NO at q ->
    side "ask" at yes-price (100-q). count/price are fixed-point strings.
    The client_order_id encodes our LIP side ("lip-yes-..."/"lip-no-...")
    so sync_from_exchange can reconstruct side from the listing.
    post_only is attempted; if the API rejects it we proceed WITHOUT it and
    accept bounded cross risk (quotes are price-capped cheap bids; a cross
    costs at most the capped per-market loss) — logged loudly once.
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
    }
    if _post_only_supported:
        body["post_only"] = True
    resp = _write("POST", path, body, private_key, api_key_id)
    if resp.status_code == 400 and "post_only" in (resp.text or ""):
        _post_only_supported = False
        log.warning("post_only NOT supported on V2 — proceeding without it; "
                    "cross risk bounded by per-market price caps")
        body.pop("post_only", None)
        resp = _write("POST", path, body, private_key, api_key_id)
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code}: {resp.text}", response=resp)
    o = resp.json().get("order") or resp.json()
    if not o.get("order_id"):
        raise RuntimeError(f"order accepted but no order_id in response: {o}")
    return o


def cancel_order(private_key, api_key_id, order_id: str) -> bool:
    """Dual-path cancel: V2 events path first, legacy fallback — mirrors
    kalshi_trade.cancel_order. The cancel path is the emergency brake; it
    must survive either endpoint being retired."""
    for path in (f"/trade-api/v2/portfolio/events/orders/{order_id}",
                 f"/trade-api/v2/portfolio/orders/{order_id}"):
        try:
            resp = _write("DELETE", path, None, private_key, api_key_id)
            if resp.ok:
                return True
            # already filled/cancelled is fine for our purposes
            if resp.status_code in (404, 409):
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
    """Balance in dollars, or None if unparseable (caller must fail safe —
    the *_fp field quirk means integer fields can come back null)."""
    data = _get("/trade-api/v2/portfolio/balance", auth=(private_key, api_key_id))
    # only fields with VERIFIED units — guessing a scale wrong would
    # over-report balance and bypass the floor
    for key, scale in (("balance", 100.0), ("balance_dollars", 1.0)):
        v = data.get(key)
        if v is not None:
            try:
                return float(v) / scale
            except (TypeError, ValueError):
                continue
    log.warning("balance response unparseable: %s", str(data)[:200])
    return None
