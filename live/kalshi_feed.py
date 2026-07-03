"""
Real-time Kalshi market bid/ask via Kalshi WebSocket v2.
Runs in a daemon thread; call set_ticker() when the market window changes.

Usage:
    import kalshi_feed
    kalshi_feed.start(private_key, api_key_id, initial_ticker)

    # in DH loop:
    bid = kalshi_feed.get_bid()
    ask = kalshi_feed.get_ask()
    age = kalshi_feed.get_age()
"""

import json
import logging
import threading
import time

import websocket

import kalshi_auth

log = logging.getLogger("kalshi_feed")

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

_lock          = threading.Lock()
_yes_bid: float | None = None
_yes_ask: float | None = None
_last_update:  float   = 0.0

# Full order-book depth (from the orderbook_delta channel).
# Kalshi expresses the book as resting YES bids and resting NO bids. Since the
# V2 migration (2026) prices arrive as fixed-point dollar STRINGS (sub-penny,
# e.g. "0.0010") and sizes as fixed-point contract strings (fractional, e.g.
# "75.00"). We key levels by the exact price string (lossless) and store sizes
# as floats. A resting NO order at price q is an offer to SELL yes at 1-q, i.e.
# it is the ask side for yes. So:
#   buying YES  -> consume _no_levels  (yes_ask = 1 - no_price)
#   buying NO   -> consume _yes_levels (no_ask  = 1 - yes_price)
_yes_levels:   dict[str, float] = {}   # yes price_dollars str -> size (contracts)
_no_levels:    dict[str, float] = {}   # no  price_dollars str -> size (contracts)
_book_ticker:  str | None = None       # ticker the current book belongs to
_book_update:  float   = 0.0

_ticker:       str | None = None   # currently subscribed market ticker
_ws:           websocket.WebSocketApp | None = None

_private_key   = None
_api_key_id:   str = ""

_msg_seq       = 0   # incrementing id for outbound messages


# ── Public API ────────────────────────────────────────────────────────────────

def get_bid() -> float | None:
    """Latest yes_bid in dollars, or None if not yet received."""
    with _lock:
        return _yes_bid


def get_ask() -> float | None:
    """Latest yes_ask in dollars, or None if not yet received."""
    with _lock:
        return _yes_ask


def get_age() -> float:
    """Seconds since last bid/ask update."""
    with _lock:
        return time.time() - _last_update if _last_update else float("inf")


def get_book() -> dict:
    """Snapshot of the full depth as sorted ask ladders, in dollars.

    Returns {ticker, age, yes_asks, no_asks} where each *_asks is a list of
    (price_dollars, size) sorted best (cheapest) first — i.e. the price you'd
    pay to BUY that side, walking the book. Empty lists if no book yet.
    """
    with _lock:
        yes_asks = sorted((round(1.0 - float(q), 4), s) for q, s in _no_levels.items() if s > 0)
        no_asks  = sorted((round(1.0 - float(p), 4), s) for p, s in _yes_levels.items() if s > 0)
        return {
            "ticker": _book_ticker,
            "age": (time.time() - _book_update) if _book_update else float("inf"),
            "yes_asks": yes_asks,
            "no_asks": no_asks,
        }


def expected_fill(side: str, contracts: float) -> dict:
    """VWAP price (dollars) to BUY `contracts` of `side` by walking the real book.

    Returns {vwap, filled, exhausted, top} where:
      vwap      = size-weighted avg fill price in dollars (None if no book)
      filled    = contracts the book can actually fill (< contracts if thin)
      exhausted = True if the book ran out before `contracts` was reached
      top       = best (touch) price in dollars, or None
    """
    book = get_book()
    ladder = book["yes_asks"] if side == "yes" else book["no_asks"]
    if not ladder:
        return {"vwap": None, "filled": 0.0, "exhausted": True, "top": None}
    remaining = float(contracts)
    cost = 0.0
    got = 0.0
    for price, size in ladder:
        take = min(remaining, float(size))
        cost += take * price
        got += take
        remaining -= take
        if remaining <= 1e-9:
            break
    return {
        "vwap": (cost / got) if got > 0 else None,
        "filled": got,
        "exhausted": remaining > 1e-9,
        "top": ladder[0][0],
    }


def set_ticker(ticker: str):
    """
    Switch the WebSocket subscription to a new market ticker.
    Call this at the start of each 15-minute window.
    """
    global _ticker, _yes_levels, _no_levels, _book_ticker, _book_update
    with _lock:
        changed = ticker != _ticker
        _ticker = ticker
        if changed:
            # Drop the previous market's book; the new snapshot will repopulate.
            _yes_levels = {}
            _no_levels = {}
            _book_ticker = None
            _book_update = 0.0
    if changed:
        _send_subscribe(ticker)
        log.info(f"[kalshi_feed] Subscribed to {ticker}")


# ── Internal ──────────────────────────────────────────────────────────────────

def _send_subscribe(ticker: str):
    global _msg_seq
    ws = _ws
    if ws is None:
        return
    _msg_seq += 1
    try:
        ws.send(json.dumps({
            "id":     _msg_seq,
            "cmd":    "subscribe",
            "params": {
                "channels":       ["ticker", "orderbook_delta"],
                "market_tickers": [ticker],
            },
        }))
    except Exception as e:
        log.warning(f"[kalshi_feed] subscribe send failed: {e}")


def _on_open(ws):
    global _ws
    _ws = ws
    log.info("[kalshi_feed] WebSocket connected")
    with _lock:
        ticker = _ticker
    if ticker:
        _send_subscribe(ticker)



def _on_message(ws, raw):
    global _yes_bid, _yes_ask, _last_update
    global _yes_levels, _no_levels, _book_ticker, _book_update
    try:
        msg      = json.loads(raw)
        msg_type = msg.get("type")
        data     = msg.get("msg", {})

        if msg_type == "ticker":
            bid = data.get("yes_bid_dollars")
            ask = data.get("yes_ask_dollars")
            if bid is not None and ask is not None:
                with _lock:
                    _yes_bid     = float(bid)
                    _yes_ask     = float(ask)
                    _last_update = time.time()

        elif msg_type == "orderbook_snapshot":
            # Full replacement of the book. V2 shape: yes_dollars_fp /
            # no_dollars_fp = [[price_dollars_str, size_fp_str], ...].
            with _lock:
                _book_ticker = data.get("market_ticker")
                _yes_levels = {str(p): float(s)
                               for p, s in (data.get("yes_dollars_fp") or [])}
                _no_levels  = {str(p): float(s)
                               for p, s in (data.get("no_dollars_fp") or [])}
                _book_update = time.time()

        elif msg_type == "orderbook_delta":
            # Incremental change. V2 shape: price_dollars (str), delta_fp
            # (signed str), side. Zero/negative size removes the level.
            with _lock:
                side  = data.get("side")
                price = data.get("price_dollars")
                delta = data.get("delta_fp")
                if side in ("yes", "no") and price is not None and delta is not None:
                    levels = _yes_levels if side == "yes" else _no_levels
                    key = str(price)
                    new = levels.get(key, 0.0) + float(delta)
                    if new > 1e-9:
                        levels[key] = new
                    else:
                        levels.pop(key, None)
                    _book_update = time.time()

        elif msg_type == "error":
            log.warning(f"[kalshi_feed] server error: {data}")

        elif msg_type not in ("subscribed", "heartbeat", None):
            log.debug(f"[kalshi_feed] unhandled type={msg_type!r}")

    except Exception as e:
        log.debug(f"[kalshi_feed] parse error: {e}  raw={raw[:200]}")


def _on_error(ws, error):
    log.warning(f"[kalshi_feed] WebSocket error: {error}")
    print(f"[kalshi_feed] error: {error}")


def _on_close(ws, code, msg):
    global _ws
    _ws = None
    log.info(f"[kalshi_feed] WebSocket closed: {code} {msg}")
    print(f"[kalshi_feed] closed: {code} {msg}")


def _run_loop():
    """
    Reconnect loop. Creates a fresh WebSocketApp with fresh auth headers on
    each attempt (auth timestamp is embedded in the signature).
    """
    while True:
        try:
            # Auth headers must be generated fresh each connection (timestamp in sig).
            # websocket-client wants a list of "Key: Value" strings for custom headers.
            raw_headers = kalshi_auth.make_auth_headers(
                _private_key, _api_key_id, "GET", "/trade-api/ws/v2"
            )
            headers = [
                f"KALSHI-ACCESS-KEY: {raw_headers['KALSHI-ACCESS-KEY']}",
                f"KALSHI-ACCESS-TIMESTAMP: {raw_headers['KALSHI-ACCESS-TIMESTAMP']}",
                f"KALSHI-ACCESS-SIGNATURE: {raw_headers['KALSHI-ACCESS-SIGNATURE']}",
            ]
            ws = websocket.WebSocketApp(
                WS_URL,
                header=headers,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            # Aggressive ping cadence: 10s/5s catches dead connections within ~15s
            # vs default 30s/10s which can let WS drift for >40s before noticing.
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            log.warning(f"[kalshi_feed] connection error: {e}")
        log.info("[kalshi_feed] Reconnecting in 2s...")
        time.sleep(2)


def start(private_key, api_key_id: str, initial_ticker: str | None = None) -> threading.Thread:
    """
    Start the Kalshi WebSocket feed in a daemon thread.
    private_key: loaded RSA key from kalshi_auth.load_private_key()
    api_key_id:  your Kalshi API key ID
    initial_ticker: optional market ticker to subscribe to immediately
    """
    global _private_key, _api_key_id, _ticker
    _private_key = private_key
    _api_key_id  = api_key_id
    if initial_ticker:
        _ticker = initial_ticker

    thread = threading.Thread(target=_run_loop, daemon=True, name="kalshi-feed")
    thread.start()
    return thread


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    from dotenv import dotenv_values
    import kalshi_trade

    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env       = dotenv_values(_env_path)
    key       = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    key_id    = env["KALSHI_API_KEY_ID"]

    print("Fetching current market ticker...")
    market = kalshi_trade.get_open_market()
    if not market:
        print("No open market found.")
        sys.exit(1)

    ticker = market["ticker"]
    rest_bid = float(market["yes_bid_dollars"])
    rest_ask = float(market["yes_ask_dollars"])
    print(f"Ticker: {ticker}")
    print(f"REST bid/ask: {rest_bid:.3f} / {rest_ask:.3f}")
    print()
    print("Starting WebSocket feed. Comparing REST vs WebSocket prices for 60s...")
    print("(If WebSocket prices differ from REST, the REST API is stale)")
    print()

    start(key, key_id, ticker)

    for i in range(60):
        bid = get_bid()
        ask = get_ask()
        age = get_age()
        if bid is not None:
            diff_bid = bid - rest_bid
            diff_ask = ask - rest_ask
            print(f"  WS bid={bid:.3f} ask={ask:.3f}  age={age:.1f}s  "
                  f"vs REST: bid{diff_bid:+.3f} ask{diff_ask:+.3f}")
        else:
            print(f"  {i+1:2d}s: waiting for first WebSocket message...")
        time.sleep(1)
