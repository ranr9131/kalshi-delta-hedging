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


def set_ticker(ticker: str):
    """
    Switch the WebSocket subscription to a new market ticker.
    Call this at the start of each 15-minute window.
    """
    global _ticker
    with _lock:
        changed = ticker != _ticker
        _ticker = ticker
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
                "channels":       ["ticker"],
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
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            log.warning(f"[kalshi_feed] connection error: {e}")
        log.info("[kalshi_feed] Reconnecting in 5s...")
        time.sleep(5)


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
