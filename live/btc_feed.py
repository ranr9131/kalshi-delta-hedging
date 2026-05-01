"""
Real-time BTC/USD price via Coinbase Exchange WebSocket.
No authentication required. Runs in a daemon thread.
"""

import json
import threading
import time
import websocket

WS_URL = "wss://ws-feed.exchange.coinbase.com"

_lock  = threading.Lock()
_price: float | None = None
_last_update: float = 0.0   # unix timestamp of last received price


def get_price() -> float | None:
    """Return latest BTC-USD price, or None if not yet received."""
    with _lock:
        return _price


def get_price_age() -> float:
    """Seconds since last price update. Useful for staleness checks."""
    with _lock:
        return time.time() - _last_update if _last_update else float("inf")


def _on_open(ws):
    ws.send(json.dumps({
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}],
    }))


def _on_message(ws, message):
    global _price, _last_update
    try:
        msg = json.loads(message)
        if msg.get("type") == "ticker" and "price" in msg:
            with _lock:
                _price = float(msg["price"])
                _last_update = time.time()
    except Exception:
        pass


def _on_error(ws, error):
    print(f"[btc_feed] WebSocket error: {error}")


def _on_close(ws, code, msg):
    print(f"[btc_feed] WebSocket closed: {code} {msg}")


def start() -> threading.Thread:
    """
    Start the Coinbase WebSocket feed in a daemon thread.
    reconnect=5: on unexpected close, wait 5s and reconnect automatically.
    """
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    thread = threading.Thread(
        target=lambda: ws.run_forever(reconnect=5),
        daemon=True,
        name="btc-feed",
    )
    thread.start()
    return thread


if __name__ == "__main__":
    print("Starting BTC feed. Streaming prices for 30s (Ctrl+C to stop)...")
    start()
    for _ in range(30):
        price = get_price()
        age   = get_price_age()
        print(f"  BTC-USD: {price}  (age: {age:.1f}s)")
        time.sleep(1)
