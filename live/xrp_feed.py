"""
Real-time XRP/USD price via Coinbase Exchange WebSocket. Mirrors eth_feed.py
exactly, just subscribes to XRP-USD instead of ETH-USD.
"""

import json
import threading
import time
import websocket

WS_URL = "wss://ws-feed.exchange.coinbase.com"

_lock  = threading.Lock()
_price: float | None = None
_last_update: float = 0.0


def get_price() -> float | None:
    with _lock:
        return _price


def get_price_age() -> float:
    with _lock:
        return time.time() - _last_update if _last_update else float("inf")


def _on_open(ws):
    ws.send(json.dumps({
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": ["XRP-USD"]}],
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
    print(f"[xrp_feed] WebSocket error: {error}")


def _on_close(ws, code, msg):
    print(f"[xrp_feed] WebSocket closed: {code} {msg}")


def start() -> threading.Thread:
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    thread = threading.Thread(
        target=lambda: ws.run_forever(reconnect=2, ping_interval=10, ping_timeout=5),
        daemon=True,
        name="xrp-feed",
    )
    thread.start()
    return thread


if __name__ == "__main__":
    print("Starting XRP feed. Streaming for 30s...")
    start()
    for _ in range(30):
        print(f"  XRP-USD: {get_price()}  (age: {get_price_age():.1f}s)")
        time.sleep(1)
