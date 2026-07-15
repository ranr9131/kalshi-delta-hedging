"""
Real-time BTCUSDT perpetual price from Binance Futures WebSocket.
Perp price typically leads spot by 50-200ms because perp has 10x the volume —
price discovery happens there first.

No authentication required. Runs in a daemon thread.
"""

import json
import threading
import time
import websocket

WS_URL = "wss://fstream.binance.com/ws/btcusdt@aggTrade"

_lock  = threading.Lock()
_price: float | None = None
_last_update: float = 0.0


def get_price() -> float | None:
    with _lock:
        return _price


def get_price_age() -> float:
    with _lock:
        return time.time() - _last_update if _last_update else float("inf")


def _on_message(ws, message):
    global _price, _last_update
    try:
        msg = json.loads(message)
        # aggTrade message: {"e":"aggTrade","s":"BTCUSDT","p":"79123.45",...}
        if msg.get("e") == "aggTrade" and "p" in msg:
            with _lock:
                _price = float(msg["p"])
                _last_update = time.time()
    except Exception:
        pass


def _on_error(ws, error):
    print(f"[binance_feed] WebSocket error: {error}")


def _on_close(ws, code, msg):
    print(f"[binance_feed] WebSocket closed: {code} {msg}")


def start() -> threading.Thread:
    """Start Binance perp WS in a daemon thread."""
    def _run():
        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        ws.run_forever(reconnect=5, ping_interval=20, ping_timeout=10)

    t = threading.Thread(target=_run, daemon=True, name="binance-feed")
    t.start()
    return t
