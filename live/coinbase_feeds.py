"""
Generic Coinbase WebSocket price feeds for any product (BNB-USD, TON-USD, ...).

We already have asset-specific feed modules (btc_feed, eth_feed, sol_feed,
xrp_feed, hype_feed) that all do essentially the same thing — clone of
btc_feed.py with the product_id swapped.  Adding new assets one-file-per
gets old fast.

This module exposes a small factory:
    feed = make_feed("BNB-USD")
    feed.start()
    feed.get_price()      # latest price or None
    feed.get_price_age()  # seconds since last tick
"""
from __future__ import annotations
import json
import threading
import time
import websocket

WS_URL = "wss://ws-feed.exchange.coinbase.com"


class CoinbaseFeed:
    def __init__(self, product_id: str):
        self.product_id = product_id
        self._lock = threading.Lock()
        self._price: float | None = None
        self._last_update: float = 0.0
        self._thread: threading.Thread | None = None

    def get_price(self) -> float | None:
        with self._lock:
            return self._price

    def get_price_age(self) -> float:
        with self._lock:
            return time.time() - self._last_update if self._last_update else float("inf")

    def _on_open(self, ws):
        ws.send(json.dumps({
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": [self.product_id]}],
        }))

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            if msg.get("type") == "ticker" and "price" in msg:
                with self._lock:
                    self._price = float(msg["price"])
                    self._last_update = time.time()
        except Exception:
            pass

    def _on_error(self, ws, error):
        print(f"[coinbase {self.product_id}] WS error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[coinbase {self.product_id}] WS closed: {code} {msg}")

    def start(self) -> threading.Thread:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=lambda: ws.run_forever(reconnect=2, ping_interval=10, ping_timeout=5),
            daemon=True,
            name=f"coinbase-{self.product_id.lower()}",
        )
        self._thread.start()
        return self._thread


def make_feed(product_id: str) -> CoinbaseFeed:
    return CoinbaseFeed(product_id)


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "BNB-USD"
    f = make_feed(pid)
    f.start()
    for _ in range(15):
        print(f"  {pid}: {f.get_price()}  (age {f.get_price_age():.1f}s)")
        time.sleep(1)
