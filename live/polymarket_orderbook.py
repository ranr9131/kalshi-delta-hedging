"""Small public Polymarket full-depth order-book client.

No credentials and no trading endpoints.  The client keeps absolute price
levels from the public market WebSocket and reconnects with a fresh snapshot
whenever the token set changes.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field

import websocket


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
log = logging.getLogger("polymarket_orderbook")


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class _Book:
    token_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update: float = 0.0
    source_timestamp: float = 0.0
    snapshot_seen: bool = False


@dataclass(frozen=True)
class BookView:
    token_id: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    last_update: float
    source_timestamp: float
    snapshot_seen: bool

    def best_bid(self):
        return self.bids[0][0] if self.bids else None

    def best_ask(self):
        return self.asks[0][0] if self.asks else None

    def midpoint(self):
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def age(self):
        return time.time() - self.last_update if self.last_update else float("inf")


_lock = threading.Lock()
_books: dict[str, _Book] = {}
_tokens: list[str] = []
_ws: websocket.WebSocketApp | None = None
_started = False


def get_book(token_id: str) -> BookView | None:
    with _lock:
        book = _books.get(str(token_id))
        if not book:
            return None
        bids = tuple(sorted(
            ((price, size) for price, size in book.bids.items() if size > 0),
            key=lambda item: -item[0],
        ))
        asks = tuple(sorted(
            ((price, size) for price, size in book.asks.items() if size > 0),
            key=lambda item: item[0],
        ))
        return BookView(
            token_id=book.token_id,
            bids=bids,
            asks=asks,
            last_update=book.last_update,
            source_timestamp=book.source_timestamp,
            snapshot_seen=book.snapshot_seen,
        )


def connected() -> bool:
    return _ws is not None


def set_tokens(token_ids) -> None:
    """Replace the token set and reconnect to obtain one coherent fresh dump."""
    global _tokens
    wanted = sorted({str(token) for token in token_ids if token})
    with _lock:
        if wanted == sorted(_tokens):
            return
        _tokens = wanted
        _books.clear()
        ws = _ws
    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass


def _timestamp_seconds(value) -> float:
    stamp = _float(value)
    if stamp > 10_000_000_000:
        stamp /= 1000.0
    return stamp


def _snapshot(message: dict) -> None:
    token = str(message.get("asset_id") or "")
    if not token:
        return
    now = time.time()
    bids, asks = {}, {}
    for level in message.get("bids") or []:
        price, size = _float(level.get("price")), _float(level.get("size"))
        if 0 < price < 1 and size > 0:
            bids[price] = size
    for level in message.get("asks") or []:
        price, size = _float(level.get("price")), _float(level.get("size"))
        if 0 < price < 1 and size > 0:
            asks[price] = size
    with _lock:
        if token not in _tokens:
            return
        _books[token] = _Book(
            token_id=token,
            bids=bids,
            asks=asks,
            last_update=now,
            source_timestamp=_timestamp_seconds(message.get("timestamp")),
            snapshot_seen=True,
        )


def _price_changes(message: dict) -> None:
    now = time.time()
    source_ts = _timestamp_seconds(message.get("timestamp"))
    for change in message.get("price_changes") or []:
        token = str(change.get("asset_id") or "")
        side = str(change.get("side") or "").upper()
        price, size = _float(change.get("price")), _float(change.get("size"))
        if not token or side not in ("BUY", "SELL") or not 0 < price < 1:
            continue
        with _lock:
            if token not in _tokens:
                continue
            book = _books.get(token)
            if book is None:
                # Refuse delta-only state; wait for a full reconnect snapshot.
                continue
            levels = book.bids if side == "BUY" else book.asks
            if size <= 0:
                levels.pop(price, None)
            else:
                levels[price] = size
            book.last_update = now
            book.source_timestamp = source_ts or book.source_timestamp


def _on_message(_ws_app, raw) -> None:
    if raw in ("PONG", "PING"):
        return
    try:
        payload = json.loads(raw)
        messages = payload if isinstance(payload, list) else [payload]
        for message in messages:
            if not isinstance(message, dict):
                continue
            event = message.get("event_type")
            # Initial dumps currently omit event_type and arrive as an array.
            if event == "book" or (message.get("asset_id") and "bids" in message and "asks" in message):
                _snapshot(message)
            elif event == "price_change":
                _price_changes(message)
    except Exception as exc:
        log.debug("parse error: %s", exc)


def _on_open(ws_app) -> None:
    global _ws
    _ws = ws_app
    with _lock:
        tokens = list(_tokens)
        _books.clear()
    if not tokens:
        log.info("connected with no tokens; waiting for discovery")
        return
    ws_app.send(json.dumps({
        "assets_ids": tokens,
        "type": "market",
        "custom_feature_enabled": True,
    }))
    log.info("connected and subscribed to %d tokens", len(tokens))


def _on_close(_ws_app, code, reason) -> None:
    global _ws
    _ws = None
    log.info("closed code=%s reason=%s", code, reason)


def _on_error(_ws_app, error) -> None:
    log.warning("WebSocket error: %s", error)


def _heartbeat() -> None:
    while True:
        time.sleep(8)
        ws = _ws
        if ws is not None:
            try:
                ws.send("PING")
            except Exception:
                pass


def _run() -> None:
    while True:
        try:
            app = websocket.WebSocketApp(
                WS_URL,
                on_open=_on_open,
                on_message=_on_message,
                on_close=_on_close,
                on_error=_on_error,
            )
            app.run_forever(ping_interval=20, ping_timeout=8)
        except Exception as exc:
            log.warning("connection error: %s", exc)
        time.sleep(1)


def start(initial_tokens=None) -> None:
    global _started
    if initial_tokens is not None:
        set_tokens(initial_tokens)
    if _started:
        return
    _started = True
    threading.Thread(target=_run, daemon=True, name="poly-book-ws").start()
    threading.Thread(target=_heartbeat, daemon=True, name="poly-book-heartbeat").start()
