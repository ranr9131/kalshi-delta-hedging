"""
Kalshi WS order book for the cross-market logger — FIXED subscription model.

Why this exists separate from kalshi_orderbook.py: that module adds tickers by
sending a SEPARATE `subscribe` command per ticker. Verified live (ws_diag.py):
Kalshi then streams orderbook_delta for only some of them — the rest get a
snapshot and then nothing, freezing the book at its opening value for the whole
game. Style that works: ONE subscription carrying the FULL ticker list.

So here: we keep one subscription. Adding a game marks the set dirty; a
maintainer thread (debounced) drops the socket; on reconnect we re-subscribe
the COMPLETE current ticker list in a single command. Brief (~1-2s) re-snapshot
per new game; the logger falls back to REST during that window.
"""
from __future__ import annotations

import json, logging, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import websocket
import kalshi_auth

log = logging.getLogger("xmarket_ob")
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
RESUB_DEBOUNCE = 2.0   # s of quiet after a ticker add before we reconnect


def _f(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


@dataclass
class Book:
    ticker: str
    yes_levels: Dict[float, float] = field(default_factory=dict)
    no_levels:  Dict[float, float] = field(default_factory=dict)
    last_update: float = 0.0
    snapshot_seen: bool = False

    def yes_bid(self):
        ps = [p for p, q in list(self.yes_levels.items()) if q > 0]
        return max(ps) if ps else None

    def no_bid(self):
        ps = [p for p, q in list(self.no_levels.items()) if q > 0]
        return max(ps) if ps else None

    def yes_ask(self):
        nb = self.no_bid()
        return None if nb is None else round(1.0 - nb, 4)

    def yes_ask_size(self):
        """Contracts available at the best YES ask = size at the best NO bid."""
        items = list(self.no_levels.items())
        ps = [p for p, q in items if q > 0]
        if not ps:
            return None
        nb = max(ps)
        return self.no_levels.get(nb)

    def age(self):
        return time.time() - self.last_update if self.last_update else float("inf")


_lock = threading.Lock()
_books: Dict[str, Book] = {}
_tickers: List[str] = []
_ws: websocket.WebSocketApp | None = None
_private_key = None
_api_key_id = ""
_msg_seq = 0
_dirty = False
_last_change = 0.0


def get_book(ticker: str):
    with _lock:
        return _books.get(ticker)


def add_ticker(ticker: str) -> None:
    global _dirty, _last_change
    with _lock:
        if ticker in _tickers:
            return
        _tickers.append(ticker)
        _dirty = True
        _last_change = time.time()


def _send_subscribe_full():
    """ONE subscribe command with the entire current ticker list (Style A)."""
    global _msg_seq
    ws = _ws
    with _lock:
        tickers = list(_tickers)
    if ws is None or not tickers:
        return
    _msg_seq += 1
    try:
        ws.send(json.dumps({"id": _msg_seq, "cmd": "subscribe",
                            "params": {"channels": ["orderbook_delta"],
                                       "market_tickers": tickers}}))
        log.info(f"[xob] subscribed {len(tickers)} tickers (single subscription)")
    except Exception as e:
        log.warning(f"[xob] subscribe failed: {e}")


def _apply_snapshot(ticker, yes_arr, no_arr):
    now = time.time()
    with _lock:
        b = _books.get(ticker) or Book(ticker=ticker)
        b.yes_levels = {round(_f(e[0]), 4): _f(e[1]) for e in (yes_arr or []) if _f(e[1]) > 0}
        b.no_levels = {round(_f(e[0]), 4): _f(e[1]) for e in (no_arr or []) if _f(e[1]) > 0}
        b.last_update = now
        b.snapshot_seen = True
        _books[ticker] = b


def _apply_delta(ticker, side, price, delta):
    now = time.time()
    with _lock:
        b = _books.get(ticker) or Book(ticker=ticker)
        levels = b.yes_levels if side == "yes" else b.no_levels
        p = round(price, 4)
        q = levels.get(p, 0.0) + delta
        if q <= 1e-9:
            levels.pop(p, None)
        else:
            levels[p] = q
        b.last_update = now
        _books[ticker] = b


def _on_open(ws):
    global _ws, _dirty
    _ws = ws
    log.info("[xob] connected")
    with _lock:
        for t in _tickers:        # drop stale books; fresh snapshots incoming
            _books.pop(t, None)
        _dirty = False            # on_open subscribes the full list below
    _send_subscribe_full()


def _on_message(ws, raw):
    try:
        m = json.loads(raw)
        t = m.get("type")
        d = m.get("msg", {}) or {}
        if t == "orderbook_snapshot":
            tk = d.get("market_ticker")
            if tk:
                _apply_snapshot(tk, d.get("yes_dollars_fp") or d.get("yes"),
                                d.get("no_dollars_fp") or d.get("no"))
        elif t == "orderbook_delta":
            tk = d.get("market_ticker"); side = d.get("side")
            price = _f(d.get("price_dollars", d.get("price")))
            delta = _f(d.get("delta_fp", d.get("delta")))
            if tk and side in ("yes", "no") and price > 0:
                _apply_delta(tk, side, price, delta)
    except Exception as e:
        log.debug(f"[xob] parse err: {e}")


def _on_close(ws, *a):
    global _ws
    _ws = None
    log.info("[xob] closed")


def _run_loop():
    while True:
        try:
            h = kalshi_auth.make_auth_headers(_private_key, _api_key_id, "GET", "/trade-api/ws/v2")
            headers = [f"{k}: {v}" for k, v in h.items() if k.startswith("KALSHI-ACCESS")]
            ws = websocket.WebSocketApp(WS_URL, header=headers, on_open=_on_open,
                                        on_message=_on_message, on_close=_on_close,
                                        on_error=lambda w, e: log.warning(f"[xob] err {e}"))
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            log.warning(f"[xob] conn err: {e}")
        time.sleep(2)


def _maintainer():
    """When new tickers were added and the set has settled, force a reconnect so
    on_open re-subscribes the COMPLETE list in one command."""
    global _dirty
    while True:
        time.sleep(0.5)
        with _lock:
            dirty = _dirty
            quiesced = (time.time() - _last_change) >= RESUB_DEBOUNCE
            ws = _ws
        if dirty and quiesced and ws is not None:
            log.info("[xob] ticker set changed -> reconnect to re-subscribe full list")
            with _lock:
                _dirty = False
            try:
                ws.close()   # _run_loop reconnects; _on_open subscribes full list
            except Exception:
                pass


def start(private_key, api_key_id: str, initial=None):
    global _private_key, _api_key_id, _tickers
    _private_key = private_key
    _api_key_id = api_key_id
    if initial:
        _tickers = list(initial)
    threading.Thread(target=_run_loop, daemon=True, name="xob-ws").start()
    threading.Thread(target=_maintainer, daemon=True, name="xob-maint").start()
