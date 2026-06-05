"""
Full-depth Kalshi orderbook via the v2 WebSocket `orderbook_delta` channel.

Kalshi v2 wire format (confirmed live 2026-06-02):
  orderbook_snapshot.msg.yes_dollars_fp = [["0.4200", "237.00"], ...]
  orderbook_snapshot.msg.no_dollars_fp  = [["0.0010", "228576.00"], ...]
  orderbook_delta.msg = {
      market_ticker, side: "yes"|"no",
      price_dollars: "0.9830", delta_fp: "-91.00",
  }

Prices are dollar-precision floats with 4 decimal places (so 0.0001 = 0.01¢
sub-cent resolution).  Quantities are floats (fractional contracts allowed).
We store {price_dollars: qty} per side as float→float dicts.

A YES bid at $0.42 means someone wants to BUY YES at 42¢, equivalently
willing to SELL NO at (1.00 − 0.42) = 58¢.

Usage:
    import kalshi_orderbook as ob
    ob.start(private_key, api_key_id, ["KXBTC-..", "KXETH-.."])
    book = ob.get_book("KXBTC-..")
    book.yes_ask()           # float dollars, lowest price to BUY YES
    book.yes_asks_sorted()   # [(price_dollars, qty), ...] ascending
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import websocket

import kalshi_auth

log = logging.getLogger("kalshi_orderbook")

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"


def _f(v) -> float:
    """Kalshi sends prices/qtys as either strings or numbers — normalise."""
    try:
        return float(v)
    except Exception:
        return 0.0


# ── Per-market book ─────────────────────────────────────────────────────────

@dataclass
class Book:
    ticker: str
    # price_dollars → qty.  yes_levels = resting BUY-YES quotes;
    # no_levels = resting BUY-NO quotes.
    yes_levels: Dict[float, float] = field(default_factory=dict)
    no_levels:  Dict[float, float] = field(default_factory=dict)
    # price_dollars → unix-ts of the last delta that touched this level.
    # Used by the sniper to gate on "level has been resting > N sec" → stale.
    yes_level_ts: Dict[float, float] = field(default_factory=dict)
    no_level_ts:  Dict[float, float] = field(default_factory=dict)
    last_update: float = 0.0
    snapshot_seen: bool = False
    snapshot_seq: int = 0
    last_seq:    int = 0

    def yes_ask_age(self) -> float:
        """Seconds since the best YES ask (= 1.0 − max no_bid) was last touched."""
        items = list(self.no_levels.items())
        ps = [p for p, q in items if q > 0]
        if not ps: return float("inf")
        best_no_bid = max(ps)
        ts = self.no_level_ts.get(best_no_bid, 0.0)
        return time.time() - ts if ts else float("inf")

    def no_ask_age(self) -> float:
        items = list(self.yes_levels.items())
        ps = [p for p, q in items if q > 0]
        if not ps: return float("inf")
        best_yes_bid = max(ps)
        ts = self.yes_level_ts.get(best_yes_bid, 0.0)
        return time.time() - ts if ts else float("inf")

    # ── Best-of-book ────────────────────────────────────────────────────
    # Each method snapshots the dict via list() at the start.  list(d.items())
    # is a single C-level call protected by the GIL → atomic against the WS
    # thread's _apply_delta mutations.  Without this, we hit
    # "RuntimeError: dictionary changed size during iteration" sporadically.
    def yes_bid(self) -> float | None:
        items = list(self.yes_levels.items())
        ps = [p for p, q in items if q > 0]
        return max(ps) if ps else None

    def no_bid(self) -> float | None:
        items = list(self.no_levels.items())
        ps = [p for p, q in items if q > 0]
        return max(ps) if ps else None

    def yes_ask(self) -> float | None:
        """Lowest price you can BUY YES at = 1.0 − max(no_bid)."""
        nb = self.no_bid()
        return None if nb is None else round(1.0 - nb, 4)

    def no_ask(self) -> float | None:
        yb = self.yes_bid()
        return None if yb is None else round(1.0 - yb, 4)

    # ── Sorted ladders ──────────────────────────────────────────────────
    def yes_asks_sorted(self) -> List[Tuple[float, float]]:
        items = list(self.no_levels.items())
        out = [(round(1.0 - p, 4), q) for p, q in items if q > 0]
        out.sort(key=lambda t: t[0])
        return out

    def no_asks_sorted(self) -> List[Tuple[float, float]]:
        items = list(self.yes_levels.items())
        out = [(round(1.0 - p, 4), q) for p, q in items if q > 0]
        out.sort(key=lambda t: t[0])
        return out

    def yes_bids_sorted(self) -> List[Tuple[float, float]]:
        items = list(self.yes_levels.items())
        out = [(p, q) for p, q in items if q > 0]
        out.sort(key=lambda t: -t[0])
        return out

    def no_bids_sorted(self) -> List[Tuple[float, float]]:
        items = list(self.no_levels.items())
        out = [(p, q) for p, q in items if q > 0]
        out.sort(key=lambda t: -t[0])
        return out

    def age(self) -> float:
        return time.time() - self.last_update if self.last_update else float("inf")


# ── Module state ────────────────────────────────────────────────────────────

_lock   = threading.Lock()
_books: Dict[str, Book] = {}
_tickers: List[str] = []
_subscribed: set[str] = set()

_ws: websocket.WebSocketApp | None = None
_private_key = None
_api_key_id  = ""
_msg_seq = 0


# ── Public API ──────────────────────────────────────────────────────────────

def get_book(ticker: str) -> Book | None:
    with _lock:
        return _books.get(ticker)


def all_books() -> Dict[str, Book]:
    with _lock:
        return dict(_books)


def set_tickers(tickers: List[str]) -> None:
    """Replace the subscription set; deltas for dropped tickers are ignored."""
    global _tickers
    with _lock:
        old = set(_tickers)
        new = set(tickers)
        _tickers = list(tickers)
    add = new - old
    drop = old - new
    if drop:
        with _lock:
            for t in drop:
                _books.pop(t, None)
                _subscribed.discard(t)
    if add:
        _send_subscribe(list(add))


def add_ticker(ticker: str) -> None:
    with _lock:
        if ticker in _tickers:
            return
        _tickers.append(ticker)
    _send_subscribe([ticker])


# ── WS internals ────────────────────────────────────────────────────────────

def _send_subscribe(tickers: List[str]) -> None:
    global _msg_seq
    ws = _ws
    if ws is None or not tickers:
        return
    _msg_seq += 1
    try:
        ws.send(json.dumps({
            "id":   _msg_seq,
            "cmd":  "subscribe",
            "params": {
                "channels":       ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }))
        log.info(f"[ob] subscribed orderbook_delta: {tickers}")
    except Exception as e:
        log.warning(f"[ob] subscribe send failed: {e}")


def _on_open(ws):
    global _ws
    _ws = ws
    log.info("[ob] WebSocket connected")
    with _lock:
        tickers = list(_tickers)
        _subscribed.clear()
        # Drop stale books so we don't keep applying deltas to old state
        # while waiting for the new snapshot.
        for t in tickers:
            _books.pop(t, None)
    if tickers:
        _send_subscribe(tickers)


def _apply_snapshot(ticker: str, yes_arr, no_arr, seq: int):
    now = time.time()
    with _lock:
        b = _books.get(ticker)
        if b is None:
            b = Book(ticker=ticker)
            _books[ticker] = b
        b.yes_levels = {}
        b.yes_level_ts = {}
        for entry in (yes_arr or []):
            p, q = _f(entry[0]), _f(entry[1])
            if q > 0:
                pr = round(p, 4)
                b.yes_levels[pr] = q
                b.yes_level_ts[pr] = now
        b.no_levels = {}
        b.no_level_ts = {}
        for entry in (no_arr or []):
            p, q = _f(entry[0]), _f(entry[1])
            if q > 0:
                pr = round(p, 4)
                b.no_levels[pr] = q
                b.no_level_ts[pr] = now
        b.last_update   = now
        b.snapshot_seen = True
        b.snapshot_seq  = seq
        b.last_seq      = seq
        _subscribed.add(ticker)


def _apply_delta(ticker: str, side: str, price: float, delta: float, seq: int):
    now = time.time()
    with _lock:
        b = _books.get(ticker)
        if b is None:
            b = Book(ticker=ticker)
            _books[ticker] = b
        p = round(price, 4)
        if side == "yes":
            levels, lvl_ts = b.yes_levels, b.yes_level_ts
        else:
            levels, lvl_ts = b.no_levels, b.no_level_ts
        new_q = levels.get(p, 0.0) + delta
        if new_q <= 1e-9:
            levels.pop(p, None)
            lvl_ts.pop(p, None)
        else:
            levels[p] = new_q
            lvl_ts[p] = now
        b.last_update = now
        b.last_seq    = seq


def _on_message(ws, raw):
    try:
        msg = json.loads(raw)
        t   = msg.get("type")
        d   = msg.get("msg", {}) or {}
        seq = int(msg.get("seq", 0) or 0)

        if t == "orderbook_snapshot":
            ticker = d.get("market_ticker")
            if not ticker:
                return
            _apply_snapshot(ticker,
                            d.get("yes_dollars_fp") or d.get("yes"),
                            d.get("no_dollars_fp")  or d.get("no"),
                            seq)

        elif t == "orderbook_delta":
            ticker = d.get("market_ticker")
            side   = d.get("side")
            # Kalshi sends price as `price_dollars` (string) and delta as `delta_fp`.
            price  = _f(d.get("price_dollars", d.get("price")))
            delta  = _f(d.get("delta_fp",       d.get("delta")))
            if not (ticker and side in ("yes", "no")) or price <= 0:
                return
            _apply_delta(ticker, side, price, delta, seq)

        elif t == "error":
            log.warning(f"[ob] server error: {d}")
        elif t == "subscribed":
            log.info(f"[ob] sub ack: {d}")
        elif t in ("ok", "heartbeat", None):
            pass
        else:
            log.debug(f"[ob] unhandled type={t!r}: {str(d)[:200]}")

    except Exception as e:
        log.debug(f"[ob] parse err: {e}  raw={raw[:200]}")


def _on_error(ws, error):
    log.warning(f"[ob] WS error: {error}")


def _on_close(ws, code, msg):
    global _ws
    _ws = None
    log.info(f"[ob] WS closed: {code} {msg}")


def _run_loop():
    while True:
        try:
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
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            log.warning(f"[ob] connection err: {e}")
        log.info("[ob] reconnecting in 2s")
        time.sleep(2)


def start(private_key, api_key_id: str, initial_tickers: List[str] | None = None) -> threading.Thread:
    global _private_key, _api_key_id, _tickers
    _private_key = private_key
    _api_key_id  = api_key_id
    if initial_tickers:
        _tickers = list(initial_tickers)
    t = threading.Thread(target=_run_loop, daemon=True, name="kalshi-orderbook")
    t.start()
    return t


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    from dotenv import dotenv_values
    import kalshi_trade

    logging.basicConfig(level=logging.INFO)

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env      = dotenv_values(env_path)
    key      = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    key_id   = env["KALSHI_API_KEY_ID"]

    mk = kalshi_trade.get_open_market()
    if not mk:
        print("No open market.")
        sys.exit(1)
    ticker = mk["ticker"]
    print(f"Subscribing to {ticker}")
    start(key, key_id, [ticker])

    for i in range(20):
        time.sleep(1)
        b = get_book(ticker)
        if b is None or not b.snapshot_seen:
            print(f"  {i+1:2d}s: waiting for snapshot...")
            continue
        ybids = [(f"${p:.4f}", q) for p, q in b.yes_bids_sorted()[:3]]
        yasks = [(f"${p:.4f}", q) for p, q in b.yes_asks_sorted()[:3]]
        yb = b.yes_bid(); ya = b.yes_ask()
        print(f"  {i+1:2d}s  TOP yes_bid=${yb}/yes_ask=${ya}  "
              f"depth_yes_bids={ybids}  depth_yes_asks={yasks}  age={b.age():.1f}s")
