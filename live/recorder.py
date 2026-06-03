"""
Passive recorder: dump every Kalshi orderbook_delta + crypto WS message to
daily JSON-lines files so we can build a real backtest corpus.

Output layout:
  recordings/YYYY-MM-DD/kalshi.jsonl    (gzipped after rollover)
  recordings/YYYY-MM-DD/crypto.jsonl    (gzipped after rollover)

Each line is JSON with a `_t` field (wall-clock millis when received) plus
the raw message payload.  Daily rollover + gzip on rollover keeps disk use
in check (~300 MB/day uncompressed → ~50-80 MB gzipped).

Run as a long-lived systemd service.  Auto-reconnects on any WS drop.
"""
from __future__ import annotations
import gzip
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone

import requests
import websocket
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_auth

ROOT = os.environ.get("REC_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "recordings"))
ROOT = os.path.abspath(ROOT)

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

KALSHI_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
KALSHI_REST = "https://api.elections.kalshi.com"

SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"]
CRYPTO_PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
MARKET_REFRESH_SEC = 60.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  recorder  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recorder")


# ── Daily-rotating writer ──────────────────────────────────────────────────

class DailyWriter:
    """Append-only writer that auto-rotates to a new file each UTC day and
    gzips the previous day's file so old data stays small on disk."""

    def __init__(self, stream_name: str):
        self.stream = stream_name
        self.cur_date: str | None = None
        self.fh = None
        self.lock = threading.Lock()

    def _path(self, date_str: str) -> str:
        d = os.path.join(ROOT, date_str)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.stream}.jsonl")

    def _maybe_rollover(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self.cur_date and self.fh is not None:
            return
        # Close old
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
            self.fh = None
            if self.cur_date and self.cur_date != today:
                # Gzip yesterday's file in the background
                old = self._path(self.cur_date)
                threading.Thread(target=self._gzip_file, args=(old,),
                                 daemon=True).start()
        # Open new
        path = self._path(today)
        self.fh = open(path, "a", buffering=1)
        self.cur_date = today
        log.info(f"{self.stream}: writing to {path}")

    def _gzip_file(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        gz = path + ".gz"
        try:
            with open(path, "rb") as fin, gzip.open(gz, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            os.remove(path)
            log.info(f"{self.stream}: gzipped {os.path.basename(path)} → "
                     f"{os.path.getsize(gz)//1024} KB")
        except Exception as e:
            log.warning(f"gzip failed for {path}: {e}")

    def write(self, payload: dict):
        with self.lock:
            self._maybe_rollover()
            try:
                self.fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
            except Exception as e:
                log.warning(f"{self.stream}: write failed: {e}")


kalshi_writer = DailyWriter("kalshi")
crypto_writer = DailyWriter("crypto")


# ── Currently-open Kalshi markets (refresh every minute) ──────────────────

_open_tickers: list[str] = []
_open_tickers_lock = threading.Lock()


def refresh_open_markets() -> list[str]:
    out: list[str] = []
    for series in SERIES:
        try:
            r = requests.get(
                f"{KALSHI_REST}/trade-api/v2/markets",
                params={"series_ticker": series, "status": "open", "limit": 50},
                timeout=10,
            )
            r.raise_for_status()
            for mk in r.json().get("markets", []):
                t = mk.get("ticker")
                if t:
                    out.append(t)
        except Exception as e:
            log.warning(f"market fetch failed for {series}: {e}")
    return out


# ── Kalshi WS ──────────────────────────────────────────────────────────────

def _kalshi_subscribe(ws, tickers: list[str]):
    if not tickers:
        return
    try:
        ws.send(json.dumps({
            "id": int(time.time()),
            "cmd": "subscribe",
            "params": {
                "channels":       ["orderbook_delta", "ticker"],
                "market_tickers": tickers,
            },
        }))
        log.info(f"kalshi: subscribed to {len(tickers)} tickers")
    except Exception as e:
        log.warning(f"kalshi subscribe failed: {e}")


def _kalshi_run(priv_key, key_id: str):
    while True:
        # Track which subscription set we're currently on so we can
        # re-subscribe when a new 15M window opens.
        subscribed: set[str] = set()

        def on_open(ws):
            log.info("kalshi WS connected")
            tickers = refresh_open_markets()
            with _open_tickers_lock:
                _open_tickers[:] = tickers
            subscribed.clear()
            subscribed.update(tickers)
            _kalshi_subscribe(ws, tickers)

        def on_message(ws, raw):
            try:
                msg = json.loads(raw)
                msg["_t"] = int(time.time() * 1000)
                kalshi_writer.write(msg)
            except Exception:
                pass

        def on_error(ws, error):
            log.warning(f"kalshi WS err: {error}")

        def on_close(ws, code, msg):
            log.info(f"kalshi WS closed: {code} {msg}")

        try:
            raw_headers = kalshi_auth.make_auth_headers(
                priv_key, key_id, "GET", "/trade-api/ws/v2"
            )
            headers = [
                f"KALSHI-ACCESS-KEY: {raw_headers['KALSHI-ACCESS-KEY']}",
                f"KALSHI-ACCESS-TIMESTAMP: {raw_headers['KALSHI-ACCESS-TIMESTAMP']}",
                f"KALSHI-ACCESS-SIGNATURE: {raw_headers['KALSHI-ACCESS-SIGNATURE']}",
            ]
            ws = websocket.WebSocketApp(
                KALSHI_WS, header=headers,
                on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close,
            )

            # Background thread: re-subscribe when the open-markets set changes
            stop_flag = threading.Event()
            def watcher():
                while not stop_flag.is_set():
                    time.sleep(MARKET_REFRESH_SEC)
                    fresh = refresh_open_markets()
                    new_set = set(fresh)
                    with _open_tickers_lock:
                        _open_tickers[:] = fresh
                    added = new_set - subscribed
                    if added:
                        _kalshi_subscribe(ws, list(added))
                        subscribed.update(added)
            t = threading.Thread(target=watcher, daemon=True)
            t.start()

            ws.run_forever(ping_interval=10, ping_timeout=5)
            stop_flag.set()
        except Exception as e:
            log.warning(f"kalshi loop err: {e}")
        log.info("kalshi reconnecting in 3s")
        time.sleep(3)


# ── Coinbase WS ────────────────────────────────────────────────────────────

def _coinbase_run():
    while True:
        try:
            def on_open(ws):
                log.info("coinbase WS connected")
                ws.send(json.dumps({
                    "type": "subscribe",
                    "channels": [{"name": "ticker", "product_ids": CRYPTO_PRODUCTS}],
                }))

            def on_message(ws, raw):
                try:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        return
                    crypto_writer.write({
                        "_t": int(time.time() * 1000),
                        "product_id": msg.get("product_id"),
                        "price":      msg.get("price"),
                        "time":       msg.get("time"),
                        "best_bid":   msg.get("best_bid"),
                        "best_ask":   msg.get("best_ask"),
                    })
                except Exception:
                    pass

            def on_error(ws, error):
                log.warning(f"coinbase WS err: {error}")

            def on_close(ws, code, msg):
                log.info(f"coinbase WS closed: {code} {msg}")

            ws = websocket.WebSocketApp(
                COINBASE_WS,
                on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close,
            )
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            log.warning(f"coinbase loop err: {e}")
        log.info("coinbase reconnecting in 3s")
        time.sleep(3)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(ROOT, exist_ok=True)
    log.info(f"recorder start  root={ROOT}")
    env = dotenv_values(ENV_PATH)
    priv = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    key_id = env["KALSHI_API_KEY_ID"]
    t1 = threading.Thread(target=_kalshi_run, args=(priv, key_id), daemon=True)
    t2 = threading.Thread(target=_coinbase_run, daemon=True)
    t1.start(); t2.start()
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
