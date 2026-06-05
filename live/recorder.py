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

RETENTION_DAYS = int(os.environ.get("RECORDER_RETENTION_DAYS", "5"))
MAX_FILE_MB    = int(os.environ.get("RECORDER_MAX_FILE_MB",    "300"))
DISK_FREE_MIN_GB = float(os.environ.get("RECORDER_MIN_FREE_GB", "1.5"))


def _disk_free_gb() -> float:
    """Free space on the partition containing ROOT, in gigabytes."""
    try:
        st = os.statvfs(ROOT)
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        return float("inf")


def _rotate_if_oversize(writer: "DailyWriter"):
    """Force-rotate the writer's current file if it exceeds MAX_FILE_MB.
    The rotated file gets a .part-N suffix and is gzipped immediately."""
    if writer.fh is None or writer.cur_date is None:
        return
    path = writer._path(writer.cur_date)
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return
    if size_mb < MAX_FILE_MB:
        return
    # Find next .part-N suffix
    n = 1
    while True:
        rotated = path + f".part-{n}"
        if not os.path.exists(rotated) and not os.path.exists(rotated + ".gz"):
            break
        n += 1
    try:
        with writer.lock:
            writer.fh.close()
            os.rename(path, rotated)
            writer.fh = open(path, "a", buffering=1)
        # gzip the rotated chunk in background
        threading.Thread(target=writer._gzip_file, args=(rotated,),
                         daemon=True).start()
        log.info(f"{writer.stream}: rotated at {size_mb:.0f}MB → "
                 f"{os.path.basename(rotated)}")
    except Exception as e:
        log.warning(f"{writer.stream}: rotation failed: {e}")


def _emergency_purge():
    """When disk is critically low: delete the oldest .gz files first, then
    if still tight, truncate the current jsonl to its last 10MB."""
    free_gb = _disk_free_gb()
    if free_gb >= DISK_FREE_MIN_GB:
        return
    log.warning(f"disk pressure: only {free_gb:.2f}GB free, purging")
    # 1) delete oldest .gz files
    gzips = []
    for day_dir in os.listdir(ROOT):
        day_path = os.path.join(ROOT, day_dir)
        if not os.path.isdir(day_path):
            continue
        for fn in os.listdir(day_path):
            if fn.endswith(".gz"):
                full = os.path.join(day_path, fn)
                gzips.append((os.path.getmtime(full), full))
    gzips.sort()
    for _, p in gzips:
        if _disk_free_gb() >= DISK_FREE_MIN_GB:
            break
        try:
            sz = os.path.getsize(p)
            os.remove(p)
            log.warning(f"purged old gz: {os.path.basename(p)} (-{sz//1024//1024}MB)")
        except Exception:
            pass
    # 2) if still tight, truncate current jsonls
    if _disk_free_gb() < DISK_FREE_MIN_GB:
        for w in (kalshi_writer, crypto_writer):
            if w.fh is None or w.cur_date is None:
                continue
            path = w._path(w.cur_date)
            try:
                with w.lock:
                    w.fh.close()
                    os.remove(path)
                    w.fh = open(path, "a", buffering=1)
                log.warning(f"emergency: truncated {os.path.basename(path)}")
            except Exception:
                pass


def _retention_pass():
    """Run frequently: rotate oversized current files, gzip past-day files,
    delete .gz files older than RETENTION_DAYS, emergency purge if disk low."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1) Size-cap the currently-open files (the recurring failure mode)
    _rotate_if_oversize(kalshi_writer)
    _rotate_if_oversize(crypto_writer)

    # 2) Daily rollover gzip + retention
    try:
        for day_dir in sorted(os.listdir(ROOT)):
            day_path = os.path.join(ROOT, day_dir)
            if not os.path.isdir(day_path):
                continue
            for fn in os.listdir(day_path):
                full = os.path.join(day_path, fn)
                # Gzip non-current-day plain .jsonl files (and any orphan .part-N)
                is_old_jsonl = (fn.endswith(".jsonl") and day_dir != today) or \
                               (".part-" in fn and not fn.endswith(".gz"))
                if is_old_jsonl:
                    gz = full + ".gz"
                    if not os.path.exists(gz):
                        try:
                            with open(full, "rb") as fin, gzip.open(gz, "wb", compresslevel=6) as fout:
                                shutil.copyfileobj(fin, fout, length=1024*1024)
                            os.remove(full)
                            log.info(f"gzipped {day_dir}/{fn}")
                        except Exception as e:
                            log.warning(f"gzip {day_dir}/{fn} failed: {e}")
                # Delete files older than retention
                try:
                    dt = datetime.strptime(day_dir, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - dt).days
                    if age_days > RETENTION_DAYS:
                        os.remove(full)
                        log.info(f"deleted {day_dir}/{fn} ({age_days}d old)")
                except Exception:
                    pass
            try:
                if not os.listdir(day_path):
                    os.rmdir(day_path)
            except Exception:
                pass
    except Exception as e:
        log.warning(f"retention pass error: {e}")

    # 3) Emergency: if disk still tight, purge aggressively
    _emergency_purge()


def _retention_loop():
    """Check every 60s.  Light when nothing to do; aggressive only when needed."""
    while True:
        try:
            _retention_pass()
        except Exception as e:
            log.warning(f"retention loop error: {e}")
        time.sleep(60)


def main():
    os.makedirs(ROOT, exist_ok=True)
    log.info(f"recorder start  root={ROOT}  retention={RETENTION_DAYS}d")
    env = dotenv_values(ENV_PATH)
    priv = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    key_id = env["KALSHI_API_KEY_ID"]
    t1 = threading.Thread(target=_kalshi_run, args=(priv, key_id), daemon=True)
    t2 = threading.Thread(target=_coinbase_run, daemon=True)
    t3 = threading.Thread(target=_retention_loop, daemon=True)
    t1.start(); t2.start(); t3.start()
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
