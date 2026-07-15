"""
PAPER shadow maker — hourly crypto ladders (KXBTCD/KXETHD/KXSOLD).
Places NO orders. Simulates the tick-improve deep-favorite strategy and logs
settlement-marked virtual fills under two queue models.

Strategy (from 30d backtest, 2026-07-10):
  - markets: strikes of the active hourly event, either side priced 96.5-99c
  - entry:   6-10 minutes before close (ENTRY_S window)
  - quote:   virtual bid = best bid + IMPROVE (price priority -> front of queue)
  - no re-quoting inside 5 min; one virtual order per (market, side)
  - fill models, driven by the live TRADE channel:
      floor   : filled only when a trade prints at/through our price for
                cumulative size >= our size (we're a new level, so trades AT
                our price are ours; strictly-below always ours)
      ceiling : filled on first trade at/below our price (any size)
  - mark to settlement via REST after close.

Output: ~/research_jobs/shadow_hourly_fills.csv (one row per virtual fill,
floor/ceiling flags), shadow_hourly_quotes.csv (every quote posted).
Run:  cd ~/kalshi-delta-hedging/live && python3 ~/research_jobs/mm_shadow_hourly.py
(needs live/.env for WS auth; uses kalshi_auth from that dir)
"""

import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import requests
import websocket

LIVE_DIR = os.path.expanduser("~/kalshi-delta-hedging/live")
sys.path.insert(0, LIVE_DIR)
import kalshi_auth                                    # noqa: E402
from dotenv import dotenv_values                      # noqa: E402

env = dotenv_values(os.path.join(LIVE_DIR, ".env"))
PRIVATE_KEY = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")

BASE = "https://api.elections.kalshi.com/trade-api/v2"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
SERIES = ("KXBTCD", "KXETHD", "KXSOLD")
ENTRY_S = (360, 600)          # post window: 6-10 min before close
NO_NEW_S = 300                # never post inside 5 min
TIER = (0.965, 0.99)          # deep-favorite band for the EXISTING best bid
IMPROVE = 0.01                # our price = best bid + 1c (price priority)
SIZE = 10.0                   # virtual contracts per quote
OUT = os.path.expanduser("~/research_jobs")

FILLS = os.path.join(OUT, "shadow_hourly_fills.csv")
QUOTES = os.path.join(OUT, "shadow_hourly_quotes.csv")
FILL_FIELDS = ["ts", "ticker", "side", "price", "size", "model", "ttc_s",
               "best_bid_at_post", "trade_px", "trade_sz", "result", "pnl_ct"]
QUOTE_FIELDS = ["ts", "ticker", "side", "price", "size", "ttc_s", "best_bid", "best_ask"]


def csv_append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def rest(path, params=None):
    for a in range(5):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** a)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1 + a)
    return None


# ── shared state ─────────────────────────────────────────────────────────────
lock = threading.Lock()
books = {}        # ticker -> {"yes": {price_str: size}, "no": {...}}
subscribed = set()
orders = {}       # (ticker, side) -> dict(price, size, posted_ts, close_ts,
                  #                        floor_left, floor_done, ceil_done, best_bid)
pending_mark = [] # fills awaiting settlement: rows w/o result
ws_app = None
msg_seq = 0


def ws_send(obj):
    global msg_seq
    if ws_app is None:
        return
    msg_seq += 1
    obj["id"] = msg_seq
    try:
        ws_app.send(json.dumps(obj))
    except Exception:
        pass


def subscribe(tickers):
    fresh = [t for t in tickers if t not in subscribed]
    if not fresh:
        return
    ws_send({"cmd": "subscribe",
             "params": {"channels": ["orderbook_delta", "trade"],
                        "market_tickers": fresh}})
    subscribed.update(fresh)


def best_bid(tk, side):
    b = books.get(tk)
    if not b:
        return None
    if side == "yes":
        lv = b["yes"]
    else:
        lv = b["no"]
    prices = [float(p) for p, s in lv.items() if s > 0.5]
    return max(prices) if prices else None


def on_message(ws, raw):
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        return
    typ = m.get("type")
    d = m.get("msg", {})
    tk = d.get("market_ticker")
    if typ == "orderbook_snapshot" and tk:
        with lock:
            books[tk] = {
                "yes": {str(p): float(s) for p, s in (d.get("yes_dollars_fp") or [])},
                "no": {str(p): float(s) for p, s in (d.get("no_dollars_fp") or [])},
            }
    elif typ == "orderbook_delta" and tk:
        with lock:
            b = books.setdefault(tk, {"yes": {}, "no": {}})
            side, price, delta = d.get("side"), d.get("price_dollars"), d.get("delta_fp")
            if side in ("yes", "no") and price is not None and delta is not None:
                lv = b[side]
                new = lv.get(str(price), 0.0) + float(delta)
                if new > 1e-9:
                    lv[str(price)] = new
                else:
                    lv.pop(str(price), None)
    elif typ == "trade" and tk:
        # public execution: count against our virtual orders
        try:
            yes_px = float(d.get("yes_price_dollars") or d.get("yes_price") or 0)
            cnt = float(d.get("count_fp") or d.get("count") or 0)
        except (TypeError, ValueError):
            return
        if yes_px <= 0 or cnt <= 0:
            return
        now = time.time()
        with lock:
            for (otk, side), o in list(orders.items()):
                if otk != tk or o.get("ceil_done") and o.get("floor_done"):
                    continue
                px = yes_px if side == "yes" else 1.0 - yes_px
                if px > o["price"] + 1e-9:
                    continue
                row_base = {"ts": now, "close_ts": o["close_ts"],
                            "ticker": tk, "side": side,
                            "price": round(o["price"], 4), "size": SIZE,
                            "ttc_s": round(o["close_ts"] - now),
                            "best_bid_at_post": o["best_bid"],
                            "trade_px": px, "trade_sz": cnt}
                if not o.get("ceil_done"):
                    o["ceil_done"] = True
                    pending_mark.append({**row_base, "model": "ceiling"})
                if not o.get("floor_done"):
                    o["floor_left"] -= cnt
                    if o["floor_left"] <= 0:
                        o["floor_done"] = True
                        pending_mark.append({**row_base, "model": "floor"})


def on_open(ws):
    global ws_app
    ws_app = ws
    with lock:
        tks = list(subscribed)
    subscribed.clear()
    subscribe(tks)


def ws_loop():
    global ws_app
    while True:
        try:
            h = kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", "/trade-api/ws/v2")
            headers = [f"{k}: {v}" for k, v in h.items()]
            app = websocket.WebSocketApp(WS_URL, header=headers, on_open=on_open,
                                         on_message=on_message)
            app.run_forever(ping_interval=10, ping_timeout=5)
        except Exception:
            pass
        ws_app = None
        time.sleep(2)


def scanner():
    """Every 20s: find active hourly events closing in 5-11 min; post virtual
    improving quotes on deep-favorite sides in the entry window."""
    while True:
        try:
            now = time.time()
            for series in SERIES:
                d = rest("/markets", {"series_ticker": series, "status": "open", "limit": 200})
                for m in (d or {}).get("markets", []):
                    try:
                        close = datetime.fromisoformat(
                            m["close_time"].replace("Z", "+00:00")).timestamp()
                    except (KeyError, ValueError):
                        continue
                    ttc = close - now
                    if not (NO_NEW_S <= ttc <= ENTRY_S[1] + 60):
                        continue
                    tk = m["ticker"]
                    subscribe([tk])
                    if not (ENTRY_S[0] <= ttc <= ENTRY_S[1]):
                        continue
                    with lock:
                        for side in ("yes", "no"):
                            if (tk, side) in orders:
                                continue
                            bb = best_bid(tk, side)
                            if bb is None or not (TIER[0] <= bb < TIER[1]):
                                continue
                            price = round(bb + IMPROVE, 4)
                            if price >= 0.995:
                                continue
                            # queue ahead at OUR price: size resting at price
                            # (we improved, so normally ~0 -> floor_left = SIZE)
                            lv = books.get(tk, {}).get(side, {})
                            ahead = sum(s for p, s in lv.items()
                                        if abs(float(p) - price) < 5e-4)
                            orders[(tk, side)] = {
                                "price": price, "size": SIZE, "posted_ts": now,
                                "close_ts": close, "best_bid": bb,
                                "floor_left": ahead + SIZE,
                                "floor_done": False, "ceil_done": False,
                            }
                            csv_append(QUOTES, QUOTE_FIELDS, {
                                "ts": now, "ticker": tk, "side": side,
                                "price": price, "size": SIZE, "ttc_s": round(ttc),
                                "best_bid": bb,
                                "best_ask": "",
                            })
                            print(f"[quote] {tk} {side} @ {price} (bb {bb}, ttc {ttc:.0f}s)",
                                  flush=True)
            # settle pending fills for closed markets
            marked = []
            with lock:
                todo = [r for r in pending_mark if r.get("result") == ""
                        or "result" not in r]
            for r in todo:
                if time.time() < r.get("close_ts", 0) + 90:   # settle after close
                    continue
                md = rest(f"/markets/{r['ticker']}")
                mm = (md or {}).get("market", {})
                res = mm.get("result")
                if res in ("yes", "no"):
                    win = (res == "yes") if r["side"] == "yes" else (res == "no")
                    r["result"] = res
                    r["pnl_ct"] = round((1 - r["price"]) if win else -r["price"], 4)
                    csv_append(FILLS, FILL_FIELDS, r)
                    marked.append(r)
                    print(f"[settle] {r['ticker']} {r['side']} {r['model']} "
                          f"pnl {r['pnl_ct']:+.2f}", flush=True)
            with lock:
                for r in marked:
                    pending_mark.remove(r)
                # GC old orders
                for k, o in list(orders.items()):
                    if time.time() > o["close_ts"] + 900:
                        del orders[k]
        except Exception as e:
            print(f"[scanner] error: {e}", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    print(f"PAPER shadow maker starting — improve +{IMPROVE*100:.1f}c, "
          f"entry {ENTRY_S[0]//60}-{ENTRY_S[1]//60}min, tier {TIER}", flush=True)
    threading.Thread(target=ws_loop, daemon=True, name="ws").start()
    scanner()
