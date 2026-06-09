"""Diagnose Kalshi WS freeze: do orderbook deltas actually stream?
Tests two subscription styles against live markets for ~35s each and counts
snapshot vs delta messages PER ticker.

  style A: ONE subscribe command with the full ticker list  (correct per API)
  style B: N separate subscribe commands, one per ticker     (what the logger did)
"""
import json, time, threading, collections, sys
import websocket, requests
import kalshi_auth
from dotenv import dotenv_values

env = dotenv_values(".env")
KEY = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
KID = env["KALSHI_API_KEY_ID"]
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# grab a few currently-open MLB markets
ms = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                  params={"series_ticker": "KXMLBGAME", "status": "open", "limit": 8},
                  timeout=10).json().get("markets", [])
TICKERS = [m["ticker"] for m in ms[:6]]
print("testing tickers:", TICKERS)


def run(style, secs=35):
    snaps = collections.Counter()
    deltas = collections.Counter()
    done = threading.Event()

    def on_open(ws):
        if style == "A":
            ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                "params": {"channels": ["orderbook_delta"],
                                           "market_tickers": TICKERS}}))
        else:
            for i, t in enumerate(TICKERS):
                ws.send(json.dumps({"id": i + 1, "cmd": "subscribe",
                                    "params": {"channels": ["orderbook_delta"],
                                               "market_tickers": [t]}}))

    def on_message(ws, raw):
        m = json.loads(raw)
        t = m.get("type")
        d = m.get("msg", {}) or {}
        tk = d.get("market_ticker", "?")
        if t == "orderbook_snapshot":
            snaps[tk] += 1
        elif t == "orderbook_delta":
            deltas[tk] += 1

    hdr = kalshi_auth.make_auth_headers(KEY, KID, "GET", "/trade-api/ws/v2")
    headers = [f"{k}: {v}" for k, v in hdr.items()
               if k.startswith("KALSHI-ACCESS")]
    ws = websocket.WebSocketApp(WS_URL, header=headers,
                                on_open=on_open, on_message=on_message)
    th = threading.Thread(target=ws.run_forever,
                          kwargs={"ping_interval": 10, "ping_timeout": 5}, daemon=True)
    th.start()
    time.sleep(secs)
    ws.close()

    print(f"\n=== style {style}: {secs}s ===")
    print(f"  {'ticker':40} {'snaps':>6} {'deltas':>7}")
    for t in TICKERS:
        print(f"  {t:40} {snaps[t]:>6} {deltas[t]:>7}")
    streamed = sum(1 for t in TICKERS if deltas[t] > 0)
    print(f"  -> {streamed}/{len(TICKERS)} tickers received deltas; "
          f"total deltas={sum(deltas.values())}")


def run_C(secs=40):
    """style C: subscribe 2 tickers, then update_subscription add_markets the
    rest. Confirms we can grow ONE subscription without dropping deltas."""
    snaps = collections.Counter(); deltas = collections.Counter()
    sid = [None]
    first2, rest = TICKERS[:2], TICKERS[2:]

    def on_open(ws):
        ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                            "params": {"channels": ["orderbook_delta"],
                                       "market_tickers": first2}}))

    def on_message(ws, raw):
        m = json.loads(raw); t = m.get("type"); d = m.get("msg", {}) or {}
        if t == "subscribed":
            sid[0] = d.get("sid")
            # now grow the subscription with the remaining tickers
            ws.send(json.dumps({"id": 99, "cmd": "update_subscription",
                                "params": {"sids": [sid[0]],
                                           "market_tickers": rest,
                                           "action": "add_markets"}}))
        elif t == "orderbook_snapshot":
            snaps[d.get("market_ticker", "?")] += 1
        elif t == "orderbook_delta":
            deltas[d.get("market_ticker", "?")] += 1
        elif t == "error":
            print("  ERROR:", d)

    hdr = kalshi_auth.make_auth_headers(KEY, KID, "GET", "/trade-api/ws/v2")
    headers = [f"{k}: {v}" for k, v in hdr.items() if k.startswith("KALSHI-ACCESS")]
    ws = websocket.WebSocketApp(WS_URL, header=headers, on_open=on_open, on_message=on_message)
    threading.Thread(target=ws.run_forever,
                     kwargs={"ping_interval": 10, "ping_timeout": 5}, daemon=True).start()
    time.sleep(secs); ws.close()
    print(f"\n=== style C: subscribe 2 + update_subscription add {len(rest)}  ({secs}s) ===")
    print(f"  sid captured: {sid[0]}")
    for t in TICKERS:
        tag = "(added)" if t in rest else "(initial)"
        print(f"  {t:40} snaps={snaps[t]} deltas={deltas[t]} {tag}")
    added_streaming = sum(1 for t in rest if deltas[t] > 0)
    print(f"  -> added tickers streaming deltas: {added_streaming}/{len(rest)}")


run("A")
run_C()
