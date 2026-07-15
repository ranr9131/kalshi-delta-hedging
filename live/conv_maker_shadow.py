"""
Convergence-aware MARKET MAKER shadow sim (paper) — Strategy 1.

We EARN the spread instead of paying it.  We rest a buy a little below our fair
value and a sell a little above it; when an impatient taker crosses to us, we
pocket the gap.  We don't predict price — we collect the spread, defended by the
v3 model's knowledge of how LOCKED-IN the 60s-average settlement already is.

Quoting rule (this is the whole strategy):
  fair  = v3.fair_p(...)                      # true odds, partial-avg aware near expiry
  We post a YES bid only if  best_bid+1 <= fair_c - MARGIN   (buying there still
    beats our fair by >= MARGIN), and a YES ask only if best_ask-1 >= fair_c +
    MARGIN.  So we quote ONLY the side(s) that are profitable vs fair.
  ENDGAME DEFENSE falls out for free: as the average locks, fair -> ~0 or ~1, so
  fair+MARGIN climbs above any sane ask (we stop selling the near-certain side)
  while fair-MARGIN lets us keep buying it cheap.  We never get picked off
  selling YES that's actually ~95%.
  Inventory skew: shift both quotes against our net position so we mean-revert
  to flat instead of accumulating a directional bet.

Honest fills (paper): a resting quote is "sticky" (rests REQUOTE_SEC = our
reaction latency) and only fills off the PUBLIC trade channel — a real taker
trade crossing our price.  A taker hitting our stale price IS the adverse-
selection event; markout-to-settlement measures whether we kept the spread.

Run (collect):  python3 conv_maker_shadow.py
Run (analyze):  python3 conv_maker_shadow.py --report   # P&L by market / horizon
Self-test:      python3 conv_maker_shadow.py --selftest
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
import websocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import kalshi_auth
import kalshi_orderbook
from coinbase_feeds import make_feed
from fair_price_model_v2 import record_price
import fair_price_model_v3 as v3
import asian_pricer

ROOT = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(ROOT, ".env"))
API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")
PRIVATE_KEY = kalshi_auth.load_private_key(env.get("KALSHI_PRIVATE_KEY", "")) \
    if env.get("KALSHI_PRIVATE_KEY") else None
BASE_URL = "https://api.elections.kalshi.com"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

FILL_CSV = os.path.join(ROOT, "conv_maker_fills.csv")

# ── Config ───────────────────────────────────────────────────────────────────
SERIES = os.environ.get("CONV_MM_SERIES",
                        "KXXRP15M,KXSOL15M,KXDOGE15M,KXXRPD,KXSOLD,KXDOGED").split(",")
MARGIN_C       = float(os.environ.get("CONV_MM_MARGIN_C", "3.0"))   # edge each side vs fair
INSIDE_TICKS   = int(os.environ.get("CONV_MM_INSIDE", "1"))         # improve book by N c
MIN_SPREAD_C   = int(os.environ.get("CONV_MM_MIN_SPREAD", "4"))     # need room to rest inside
REQUOTE_SEC    = float(os.environ.get("CONV_MM_REQUOTE_SEC", "1.0"))# reaction latency
MAX_POS        = float(os.environ.get("CONV_MM_MAX_POS", "100"))    # net inventory cap
SKEW_C_PER_POS = float(os.environ.get("CONV_MM_SKEW_C", "0.05"))    # cents skew per contract
QUOTE_SIZE     = float(os.environ.get("CONV_MM_SIZE", "50"))        # our resting size
MAKER_FEE_MULT = float(os.environ.get("CONV_MM_FEE_MULT", "1.75"))  # maker fee ≈ 1.75*p*(1-p)c
TRACK_WITHIN_SEC = float(os.environ.get("CONV_MM_TRACK_SEC", "1200"))  # quote final 20 min
REFRESH_SEC    = float(os.environ.get("CONV_MM_REFRESH_SEC", "30"))
# Final-window regime: inside the last ASIAN_ONLY_SEC before close, the v3
# table fair is not trusted (fills there ran -1.3 to -4.6 c/ct in the 7/13
# decomposition). Quotes are priced off the live banked settlement average
# (asian_pricer), and WITHDRAWN entirely when it can't price responsibly.
ASIAN_ONLY_SEC = float(os.environ.get("CONV_MM_ASIAN_SEC", "120"))
MID_LO, MID_HI = 0.02, 0.98
TICK_SEC       = float(os.environ.get("CONV_MM_TICK_SEC", "0.5"))

COIN_PRODUCT = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
                "XRP": "XRP-USD", "DOGE": "DOGE-USD", "LTC": "LTC-USD",
                "ADA": "ADA-USD", "BNB": "BNB-USD"}
COIN_SYMBOLS = sorted(COIN_PRODUCT, key=len, reverse=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  mm  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("convmm")

_running = True
_tracked: dict = {}
_tracked_lock = threading.Lock()
_feeds: dict = {}
_quotes: dict = {}          # ticker -> {bid, ask, set_ts, bb, ba}
_pos: dict = defaultdict(float)   # ticker -> net YES contracts
_seen_trades: set = set()
_csv_lock = threading.Lock()

FILL_COLS = ["ts", "ticker", "asset", "horizon", "action", "price_c", "size",
             "fair_c", "mid_c", "ttc_sec", "quote_age_s"]


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def coin_of(ticker):
    body = ticker.split("-")[0].upper()
    body = body[2:] if body.startswith("KX") else body
    for s in COIN_SYMBOLS:
        if body.startswith(s):
            return s
    return None


def horizon_of(ticker, strike_type):
    if "15M" in ticker.upper() or (strike_type or "").lower() == "greater_or_equal":
        return "15m"
    return "hourly"


def maker_fee_c(price_dollars):
    p = min(max(price_dollars, 0.0), 1.0)
    return MAKER_FEE_MULT * p * (1.0 - p)


def _spot(asset):
    f = _feeds.get(asset)
    return f.get_price() if f else None


def _get(path, params=None):
    h = kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path) if PRIVATE_KEY else None
    return requests.get(BASE_URL + path, params=params, headers=h, timeout=15)


def refresh_candidates():
    out = {}
    now = datetime.now(timezone.utc)
    for st in SERIES:
        st = st.strip()
        try:
            r = _get("/trade-api/v2/markets", {"series_ticker": st, "status": "open", "limit": 200})
            r.raise_for_status()
            markets = r.json().get("markets", [])
        except Exception as e:
            log.warning(f"refresh {st}: {e}")
            continue
        for m in markets:
            asset = coin_of(m["ticker"])
            if asset is None or not m.get("close_time"):
                continue
            try:
                close_dt = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
            except Exception:
                continue
            ttc = (close_dt - now).total_seconds()
            if not (0 < ttc <= TRACK_WITHIN_SEC):
                continue
            stt = m.get("strike_type") or "greater"
            out[m["ticker"]] = {
                "asset": asset, "close_dt": close_dt, "strike_type": stt,
                "horizon": horizon_of(m["ticker"], stt),
                "floor_strike": _f(m.get("floor_strike")),
                "cap_strike": _f(m.get("cap_strike")),
            }
    return out


def _append(row):
    with _csv_lock:
        new = not os.path.exists(FILL_CSV)
        with open(FILL_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FILL_COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def _fair_c(meta, spot, ttc):
    ml = ttc / 60.0
    return v3.fair_p(spot, ml, meta["asset"], floor_strike=meta.get("floor_strike"),
                     cap_strike=meta.get("cap_strike"),
                     strike_type=meta.get("strike_type", "greater"),
                     use_partial_history=True) * 100.0


_asian: dict = {}   # asset -> AsianPricer (1s buffer per asset)


def _asian_fair_c(meta):
    """Fair (cents) from the banked-average pricer, or None if it refuses
    (stale feed, holes in the banked window, <2 min of vol samples)."""
    ap = _asian.get(meta["asset"])
    if ap is None:
        return None
    close_ts = meta["close_dt"].timestamp()
    st = (meta.get("strike_type") or "greater").lower()
    fl, cp = meta.get("floor_strike"), meta.get("cap_strike")
    if st in ("greater", "greater_or_equal") and fl:
        p = ap.p_up(fl, close_ts)
    elif st in ("less", "less_or_equal") and cp:
        pu = ap.p_up(cp, close_ts)
        p = None if pu is None else 1.0 - pu
    elif st == "between" and fl and cp:
        p1, p2 = ap.p_up(fl, close_ts), ap.p_up(cp, close_ts)
        p = None if (p1 is None or p2 is None) else max(0.001, p1 - p2)
    else:
        return None
    return None if p is None else p * 100.0


# ── Core quoting decision (pure, unit-testable) ────────────────────────────────
def desired_quotes(fair_c, best_bid_c, best_ask_c, net_pos):
    """Return (bid_c, ask_c) we want to rest, or None on a side.

    Quote a side only if pennying the book still beats fair by MARGIN.  Skew both
    quotes against inventory so we revert to flat."""
    skew = SKEW_C_PER_POS * net_pos            # >0 if long YES -> push quotes down
    f = fair_c - skew
    bid = best_bid_c + INSIDE_TICKS
    ask = best_ask_c - INSIDE_TICKS
    want_bid = bid if (bid <= f - MARGIN_C and bid < ask) else None
    want_ask = ask if (ask >= f + MARGIN_C and ask > bid) else None
    # inventory caps: stop adding to a maxed-out side
    if net_pos >= MAX_POS:
        want_bid = None
    if net_pos <= -MAX_POS:
        want_ask = None
    return want_bid, want_ask


# ── Fill detection off the public trade channel ───────────────────────────────
def _on_trade(ticker, yes_price_c, taker_side, size):
    q = _quotes.get(ticker)
    meta = _tracked.get(ticker)
    if not q or not meta:
        return
    ttc = (meta["close_dt"] - datetime.now(timezone.utc)).total_seconds()
    age = time.time() - q["set_ts"]
    action = price = None
    if taker_side == "no" and q["bid"] is not None and q["bid"] >= yes_price_c:
        action, price = "buy_yes", q["bid"]; q["bid"] = None     # filled once
        _pos[ticker] += min(size, QUOTE_SIZE)
    elif taker_side == "yes" and q["ask"] is not None and q["ask"] <= yes_price_c:
        action, price = "sell_yes", q["ask"]; q["ask"] = None
        _pos[ticker] -= min(size, QUOTE_SIZE)
    if not action:
        return
    fair = _fair_c(meta, _spot(meta["asset"]) or 0, ttc) if _spot(meta["asset"]) else None
    _append({
        "ts": round(time.time(), 3), "ticker": ticker, "asset": meta["asset"],
        "horizon": meta["horizon"], "action": action,
        "price_c": round(price, 1), "size": round(min(size, QUOTE_SIZE), 1),
        "fair_c": round(fair, 1) if fair is not None else "",
        "mid_c": round((q["bb"] + q["ba"]) / 2, 1) if q.get("bb") else "",
        "ttc_sec": round(ttc, 1), "quote_age_s": round(age, 2),
    })
    log.info(f"FILL {action:8} {ticker[:28]:28} @{price:.0f}c fair={fair:.0f}c "
             f"pos={_pos[ticker]:.0f} ttc={ttc:.0f}s age={age:.1f}s")


def _trade_ws_loop():
    while _running:
        try:
            hh = kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", "/trade-api/ws/v2")
            hdr = [f"KALSHI-ACCESS-KEY: {hh['KALSHI-ACCESS-KEY']}",
                   f"KALSHI-ACCESS-TIMESTAMP: {hh['KALSHI-ACCESS-TIMESTAMP']}",
                   f"KALSHI-ACCESS-SIGNATURE: {hh['KALSHI-ACCESS-SIGNATURE']}"]

            def on_open(ws):
                with _tracked_lock:
                    tk = list(_tracked.keys())
                if tk:
                    ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                        "params": {"channels": ["trade"], "market_tickers": tk}}))
                ws._sub = set(tk)

            def on_msg(ws, raw):
                try:
                    m = json.loads(raw)
                    if m.get("type") != "trade":
                        return
                    d = m.get("msg", {})
                    tid = d.get("trade_id")
                    if tid is not None:
                        if tid in _seen_trades:
                            return
                        _seen_trades.add(tid)
                    tkr = d.get("ticker") or d.get("market_ticker")
                    if tkr in _quotes:
                        _on_trade(tkr, _f(d.get("yes_price_dollars")) * 100.0,
                                  d.get("taker_side"), _f(d.get("count_fp"), 0))
                except Exception:
                    pass

            def resub():
                while _running:
                    time.sleep(REFRESH_SEC)
                    try:
                        with _tracked_lock:
                            tk = list(_tracked.keys())
                        cur = getattr(ws, "_sub", set())
                        add = [t for t in tk if t not in cur]
                        if add:
                            ws.send(json.dumps({"id": 2, "cmd": "subscribe",
                                                "params": {"channels": ["trade"], "market_tickers": add}}))
                            ws._sub = set(tk)
                    except Exception:
                        return

            ws = websocket.WebSocketApp(WS_URL, header=hdr, on_open=on_open, on_message=on_msg,
                                        on_error=lambda w, e: log.warning(f"[trade] {e}"),
                                        on_close=lambda w, c, m: None)
            threading.Thread(target=resub, daemon=True).start()
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            log.warning(f"[trade] loop: {e}")
        if _running:
            time.sleep(2)


# ── Collection ─────────────────────────────────────────────────────────────────
def _ensure_feeds():
    with _tracked_lock:
        assets = {m["asset"] for m in _tracked.values()}
    for a in assets:
        if a not in _feeds and a in COIN_PRODUCT:
            f = make_feed(COIN_PRODUCT[a]); f.start(); _feeds[a] = f
            log.info(f"[feed] {COIN_PRODUCT[a]}")
        if a in _feeds and a not in _asian:
            ap = asian_pricer.AsianPricer()
            ap.start(lambda a=a: _spot(a))
            _asian[a] = ap
            log.info(f"[asian] 1s sampler started for {a}")


def _refresh_thread():
    while _running:
        cands = refresh_candidates()
        with _tracked_lock:
            _tracked.clear(); _tracked.update(cands)
        _ensure_feeds()
        try:
            kalshi_orderbook.set_tickers(list(cands.keys()))
        except Exception:
            pass
        log.info(f"[refresh] {len(cands)} markets")
        time.sleep(REFRESH_SEC)


def collect():
    log.info(f"series={SERIES} margin={MARGIN_C}c size={QUOTE_SIZE} requote={REQUOTE_SEC}s "
             f"max_pos={MAX_POS} min_spread={MIN_SPREAD_C}c asian_window={ASIAN_ONLY_SEC}s")
    cands = refresh_candidates()
    with _tracked_lock:
        _tracked.update(cands)
    _ensure_feeds()
    kalshi_orderbook.start(PRIVATE_KEY, API_KEY_ID, list(_tracked.keys()))
    threading.Thread(target=_trade_ws_loop, daemon=True).start()
    threading.Thread(target=_refresh_thread, daemon=True).start()
    time.sleep(3)
    log.info(f"making -> {FILL_CSV}")

    while _running:
        t0 = time.time()
        with _tracked_lock:
            items = list(_tracked.items())
        for ticker, meta in items:
            sp = _spot(meta["asset"])
            if sp:
                record_price(meta["asset"], sp)
            ttc = (meta["close_dt"] - datetime.now(timezone.utc)).total_seconds()
            if ttc <= 0 or not sp:
                continue
            book = kalshi_orderbook.get_book(ticker)
            if not book:
                continue
            bb, ba = book.yes_bid(), book.yes_ask()
            if bb is None or ba is None:
                continue
            bb_c, ba_c = round(bb * 100), round(ba * 100)
            mid = (bb_c + ba_c) / 200.0
            if ba_c - bb_c < MIN_SPREAD_C or not (MID_LO <= mid <= MID_HI):
                # not enough room to rest inside, or trivial market -> no quote
                _quotes[ticker] = {"bid": None, "ask": None, "set_ts": time.time(),
                                   "bb": bb_c, "ba": ba_c}
                continue
            if ttc <= ASIAN_ONLY_SEC:
                fair_c = _asian_fair_c(meta)
                if fair_c is None:
                    # final window and the banked average can't be priced
                    # responsibly -> pull quotes rather than rest stale ones
                    _quotes[ticker] = {"bid": None, "ask": None,
                                       "set_ts": time.time(),
                                       "bb": bb_c, "ba": ba_c}
                    continue
            else:
                fair_c = _fair_c(meta, sp, ttc)
            q = _quotes.get(ticker)
            need = (q is None or (time.time() - q["set_ts"]) >= REQUOTE_SEC
                    or (q["bid"] is not None and q["bid"] >= ba_c)
                    or (q["ask"] is not None and q["ask"] <= bb_c))
            if need:
                wb, wa = desired_quotes(fair_c, bb_c, ba_c, _pos[ticker])
                _quotes[ticker] = {"bid": wb, "ask": wa, "set_ts": time.time(),
                                   "bb": bb_c, "ba": ba_c}
            else:
                q["bb"], q["ba"] = bb_c, ba_c
        time.sleep(max(0.0, TICK_SEC - (time.time() - t0)))


# ── Settlement + report ────────────────────────────────────────────────────────
def _settle(ticker, cache):
    """Only TERMINAL results are cached — caching a pre-finalization None with
    a long-lived cache made callers treat the market as never-settling
    (conv_maker_live's daily-loss cap was dead because of this; found 7/14)."""
    if ticker in cache:
        return cache[ticker]
    try:
        r = requests.get(f"{BASE_URL}/trade-api/v2/markets/{ticker}", timeout=10)
        m = r.json().get("market", {}) if r.ok else {}
        res = (m.get("result") or "").lower()
        if res in ("yes", "no"):
            cache[ticker] = 1 if res == "yes" else 0
            return cache[ticker]
        return None
    except Exception:
        return None


def report():
    if not os.path.exists(FILL_CSV):
        print("No fills yet."); return
    rows = list(csv.DictReader(open(FILL_CSV)))
    # group fills per ticker -> reconstruct cash + net position, then settle
    byt = defaultdict(list)
    for r in rows:
        byt[r["ticker"]].append(r)
    cache = {}
    mkts = []   # per-market result
    for ticker, fills in byt.items():
        y = _settle(ticker, cache)
        if y is None:
            continue
        cash = pos = fee = nfill = 0.0
        for f in fills:
            pc = float(f["price_c"]); sz = float(f["size"])
            if f["action"] == "buy_yes":
                cash -= pc * sz; pos += sz
            else:
                cash += pc * sz; pos -= sz
            fee += maker_fee_c(pc / 100.0) * sz
            nfill += 1
        pnl = cash + pos * 100.0 * y - fee   # settle leftover inventory at outcome
        mkts.append({"ticker": ticker, "horizon": fills[0].get("horizon", "?"),
                     "asset": fills[0]["asset"], "pnl_c": pnl, "nfill": nfill,
                     "net_pos": pos})
    if not mkts:
        print("No settled markets with fills yet."); return

    def agg(name, ms):
        if not ms:
            return
        n = len(ms); fills = sum(m["nfill"] for m in ms)
        pnl = sum(m["pnl_c"] for m in ms)
        print(f"  {name:>16}: markets={n:>4} fills={fills:>5}  "
              f"pnl/market={pnl/n:+7.2f}c  pnl/fill={(pnl/fills if fills else 0):+6.2f}c  "
              f"total={pnl:+8.1f}c")

    print(f"\n{len(rows)} fills across {len(byt)} markets; {len(mkts)} settled\n")
    def block(label, ms):
        if not ms:
            return
        print(f"################  {label}  ################")
        agg("ALL", ms)
        for a in sorted(set(m["asset"] for m in ms)):
            agg(a, [m for m in ms if m["asset"] == a])
    block("ALL HORIZONS", mkts)
    block("15-MIN (KX*15M)", [m for m in mkts if m["horizon"] == "15m"])
    block("1-HOUR (KX*D)", [m for m in mkts if m["horizon"] == "hourly"])


def selftest():
    print("normal: fair=58, book 35/78, flat ->", desired_quotes(58, 35, 78, 0),
          "(bid 36 & ask 77: quote both, earn the gap)")
    print("ENDGAME: fair=95 (near-certain YES), someone offers YES cheap, book 85/93 ->",
          desired_quotes(95, 85, 93, 0),
          "(bid 86 = buy cheap YES; ask None = REFUSE to sell near-certain YES cheap)")
    print("inventory skew: fair=58, long 80 YES, book 35/78 ->",
          desired_quotes(58, 35, 78, 80), "(quotes pushed down to shed inventory)")
    print("maker fee at p=0.5:", round(maker_fee_c(0.5), 3), "c ; at 0.9:", round(maker_fee_c(0.9), 3), "c")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.report:
        report(); return
    global _running
    try:
        collect()
    except KeyboardInterrupt:
        _running = False


if __name__ == "__main__":
    main()
