"""
Passive-fill shadow logger — Step 2 of the thin-coin MM exploration.
(See STRATEGY_LOG.md §2 "Revisit (6/06)".)

GOAL: measure whether a passive maker quote on low-volume coin markets actually
keeps any of its spread once adverse selection is accounted for — WITHOUT
risking capital.

How it works
------------
For a set of candidate markets (default: XRP daily + DOGE), we:

  1. Track the live book via kalshi_orderbook (WS orderbook_delta).
  2. Track the model fair value via fair_price_model_v2 (spot feed + realized σ).
  3. Simulate a *resting* maker quote INSIDE_TICKS inside each side
     (bid = best_bid+1c, ask = best_ask-1c).
  4. Listen to the public `trade` channel. When a real taker trade would have
     hit our resting quote, we record a simulated FILL.
  5. Log a continuous snapshot stream (book + fair) so markout / adverse
     selection can be computed at ANY horizon offline (`--report`).

HONESTY KNOB — sticky quotes with reaction latency
--------------------------------------------------
Our quote is NOT recomputed from the book on every tick. If it were, we'd
"cancel" stale quotes before a taker could hit them and massively understate
adverse selection. Instead the quote rests until one of:
  - it gets filled,
  - REQUOTE_SEC elapses (our reaction latency — the key parameter),
  - the book moves enough that the quote is no longer strictly inside.
A taker hitting our stale resting price IS the adverse-selection event, and the
markout (how far fair moved against us afterward) measures its cost.
Smaller REQUOTE_SEC = faster MM = less adverse selection. Sweep it to see how
fast you'd need to be.

Fill semantics (from the trade channel, confirmed live)
-------------------------------------------------------
A trade has `taker_side` ∈ {yes, no} and `yes_price_dollars`:
  - taker_side == "no"  → taker SOLD yes (hit a bid). Our resting YES bid B
    fills (we BOUGHT yes) if B is active and B >= traded yes_price.
  - taker_side == "yes" → taker BOUGHT yes (lifted an ask). Our resting YES ask
    A fills (we SOLD yes) if A is active and A <= traded yes_price.
Because our quote is one tick inside, it has price priority, so any same-side
aggression while we're resting fills us.

Run (collect):  python3 mm_shadow_logger.py
Run (analyze):  python3 mm_shadow_logger.py --report
Options:        --series KXXRPD,KXDOGE  --requote-sec 1.0  --inside 1
"""
from __future__ import annotations

import os
import sys
import csv
import json
import math
import time
import signal
import logging
import argparse
import threading
from datetime import datetime, timezone

import requests
import websocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import kalshi_auth
import kalshi_orderbook
from coinbase_feeds import make_feed
from fair_price_model_v2 import record_price
from daily_calibration import fair_p_yes_daily, daily_calibration_active

# Optional v3 fair-value model (60s-avg settlement + fat tails + index basis).
# Enable with FAIR_MODEL=v3; defaults to the existing daily-calibrated v2 model.
_FAIR_MODEL = os.environ.get("FAIR_MODEL", "v2").lower()
if _FAIR_MODEL == "v3":
    import fair_price_model_v3 as _v3
    # use the live partial-average sharpening in the last minute (Coinbase feed)
    _V3_PARTIAL = os.environ.get("FAIR_V3_PARTIAL", "0") == "1"

ROOT = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(ROOT, ".env"))
API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
PRIVATE_KEY = kalshi_auth.load_private_key(env.get("KALSHI_PRIVATE_KEY", "")) \
    if env.get("KALSHI_PRIVATE_KEY") else None

BASE_URL = "https://api.elections.kalshi.com"
WS_URL   = "wss://api.elections.kalshi.com/trade-api/ws/v2"

SNAP_CSV = os.path.join(ROOT, "mm_shadow_snapshots.csv")
FILL_CSV = os.path.join(ROOT, "mm_shadow_fills.csv")

# ── Config ───────────────────────────────────────────────────────────────────
SERIES          = os.environ.get("MM_SHADOW_SERIES", "KXXRPD,KXDOGE,KXDOGED").split(",")
INSIDE_TICKS    = int(os.environ.get("MM_SHADOW_INSIDE", "1"))      # cents inside
MIN_SPREAD_C    = int(os.environ.get("MM_SHADOW_MIN_SPREAD", "3"))  # need room to rest inside
REQUOTE_SEC     = float(os.environ.get("MM_SHADOW_REQUOTE_SEC", "1.0"))
MID_MIN         = float(os.environ.get("MM_SHADOW_MID_MIN", "0.10"))  # near-the-money band
MID_MAX         = float(os.environ.get("MM_SHADOW_MID_MAX", "0.90"))
SNAPSHOT_SEC    = float(os.environ.get("MM_SHADOW_SNAPSHOT_SEC", "1.0"))
MARKET_REFRESH  = float(os.environ.get("MM_SHADOW_REFRESH_SEC", "60"))
QUOTE_SIZE      = float(os.environ.get("MM_SHADOW_QUOTE_SIZE", "10"))  # contracts, for $ est

# Coin symbol → Coinbase product. Extend as needed.
COIN_PRODUCT = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "DOGE": "DOGE-USD", "BNB": "BNB-USD", "ADA": "ADA-USD", "LTC": "LTC-USD",
    "HYPE": "HYPE-USD",
}
COIN_SYMBOLS = sorted(COIN_PRODUCT.keys(), key=len, reverse=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mm_shadow")


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def coin_of(ticker: str):
    body = ticker.split("-")[0].upper()
    body = body[2:] if body.startswith("KX") else body
    for s in COIN_SYMBOLS:
        if body.startswith(s):
            return s
    return None


# ── REST helpers ──────────────────────────────────────────────────────────────
def _get(path, params=None):
    h = kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path) if PRIVATE_KEY else None
    return requests.get(BASE_URL + path, params=params, headers=h, timeout=15)


def refresh_candidates():
    """Return {ticker: {asset, strike, close_dt}} for near-the-money, two-sided,
    wide-enough markets in the configured series."""
    out = {}
    for st in SERIES:
        try:
            r = _get("/trade-api/v2/markets", {"series_ticker": st, "status": "open", "limit": 200})
            r.raise_for_status()
            markets = r.json().get("markets", [])
        except Exception as e:
            log.warning(f"refresh {st} failed: {e}")
            continue
        for m in markets:
            bid = _f(m.get("yes_bid_dollars"))
            ask = _f(m.get("yes_ask_dollars"))
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            spread_c = round(ask * 100) - round(bid * 100)
            mid = (bid + ask) / 2.0
            if spread_c < MIN_SPREAD_C or not (MID_MIN <= mid <= MID_MAX):
                continue
            asset = coin_of(m["ticker"])
            strike = None
            for k in ("floor_strike", "strike", "cap_strike"):
                if m.get(k) is not None:
                    strike = _f(m.get(k)); break
            if asset is None or not strike:
                continue
            close = m.get("close_time")
            try:
                close_dt = datetime.fromisoformat(close.replace("Z", "+00:00"))
            except Exception:
                continue
            out[m["ticker"]] = {"asset": asset, "strike": strike, "close_dt": close_dt,
                                "strike_type": m.get("strike_type") or "greater",
                                "floor_strike": _f(m.get("floor_strike")) if m.get("floor_strike") is not None else None,
                                "cap_strike": _f(m.get("cap_strike")) if m.get("cap_strike") is not None else None}
    return out


# ── State ─────────────────────────────────────────────────────────────────────
_tracked: dict[str, dict] = {}          # ticker -> meta (asset, strike, close_dt)
_tracked_lock = threading.Lock()
_quotes: dict[str, dict] = {}           # ticker -> {bid, ask, set_ts}
_feeds: dict[str, object] = {}          # asset -> CoinbaseFeed
_pending_fills: list[dict] = []         # for inline horizon markout
_seen_trades: set = set()               # trade_id dedup (sweeps print same id twice)
_running = True


def _spot(asset):
    f = _feeds.get(asset)
    return f.get_price() if f else None


def _minutes_left(meta):
    return max(0.0, (meta["close_dt"] - datetime.now(timezone.utc)).total_seconds() / 60.0)


def _fair_c(meta):
    sp = _spot(meta["asset"])
    if sp is None:
        return None
    ml = _minutes_left(meta)
    if ml <= 0:
        return None
    if _FAIR_MODEL == "v3":
        floor = meta.get("floor_strike")
        cap = meta.get("cap_strike")
        st = meta.get("strike_type") or "greater"
        if floor is None and cap is None:
            floor = meta["strike"]  # fall back to the single captured strike
        return _v3.fair_p(sp, ml, meta["asset"], floor_strike=floor, cap_strike=cap,
                          strike_type=st, use_partial_history=_V3_PARTIAL) * 100.0
    return fair_p_yes_daily(sp, meta["strike"], ml, meta["asset"]) * 100.0


# ── Trade channel (fill detection) ────────────────────────────────────────────
def _on_trade(ticker, yes_price_c, taker_side, size):
    q = _quotes.get(ticker)
    if not q:
        return
    meta = _tracked.get(ticker)
    if not meta:
        return
    fair = _fair_c(meta)
    if fair is None:
        return
    age = time.time() - q["set_ts"]
    bb, ba = q.get("bb"), q.get("ba")
    fill = None
    if taker_side == "no" and q["bid"] is not None and q["bid"] >= yes_price_c:
        # taker sold yes → our resting YES bid is hit → we BOUGHT yes at our bid
        fill = ("buy_yes", q["bid"], fair - q["bid"])
    elif taker_side == "yes" and q["ask"] is not None and q["ask"] <= yes_price_c:
        # taker bought yes → our resting YES ask is lifted → we SOLD yes at our ask
        fill = ("sell_yes", q["ask"], q["ask"] - fair)
    if not fill:
        return
    side, price, imm_edge = fill
    # A finite-size resting order fills ONCE, then it's gone until we re-quote.
    # Without this, a taker sweep (multiple trade prints at the same instant)
    # re-fills the same quote repeatedly and inflates the fill count.
    if side == "buy_yes":
        q["bid"] = None
    else:
        q["ask"] = None
    row = {
        "ts": round(time.time(), 3), "ticker": ticker, "asset": meta["asset"],
        "side": side, "fill_price_c": round(price, 2), "fair_at_fill_c": round(fair, 2),
        "immediate_edge_c": round(imm_edge, 2), "bb_c": bb, "ba_c": ba,
        "spread_c": (round(ba - bb, 2) if (bb and ba) else ""),
        "mins_left": round(_minutes_left(meta), 2), "quote_age_s": round(age, 2),
        "trade_size": round(size, 2),
    }
    _append_csv(FILL_CSV, FILL_COLS, row)
    log.info(f"FILL {side:8} {ticker[:28]:28} @{price:.0f}c fair={fair:.1f}c "
             f"edge={imm_edge:+.1f}c age={age:.1f}s")


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
                    log.info(f"[trade] subscribed: {len(tk)} tickers")
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
                    if tkr not in _quotes:
                        return
                    _on_trade(tkr, _f(d.get("yes_price_dollars")) * 100.0,
                              d.get("taker_side"), _f(d.get("count_fp")))
                except Exception as e:
                    log.debug(f"[trade] parse err: {e}")

            # Periodically resubscribe to pick up new tickers after refresh.
            def resub():
                while _running:
                    time.sleep(MARKET_REFRESH)
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
                                        on_error=lambda w, e: log.warning(f"[trade] WS err: {e}"),
                                        on_close=lambda w, c, m: log.info("[trade] WS closed"))
            threading.Thread(target=resub, daemon=True).start()
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            log.warning(f"[trade] loop err: {e}")
        if _running:
            time.sleep(2)


# ── CSV ───────────────────────────────────────────────────────────────────────
SNAP_COLS = ["ts", "ticker", "asset", "spot", "strike", "mins_left",
             "bb_c", "ba_c", "bb_sz", "ba_sz", "spread_c",
             "my_bid_c", "my_ask_c", "fair_c", "quote_age_s"]
FILL_COLS = ["ts", "ticker", "asset", "side", "fill_price_c", "fair_at_fill_c",
             "immediate_edge_c", "bb_c", "ba_c", "spread_c", "mins_left",
             "quote_age_s", "trade_size"]
_csv_lock = threading.Lock()


def _append_csv(path, cols, row):
    with _csv_lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)


# ── Main collection loop ──────────────────────────────────────────────────────
def _ensure_feeds():
    with _tracked_lock:
        assets = {m["asset"] for m in _tracked.values()}
    for a in assets:
        if a not in _feeds and a in COIN_PRODUCT:
            f = make_feed(COIN_PRODUCT[a]); f.start(); _feeds[a] = f
            log.info(f"[feed] started {COIN_PRODUCT[a]}")


def _refresh_thread():
    while _running:
        cands = refresh_candidates()
        if cands:
            with _tracked_lock:
                _tracked.clear()
                _tracked.update(cands)
            _ensure_feeds()
            kalshi_orderbook.set_tickers(list(cands.keys()))
            log.info(f"[refresh] tracking {len(cands)} markets: "
                     f"{', '.join(sorted(cands.keys()))[:160]}")
        else:
            log.warning("[refresh] no candidates found")
        time.sleep(MARKET_REFRESH)


def collect():
    log.info(f"series={SERIES} inside={INSIDE_TICKS}c min_spread={MIN_SPREAD_C}c "
             f"requote={REQUOTE_SEC}s band={MID_MIN}-{MID_MAX}")
    # initial candidate set + feeds before starting WS
    cands = refresh_candidates()
    with _tracked_lock:
        _tracked.update(cands)
    _ensure_feeds()
    kalshi_orderbook.start(PRIVATE_KEY, API_KEY_ID, list(_tracked.keys()))
    threading.Thread(target=_trade_ws_loop, daemon=True).start()
    threading.Thread(target=_refresh_thread, daemon=True).start()
    log.info(f"warming up {len(_tracked)} markets; logging to {SNAP_CSV} / {FILL_CSV}")
    time.sleep(3)

    while _running:
        t0 = time.time()
        with _tracked_lock:
            items = list(_tracked.items())
        for ticker, meta in items:
            # feed realized-vol history
            sp = _spot(meta["asset"])
            if sp:
                record_price(meta["asset"], sp)
            book = kalshi_orderbook.get_book(ticker)
            if not book:
                continue
            bb = book.yes_bid(); ba = book.yes_ask()
            if bb is None or ba is None:
                continue
            bb_c, ba_c = round(bb * 100), round(ba * 100)
            spread_c = ba_c - bb_c
            bids = book.yes_bids_sorted(); asks = book.yes_asks_sorted()
            bb_sz = bids[0][1] if bids else 0
            ba_sz = asks[0][1] if asks else 0

            # --- sticky quote logic ---
            q = _quotes.get(ticker)
            want_bid = bb_c + INSIDE_TICKS
            want_ask = ba_c - INSIDE_TICKS
            inside_ok = spread_c >= MIN_SPREAD_C and want_bid < want_ask
            need_requote = (
                q is None
                or (time.time() - q["set_ts"]) >= REQUOTE_SEC
                or (q["bid"] is not None and q["bid"] >= ba_c)    # our bid now marketable → would cancel
                or (q["ask"] is not None and q["ask"] <= bb_c)    # our ask now marketable → would cancel
            )
            if need_requote:
                if inside_ok:
                    q = {"bid": want_bid, "ask": want_ask, "set_ts": time.time(),
                         "bb": bb_c, "ba": ba_c}
                else:
                    q = {"bid": None, "ask": None, "set_ts": time.time(),
                         "bb": bb_c, "ba": ba_c}
                _quotes[ticker] = q
            else:
                q["bb"], q["ba"] = bb_c, ba_c   # keep latest touch for fill context

            fair = _fair_c(meta)
            _append_csv(SNAP_CSV, SNAP_COLS, {
                "ts": round(t0, 3), "ticker": ticker, "asset": meta["asset"],
                "spot": round(sp, 6) if sp else "", "strike": meta["strike"],
                "mins_left": round(_minutes_left(meta), 2),
                "bb_c": bb_c, "ba_c": ba_c, "bb_sz": round(bb_sz, 1), "ba_sz": round(ba_sz, 1),
                "spread_c": spread_c, "my_bid_c": q["bid"] or "", "my_ask_c": q["ask"] or "",
                "fair_c": round(fair, 2) if fair is not None else "",
                "quote_age_s": round(time.time() - q["set_ts"], 2),
            })
        dt = time.time() - t0
        time.sleep(max(0.0, SNAPSHOT_SEC - dt))


# ── Report / analysis ─────────────────────────────────────────────────────────
def report(horizons=(15, 60, 300)):
    if not os.path.exists(FILL_CSV):
        print("No fills logged yet.");
    # load snapshot fair series per ticker
    series: dict[str, list] = {}
    if os.path.exists(SNAP_CSV):
        with open(SNAP_CSV) as fh:
            for r in csv.DictReader(fh):
                if r["fair_c"] == "":
                    continue
                series.setdefault(r["ticker"], []).append((float(r["ts"]), float(r["fair_c"])))
    for t in series:
        series[t].sort()

    def fair_at(ticker, ts):
        arr = series.get(ticker)
        if not arr:
            return None
        # nearest snapshot at or after ts (within 30s), else last before
        lo, hi = 0, len(arr) - 1
        best = None
        for s_ts, s_fair in arr:
            if s_ts >= ts:
                best = s_fair; break
        if best is None and arr:
            best = arr[-1][1]
        return best

    fills = []
    if os.path.exists(FILL_CSV):
        with open(FILL_CSV) as fh:
            fills = list(csv.DictReader(fh))
    if not fills:
        print("No fills to analyze. Let the logger run longer (thin markets → rare fills).")
        # still report coverage
        if series:
            span = max(a[-1][0] for a in series.values()) - min(a[0][0] for a in series.values())
            print(f"Snapshot coverage: {sum(len(a) for a in series.values())} rows across "
                  f"{len(series)} markets, ~{span/3600:.1f}h.")
        return

    # aggregate
    print(f"\n=== Passive-fill shadow report ({len(fills)} fills) ===\n")
    by_ticker: dict[str, list] = {}
    for f in fills:
        by_ticker.setdefault(f["ticker"], []).append(f)

    hcols = "  ".join(f"mk{h}s" for h in horizons)
    print(f"{'ticker':30} {'n':>3} {'imm¢':>6}  {hcols}   {'advSel¢(60s)':>12}")
    print("-" * 90)
    tot = {"n": 0, "imm": 0.0, "mk": {h: 0.0 for h in horizons}, "adv60": 0.0}
    for ticker, fs in sorted(by_ticker.items()):
        n = len(fs)
        imm = sum(float(f["immediate_edge_c"]) for f in fs) / n
        mk = {}
        for h in horizons:
            vals = []
            for f in fs:
                fa = fair_at(ticker, float(f["ts"]) + h)
                if fa is None:
                    continue
                if f["side"] == "buy_yes":
                    vals.append(fa - float(f["fill_price_c"]))      # markout for a long
                else:
                    vals.append(float(f["fill_price_c"]) - fa)      # markout for a short
            mk[h] = sum(vals) / len(vals) if vals else float("nan")
        adv60 = imm - mk.get(60, float("nan"))
        mkstr = "  ".join(f"{mk[h]:+5.1f}" for h in horizons)
        print(f"{ticker[:30]:30} {n:>3} {imm:>+6.1f}  {mkstr}   {adv60:>+12.1f}")
        tot["n"] += n; tot["imm"] += imm * n
        for h in horizons:
            tot["mk"][h] += (mk[h] if mk[h] == mk[h] else 0) * n
        tot["adv60"] += (adv60 if adv60 == adv60 else 0) * n

    n = tot["n"]
    print("-" * 90)
    mkstr = "  ".join(f"{tot['mk'][h]/n:+5.1f}" for h in horizons)
    print(f"{'ALL':30} {n:>3} {tot['imm']/n:>+6.1f}  {mkstr}   {tot['adv60']/n:>+12.1f}")
    print("\nLegend:")
    print("  imm¢      = immediate model edge at fill (fair vs our price)")
    print("  mkNs      = markout: realized edge if offloaded at fair after N sec")
    print("  advSel¢   = adverse selection = imm − mk60 (how much fair moved against us)")
    print("\nVERDICT: passive MM survives only if markout (mkNs) stays POSITIVE.")
    print("If imm is positive but mk60 ≈ 0 or negative, the spread is an illusion —")
    print("you're being picked off and the edge evaporates after the fill.")
    # crude $/day estimate from mk60
    if series:
        span_h = (max(a[-1][0] for a in series.values()) - min(a[0][0] for a in series.values())) / 3600
        if span_h > 0:
            fills_per_day = n / span_h * 24
            mk60 = tot["mk"][60] / n
            print(f"\nObserved: {n} fills over ~{span_h:.1f}h → ~{fills_per_day:.0f} fills/day.")
            print(f"At {QUOTE_SIZE:.0f} contracts/fill and mk60={mk60:+.1f}¢: "
                  f"~${fills_per_day * QUOTE_SIZE * mk60 / 100:+.1f}/day (rough, pre-inventory-risk).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--series", default=None)
    ap.add_argument("--requote-sec", type=float, default=None)
    ap.add_argument("--inside", type=int, default=None)
    ap.add_argument("--horizons", default="15,60,300")
    args = ap.parse_args()

    global SERIES, REQUOTE_SEC, INSIDE_TICKS, _running
    if args.series:
        SERIES = args.series.split(",")
    if args.requote_sec is not None:
        REQUOTE_SEC = args.requote_sec
    if args.inside is not None:
        INSIDE_TICKS = args.inside

    if args.report:
        report(tuple(int(x) for x in args.horizons.split(",")))
        return

    def _stop(*_):
        global _running
        _running = False
        log.info("stopping...")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    collect()


if __name__ == "__main__":
    main()
