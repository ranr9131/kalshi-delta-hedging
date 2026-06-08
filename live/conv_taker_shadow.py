"""
Convergence-taker shadow sim (paper) — thin-alt crypto, near expiry.

THESIS (the one structural, non-illusory edge from the v3 work):
  Settlement = the 60-second AVERAGE of the index over the final minute.  In the
  last ~90s a large fraction of that average is ALREADY LOCKED.  A market pricing
  off spot over-reacts to late spot moves the settlement average can no longer
  follow.  The partial-average-aware v3 fair knows the settlement to ~0.1 bps at
  5s out; when the market diverges from it by MORE than the spread + fee, we take.

This is a MECHANICAL edge (the settlement is an average), not a forecasting edge,
so it should survive the selection-effect critique IF it survives the spread.
That "if" is the whole question — hence shadow sim, not live money.

How it works (taker, paper):
  * Track thin-alt 15M + hourly markets closing soon (kalshi_orderbook WS book).
  * Feed Coinbase spot to record_price (v3 realized sigma + running partial avg).
  * Only in the final WINDOW_SEC seconds, every TICK_SEC:
      fair = v3.fair_p(spot, mins_left, asset, strike..., use_partial_history=True)
      yes-buy edge = fair - yes_ask ;  no-buy edge = yes_bid - fair
      if edge*100 > fee_c + MIN_EDGE_C and touch size available -> record a paper
      fill at the touch (we CROSS the spread — the honest taker cost).
  * --report joins true settlements (Kalshi API) and computes markout-to-
    settlement P&L per fill, net of the Kalshi fee, sliced by asset / ttc / edge.

Horizons: runs on BOTH Kalshi crypto cadences and reports them SEPARATELY —
  * 15-min up/down   (KX*15M)   -> horizon "15m"
  * 1-hour threshold (KX*D)     -> horizon "hourly"
(Kalshi has no 5-min crypto market; that's Polymarket-only.)  Per-horizon action
window is configurable: CONV_WINDOW_15M / CONV_WINDOW_HOURLY (default 90s each).

Run (collect):  python3 conv_taker_shadow.py
Run (analyze):  python3 conv_taker_shadow.py --report   # splits ALL / 15m / hourly
Self-test:      python3 conv_taker_shadow.py --selftest
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
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import kalshi_auth
import kalshi_orderbook
from coinbase_feeds import make_feed
from fair_price_model_v2 import record_price
import fair_price_model_v3 as v3

ROOT = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(ROOT, ".env"))
API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")
PRIVATE_KEY = kalshi_auth.load_private_key(env.get("KALSHI_PRIVATE_KEY", "")) \
    if env.get("KALSHI_PRIVATE_KEY") else None
BASE_URL = "https://api.elections.kalshi.com"

FILL_CSV = os.path.join(ROOT, "conv_taker_fills.csv")

# ── Config ───────────────────────────────────────────────────────────────────
SERIES = os.environ.get("CONV_SERIES",
                        "KXXRP15M,KXSOL15M,KXDOGE15M,KXXRPD,KXSOLD,KXDOGED").split(",")
# Per-horizon action window (final N seconds before close). The mechanical 60s-avg
# edge lives near expiry for BOTH cadences, but you can widen per horizon to map
# the decay. 15m markets only live 15 min; hourly live 60 min.
WINDOW_SEC_15M    = float(os.environ.get("CONV_WINDOW_15M", "90"))
WINDOW_SEC_HOURLY = float(os.environ.get("CONV_WINDOW_HOURLY", "90"))
WINDOW_SEC   = max(WINDOW_SEC_15M, WINDOW_SEC_HOURLY)           # tracking horizon
TICK_SEC     = float(os.environ.get("CONV_TICK_SEC", "0.5"))    # sampling in window
MIN_EDGE_C   = float(os.environ.get("CONV_MIN_EDGE_C", "2.0"))  # edge over fee, cents
MAX_SIZE     = float(os.environ.get("CONV_MAX_SIZE", "50"))     # contracts/signal cap
TRACK_WITHIN_SEC = float(os.environ.get("CONV_TRACK_WITHIN_SEC", "300"))  # subscribe horizon
REFRESH_SEC  = float(os.environ.get("CONV_REFRESH_SEC", "30"))
MID_LO, MID_HI = 0.02, 0.98     # only near-money strikes (outcome in doubt)
ONE_SHOT_PER_TICKER = os.environ.get("CONV_ONE_SHOT", "1") == "1"  # one fill/ticker

COIN_PRODUCT = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
                "XRP": "XRP-USD", "DOGE": "DOGE-USD", "LTC": "LTC-USD",
                "ADA": "ADA-USD", "BNB": "BNB-USD"}
COIN_SYMBOLS = sorted(COIN_PRODUCT, key=len, reverse=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  conv  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("conv")

_running = True
_tracked: dict = {}
_tracked_lock = threading.Lock()
_feeds: dict = {}
_fired: set = set()
_csv_lock = threading.Lock()

FILL_COLS = ["ts", "ticker", "asset", "horizon", "side", "fill_price_c", "fair_c",
             "mid_c", "spot", "partial_avg", "strike", "strike_type", "ttc_sec",
             "mins_left", "edge_c", "fee_c", "size", "touch_size"]


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def horizon_of(ticker: str, strike_type: str) -> str:
    """15m up/down (strike_type greater_or_equal, or *15M ticker) vs hourly."""
    if "15M" in ticker.upper() or (strike_type or "").lower() == "greater_or_equal":
        return "15m"
    return "hourly"


def window_for(horizon: str) -> float:
    return WINDOW_SEC_15M if horizon == "15m" else WINDOW_SEC_HOURLY


def coin_of(ticker: str):
    body = ticker.split("-")[0].upper()
    body = body[2:] if body.startswith("KX") else body
    for s in COIN_SYMBOLS:
        if body.startswith(s):
            return s
    return None


def fee_cents(price_dollars: float) -> float:
    """Kalshi trading fee per contract ≈ 7 * p * (1-p) cents (0.07*C*P*(1-P))."""
    p = min(max(price_dollars, 0.0), 1.0)
    return 7.0 * p * (1.0 - p)


def _spot(asset):
    f = _feeds.get(asset)
    return f.get_price() if f else None


# ── REST ──────────────────────────────────────────────────────────────────────
def _get(path, params=None):
    h = kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path) if PRIVATE_KEY else None
    return requests.get(BASE_URL + path, params=params, headers=h, timeout=15)


def refresh_candidates():
    """{ticker: meta} for thin-alt crypto markets closing within TRACK_WITHIN_SEC."""
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
            if asset is None:
                continue
            ci = m.get("close_time")
            if not ci:
                continue
            try:
                close_dt = datetime.fromisoformat(ci.replace("Z", "+00:00"))
            except Exception:
                continue
            ttc = (close_dt - now).total_seconds()
            if not (0 < ttc <= TRACK_WITHIN_SEC):
                continue
            st_type = m.get("strike_type") or "greater"
            out[m["ticker"]] = {
                "asset": asset, "close_dt": close_dt,
                "strike_type": st_type,
                "horizon": horizon_of(m["ticker"], st_type),
                "floor_strike": _f(m.get("floor_strike")),
                "cap_strike": _f(m.get("cap_strike")),
                "strike": _f(m.get("floor_strike")) if m.get("floor_strike") is not None
                          else _f(m.get("cap_strike")),
            }
    return out


def _append(path, cols, row):
    with _csv_lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)


def _fair(meta, spot, mins_left):
    return v3.fair_p(spot, mins_left, meta["asset"],
                     floor_strike=meta.get("floor_strike"),
                     cap_strike=meta.get("cap_strike"),
                     strike_type=meta.get("strike_type", "greater"),
                     use_partial_history=True)


# ── Core decision (pure, unit-testable) ────────────────────────────────────────
def decide(fair, yes_bid, yes_ask, min_edge_c=MIN_EDGE_C):
    """Return (side, fill_price_dollars, edge_c, fee_c) or None.
    Crosses the spread (taker).  side in {buy_yes, buy_no}."""
    if fair is None or yes_bid is None or yes_ask is None:
        return None
    yes_edge = fair - yes_ask                 # buy YES at ask
    no_edge = yes_bid - fair                  # buy NO at (1 - yes_bid)
    if yes_edge >= no_edge and yes_edge > 0:
        fee = fee_cents(yes_ask)
        net = yes_edge * 100.0 - fee
        if net >= min_edge_c:
            return ("buy_yes", yes_ask, yes_edge * 100.0, fee)
    if no_edge > 0:
        no_price = 1.0 - yes_bid
        fee = fee_cents(no_price)
        net = no_edge * 100.0 - fee
        if net >= min_edge_c:
            return ("buy_no", no_price, no_edge * 100.0, fee)
    return None


# ── Collection loop ────────────────────────────────────────────────────────────
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
        with _tracked_lock:
            _tracked.clear(); _tracked.update(cands)
        _ensure_feeds()
        try:
            kalshi_orderbook.set_tickers(list(cands.keys()))
        except Exception:
            pass
        log.info(f"[refresh] tracking {len(cands)} markets closing < {TRACK_WITHIN_SEC:.0f}s")
        time.sleep(REFRESH_SEC)


def collect():
    log.info(f"series={SERIES} window(15m={WINDOW_SEC_15M}s,hourly={WINDOW_SEC_HOURLY}s) "
             f"tick={TICK_SEC}s min_edge={MIN_EDGE_C}c")
    cands = refresh_candidates()
    with _tracked_lock:
        _tracked.update(cands)
    _ensure_feeds()
    kalshi_orderbook.start(PRIVATE_KEY, API_KEY_ID, list(_tracked.keys()))
    threading.Thread(target=_refresh_thread, daemon=True).start()
    time.sleep(3)
    log.info(f"collecting -> {FILL_CSV}")

    while _running:
        t0 = time.time()
        with _tracked_lock:
            items = list(_tracked.items())
        for ticker, meta in items:
            sp = _spot(meta["asset"])
            if sp:
                record_price(meta["asset"], sp)   # always feed σ + partial-avg
            ttc = (meta["close_dt"] - datetime.now(timezone.utc)).total_seconds()
            if ttc > window_for(meta["horizon"]) or ttc <= 0:
                continue
            if ONE_SHOT_PER_TICKER and ticker in _fired:
                continue
            if not sp:
                continue
            book = kalshi_orderbook.get_book(ticker)
            if not book:
                continue
            bb = book.yes_bid(); ba = book.yes_ask()
            if bb is None or ba is None:
                continue
            mid = (bb + ba) / 2.0
            if not (MID_LO <= mid <= MID_HI):
                continue
            ml = ttc / 60.0
            fair = _fair(meta, sp, ml)
            sig = decide(fair, bb, ba)
            if not sig:
                continue
            side, price, edge_c, fee = sig
            asks = book.yes_asks_sorted(); bids = book.yes_bids_sorted()
            touch_sz = (asks[0][1] if asks else 0) if side == "buy_yes" else (bids[0][1] if bids else 0)
            size = min(MAX_SIZE, touch_sz) if touch_sz else 0
            if size <= 0:
                continue
            pavg = v3.trailing_avg(meta["asset"], max(0.0, (1.0 - ml)) * 60.0) if ml < 1 else None
            _append(FILL_CSV, FILL_COLS, {
                "ts": round(time.time(), 3), "ticker": ticker, "asset": meta["asset"],
                "horizon": meta["horizon"],
                "side": side, "fill_price_c": round(price * 100, 2),
                "fair_c": round(fair * 100, 2), "mid_c": round(mid * 100, 2),
                "spot": round(sp, 6), "partial_avg": round(pavg, 6) if pavg else "",
                "strike": meta["strike"], "strike_type": meta["strike_type"],
                "ttc_sec": round(ttc, 1), "mins_left": round(ml, 3),
                "edge_c": round(edge_c, 2), "fee_c": round(fee, 2),
                "size": round(size, 1), "touch_size": round(touch_sz, 1),
            })
            _fired.add(ticker)
            log.info(f"TAKE {side:7} {ticker[:30]:30} @{price*100:.0f}c fair={fair*100:.0f}c "
                     f"mid={mid*100:.0f}c edge={edge_c:+.1f}c ttc={ttc:.0f}s sz={size:.0f}")
        time.sleep(max(0.0, TICK_SEC - (time.time() - t0)))


# ── Settlement + report ────────────────────────────────────────────────────────
def _settle(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    try:
        r = requests.get(f"{BASE_URL}/trade-api/v2/markets/{ticker}", timeout=10)
        m = r.json().get("market", {}) if r.ok else {}
        res = (m.get("result") or "").lower()
        cache[ticker] = (1 if res == "yes" else 0) if res in ("yes", "no") else None
    except Exception:
        cache[ticker] = None
    return cache[ticker]


def report():
    if not os.path.exists(FILL_CSV):
        print("No fills logged yet."); return
    rows = list(csv.DictReader(open(FILL_CSV)))
    print(f"{len(rows)} paper fills logged")
    cache = {}
    trades = []
    for r in rows:
        y = _settle(r["ticker"], cache)
        if y is None:
            continue
        fp = float(r["fill_price_c"]); fee = float(r["fee_c"])
        if r["side"] == "buy_yes":
            pnl = (100.0 * y - fp) - fee
        else:  # buy_no: pays fp cents, worth 100 if outcome NO (y==0)
            pnl = (100.0 * (1 - y) - fp) - fee
        trades.append({**r, "outcome": y, "pnl_c": pnl,
                       "edge_c": float(r["edge_c"]), "ttc": float(r["ttc_sec"]),
                       "asset": r["asset"],
                       "horizon": r.get("horizon") or horizon_of(r["ticker"], r.get("strike_type", ""))})
    if not trades:
        print("No settled fills yet."); return

    def agg(name, ts):
        n = len(ts)
        if not n:
            return
        pnl = sum(t["pnl_c"] for t in ts) / n
        hit = sum(1 for t in ts if t["pnl_c"] > 0) / n
        tot = sum(t["pnl_c"] for t in ts)
        ed = sum(t["edge_c"] for t in ts) / n
        print(f"  {name:>16}: n={n:>4}  pnl/trade={pnl:+6.2f}c  hit={hit:5.1%}  "
              f"pred_edge={ed:+5.1f}c  total={tot:+8.1f}c")

    def report_block(label, ts):
        if not ts:
            return
        print(f"\n################  {label}  (n={len(ts)})  ################")
        agg("ALL", ts)
        print("  -- by asset --")
        for a in sorted(set(t["asset"] for t in ts)):
            agg(a, [t for t in ts if t["asset"] == a])
        print("  -- by side --")
        for s in ("buy_yes", "buy_no"):
            agg(s, [t for t in ts if t["side"] == s])
        print("  -- by time-to-close --")
        for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 90), (90, 1e9)]:
            agg(f"{lo}-{int(hi) if hi < 1e8 else 'inf'}s", [t for t in ts if lo <= t["ttc"] < hi])
        print("  -- by predicted edge --")
        for lo, hi in [(2, 4), (4, 7), (7, 12), (12, 100)]:
            agg(f"{lo}-{hi}c", [t for t in ts if lo <= t["edge_c"] < hi])

    print(f"\nsettled fills: {len(trades)}")
    report_block("ALL HORIZONS", trades)
    report_block("15-MIN markets (KX*15M)", [t for t in trades if t["horizon"] == "15m"])
    report_block("1-HOUR markets (KX*D)", [t for t in trades if t["horizon"] == "hourly"])


# ── Self-test (no network): exercises decide() + P&L accounting ────────────────
def selftest():
    print("decide() unit checks:")
    # fair 0.90, ask 0.70 -> big YES edge, fee ~ 7*.7*.3=1.47c, net ~ 18.5c
    print("  YES underpriced:", decide(0.90, 0.60, 0.70))
    # fair 0.10, bid 0.40 -> buy NO at 0.60, edge = 0.40-0.10=0.30, fee 7*.6*.4=1.68
    print("  YES overpriced :", decide(0.10, 0.40, 0.50))
    # fair 0.50, bid .48 ask .52 -> no edge clears fee
    print("  no edge        :", decide(0.50, 0.48, 0.52))
    # P&L accounting sanity
    print("\nP&L accounting (fee excluded for clarity):")
    print("  buy_yes @70c, outcome YES:", 100*1 - 70, "c gross")
    print("  buy_yes @70c, outcome NO :", 100*0 - 70, "c gross")
    print("  buy_no  @60c, outcome NO :", 100*1 - 60, "c gross (NO pays 100)")
    print("  fee at 0.5 =", round(fee_cents(0.5), 3), "c ; at 0.9 =", round(fee_cents(0.9), 3), "c")


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
        log.info("stopped")


if __name__ == "__main__":
    main()
