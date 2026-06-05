"""
Kalshi 15M crypto sniper — variant v2.

Differences from v1 (sniper.py):
  1. Uses fair_price_model_v2 → realized σ instead of constant, and loads
     calibration_v2.json (live-data-fit Platt scaling) instead of v1's
     historical-data calibration.json.
  2. Directional gate: only fires YES if the recent crypto move was UP
     (favorable for YES winning) and only fires NO if the move was DOWN.
     Stops the "buy YES on a downtrend" failure mode that gave us the −$58
     trough early on.
  3. Writes to snipes_v2.csv so v1 and v2 can be compared side-by-side
     without contaminating each other.

Designed to run in parallel with v1 as its own systemd service.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Optional, Deque, Tuple

import requests
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kalshi_auth
import kalshi_orderbook as ob
import kalshi_trade
import btc_feed, eth_feed, sol_feed, xrp_feed, hype_feed
from coinbase_feeds import make_feed
from fair_price_model_v2 import (
    fair_p_yes_v2, fair_p_yes_raw_v2,
    effective_sigma_per_min, record_price, FALLBACK_SIGMA_PER_MIN,
)

# Generic Coinbase feeds for the 4 newer assets (single source via factory
# instead of cloning ${coin}_feed.py 4 more times).
_bnb_feed  = make_feed("BNB-USD")
_ton_feed  = make_feed("TON-USD")
_doge_feed = make_feed("DOGE-USD")
_ada_feed  = make_feed("ADA-USD")


# ── Config ──────────────────────────────────────────────────────────────────

PAPER_MODE          = os.environ.get("PAPER_MODE", "true").lower() != "false"
MIN_EDGE_CENTS      = int(os.environ.get("MIN_EDGE_CENTS", "5"))
MAX_STAKE_PER_SNIPE = float(os.environ.get("MAX_STAKE_PER_SNIPE", "5.0"))
MIN_MINUTES_LEFT    = float(os.environ.get("MIN_MINUTES_LEFT", "0.5"))
MAX_MINUTES_LEFT    = float(os.environ.get("MAX_MINUTES_LEFT", "14.0"))
TICK_SECONDS        = float(os.environ.get("TICK_SECONDS", "0.05"))
MIN_FAIR_P          = float(os.environ.get("MIN_FAIR_P", "0.15"))
MAX_FAIR_P          = float(os.environ.get("MAX_FAIR_P", "0.85"))

MIN_LEVEL_AGE_SEC = float(os.environ.get("MIN_LEVEL_AGE_SEC", "2.0"))
MOVE_WINDOW_SEC   = float(os.environ.get("MOVE_WINDOW_SEC",   "3.0"))
MOVE_GATE_MULT    = float(os.environ.get("MOVE_GATE_MULT",    "1.2"))

# Directional gate threshold — the signed move (bps over MOVE_WINDOW_SEC)
# must be at least this magnitude AND on the favorable side of zero.
# Same as the move gate; the directional gate just adds the SIGN check.
def _min_move_bps_for(asset: str) -> float:
    override = os.environ.get(f"MIN_MOVE_BPS_{asset.upper()}")
    if override:
        return float(override)
    sigma = FALLBACK_SIGMA_PER_MIN.get(asset.upper(), 0.0015)
    typical_bps = sigma * math.sqrt(MOVE_WINDOW_SEC / 60.0) * 10000.0
    return typical_bps * MOVE_GATE_MULT

_ASSETS = ("BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "TON", "DOGE", "ADA")
_MIN_MOVE_BPS = {a: _min_move_bps_for(a) for a in _ASSETS}

SERIES = os.environ.get(
    "SERIES",
    "KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXHYPE15M,KXBNB15M,KXTON15M,KXDOGE15M,KXADA15M"
).split(",")
LOG_PATH = os.environ.get(
    "LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "snipes_v2.csv"),
)
MARKET_REFRESH_SEC = 20.0
SNIPE_COOLDOWN_SEC = 3.0

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
KALSHI_BASE = "https://api.elections.kalshi.com"


# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sniper_v2")


SERIES_TO_ASSET = {
    "KXBTC15M":  "BTC",  "KXETH15M": "ETH",
    "KXSOL15M":  "SOL",  "KXXRP15M": "XRP",
    "KXHYPE15M": "HYPE",
    "KXBNB15M":  "BNB",  "KXTON15M":  "TON",
    "KXDOGE15M": "DOGE", "KXADA15M":  "ADA",
}


def _asset_for_ticker(ticker: str) -> Optional[str]:
    for prefix, sym in SERIES_TO_ASSET.items():
        if ticker.startswith(prefix):
            return sym
    return None


def _price_for_asset(asset: str) -> Optional[float]:
    if asset == "BTC":  return btc_feed.get_price()
    if asset == "ETH":  return eth_feed.get_price()
    if asset == "SOL":  return sol_feed.get_price()
    if asset == "XRP":  return xrp_feed.get_price()
    if asset == "HYPE": return hype_feed.get_price()
    if asset == "BNB":  return _bnb_feed.get_price()
    if asset == "TON":  return _ton_feed.get_price()
    if asset == "DOGE": return _doge_feed.get_price()
    if asset == "ADA":  return _ada_feed.get_price()
    return None


def _price_age_for_asset(asset: str) -> float:
    if asset == "BTC":  return btc_feed.get_price_age()
    if asset == "ETH":  return eth_feed.get_price_age()
    if asset == "SOL":  return sol_feed.get_price_age()
    if asset == "XRP":  return xrp_feed.get_price_age()
    if asset == "HYPE": return hype_feed.get_price_age()
    if asset == "BNB":  return _bnb_feed.get_price_age()
    if asset == "TON":  return _ton_feed.get_price_age()
    if asset == "DOGE": return _doge_feed.get_price_age()
    if asset == "ADA":  return _ada_feed.get_price_age()
    return float("inf")


def fetch_open_markets() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for series in SERIES:
        try:
            r = requests.get(
                f"{KALSHI_BASE}/trade-api/v2/markets",
                params={"series_ticker": series, "status": "open", "limit": 50},
                timeout=10,
            )
            r.raise_for_status()
            for mk in r.json().get("markets", []):
                t = mk.get("ticker")
                if t:
                    out[t] = mk
        except Exception as e:
            log.warning(f"market fetch failed for {series}: {e}")
    return out


def _parse_close(market: dict) -> Optional[datetime]:
    ts = market.get("close_time") or market.get("expected_expiration_time")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _strike(market: dict) -> Optional[float]:
    for k in ("floor_strike", "strike", "strike_price"):
        v = market.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    try:
        return float(market.get("ticker", "").split("-")[-1])
    except Exception:
        return None


# ── Log CSV ────────────────────────────────────────────────────────────────

LOG_FIELDS = [
    "ts_iso", "mode", "ticker", "asset", "side",
    "limit_cents", "fill_cents_est", "qty", "stake_dollars",
    "fair_p", "edge_cents", "crypto_price", "strike",
    "minutes_left", "yes_bid", "yes_ask", "no_bid", "no_ask",
    "lvl_age_sec", "move_bps_signed", "sigma_per_min",
    "result", "settled_pnl",
]


def _log_init():
    new = not os.path.exists(LOG_PATH)
    f = open(LOG_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
    if new:
        w.writeheader()
        f.flush()
    return f, w


# ── Move tracking ──────────────────────────────────────────────────────────

_last_snipe_ts: Dict[tuple, float] = defaultdict(float)

# Rolling crypto price history per asset for the move gate (short window).
# Separate from the realized-vol history maintained by fair_price_model_v2.
_move_hist: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
    lambda: deque(maxlen=200)
)


def _record_move(asset: str, price: float) -> None:
    now = time.time()
    buf = _move_hist[asset]
    buf.append((now, price))
    while buf and now - buf[0][0] > 10.0:
        buf.popleft()


def _recent_move_signed_bps(asset: str, window_sec: float) -> float:
    """Signed (price_now - price_window_ago) / price_now in bps.  Positive
    = up move (favors YES), negative = down move (favors NO)."""
    buf = _move_hist[asset]
    if len(buf) < 2:
        return 0.0
    now_ts, now_px = buf[-1]
    if now_px <= 0:
        return 0.0
    cutoff = now_ts - window_sec
    ref_px = None
    for ts, px in buf:
        if ts >= cutoff:
            ref_px = px
            break
    if ref_px is None:
        return 0.0
    return (now_px - ref_px) / now_px * 10000.0


# ── Snipe execution ────────────────────────────────────────────────────────

def _maybe_snipe(
    ticker: str, market: dict, book: ob.Book,
    crypto_price: float, minutes_left: float,
    log_writer, log_file, private_key, api_key_id,
):
    asset  = _asset_for_ticker(ticker)
    strike = _strike(market)
    if asset is None or strike is None:
        return

    sigma = effective_sigma_per_min(asset)
    fpy = fair_p_yes_v2(crypto_price, strike, minutes_left, asset, sigma)
    fpn = 1.0 - fpy

    yes_ask = book.yes_ask()
    no_ask  = book.no_ask()
    edge_dollars = MIN_EDGE_CENTS / 100.0

    # Signed move + magnitude
    move_signed = _recent_move_signed_bps(asset, MOVE_WINDOW_SEC)
    move_mag = abs(move_signed)
    move_threshold = _MIN_MOVE_BPS.get(asset, 5.0)

    # Magnitude gate — same as v1
    if move_mag < move_threshold:
        return

    snipes = []  # (side, limit_dollars, fair_p, edge_cents, lvl_age, move_signed)

    # ── YES side — only fire if recent move was UP ─────────────────────
    # Up move → fair_p_yes just increased → resting YES ask is the one
    # most likely to be stale (didn't have time to reprice up).
    if (yes_ask is not None and yes_ask >= 0.01
            and MIN_FAIR_P <= fpy <= MAX_FAIR_P
            and move_signed > 0):
        max_pay_yes = fpy - edge_dollars
        if yes_ask <= max_pay_yes:
            lvl_age = book.yes_ask_age()
            if lvl_age >= MIN_LEVEL_AGE_SEC:
                limit_d = min(0.99, max_pay_yes)
                snipes.append(("yes", round(limit_d, 4), fpy,
                               (fpy - yes_ask) * 100.0, lvl_age, move_signed))

    # ── NO side — only fire if recent move was DOWN ────────────────────
    if (no_ask is not None and no_ask >= 0.01
            and MIN_FAIR_P <= fpn <= MAX_FAIR_P
            and move_signed < 0):
        max_pay_no = fpn - edge_dollars
        if no_ask <= max_pay_no:
            lvl_age = book.no_ask_age()
            if lvl_age >= MIN_LEVEL_AGE_SEC:
                limit_d = min(0.99, max_pay_no)
                snipes.append(("no", round(limit_d, 4), fpn,
                               (fpn - no_ask) * 100.0, lvl_age, move_signed))

    now = time.time()
    for side, limit_d, fair_p, edge_c, lvl_age, mv_s in snipes:
        key = (ticker, side)
        if now - _last_snipe_ts[key] < SNIPE_COOLDOWN_SEC:
            continue
        _last_snipe_ts[key] = now

        ladder = book.yes_asks_sorted() if side == "yes" else book.no_asks_sorted()
        avail_qty = sum(q for p, q in ladder if p <= limit_d)
        if avail_qty <= 0:
            continue
        max_qty_by_stake = MAX_STAKE_PER_SNIPE / max(limit_d, 0.01)
        qty = max(1.0, min(avail_qty, max_qty_by_stake))

        filled = 0.0
        cost   = 0.0
        for p, q in ladder:
            if p > limit_d or filled >= qty:
                break
            take = min(q, qty - filled)
            filled += take
            cost   += take * p
        if filled <= 0:
            continue
        avg_fill_d = cost / filled
        stake = cost

        log.info(
            f"V2 SNIPE  {ticker}  {side.upper()}  limit=${limit_d:.4f}  "
            f"fill_est=${avg_fill_d:.4f}  qty={filled:.1f}  "
            f"fair={fair_p*100:.1f}¢  edge={edge_c:.1f}¢  "
            f"lvl_age={lvl_age:.1f}s  mv={mv_s:+.1f}bps  σ={sigma*100:.3f}%/min  "
            f"[{'PAPER' if PAPER_MODE else 'LIVE'}]"
        )

        if not PAPER_MODE:
            try:
                limit_cents = max(1, min(99, int(round(limit_d * 100))))
                if side == "yes":
                    base_ask_c = int(round(float(market["yes_ask_dollars"]) * 100))
                    extra = limit_cents - base_ask_c - kalshi_trade.FILL_BUFFER_CENTS
                else:
                    base_yes_bid_c   = int(round(float(market["yes_bid_dollars"]) * 100))
                    target_yes_price = 100 - limit_cents
                    extra = base_yes_bid_c - target_yes_price - kalshi_trade.FILL_BUFFER_CENTS
                kalshi_trade.place_order(
                    private_key, api_key_id, ticker, side, market,
                    stake_dollars=stake, extra_buffer_cents=extra, ioc=True,
                )
            except Exception as e:
                log.warning(f"order failed: {e}")

        log_writer.writerow({
            "ts_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode":   "paper" if PAPER_MODE else "live",
            "ticker": ticker, "asset": asset, "side": side,
            "limit_cents": round(limit_d * 100, 2),
            "fill_cents_est": round(avg_fill_d * 100, 2),
            "qty": round(filled, 3), "stake_dollars": round(stake, 2),
            "fair_p": round(fair_p, 4),
            "edge_cents": round(edge_c, 2),
            "crypto_price": round(crypto_price, 4),
            "strike": strike,
            "minutes_left": round(minutes_left, 3),
            "yes_bid": book.yes_bid(), "yes_ask": book.yes_ask(),
            "no_bid":  book.no_bid(),  "no_ask":  book.no_ask(),
            "lvl_age_sec": round(lvl_age, 2),
            "move_bps_signed": round(mv_s, 2),
            "sigma_per_min":  round(sigma, 6),
            "result": "", "settled_pnl": "",
        })
        log_file.flush()


def main():
    env = dotenv_values(ENV_PATH)
    private_key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    api_key_id  = env["KALSHI_API_KEY_ID"]

    log.info(
        f"sniper_v2 start  PAPER_MODE={PAPER_MODE}  series={SERIES}  "
        f"min_edge={MIN_EDGE_CENTS}¢  max_stake=${MAX_STAKE_PER_SNIPE}  "
        f"tick={TICK_SECONDS*1000:.0f}ms"
    )
    log.info("move thresholds (bps over %.1fs, mult=%.2f): %s",
             MOVE_WINDOW_SEC, MOVE_GATE_MULT,
             {a: round(v, 2) for a, v in _MIN_MOVE_BPS.items()})
    log.info("coinflip filter: %.2f <= fair_p <= %.2f", MIN_FAIR_P, MAX_FAIR_P)
    log.info("directional gate ON: YES only on +move, NO only on -move")
    log.info("realized vol: ENABLED  (calibration_v2: see fair_price_model_v2)")

    if kalshi_trade.warmup_session():
        log.info("kalshi session pre-warmed (TLS handshake done)")
    else:
        log.warning("kalshi session warmup failed (not fatal)")

    btc_feed.start(); eth_feed.start(); sol_feed.start(); xrp_feed.start(); hype_feed.start()
    _bnb_feed.start(); _ton_feed.start(); _doge_feed.start(); _ada_feed.start()
    ob.start(private_key, api_key_id, [])

    log_file, log_writer = _log_init()

    open_mkts: Dict[str, dict] = {}
    last_market_refresh = 0.0

    try:
        while True:
            now = time.time()

            if now - last_market_refresh >= MARKET_REFRESH_SEC:
                fresh = fetch_open_markets()
                if fresh:
                    open_mkts = fresh
                    ob.set_tickers(list(open_mkts.keys()))
                    log.info(f"open markets: {len(open_mkts)}  "
                             f"({', '.join(open_mkts.keys())})")
                last_market_refresh = now

            # Feed price history (both vol estimator + move tracker)
            for sym in _ASSETS:
                p = _price_for_asset(sym)
                if p is not None and _price_age_for_asset(sym) < 5.0:
                    record_price(sym, p)
                    _record_move(sym, p)

            for ticker, market in list(open_mkts.items()):
                asset = _asset_for_ticker(ticker)
                if asset is None:
                    continue
                px = _price_for_asset(asset)
                if px is None:
                    continue
                if _price_age_for_asset(asset) > 5.0:
                    continue

                close_dt = _parse_close(market)
                if close_dt is None:
                    continue
                minutes_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60.0
                if minutes_left < MIN_MINUTES_LEFT or minutes_left > MAX_MINUTES_LEFT:
                    continue

                book = ob.get_book(ticker)
                if book is None or not book.snapshot_seen:
                    continue
                if book.age() > 5.0:
                    continue

                _maybe_snipe(
                    ticker, market, book, px, minutes_left,
                    log_writer, log_file, private_key, api_key_id,
                )

            time.sleep(TICK_SECONDS)

    except KeyboardInterrupt:
        log.info("sniper_v2 stopped")
    finally:
        try: log_file.close()
        except Exception: pass


if __name__ == "__main__":
    main()
