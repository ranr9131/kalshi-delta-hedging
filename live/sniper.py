"""
Kalshi 15M crypto stale-order sniper.

Strategy:
  - Subscribe to live crypto prices (BTC, ETH, SOL, XRP via Coinbase WS).
  - Subscribe to the full Kalshi orderbook for each currently-open 15M
    market across all 4 series.
  - On every loop tick (~200 ms):
      * Compute fair P(YES) from current crypto price, strike, minutes left.
      * Look at the cheapest YES ask in the book.
      * If fair_p_yes * 100 - yes_ask_cents >= MIN_EDGE_CENTS  →  snipe:
        fire an IOC buy at limit = round(fair*100 - EDGE_BUFFER) so the
        matching engine sweeps every mispriced level in one shot.
      * Mirror logic for NO side.

  - PAPER_MODE=true (default): we never place real orders.  We log each
    snipe to snipes.csv with the price we would have paid, then settle at
    market close from the actual result (or current crypto price if we
    re-run later).

Env vars (with defaults):
  PAPER_MODE          true       — must be exactly "false" to go live
  MIN_EDGE_CENTS      5          — required (fair_cents − ask_cents)
  MAX_STAKE_PER_SNIPE 5.0        — dollars risked per snipe
  MIN_MINUTES_LEFT    0.5        — skip markets w/ < 30 s to close
  MAX_MINUTES_LEFT    14         — skip the first ~1 min (book settling)
  SERIES              KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M
  TICK_SECONDS        0.2
  LOG_PATH            snipes.csv
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
import btc_feed, eth_feed, sol_feed, xrp_feed
from fair_price_model import fair_p_yes, fair_p_no, ASSET_VOL_PER_MIN


# ── Config ──────────────────────────────────────────────────────────────────

PAPER_MODE          = os.environ.get("PAPER_MODE", "true").lower() != "false"
MIN_EDGE_CENTS      = int(os.environ.get("MIN_EDGE_CENTS", "5"))
MAX_STAKE_PER_SNIPE = float(os.environ.get("MAX_STAKE_PER_SNIPE", "5.0"))
MIN_MINUTES_LEFT    = float(os.environ.get("MIN_MINUTES_LEFT", "0.5"))
MAX_MINUTES_LEFT    = float(os.environ.get("MAX_MINUTES_LEFT", "14.0"))
TICK_SECONDS        = float(os.environ.get("TICK_SECONDS", "0.2"))

# ── Coinflip-zone filter ────────────────────────────────────────────────────
# Only fire when our model's fair probability sits inside this range.  Outside
# the range:
#   - fair_p < MIN_FAIR_P  →  longshot territory where the model's tail-mass
#     overestimate dominates and bleed risk is highest
#   - fair_p > MAX_FAIR_P  →  near-certainty bets where edge ≥ 5¢ is rare and
#     marginal anyway
# The middle gives a measurable win-rate signal within ~tens of fires.
MIN_FAIR_P = float(os.environ.get("MIN_FAIR_P", "0.30"))
MAX_FAIR_P = float(os.environ.get("MAX_FAIR_P", "0.70"))

# ── Sniper gates (filter out "model disagreement" noise) ────────────────────
# A snipe only fires when BOTH gates hold:
#   1. The best ask level has been resting ≥ MIN_LEVEL_AGE_SEC (genuinely stale,
#      not freshly placed by a maker who's actively quoting).
#   2. The underlying crypto has moved ≥ MIN_RECENT_MOVE_BPS_<asset> in the last
#      MOVE_WINDOW_SEC (real new info that the resting quote hasn't priced in).
#
# The recent-move threshold is *per-asset*: a 5bps move in 3s is barely above
# noise on SOL but is a ~2σ event on BTC.  We derive each asset's threshold
# from its per-minute vol so all 4 assets have a similar "rarity" of firing.
#   typical_move_bps_3s = σ_per_min × √(window/60) × 10000
#   threshold = typical_move_bps × MOVE_GATE_MULT
# Override per asset with MIN_MOVE_BPS_BTC / _ETH / _SOL / _XRP if needed.
MIN_LEVEL_AGE_SEC = float(os.environ.get("MIN_LEVEL_AGE_SEC", "2.0"))
MOVE_WINDOW_SEC   = float(os.environ.get("MOVE_WINDOW_SEC",   "3.0"))
MOVE_GATE_MULT    = float(os.environ.get("MOVE_GATE_MULT",    "1.2"))


def _min_move_bps_for(asset: str) -> float:
    override = os.environ.get(f"MIN_MOVE_BPS_{asset.upper()}")
    if override:
        return float(override)
    sigma = ASSET_VOL_PER_MIN.get(asset.upper(), 0.0015)
    typical_bps = sigma * math.sqrt(MOVE_WINDOW_SEC / 60.0) * 10000.0
    return typical_bps * MOVE_GATE_MULT


_MIN_MOVE_BPS = {a: _min_move_bps_for(a) for a in ("BTC", "ETH", "SOL", "XRP")}
SERIES              = os.environ.get(
    "SERIES", "KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M"
).split(",")
LOG_PATH            = os.environ.get(
    "LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "snipes.csv"),
)
MARKET_REFRESH_SEC  = 20.0   # how often to re-poll the open-markets list
SNIPE_COOLDOWN_SEC  = 3.0    # don't re-snipe same (ticker,side) within N sec

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

KALSHI_BASE = "https://api.elections.kalshi.com"


# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sniper")


# ── Series → asset symbol ───────────────────────────────────────────────────

SERIES_TO_ASSET = {
    "KXBTC15M": "BTC",
    "KXETH15M": "ETH",
    "KXSOL15M": "SOL",
    "KXXRP15M": "XRP",
}


def _asset_for_ticker(ticker: str) -> Optional[str]:
    for prefix, sym in SERIES_TO_ASSET.items():
        if ticker.startswith(prefix):
            return sym
    return None


def _price_for_asset(asset: str) -> Optional[float]:
    if asset == "BTC": return btc_feed.get_price()
    if asset == "ETH": return eth_feed.get_price()
    if asset == "SOL": return sol_feed.get_price()
    if asset == "XRP": return xrp_feed.get_price()
    return None


def _price_age_for_asset(asset: str) -> float:
    if asset == "BTC": return btc_feed.get_price_age()
    if asset == "ETH": return eth_feed.get_price_age()
    if asset == "SOL": return sol_feed.get_price_age()
    if asset == "XRP": return xrp_feed.get_price_age()
    return float("inf")


# ── Market discovery ────────────────────────────────────────────────────────

def fetch_open_markets() -> Dict[str, dict]:
    """
    Return {ticker: market} for every currently-open 15M crypto market
    across the configured series.  No auth required.
    """
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
    """Lower bound of the binary range.  KXBTC15M-...-30 means ≥ strike."""
    for k in ("floor_strike", "strike", "strike_price"):
        v = market.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # Try parsing from ticker suffix as a last resort
    try:
        tail = market.get("ticker", "").split("-")[-1]
        return float(tail)
    except Exception:
        return None


# ── Snipe log ───────────────────────────────────────────────────────────────

LOG_FIELDS = [
    "ts_iso", "mode", "ticker", "asset", "side",
    "limit_cents", "fill_cents_est", "qty", "stake_dollars",
    "fair_p", "edge_cents", "crypto_price", "strike",
    "minutes_left", "yes_bid", "yes_ask", "no_bid", "no_ask",
    "lvl_age_sec", "move_bps",
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


# ── Snipe execution ─────────────────────────────────────────────────────────

_last_snipe_ts: Dict[tuple, float] = defaultdict(float)

# Rolling crypto price history per asset: deque[(ts, price)] last 10 sec.
_price_hist: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
    lambda: deque(maxlen=200)
)

def _record_price(asset: str, price: float) -> None:
    """Append current price to rolling buffer; prune entries > 10s old."""
    now = time.time()
    buf = _price_hist[asset]
    buf.append((now, price))
    while buf and now - buf[0][0] > 10.0:
        buf.popleft()

def _recent_move_bps(asset: str, window_sec: float) -> float:
    """abs(price_now - price_window_ago) / price_now in basis points (1bp = 0.01%)."""
    buf = _price_hist[asset]
    if len(buf) < 2:
        return 0.0
    now_ts, now_px = buf[-1]
    if now_px <= 0:
        return 0.0
    cutoff = now_ts - window_sec
    # Find oldest entry within window
    ref_px = None
    for ts, px in buf:
        if ts >= cutoff:
            ref_px = px
            break
    if ref_px is None:
        return 0.0
    return abs(now_px - ref_px) / now_px * 10000.0


def _maybe_snipe(
    ticker: str, market: dict, book: ob.Book,
    crypto_price: float, minutes_left: float,
    log_writer, log_file, private_key, api_key_id,
):
    """All prices in float dollars (probability units).  We log edge in
    cents because that's the human-readable unit."""
    asset  = _asset_for_ticker(ticker)
    strike = _strike(market)
    if asset is None or strike is None:
        return

    fpy = fair_p_yes(crypto_price, strike, minutes_left, asset)
    fpn = 1.0 - fpy

    yes_ask = book.yes_ask()   # $ to BUY YES, or None
    no_ask  = book.no_ask()    # $ to BUY NO, or None

    edge_dollars = MIN_EDGE_CENTS / 100.0

    # ── Gate 2: recent crypto move (real new info) ─────────────────────
    move_bps = _recent_move_bps(asset, MOVE_WINDOW_SEC)
    move_gate_ok = move_bps >= _MIN_MOVE_BPS.get(asset, 5.0)

    snipes = []  # (side, limit_dollars, fair_p, edge_cents, level_age, move_bps)

    # ── YES side ───────────────────────────────────────────────────────
    # Coinflip-zone filter: only fire if our model's fair sits in [MIN, MAX].
    # Outside that band the per-side fair has either too much downside (model
    # error dominates EV) or too little upside (5¢ edge isn't enough payoff).
    if yes_ask is not None and yes_ask >= 0.01 and MIN_FAIR_P <= fpy <= MAX_FAIR_P:
        max_pay_yes = fpy - edge_dollars
        if yes_ask <= max_pay_yes:
            # Gate 1: best YES ask level has been resting long enough.
            lvl_age = book.yes_ask_age()
            if move_gate_ok and lvl_age >= MIN_LEVEL_AGE_SEC:
                limit_d = min(0.99, max_pay_yes)
                snipes.append(("yes", round(limit_d, 4), fpy,
                               (fpy - yes_ask) * 100.0, lvl_age, move_bps))

    # ── NO side ────────────────────────────────────────────────────────
    if no_ask is not None and no_ask >= 0.01 and MIN_FAIR_P <= fpn <= MAX_FAIR_P:
        max_pay_no = fpn - edge_dollars
        if no_ask <= max_pay_no:
            lvl_age = book.no_ask_age()
            if move_gate_ok and lvl_age >= MIN_LEVEL_AGE_SEC:
                limit_d = min(0.99, max_pay_no)
                snipes.append(("no", round(limit_d, 4), fpn,
                               (fpn - no_ask) * 100.0, lvl_age, move_bps))

    now = time.time()
    for side, limit_d, fair_p, edge_c, lvl_age, mv_bps in snipes:
        key = (ticker, side)
        if now - _last_snipe_ts[key] < SNIPE_COOLDOWN_SEC:
            continue
        _last_snipe_ts[key] = now

        ladder = book.yes_asks_sorted() if side == "yes" else book.no_asks_sorted()
        # Walk the ladder up to limit_d to estimate fill
        avail_qty = sum(q for p, q in ladder if p <= limit_d)
        if avail_qty <= 0:
            continue
        # Cap by stake budget @ worst-case fill = limit_d
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
        qty_int = int(round(filled))

        log.info(
            f"SNIPE  {ticker}  {side.upper()}  limit=${limit_d:.4f}  "
            f"fill_est=${avg_fill_d:.4f}  qty={filled:.1f}  "
            f"fair={fair_p*100:.1f}¢  edge={edge_c:.1f}¢  "
            f"lvl_age={lvl_age:.1f}s  mv={mv_bps:.1f}bps  "
            f"px={crypto_price:.2f} strike={strike} t={minutes_left:.2f}m  "
            f"[{'PAPER' if PAPER_MODE else 'LIVE'}]"
        )

        # Real-order path: convert to integer cents for Kalshi REST.
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

        # Record snipe
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
            "move_bps":    round(mv_bps, 2),
            "result": "", "settled_pnl": "",
        })
        log_file.flush()


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    env = dotenv_values(ENV_PATH)
    private_key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    api_key_id  = env["KALSHI_API_KEY_ID"]

    log.info(
        f"sniper start  PAPER_MODE={PAPER_MODE}  series={SERIES}  "
        f"min_edge={MIN_EDGE_CENTS}¢  max_stake=${MAX_STAKE_PER_SNIPE}"
    )
    log.info(
        "move thresholds (bps over %.1fs, mult=%.2f): %s",
        MOVE_WINDOW_SEC, MOVE_GATE_MULT,
        {a: round(v, 2) for a, v in _MIN_MOVE_BPS.items()},
    )
    log.info(
        "coinflip filter: %.2f <= fair_p <= %.2f", MIN_FAIR_P, MAX_FAIR_P,
    )

    # Crypto feeds
    btc_feed.start(); eth_feed.start(); sol_feed.start(); xrp_feed.start()

    # Kalshi orderbook WS
    ob.start(private_key, api_key_id, [])

    # Snipe log
    log_file, log_writer = _log_init()

    open_mkts: Dict[str, dict] = {}
    last_market_refresh = 0.0

    try:
        while True:
            now = time.time()

            # ── Refresh open markets every MARKET_REFRESH_SEC ───────────
            if now - last_market_refresh >= MARKET_REFRESH_SEC:
                fresh = fetch_open_markets()
                if fresh:
                    open_mkts = fresh
                    ob.set_tickers(list(open_mkts.keys()))
                    log.info(f"open markets: {len(open_mkts)}  "
                             f"({', '.join(open_mkts.keys())})")
                last_market_refresh = now

            # ── Refresh rolling crypto history once per tick ───────────
            for sym in ("BTC", "ETH", "SOL", "XRP"):
                p = _price_for_asset(sym)
                if p is not None and _price_age_for_asset(sym) < 5.0:
                    _record_price(sym, p)

            # ── Per-market scan ────────────────────────────────────────
            for ticker, market in list(open_mkts.items()):
                asset = _asset_for_ticker(ticker)
                if asset is None:
                    continue
                px = _price_for_asset(asset)
                if px is None:
                    continue
                if _price_age_for_asset(asset) > 5.0:
                    continue  # stale crypto feed, don't trust fair calc

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
        log.info("sniper stopped")
    finally:
        try: log_file.close()
        except Exception: pass


if __name__ == "__main__":
    main()
