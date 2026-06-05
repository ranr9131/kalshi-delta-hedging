"""
Multi-account sniper.  Same strategy as sniper.py (V1) but fans out each
fire across multiple Kalshi accounts with pro-rata stake allocation.

Use case: a partnership where you and a friend each have your own Kalshi
account.  Both accounts trade the SAME signals at the SAME instant.  Each
account's stake on a given fire is proportional to its current balance,
so over time each account's PnL naturally reflects its share of capital
— no inter-account transfers ever needed.

Configuration via env vars (suffix _<LABEL> per account):
  ACCOUNTS=LEO,FRIEND
  LEO_KEY_ID=...
  LEO_PRIVATE_KEY=...           (PEM, \\n-escaped if single-line)
  FRIEND_KEY_ID=...
  FRIEND_PRIVATE_KEY=...

  TOTAL_STAKE_PER_SNIPE=10      # combined stake across all accounts
  TOTAL_DAILY_LOSS_LIMIT=200    # combined across all accounts
  PAPER_MODE=false              # set "true" to disable real orders for testing

  (Everything else — SERIES, gates, calibration, etc. — same as sniper.py)

Behavior:
  - At startup and every 60s thereafter, polls each account's balance.
  - Per-snipe stake for account A = TOTAL_STAKE × (balance_A / sum_balances)
  - Both IOCs fire in parallel threads — both arrive at Kalshi within ms.
  - Per-account CSV at snipes_<label>.csv.
  - Per-account daily loss limit = TOTAL_LIMIT × (balance_A / sum_balances).
  - If only one account configured, behaves identically to sniper.py.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Deque, Tuple

import requests
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kalshi_auth
import kalshi_orderbook as ob
import kalshi_trade
import btc_feed, eth_feed, sol_feed, xrp_feed, hype_feed
from coinbase_feeds import make_feed
# Generic feeds for newer coins (same pattern as sniper_v2)
_bnb_feed  = make_feed("BNB-USD")
_doge_feed = make_feed("DOGE-USD")
# V1 pricing: constant per-asset σ + calibration.json (historical fit).
# Note: directional gate + opposite-side guard from V2 stay active (built
# into sniper_multi itself, not the model).  This gives a hybrid: V1 model
# without V1's worst failure modes.
from fair_price_model import fair_p_yes, fair_p_no, ASSET_VOL_PER_MIN
# No-op stub so the tick loop's _record_price_for_vol(...) call still works
def _record_price_for_vol(asset, price): pass


# ── Config (shared) ─────────────────────────────────────────────────────────

PAPER_MODE              = os.environ.get("PAPER_MODE", "true").lower() != "false"
MIN_EDGE_CENTS          = int(os.environ.get("MIN_EDGE_CENTS", "5"))
TOTAL_STAKE_PER_SNIPE   = float(os.environ.get("TOTAL_STAKE_PER_SNIPE", "10.0"))
TOTAL_DAILY_LOSS_LIMIT  = float(os.environ.get("TOTAL_DAILY_LOSS_LIMIT", "200.0"))
MIN_MINUTES_LEFT        = float(os.environ.get("MIN_MINUTES_LEFT", "0.5"))
MAX_MINUTES_LEFT        = float(os.environ.get("MAX_MINUTES_LEFT", "14.0"))
TICK_SECONDS            = float(os.environ.get("TICK_SECONDS", "0.05"))
MIN_FAIR_P              = float(os.environ.get("MIN_FAIR_P", "0.15"))
MAX_FAIR_P              = float(os.environ.get("MAX_FAIR_P", "0.85"))
MIN_LEVEL_AGE_SEC       = float(os.environ.get("MIN_LEVEL_AGE_SEC", "2.0"))
MOVE_WINDOW_SEC         = float(os.environ.get("MOVE_WINDOW_SEC",   "3.0"))
MOVE_GATE_MULT          = float(os.environ.get("MOVE_GATE_MULT",    "1.2"))

BALANCE_REFRESH_SEC     = float(os.environ.get("BALANCE_REFRESH_SEC", "60"))
SERIES                  = os.environ.get(
    "SERIES", "KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXHYPE15M"
).split(",")
MARKET_REFRESH_SEC      = 20.0
SNIPE_COOLDOWN_SEC      = 3.0

ROOT      = os.path.dirname(os.path.abspath(__file__))
ENV_PATH  = os.path.join(ROOT, ".env")
KALSHI_BASE = "https://api.elections.kalshi.com"


def _min_move_bps_for(asset: str) -> float:
    override = os.environ.get(f"MIN_MOVE_BPS_{asset.upper()}")
    if override:
        return float(override)
    sigma = ASSET_VOL_PER_MIN.get(asset.upper(), 0.0015)
    return sigma * math.sqrt(MOVE_WINDOW_SEC / 60.0) * 10000.0 * MOVE_GATE_MULT

_MIN_MOVE_BPS = {a: _min_move_bps_for(a) for a in ("BTC","ETH","SOL","XRP","HYPE","BNB","DOGE")}


# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sniper_multi")


# ── Account model ───────────────────────────────────────────────────────────

@dataclass
class Account:
    label: str
    key_id: str
    private_key: object  # loaded RSA key
    log_path: str
    log_file: object = None
    log_writer: object = None
    # Balance + ratio (refreshed periodically)
    balance: float = 0.0
    ratio:   float = 0.0   # share of total capital, 0..1
    last_balance_ts: float = 0.0
    # Per-account loss-limit cache
    today_pnl: float = 0.0
    pnl_cache_ts: float = 0.0
    pnl_cache_date: str = ""
    breached: bool = False


LOG_FIELDS = [
    "ts_iso", "mode", "account", "ticker", "asset", "side",
    "limit_cents", "fill_cents_est", "qty", "stake_dollars",
    "fair_p", "edge_cents", "crypto_price", "strike",
    "minutes_left", "yes_bid", "yes_ask", "no_bid", "no_ask",
    "lvl_age_sec", "move_bps",
    "result", "settled_pnl",
]


def load_accounts() -> List[Account]:
    """Read account list from env.  Order matters — used for ratio computation
    and for the systemd unit's logs to be deterministic."""
    raw_env = dotenv_values(ENV_PATH)
    labels_csv = os.environ.get("ACCOUNTS") or raw_env.get("ACCOUNTS", "")
    labels = [s.strip() for s in labels_csv.split(",") if s.strip()]
    if not labels:
        # Fallback: single account using existing KALSHI_* envs (back-compat).
        labels = ["DEFAULT"]
    accounts: List[Account] = []
    for label in labels:
        if label == "DEFAULT":
            key_id  = os.environ.get("KALSHI_API_KEY_ID")  or raw_env.get("KALSHI_API_KEY_ID")
            pem     = os.environ.get("KALSHI_PRIVATE_KEY") or raw_env.get("KALSHI_PRIVATE_KEY")
        else:
            key_id = os.environ.get(f"{label}_KEY_ID") or raw_env.get(f"{label}_KEY_ID")
            pem    = os.environ.get(f"{label}_PRIVATE_KEY") or raw_env.get(f"{label}_PRIVATE_KEY")
        if not key_id or not pem:
            log.error(f"missing key_id or private_key for account '{label}'")
            continue
        priv = kalshi_auth.load_private_key(pem)
        log_path = os.path.join(ROOT, f"snipes_{label.lower()}.csv")
        accounts.append(Account(label=label, key_id=key_id, private_key=priv,
                                log_path=log_path))
    return accounts


def _open_csv(acct: Account):
    new = not os.path.exists(acct.log_path)
    f = open(acct.log_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
    if new:
        w.writeheader()
        f.flush()
    acct.log_file = f
    acct.log_writer = w


# ── Balance / ratio polling ─────────────────────────────────────────────────

def refresh_balances(accounts: List[Account]) -> None:
    """Query Kalshi for each account's current balance.  Update each
    account's `balance` and recompute `ratio` so they sum to 1.0."""
    for acct in accounts:
        try:
            bal = kalshi_trade.get_balance(acct.private_key, acct.key_id)
        except Exception as e:
            log.warning(f"balance fetch failed for {acct.label}: {e}")
            bal = None
        if bal is not None:
            acct.balance = bal
            acct.last_balance_ts = time.time()
    total = sum(max(0.0, a.balance) for a in accounts)
    if total > 0:
        for acct in accounts:
            acct.ratio = max(0.0, acct.balance) / total
    else:
        # If we somehow can't read any balance, fall back to equal split.
        for acct in accounts:
            acct.ratio = 1.0 / max(1, len(accounts))


# ── Per-account daily loss limit ────────────────────────────────────────────

SETTLE_PATH = os.path.join(ROOT, "settlements.csv")


def _today_realized_pnl(acct: Account) -> float:
    """Realized PnL for this account today UTC.  Cached 15s.  Reads the
    per-account CSV (snipes_<label>.csv)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != acct.pnl_cache_date:
        acct.pnl_cache_ts = 0.0
    if time.time() - acct.pnl_cache_ts < 15.0:
        return acct.today_pnl

    settlements: Dict[str, dict] = {}
    if os.path.exists(SETTLE_PATH):
        try:
            with open(SETTLE_PATH, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("ticker"):
                        settlements[r["ticker"]] = r
        except Exception:
            pass

    pnl = 0.0
    if os.path.exists(acct.log_path):
        try:
            with open(acct.log_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("mode") != "live":
                        continue
                    if not r.get("ts_iso", "").startswith(today):
                        continue
                    s = settlements.get(r.get("ticker", ""))
                    if not s:
                        continue
                    result = (s.get("result") or "").lower()
                    if result not in ("yes", "no"):
                        continue
                    try:
                        qty   = float(r.get("qty") or 0)
                        stake = float(r.get("stake_dollars") or 0)
                    except Exception:
                        continue
                    pnl += (qty - stake) if r.get("side") == result else -stake
        except Exception:
            pass

    acct.today_pnl = pnl
    acct.pnl_cache_ts = time.time()
    acct.pnl_cache_date = today
    return pnl


def _loss_limit_ok(acct: Account) -> bool:
    """Account-level limit = TOTAL_LIMIT × this account's current ratio."""
    if TOTAL_DAILY_LOSS_LIMIT <= 0:
        return True
    limit = TOTAL_DAILY_LOSS_LIMIT * max(acct.ratio, 0.0)
    pnl = _today_realized_pnl(acct)
    breached = pnl <= -limit
    if breached and not acct.breached:
        log.warning(f"[{acct.label}] daily loss breached: pnl=$%+.2f <= -$%.2f. "
                    f"Halting fires for this account until UTC midnight." % (pnl, limit))
        acct.breached = True
    elif not breached and acct.breached:
        log.info(f"[{acct.label}] loss limit cleared (pnl=$%+.2f), resuming" % pnl)
        acct.breached = False
    return not breached


# ── Asset helpers (same as sniper.py) ───────────────────────────────────────

SERIES_TO_ASSET = {
    "KXBTC15M":  "BTC",  "KXETH15M": "ETH",
    "KXSOL15M":  "SOL",  "KXXRP15M": "XRP",
    "KXHYPE15M": "HYPE",
    "KXBNB15M":  "BNB",  "KXDOGE15M": "DOGE",
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
    if asset == "DOGE": return _doge_feed.get_price()
    return None


def _price_age_for_asset(asset: str) -> float:
    if asset == "BTC":  return btc_feed.get_price_age()
    if asset == "ETH":  return eth_feed.get_price_age()
    if asset == "SOL":  return sol_feed.get_price_age()
    if asset == "XRP":  return xrp_feed.get_price_age()
    if asset == "HYPE": return hype_feed.get_price_age()
    if asset == "BNB":  return _bnb_feed.get_price_age()
    if asset == "DOGE": return _doge_feed.get_price_age()
    return float("inf")


def fetch_open_markets() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for series in SERIES:
        try:
            r = requests.get(f"{KALSHI_BASE}/trade-api/v2/markets",
                             params={"series_ticker": series, "status": "open", "limit": 50},
                             timeout=10)
            r.raise_for_status()
            for mk in r.json().get("markets", []):
                if mk.get("ticker"):
                    out[mk["ticker"]] = mk
        except Exception as e:
            log.warning(f"market fetch {series}: {e}")
    return out


def _parse_close(market: dict) -> Optional[datetime]:
    ts = market.get("close_time") or market.get("expected_expiration_time")
    if not ts: return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _strike(market: dict) -> Optional[float]:
    for k in ("floor_strike", "strike", "strike_price"):
        v = market.get(k)
        if v is not None:
            try: return float(v)
            except Exception: pass
    return None


# ── Recent-move tracking (same as sniper.py) ────────────────────────────────

_price_hist: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
    lambda: deque(maxlen=200))

def _record_price(asset: str, price: float) -> None:
    now = time.time()
    buf = _price_hist[asset]
    buf.append((now, price))
    while buf and now - buf[0][0] > 10.0:
        buf.popleft()

def _recent_move_bps(asset: str, window_sec: float) -> float:
    """Magnitude of recent move in bps (always >= 0).  Used by the move-
    threshold gate."""
    buf = _price_hist[asset]
    if len(buf) < 2: return 0.0
    now_ts, now_px = buf[-1]
    if now_px <= 0: return 0.0
    cutoff = now_ts - window_sec
    ref = next((p for t, p in buf if t >= cutoff), None)
    if ref is None: return 0.0
    return abs(now_px - ref) / now_px * 10000.0

def _recent_move_signed_bps(asset: str, window_sec: float) -> float:
    """Signed move in bps over `window_sec`.  +ve = price went up, -ve = down.
    Used by the DIRECTIONAL gate: YES side only fires on +ve move, NO side
    only on -ve.  Stops us from buying YES into a falling market (the
    failure mode that produced the 16% WR hour on 2026-06-05 14:00 UTC)."""
    buf = _price_hist[asset]
    if len(buf) < 2: return 0.0
    now_ts, now_px = buf[-1]
    if now_px <= 0: return 0.0
    cutoff = now_ts - window_sec
    ref = next((p for t, p in buf if t >= cutoff), None)
    if ref is None: return 0.0
    return (now_px - ref) / now_px * 10000.0


# Track in-flight exposure per ticker so we don't fire BOTH sides on the
# same market in quick succession (the same-market opposite-side bug).
# Key: ticker → set of sides we've fired today.  Resets at UTC midnight.
_sides_fired: Dict[str, set] = defaultdict(set)
_sides_fired_date: str = ""

def _check_and_record_side(ticker: str, side: str) -> bool:
    """Return True if firing `side` on `ticker` is allowed.  Blocks if we've
    already fired the opposite side today on the same market."""
    global _sides_fired_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _sides_fired_date:
        _sides_fired.clear()
        _sides_fired_date = today
    opp = "no" if side == "yes" else "yes"
    if opp in _sides_fired[ticker]:
        return False  # blocked — opposite side already in play
    _sides_fired[ticker].add(side)
    return True


# ── Per-account order placement (parallel) ──────────────────────────────────

def _place_for_account(acct: Account, market: dict, side: str, limit_d: float,
                       stake: float, qty_hint: float, fair_p: float, edge_c: float,
                       crypto_price: float, strike: float, minutes_left: float,
                       book: ob.Book, lvl_age: float, mv_bps: float):
    """Fire IOC for one account.  Independently logs to its own CSV."""
    if not _loss_limit_ok(acct):
        return  # this account is halted

    # Real order
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
                acct.private_key, acct.key_id, market["ticker"], side, market,
                stake_dollars=stake, extra_buffer_cents=extra, ioc=True,
            )
        except Exception as e:
            log.warning(f"[{acct.label}] order failed: {e}")

    # CSV log (same row whether paper or live — fills get reconciled later)
    acct.log_writer.writerow({
        "ts_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode":   "paper" if PAPER_MODE else "live",
        "account": acct.label,
        "ticker": market["ticker"], "asset": _asset_for_ticker(market["ticker"]),
        "side":   side,
        "limit_cents": round(limit_d * 100, 2),
        "fill_cents_est": round((stake/qty_hint*100) if qty_hint else 0, 2),
        "qty": round(qty_hint, 3), "stake_dollars": round(stake, 2),
        "fair_p": round(fair_p, 4), "edge_cents": round(edge_c, 2),
        "crypto_price": round(crypto_price, 4),
        "strike": strike, "minutes_left": round(minutes_left, 3),
        "yes_bid": book.yes_bid(), "yes_ask": book.yes_ask(),
        "no_bid":  book.no_bid(),  "no_ask":  book.no_ask(),
        "lvl_age_sec": round(lvl_age, 2), "move_bps": round(mv_bps, 2),
        "result": "", "settled_pnl": "",
    })
    acct.log_file.flush()


_last_snipe_ts: Dict[tuple, float] = defaultdict(float)


def _maybe_snipe(ticker: str, market: dict, book: ob.Book,
                 crypto_price: float, minutes_left: float,
                 accounts: List[Account]):
    asset  = _asset_for_ticker(ticker)
    strike = _strike(market)
    if asset is None or strike is None:
        return

    fpy = fair_p_yes(crypto_price, strike, minutes_left, asset)
    fpn = 1.0 - fpy
    yes_ask = book.yes_ask(); no_ask = book.no_ask()
    edge_dollars = MIN_EDGE_CENTS / 100.0

    # Gate 1: magnitude — recent crypto move must be at least typical-vol size.
    # (Directional gate REMOVED per user request — both YES and NO can fire
    #  regardless of move sign, as long as magnitude is sufficient.)
    move_bps = _recent_move_bps(asset, MOVE_WINDOW_SEC)
    if move_bps < _MIN_MOVE_BPS.get(asset, 5.0):
        return

    snipes = []  # (side, limit_d, fair_p, edge_c, lvl_age)
    # YES side
    if (yes_ask is not None and yes_ask >= 0.01
            and MIN_FAIR_P <= fpy <= MAX_FAIR_P):
        max_pay = fpy - edge_dollars
        if yes_ask <= max_pay and book.yes_ask_age() >= MIN_LEVEL_AGE_SEC:
            snipes.append(("yes", round(min(0.99, max_pay), 4), fpy,
                           (fpy - yes_ask) * 100.0, book.yes_ask_age()))
    # NO side
    if (no_ask is not None and no_ask >= 0.01
            and MIN_FAIR_P <= fpn <= MAX_FAIR_P):
        max_pay = fpn - edge_dollars
        if no_ask <= max_pay and book.no_ask_age() >= MIN_LEVEL_AGE_SEC:
            snipes.append(("no", round(min(0.99, max_pay), 4), fpn,
                           (fpn - no_ask) * 100.0, book.no_ask_age()))

    now = time.time()
    for side, limit_d, fair_p, edge_c, lvl_age in snipes:
        key = (ticker, side)
        if now - _last_snipe_ts[key] < SNIPE_COOLDOWN_SEC:
            continue
        # Block if we already fired the OPPOSITE side on this ticker today
        # (avoids accumulating self-hedged-net-loss exposure when the
        # underlying whipsaws across our fair).
        if not _check_and_record_side(ticker, side):
            log.info(f"  skipping {ticker} {side}: opposite side already fired today")
            continue
        _last_snipe_ts[key] = now

        # How many contracts available at-or-below our limit
        ladder = book.yes_asks_sorted() if side == "yes" else book.no_asks_sorted()
        avail_qty = sum(q for p, q in ladder if p <= limit_d)
        if avail_qty <= 0:
            continue
        total_qty_by_stake = TOTAL_STAKE_PER_SNIPE / max(limit_d, 0.01)
        total_qty = max(1.0, min(avail_qty, total_qty_by_stake))

        # Estimate fill (walk the ladder)
        filled = 0.0; cost = 0.0
        for p, q in ladder:
            if p > limit_d or filled >= total_qty: break
            take = min(q, total_qty - filled)
            filled += take; cost += take * p
        if filled <= 0:
            continue
        avg_fill_d = cost / filled

        log.info(
            f"SNIPE  {ticker}  {side.upper()}  total_qty={filled:.1f}  "
            f"avg_fill=${avg_fill_d:.4f}  fair={fair_p*100:.1f}¢  "
            f"edge={edge_c:.1f}¢  mv={move_bps:.1f}bps  lvl_age={lvl_age:.1f}s  "
            f"[{'PAPER' if PAPER_MODE else 'LIVE'}]"
        )

        # Fan out across accounts: each gets stake proportional to its ratio
        threads = []
        for acct in accounts:
            if acct.ratio <= 0:
                continue
            acct_stake = round(cost * acct.ratio, 4)   # this account's $ of the fill
            acct_qty   = round(filled * acct.ratio, 4)
            if acct_stake < 0.01 or acct_qty < 0.1:
                continue
            log.info(f"  → {acct.label}: stake=${acct_stake:.2f}  qty={acct_qty:.2f}  "
                     f"(ratio={acct.ratio*100:.0f}%, bal=${acct.balance:.0f})")
            t = threading.Thread(
                target=_place_for_account,
                args=(acct, market, side, limit_d, acct_stake, acct_qty,
                      fair_p, edge_c, crypto_price, strike, minutes_left,
                      book, lvl_age, move_bps),
                daemon=True, name=f"place-{acct.label}",
            )
            threads.append(t)
            t.start()
        # Don't .join() — they fire-and-forget so the next tick isn't blocked
        # on HTTPS round-trips.  Daemon threads exit with the process.


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("sniper_multi start  PAPER_MODE=%s  total_stake=$%.2f  "
             "total_loss_limit=$%.0f  tick=%.0fms",
             PAPER_MODE, TOTAL_STAKE_PER_SNIPE, TOTAL_DAILY_LOSS_LIMIT,
             TICK_SECONDS*1000)
    log.info("series=%s", SERIES)
    log.info("move thresholds (bps/3s): %s",
             {a: round(v,2) for a,v in _MIN_MOVE_BPS.items()})
    log.info("coinflip filter: %.2f <= fair_p <= %.2f", MIN_FAIR_P, MAX_FAIR_P)
    log.info("directional gate OFF: both sides eligible regardless of move direction")
    log.info("opposite-side guard ON: won't fire NO on a ticker if we've fired YES today")
    log.info("pricing model: V1 (constant σ + calibration.json)")

    accounts = load_accounts()
    if not accounts:
        log.error("no accounts loaded — set ACCOUNTS=LABEL1,LABEL2 with "
                  "matching <LABEL>_KEY_ID and <LABEL>_PRIVATE_KEY env vars")
        sys.exit(1)
    log.info(f"loaded {len(accounts)} account(s): {[a.label for a in accounts]}")

    for acct in accounts:
        _open_csv(acct)
        log.info(f"  [{acct.label}] log → {acct.log_path}")

    refresh_balances(accounts)
    for acct in accounts:
        log.info(f"  [{acct.label}] balance=$%.2f  ratio=%.1f%%" % (
            acct.balance, acct.ratio*100))

    if kalshi_trade.warmup_session():
        log.info("kalshi session pre-warmed")

    # Crypto feeds + Kalshi orderbook
    btc_feed.start(); eth_feed.start(); sol_feed.start(); xrp_feed.start(); hype_feed.start()
    _bnb_feed.start(); _doge_feed.start()
    # Use the FIRST account's credentials for the WS orderbook subscription
    # (it's just an authenticated read; doesn't depend on which account).
    ob.start(accounts[0].private_key, accounts[0].key_id, [])

    open_mkts: Dict[str, dict] = {}
    last_market_refresh = 0.0
    last_balance_refresh = time.time()

    try:
        while True:
            now = time.time()

            if now - last_market_refresh >= MARKET_REFRESH_SEC:
                fresh = fetch_open_markets()
                if fresh:
                    open_mkts = fresh
                    ob.set_tickers(list(open_mkts.keys()))
                    log.info(f"open markets: {len(open_mkts)}")
                last_market_refresh = now

            if now - last_balance_refresh >= BALANCE_REFRESH_SEC:
                refresh_balances(accounts)
                last_balance_refresh = now

            for sym in ("BTC","ETH","SOL","XRP","HYPE","BNB","DOGE"):
                p = _price_for_asset(sym)
                if p is not None and _price_age_for_asset(sym) < 5.0:
                    _record_price(sym, p)            # short window for move gate
                    _record_price_for_vol(sym, p)    # long window for realized σ

            for ticker, market in list(open_mkts.items()):
                asset = _asset_for_ticker(ticker)
                if asset is None: continue
                px = _price_for_asset(asset)
                if px is None: continue
                if _price_age_for_asset(asset) > 5.0: continue
                close_dt = _parse_close(market)
                if close_dt is None: continue
                minutes_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60.0
                if minutes_left < MIN_MINUTES_LEFT or minutes_left > MAX_MINUTES_LEFT:
                    continue
                book = ob.get_book(ticker)
                if book is None or not book.snapshot_seen: continue
                if book.age() > 5.0: continue
                _maybe_snipe(ticker, market, book, px, minutes_left, accounts)

            time.sleep(TICK_SECONDS)

    except KeyboardInterrupt:
        log.info("sniper_multi stopped")
    finally:
        for acct in accounts:
            try: acct.log_file.close()
            except Exception: pass


if __name__ == "__main__":
    main()
