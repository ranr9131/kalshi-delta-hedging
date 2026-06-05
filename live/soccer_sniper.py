"""
Soccer goal sniper.

Strategy: when a goal is scored in a live soccer game, immediately buy
YES on the scoring team's Kalshi market BEFORE the market reprices.
Hold the position until either (a) the market reprices favorably and we
hit our take-profit, or (b) the game ends and the market settles.

Event-driven (unlike the crypto sniper which scans continuously):
  - Discovery thread: every 30s, scan ESPN for live soccer games across
    all major leagues we follow.
  - Per-game observer thread: every 1.5s, poll ESPN for the current
    score.  On score change, fire BUY YES IOC on the scoring team's
    Kalshi market.
  - (Optional) Take-profit thread: after a buy, monitor the Kalshi
    market.  If best-bid moves above entry + TAKE_PROFIT_CENTS, fire a
    SELL order (= BUY NO at 100 − sell_price) to lock in the gain.

Leagues monitored (configurable via env):
  EPL, Champions League, La Liga, Bundesliga, Serie A, MLS, NWSL,
  Ligue 1, Europa League, World Cup, Liga Portugal, Liga DIMAYOR,
  Liga MX, FIFA Women's
  → Designed to come alive for the World Cup 2026 starting June 14.

Configuration via env vars:
  PAPER_MODE=true              must be exactly "false" to fire real orders
  STAKE_PER_GOAL=10            $ per snipe (capped by stake AND Kalshi qty available)
  TAKE_PROFIT_CENTS=8          sell if best bid rises this many cents above our fill
  HOLD_FOR_SECONDS=600         only attempt sell within this window; otherwise hold to settlement
  DAILY_LOSS_LIMIT=100         halts after losing this much
  LEAGUES=eng.1,uefa.champions,esp.1,ger.1,ita.1,usa.1,fra.1,fifa.worldcup
  LOG_PATH=...                 default soccer_snipes.csv

No orders placed unless PAPER_MODE=false.  Safe to run alongside any
existing crypto sniper services — they don't interact.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_auth
import kalshi_trade


ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT, ".env")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
KALSHI_BASE = "https://api.elections.kalshi.com"

# ── Config ──────────────────────────────────────────────────────────────────

PAPER_MODE         = os.environ.get("PAPER_MODE", "true").lower() != "false"
STAKE_PER_GOAL     = float(os.environ.get("STAKE_PER_GOAL", "10.0"))
TAKE_PROFIT_CENTS  = int(os.environ.get("TAKE_PROFIT_CENTS", "8"))
HOLD_FOR_SECONDS   = int(os.environ.get("HOLD_FOR_SECONDS", "600"))
DAILY_LOSS_LIMIT   = float(os.environ.get("DAILY_LOSS_LIMIT", "100.0"))
MIN_PRICE_TO_BUY   = float(os.environ.get("MIN_PRICE_TO_BUY",  "0.05"))   # ignore <5¢ asks (basically free options nobody fills)
MAX_PRICE_TO_BUY   = float(os.environ.get("MAX_PRICE_TO_BUY",  "0.85"))   # ignore >85¢ — already priced in
DISCOVERY_INTERVAL = int(os.environ.get("DISCOVERY_INTERVAL_SEC", "30"))
SCORE_POLL_SEC     = float(os.environ.get("SCORE_POLL_SEC", "1.5"))

LEAGUES = os.environ.get(
    "LEAGUES",
    "eng.1,uefa.champions,esp.1,ger.1,ita.1,usa.1,fra.1,fifa.worldcup,"
    "uefa.europa,por.1,col.1,mex.1,fifa.wwc,usa.nwsl"
).split(",")

LOG_PATH = os.environ.get(
    "LOG_PATH", os.path.join(ROOT, "soccer_snipes.csv")
)


# ── Kalshi soccer series patterns to try ────────────────────────────────────
# When a goal happens, we need to find the Kalshi market for that game.
# We try a list of plausible series tickers (Kalshi doesn't have a stable
# naming convention so we cast a wide net).
KALSHI_SOCCER_SERIES = [
    "KXWORLDCUPGAME",   # World Cup (when it starts)
    "KXEPLGAME",        # English Premier League
    "KXUCLGAME",        # UEFA Champions League
    "KXLALIGAGAME",
    "KXBUNDESGAME",
    "KXSERIEAGAME",
    "KXMLSGAME",
    "KXLIGUE1GAME",
    "KXEUROPAGAME",
    "KXLIGAPORTUGALGAME",
    "KXDIMAYORGAME",
    "KXLIGAMXGAME",
    "KXFIFAWGAME",
    "KXNWSLGAME",
    "KXSOCCERGAME",     # generic fallback
]


# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  soccer  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("soccer_sniper")


# ── CSV ─────────────────────────────────────────────────────────────────────

LOG_FIELDS = [
    "ts_iso", "mode", "event_type",
    "league", "game", "scoring_team", "score_after",
    "kalshi_ticker", "side", "limit_cents", "fill_cents_est",
    "qty", "stake_dollars",
    "entry_cents", "exit_cents", "realized_pnl",
    "note",
]

_csv_lock = threading.Lock()
_csv_file = None
_csv_writer = None


def _open_csv():
    global _csv_file, _csv_writer
    new = not os.path.exists(LOG_PATH)
    _csv_file = open(LOG_PATH, "a", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=LOG_FIELDS)
    if new:
        _csv_writer.writeheader()
        _csv_file.flush()


def csv_log(row: dict):
    with _csv_lock:
        _csv_writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})
        _csv_file.flush()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── ESPN ────────────────────────────────────────────────────────────────────

_session = requests.Session()


def espn_scoreboard(league_id: str) -> list:
    try:
        r = _session.get(f"{ESPN_BASE}/{league_id}/scoreboard", timeout=10)
        return r.json().get("events", [])
    except Exception:
        return []


def parse_espn_event(e: dict) -> Optional[dict]:
    """Return normalized in-progress game dict, or None if not live."""
    state = e.get("status", {}).get("type", {}).get("state", "")
    if state != "in":
        return None
    comp = (e.get("competitions") or [{}])[0]
    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    return {
        "espn_id":     e.get("id"),
        "label":       e.get("shortName", "?"),
        "home_id":     home.get("team", {}).get("id"),
        "home_abbr":   home.get("team", {}).get("abbreviation", ""),
        "home_name":   home.get("team", {}).get("displayName", ""),
        "home_score":  int(home.get("score", 0)),
        "away_id":     away.get("team", {}).get("id"),
        "away_abbr":   away.get("team", {}).get("abbreviation", ""),
        "away_name":   away.get("team", {}).get("displayName", ""),
        "away_score":  int(away.get("score", 0)),
        "status":      e.get("status", {}).get("type", {}).get("description", ""),
        "clock":       e.get("status", {}).get("displayClock", ""),
    }


# ── Kalshi ──────────────────────────────────────────────────────────────────

def kalshi_get(path: str, params=None, signed=False, priv_key=None, key_id=None):
    headers = {}
    if signed:
        headers = kalshi_auth.make_auth_headers(priv_key, key_id, "GET", path)
    try:
        r = _session.get(KALSHI_BASE + path, headers=headers, params=params, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def find_kalshi_ticker_for_team(scoring_team_abbr: str, opp_team_abbr: str,
                                game_start_iso: str = "") -> Optional[str]:
    """Search Kalshi for an open per-game market matching this game.

    Strategy: query each plausible series for open markets containing
    BOTH team abbreviations in the ticker.  Return the one whose ticker
    ENDS with the scoring team's abbreviation (= the YES market for that
    team's win)."""
    scoring = scoring_team_abbr.upper()
    opp     = opp_team_abbr.upper()
    if not scoring or not opp:
        return None
    for series in KALSHI_SOCCER_SERIES:
        data = kalshi_get("/trade-api/v2/markets",
                          params={"series_ticker": series, "status": "open", "limit": 200})
        if not data:
            continue
        mks = data.get("markets", [])
        # Look for tickers that contain both abbreviations
        for m in mks:
            t = m.get("ticker", "")
            tu = t.upper()
            if scoring in tu and opp in tu and tu.endswith("-" + scoring):
                return t
    return None


def kalshi_yes_ask(ticker: str) -> Optional[float]:
    """Current best yes_ask in dollars."""
    data = kalshi_get(f"/trade-api/v2/markets/{ticker}")
    if not data:
        return None
    mk = data.get("market", {})
    try:
        return float(mk.get("yes_ask_dollars") or 0)
    except Exception:
        return None


def kalshi_yes_bid(ticker: str) -> Optional[float]:
    data = kalshi_get(f"/trade-api/v2/markets/{ticker}")
    if not data:
        return None
    mk = data.get("market", {})
    try:
        return float(mk.get("yes_bid_dollars") or 0)
    except Exception:
        return None


# ── Daily-loss tracking ─────────────────────────────────────────────────────

_today_pnl = 0.0
_today_date = ""
_pnl_lock = threading.Lock()


def add_pnl(amount: float):
    global _today_pnl, _today_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _pnl_lock:
        if today != _today_date:
            _today_pnl = 0.0
            _today_date = today
        _today_pnl += amount


def loss_limit_ok() -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _pnl_lock:
        if today != _today_date:
            return True
        return _today_pnl > -DAILY_LOSS_LIMIT


# ── Position management ─────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    side: str           # "yes"
    entry_cents: float  # what we paid per contract
    qty: float
    bought_at: float    # unix ts
    league: str
    game_label: str
    scoring_team: str


_positions: List[Position] = []
_pos_lock = threading.Lock()


def manage_positions_loop(priv_key, key_id):
    """Background thread: monitor every open position, fire take-profit sell
    if Kalshi best-bid is high enough."""
    while True:
        try:
            with _pos_lock:
                positions = list(_positions)
            now = time.time()
            for p in positions:
                # If we've held longer than HOLD_FOR_SECONDS, give up active management
                # (the position will settle automatically at game end)
                if now - p.bought_at > HOLD_FOR_SECONDS:
                    continue
                # Check current best bid; if it's above our target, sell
                bid = kalshi_yes_bid(p.ticker)
                if bid is None:
                    continue
                bid_cents = bid * 100
                target_cents = p.entry_cents + TAKE_PROFIT_CENTS
                if bid_cents < target_cents:
                    continue
                # Fire sell — equivalent to buying NO at (1 - bid_cents/100)
                sell_at = bid_cents
                profit_per_contract = sell_at - p.entry_cents
                pnl_est = profit_per_contract * p.qty / 100.0
                log.info(f"  SELL  {p.ticker}  entry={p.entry_cents:.1f}¢ "
                         f"→ sell@{sell_at:.1f}¢ (best bid)  qty={p.qty}  "
                         f"est_profit=${pnl_est:.2f}  [{'PAPER' if PAPER_MODE else 'LIVE'}]")

                if not PAPER_MODE:
                    # Submit IOC SELL — we're closing a YES position by buying NO
                    # at (100 - sell_price).  kalshi_trade.place_order doesn't
                    # have a "sell" mode directly, so we fire BUY NO equivalent.
                    try:
                        market = {
                            "yes_bid_dollars": str(bid),
                            "yes_ask_dollars": str(bid + 0.01),
                            "ticker":         p.ticker,
                        }
                        target_no_price_cents = 100 - int(round(sell_at))
                        # buffer is 0 — we want exact match
                        base_yes_bid_c = int(round(bid * 100))
                        extra = base_yes_bid_c - target_no_price_cents - kalshi_trade.FILL_BUFFER_CENTS
                        kalshi_trade.place_order(
                            priv_key, key_id, p.ticker, "no", market,
                            stake_dollars=p.qty * (100 - sell_at) / 100.0,
                            extra_buffer_cents=extra, ioc=True,
                        )
                    except Exception as e:
                        log.warning(f"  sell failed: {e}")

                # Record + remove
                add_pnl(pnl_est)
                csv_log({
                    "ts_iso": _now_iso(),
                    "mode":   "paper" if PAPER_MODE else "live",
                    "event_type": "SELL",
                    "league": p.league, "game": p.game_label,
                    "scoring_team": p.scoring_team,
                    "kalshi_ticker": p.ticker, "side": "yes",
                    "entry_cents": round(p.entry_cents, 2),
                    "exit_cents":  round(sell_at, 2),
                    "qty": p.qty, "realized_pnl": round(pnl_est, 2),
                    "note": "take_profit",
                })
                with _pos_lock:
                    if p in _positions:
                        _positions.remove(p)
        except Exception as e:
            log.warning(f"position-mgr err: {e}")
        time.sleep(2)


# ── Per-game observer ──────────────────────────────────────────────────────

_active_games: Dict[str, threading.Thread] = {}
_active_lock  = threading.Lock()


def observe_game(league_id: str, league_label: str, game: dict,
                 priv_key, key_id):
    label = f"{game['away_abbr']}@{game['home_abbr']}"
    log.info(f"[{league_label} {label}] starting observation "
             f"({game['status']}, score {game['away_score']}-{game['home_score']})")
    csv_log({
        "ts_iso": _now_iso(),
        "mode":   "paper" if PAPER_MODE else "live",
        "event_type": "GAME_START_OBS",
        "league": league_label, "game": label,
        "score_after": f"{game['away_score']}-{game['home_score']}",
        "note": game.get("status", ""),
    })

    last_home = game["home_score"]
    last_away = game["away_score"]
    espn_id   = game["espn_id"]

    while True:
        try:
            events = espn_scoreboard(league_id)
            cur = None
            for e in events:
                p = parse_espn_event(e)
                if p and p["espn_id"] == espn_id:
                    cur = p
                    break
            if cur is None:
                # Game ended
                log.info(f"[{league_label} {label}] game ended")
                csv_log({
                    "ts_iso": _now_iso(),
                    "mode":   "paper" if PAPER_MODE else "live",
                    "event_type": "GAME_END",
                    "league": league_label, "game": label,
                })
                break
        except Exception as e:
            log.warning(f"[{label}] espn err: {e}")
            time.sleep(5); continue

        # Score change → goal!
        if cur["home_score"] != last_home or cur["away_score"] != last_away:
            home_diff = cur["home_score"] - last_home
            away_diff = cur["away_score"] - last_away
            if home_diff > 0:
                scoring_abbr = cur["home_abbr"]; scoring_name = cur["home_name"]
                opp_abbr     = cur["away_abbr"]
            else:
                scoring_abbr = cur["away_abbr"]; scoring_name = cur["away_name"]
                opp_abbr     = cur["home_abbr"]

            new_score = f"{cur['away_abbr']} {cur['away_score']} - {cur['home_abbr']} {cur['home_score']}"
            log.info(f"[{league_label} {label}] GOAL  {scoring_name} scored  →  {new_score}")
            handle_goal(league_label, label, cur, scoring_abbr, scoring_name, opp_abbr,
                        new_score, priv_key, key_id)

            last_home = cur["home_score"]
            last_away = cur["away_score"]

        time.sleep(SCORE_POLL_SEC)

    with _active_lock:
        _active_games.pop(espn_id, None)


def handle_goal(league_label, game_label, game, scoring_abbr, scoring_name,
                opp_abbr, score_after, priv_key, key_id):
    """Goal just happened — find the Kalshi market and fire IOC BUY YES."""
    if not loss_limit_ok():
        log.warning("daily loss limit reached — not firing")
        csv_log({"ts_iso": _now_iso(), "mode": "paper" if PAPER_MODE else "live",
                 "event_type": "GOAL_BLOCKED_LOSS_LIMIT",
                 "league": league_label, "game": game_label,
                 "scoring_team": scoring_abbr, "score_after": score_after})
        return

    ticker = find_kalshi_ticker_for_team(scoring_abbr, opp_abbr)
    if not ticker:
        log.warning(f"  no Kalshi market found for {scoring_abbr} vs {opp_abbr}")
        csv_log({"ts_iso": _now_iso(), "mode": "paper" if PAPER_MODE else "live",
                 "event_type": "NO_MARKET",
                 "league": league_label, "game": game_label,
                 "scoring_team": scoring_abbr, "score_after": score_after})
        return

    ask = kalshi_yes_ask(ticker)
    if ask is None:
        log.warning(f"  couldn't read ask for {ticker}")
        return

    if ask < MIN_PRICE_TO_BUY or ask > MAX_PRICE_TO_BUY:
        log.info(f"  ask {ask:.2f} outside [{MIN_PRICE_TO_BUY},{MAX_PRICE_TO_BUY}] — skipping")
        return

    qty = max(1.0, STAKE_PER_GOAL / ask)
    actual_stake = qty * ask
    log.info(f"  → fire BUY YES  {ticker}  ask={ask:.4f}  qty={qty:.1f}  "
             f"stake=${actual_stake:.2f}  [{'PAPER' if PAPER_MODE else 'LIVE'}]")

    if not PAPER_MODE:
        try:
            market = {
                "yes_ask_dollars": str(ask),
                "yes_bid_dollars": str(ask - 0.01),
                "ticker":          ticker,
            }
            # Buy at exactly ask (no extra buffer beyond default)
            kalshi_trade.place_order(
                priv_key, key_id, ticker, "yes", market,
                stake_dollars=actual_stake, extra_buffer_cents=0, ioc=True,
            )
        except Exception as e:
            log.warning(f"  buy failed: {e}")
            csv_log({"ts_iso": _now_iso(), "mode": "live",
                     "event_type": "BUY_FAILED",
                     "league": league_label, "game": game_label,
                     "scoring_team": scoring_abbr, "kalshi_ticker": ticker,
                     "note": str(e)})
            return

    # Record + add to position management
    with _pos_lock:
        _positions.append(Position(
            ticker=ticker, side="yes", entry_cents=ask*100, qty=qty,
            bought_at=time.time(), league=league_label,
            game_label=game_label, scoring_team=scoring_abbr,
        ))
    csv_log({
        "ts_iso": _now_iso(),
        "mode":   "paper" if PAPER_MODE else "live",
        "event_type": "BUY",
        "league": league_label, "game": game_label,
        "scoring_team": scoring_abbr, "score_after": score_after,
        "kalshi_ticker": ticker, "side": "yes",
        "limit_cents": round(ask * 100, 1),
        "fill_cents_est": round(ask * 100, 1),
        "qty": round(qty, 3), "stake_dollars": round(actual_stake, 2),
        "entry_cents": round(ask * 100, 2),
    })


# ── Discovery loop ─────────────────────────────────────────────────────────

LEAGUE_LABELS = {
    "eng.1": "EPL", "uefa.champions": "UCL", "esp.1": "La Liga",
    "ger.1": "Bundesliga", "ita.1": "Serie A", "usa.1": "MLS",
    "fra.1": "Ligue 1", "fifa.worldcup": "World Cup",
    "uefa.europa": "Europa", "por.1": "Liga Portugal",
    "col.1": "Liga DIMAYOR", "mex.1": "Liga MX", "fifa.wwc": "FIFA WWC",
    "usa.nwsl": "NWSL",
}


def discovery_loop(priv_key, key_id):
    log.info("discovery loop start")
    while True:
        try:
            for league_id in LEAGUES:
                label = LEAGUE_LABELS.get(league_id, league_id)
                events = espn_scoreboard(league_id)
                for e in events:
                    g = parse_espn_event(e)
                    if g is None:
                        continue
                    gid = g["espn_id"]
                    with _active_lock:
                        if gid in _active_games:
                            continue
                        log.info(f"[{label}] live game: {g['away_abbr']}@{g['home_abbr']} "
                                 f"({g['status']}, {g['away_score']}-{g['home_score']})")
                        t = threading.Thread(
                            target=observe_game,
                            args=(league_id, label, g, priv_key, key_id),
                            daemon=True, name=f"obs-{label}-{gid}",
                        )
                        _active_games[gid] = t
                        t.start()
        except Exception as e:
            log.warning(f"discovery err: {e}")
        time.sleep(DISCOVERY_INTERVAL)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("soccer sniper starting")
    log.info(f"  PAPER_MODE={PAPER_MODE}  STAKE=${STAKE_PER_GOAL}  "
             f"TAKE_PROFIT={TAKE_PROFIT_CENTS}¢  LOSS_LIMIT=${DAILY_LOSS_LIMIT}")
    log.info(f"  leagues: {LEAGUES}")
    log.info(f"  log path: {LOG_PATH}")

    env = dotenv_values(ENV_PATH)
    # Use LEO key if present, else fall back to KALSHI_*
    priv_pem = env.get("LEO_PRIVATE_KEY") or env.get("KALSHI_PRIVATE_KEY")
    key_id   = env.get("LEO_KEY_ID")      or env.get("KALSHI_API_KEY_ID")
    priv_key = kalshi_auth.load_private_key(priv_pem)

    _open_csv()
    if kalshi_trade.warmup_session():
        log.info("kalshi session pre-warmed")

    # Background position manager
    t = threading.Thread(
        target=manage_positions_loop, args=(priv_key, key_id),
        daemon=True, name="position-mgr",
    )
    t.start()

    discovery_loop(priv_key, key_id)


if __name__ == "__main__":
    main()
