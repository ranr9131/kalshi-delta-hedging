"""
Auto-starting sports observer.

Purpose: when any live NBA / NHL / MLB game starts, automatically observe
the Kalshi market for that game and log how fast Kalshi's market makers
reprice after each scoring event.  This is the diagnostic that tells us
whether goal-sniping has real edge on Kalshi.

What it does:
  - Background thread polls ESPN every 30s for live games (status=='in')
  - When a new live game is detected, spawns a per-game observer thread:
      * Polls ESPN every 1s for the current score
      * Polls Kalshi REST every 1s for the per-team market bid/ask
      * On every score change, logs the event + tracks Kalshi quote
        for the next 30s (so we can measure reprice latency)
  - Outputs:
      * sports_observer.log   — human-readable timeline
      * sports_events.csv     — machine-readable for analysis

No orders placed.  Pure observation.  Run as systemd service so it's
always ready when a game starts.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import requests


ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_TXT = os.path.join(ROOT, "sports_observer.log")
LOG_CSV = os.path.join(ROOT, "sports_events.csv")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
KALSHI_BASE = "https://api.elections.kalshi.com"

# ESPN scoreboards we care about
SCOREBOARDS = {
    "NBA": "basketball/nba",
    "NHL": "hockey/nhl",
    "MLB": "baseball/mlb",
    "NFL": "football/nfl",
}

# Kalshi series per sport
KALSHI_SERIES = {
    "NBA": "KXNBAGAME",
    "NHL": "KXNHLGAME",
    "MLB": "KXMLBGAME",
    "NFL": "KXNFLGAME",
}

DISCOVERY_INTERVAL = 30   # seconds
GAME_POLL_INTERVAL = 1.0  # seconds — score + Kalshi quote sampling rate
POST_SCORE_TRACK   = 30   # seconds to closely track quotes after a score


# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sports_obs")


# ── Shared CSV writer ──────────────────────────────────────────────────────
_csv_lock = threading.Lock()


def _csv_write(row: dict):
    new = not os.path.exists(LOG_CSV)
    with _csv_lock:
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts_iso", "sport", "game_label", "event_type",
                "home_team", "away_team", "home_score", "away_score",
                "scoring_team", "secs_since_score",
                "kalshi_ticker", "yes_ask", "yes_bid", "spread_c",
                "note",
            ])
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in w.fieldnames})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ── ESPN ────────────────────────────────────────────────────────────────────
_sess = requests.Session()


def _espn_scoreboard(sport_path: str) -> list:
    try:
        r = _sess.get(f"{ESPN_BASE}/{sport_path}/scoreboard", timeout=10)
        return r.json().get("events", [])
    except Exception:
        return []


def _parse_espn_game(e: dict) -> dict | None:
    """Return a dict with the info we care about, or None if not in-progress."""
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
        "espn_id":    e.get("id"),
        "name":       e.get("shortName", e.get("name", "?")),
        "home_team":  home.get("team", {}).get("abbreviation", ""),
        "away_team":  away.get("team", {}).get("abbreviation", ""),
        "home_score": int(home.get("score", 0)),
        "away_score": int(away.get("score", 0)),
        "status":     e.get("status", {}).get("type", {}).get("description", ""),
        "date":       e.get("date", ""),
    }


# ── Kalshi market discovery ────────────────────────────────────────────────

def _find_kalshi_tickers(sport: str, home: str, away: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the two per-team Kalshi tickers for this game.  Returns
    (home_ticker, away_ticker) or (None, None) if not found."""
    series = KALSHI_SERIES.get(sport)
    if not series:
        return None, None
    try:
        r = _sess.get(f"{KALSHI_BASE}/trade-api/v2/markets",
                      params={"series_ticker": series, "status": "open", "limit": 200},
                      timeout=10)
        if not r.ok:
            return None, None
        mks = r.json().get("markets", [])
    except Exception:
        return None, None
    # Look for tickers containing BOTH team codes, ending with -HOME / -AWAY
    home_t = None; away_t = None
    for m in mks:
        t = m.get("ticker", "")
        if home in t and away in t:
            if t.endswith(f"-{home}"):
                home_t = t
            elif t.endswith(f"-{away}"):
                away_t = t
    return home_t, away_t


def _kalshi_market(ticker: str) -> dict | None:
    try:
        r = _sess.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}", timeout=8)
        if not r.ok:
            return None
        return r.json().get("market", {})
    except Exception:
        return None


def _f(v):
    try: return float(v)
    except: return None


# ── Per-game observer ───────────────────────────────────────────────────────

_active_games: Dict[str, threading.Thread] = {}
_active_lock = threading.Lock()


def _observe_game(sport: str, game: dict):
    label = f"{game['away_team']}@{game['home_team']}"
    log.info(f"[{sport} {label}] starting observation ({game['status']})")

    # Find Kalshi tickers
    home_ticker, away_ticker = _find_kalshi_tickers(sport, game["home_team"], game["away_team"])
    if not home_ticker and not away_ticker:
        log.warning(f"[{sport} {label}] no matching Kalshi tickers found, "
                    f"will retry every poll cycle")

    _csv_write({
        "ts_iso": _now_iso(), "sport": sport, "game_label": label,
        "event_type": "GAME_OBSERVATION_START",
        "home_team": game["home_team"], "away_team": game["away_team"],
        "home_score": game["home_score"], "away_score": game["away_score"],
        "kalshi_ticker": (home_ticker or "") + "|" + (away_ticker or ""),
        "note": game.get("status", ""),
    })

    last_home = game["home_score"]; last_away = game["away_score"]
    last_score_ts: float = 0.0
    last_scoring_team: str = ""

    while True:
        # Re-fetch game from ESPN
        try:
            events = _espn_scoreboard(SCOREBOARDS[sport])
            cur = None
            for e in events:
                p = _parse_espn_game(e)
                if p and p["espn_id"] == game["espn_id"]:
                    cur = p
                    break
            if cur is None:
                # Game ended
                log.info(f"[{sport} {label}] game ended, observer stopping")
                _csv_write({
                    "ts_iso": _now_iso(), "sport": sport, "game_label": label,
                    "event_type": "GAME_END",
                })
                break
        except Exception as e:
            log.warning(f"[{sport} {label}] espn err: {e}")
            time.sleep(5); continue

        # Score change detection
        if cur["home_score"] != last_home or cur["away_score"] != last_away:
            home_diff = cur["home_score"] - last_home
            away_diff = cur["away_score"] - last_away
            scoring_team = cur["home_team"] if home_diff > 0 else cur["away_team"]
            last_scoring_team = scoring_team
            last_score_ts = time.time()
            log.info(f"[{sport} {label}] SCORE  {scoring_team} scored  "
                     f"now {cur['away_team']} {cur['away_score']} - "
                     f"{cur['home_team']} {cur['home_score']}")
            # Snapshot Kalshi quotes at scoring moment
            for t in (home_ticker, away_ticker):
                if not t: continue
                mk = _kalshi_market(t)
                if mk:
                    _csv_write({
                        "ts_iso": _now_iso(), "sport": sport, "game_label": label,
                        "event_type": "SCORE",
                        "home_team": cur["home_team"], "away_team": cur["away_team"],
                        "home_score": cur["home_score"], "away_score": cur["away_score"],
                        "scoring_team": scoring_team,
                        "secs_since_score": 0,
                        "kalshi_ticker": t,
                        "yes_ask": _f(mk.get("yes_ask_dollars")),
                        "yes_bid": _f(mk.get("yes_bid_dollars")),
                        "spread_c": (_f(mk.get("yes_ask_dollars")) - _f(mk.get("yes_bid_dollars")))*100
                                     if _f(mk.get("yes_ask_dollars")) and _f(mk.get("yes_bid_dollars")) else None,
                    })
            last_home = cur["home_score"]
            last_away = cur["away_score"]

        # Tight post-score tracking — log Kalshi quotes every 1s for 30s after
        if last_score_ts and (time.time() - last_score_ts) <= POST_SCORE_TRACK:
            secs = time.time() - last_score_ts
            for t in (home_ticker, away_ticker):
                if not t: continue
                mk = _kalshi_market(t)
                if not mk: continue
                ya = _f(mk.get("yes_ask_dollars"))
                yb = _f(mk.get("yes_bid_dollars"))
                _csv_write({
                    "ts_iso": _now_iso(), "sport": sport, "game_label": label,
                    "event_type": "QUOTE",
                    "home_team": cur["home_team"], "away_team": cur["away_team"],
                    "home_score": cur["home_score"], "away_score": cur["away_score"],
                    "scoring_team": last_scoring_team,
                    "secs_since_score": round(secs, 1),
                    "kalshi_ticker": t,
                    "yes_ask": ya, "yes_bid": yb,
                    "spread_c": (ya - yb)*100 if ya and yb else None,
                })
        time.sleep(GAME_POLL_INTERVAL)

    with _active_lock:
        _active_games.pop(game["espn_id"], None)


# ── Discovery loop ─────────────────────────────────────────────────────────

def _discovery_loop():
    log.info("discovery loop starting")
    while True:
        try:
            for sport, path in SCOREBOARDS.items():
                events = _espn_scoreboard(path)
                for e in events:
                    g = _parse_espn_game(e)
                    if g is None:
                        continue
                    gid = g["espn_id"]
                    with _active_lock:
                        if gid in _active_games:
                            continue
                        log.info(f"[{sport}] live game detected: "
                                 f"{g['away_team']}@{g['home_team']}  "
                                 f"({g['status']})")
                        t = threading.Thread(
                            target=_observe_game, args=(sport, g),
                            daemon=True, name=f"obs-{sport}-{gid}",
                        )
                        _active_games[gid] = t
                        t.start()
        except Exception as e:
            log.warning(f"discovery err: {e}")
        time.sleep(DISCOVERY_INTERVAL)


def main():
    log.info("sports observer starting  log=%s  csv=%s", LOG_TXT, LOG_CSV)
    _discovery_loop()


if __name__ == "__main__":
    main()
