"""
Cross-market sports order-book logger (Kalshi <-> Polymarket).

Purpose: build the dataset neither platform's history API gives you — aligned,
two-sided (bid/ask) quotes for the SAME game on BOTH venues, sampled every
~1-2s, so you can later test whether cross-venue sports arbs were actually
*executable* (not just whether mids diverged).

Why a custom logger:
  - Kalshi candlesticks give bid/ask but only 1-min granularity.
  - Polymarket /prices-history gives a single price/min, NO bid/ask.
  Real arb lives intra-second and needs both sides on both venues.

Generalised N-outcome model (one code path for all sports):
  - 2-way games (NBA/WNBA/NHL/MLB/NFL): outcomes = [home, away]
  - 3-way games (soccer / World Cup):    outcomes = [home, away, draw]
  Each outcome is matched to a Kalshi market AND a Polymarket token, then
  sampled every poll. Arb = sum over outcomes of cheapest ask < $1.00.

Matching keys (robust, name-based, venue-symmetric):
  - Kalshi:     market `yes_sub_title`  ("Congo DR" / "Tie" / "Uzbekistan")
  - Polymarket: market `groupItemTitle` ("Denmark"  / "Draw (...)" )

Output: xmarket_quotes.csv  (one wide row per game per poll), xmarket.log
Run:    python3 xmarket_logger.py        (no auth — all endpoints public)
Pure observation. NO orders placed.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

# Optional: live Kalshi WebSocket order book (full depth). If auth/deps are
# missing we fall back to REST quotes. Using the WS book removes our own
# measurement lag (REST /markets can be cached seconds) and gives ask DEPTH so
# arbs can be sized, not just detected.
try:
    from dotenv import dotenv_values
    import kalshi_auth
    import xmarket_ob as ob          # fixed single-subscription WS book
    _OB_AVAILABLE = True
except Exception:
    _OB_AVAILABLE = False

_ob_ready = False   # set True once the WS book is started and authed
WS_FRESH_S = 20     # use WS book if its last update is within this many seconds

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_TXT = os.path.join(ROOT, "xmarket.log")
LOG_CSV = os.path.join(ROOT, "xmarket_quotes.csv")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
KALSHI_BASE = "https://api.elections.kalshi.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Polymarket "live games" tag (all sports single-game markets).
POLY_GAMES_TAG = 100639

# League config. Each league lists the ESPN scoreboard(s) to discover live
# games from, the candidate Kalshi series to match against, and whether the
# sport has a draw outcome (soccer).
LEAGUES: Dict[str, dict] = {
    "NBA":  dict(espn=["basketball/nba"],  kalshi=["KXNBAGAME"],  draw=False),
    "WNBA": dict(espn=["basketball/wnba"], kalshi=["KXWNBAGAME"], draw=False),
    "NHL":  dict(espn=["hockey/nhl"],      kalshi=["KXNHLGAME"],  draw=False),
    "MLB":  dict(espn=["baseball/mlb"],    kalshi=["KXMLBGAME"],  draw=False),
    "NFL":  dict(espn=["football/nfl"],    kalshi=["KXNFLGAME"],  draw=False),
    "SOCCER": dict(
        espn=["soccer/fifa.world", "soccer/fifa.friendly",
              "soccer/uefa.nations", "soccer/fifa.cwc"],
        kalshi=["KXWCGAME", "KXFIFAWGAME"],
        draw=True,
    ),
}

DISCOVERY_INTERVAL = 30   # s — how often we scan ESPN for new live games
POLL_INTERVAL = 1.5       # s — quote sampling cadence (REST on both venues)
SCORE_EVERY = 2           # re-poll ESPN score every Nth sample
MAX_OUTCOMES = 3          # CSV slots (home, away, draw)

# Log to stdout only; under systemd, StandardOutput=append writes it to
# xmarket.log. (A Python FileHandler to that same path fails — systemd opens
# the append file as root before dropping to the service User.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xmarket")

_sess = requests.Session()
_sess.headers.update({"User-Agent": "xmarket-logger/1.0"})


# ── CSV (wide: up to 3 outcome slots) ────────────────────────────────────────

_csv_lock = threading.Lock()


def _csv_fields() -> List[str]:
    base = ["ts_iso", "ts_unix", "league", "game_label", "espn_id",
            "status", "home_score", "away_score", "n_outcomes"]
    for i in range(1, MAX_OUTCOMES + 1):
        base += [f"o{i}_label", f"o{i}_k_ticker",
                 f"o{i}_k_bid", f"o{i}_k_ask", f"o{i}_k_ask_sz",
                 f"o{i}_p_bid", f"o{i}_p_ask", f"o{i}_p_ask_sz", f"o{i}_min_ask"]
    base += ["lock_cost", "arb_edge_c", "poly_slug"]
    return base


CSV_FIELDS = _csv_fields()


def _csv_write(row: dict):
    new = not os.path.exists(LOG_CSV)
    with _csv_lock:
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _norm(s: str) -> set:
    """Word set for name matching across venues. Splits concatenated CamelCase
    first ("PortlandFire" -> "Portland Fire") so exact word matching works
    without resorting to substring matches (which false-fire on "los" etc.)."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s or "")
    s = s.lower().replace(".", " ").replace("-", " ")
    return {w for w in s.split() if len(w) > 2}


def _min_ask(*asks) -> Optional[float]:
    vals = [a for a in asks if a is not None and a > 0]
    return min(vals) if vals else None


def _classify_role(name: str, home_set: set, away_set: set) -> Optional[str]:
    """Map an outcome label to 'home' / 'away' / 'draw' / None.

    Uses argmax word-overlap on the *distinctive* word, so shared tokens like
    "Los Angeles" (Dodgers vs Angels) don't cause a false first-match.
    """
    s = (name or "").strip().lower()
    if s.startswith("draw") or s in ("tie", "draw"):
        return "draw"
    nset = _norm(name)   # _norm splits CamelCase, so "PortlandFire" -> {portland, fire}
    hov = len(home_set & nset)
    aov = len(away_set & nset)
    if hov == 0 and aov == 0:
        return None
    if hov > aov:
        return "home"
    if aov > hov:
        return "away"
    return None  # ambiguous (equal non-zero overlap) — skip


# ── ESPN ─────────────────────────────────────────────────────────────────────

def _espn_scoreboard(path: str) -> list:
    try:
        r = _sess.get(f"{ESPN_BASE}/{path}/scoreboard", timeout=10)
        return r.json().get("events", [])
    except Exception:
        return []


def _parse_espn_game(e: dict) -> Optional[dict]:
    state = e.get("status", {}).get("type", {}).get("state", "")
    if state != "in":
        return None
    comp = (e.get("competitions") or [{}])[0]
    cs = comp.get("competitors", [])
    home = next((c for c in cs if c.get("homeAway") == "home"), None)
    away = next((c for c in cs if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    return {
        "espn_id":    e.get("id"),
        "home_abbr":  home.get("team", {}).get("abbreviation", ""),
        "away_abbr":  away.get("team", {}).get("abbreviation", ""),
        "home_name":  home.get("team", {}).get("displayName", "") or home.get("team", {}).get("name", ""),
        "away_name":  away.get("team", {}).get("displayName", "") or away.get("team", {}).get("name", ""),
        "home_score": int(home.get("score", 0) or 0),
        "away_score": int(away.get("score", 0) or 0),
        "status":     e.get("status", {}).get("type", {}).get("description", ""),
        "date":       e.get("date", ""),   # ISO UTC
    }


# ── Outcome model ────────────────────────────────────────────────────────────

def _build_outcomes(game: dict, has_draw: bool) -> List[dict]:
    outs = [
        {"role": "home", "label": game["home_name"], "k": None, "p": None},
        {"role": "away", "label": game["away_name"], "k": None, "p": None},
    ]
    if has_draw:
        outs.append({"role": "draw", "label": "Draw", "k": None, "p": None})
    return outs


# ── Kalshi matching (public REST) ────────────────────────────────────────────

def _kalshi_series_markets(series: str) -> list:
    try:
        r = _sess.get(f"{KALSHI_BASE}/trade-api/v2/markets",
                      params={"series_ticker": series, "status": "open", "limit": 400},
                      timeout=12)
        return r.json().get("markets", []) if r.ok else []
    except Exception:
        return []


def _match_kalshi(series_list: List[str], game: dict, outcomes: List[dict]) -> None:
    """Fill outcome['k'] with a Kalshi ticker by matching yes_sub_title."""
    markets = []
    for s in series_list:
        markets += _kalshi_series_markets(s)
    if not markets:
        return
    # group markets into events (ticker without the trailing outcome segment)
    events: Dict[str, list] = defaultdict(list)
    for m in markets:
        base = m.get("ticker", "").rsplit("-", 1)[0]
        events[base].append(m)

    home_set, away_set = _norm(game["home_name"]), _norm(game["away_name"])
    ha = (game.get("home_abbr") or "").upper()
    aa = (game.get("away_abbr") or "").upper()
    # Pick our event. Prefer matching both team CODES in the event base ticker
    # (robust — Kalshi truncates yes_sub_title to ~13 chars, which collapses
    # shared-city teams like LA Dodgers / LA Angels). Fall back to sub_titles.
    chosen = None
    for base, mks in events.items():
        b = base.upper()
        if ha and aa and ha in b and aa in b:
            chosen = mks
            break
    if chosen is None:
        for base, mks in events.items():
            tset = _norm(" ".join(m.get("yes_sub_title", "") for m in mks))
            if (home_set & tset) and (away_set & tset):
                chosen = mks
                break
    if not chosen:
        return
    for m in chosen:
        suffix = m.get("ticker", "").rsplit("-", 1)[-1].upper()
        if suffix == "TIE":
            role = "draw"
        elif ha and suffix == ha:
            role = "home"
        elif aa and suffix == aa:
            role = "away"
        else:
            role = _classify_role(m.get("yes_sub_title") or "", home_set, away_set)
        if not role:
            continue
        for o in outcomes:
            if o["role"] == role:
                o["k"] = m.get("ticker")


def _kalshi_quote_rest(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """REST fallback: (yes_bid, yes_ask). Can lag the true book by seconds."""
    try:
        r = _sess.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}", timeout=8)
        if not r.ok:
            return None, None
        mk = r.json().get("market", {})
        return _f(mk.get("yes_bid_dollars")), _f(mk.get("yes_ask_dollars"))
    except Exception:
        return None, None


def _kalshi_ask_depth(ticker: str) -> Optional[float]:
    """Best-effort YES-ask depth from the public REST orderbook. The best YES
    ask is the inverse of the best NO bid, so its size = size at the top
    NO-bid level. Response shape (confirmed live): {orderbook_fp: {no_dollars:
    [[price_str, size_str], ...] ascending, yes_dollars: [...]}}."""
    try:
        ob_j = _sess.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}/orderbook",
                         timeout=8).json().get("orderbook_fp", {}) or {}
        no = ob_j.get("no_dollars") or ob_j.get("no") or []
        if not no:
            return None
        best = max(no, key=lambda l: _f(l[0]) or 0)   # highest NO bid
        return _f(best[1])
    except Exception:
        return None


def _kalshi_quote(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(yes_bid, yes_ask, yes_ask_size).

    Prefer the live WS book (xmarket_ob — single-subscription model that
    actually streams deltas, unlike the old per-ticker kalshi_orderbook).
    Fall back to REST when the book is missing or stale (e.g. during the brief
    reconnect after a new game is added), so we never log a frozen value."""
    if _ob_ready:
        b = ob.get_book(ticker)
        if b is not None and b.snapshot_seen and b.age() < WS_FRESH_S:
            ya, yb = b.yes_ask(), b.yes_bid()
            if ya is not None or yb is not None:
                return yb, ya, b.yes_ask_size()
    yb, ya = _kalshi_quote_rest(ticker)
    sz = _kalshi_ask_depth(ticker) if ya is not None else None
    return yb, ya, sz


# ── Polymarket matching (Gamma -> CLOB) ──────────────────────────────────────

def _gamma_games() -> list:
    try:
        r = _sess.get(f"{GAMMA_BASE}/markets",
                      params={"closed": "false", "tag_id": POLY_GAMES_TAG,
                              "limit": 500, "order": "volume24hr", "ascending": "false"},
                      timeout=15)
        ms = r.json()
        return ms if isinstance(ms, list) else ms.get("data", [])
    except Exception:
        return []


def _match_poly(game: dict, outcomes: List[dict]) -> Optional[str]:
    """Fill outcome['p'] with a Polymarket CLOB token id by matching
    groupItemTitle. Returns the matched game's slug-prefix (or None)."""
    markets = _gamma_games()
    if not markets:
        return None
    game_date = (game.get("date") or "")[:10]
    home_set, away_set = _norm(game["home_name"]), _norm(game["away_name"])

    # derivative markets (spreads/totals/btts) also carry team-name outcomes —
    # exclude them so only the moneyline (per-team win / draw) is matched.
    NON_ML = ("-spread", "-total", "-btts", "-over", "-under", "-handicap")

    # Group markets by their game slug = sport-away-home-YYYY-MM-DD (first 6
    # segments). This SCOPES matching to one game, so a shared city ("Los
    # Angeles") in another same-date game's market can't leak in.
    groups: Dict[str, list] = defaultdict(list)
    for m in markets:
        slug = m.get("slug", "")
        if any(tag in slug for tag in NON_ML):
            continue
        if game_date and game_date not in slug and game_date not in (m.get("gameStartTime") or ""):
            continue
        key = "-".join(slug.split("-")[:6])
        groups[key].append(m)

    def _parse(m):
        try:
            return json.loads(m.get("outcomes") or "[]"), json.loads(m.get("clobTokenIds") or "[]")
        except Exception:
            return [], []

    def _group_roles(gm) -> set:
        roles = set()
        for m in gm:
            ocs, toks = _parse(m)
            if len(ocs) != 2 or len(toks) != 2:
                continue
            if [o.strip().lower() for o in ocs] == ["yes", "no"]:
                r = _classify_role(m.get("groupItemTitle") or "", home_set, away_set)
                if r:
                    roles.add(r)
            else:
                for name in ocs:
                    r = _classify_role(name, home_set, away_set)
                    if r:
                        roles.add(r)
        return roles

    # pick the one group that covers BOTH our teams
    chosen_key = None
    for key, gm in groups.items():
        roles = _group_roles(gm)
        if "home" in roles and "away" in roles:
            chosen_key = key
            break
    if not chosen_key:
        return None

    def _assign(role, tok):
        for o in outcomes:
            if o["role"] == role and o["p"] is None:
                o["p"] = tok

    for m in groups[chosen_key]:
        ocs, tokens = _parse(m)
        if len(ocs) != 2 or len(tokens) != 2:
            continue
        if [o.strip().lower() for o in ocs] == ["yes", "no"]:
            # split market: team/Draw named in groupItemTitle; token[0]="Yes".
            role = _classify_role(m.get("groupItemTitle") or "", home_set, away_set)
            if role:
                _assign(role, tokens[0])
        else:
            # 2-way single market: outcomes ARE the team names.
            for idx, name in enumerate(ocs):
                role = _classify_role(name, home_set, away_set)
                if role:
                    _assign(role, tokens[idx])
    return chosen_key


def _poly_book_quote(token_id: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(best_bid, best_ask, best_ask_size): bids ascending (best=last),
    asks descending (best=last)."""
    try:
        r = _sess.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=8)
        if not r.ok:
            return None, None, None
        b = r.json()
        bids = b.get("bids") or []
        asks = b.get("asks") or []
        bb = _f(bids[-1]["price"]) if bids else None
        ba = _f(asks[-1]["price"]) if asks else None
        bsz = _f(asks[-1]["size"]) if asks else None
        return bb, ba, bsz
    except Exception:
        return None, None, None


# ── Per-game sampler ─────────────────────────────────────────────────────────

_active: Dict[str, threading.Thread] = {}
_active_lock = threading.Lock()


def _observe(league: str, cfg: dict, game: dict):
    label = f"{game['away_abbr'] or game['away_name']}@{game['home_abbr'] or game['home_name']}"
    log.info(f"[{league} {label}] matching markets...")

    outcomes = _build_outcomes(game, cfg["draw"])
    _match_kalshi(cfg["kalshi"], game, outcomes)
    _subscribe_kalshi(outcomes)
    poly_slug = _match_poly(game, outcomes)

    have_k = any(o["k"] for o in outcomes)
    have_p = any(o["p"] for o in outcomes)
    log.info(f"[{league} {label}] kalshi={'Y' if have_k else '-'} "
             f"poly={poly_slug or '-'}  outcomes="
             + ",".join(f"{o['role']}:k{'Y' if o['k'] else '-'}p{'Y' if o['p'] else '-'}"
                        for o in outcomes))
    if not have_k and not have_p:
        log.warning(f"[{league} {label}] no markets on either venue, dropping")
        with _active_lock:
            _active.pop(game["espn_id"], None)
        return

    cur = game
    n = 0
    while True:
        n += 1
        if n % SCORE_EVERY == 1:
            ended = True
            for e in _espn_scoreboard_any(cfg["espn"]):
                p = _parse_espn_game(e)
                if p and p["espn_id"] == game["espn_id"]:
                    cur = p
                    ended = False
                    break
            if ended:
                log.info(f"[{league} {label}] game ended, sampler stopping")
                break
            # late-appearing markets: retry whichever venue is still missing
            if not have_k:
                _match_kalshi(cfg["kalshi"], cur, outcomes)
                _subscribe_kalshi(outcomes)
                have_k = any(o["k"] for o in outcomes)
            if not poly_slug:
                poly_slug = _match_poly(cur, outcomes)
                have_p = any(o["p"] for o in outcomes)

        row = {
            "ts_iso": _now_iso(), "ts_unix": round(time.time(), 3),
            "league": league, "game_label": label, "espn_id": game["espn_id"],
            "status": cur.get("status", ""),
            "home_score": cur["home_score"], "away_score": cur["away_score"],
            "n_outcomes": len(outcomes), "poly_slug": poly_slug or "",
        }
        min_asks = []
        for i, o in enumerate(outcomes, start=1):
            kb, ka, ksz = _kalshi_quote(o["k"]) if o["k"] else (None, None, None)
            pb, pa, psz = _poly_book_quote(o["p"]) if o["p"] else (None, None, None)
            ma = _min_ask(ka, pa)
            min_asks.append(ma)
            row.update({
                f"o{i}_label": o["label"], f"o{i}_k_ticker": o["k"] or "",
                f"o{i}_k_bid": kb, f"o{i}_k_ask": ka, f"o{i}_k_ask_sz": ksz,
                f"o{i}_p_bid": pb, f"o{i}_p_ask": pa, f"o{i}_p_ask_sz": psz,
                f"o{i}_min_ask": ma,
            })

        # arb only computable if EVERY outcome has at least one venue ask
        if min_asks and all(a is not None for a in min_asks):
            lock = sum(min_asks)
            edge = round((1.0 - lock) * 100, 2)
            row["lock_cost"] = round(lock, 4)
            row["arb_edge_c"] = edge
            if edge > 0:
                log.info(f"[{league} {label}] ARB? lock={lock:.4f} edge={edge:.2f}c "
                         + " ".join(f"{o['label'][:8]}={a}" for o, a in zip(outcomes, min_asks)))

        _csv_write(row)
        time.sleep(POLL_INTERVAL)

    with _active_lock:
        _active.pop(game["espn_id"], None)


def _espn_scoreboard_any(paths: List[str]) -> list:
    """Union of events across a league's ESPN scoreboards."""
    out = []
    for p in paths:
        out += _espn_scoreboard(p)
    return out


def _subscribe_kalshi(outcomes: List[dict]) -> None:
    """Subscribe a game's matched Kalshi tickers to the live WS order book."""
    if not _ob_ready:
        return
    for o in outcomes:
        if o["k"]:
            try:
                ob.add_ticker(o["k"])
            except Exception as e:
                log.warning(f"ob.add_ticker({o['k']}) failed: {e}")


def _start_kalshi_ws() -> None:
    """Start the Kalshi WS order book from .env creds. Sets _ob_ready on
    success; on any failure we silently keep using REST quotes."""
    global _ob_ready
    if not _OB_AVAILABLE:
        log.warning("kalshi_orderbook/deps unavailable — using REST quotes")
        return
    try:
        env = dotenv_values(os.path.join(ROOT, ".env"))
        pk = env.get("KALSHI_PRIVATE_KEY")
        kid = env.get("KALSHI_API_KEY_ID")
        if not pk or not kid:
            log.warning("no Kalshi creds in .env — using REST quotes")
            return
        key = kalshi_auth.load_private_key(pk)
        ob.start(key, kid, [])
        _ob_ready = True
        log.info("kalshi WS order book started (live bid/ask + depth)")
    except Exception as e:
        log.warning(f"kalshi WS init failed ({e}) — using REST quotes")


# ── Discovery ────────────────────────────────────────────────────────────────

def _discovery():
    log.info("discovery loop starting: %s", ", ".join(LEAGUES))
    while True:
        try:
            for league, cfg in LEAGUES.items():
                for e in _espn_scoreboard_any(cfg["espn"]):
                    g = _parse_espn_game(e)
                    if not g:
                        continue
                    gid = g["espn_id"]
                    with _active_lock:
                        if gid in _active:
                            continue
                        log.info(f"[{league}] live: {g['away_name']} @ {g['home_name']} ({g['status']})")
                        t = threading.Thread(target=_observe, args=(league, cfg, g),
                                             daemon=True, name=f"obs-{gid}")
                        _active[gid] = t
                        t.start()
        except Exception as e:
            log.warning(f"discovery err: {e}")
        time.sleep(DISCOVERY_INTERVAL)


def main():
    log.info("xmarket logger starting  csv=%s", LOG_CSV)
    _start_kalshi_ws()   # single-subscription WS book; REST is the fallback
    _discovery()


if __name__ == "__main__":
    main()
