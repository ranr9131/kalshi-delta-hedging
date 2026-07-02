"""
Sports latency taker — PAPER evidence collector (strategy #3, RN1 pattern).

Thesis: league CDN feeds (NBA liveData, MLB statsapi GUMBO) run ~1-3s behind
the arena, while many quoters follow TV/streams 15-60s behind. After a scoring
event, stale quotes survive for seconds-to-minutes. This bot:

  1. Polls the fast feed (1s) and maintains a win-probability (WP) estimate,
     PRIOR-ANCHORED to the market's pregame price (so model deviations are
     event-driven, not model-bias-driven).
  2. On a score change, snapshots BOTH venues' books (Kalshi REST orderbook,
     Polymarket CLOB book). If a touch price deviates from model WP by more
     than (threshold + taker fee), records a SIMULATED take at the touch with
     its real size.
  3. Tracks each take's book for 180s (time-to-reprice, post-event drift) and
     settles every take at the final result — ground truth P&L.

Everything is paper. No orders are placed, no keys needed.

Run (during live games):
  python3 latency_taker.py                  # auto-discovers today's NBA + MLB
  python3 latency_taker.py --leagues nba    # NBA only
  python3 latency_taker.py --min-edge 6     # take threshold in cents
  python3 latency_taker.py --report         # summarize takes CSV

Outputs:
  latency_takes.csv  — one row per simulated take (+ reprice/markout columns)
  latency_quotes.csv — 1s book snapshots around events (for offline analysis)
  latency_taker.out  — log
"""
from __future__ import annotations

import os
import re
import csv
import json
import math
import time
import argparse
import threading
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
TAKES_CSV = os.path.join(ROOT, "latency_takes.csv")
QUOTES_CSV = os.path.join(ROOT, "latency_quotes.csv")

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

NBA_SB = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
MLB_SCHED = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=team"
NBA_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
MLB_LIVE = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

S = requests.Session()        # Kalshi / Gamma / CLOB / statsapi — plain headers
S_NBA = requests.Session()    # cdn.nba.com requires browser-like headers (Akamai)
S_NBA.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
})

TAKE_COLS = ["ts_take", "league", "game", "venue", "side_team", "contract", "event_desc",
             "score", "clock", "wp_model", "touch_px_c", "touch_sz", "edge_c", "fee_c",
             "ts_event_feed", "stale_age_s", "reprice_s", "px_after_60s_c",
             "px_after_180s_c", "final_result", "settle_pnl_c"]
QUOTE_COLS = ["ts", "league", "game", "venue", "team", "bid_c", "ask_c", "wp_model",
              "score", "clock", "phase"]


def log(msg):
    print(f'{datetime.now().strftime("%H:%M:%S")} {msg}', flush=True)


def _csv_append(path, cols, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        w.writerow([row.get(c, "") for c in cols])


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_phi(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── Win-probability models (margin = home − away) ────────────────────────────
NBA_SD_PER_SQRT_MIN = 1.73   # final-margin sd ≈ 12 pts over 48 min

def wp_nba(margin, mins_left, prior_z):
    """P(home win). prior_z = pregame strength in z-units, decays with time."""
    t = max(mins_left, 0.05)
    drift = prior_z * NBA_SD_PER_SQRT_MIN * math.sqrt(48.0) * (t / 48.0)
    return phi((margin + drift) / (NBA_SD_PER_SQRT_MIN * math.sqrt(t)))


MLB_SD_PER_SQRT_INN = 1.55   # final run-margin sd ≈ 4.6 over 9 innings

# RE24: expected runs in remainder of half-inning, (1st,2nd,3rd occupied) x outs
RE24 = {
    (0, 0, 0): (0.51, 0.27, 0.10), (1, 0, 0): (0.90, 0.55, 0.23),
    (0, 1, 0): (1.13, 0.69, 0.32), (0, 0, 1): (1.38, 0.95, 0.36),
    (1, 1, 0): (1.49, 0.93, 0.45), (1, 0, 1): (1.80, 1.19, 0.50),
    (0, 1, 1): (2.00, 1.41, 0.59), (1, 1, 1): (2.33, 1.57, 0.74),
}
RE_BASELINE = 0.51            # empty bases, 0 out (start of half-inning)

def wp_mlb(margin, innings_left, prior_z, dre_home=0.0):
    """dre_home = base-out run expectancy surplus, signed for the home team
    (negative when the away team has runners on / outs in hand)."""
    t = max(innings_left, 0.1)
    drift = prior_z * MLB_SD_PER_SQRT_INN * math.sqrt(9.0) * (t / 9.0)
    return phi((margin + dre_home + drift) / (MLB_SD_PER_SQRT_INN * math.sqrt(t)))


# ── Fee math ──────────────────────────────────────────────────────────────────
def kalshi_fee_c(px_c):
    p = px_c / 100.0
    return math.ceil(7.0 * p * (1 - p)) if px_c else 0


def poly_fee_c(px_c):           # sports taker rate 0.03
    p = px_c / 100.0
    return 100 * 0.03 * p * (1 - p)


# ── Market discovery ─────────────────────────────────────────────────────────
def kalshi_game_markets(series, date_token, abbr_a, abbr_b):
    """Find the two team tickers for a game. Ticker form:
    KXNBAGAME-26JUN10SASNYK-SAS"""
    try:
        r = S.get(f"{KALSHI}/markets",
                  params={"series_ticker": series, "status": "open", "limit": 100},
                  timeout=10).json()
    except Exception:
        return {}
    out = {}
    for m in r.get("markets", []):
        t = m["ticker"]
        if date_token in t and abbr_a in t and abbr_b in t:
            side = t.rsplit("-", 1)[-1]
            out[side] = t
    return out


def poly_game_tokens(slug_candidates):
    """gamma slug -> {team_abbr_or_name: yes_token}. Moneyline market has
    outcomes = [away, home] team names."""
    for slug in slug_candidates:
        try:
            g = S.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=10).json()
        except Exception:
            continue
        if isinstance(g, list) and g:
            m = g[0]
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
                outs = json.loads(m.get("outcomes") or "[]")
            except Exception:
                continue
            if len(toks) == 2 and len(outs) == 2:
                return {outs[i]: toks[i] for i in range(2)}, slug
    return {}, None


def kalshi_touch(ticker):
    """(bid_c, ask_c, ask_sz) for YES via orderbook."""
    try:
        ob = S.get(f"{KALSHI}/markets/{ticker}/orderbook", timeout=6).json().get("orderbook") or {}
    except Exception:
        return None, None, 0
    yes = ob.get("yes") or []     # resting YES bids [[px, sz]]
    no = ob.get("no") or []       # resting NO bids; YES ask = 100 - best NO bid
    bid = max((p for p, _ in yes), default=None)
    best_no = max((p for p, _ in no), default=None)
    ask = 100 - best_no if best_no is not None else None
    ask_sz = next((s for p, s in no if p == best_no), 0) if best_no is not None else 0
    return bid, ask, ask_sz


def poly_touch(token):
    """(bid_c, ask_c, ask_sz_shares) from CLOB book."""
    try:
        b = S.get(f"{CLOB}/book", params={"token_id": token}, timeout=6).json()
    except Exception:
        return None, None, 0
    bids = b.get("bids") or []
    asks = b.get("asks") or []
    bid = max((float(x["price"]) for x in bids), default=None)
    ask = min((float(x["price"]) for x in asks), default=None)
    ask_sz = sum(float(x["size"]) for x in asks if ask is not None and abs(float(x["price"]) - ask) < 1e-9)
    return (bid * 100 if bid else None), (ask * 100 if ask else None), ask_sz


# ── Game feed adapters ───────────────────────────────────────────────────────
class NbaGame:
    league = "nba"

    def __init__(self, meta):
        self.gid = meta["gameId"]
        self.home = meta["homeTeam"]["teamTricode"]
        self.away = meta["awayTeam"]["teamTricode"]
        self.name = f'{self.away}@{self.home}'
        self.status = meta.get("gameStatus", 1)   # 1 sched, 2 live, 3 final
        self.margin = 0
        self.mins_left = 48.0
        self.clock = ""
        self.last_desc = ""

    def update(self):
        try:
            js = S_NBA.get(NBA_PBP.format(gid=self.gid), timeout=5).json()
            acts = js.get("game", {}).get("actions", [])
        except Exception:
            return False
        if not acts:
            return False
        a = acts[-1]
        # pbp exists -> live; explicit end action (actionType=game, subType=end) -> final
        if a.get("actionType") == "game" and a.get("subType") == "end":
            self.status = 3
        elif self.status == 1:
            self.status = 2
        try:
            h, aw = int(a.get("scoreHome", 0)), int(a.get("scoreAway", 0))
        except Exception:
            return False
        per = int(a.get("period", 1))
        m = re.match(r"PT(\d+)M([\d.]+)S", a.get("clock") or "")
        rem_in_period = (int(m.group(1)) + float(m.group(2)) / 60) if m else 0.0
        reg_left = max(0.0, (4 - min(per, 4)) * 12.0) + rem_in_period
        changed = (h - aw) != self.margin or abs(reg_left - self.mins_left) > 3
        self.margin = h - aw
        self.mins_left = reg_left if per <= 4 else rem_in_period
        self.clock = f'Q{per} {a.get("clock", "")}'
        self.last_desc = (a.get("description") or "")[:60]
        self.score = f"{aw}-{h}"
        return changed

    def wp_home(self, prior_z):
        return wp_nba(self.margin, self.mins_left, prior_z)


class MlbGame:
    league = "mlb"

    def __init__(self, meta):
        self.pk = meta["gamePk"]
        t = meta["teams"]
        self.home = (t["home"]["team"].get("abbreviation") or t["home"]["team"]["name"]).upper()
        self.away = (t["away"]["team"].get("abbreviation") or t["away"]["team"]["name"]).upper()
        self.home_name = t["home"]["team"]["name"]
        self.away_name = t["away"]["team"]["name"]
        self.name = f"{self.away}@{self.home}"
        self.status = 1
        self.margin = 0
        self.innings_left = 9.0
        self.clock = ""
        self.last_desc = ""
        self.score = "0-0"

    def update(self):
        try:
            js = S.get(MLB_LIVE.format(pk=self.pk), timeout=5).json()
            ls = js["liveData"]["linescore"]
            state = js["gameData"]["status"]["abstractGameState"]
        except Exception:
            return False
        self.status = {"Preview": 1, "Live": 2, "Final": 3}.get(state, 1)
        h = ls.get("teams", {}).get("home", {}).get("runs", 0) or 0
        a = ls.get("teams", {}).get("away", {}).get("runs", 0) or 0
        inn = ls.get("currentInning", 1) or 1
        half = ls.get("inningHalf", "Top")
        outs = min(ls.get("outs", 0) or 0, 2)
        # base-out run expectancy for the batting team (mid-inning state)
        off = ls.get("offense", {}) or {}
        bases = (1 if off.get("first") else 0, 1 if off.get("second") else 0,
                 1 if off.get("third") else 0)
        dre = RE24.get(bases, RE24[(0, 0, 0)])[outs] - RE_BASELINE
        away_batting = half.lower() != "bottom"
        self.dre_home = -dre if away_batting else dre
        done = (inn - 1) + (0.5 if half.lower() == "bottom" else 0.0) + outs / 6.0
        changed = (h - a) != self.margin or abs(self.dre_home - getattr(self, "_last_dre", 0)) > 0.45
        self._last_dre = self.dre_home
        self.margin = h - a
        self.innings_left = max(0.0, 9.0 - done)
        self.clock = f"{half} {inn}, {outs} out, b{''.join(map(str,bases))}"
        self.score = f"{a}-{h}"
        desc = js.get("liveData", {}).get("plays", {}).get("currentPlay", {}) \
                 .get("result", {}).get("description") or ""
        self.last_desc = desc[:60]
        return changed

    def wp_home(self, prior_z):
        return wp_mlb(self.margin, self.innings_left, prior_z,
                      getattr(self, "dre_home", 0.0))


# ── Per-game tracker ─────────────────────────────────────────────────────────
class Tracker:
    def __init__(self, game, kalshi_tk, poly_tok, poly_team_map):
        self.g = game
        self.kalshi = kalshi_tk          # {abbr: ticker}
        self.poly = poly_tok             # {team_name_or_abbr: token}
        self.poly_team_map = poly_team_map  # {'home': name_key, 'away': name_key}
        self.prior_z = None
        self.open_takes = []
        self.takes_done = 0

    def anchor_prior(self):
        """Set pregame strength so model == market before tip."""
        px = None
        home_key = self.poly_team_map.get("home")
        if home_key and self.poly.get(home_key):
            b, a, _ = poly_touch(self.poly[home_key])
            if b and a:
                px = (b + a) / 200.0
        if px is None and self.kalshi.get(self.g.home):
            b, a, _ = kalshi_touch(self.kalshi[self.g.home])
            if b and a:
                px = (b + a) / 200.0
        if px:
            self.prior_z = inv_phi(px)
            log(f"[{self.g.name}] prior anchored: home {px*100:.0f}c -> z={self.prior_z:+.2f}")
        return px is not None

    def books(self):
        """[(venue, team_label, is_home, bid, ask, ask_sz, contract)]"""
        out = []
        for abbr, tk in self.kalshi.items():
            b, a, sz = kalshi_touch(tk)
            out.append(("kalshi", abbr, abbr == self.g.home, b, a, sz, tk))
        for key, tok in self.poly.items():
            b, a, sz = poly_touch(tok)
            is_home = key == self.poly_team_map.get("home")
            out.append(("poly", str(key)[:12], is_home, b, a, sz, tok))
        return out

    _last_quote_log = 0.0

    def on_tick(self, changed, min_edge_c, ts_event):
        if self.prior_z is None:
            return
        now = time.time()
        # full book reads are needed on events; on quiet ticks only every 10s
        if not changed and now - self._last_quote_log < 10:
            return
        self._last_quote_log = now
        wp_h = self.g.wp_home(self.prior_z)
        wp_move = wp_h - getattr(self, "_prev_wp_h", wp_h)   # event direction
        self._prev_wp_h = wp_h
        # states the Gaussian can't price — observe, never take
        g = self.g
        untakeable = False
        if g.league == "mlb":
            loaded = getattr(g, "dre_home", 0) and abs(g.dre_home) > 1.0
            untakeable = g.innings_left < 1.0 or loaded
        elif g.league == "nba":
            untakeable = g.mins_left < 1.0
        rows = self.books()
        for venue, team, is_home, bid, ask, sz, contract in rows:
            wp = wp_h if is_home else 1 - wp_h
            _csv_append(QUOTES_CSV, QUOTE_COLS, {
                "ts": round(now, 1), "league": self.g.league, "game": self.g.name,
                "venue": venue, "team": team, "bid_c": bid, "ask_c": ask,
                "wp_model": round(wp * 100, 1), "score": self.g.score,
                "clock": self.g.clock, "phase": "event" if changed else "tick"})
            if not changed or ask is None or sz <= 0:
                continue
            fee = kalshi_fee_c(ask) if venue == "kalshi" else poly_fee_c(ask)
            edge = wp * 100 - ask - fee
            # gate: only chase the event's own direction (stale-quote pattern),
            # never fight the market in states the model prices worse than it
            event_favors_side = (wp_move > 0.005) if is_home else (wp_move < -0.005)
            if edge >= min_edge_c and event_favors_side and not untakeable:
                take = {
                    "ts_take": round(now, 1), "league": self.g.league, "game": self.g.name,
                    "venue": venue, "side_team": team, "contract": contract,
                    "event_desc": self.g.last_desc, "score": self.g.score,
                    "clock": self.g.clock, "wp_model": round(wp * 100, 1),
                    "touch_px_c": round(ask, 1), "touch_sz": round(sz, 1),
                    "edge_c": round(edge, 1), "fee_c": round(fee, 2),
                    "ts_event_feed": round(ts_event, 1),
                    "stale_age_s": round(now - ts_event, 1),
                    "_watch_until": now + 180, "_repriced": None, "_p60": None,
                    "_is_home": is_home,
                }
                self.open_takes.append(take)
                log(f"  TAKE[{venue}] {team} ask {ask:.0f}c sz{sz:.0f} vs wp {wp*100:.0f} "
                    f"edge +{edge:.1f}c after '{self.g.last_desc}' ({self.g.score} {self.g.clock})")

    def watch_takes(self):
        now = time.time()
        for t in self.open_takes[:]:
            venue, contract = t["venue"], t["contract"]
            b, a, _ = kalshi_touch(contract) if venue == "kalshi" else poly_touch(contract)
            age = now - t["ts_take"]
            if a is not None:
                if t["_repriced"] is None and a >= t["wp_model"] - 2:
                    t["_repriced"] = age
                if t["_p60"] is None and age >= 60:
                    t["_p60"] = a
            if age >= 180 or self.g.status == 3:
                t["reprice_s"] = round(t["_repriced"], 1) if t["_repriced"] else ""
                t["px_after_60s_c"] = t["_p60"] if t["_p60"] is not None else ""
                t["px_after_180s_c"] = a if a is not None else ""
                self.open_takes.remove(t)
                self._settle_later.append(t)
                self.takes_done += 1

    _settle_later: list

    def settle(self):
        """At game end: final result from feed margin."""
        home_won = self.g.margin > 0
        for t in self._settle_later:
            won = home_won if t["_is_home"] else not home_won
            pnl = (100 - t["touch_px_c"] - t["fee_c"]) if won else (-t["touch_px_c"] - t["fee_c"])
            t["final_result"] = "W" if won else "L"
            t["settle_pnl_c"] = round(pnl, 1)
            _csv_append(TAKES_CSV, TAKE_COLS, t)
        n = len(self._settle_later)
        if n:
            tot = sum(t["settle_pnl_c"] for t in self._settle_later)
            log(f"[{self.g.name}] settled {n} takes: {tot:+.0f}c total")
        self._settle_later = []


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover_nba():
    out = []
    games = []
    try:
        js = S_NBA.get(NBA_SB, timeout=8).json()
        games = js.get("scoreboard", {}).get("games", [])
    except Exception as e:
        log(f"nba scoreboard err: {e}")
    if not games:
        # scoreboard hasn't rolled over yet — fall back to the season schedule
        try:
            sch = S_NBA.get(NBA_SCHEDULE, timeout=20).json()
            want = datetime.now().strftime("%m/%d/%Y")
            for d in sch.get("leagueSchedule", {}).get("gameDates", []):
                if want in (d.get("gameDate") or ""):
                    games = [{"gameId": gm["gameId"], "gameStatus": 1,
                              "homeTeam": {"teamTricode": gm["homeTeam"]["teamTricode"]},
                              "awayTeam": {"teamTricode": gm["awayTeam"]["teamTricode"]}}
                             for gm in d.get("games", [])]
                    break
            if games:
                log(f"nba: scoreboard empty, schedule fallback found {len(games)} games")
        except Exception as e:
            log(f"nba schedule err: {e}")
    for gm in games:
        g = NbaGame(gm)
        date_tok = datetime.now().strftime("%y%b%d").upper()
        ktk = kalshi_game_markets("KXNBAGAME", date_tok, g.away, g.home)
        d = datetime.now().strftime("%Y-%m-%d")
        slugs = [f"nba-{g.away.lower()}-{g.home.lower()}-{d}",
                 f"nba-{g.home.lower()}-{g.away.lower()}-{d}"]
        ptok, slug = poly_game_tokens(slugs)
        # verified: gamma outcomes order is [away, home] for nba-/mlb- game slugs
        keys = list(ptok.keys())
        team_map = {"away": keys[0], "home": keys[1]} if len(keys) == 2 else {}
        if ktk or ptok:
            tr = Tracker(g, ktk, ptok, team_map)
            tr._settle_later = []
            out.append(tr)
            log(f"NBA {g.name}: kalshi={list(ktk.values())} poly={slug} status={g.status}")
    return out


def discover_mlb():
    out = []
    d = datetime.now().strftime("%Y-%m-%d")
    try:
        js = S.get(MLB_SCHED.format(date=d), timeout=8).json()
        games = [g for day in js.get("dates", []) for g in day.get("games", [])]
    except Exception as e:
        log(f"mlb sched err: {e}")
        return out
    for gm in games:
        try:
            g = MlbGame(gm)
        except Exception:
            continue
        date_tok = datetime.now().strftime("%y%b%d").upper()
        ktk = {}
        try:
            r = S.get(f"{KALSHI}/markets", params={"series_ticker": "KXMLBGAME",
                      "status": "open", "limit": 200}, timeout=10).json()
            for m in r.get("markets", []):
                t = m["ticker"]
                # ticker mid-segment: {YY}{MON}{DD}{HHMM}{AWAY}{HOME}
                seg = t.split("-")[1] if "-" in t else ""
                if date_tok in seg and seg.endswith(g.away + g.home):
                    ktk[t.rsplit("-", 1)[-1]] = t
        except Exception:
            pass
        slugs = [f"mlb-{g.away.lower()}-{g.home.lower()}-{d}"]
        ptok, slug = poly_game_tokens(slugs)
        team_map = {"away": list(ptok)[0], "home": list(ptok)[1]} if len(ptok) == 2 else {}
        if ktk or ptok:
            tr = Tracker(g, ktk, ptok, team_map)
            tr._settle_later = []
            out.append(tr)
            log(f"MLB {g.name}: kalshi={len(ktk)} poly={slug}")
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(leagues, min_edge_c, poll_s):
    trackers = []
    if "nba" in leagues:
        trackers += discover_nba()
    if "mlb" in leagues:
        trackers += discover_mlb()
    if not trackers:
        log("no games matched today — exiting")
        return
    log(f"tracking {len(trackers)} games, min_edge={min_edge_c}c, paper only")
    anchored = set()
    last_pregame_poll = {}
    while trackers:
        for tr in trackers[:]:
            g = tr.g
            now = time.time()
            # pregame: poll feed/books only every 60s to stay polite all afternoon
            if g.status == 1 and now - last_pregame_poll.get(id(tr), 0) < 60:
                continue
            changed = g.update()
            ts_event = time.time()
            if g.status == 1:
                last_pregame_poll[id(tr)] = now
                if id(tr) not in anchored and tr.anchor_prior():
                    anchored.add(id(tr))
                continue
            if id(tr) not in anchored:
                # never anchor mid-game: a live mid reflects the current score,
                # so using it as the pregame prior double-counts the game state
                log(f"[{g.name}] went live without a pregame anchor — dropping")
                trackers.remove(tr)
                continue
            if g.status == 2:
                tr.on_tick(changed, min_edge_c, ts_event)
                tr.watch_takes()
            elif g.status == 3:
                tr.watch_takes()
                tr.settle()
                trackers.remove(tr)
                log(f"[{g.name}] FINAL {g.score} — done ({tr.takes_done} takes)")
        time.sleep(poll_s)
    log("all games final — exiting")


def report():
    if not os.path.exists(TAKES_CSV):
        print("no takes logged yet")
        return
    rows = list(csv.DictReader(open(TAKES_CSV)))
    if not rows:
        print("no takes")
        return
    import statistics
    pnl = [float(r["settle_pnl_c"]) for r in rows if r["settle_pnl_c"]]
    won = sum(1 for r in rows if r["final_result"] == "W")
    stale = [float(r["stale_age_s"]) for r in rows if r["stale_age_s"]]
    rep = [float(r["reprice_s"]) for r in rows if r["reprice_s"]]
    by_venue = {}
    for r in rows:
        by_venue.setdefault(r["venue"], []).append(float(r["settle_pnl_c"] or 0))
    print(f"takes: {len(rows)}  win: {won}/{len(pnl)}  "
          f"avg pnl: {statistics.mean(pnl):+.1f}c  total: {sum(pnl):+.0f}c")
    print(f"stale age at take: median {statistics.median(stale):.1f}s" if stale else "")
    print(f"book reprice time: median {statistics.median(rep):.0f}s" if rep else "(no reprices tracked)")
    for v, p in by_venue.items():
        print(f"  {v}: n={len(p)} avg {statistics.mean(p):+.1f}c")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="nba,mlb")
    ap.add_argument("--min-edge", type=float, default=5.0)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run([x.strip() for x in a.leagues.split(",")], a.min_edge, a.poll)
