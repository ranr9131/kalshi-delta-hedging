#!/usr/bin/env python3
"""Execution-quality paper audit for Kalshi/Polymarket live-tennis locks.

This process is deliberately unable to trade.  It imports only market-data
clients, reads public venue metadata, and writes two CSVs:

* ``tennis_lock_quotes_v2.csv``: fresh, full-depth live-match snapshots.
* ``tennis_lock_attempts_v2.csv``: simulated Kalshi-first/FOK then delayed
  Polymarket hedge attempts.

The original tennis logger was useful for discovery but had two material
false-positive paths: first-name token matching and cached Kalshi REST quotes
after HTTP 429 responses.  This collector uses family-name matching and the
authenticated Kalshi order-book WebSocket, records depth/age, requires actual
in-play price movement, and never serves cached REST quotes as current.

No order module, wallet, private key signing for orders, POST order endpoint,
or cancellation endpoint is present.  Kalshi credentials are used only to
authenticate the read-only order-book WebSocket.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import signal
import sys
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable

import requests
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_auth
import kalshi_orderbook
import polymarket_orderbook


ROOT = os.path.dirname(os.path.abspath(__file__))
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KALSHI_SERIES = ("KXATPMATCH", "KXWTAMATCH", "KXCHALLENGERMATCH")
MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

SIZE = float(os.environ.get("TENNIS_LOCK_SIZE", "5"))
MIN_EDGE = float(os.environ.get("TENNIS_LOCK_MIN_EDGE", "0.04"))
LIVE_RANGE = float(os.environ.get("TENNIS_LOCK_LIVE_RANGE", "0.02"))
LIVE_WINDOW_SEC = float(os.environ.get("TENNIS_LOCK_LIVE_WINDOW", "600"))
MIN_LIVE_OBS = int(os.environ.get("TENNIS_LOCK_MIN_LIVE_OBS", "5"))
MAX_BOOK_AGE = float(os.environ.get("TENNIS_LOCK_MAX_BOOK_AGE", "3"))
PERSIST_SEC = float(os.environ.get("TENNIS_LOCK_PERSIST_SEC", "1"))
HEDGE_DELAY_SEC = float(os.environ.get("TENNIS_LOCK_HEDGE_DELAY", "0.25"))
HEDGE_SLIP = float(os.environ.get("TENNIS_LOCK_HEDGE_SLIP", "0.03"))
REARM_SEC = float(os.environ.get("TENNIS_LOCK_REARM_SEC", "10"))
DISCOVER_SEC = float(os.environ.get("TENNIS_LOCK_DISCOVER_SEC", "180"))
TICK_SEC = float(os.environ.get("TENNIS_LOCK_TICK_SEC", "0.20"))
QUOTE_LOG_SEC = float(os.environ.get("TENNIS_LOCK_QUOTE_LOG_SEC", "2"))
DATE_WINDOW_DAYS = int(os.environ.get("TENNIS_LOCK_DATE_WINDOW", "1"))

QUOTE_CSV = os.environ.get(
    "TENNIS_LOCK_QUOTE_CSV", os.path.join(ROOT, "tennis_lock_quotes_v2.csv")
)
ATTEMPT_CSV = os.environ.get(
    "TENNIS_LOCK_ATTEMPT_CSV", os.path.join(ROOT, "tennis_lock_attempts_v2.csv")
)
PAIR_CSV = os.environ.get(
    "TENNIS_LOCK_PAIR_CSV", os.path.join(ROOT, "tennis_lock_pairs_v2.csv")
)

QUOTE_COLS = [
    "ts", "ts_iso", "key", "series", "kalshi_stem", "poly_slug",
    "p1", "p2", "live_range_c", "live_observations", "direction",
    "k_ticker", "k_top", "k_vwap", "k_visible", "k_age",
    "p_token", "p_top", "p_vwap", "p_visible", "p_age",
    "p_source_age", "k_fee", "p_fee", "lock_cost", "edge_c",
    "eligible", "reason",
]
ATTEMPT_COLS = [
    "attempt_id", "signal_ts", "signal_iso", "hedge_ts", "key",
    "series", "kalshi_stem", "poly_slug", "p1", "p2", "direction",
    "k_player", "p_player", "size", "live_range_c", "persist_sec",
    "signal_edge_c", "k_ticker", "k_top", "k_vwap", "k_visible",
    "k_age", "k_fee", "p_token", "p_signal_top", "p_signal_vwap",
    "p_signal_visible", "p_signal_age", "p_limit", "delay_sec",
    "p_hedge_top", "p_hedge_vwap", "p_hedge_visible_at_limit",
    "p_hedge_age", "p_fee", "status", "completed_lock_cost",
    "completed_edge_c", "completed_profit", "unhedged_contracts", "note",
]
PAIR_COLS = [
    "discovered_ts", "key", "series", "kalshi_stem", "kalshi_date",
    "poly_slug", "poly_date", "p1", "p2", "k1_name", "k2_name",
    "k1_ticker", "k2_ticker", "condition_id", "fee_rate", "fee_exponent",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s tennis-lock %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tennis_lock_shadow")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "tennis-lock-shadow/2.0"})
_shutdown = False


def _stop(_signum, _frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return result if isinstance(result, list) else []


def _get(base: str, path: str, params=None) -> dict | list:
    response = SESSION.get(base + path, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def _append(path: str, columns: list[str], row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()


def normalized_tokens(name: str) -> list[str]:
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    tokens = re.split(r"[^a-z]+", text.lower())
    particles = {"van", "von", "der", "den", "del", "della", "de", "da", "la", "le"}
    return [token for token in tokens if len(token) >= 2 and token not in particles]


def family_name(name: str) -> str:
    """Return the trailing normalized token; never match solely on first name."""
    tokens = normalized_tokens(name)
    return tokens[-1] if tokens else ""


def surname_tokens(name: str) -> list[str]:
    """Possible surname components, excluding the first/given-name token.

    This retains compound surnames and remains safe against the original bug:
    two players sharing only ``Luka/Lukas`` or ``Carlos`` cannot match.
    """
    tokens = normalized_tokens(name)
    return tokens[1:] if len(tokens) > 1 else tokens


def strict_name_match(kalshi_name: str, poly_name: str) -> bool:
    """Conservative surname match with support for compounds/truncation."""
    for left in surname_tokens(kalshi_name):
        for right in surname_tokens(poly_name):
            if (
                len(left) >= 4 and len(right) >= 4
                and (left.startswith(right) or right.startswith(left))
            ):
                return True
    return False


def kalshi_date(ticker: str) -> str | None:
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ticker)
    if not match or match.group(2) not in MONTH:
        return None
    return f"20{match.group(1)}-{MONTH[match.group(2)]:02d}-{int(match.group(3)):02d}"


def kalshi_fee(fills: Iterable[tuple[float, float]]) -> float:
    raw = sum(0.07 * qty * price * (1.0 - price) for price, qty in fills)
    return math.ceil(max(0.0, raw) * 100.0 - 1e-12) / 100.0


def poly_fee(fills: Iterable[tuple[float, float]], rate: float, exponent: float) -> float:
    if rate <= 0:
        return 0.0
    if abs(exponent - 1.0) > 1e-9:
        raise ValueError(f"unsupported Polymarket fee exponent {exponent}")
    return round(sum(qty * rate * price * (1.0 - price) for price, qty in fills), 5)


def walk_book(
    levels: Iterable[tuple[float, float]], size: float, limit: float | None = None
) -> tuple[float | None, list[tuple[float, float]], float, float | None]:
    """Return VWAP, fills, visible-at-limit, worst price; require a full fill."""
    clean = sorted(
        (float(price), float(qty))
        for price, qty in levels
        if 0 < float(price) < 1 and float(qty) > 0
        and (limit is None or float(price) <= limit + 1e-12)
    )
    visible = sum(qty for _, qty in clean)
    remaining, cost, fills, worst = float(size), 0.0, [], None
    for price, qty in clean:
        take = min(remaining, qty)
        if take > 0:
            fills.append((price, take))
            cost += price * take
            remaining -= take
            worst = price
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        return None, fills, visible, None
    return cost / size, fills, visible, worst


@dataclass(frozen=True)
class MatchPair:
    key: str
    series: str
    stem: str
    kalshi_date: str
    poly_slug: str
    poly_date: str
    p1: str
    p2: str
    k1_name: str
    k2_name: str
    k1_ticker: str
    k2_ticker: str
    token1: str
    token2: str
    condition_id: str
    fee_rate: float
    fee_exponent: float


def discover_poly() -> list[dict]:
    out = []
    for offset in range(0, 500, 100):
        events = _get(
            GAMMA, "/events",
            {"tag_slug": "tennis", "closed": "false", "limit": 100, "offset": offset},
        )
        if not isinstance(events, list) or not events:
            break
        for event in events:
            slug = str(event.get("slug") or "")
            match = re.match(
                r"^(atp|wta|itf|challenger)-(?!doubles)\S*-(\d{4}-\d{2}-\d{2})$",
                slug,
            )
            if not match:
                continue
            for market in event.get("markets") or []:
                question = str(market.get("question") or "")
                blocked = ("Completed", "Set", "O/U", "Doubles", "Total")
                if " vs" not in question or any(value in question for value in blocked):
                    continue
                if market.get("closed") or not market.get("active") or not market.get("acceptingOrders"):
                    continue
                outcomes = _json_list(market.get("outcomes"))
                tokens = _json_list(market.get("clobTokenIds"))
                if len(outcomes) != 2 or len(tokens) != 2:
                    continue
                condition = str(market.get("conditionId") or "")
                out.append({
                    "slug": slug,
                    "date": match.group(2),
                    "p1": str(outcomes[0]),
                    "p2": str(outcomes[1]),
                    "token1": str(tokens[0]),
                    "token2": str(tokens[1]),
                    "condition_id": condition,
                    "fee_rate": 0.0,
                    "fee_exponent": 1.0,
                })
                break
        if len(events) < 100:
            break
    return out


def discover_kalshi() -> dict[str, dict]:
    stems = {}
    for series in KALSHI_SERIES:
        payload = _get(
            KALSHI, "/markets",
            {"series_ticker": series, "status": "open", "limit": 1000},
        )
        for market in payload.get("markets") or []:
            ticker = str(market.get("ticker") or "")
            date = kalshi_date(ticker)
            if not ticker or not date:
                continue
            stem = ticker.rsplit("-", 1)[0]
            name = str(market.get("yes_sub_title") or market.get("subtitle") or "")
            row = stems.setdefault(
                stem, {"date": date, "series": series, "players": []}
            )
            row["players"].append({"ticker": ticker, "name": name})
    return {
        stem: row for stem, row in stems.items()
        if len(row["players"]) == 2 and all(item["name"] for item in row["players"])
    }


def pair_matches(poly: list[dict], kalshi: dict[str, dict]) -> list[MatchPair]:
    """Conservative one-to-one pairing; ambiguous matches are rejected."""
    candidates = []
    for stem, kmatch in kalshi.items():
        kdate = datetime.strptime(kmatch["date"], "%Y-%m-%d")
        first, second = kmatch["players"]
        found = []
        for pmatch in poly:
            pdate = datetime.strptime(pmatch["date"], "%Y-%m-%d")
            if abs((kdate - pdate).days) > DATE_WINDOW_DAYS:
                continue
            if strict_name_match(first["name"], pmatch["p1"]) and strict_name_match(
                second["name"], pmatch["p2"]
            ):
                k1, k2 = first, second
            elif strict_name_match(first["name"], pmatch["p2"]) and strict_name_match(
                second["name"], pmatch["p1"]
            ):
                k1, k2 = second, first
            else:
                continue
            found.append((pmatch, k1, k2))
        if len(found) != 1:
            if len(found) > 1:
                log.warning("rejecting ambiguous pair %s candidates=%d", stem, len(found))
            continue
        pmatch, k1, k2 = found[0]
        candidates.append(MatchPair(
            key=f"{kmatch['date']}|{pmatch['slug']}",
            series=kmatch["series"], stem=stem,
            kalshi_date=kmatch["date"], poly_slug=pmatch["slug"],
            poly_date=pmatch["date"], p1=pmatch["p1"], p2=pmatch["p2"],
            k1_name=k1["name"], k2_name=k2["name"],
            k1_ticker=k1["ticker"], k2_ticker=k2["ticker"],
            token1=pmatch["token1"], token2=pmatch["token2"],
            condition_id=pmatch["condition_id"], fee_rate=pmatch["fee_rate"],
            fee_exponent=pmatch["fee_exponent"],
        ))

    # A Poly market may map to only one Kalshi match.  Reject duplicate slugs.
    counts = defaultdict(int)
    for pair in candidates:
        counts[pair.poly_slug] += 1
    return [pair for pair in candidates if counts[pair.poly_slug] == 1]


def fee_terms(condition_id: str) -> tuple[float, float] | None:
    """Fetch live fee terms; unknown fees cause the pair to be dropped."""
    if not condition_id:
        return None
    try:
        info = _get(CLOB, f"/clob-markets/{condition_id}")
        fee = info.get("fd") or {}
        return _float(fee.get("r"), 0.0), _float(fee.get("e"), 1.0)
    except requests.RequestException as error:
        log.warning("fee metadata failed condition=%s: %s", condition_id, error)
        return None


@dataclass
class DirectionQuote:
    direction: str
    k_player: str
    p_player: str
    k_ticker: str
    p_token: str
    k_top: float
    k_vwap: float
    k_visible: float
    k_age: float
    k_fee: float
    p_top: float
    p_vwap: float
    p_visible: float
    p_age: float
    p_source_age: float
    p_fee: float
    lock_cost: float
    edge: float


def _direction_quote(pair: MatchPair, direction: int, now: float) -> DirectionQuote | None:
    if direction == 0:
        k_player, p_player = pair.p1, pair.p2
        k_ticker, p_token = pair.k1_ticker, pair.token2
        label = "K:p1+P:p2"
    else:
        k_player, p_player = pair.p2, pair.p1
        k_ticker, p_token = pair.k2_ticker, pair.token1
        label = "K:p2+P:p1"
    kbook = kalshi_orderbook.get_book(k_ticker)
    pbook = polymarket_orderbook.get_book(p_token)
    if not kbook or not kbook.snapshot_seen or not pbook or not pbook.snapshot_seen:
        return None
    k_levels = kbook.yes_asks_sorted()
    k_vwap, k_fills, k_visible, _ = walk_book(k_levels, SIZE)
    p_vwap, p_fills, p_visible, _ = walk_book(pbook.asks, SIZE)
    if k_vwap is None or p_vwap is None:
        return None
    try:
        k_fee = kalshi_fee(k_fills)
        p_fee = poly_fee(p_fills, pair.fee_rate, pair.fee_exponent)
    except ValueError:
        return None
    lock = k_vwap + p_vwap + (k_fee + p_fee) / SIZE
    source_age = now - pbook.source_timestamp if pbook.source_timestamp else math.inf
    return DirectionQuote(
        direction=label, k_player=k_player, p_player=p_player,
        k_ticker=k_ticker, p_token=p_token,
        k_top=k_levels[0][0], k_vwap=k_vwap, k_visible=k_visible,
        k_age=kbook.age(), k_fee=k_fee,
        p_top=pbook.asks[0][0], p_vwap=p_vwap, p_visible=p_visible,
        p_age=pbook.age(), p_source_age=source_age, p_fee=p_fee,
        lock_cost=lock, edge=1.0 - lock,
    )


def _eligibility(quote: DirectionQuote, live: bool) -> tuple[bool, str]:
    if not live:
        return False, "not_live"
    if quote.k_age > MAX_BOOK_AGE:
        return False, "kalshi_book_old"
    if quote.p_age > MAX_BOOK_AGE or quote.p_source_age > MAX_BOOK_AGE:
        return False, "poly_book_old"
    if not (0.01 <= quote.k_top <= 0.98 and 0.01 <= quote.p_top <= 0.98):
        return False, "terminal_price"
    if quote.edge + 1e-12 < MIN_EDGE:
        return False, "edge_below_threshold"
    return True, "eligible"


@dataclass
class PendingAttempt:
    pair: MatchPair
    quote: DirectionQuote
    signal_ts: float
    live_range: float
    persist_sec: float
    p_limit: float


class Shadow:
    def __init__(self):
        self.pairs: list[MatchPair] = []
        self.history: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self.eligible_since: dict[tuple[str, str], float] = {}
        self.armed: dict[tuple[str, str], bool] = defaultdict(lambda: True)
        self.last_ineligible: dict[tuple[str, str], float] = {}
        self.pending: list[PendingAttempt] = []
        self.last_quote_log: dict[tuple[str, str], float] = {}
        self.attempt_sequence = 0
        self.last_discover = 0.0

    def discover(self, now: float) -> None:
        poly = discover_poly()
        kalshi = discover_kalshi()
        pairs = pair_matches(poly, kalshi)
        fee_verified = []
        for pair in pairs:
            terms = fee_terms(pair.condition_id)
            if terms is not None:
                fee_verified.append(replace(
                    pair, fee_rate=terms[0], fee_exponent=terms[1]
                ))
        pairs = fee_verified
        self.pairs = pairs
        kalshi_orderbook.set_tickers(
            [ticker for pair in pairs for ticker in (pair.k1_ticker, pair.k2_ticker)]
        )
        polymarket_orderbook.set_tokens(
            [token for pair in pairs for token in (pair.token1, pair.token2)]
        )
        self.last_discover = now
        for pair in pairs:
            _append(PAIR_CSV, PAIR_COLS, {
                "discovered_ts": now, "key": pair.key, "series": pair.series,
                "kalshi_stem": pair.stem, "kalshi_date": pair.kalshi_date,
                "poly_slug": pair.poly_slug, "poly_date": pair.poly_date,
                "p1": pair.p1, "p2": pair.p2,
                "k1_name": pair.k1_name, "k2_name": pair.k2_name,
                "k1_ticker": pair.k1_ticker, "k2_ticker": pair.k2_ticker,
                "condition_id": pair.condition_id, "fee_rate": pair.fee_rate,
                "fee_exponent": pair.fee_exponent,
            })
        log.info(
            "discover poly=%d kalshi=%d strict_pairs=%d tickers=%d tokens=%d",
            len(poly), len(kalshi), len(pairs), 2 * len(pairs), 2 * len(pairs),
        )

    def _update_live(self, pair: MatchPair, now: float) -> tuple[bool, float, int]:
        pbook = polymarket_orderbook.get_book(pair.token1)
        if pbook and pbook.snapshot_seen and pbook.midpoint() is not None:
            history = self.history[pair.key]
            midpoint = pbook.midpoint()
            if not history or now - history[-1][0] >= 1.0:
                history.append((now, midpoint))
            cutoff = now - LIVE_WINDOW_SEC
            while history and history[0][0] < cutoff:
                history.popleft()
        history = self.history[pair.key]
        if not history:
            return False, 0.0, 0
        values = [value for _, value in history]
        price_range = max(values) - min(values)
        live = len(values) >= MIN_LIVE_OBS and price_range + 1e-12 >= LIVE_RANGE
        return live, price_range, len(values)

    def _log_quote(
        self, pair: MatchPair, quote: DirectionQuote, now: float,
        live_range: float, observations: int, eligible: bool, reason: str,
    ) -> None:
        key = (pair.key, quote.direction)
        if now - self.last_quote_log.get(key, 0.0) < QUOTE_LOG_SEC:
            return
        self.last_quote_log[key] = now
        _append(QUOTE_CSV, QUOTE_COLS, {
            "ts": now, "ts_iso": _iso(now), "key": pair.key,
            "series": pair.series, "kalshi_stem": pair.stem,
            "poly_slug": pair.poly_slug, "p1": pair.p1, "p2": pair.p2,
            "live_range_c": round(100 * live_range, 3),
            "live_observations": observations, "direction": quote.direction,
            "k_ticker": quote.k_ticker, "k_top": quote.k_top,
            "k_vwap": round(quote.k_vwap, 5), "k_visible": quote.k_visible,
            "k_age": round(quote.k_age, 3), "p_token": quote.p_token,
            "p_top": quote.p_top, "p_vwap": round(quote.p_vwap, 5),
            "p_visible": quote.p_visible, "p_age": round(quote.p_age, 3),
            "p_source_age": round(quote.p_source_age, 3),
            "k_fee": quote.k_fee, "p_fee": quote.p_fee,
            "lock_cost": round(quote.lock_cost, 6),
            "edge_c": round(100 * quote.edge, 3),
            "eligible": int(eligible), "reason": reason,
        })

    def _schedule(
        self, pair: MatchPair, quote: DirectionQuote, now: float,
        live_range: float, persist_sec: float,
    ) -> None:
        # Precommit a Poly hedge limit.  Require the lock to remain positive at
        # that full slippage limit before simulating the Kalshi first leg.
        p_limit = min(0.99, quote.p_top + HEDGE_SLIP)
        limit_fee_per_share = (
            pair.fee_rate * p_limit * (1.0 - p_limit)
            if pair.fee_rate > 0 and abs(pair.fee_exponent - 1.0) <= 1e-9 else 0.0
        )
        worst_lock = quote.k_vwap + quote.k_fee / SIZE + p_limit + limit_fee_per_share
        if worst_lock >= 1.0:
            return
        self.pending.append(PendingAttempt(
            pair=pair, quote=quote, signal_ts=now, live_range=live_range,
            persist_sec=persist_sec, p_limit=p_limit,
        ))
        self.armed[(pair.key, quote.direction)] = False
        log.info(
            "PAPER LEG1 %s %s size=%.2f K@%.3f edge=%.2fc liveRange=%.1fc",
            pair.stem, quote.direction, SIZE, quote.k_vwap,
            100 * quote.edge, 100 * live_range,
        )

    def _finish_pending(self, now: float) -> None:
        ready = [item for item in self.pending if now - item.signal_ts >= HEDGE_DELAY_SEC]
        for item in ready:
            self.pending.remove(item)
            pbook = polymarket_orderbook.get_book(item.quote.p_token)
            p_vwap = None
            p_fills = []
            p_visible = 0.0
            p_top = None
            p_age = math.inf
            if pbook and pbook.snapshot_seen:
                p_top = pbook.best_ask()
                p_age = pbook.age()
                p_vwap, p_fills, p_visible, _ = walk_book(
                    pbook.asks, SIZE, item.p_limit
                )
            p_fee = 0.0
            status = "complete"
            note = "displayed Kalshi leg plus delayed displayed Poly hedge"
            if p_vwap is None:
                status = "underhedged"
                note = "delayed Poly depth could not fill full size at precommitted limit"
            elif p_age > MAX_BOOK_AGE:
                status = "stale_hedge_book"
                note = "delayed Poly book exceeded freshness limit"
            else:
                try:
                    p_fee = poly_fee(
                        p_fills, item.pair.fee_rate, item.pair.fee_exponent
                    )
                except ValueError as error:
                    status, note = "unsupported_fee", str(error)
            completed_lock = ""
            completed_edge = ""
            completed_profit = ""
            unhedged = SIZE
            if status == "complete" and p_vwap is not None:
                completed_lock_value = (
                    item.quote.k_vwap + p_vwap
                    + (item.quote.k_fee + p_fee) / SIZE
                )
                completed_lock = round(completed_lock_value, 6)
                completed_edge = round(100 * (1.0 - completed_lock_value), 3)
                completed_profit = round(SIZE * (1.0 - completed_lock_value), 4)
                unhedged = 0.0
            self.attempt_sequence += 1
            attempt_id = f"{int(item.signal_ts * 1000)}-{self.attempt_sequence}"
            _append(ATTEMPT_CSV, ATTEMPT_COLS, {
                "attempt_id": attempt_id, "signal_ts": item.signal_ts,
                "signal_iso": _iso(item.signal_ts), "hedge_ts": now,
                "key": item.pair.key, "series": item.pair.series,
                "kalshi_stem": item.pair.stem, "poly_slug": item.pair.poly_slug,
                "p1": item.pair.p1, "p2": item.pair.p2,
                "direction": item.quote.direction,
                "k_player": item.quote.k_player, "p_player": item.quote.p_player,
                "size": SIZE, "live_range_c": round(100 * item.live_range, 3),
                "persist_sec": round(item.persist_sec, 3),
                "signal_edge_c": round(100 * item.quote.edge, 3),
                "k_ticker": item.quote.k_ticker, "k_top": item.quote.k_top,
                "k_vwap": round(item.quote.k_vwap, 5),
                "k_visible": item.quote.k_visible, "k_age": round(item.quote.k_age, 3),
                "k_fee": item.quote.k_fee, "p_token": item.quote.p_token,
                "p_signal_top": item.quote.p_top,
                "p_signal_vwap": round(item.quote.p_vwap, 5),
                "p_signal_visible": item.quote.p_visible,
                "p_signal_age": round(item.quote.p_age, 3),
                "p_limit": item.p_limit, "delay_sec": round(now - item.signal_ts, 3),
                "p_hedge_top": p_top if p_top is not None else "",
                "p_hedge_vwap": round(p_vwap, 5) if p_vwap is not None else "",
                "p_hedge_visible_at_limit": round(p_visible, 4),
                "p_hedge_age": round(p_age, 3) if math.isfinite(p_age) else "",
                "p_fee": p_fee, "status": status,
                "completed_lock_cost": completed_lock,
                "completed_edge_c": completed_edge,
                "completed_profit": completed_profit,
                "unhedged_contracts": unhedged, "note": note,
            })
            log.info(
                "PAPER %s %s status=%s delayedP=%s completedEdge=%s",
                item.pair.stem, item.quote.direction, status,
                "n/a" if p_vwap is None else f"{p_vwap:.3f}",
                "n/a" if completed_edge == "" else f"{completed_edge:.2f}c",
            )

    def tick(self, now: float) -> None:
        self._finish_pending(now)
        for pair in self.pairs:
            live, live_range, observations = self._update_live(pair, now)
            for direction in (0, 1):
                quote = _direction_quote(pair, direction, now)
                if quote is None:
                    continue
                key = (pair.key, quote.direction)
                eligible, reason = _eligibility(quote, live)
                if live or quote.edge >= 0:
                    self._log_quote(
                        pair, quote, now, live_range, observations, eligible, reason
                    )
                if not eligible:
                    self.eligible_since.pop(key, None)
                    self.last_ineligible.setdefault(key, now)
                    if not self.armed[key] and now - self.last_ineligible[key] >= REARM_SEC:
                        self.armed[key] = True
                    continue
                self.last_ineligible.pop(key, None)
                start = self.eligible_since.setdefault(key, now)
                persisted = now - start
                if self.armed[key] and persisted + 1e-12 >= PERSIST_SEC:
                    self._schedule(pair, quote, now, live_range, persisted)

    def run(self) -> None:
        while not _shutdown:
            now = time.time()
            if not self.pairs or now - self.last_discover >= DISCOVER_SEC:
                try:
                    self.discover(now)
                except Exception as error:
                    log.exception("discovery failed: %s", error)
                    self.last_discover = now - DISCOVER_SEC + 30
            try:
                self.tick(now)
            except Exception as error:
                log.exception("tick failed: %s", error)
            time.sleep(TICK_SEC)


def main() -> None:
    env = dotenv_values(os.path.join(ROOT, ".env"))
    private_key_text = env.get("KALSHI_PRIVATE_KEY")
    api_key_id = env.get("KALSHI_API_KEY_ID")
    if not private_key_text or not api_key_id:
        raise RuntimeError("Kalshi read-only WebSocket credentials are missing from live/.env")
    private_key = kalshi_auth.load_private_key(private_key_text)
    kalshi_orderbook.start(private_key, api_key_id, [])
    polymarket_orderbook.start([])
    log.info(
        "starting PAPER ONLY size=%.2f minEdge=%.1fc liveRange=%.1fc/%ss "
        "maxAge=%.1fs persist=%.1fs hedgeDelay=%dms",
        SIZE, 100 * MIN_EDGE, 100 * LIVE_RANGE, int(LIVE_WINDOW_SEC),
        MAX_BOOK_AGE, PERSIST_SEC, round(1000 * HEDGE_DELAY_SEC),
    )
    Shadow().run()
    log.info("stopped")


if __name__ == "__main__":
    main()
