#!/usr/bin/env python3.11
"""
Tennis cross-market logger: Kalshi (KXATPMATCH/KXWTAMATCH/KXCHALLENGERMATCH)
vs Polymarket (tag=tennis) aligned two-sided books for the same match.

Observation ONLY — places no orders. Modeled on xmarket_logger.py.

Discovery (every DISCOVER_SEC):
  - Poly: gamma /events?tag_slug=tennis&closed=false, singles slugs
    '(atp|wta|itf|challenger)-<p1>-<p2>-YYYY-MM-DD'; match-winner market =
    the "A vs B" question (not Completed/Set/O-U/Doubles), outcomes = players.
  - Kalshi: /markets?series_ticker=... (quotes only populate per-series);
    two markets per match, grouped by ticker event stem
    (KXWTAMATCH-26JUL14PUTLIU-PUT -> stem 26JUL14PUTLIU).
  - Pair by (date within ±1 day) + both player last-name tokens matching
    (prefix match both ways — Kalshi yes_sub_title truncates ~20 chars).

Quote loop (every POLL_SEC):
  - Kalshi bulk per-series GET (bid/ask per player market).
  - Poly CLOB POST /books batch (token ids from gamma clobTokenIds).
  - Writes one CSV row per matched match per tick; prints LOCK lines when
    fee-adjusted cross-venue lock cost < $1.

Resolution caveat (logged, not handled): Kalshi resolves "wins after a ball
has been played"; Poly resolves "advances" and goes 50-50 on cancellation or
7-day delay. Retirement/walkover rows are NOT directly comparable locks.
"""
import csv
import json
import math
import os
import re
import signal
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "tennis_xmarket_quotes.csv")

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

KALSHI_SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXCHALLENGERMATCH"]
POLL_SEC = 8            # box shares Kalshi's per-IP rate limit with many services
DISCOVER_SEC = 600
KALSHI_429_COOLDOWN = 60
LOCK_ALERT_EDGE = 0.02          # print LOCK when net edge >= 2c
DAY_WINDOW = 1                  # pair matches within +/- 1 day

MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
       "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

_shutdown = False


def _sig(_s, _f):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)


def http_json(url, payload=None, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": "tennis-xmarket/1.0",
                                               "Content-Type": "application/json"})
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
        return json.load(r)


def norm_tokens(name):
    """Normalized name tokens, accents stripped, len>=3, drop particles."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    toks = re.split(r"[^a-z]+", s.lower())
    return [t for t in toks if len(t) >= 3 and t not in ("van", "der", "den", "del", "de", "la")]


def names_match(a, b):
    """True if names share a token by prefix either way (handles truncation)."""
    ta, tb = norm_tokens(a), norm_tokens(b)
    for x in ta:
        for y in tb:
            if len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x)):
                return True
    return False


def kalshi_fee(price):
    """Taker fee per contract, dollars (ceil to cent per exchange docs)."""
    return math.ceil(7 * price * (1 - price)) / 100.0


# ── discovery ──────────────────────────────────────────────────────────────────

def poly_tennis_matches():
    """[{slug, date, p1, p2, q, token1, token2}] singles match-winner markets."""
    out = []
    offset = 0
    while offset < 500:
        try:
            evs = http_json(f"{GAMMA_BASE}/events?tag_slug=tennis&closed=false"
                            f"&limit=100&offset={offset}")
        except Exception as e:
            print(f"[discover] poly events error: {e}", flush=True)
            break
        if not evs:
            break
        for e in evs:
            slug = e.get("slug", "")
            m = re.match(r"^(atp|wta|itf|challenger)-(?!doubles)\S*-(\d{4}-\d{2}-\d{2})$", slug)
            if not m:
                continue
            date = m.group(2)
            for mk in e.get("markets", []):
                q = mk.get("question", "")
                if (" vs" not in q or "Completed" in q or "Set" in q
                        or "O/U" in q or "Doubles" in q or "Total" in q):
                    continue
                try:
                    outcomes = json.loads(mk.get("outcomes") or "[]")
                    tokens = json.loads(mk.get("clobTokenIds") or "[]")
                except Exception:
                    continue
                if len(outcomes) != 2 or len(tokens) != 2:
                    continue
                out.append({"slug": slug, "date": date, "p1": outcomes[0],
                            "p2": outcomes[1], "q": q,
                            "token1": tokens[0], "token2": tokens[1]})
                break   # one winner market per event
        if len(evs) < 100:
            break
        offset += 100
    return out


def kalshi_ticker_date(ticker):
    """KXWTAMATCH-26JUL14PUTLIU-PUT -> '2026-07-14' (None if unparseable)."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ticker)
    if not m or m.group(2) not in MON:
        return None
    return f"20{m.group(1)}-{MON[m.group(2)]:02d}-{int(m.group(3)):02d}"


def kalshi_matches():
    """{stem: {date, series, players: [{ticker, name}]}} from open match markets.

    Returns {} when rate-limited — caller must keep its existing pairs then.
    """
    global _k_cooldown_until
    if time.time() < _k_cooldown_until:
        return {}
    stems = {}
    for series in KALSHI_SERIES:
        try:
            d = http_json(f"{KALSHI_BASE}/markets?series_ticker={series}"
                          f"&status=open&limit=1000")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _k_cooldown_until = time.time() + KALSHI_429_COOLDOWN
                print(f"[discover] kalshi 429 — cooldown, keeping prior pairs",
                      flush=True)
                return {}
            print(f"[discover] kalshi {series} error: {e}", flush=True)
            continue
        except Exception as e:
            print(f"[discover] kalshi {series} error: {e}", flush=True)
            continue
        for mk in d.get("markets", []):
            ticker = mk.get("ticker", "")
            stem = ticker.rsplit("-", 1)[0]
            date = kalshi_ticker_date(ticker)
            if not date:
                continue
            name = mk.get("yes_sub_title") or mk.get("subtitle") or ""
            stems.setdefault(stem, {"date": date, "series": series, "players": []})
            stems[stem]["players"].append({"ticker": ticker, "name": name})
    return {k: v for k, v in stems.items() if len(v["players"]) == 2}


def pair_venues(poly, kalshi):
    """Return [{key, poly:{...}, k1/k2 tickers aligned to poly p1/p2, ...}]."""
    pairs = []
    used = set()
    for stem, km in kalshi.items():
        kd = datetime.strptime(km["date"], "%Y-%m-%d")
        ka, kb = km["players"]
        for pm in poly:
            if pm["slug"] in used:
                continue
            pd = datetime.strptime(pm["date"], "%Y-%m-%d")
            if abs((kd - pd).days) > DAY_WINDOW:
                continue
            if names_match(ka["name"], pm["p1"]) and names_match(kb["name"], pm["p2"]):
                k1, k2 = ka, kb
            elif names_match(ka["name"], pm["p2"]) and names_match(kb["name"], pm["p1"]):
                k1, k2 = kb, ka
            else:
                continue
            used.add(pm["slug"])
            pairs.append({"key": f"{km['date']}|{pm['slug']}", "stem": stem,
                          "series": km["series"], "poly": pm,
                          "k1_ticker": k1["ticker"], "k2_ticker": k2["ticker"],
                          "p1": pm["p1"], "p2": pm["p2"]})
            break
    return pairs


# ── quotes ─────────────────────────────────────────────────────────────────────

_k_cooldown_until = 0.0
_k_last_quotes = {}


def kalshi_quotes(series_list):
    """{ticker: (bid, ask)} for all open markets in the given series.

    On 429 (shared per-IP limit), back off KALSHI_429_COOLDOWN s and serve the
    last good snapshot so Poly-side logging continues uninterrupted.
    """
    global _k_cooldown_until, _k_last_quotes
    now = time.time()
    if now < _k_cooldown_until:
        return _k_last_quotes
    q = {}
    for series in series_list:
        try:
            d = http_json(f"{KALSHI_BASE}/markets?series_ticker={series}"
                          f"&status=open&limit=1000")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _k_cooldown_until = time.time() + KALSHI_429_COOLDOWN
                print(f"[kalshi] 429 — cooling down {KALSHI_429_COOLDOWN}s, "
                      f"serving cached quotes", flush=True)
                return _k_last_quotes
            continue
        except Exception:
            continue
        for mk in d.get("markets", []):
            try:
                bid = float(mk.get("yes_bid_dollars") or 0)
                ask = float(mk.get("yes_ask_dollars") or 0)
            except (TypeError, ValueError):
                continue
            q[mk.get("ticker", "")] = (bid, ask)
    if q:
        _k_last_quotes = q
    return _k_last_quotes


def poly_books(token_ids):
    """{token_id: (best_bid, best_ask)} via batched CLOB /books."""
    out = {}
    for i in range(0, len(token_ids), 50):
        chunk = token_ids[i:i + 50]
        try:
            books = http_json(f"{CLOB_BASE}/books",
                              payload=[{"token_id": t} for t in chunk])
        except Exception:
            continue
        for b in books or []:
            tid = b.get("asset_id") or b.get("token_id") or ""
            bids = [float(x["price"]) for x in b.get("bids", []) if x.get("price")]
            asks = [float(x["price"]) for x in b.get("asks", []) if x.get("price")]
            out[tid] = (max(bids) if bids else 0.0, min(asks) if asks else 0.0)
    return out


CSV_FIELDS = ["ts_iso", "ts_unix", "key", "series", "k_stem", "poly_slug",
              "p1", "p2",
              "k1_bid", "k1_ask", "k2_bid", "k2_ask",
              "pp1_bid", "pp1_ask", "pp2_bid", "pp2_ask",
              "lock_k1p2", "lock_k2p1", "best_edge_c"]


def main():
    once = "--once" in sys.argv
    writer_new = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if writer_new:
        writer.writeheader()

    pairs, last_disc = [], 0.0
    print(f"tennis xmarket logger starting | poll={POLL_SEC}s "
          f"discover={DISCOVER_SEC}s csv={CSV_PATH}", flush=True)
    while not _shutdown:
        now = time.time()
        if now - last_disc > DISCOVER_SEC or not pairs:
            poly = poly_tennis_matches()
            kalshi = kalshi_matches()
            if kalshi:
                new_pairs = pair_venues(poly, kalshi)
                changed = {p["key"] for p in new_pairs} != {p["key"] for p in pairs}
                pairs = new_pairs
                last_disc = now
                print(f"[discover] poly singles={len(poly)} kalshi matches="
                      f"{len(kalshi)} paired={len(pairs)}", flush=True)
                if changed:
                    for p in pairs:
                        print(f"  paired: {p['stem']} <-> {p['poly']['slug']}",
                              flush=True)
            else:
                # rate-limited: keep prior pairs, retry discovery in 60s
                last_disc = now - DISCOVER_SEC + 60

        if pairs:
            kq = kalshi_quotes(sorted({p["series"] for p in pairs}))
            tokens = [t for p in pairs for t in (p["poly"]["token1"], p["poly"]["token2"])]
            pb = poly_books(tokens)
            ts = datetime.now(timezone.utc)
            for p in pairs:
                k1 = kq.get(p["k1_ticker"], (0.0, 0.0))
                k2 = kq.get(p["k2_ticker"], (0.0, 0.0))
                pp1 = pb.get(p["poly"]["token1"], (0.0, 0.0))
                pp2 = pb.get(p["poly"]["token2"], (0.0, 0.0))
                # lock A: buy p1 on Kalshi + p2 on Poly (Poly sports taker fee = 0)
                lock_a = (k1[1] + kalshi_fee(k1[1]) + pp2[1]) if (k1[1] and pp2[1]) else 0.0
                # lock B: buy p2 on Kalshi + p1 on Poly
                lock_b = (k2[1] + kalshi_fee(k2[1]) + pp1[1]) if (k2[1] and pp1[1]) else 0.0
                edges = [1.0 - c for c in (lock_a, lock_b) if c > 0]
                best_edge = max(edges) if edges else -1.0
                writer.writerow({
                    "ts_iso": ts.isoformat(timespec="milliseconds"),
                    "ts_unix": round(ts.timestamp(), 3),
                    "key": p["key"], "series": p["series"], "k_stem": p["stem"],
                    "poly_slug": p["poly"]["slug"], "p1": p["p1"], "p2": p["p2"],
                    "k1_bid": k1[0], "k1_ask": k1[1],
                    "k2_bid": k2[0], "k2_ask": k2[1],
                    "pp1_bid": pp1[0], "pp1_ask": pp1[1],
                    "pp2_bid": pp2[0], "pp2_ask": pp2[1],
                    "lock_k1p2": round(lock_a, 4), "lock_k2p1": round(lock_b, 4),
                    "best_edge_c": round(best_edge * 100, 2),
                })
                if best_edge >= LOCK_ALERT_EDGE:
                    side = "K:%s+P:%s" % ((p["p1"], p["p2"]) if lock_a and 1 - lock_a == best_edge
                                          else (p["p2"], p["p1"]))
                    print(f"LOCK {ts.strftime('%H:%M:%S')} {p['stem']} "
                          f"edge={best_edge*100:.1f}c ({side}) "
                          f"k1={k1[0]:.2f}/{k1[1]:.2f} k2={k2[0]:.2f}/{k2[1]:.2f} "
                          f"pp1={pp1[0]:.2f}/{pp1[1]:.2f} pp2={pp2[0]:.2f}/{pp2[1]:.2f}",
                          flush=True)
            fh.flush()

        if once:
            break
        time.sleep(POLL_SEC)

    fh.close()
    print("tennis xmarket logger exiting", flush=True)


if __name__ == "__main__":
    main()
