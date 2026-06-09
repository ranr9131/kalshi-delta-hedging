"""
Touch-market passive-fill shadow logger  (Phase 0 of the sharky6999 replication).

GOAL — measure, WITHOUT risking capital, whether the touch model
(fair_price_model_v2.fair_p_no_touch_v2) actually finds mispriced "No" on
Polymarket crypto TOUCH markets, and whether that edge survives once a real
taker has to cross to you. This is the @sharky6999 strategy: be the maker
buying "No" on far-OTM "Will X reach/dip to $Y" markets, collecting the
premium retail overpays for the longshot "Yes".

Why Polymarket (not the Kalshi mm_shadow_logger this is forked from):
  - The touch markets ("reach $X in June" = hit at ANY time) and the retail
    longshot flow that creates the edge both live on Polymarket.
  - Kalshi crypto markets are mostly TERMINAL (settlement) — the wrong model.

What it does, per cycle:
  1. Discover open crypto touch markets via the gamma /events API
     (events titled "What price will <asset> hit ...", whose children are the
     individual "reach/dip to $X" markets). Terminal baskets ("<asset> above
     ___", "<asset> price on ...") are excluded.
  2. Track spot via coinbase_feeds (same feed the sniper uses) and feed it to
     the realized-σ estimator (record_price).
  3. For each market compute model No = fair_p_no_touch_v2(spot, strike, ...)
     and the moneyness gate z = touch_moneyness_z(...). A quote is only
     "active" only in the liquid wing (Z_MIN<=z<=Z_MAX), with edge>=EDGE_GATE_C
#     and real depth (>=MIN_BID_SZ).
  4. Rest a simulated No bid INSIDE_TICKS inside the book's best No bid.
  5. Watch the public data-api /trades feed. When a taker SELLS No at/through
     our resting bid, record a simulated FILL (we, the maker, BOUGHT No).
  6. Log a continuous snapshot stream so markout / adverse-selection can be
     computed offline (--report).

HONESTY: one-sided (we only buy No, like sharky — not two-sided MM). Quotes are
sticky with REQUOTE_SEC reaction latency; a taker hitting a stale quote IS the
adverse-selection event, and the markout (how far model-No drifts afterward)
measures its cost. Pure observation — NO orders placed.

Run:
  python3 touch_shadow_logger.py                 # collect
  python3 touch_shadow_logger.py --report        # analyse markout
  TOUCH_Z_MIN=0.7 TOUCH_Z_MAX=1.8 TOUCH_EDGE_C=3 python3 touch_shadow_logger.py
"""
from __future__ import annotations

import os
import sys
import csv
import re
import json
import time
import signal
import logging
import argparse
import threading
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coinbase_feeds import make_feed
from fair_price_model_v2 import (
    record_price, fair_p_no_touch_v2, touch_moneyness_z, effective_sigma_per_min,
    implied_sigma_from_touch,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
DATA  = "https://data-api.polymarket.com"

SNAP_CSV = os.path.join(ROOT, "touch_shadow_snapshots.csv")
FILL_CSV = os.path.join(ROOT, "touch_shadow_fills.csv")
MAP_JSON = os.path.join(ROOT, "touch_market_map.json")    # slug -> conditionId (persists past resolution)
RES_JSON = os.path.join(ROOT, "touch_resolutions.json")   # conditionId -> winning outcome (cache)

# ── Config ────────────────────────────────────────────────────────────────────
TAGS         = os.environ.get("TOUCH_TAGS", "bitcoin,ethereum,solana,ripple").split(",")
# Liquid-wing gate (retuned from the calibration finding): the edge lives in the
# MODERATELY out-of-the-money strikes, not the deepest tails. Quote only when the
# strike sits in the z-band, the edge clears EDGE_GATE_C, and there's real depth.
Z_MIN        = float(os.environ.get("TOUCH_Z_MIN", "0.7"))     # near edge of the wing
Z_MAX        = float(os.environ.get("TOUCH_Z_MAX", "1.8"))     # far edge (beyond = illiquid lottery)
MIN_BID_SZ   = float(os.environ.get("TOUCH_MIN_BID_SZ", "100"))# contracts of depth required
EDGE_GATE_C  = float(os.environ.get("TOUCH_EDGE_C", "3.0"))    # min model-edge, cents
ANCHOR_TGT   = float(os.environ.get("TOUCH_ANCHOR_TGT", "0.35"))  # σ-anchor: strike w/ mkt-touch nearest this
INSIDE_TICKS = float(os.environ.get("TOUCH_INSIDE", "0.1"))    # cents inside best No bid (Poly tick = 0.1c)
REQUOTE_SEC  = float(os.environ.get("TOUCH_REQUOTE_SEC", "5.0"))
SNAPSHOT_SEC = float(os.environ.get("TOUCH_SNAPSHOT_SEC", "5.0"))
REFRESH_SEC  = float(os.environ.get("TOUCH_REFRESH_SEC", "300"))
QUOTE_SIZE   = float(os.environ.get("TOUCH_QUOTE_SIZE", "20")) # contracts, for $ est
DRIFT_PER_MIN = float(os.environ.get("TOUCH_DRIFT", "0.0"))

COIN_PRODUCT = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "DOGE": "DOGE-USD", "BNB": "BNB-USD", "ADA": "ADA-USD", "LTC": "LTC-USD",
}
ASSET_ALIAS = {
    "bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL", "xrp": "XRP", "ripple": "XRP",
    "dogecoin": "DOGE", "doge": "DOGE",
}
# child-market question → (asset, direction, strike)
Q_PAT = re.compile(
    r"(bitcoin|ethereum|solana|xrp|ripple|dogecoin|btc|eth|sol|doge)\b.*?"
    r"(reach|hit|dip to|dip|fall to|drop to)\s*\$?\s*([\d,]+(?:\.\d+)?)",
    re.I,
)
UP_WORDS = {"reach", "hit"}
# These parse like price markets but the strike is an index/percent, not spot
# price ("volatility index dip to 25", "dominance hit 70", "kimchi premium 8").
# They'd feed nonsense strikes to the spot model — drop them.
EXCLUDE_RE = re.compile(
    r"volatilit|dominance|kimchi|premium|implied|\bindex\b|market\s?cap|"
    r"before gta|hashrate|fear|greed", re.I)
# Strike must be within this multiplicative band of spot to be a real price
# touch market (rejects index/percent strikes that slip past EXCLUDE_RE).
STRIKE_SPOT_BAND = (0.05, 20.0)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("touch_shadow")


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def parse_question(q: str):
    """('BTC','up',75000.0) from 'Will Bitcoin reach $75,000 in June?'. None if no match."""
    m = Q_PAT.search(q or "")
    if not m:
        return None
    asset = ASSET_ALIAS.get(m.group(1).lower())
    if asset is None:
        return None
    direction = "up" if m.group(2).lower() in UP_WORDS else "down"
    strike = _f(m.group(3).replace(",", ""))
    if strike <= 0:
        return None
    return asset, direction, strike


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover():
    """Return {no_token: meta}. meta = slug, asset, direction, strike, end_dt,
    no_token, cond. Touch baskets only (events titled 'what price will ... hit')."""
    out = {}
    for tag in TAGS:
        try:
            r = requests.get(f"{GAMMA}/events",
                             params={"closed": "false", "active": "true",
                                     "limit": 200, "tag_slug": tag.strip()},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            events = r.json()
        except Exception as e:
            log.warning(f"discover {tag} failed: {e}")
            continue
        for ev in events:
            title = (ev.get("title") or "").lower()
            # touch baskets phrase the event as "what price will X hit ..."; the
            # terminal baskets are "X above ___ on ..." / "X price on ...".
            if "hit" not in title and "reach" not in title:
                continue
            if "above" in title or "price on" in title:
                continue
            for m in ev.get("markets", []):
                if m.get("closed") or not m.get("active", True):
                    continue
                q = m.get("question", "")
                if EXCLUDE_RE.search(q) or EXCLUDE_RE.search(m.get("slug", "")):
                    continue
                parsed = parse_question(q)
                if not parsed:
                    continue
                asset, direction, strike = parsed
                if asset not in COIN_PRODUCT:
                    continue
                try:
                    toks = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
                    outs = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
                    no_token = toks[outs.index("No")]
                except Exception:
                    continue
                try:
                    end_dt = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
                except Exception:
                    continue
                out[no_token] = {
                    "slug": m.get("slug", "")[:48], "asset": asset,
                    "direction": direction, "strike": strike, "end_dt": end_dt,
                    "no_token": no_token, "cond": m.get("conditionId"),
                }
    return out


# ── State ─────────────────────────────────────────────────────────────────────
_tracked: dict[str, dict] = {}
_tracked_lock = threading.Lock()
_quotes: dict[str, dict] = {}          # no_token -> {bid_c, set_ts, active}
_feeds: dict[str, object] = {}         # asset -> feed
_trade_seen: set = set()
_trade_last_ts: dict[str, int] = {}    # cond -> last trade ts seen
_calib: dict = {}                      # group_key -> implied sigma_per_min
_running = True


def _spot(asset):
    f = _feeds.get(asset)
    return f.get_price() if f else None


def _minutes_left(meta):
    return max(0.0, (meta["end_dt"] - datetime.now(timezone.utc)).total_seconds() / 60.0)


def _model_no_c(meta, spot, sigma=None):
    """Model No value in cents, direction-aware, using the per-expiry market-
    implied σ when provided (else the realized-σ fallback). None if not
    computable. Guards the already-touched case (No has already lost)."""
    if spot is None:
        return None
    ml = _minutes_left(meta)
    if ml <= 0:
        return None
    lo, hi = STRIKE_SPOT_BAND
    if not (lo * spot <= meta["strike"] <= hi * spot):
        return None   # strike implausible vs spot → not a real price market
    if meta["direction"] == "up" and spot >= meta["strike"]:
        return 0.0   # already reached → No lost
    if meta["direction"] == "down" and spot <= meta["strike"]:
        return 0.0
    return fair_p_no_touch_v2(spot, meta["strike"], ml, meta["asset"],
                              sigma_per_min=sigma, drift_per_min=DRIFT_PER_MIN) * 100.0


def _group_key(meta):
    return (meta["asset"], meta["end_dt"].date().isoformat(), meta["direction"])


def persist_map(items):
    """Merge {slug: conditionId} into MAP_JSON. Never deletes, so once a market
    has been tracked we can still resolve it after it closes and drops off the
    open list (which is how settlement P&L gets attributed to old fills)."""
    try:
        m = json.load(open(MAP_JSON)) if os.path.exists(MAP_JSON) else {}
    except Exception:
        m = {}
    for _, meta in items:
        if meta.get("cond"):
            m[meta["slug"]] = meta["cond"]
    try:
        json.dump(m, open(MAP_JSON, "w"))
    except Exception as e:
        log.debug(f"persist_map err: {e}")


def market_resolution(cond, cache=None):
    """Winning outcome ('Yes'/'No') for a resolved market, else None. Uses CLOB
    /markets/<cond> (has per-token `winner` flags). Caches to RES_JSON."""
    if not cond:
        return None
    if cache is not None and cond in cache:
        return cache[cond]
    try:
        r = requests.get(f"{CLOB}/markets/{cond}", timeout=10)
        r.raise_for_status()
        mk = r.json()
    except Exception:
        return None
    res = None
    if mk.get("closed"):
        for t in mk.get("tokens", []):
            if t.get("winner"):
                res = t.get("outcome")
                break
    if cache is not None and res is not None:
        cache[cond] = res
    return res


def get_books_batch(tokens):
    """{token: (no_bid_c, no_ask_c, bid_sz)} via one POST /books per ~100 tokens."""
    out = {}
    for i in range(0, len(tokens), 100):
        chunk = tokens[i:i + 100]
        try:
            r = requests.post(f"{CLOB}/books",
                              json=[{"token_id": t} for t in chunk], timeout=15)
            r.raise_for_status()
            books = r.json()
        except Exception as e:
            log.debug(f"batch book err: {e}")
            continue
        for b in books if isinstance(books, list) else []:
            tok = b.get("asset_id")
            bids = b.get("bids") or []
            asks = b.get("asks") or []
            if not tok or not bids or not asks:
                continue
            bb = max(_f(x["price"]) for x in bids)
            ba = min(_f(x["price"]) for x in asks)
            bsz = sum(_f(x["size"]) for x in bids if abs(_f(x["price"]) - bb) < 1e-9)
            out[tok] = (round(bb * 100, 1), round(ba * 100, 1), round(bsz, 1))
    return out


def calibrate_groups(items, books):
    """Per (asset, expiry, direction) group, read σ off a near-the-money anchor
    strike's market touch-prob. Returns {group_key: sigma_per_min}."""
    grp = {}
    for no_token, meta in items:
        bk = books.get(no_token)
        spot = _spot(meta["asset"])
        if not bk or spot is None:
            continue
        no_bid_c, no_ask_c, _ = bk
        yes_mid = 1.0 - ((no_bid_c + no_ask_c) / 2.0) / 100.0
        ml = _minutes_left(meta)
        if ml <= 0 or not (0.03 < yes_mid < 0.97):
            continue
        grp.setdefault(_group_key(meta), []).append((yes_mid, meta["strike"], spot, ml))
    out = {}
    for key, legs in grp.items():
        if len(legs) < 3:
            continue   # need a curve to anchor
        anchor = min(legs, key=lambda L: abs(L[0] - ANCHOR_TGT))
        sig = implied_sigma_from_touch(anchor[0], anchor[2], anchor[1], anchor[3])
        if sig:
            out[key] = sig
    return out


def _z(meta, spot):
    if spot is None:
        return 0.0
    return touch_moneyness_z(spot, meta["strike"], _minutes_left(meta), meta["asset"])


# ── Order book ────────────────────────────────────────────────────────────────
def get_no_book(no_token):
    """(best_no_bid_c, best_no_ask_c, bid_size). None on failure."""
    try:
        r = requests.get(f"{CLOB}/book", params={"token_id": no_token}, timeout=10)
        r.raise_for_status()
        b = r.json()
    except Exception:
        return None
    bids = b.get("bids") or []
    asks = b.get("asks") or []
    if not bids or not asks:
        return None
    best_bid = max(_f(x["price"]) for x in bids)
    best_ask = min(_f(x["price"]) for x in asks)
    bid_sz = sum(_f(x["size"]) for x in bids if abs(_f(x["price"]) - best_bid) < 1e-9)
    return round(best_bid * 100, 1), round(best_ask * 100, 1), round(bid_sz, 1)


# ── Fill detection (public trades feed) ───────────────────────────────────────
def poll_fills(no_token, meta):
    """A taker SELLing No at/through our resting No bid = we (maker) bought No."""
    q = _quotes.get(no_token)
    if not q or not q.get("active") or q.get("bid_c") is None:
        return
    cond = meta["cond"]
    try:
        r = requests.get(f"{DATA}/trades",
                         params={"market": cond, "limit": 100,
                                 "takerOnly": "true"}, timeout=10)
        r.raise_for_status()
        trades = r.json()
    except Exception:
        return
    last = _trade_last_ts.get(cond, 0)
    newest = last
    spot = _spot(meta["asset"])
    fair = _model_no_c(meta, spot, _calib.get(_group_key(meta)))
    for t in trades:
        ts = int(_f(t.get("timestamp")))
        if ts <= last:
            continue
        newest = max(newest, ts)
        if t.get("asset") != no_token:
            continue
        if (t.get("side") or "").upper() != "SELL":
            continue   # taker selling No → hits a bid
        px_c = _f(t.get("price")) * 100.0
        if q["bid_c"] is None or px_c > q["bid_c"]:
            continue   # didn't reach our resting bid
        tid = t.get("transactionHash", "") + str(ts) + str(t.get("size"))
        if tid in _trade_seen:
            continue
        _trade_seen.add(tid)
        if fair is None:
            continue
        fill_c = q["bid_c"]                     # we get our resting maker price
        imm_edge = fair - fill_c                # long-No edge vs model
        _quotes[no_token]["bid_c"] = None       # finite resting order fills once
        row = {
            "ts": round(time.time(), 3), "slug": meta["slug"], "asset": meta["asset"],
            "direction": meta["direction"], "fill_price_c": round(fill_c, 2),
            "fair_no_c": round(fair, 2), "immediate_edge_c": round(imm_edge, 2),
            "z": round(_z(meta, spot), 2), "mins_left": round(_minutes_left(meta), 1),
            "trade_size": round(_f(t.get("size")), 2),
        }
        _append_csv(FILL_CSV, FILL_COLS, row)
        log.info(f"FILL buy_no {meta['slug'][:34]:34} @{fill_c:.0f}c "
                 f"fair={fair:.1f}c edge={imm_edge:+.1f}c z={row['z']}")
    _trade_last_ts[cond] = newest


# ── CSV ───────────────────────────────────────────────────────────────────────
SNAP_COLS = ["ts", "slug", "asset", "direction", "spot", "strike", "mins_left",
             "no_bid_c", "no_ask_c", "no_bid_sz", "fair_no_c", "edge_c", "z",
             "sigma_min", "my_bid_c", "active"]
FILL_COLS = ["ts", "slug", "asset", "direction", "fill_price_c", "fair_no_c",
             "immediate_edge_c", "z", "mins_left", "trade_size"]
_csv_lock = threading.Lock()


def _append_csv(path, cols, row):
    with _csv_lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)


# ── Collection ────────────────────────────────────────────────────────────────
def _ensure_feeds():
    with _tracked_lock:
        assets = {m["asset"] for m in _tracked.values()}
    for a in assets:
        if a not in _feeds and a in COIN_PRODUCT:
            f = make_feed(COIN_PRODUCT[a]); f.start(); _feeds[a] = f
            log.info(f"[feed] started {COIN_PRODUCT[a]}")


def _refresh_thread():
    while _running:
        cands = discover()
        if cands:
            with _tracked_lock:
                _tracked.clear(); _tracked.update(cands)
            _ensure_feeds()
            persist_map(list(cands.items()))
            log.info(f"[discover] tracking {len(cands)} touch markets")
        else:
            log.warning("[discover] none found")
        for _ in range(int(REFRESH_SEC)):
            if not _running:
                return
            time.sleep(1)


def collect():
    log.info(f"tags={TAGS} z_band={Z_MIN}-{Z_MAX} edge_gate={EDGE_GATE_C}c "
             f"min_bid_sz={MIN_BID_SZ} inside={INSIDE_TICKS}c requote={REQUOTE_SEC}s")
    cands = discover()
    with _tracked_lock:
        _tracked.update(cands)
    _ensure_feeds()
    persist_map(list(cands.items()))
    log.info(f"[discover] tracking {len(_tracked)} markets; warming feeds…")
    threading.Thread(target=_refresh_thread, daemon=True).start()
    time.sleep(4)

    global _calib
    while _running:
        t0 = time.time()
        with _tracked_lock:
            items = list(_tracked.items())
        # 1) feed realized-vol history
        for _, meta in items:
            sp = _spot(meta["asset"])
            if sp:
                record_price(meta["asset"], sp)
        # 2) one batch book fetch for every tracked No token
        books = get_books_batch([tok for tok, _ in items])
        # 3) read the market's implied σ per (asset, expiry, direction)
        _calib = calibrate_groups(items, books)

        for no_token, meta in items:
            spot = _spot(meta["asset"])
            book = books.get(no_token)
            if not book or spot is None:
                continue
            no_bid_c, no_ask_c, no_bid_sz = book
            sigma = _calib.get(_group_key(meta))
            fair = _model_no_c(meta, spot, sigma)
            z = _z(meta, spot)
            edge_c = (fair - no_ask_c) if fair is not None else None

            # sticky quote, retuned gate: rest a No bid one tick inside the best
            # No bid, ACTIVE only in the liquid wing (Z_MIN..Z_MAX), with real
            # edge after paying our bid, real depth, and a feasible spread.
            want_bid = round(no_bid_c + INSIDE_TICKS, 1)
            gate_ok = (
                fair is not None and sigma is not None
                and Z_MIN <= z <= Z_MAX
                and (fair - want_bid) >= EDGE_GATE_C
                and no_bid_sz >= MIN_BID_SZ
                and want_bid < no_ask_c
            )
            q = _quotes.get(no_token)
            if q is None or (time.time() - q["set_ts"]) >= REQUOTE_SEC \
               or (q.get("bid_c") is not None and q["bid_c"] >= no_ask_c):
                q = {"bid_c": (want_bid if gate_ok else None),
                     "active": gate_ok, "set_ts": time.time()}
                _quotes[no_token] = q

            poll_fills(no_token, meta)

            _append_csv(SNAP_CSV, SNAP_COLS, {
                "ts": round(t0, 3), "slug": meta["slug"], "asset": meta["asset"],
                "direction": meta["direction"],
                "spot": round(spot, 6) if spot else "", "strike": meta["strike"],
                "mins_left": round(_minutes_left(meta), 1),
                "no_bid_c": no_bid_c, "no_ask_c": no_ask_c, "no_bid_sz": no_bid_sz,
                "fair_no_c": round(fair, 2) if fair is not None else "",
                "edge_c": round(edge_c, 2) if edge_c is not None else "",
                "z": round(z, 2), "sigma_min": round(sigma, 6) if sigma else "",
                "my_bid_c": q.get("bid_c") or "", "active": int(bool(q.get("active"))),
            })
        time.sleep(max(0.0, SNAPSHOT_SEC - (time.time() - t0)))


# ── Report ────────────────────────────────────────────────────────────────────
def report(horizons=(60, 300, 900)):
    series: dict[str, list] = {}
    if os.path.exists(SNAP_CSV):
        with open(SNAP_CSV) as fh:
            for r in csv.DictReader(fh):
                if r["fair_no_c"] == "":
                    continue
                series.setdefault(r["slug"], []).append((float(r["ts"]), float(r["fair_no_c"])))
    for s in series:
        series[s].sort()

    def fair_at(slug, ts):
        arr = series.get(slug)
        if not arr:
            return None
        for s_ts, s_fair in arr:
            if s_ts >= ts:
                return s_fair
        return arr[-1][1]

    fills = []
    if os.path.exists(FILL_CSV):
        with open(FILL_CSV) as fh:
            fills = list(csv.DictReader(fh))

    if series:
        span = max(a[-1][0] for a in series.values()) - min(a[0][0] for a in series.values())
        active_rows = 0
        if os.path.exists(SNAP_CSV):
            with open(SNAP_CSV) as fh:
                active_rows = sum(1 for r in csv.DictReader(fh) if r.get("active") == "1")
        print(f"\nCoverage: {sum(len(a) for a in series.values())} snapshots across "
              f"{len(series)} markets, ~{span/3600:.1f}h; {active_rows} gate-passing rows.")

    if not fills:
        print("No simulated fills yet. Touch markets are slow — let it run hours, "
              "and confirm some rows passed the gate (active=1).")
        return

    print(f"\n=== Touch passive-fill report ({len(fills)} fills) ===\n")
    by = {}
    for f in fills:
        by.setdefault(f["slug"], []).append(f)
    hcols = "  ".join(f"mk{h}s" for h in horizons)
    print(f"{'market':36} {'n':>3} {'imm¢':>6}  {hcols}   {'advSel¢':>8}")
    print("-" * 92)
    tot = {"n": 0, "imm": 0.0, "mk": {h: 0.0 for h in horizons}}
    for slug, fs in sorted(by.items()):
        n = len(fs)
        imm = sum(float(f["immediate_edge_c"]) for f in fs) / n
        mk = {}
        for h in horizons:
            vals = []
            for f in fs:
                fa = fair_at(slug, float(f["ts"]) + h)
                if fa is not None:
                    vals.append(fa - float(f["fill_price_c"]))   # long-No markout
            mk[h] = sum(vals) / len(vals) if vals else float("nan")
        mkstr = "  ".join(f"{mk[h]:+5.1f}" for h in horizons)
        adv = imm - (mk[horizons[-1]] if mk[horizons[-1]] == mk[horizons[-1]] else 0)
        print(f"{slug[:36]:36} {n:>3} {imm:>+6.1f}  {mkstr}   {adv:>+8.1f}")
        tot["n"] += n; tot["imm"] += imm * n
        for h in horizons:
            tot["mk"][h] += (mk[h] if mk[h] == mk[h] else 0) * n
    n = tot["n"]
    print("-" * 92)
    mkstr = "  ".join(f"{tot['mk'][h]/n:+5.1f}" for h in horizons)
    print(f"{'ALL':36} {n:>3} {tot['imm']/n:>+6.1f}  {mkstr}")
    print("\n  imm¢   = model edge at fill (model-No − our bid)")
    print("  mkNs   = markout: model-No drift N sec after fill (realized long-No edge)")
    print("  advSel = imm − last markout (how much model moved against us post-fill)")
    print("\nVERDICT: the edge is real only if mkNs stays POSITIVE. If imm>0 but")
    print("markout decays to ~0/negative, you're being adversely selected — the")
    print("'No' was cheap precisely when spot was lurching toward the barrier.")
    span_h = (max(a[-1][0] for a in series.values()) - min(a[0][0] for a in series.values())) / 3600 if series else 0
    if span_h > 0:
        mkL = tot["mk"][horizons[-1]] / n
        fpd = n / span_h * 24
        print(f"\nObserved: {n} fills / ~{span_h:.1f}h → ~{fpd:.0f}/day. At {QUOTE_SIZE:.0f} "
              f"contracts/fill, mk={mkL:+.1f}¢ → ~${fpd*QUOTE_SIZE*mkL/100:+.1f}/day (pre-inventory-risk).")

    settlement_report(fills)


def settlement_report(fills):
    """Real resolved P&L: for fills whose market has settled, a buy-No pays
    (100 − fill) if 'No' won, else −fill. This is ground truth — the markout
    above is only vs the model; this is the money."""
    smap = json.load(open(MAP_JSON)) if os.path.exists(MAP_JSON) else {}
    cache = json.load(open(RES_JSON)) if os.path.exists(RES_JSON) else {}
    if not smap:
        print("\n[settlement] no market map yet — restart the logger so it writes "
              "touch_market_map.json, then markets can be resolved as they settle.")
        return

    resolved, unresolved, no_map = [], 0, 0
    for f in fills:
        cond = smap.get(f["slug"])
        if not cond:
            no_map += 1
            continue
        res = market_resolution(cond, cache)
        if res is None:
            unresolved += 1
            continue
        fill_c = float(f["fill_price_c"])
        pnl_c = (100.0 - fill_c) if res == "No" else (-fill_c)
        resolved.append({**f, "winner": res, "pnl_c": pnl_c})
    try:
        json.dump(cache, open(RES_JSON, "w"))
    except Exception:
        pass

    print(f"\n=== Settlement P&L (ground truth) ===")
    print(f"fills: {len(fills)}  resolved: {len(resolved)}  "
          f"still-open: {unresolved}  no-map: {no_map}")
    if not resolved:
        print("No tracked markets have settled yet. The near-dated (June) strikes "
              "resolve first — check back after they close.")
        return
    wins = sum(1 for r in resolved if r["winner"] == "No")
    tot_c = sum(r["pnl_c"] for r in resolved)
    avg_c = tot_c / len(resolved)
    print(f"win rate (No held): {wins}/{len(resolved)} = {100*wins/len(resolved):.0f}%")
    print(f"total realized: {tot_c:+.1f}¢/contract summed  |  avg {avg_c:+.2f}¢/fill  "
          f"|  ${tot_c*QUOTE_SIZE/100:+.2f} at {QUOTE_SIZE:.0f} contracts/fill")
    losers = sorted([r for r in resolved if r["pnl_c"] < 0], key=lambda r: r["pnl_c"])
    if losers:
        print(f"\nlosers ({len(losers)}) — the tail that matters:")
        for r in losers[:8]:
            print(f"  {r['pnl_c']:>+7.1f}¢  {r['slug'][:42]:42} (fill {float(r['fill_price_c']):.0f}c, Yes touched)")
    print("\nThis is the real test, not the markout. Watch avg¢/fill stay positive")
    print("as MORE markets settle — and watch whether one barrier-touch loser wipes")
    print("out many small No-wins (the correlated-tail risk).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tags", default=None)
    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--inside", type=float, default=None)
    ap.add_argument("--horizons", default="60,300,900")
    ap.add_argument("--discover-only", action="store_true",
                    help="print discovered markets and exit (no logging)")
    args = ap.parse_args()

    global TAGS, Z_MIN, Z_MAX, INSIDE_TICKS
    if args.tags:
        TAGS = args.tags.split(",")
    if args.z_min is not None:
        Z_MIN = args.z_min
    if args.z_max is not None:
        Z_MAX = args.z_max
    if args.inside is not None:
        INSIDE_TICKS = args.inside

    if args.report:
        report(tuple(int(x) for x in args.horizons.split(",")))
        return
    if args.discover_only:
        d = discover()
        print(f"discovered {len(d)} touch markets:")
        for meta in sorted(d.values(), key=lambda m: (m["asset"], m["strike"])):
            print(f"  {meta['asset']:4} {meta['direction']:4} {meta['strike']:>12,.2f}  {meta['slug']}")
        return

    def _stop(*_):
        global _running
        _running = False
        log.info("stopping…")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    collect()


if __name__ == "__main__":
    main()
