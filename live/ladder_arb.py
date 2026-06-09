"""
Intra-ladder relative-value / no-arbitrage scanner for Kalshi crypto hourly
ladders (Strategy #1).  MARKET-NEUTRAL and SETTLEMENT-INDEX-NEUTRAL: every strike
in an event settles on the SAME 60s-average index at the SAME time, so their
prices must obey hard no-arb relations.  We hunt internal inconsistencies and
hedge them with neighbors — no directional bet, no BRTI-vs-Coinbase basis risk.

Two ladders per asset/event:
  * THRESHOLD  (KXBTCD / KXETHD): YES = settle > strike.  P(>k) must be
    non-increasing in k.
  * RANGE      (KXBTC / KXETH):  mutually-exclusive, exhaustive buckets
    (less-tail + betweens + greater-tail) -> YES prices must sum to ~1.

Checks (model-FREE locked arbs first, then model edge):
  A. THRESHOLD monotonicity: pair k_i<k_j with bid(k_j) >= ask(k_i) + fees ->
     buy YES(k_i)@ask, sell YES(k_j)@bid; payoff 1{k_i<settle<=k_j} >= 0 and you
     got a credit -> LOCKED.
  B. RANGE bucket-sum: buy ALL buckets' YES @ask; exactly one pays $1; if
     sum(ask)+fees < 100c -> LOCKED.  (Or sum(bid)-fees > 100c -> sell all.)
  C. CROSS-LADDER: RANGE between[a,b] has the SAME payoff as THRESHOLD
     [YES(>a-eps) - YES(>b)].  If one side is cheaper than the other by > fees ->
     LOCKED (same index, same close).
  D. MODEL edge (not locked): a strike cheap/rich vs v3 fair beyond half-spread,
     flagged for a neighbor-hedged relative-value trade.

Fees: Kalshi taker ~ 7*p*(1-p) c/contract/leg (low at extreme prices).

Usage:
  python3 ladder_arb.py --assets BTC,ETH            # one scan
  python3 ladder_arb.py --assets BTC --loop 5       # scan every 5s, log to CSV
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_auth
from dotenv import dotenv_values
import fair_price_model_v3 as v3

ROOT = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(ROOT, ".env"))
KID = env.get("KALSHI_API_KEY_ID", "")
PK = kalshi_auth.load_private_key(env.get("KALSHI_PRIVATE_KEY", "")) if env.get("KALSHI_PRIVATE_KEY") else None
B = "https://api.elections.kalshi.com"
CB = "https://api.exchange.coinbase.com"
LOG = os.path.join(ROOT, "ladder_arb_hits.csv")

THRESH = {"BTC": "KXBTCD", "ETH": "KXETHD", "SOL": "KXSOLD", "XRP": "KXXRPD"}
RANGE = {"BTC": "KXBTC", "ETH": "KXETH"}
PRODUCT = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD"}


def fee_c(p):
    p = min(max(p, 0.0), 1.0)
    return 7.0 * p * (1.0 - p)


def spot(asset):
    try:
        return float(requests.get(f"{CB}/products/{PRODUCT[asset]}/ticker", timeout=10).json()["price"])
    except Exception:
        return None


def _get(path, params=None):
    h = kalshi_auth.make_auth_headers(PK, KID, "GET", path) if PK else None
    return requests.get(B + path, params=params, headers=h, timeout=15)


def soonest_event(series):
    r = _get("/trade-api/v2/markets", {"series_ticker": series, "status": "open", "limit": 1000})
    if not r.ok:
        return None, []
    ms = r.json().get("markets", [])
    by = defaultdict(list)
    for m in ms:
        by[m.get("event_ticker")].append(m)
    if not by:
        return None, []
    ev = min(by, key=lambda k: by[k][0].get("close_time") or "9999")
    return ev, by[ev]


def book(ticker):
    """Return (yes_bid, yes_ask, bid_sz, ask_sz) in dollars, or Nones."""
    r = _get(f"/trade-api/v2/markets/{ticker}/orderbook")
    if not r.ok:
        return None, None, 0, 0
    ob = (r.json() or {}).get("orderbook_fp") or {}
    yes = ob.get("yes_dollars") or []     # YES bids
    no = ob.get("no_dollars") or []       # NO bids == YES asks at 1-price
    yb = ya = None; bsz = asz = 0.0
    if yes:
        lvl = max(yes, key=lambda x: float(x[0]))
        yb = float(lvl[0]); bsz = float(lvl[1])
    if no:
        lvl = max(no, key=lambda x: float(x[0]))
        ya = 1.0 - float(lvl[0]); asz = float(lvl[1])
    return yb, ya, bsz, asz


def strike_of(m):
    return m.get("floor_strike") if m.get("floor_strike") is not None else m.get("cap_strike")


def load_books(markets, sp, band):
    """Fetch books for near-money strikes; return list of dicts sorted by strike."""
    out = []
    for m in markets:
        k = strike_of(m)
        if k is None or abs(k - sp) > band:
            continue
        yb, ya, bsz, asz = book(m["ticker"])
        out.append({"ticker": m["ticker"], "type": m.get("strike_type"),
                    "floor": m.get("floor_strike"), "cap": m.get("cap_strike"),
                    "k": k, "yb": yb, "ya": ya, "bsz": bsz, "asz": asz})
        time.sleep(0.03)
    out.sort(key=lambda d: d["k"])
    return out


# ── Checks ─────────────────────────────────────────────────────────────────────
def check_monotonicity(thr):
    """A: adjacent threshold pairs where higher-strike bid >= lower-strike ask."""
    hits = []
    for i in range(len(thr)):
        for j in range(i + 1, min(i + 4, len(thr))):   # check next few strikes
            a, b = thr[i], thr[j]   # a.k < b.k
            if a["ya"] is None or b["yb"] is None:
                continue
            credit = (b["yb"] - a["ya"]) * 100.0           # cents per pair
            f = fee_c(a["ya"]) + fee_c(b["yb"])
            net = credit - f
            if net > 0:
                sz = min(a["asz"], b["bsz"])
                hits.append(("A-monotonic", f"buy>{a['k']:.0f}@{a['ya']*100:.0f} / sell>{b['k']:.0f}@{b['yb']*100:.0f}",
                             net, sz))
    return hits


def check_bucket_sum(rng):
    """B: mutually-exclusive buckets' YES asks sum < 100 (buy-all) or bids > 100 (sell-all)."""
    hits = []
    have = [d for d in rng if d["ya"] is not None]
    if len(have) < 3:
        return hits
    ask_sum = sum(d["ya"] for d in have) * 100.0
    fee_sum = sum(fee_c(d["ya"]) for d in have)
    # NOTE: 'have' is only near-money buckets; full exhaustiveness needs the tails
    # (which are ~0/1). We report the near-money partial sum as a sanity flag only
    # when the covered region already exceeds 100 / falls short with tail-prob est.
    bidv = [d for d in rng if d["yb"] is not None]
    bid_sum = sum(d["yb"] for d in bidv) * 100.0
    return {"n_ask": len(have), "ask_sum_c": round(ask_sum, 1), "ask_fees_c": round(fee_sum, 1),
            "n_bid": len(bidv), "bid_sum_c": round(bid_sum, 1)}


def check_cross_ladder(thr, rng):
    """C: RANGE between[a,b] vs THRESHOLD synthetic [YES(>a-eps) - YES(>b)]."""
    hits = []
    tb = {round(d["k"]): d for d in thr}    # threshold by rounded strike
    for d in rng:
        if d["type"] != "between" or d["floor"] is None or d["cap"] is None:
            continue
        lo = round(d["floor"]) - 1            # KXBTCD strike just below floor (x99.99 -> floor-1 ~ matches)
        hi = round(d["cap"])                  # cap is x99.99
        tl = tb.get(round(d["floor"]) - 1) or tb.get(round(d["floor"]))
        th = tb.get(round(d["cap"])) or tb.get(round(d["cap"]) + 1)
        if not tl or not th:
            continue
        if None in (d["ya"], d["yb"], tl["ya"], tl["yb"], th["ya"], th["yb"]):
            continue
        # synthetic long (buy spread on threshold): pay ask(lo) - bid(hi)
        synth_buy = (tl["ya"] - th["yb"]) * 100.0
        synth_sell = (tl["yb"] - th["ya"]) * 100.0
        between_ask = d["ya"] * 100.0
        between_bid = d["yb"] * 100.0
        fees = fee_c(d["ya"]) + fee_c(tl["ya"]) + fee_c(th["yb"])
        # buy between on RANGE, sell synthetic on THRESHOLD
        edge1 = synth_sell - between_ask - fees   # >0 => buy range bucket, sell threshold spread
        edge2 = between_bid - synth_buy - fees     # >0 => sell range bucket, buy threshold spread
        if edge1 > 0:
            hits.append(("C-cross", f"buy between[{d['floor']:.0f},{d['cap']:.0f}]@{between_ask:.0f} vs synth sell {synth_sell:.0f}", edge1, min(d["asz"], tl["bsz"], th["asz"])))
        if edge2 > 0:
            hits.append(("C-cross", f"sell between[{d['floor']:.0f},{d['cap']:.0f}]@{between_bid:.0f} vs synth buy {synth_buy:.0f}", edge2, min(d["bsz"], tl["asz"], th["bsz"])))
    return hits


def check_model_edge(thr, asset, mins_left):
    """D: threshold strike mid vs v3 fair (not locked; needs neighbor hedge)."""
    hits = []
    for d in thr:
        if d["yb"] is None or d["ya"] is None:
            continue
        mid = (d["yb"] + d["ya"]) / 2.0
        sp = d.get("_spot")
        fair = v3.fair_p(sp, mins_left, asset, floor_strike=d["k"], strike_type="greater")
        half = (d["ya"] - d["yb"]) / 2.0
        edge = (fair - d["ya"]) if fair > mid else (d["yb"] - fair)   # vs the takeable side
        if edge * 100 > max(2.0, half * 100):   # beats half-spread and >2c
            side = "buy" if fair > mid else "sell"
            hits.append(("D-model", f"{side} >{d['k']:.0f} mid={mid*100:.0f} fair={fair*100:.0f}", edge * 100, min(d["asz"], d["bsz"])))
    return hits


def scan(assets, band, min_edge, do_model, log_fh=None):
    now = datetime.now(timezone.utc)
    for asset in assets:
        sp = spot(asset)
        if sp is None:
            continue
        all_hits = []
        # THRESHOLD ladder
        if asset in THRESH:
            ev, mk = soonest_event(THRESH[asset])
            if mk:
                close = mk[0].get("close_time")
                ml = max(0.1, (datetime.fromisoformat(close.replace("Z", "+00:00")) - now).total_seconds() / 60.0)
                thr = load_books(mk, sp, band)
                for d in thr:
                    d["_spot"] = sp
                all_hits += check_monotonicity(thr)
                if do_model:
                    all_hits += check_model_edge(thr, asset, ml)
                # RANGE ladder + cross
                if asset in RANGE:
                    evr, mkr = soonest_event(RANGE[asset])
                    rng = load_books(mkr, sp, band) if mkr else []
                    bs = check_bucket_sum(rng) if rng else None
                    all_hits += check_cross_ladder(thr, rng)
                    if bs:
                        print(f"  [{asset}] range near-money buckets: n={bs['n_ask']} ask_sum={bs['ask_sum_c']}c "
                              f"(+fees {bs['ask_fees_c']}c)  bid_sum={bs['bid_sum_c']}c  ttc={ml:.0f}m")
        # report
        keep = [h for h in all_hits if h[2] >= min_edge]
        print(f"[{now.strftime('%H:%M:%S')}] {asset} spot={sp:.1f}  hits>{min_edge}c: {len(keep)}/{len(all_hits)}")
        for typ, desc, net, sz in sorted(keep, key=lambda x: -x[2])[:12]:
            print(f"    {typ:<12} edge={net:+5.1f}c size={sz:>7.0f}  {desc}")
            if log_fh:
                log_fh.writerow([round(time.time(), 1), asset, typ, round(net, 2), round(sz, 1), desc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH")
    ap.add_argument("--band", type=float, default=2500, help="strike window around spot ($)")
    ap.add_argument("--min-edge", type=float, default=0.5, help="min net edge cents to report")
    ap.add_argument("--model", action="store_true", help="also flag v3 model edges (not locked)")
    ap.add_argument("--loop", type=float, default=0, help="scan every N seconds (0=once)")
    args = ap.parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    fh = None
    if args.loop:
        f = open(LOG, "a", newline="")
        fh = csv.writer(f)
        if os.stat(LOG).st_size == 0:
            fh.writerow(["ts", "asset", "type", "edge_c", "size", "desc"])
    while True:
        try:
            scan(assets, args.band, args.min_edge, args.model, fh)
        except Exception as e:
            print("scan err:", e)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
