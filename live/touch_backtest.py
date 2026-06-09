"""
Historical settlement backtest for the touch 'buy-No' strategy.

The live shadow logger can't show settled P&L for ~3 weeks (the open touch
markets all resolve June 30+). This replays the SAME model + gate against touch
markets that have ALREADY resolved, so we get a real win-rate / P&L verdict
TODAY — the proof that the +5¢ markout actually turns into money at settlement.

Method, per resolved market (strike K, direction, expiry E, asset A, winner R):
  1. Pull the 'No' token price history (clob /prices-history).
  2. Pull spot history for A (Coinbase candles); estimate a horizon-appropriate
     realized σ from it (hourly log-returns → per-minute) — the proper vol for
     multi-day touches, not the minute-σ×√T that over-disperses.
  3. Walk forward hour by hour. At each t compute z, model-No, and the market's
     No price. The FIRST hour the gate fires (Z_MIN≤z≤Z_MAX and
     model_No − market_No ≥ EDGE), we 'buy No' at the market price.
  4. Settle: a buy-No pays (100 − entry) if No won (R=='No'), else −entry.

This tests the real thesis — that the market over-prices the longshot 'Yes', so
buying 'No' when the model says it's cheap wins at settlement. Honest knobs:
  --cost   cents subtracted from each entry (taker/half-spread realism)
  --sigma-mult  scale the realized σ to stress the one assumption that matters

Run:  python3 touch_backtest.py
      python3 touch_backtest.py --cost 1 --sigma-mult 1.3
"""
from __future__ import annotations

import sys
import os
import math
import time
import argparse
import statistics
from datetime import datetime, timezone, timedelta

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from touch_shadow_logger import parse_question, COIN_PRODUCT
from fair_price_model_v2 import p_touch_v2, touch_moneyness_z

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CB = "https://api.exchange.coinbase.com"

# Extra weekly baskets (not always surfaced by the closed-events list).
EXTRA_SLUGS = [
    "what-price-will-bitcoin-hit-june-1-7-2026",
    "what-price-will-ethereum-hit-june-1-7-2026",
    "what-price-will-solana-hit-june-1-7-2026",
    "what-price-will-xrp-hit-june-1-7-2026",
]
TAGS = ["bitcoin", "ethereum", "solana", "xrp"]


def discover_resolved_slugs():
    """All resolved 'what price will X hit ...' basket slugs across tags."""
    import re as _re
    slugs = set(EXTRA_SLUGS)
    for tag in TAGS:
        try:
            evs = _get(f"{GAMMA}/events", closed="true", limit=500, tag_slug=tag)
        except Exception:
            continue
        for e in evs:
            if _re.search(r"what price will .* hit", e.get("title", ""), _re.I):
                if e.get("slug"):
                    slugs.add(e["slug"])
    return sorted(slugs)

# Gate (same spirit as the live logger's liquid wing).
Z_MIN, Z_MAX, EDGE_C = 0.7, 1.8, 3.0


def _get(url, **params):
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    r.raise_for_status()
    return r.json()


def resolved_markets():
    """[{asset,direction,strike,no_token,won_no,start,end,question}] across all
    discovered resolved touch baskets."""
    import json as _j
    out = []
    for slug in discover_resolved_slugs():
        try:
            ev = _get(f"{GAMMA}/events", slug=slug)
        except Exception:
            continue
        if not ev:
            continue
        ev = ev[0]
        for m in ev.get("markets", []):
            if not m.get("closed"):
                continue
            parsed = parse_question(m.get("question", ""))
            if not parsed:
                continue
            asset, direction, strike = parsed
            if asset not in COIN_PRODUCT:
                continue
            try:
                toks = _j.loads(m["clobTokenIds"]); outs = _j.loads(m["outcomes"])
                opx = _j.loads(m["outcomePrices"])
                no_i = outs.index("No")
                no_token = toks[no_i]
                won_no = float(opx[no_i]) > 0.5
                end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
                start = datetime.fromisoformat(m["startDate"].replace("Z", "+00:00")) \
                    if m.get("startDate") else end - timedelta(days=7)
            except Exception:
                continue
            out.append({"asset": asset, "direction": direction, "strike": strike,
                        "no_token": no_token, "won_no": won_no, "start": start,
                        "end": end, "question": m["question"][:46]})
    return out


DAY = 86400  # daily buckets — robust across weekly..yearly horizons


def no_price_series(token, start, end):
    """{day: no_price} from clob prices-history. The startTs/endTs form 400s, so
    pull the token's FULL history (interval=max) and bucket to days in-window."""
    try:
        d = _get(f"{CLOB}/prices-history", market=token, interval="max", fidelity=60)
    except Exception:
        return {}
    hist = d.get("history", d if isinstance(d, list) else [])
    s, e = int(start.timestamp()), int(end.timestamp())
    out = {}
    for pt in hist:
        t = int(pt["t"])
        if s <= t <= e:
            out[t // DAY] = float(pt["p"])
    return out


def spot_series(asset, start, end):
    """{day: spot} from Coinbase daily candles (one call covers ≤300 days)."""
    prod = COIN_PRODUCT[asset]
    out = {}
    try:
        d = _get(f"{CB}/products/{prod}/candles", granularity=DAY,
                 start=start.isoformat(), end=end.isoformat())
    except Exception:
        return out
    for c in d if isinstance(d, list) else []:  # [time, low, high, open, close, vol]
        out[int(c[0]) // DAY] = float(c[4])
    return out


def trailing_sigma_per_min(spot_by_day, as_of_day, window_days=14):
    """σ-per-minute from DAILY log returns in (as_of_day-window, as_of_day].
    TRAILING ONLY — no look-ahead. None if too few prior samples."""
    days = [h for h in sorted(spot_by_day) if as_of_day - window_days < h <= as_of_day]
    rets = []
    for a, b in zip(days, days[1:]):
        pa, pb = spot_by_day[a], spot_by_day[b]
        if pa > 0 and pb > 0:
            rets.append(math.log(pb / pa))
    if len(rets) < 6:
        return None
    return statistics.pstdev(rets) / math.sqrt(float(DAY) / 60.0)   # daily → per-minute


def backtest(cost_c=0.0, sigma_mult=1.0, verbose=False,
             z_min=Z_MIN, z_max=Z_MAX, edge_c=EDGE_C, min_entry=0.0, hedge=False):
    mkts = resolved_markets()
    print(f"resolved touch markets: {len(mkts)}  (gate z∈[{z_min},{z_max}], "
          f"edge≥{edge_c}c, min_entry={min_entry}c, cost={cost_c}c, σ×{sigma_mult}, "
          f"hedge={'ON' if hedge else 'off'})")
    trades = []
    skipped = 0
    for m in mkts:
        time.sleep(0.12)   # throttle Coinbase/clob
        nps = no_price_series(m["no_token"], m["start"], m["end"])
        # fetch spot from 16 days BEFORE start so a 14d trailing-σ window exists at entry
        sps = spot_series(m["asset"], m["start"] - timedelta(days=16), m["end"])
        if not nps or not sps:
            skipped += 1
            continue
        spot_end = sps[max(sps)]   # last available spot ≈ settlement spot (for hedge)
        for day in sorted(set(nps) & set(sps)):
            spot = sps[day]; no_px = nps[day]
            mins_left = (m["end"].timestamp() - day * DAY) / 60.0
            if mins_left <= 1440:        # need >1 day left
                continue
            sig = trailing_sigma_per_min(sps, day)   # NO look-ahead
            if not sig:
                continue
            sig *= sigma_mult
            # direction-aware already-touched guard
            if m["direction"] == "up" and spot >= m["strike"]:
                continue
            if m["direction"] == "down" and spot <= m["strike"]:
                continue
            z = touch_moneyness_z(spot, m["strike"], mins_left, m["asset"], sigma_per_min=sig)
            model_no = (1.0 - p_touch_v2(spot, m["strike"], mins_left,
                                         sigma_per_min=sig)) * 100.0
            entry = no_px * 100.0 + cost_c
            edge = model_no - entry
            if z_min <= z <= z_max and edge >= edge_c and min_entry <= entry < 100:
                opt_pnl = (100.0 - entry) if m["won_no"] else (-entry)
                # static delta hedge held to expiry: short Δ = d(No)/d(spot) of the
                # underlying, P&L = −Δ·(spot_end − spot_entry), in contract cents.
                hedge_pnl = 0.0
                if hedge:
                    dS = 0.005 * spot
                    no_up = 1.0 - p_touch_v2(spot + dS, m["strike"], mins_left, sigma_per_min=sig)
                    no_dn = 1.0 - p_touch_v2(spot - dS, m["strike"], mins_left, sigma_per_min=sig)
                    delta = (no_up - no_dn) / (2 * dS)          # coin/contract ($-value)
                    hedge_pnl = -delta * (spot_end - spot) * 100.0  # → contract cents
                pnl = opt_pnl + hedge_pnl
                trades.append({**m, "entry": entry, "z": z, "model_no": model_no,
                               "edge": edge, "opt_pnl": opt_pnl, "hedge_pnl": hedge_pnl,
                               "pnl": pnl})
                if verbose:
                    hp = f" hedge={hedge_pnl:+.0f}" if hedge else ""
                    print(f"  BUY {m['question'][:34]:34} @{entry:.0f}c z={z:.2f} "
                          f"edge={edge:+.1f} -> {'No WON ' if m['won_no'] else 'touched'} "
                          f"opt={opt_pnl:+.0f}{hp} net={pnl:+.0f}c")
                break  # one entry per market

    print(f"\nentered {len(trades)} of {len(mkts)} markets ({skipped} skipped: no data/σ)")
    if not trades:
        print("No gated entries. Loosen gate or check data.")
        return
    wins = sum(1 for t in trades if t["opt_pnl"] > 0)
    tot = sum(t["pnl"] for t in trades)
    avg = tot / len(trades)
    # regime split + per-direction (the correlated risk lives in 'dip'/'reach')
    for d in ("up", "down"):
        sub = [t for t in trades if t["direction"] == d]
        if sub:
            print(f"  {('reach(up)' if d=='up' else 'dip(down)'):10}: {len(sub):>3} trades  "
                  f"avg net {sum(t['pnl'] for t in sub)/len(sub):+.2f}c")
    print(f"win rate (No held): {wins}/{len(trades)} = {100*wins/len(trades):.0f}%")
    if hedge:
        gross = sum(t["opt_pnl"] for t in trades) / len(trades)
        hedj = sum(t["hedge_pnl"] for t in trades) / len(trades)
        print(f"UNHEDGED option avg: {gross:+.2f}c   hedge contribution: {hedj:+.2f}c   "
              f"→ HEDGED net: {avg:+.2f}c/trade")
    print(f"avg net P&L: {avg:+.2f}c/trade   total: {tot:+.0f}c   "
          f"(${tot*0.20:+.2f} at 20 contracts/trade)")
    print(f"avg entry: {statistics.mean(t['entry'] for t in trades):.1f}c   "
          f"avg model-edge at entry: {statistics.mean(t['edge'] for t in trades):+.1f}c")
    losers = sorted((t for t in trades if t["pnl"] < 0), key=lambda t: t["pnl"])
    if losers:
        print(f"\nlosers ({len(losers)}) — barrier-touch tail:")
        for t in losers[:8]:
            print(f"  {t['pnl']:>+7.0f}c  {t['question'][:44]:44} (entry {t['entry']:.0f}c, z={t['z']:.2f})")
    print("\nBreakeven win rate at avg entry "
          f"{statistics.mean(t['entry'] for t in trades):.0f}c = "
          f"{statistics.mean(t['entry'] for t in trades):.0f}%. "
          f"Actual {100*wins/len(trades):.0f}% → "
          f"{'EDGE IS REAL' if avg>0 else 'NO EDGE / negative'}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.0, help="cents subtracted per entry (taker/spread)")
    ap.add_argument("--sigma-mult", type=float, default=1.0, help="scale realized σ (stress test)")
    ap.add_argument("--z-min", type=float, default=Z_MIN)
    ap.add_argument("--z-max", type=float, default=Z_MAX)
    ap.add_argument("--edge", type=float, default=EDGE_C)
    ap.add_argument("--min-entry", type=float, default=0.0, help="only buy No priced >= this (deep-favorite filter)")
    ap.add_argument("--hedge", action="store_true", help="static delta-hedge each position to expiry")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    backtest(args.cost, args.sigma_mult, args.verbose,
             args.z_min, args.z_max, args.edge, args.min_entry, args.hedge)


if __name__ == "__main__":
    main()
