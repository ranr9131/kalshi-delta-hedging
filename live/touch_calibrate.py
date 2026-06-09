"""
Touch-market volatility calibration (Phase 1 of the sharky6999 replication).

THE PROBLEM Phase-0 surfaced: pricing these weeks-long touch markets with the
sniper's minute-scale σ (× √minutes) over-states dispersion, so the model
thinks 'No' is overpriced everywhere and finds no edge. The σ dial is wrong for
the horizon.

THE FIX here: let the MARKET set its own volatility. For each (asset, expiry):
  1. Pull every strike's live book (via touch_shadow_logger.discover/get_no_book).
  2. The market's 'Yes' mid = its implied touch probability for that strike.
  3. Pick a NEAR-THE-MONEY anchor strike (most liquid / most efficient) and
     invert the touch model for the σ the market is implying there
     (implied_sigma_from_touch).
  4. Re-price ALL strikes of that expiry with the anchor σ.
  5. Strikes where market 'No' is cheaper than the anchor-σ model 'No' are the
     edge: the far tail is priced as if vol were even higher than ATM — i.e.
     the longshot 'Yes' is overpriced, exactly what sharky harvests.

This anchors σ to the efficient part of the curve and reads the EDGE off the
inefficient tail — instead of trusting an absolute vol guess.

Run:
  python3 touch_calibrate.py                 # all assets/expiries
  python3 touch_calibrate.py --asset BTC
  python3 touch_calibrate.py --min-edge 2    # only show strikes with >=2c edge
"""
from __future__ import annotations

import sys
import os
import argparse
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from touch_shadow_logger import discover, get_no_book, COIN_PRODUCT
from coinbase_feeds import make_feed
from fair_price_model_v2 import (
    p_touch_v2, fair_p_no_touch_v2, touch_moneyness_z, implied_sigma_from_touch,
    effective_sigma_per_min, FALLBACK_SIGMA_PER_MIN,
)


def minutes_left(end_dt):
    return max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 60.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None, help="restrict to one asset, e.g. BTC")
    ap.add_argument("--min-edge", type=float, default=0.0, help="only print strikes with >= this edge (cents)")
    ap.add_argument("--anchor-target", type=float, default=0.35,
                    help="aim the σ-anchor at the strike whose market touch-prob is nearest this")
    args = ap.parse_args()

    print("discovering markets + fetching books (live)…")
    markets = discover()

    # spot feeds
    assets = {m["asset"] for m in markets.values()}
    if args.asset:
        assets = {args.asset.upper()}
    feeds = {}
    for a in assets:
        if a in COIN_PRODUCT:
            f = make_feed(COIN_PRODUCT[a]); f.start(); feeds[a] = f
    import time as _t
    _t.sleep(4)  # warm feeds

    # group strikes by (asset, expiry-day, direction)
    groups = defaultdict(list)
    for tok, m in markets.items():
        if args.asset and m["asset"] != args.asset.upper():
            continue
        book = get_no_book(tok)
        if not book:
            continue
        no_bid_c, no_ask_c, no_bid_sz = book
        no_mid = (no_bid_c + no_ask_c) / 2.0
        yes_mid = 1.0 - no_mid / 100.0          # market-implied touch prob
        spot = feeds[m["asset"]].get_price() if m["asset"] in feeds else None
        if spot is None:
            continue
        ml = minutes_left(m["end_dt"])
        if ml <= 0:
            continue
        key = (m["asset"], m["end_dt"].date().isoformat(), m["direction"])
        groups[key].append({
            "strike": m["strike"], "spot": spot, "ml": ml, "slug": m["slug"],
            "no_bid_c": no_bid_c, "no_ask_c": no_ask_c, "no_bid_sz": no_bid_sz,
            "yes_mid": yes_mid, "z": touch_moneyness_z(spot, m["strike"], ml, m["asset"]),
        })

    for key in sorted(groups):
        asset, expiry, direction = key
        legs = sorted(groups[key], key=lambda x: x["strike"])
        if len(legs) < 3:
            continue   # need a few strikes to anchor + compare

        # anchor = strike whose market touch-prob is nearest anchor-target and
        # is genuinely two-sided (not a degenerate 0/100 tail).
        cand = [L for L in legs if 0.03 < L["yes_mid"] < 0.97]
        if not cand:
            continue
        anchor = min(cand, key=lambda L: abs(L["yes_mid"] - args.anchor_target))
        sig = implied_sigma_from_touch(anchor["yes_mid"], anchor["spot"],
                                       anchor["strike"], anchor["ml"])
        fb = FALLBACK_SIGMA_PER_MIN.get(asset, 0.0015)
        if sig is None:
            continue

        print(f"\n=== {asset} {direction.upper()}  expiry {expiry}  "
              f"spot={anchor['spot']:.2f} ===")
        print(f"anchor strike {anchor['strike']:,.4g} (mkt touch={anchor['yes_mid']:.3f}) "
              f"→ implied σ/min={sig:.5f}   vs sniper-fallback σ={fb:.5f} "
              f"({sig/fb:.2f}x)")
        print(f"{'strike':>12} {'z':>5} {'mktNo':>6} {'modelNo':>7} {'edge¢':>6} {'bidsz':>8}")
        print("-" * 56)
        for L in legs:
            model_no = fair_p_no_touch_v2(L["spot"], L["strike"], L["ml"],
                                          asset, sigma_per_min=sig) * 100.0
            edge = model_no - L["no_ask_c"]    # buy No at ask vs model
            if edge < args.min_edge:
                continue
            flag = "  <<<" if (edge >= 2 and L["z"] >= 2) else ""
            print(f"{L['strike']:>12,.4g} {L['z']:>5.2f} {L['no_ask_c']:>6.1f} "
                  f"{model_no:>7.1f} {edge:>+6.1f} {L['no_bid_sz']:>8,.0f}{flag}")

    print("\nLegend: edge = anchor-σ model 'No' − market 'No' ask.  Positive = the")
    print("tail is pricing higher vol than ATM (overpriced longshot) = buy 'No'.")
    print("'<<<' = passes the z>=2 + edge>=2c gate. These are the sharky-style trades.")


if __name__ == "__main__":
    main()
