"""
Cross-market arb FILL TEST harness.

Question it answers: when Kalshi shows an under-priced favorite (stale ask far
below Polymarket's price), does that Kalshi ask ACTUALLY FILL when you hit it,
or is it phantom/adverse? That single fact decides whether the whole
cross-venue edge is real money or a paper mirage.

Scope: this tests the KALSHI leg only (the unknown). Polymarket is the deep,
sharp, liquid side — hedging there is reliable and is NOT automated here
(no CLOB client / wallet configured). The harness prints the exact Poly hedge
for you to place manually.

SAFETY:
  * DRY-RUN by default — detects opportunities and PRINTS the exact planned
    Kalshi order + Poly hedge. Places NOTHING.
  * --live actually sends ONE small Kalshi IOC order, then exits. Hard size
    caps below. Intended to be run BY YOU, not by an agent.
  * Buys the under-priced FAVORITE: even unhedged, a few contracts of a
    true-~85% team bought at ~0.49 is +EV, so the downside of the test itself
    is a few dollars.

Usage:
  python3 xmarket_filltest.py                 # dry-run monitor (safe)
  python3 xmarket_filltest.py --live --yes    # place ONE real test order (you)
  optional: --contracts N  --min-edge 0.08  --game WSH@SF
"""
from __future__ import annotations

import argparse, time, sys, requests
from dotenv import dotenv_values

import xmarket_logger as X
import kalshi_auth
import kalshi_trade

# ---- hard safety caps ----
MAX_CONTRACTS = 20            # absolute ceiling regardless of flags
DEFAULT_CONTRACTS = 5
MIN_FAVORITE_POLY = 0.62      # only buy a leg Polymarket considers a real favourite
DEFAULT_MIN_EDGE = 0.08       # net cross edge (after Kalshi fee) to act on
KALSHI_BASE = "https://api.elections.kalshi.com"


def kfee(p):
    import math
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0 if p else 0.0


def kalshi_market(ticker):
    r = requests.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}", timeout=8)
    return r.json().get("market", {}) if r.ok else {}


def find_opportunity(min_edge, want_game=None):
    """Scan live games; return the best under-priced-favorite-on-Kalshi lock.
    Returns dict with both legs, or None."""
    best = None
    for lg, cfg in X.LEAGUES.items():
        for e in X._espn_scoreboard_any(cfg["espn"]):
            g = X._parse_espn_game(e)
            if not g:
                continue
            lab = f"{g['away_abbr']}@{g['home_abbr']}"
            if want_game and lab != want_game:
                continue
            outs = X._build_outcomes(g, cfg["draw"])
            X._match_kalshi(cfg["kalshi"], g, outs)
            X._match_poly(g, outs)
            quotes = []
            for o in outs:
                kb, ka, ksz = X._kalshi_quote(o["k"]) if o["k"] else (None, None, None)
                pb, pa, psz = X._poly_book_quote(o["p"]) if o["p"] else (None, None, None)
                quotes.append(dict(o=o, ka=ka, ksz=ksz, pa=pa, psz=psz))
            if len(quotes) < 2:
                continue
            # favourite leg = the one Polymarket prices high; buy it on Kalshi if cheaper
            for i, q in enumerate(quotes):
                other = quotes[1 - i] if len(quotes) == 2 else None
                if not other or q["ka"] is None or q["pa"] is None or other["pa"] is None:
                    continue
                if q["pa"] < MIN_FAVORITE_POLY:      # not a real favourite on Poly
                    continue
                if q["ka"] >= q["pa"]:               # Kalshi not cheaper -> no edge
                    continue
                # lock: buy favourite on Kalshi, hedge underdog on Poly
                lock = (q["ka"] + kfee(q["ka"])) + other["pa"]
                net = 1.0 - lock
                if net <= min_edge:
                    continue
                cand = dict(
                    league=lg, game=lab, score=f"{g['away_score']}-{g['home_score']}",
                    status=g["status"],
                    fav_label=q["o"]["label"], fav_ticker=q["o"]["k"],
                    k_ask=q["ka"], k_depth=q["ksz"], poly_fav=q["pa"],
                    hedge_label=other["o"]["label"], hedge_token=other["o"]["p"],
                    hedge_poly_ask=other["pa"], hedge_depth=other["psz"],
                    net=net, lock=lock,
                )
                if best is None or cand["net"] > best["net"]:
                    best = cand
    return best


def show(c, contracts):
    cost_k = c["k_ask"] * contracts
    cost_p = c["hedge_poly_ask"] * contracts
    print("\n" + "=" * 64)
    print(f"OPPORTUNITY  {c['league']} {c['game']}  ({c['score']}, {c['status']})")
    print(f"  net edge after Kalshi fee: {c['net']*100:.1f}c   lock cost {c['lock']:.3f}")
    print(f"  Kalshi depth at ask: {c['k_depth']}   Poly hedge depth: {c['hedge_depth']}")
    print(f"\n  LEG 1  KALSHI  BUY YES  {c['fav_label']}")
    print(f"         ticker {c['fav_ticker']}")
    print(f"         limit ~{round(c['k_ask']*100)}c (IOC)  x {contracts} contracts  = ${cost_k:.2f}")
    print(f"         (Polymarket prices this team at {c['poly_fav']:.2f} -> Kalshi is stale low)")
    print(f"\n  LEG 2  POLYMARKET  BUY  {c['hedge_label']}   <-- DO THIS MANUALLY")
    print(f"         token {c['hedge_token']}")
    print(f"         ~{c['hedge_poly_ask']:.2f}  x {contracts}  = ${cost_p:.2f}")
    print(f"\n  If both fill: pay ${cost_k+cost_p:.2f} for {contracts} guaranteed "
          f"$1 payouts -> profit ${contracts - (cost_k+cost_p):.2f}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually place ONE Kalshi order")
    ap.add_argument("--yes", action="store_true", help="required confirm flag for --live")
    ap.add_argument("--contracts", type=int, default=DEFAULT_CONTRACTS)
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    ap.add_argument("--game", default=None, help="restrict to one game e.g. WSH@SF")
    ap.add_argument("--once", action="store_true", help="check once and exit (dry-run)")
    a = ap.parse_args()
    contracts = max(1, min(MAX_CONTRACTS, a.contracts))

    env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")
    key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    kid = env["KALSHI_API_KEY_ID"]

    if a.live:
        if not a.yes:
            print("--live requires --yes to confirm. Aborting."); sys.exit(1)
        bal = kalshi_trade.get_balance(key, kid)
        print(f"[LIVE] Kalshi balance: ${bal}")
        c = find_opportunity(a.min_edge, a.game)
        if not c:
            print("No qualifying opportunity right now. Try again during a live lopsided game.")
            sys.exit(0)
        show(c, contracts)
        print(f"\n[LIVE] placing Kalshi IOC BUY YES {c['fav_label']} x{contracts} ...")
        mk = kalshi_market(c["fav_ticker"])
        if not mk:
            print("could not fetch market; abort"); sys.exit(1)
        order = kalshi_trade.place_order(key, kid, c["fav_ticker"], "yes", mk,
                                         stake_dollars=contracts * c["k_ask"],
                                         extra_buffer_cents=0, ioc=True)
        oid = order.get("order", {}).get("order_id") or order.get("order_id")
        print(f"[LIVE] order submitted id={oid}; polling fill ...")
        time.sleep(4)
        filled = kalshi_trade.get_order_filled_stake(key, kid, oid)
        st = kalshi_trade.get_order_status(key, kid, oid)
        print(f"[LIVE] RESULT: filled_stake=${filled}  status={st.get('order',{}).get('status')}")
        print(f"  -> Did it fill? filled_stake>0 means the stale Kalshi ask is REAL & hittable.")
        print(f"  -> Now hedge manually on Polymarket: BUY {c['hedge_label']} "
              f"~{c['hedge_poly_ask']:.2f} x{contracts}")
        return

    # dry-run monitor
    print(f"DRY-RUN monitor (no orders). contracts={contracts} min_edge={a.min_edge*100:.0f}c "
          f"game={a.game or 'any'}")
    while True:
        c = find_opportunity(a.min_edge, a.game)
        if c:
            show(c, contracts)
            print("  [dry-run] to test for real: python3 xmarket_filltest.py --live --yes "
                  f"--game {c['game']} --contracts {contracts}")
        else:
            print(f"  {time.strftime('%H:%M:%S')} no qualifying opportunity "
                  f"(need a live lopsided game, edge>{a.min_edge*100:.0f}c)")
        if a.once:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
