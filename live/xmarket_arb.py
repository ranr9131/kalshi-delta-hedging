"""
Cross-venue sports arb EXECUTOR (Kalshi <-> Polymarket).

Pipeline:  detect lopsided cross-venue lock  ->  risk gate  ->  leg KALSHI
first (IOC, known fill instantly)  ->  hedge exactly that fill on POLYMARKET
(FAK)  ->  log + reconcile.

Why Kalshi first: it's the stale/thin/uncertain side. IOC tells us the real
fill immediately; we then commit the liquid Poly hedge for exactly what filled,
so we never over-hedge a leg that didn't go.

MODES
  paper (default) : simulate both legs at quoted prices. No keys, no orders.
                    Run this freely to watch the strategy + measure flow.
  --live --yes    : REAL orders. You run this. Needs POLY_PRIVATE_KEY in .env
                    (unless --kalshi-only). Hard caps below still apply.
  --kalshi-only   : real Kalshi leg, NO auto Poly hedge (prints it to do by hand)
  --check         : verify balances/connectivity on both venues, trade nothing

SAFETY (hard, not bypassable by flags)
  * MAX_CONTRACTS ceiling per trade
  * MAX_OPEN_EXPOSURE_USD total
  * kill switch: create file ~/STOP_ARB to halt immediately
  * re-checks live quotes at execution time (never acts on stale detection)
  * per-game cooldown + dedup so it can't hammer one market
This module never runs itself; an agent will not invoke --live.
"""
from __future__ import annotations
import argparse, math, os, time, csv, sys, requests
from dotenv import dotenv_values

import xmarket_logger as X
import kalshi_auth, kalshi_trade
import xmarket_poly_exec

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT, ".env")
TRADES_CSV = os.path.join(ROOT, "xmarket_trades.csv")
KILL_SWITCH = os.path.expanduser("~/STOP_ARB")
KALSHI_BASE = "https://api.elections.kalshi.com"

# ── hard safety caps ─────────────────────────────────────────────────────────
MAX_CONTRACTS = 25
MAX_OPEN_EXPOSURE_USD = 250.0
MIN_FAVORITE_POLY = 0.60      # only buy a leg Poly treats as a real favourite
DEFAULT_MIN_EDGE = 0.05       # net cross edge after Kalshi fee
HEDGE_SLIP_TOL = 0.04         # abort if Poly hedge price drifts > this at exec
PER_GAME_COOLDOWN_S = 60


def kfee(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0 if p else 0.0


def kalshi_market(ticker):
    r = requests.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}", timeout=8)
    return r.json().get("market", {}) if r.ok else {}


# ── detection ────────────────────────────────────────────────────────────────
def detect(min_edge, want_game=None):
    """Return list of opportunities. Each: buy FAVOURITE on Kalshi (stale low),
    hedge the OTHER outcome on Polymarket. 2-way games only (MLB/NBA/NHL/WNBA)."""
    out = []
    for lg, cfg in X.LEAGUES.items():
        if cfg["draw"]:
            continue  # 3-way soccer: detect-only elsewhere; not auto-executed
        for e in X._espn_scoreboard_any(cfg["espn"]):
            g = X._parse_espn_game(e)
            if not g:
                continue
            lab = f"{g['away_abbr']}@{g['home_abbr']}"
            if want_game and lab != want_game:
                continue
            outs = X._build_outcomes(g, False)
            X._match_kalshi(cfg["kalshi"], g, outs)
            X._match_poly(g, outs)
            q = []
            for o in outs:
                kb, ka, ksz = X._kalshi_quote(o["k"]) if o["k"] else (None, None, None)
                pb, pa, psz = X._poly_book_quote(o["p"]) if o["p"] else (None, None, None)
                q.append(dict(o=o, ka=ka, ksz=ksz, pa=pa, psz=psz))
            if len(q) != 2:
                continue
            for i in (0, 1):
                fav, hed = q[i], q[1 - i]
                if None in (fav["ka"], fav["pa"], hed["pa"]):
                    continue
                if fav["pa"] < MIN_FAVORITE_POLY or fav["ka"] >= fav["pa"]:
                    continue
                lock = fav["ka"] + kfee(fav["ka"]) + hed["pa"]
                net = 1.0 - lock
                if net <= min_edge:
                    continue
                out.append(dict(
                    league=lg, game=lab, score=f"{g['away_score']}-{g['home_score']}",
                    status=g["status"],
                    fav_label=fav["o"]["label"], fav_ticker=fav["o"]["k"],
                    k_ask=fav["ka"], k_depth=fav["ksz"] or 0,
                    poly_fav=fav["pa"],
                    hedge_label=hed["o"]["label"], hedge_token=hed["o"]["p"],
                    hedge_ask=hed["pa"], hedge_depth=hed["psz"] or 0,
                    net=net, lock=lock,
                ))
    out.sort(key=lambda c: -c["net"])
    return out


def risk_ok(c, size, open_exposure):
    if os.path.exists(KILL_SWITCH):
        return False, "KILL SWITCH active (~/STOP_ARB exists)"
    if size > MAX_CONTRACTS:
        return False, f"size {size} > MAX_CONTRACTS {MAX_CONTRACTS}"
    if c["k_depth"] < size:
        return False, f"Kalshi depth {c['k_depth']} < size {size}"
    if c["hedge_depth"] < size:
        return False, f"Poly hedge depth {c['hedge_depth']} < size {size}"
    cost = size * (c["k_ask"] + c["hedge_ask"])
    if open_exposure + cost > MAX_OPEN_EXPOSURE_USD:
        return False, f"exposure ${open_exposure+cost:.0f} > cap ${MAX_OPEN_EXPOSURE_USD:.0f}"
    return True, "ok"


def log_trade(row):
    new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts", "mode", "league", "game", "score", "fav", "size",
            "k_ask", "k_filled", "hedge", "hedge_ask", "p_filled",
            "lock_cost", "net_edge_c", "profit_$", "unhedged", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


# ── execution ────────────────────────────────────────────────────────────────
def execute(c, size, mode, key, kid, poly):
    """Kalshi IOC favourite first, then Poly FAK hedge for the filled amount."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    # re-fetch Kalshi market fresh (never act on stale detection)
    mk = kalshi_market(c["fav_ticker"])
    if not mk:
        return {"ok": False, "note": "no kalshi market"}
    live_ask = float(mk.get("yes_ask_dollars") or 0)
    if live_ask <= 0 or live_ask > c["k_ask"] + HEDGE_SLIP_TOL:
        return {"ok": False, "note": f"kalshi ask moved {c['k_ask']}->{live_ask}, abort"}

    base = dict(ts=ts, mode=mode, league=c["league"], game=c["game"], score=c["score"],
                fav=c["fav_label"], size=size, k_ask=live_ask, hedge=c["hedge_label"],
                hedge_ask=c["hedge_ask"])

    # ---- KALSHI leg ----
    if mode == "paper":
        k_filled = size
    else:
        order = kalshi_trade.place_order(key, kid, c["fav_ticker"], "yes", mk,
                                         stake_dollars=size * live_ask,
                                         extra_buffer_cents=0, ioc=True)
        oid = order.get("order", {}).get("order_id") or order.get("order_id")
        time.sleep(4)
        filled_stake = kalshi_trade.get_order_filled_stake(key, kid, oid) or 0.0
        k_filled = round(filled_stake / live_ask) if live_ask else 0
        if k_filled == 0:
            row = {**base, "k_filled": 0, "p_filled": 0, "lock_cost": "", "net_edge_c": "",
                   "profit_$": 0, "unhedged": 0, "note": "KALSHI DID NOT FILL (phantom ask)"}
            log_trade(row)
            return {"ok": True, "k_filled": 0, "note": "kalshi unfilled"}

    # ---- POLY hedge for exactly what filled on Kalshi ----
    if mode == "kalshi-only":
        p_filled = 0
        note = f"MANUAL HEDGE: buy {c['hedge_label']} ~{c['hedge_ask']:.2f} x{k_filled} on Poly"
    else:
        res = poly.buy(c["hedge_token"], size=k_filled, max_price=c["hedge_ask"] + HEDGE_SLIP_TOL)
        p_filled = res.get("filled_size", 0)
        note = "hedged" if p_filled >= k_filled - 0.5 else f"UNDERHEDGED {p_filled}/{k_filled}"

    lock = live_ask + c["hedge_ask"]
    profit = k_filled * (1.0 - lock) if (mode != "kalshi-only") else None
    unhedged = max(0, k_filled - (p_filled or 0)) if mode != "kalshi-only" else k_filled
    row = {**base, "k_filled": k_filled, "p_filled": p_filled,
           "lock_cost": round(lock, 4), "net_edge_c": round((1 - lock) * 100, 2),
           "profit_$": round(profit, 2) if profit is not None else "",
           "unhedged": unhedged, "note": note}
    log_trade(row)
    return {"ok": True, "k_filled": k_filled, "p_filled": p_filled, "note": note}


def fmt(c, size):
    return (f"{c['league']} {c['game']} ({c['score']})  net {c['net']*100:.1f}c | "
            f"BUY {c['fav_label'][:14]} K@{c['k_ask']:.2f}(d{c['k_depth']:.0f}) x{size} | "
            f"HEDGE {c['hedge_label'][:14]} P@{c['hedge_ask']:.2f}(d{c['hedge_depth']:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm --live")
    ap.add_argument("--kalshi-only", action="store_true", help="real Kalshi, manual Poly hedge")
    ap.add_argument("--check", action="store_true", help="verify both venues, trade nothing")
    ap.add_argument("--paper", action="store_true", help="force paper (default if not --live)")
    ap.add_argument("--contracts", type=int, default=5)
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    ap.add_argument("--game", default=None)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    size = max(1, min(MAX_CONTRACTS, a.contracts))
    live = a.live and not a.paper
    if live and not a.yes:
        print("--live requires --yes. Aborting."); sys.exit(1)
    mode = "kalshi-only" if (live and a.kalshi_only) else ("live" if live else "paper")

    env = dotenv_values(ENV_PATH)
    key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
    kid = env["KALSHI_API_KEY_ID"]
    poly = None
    need_poly = (mode == "live") or a.check
    if need_poly:
        try:
            poly = xmarket_poly_exec.make(paper=False, env_path=ENV_PATH)
        except Exception as e:
            print(f"Polymarket client init failed: {e}")
            if mode == "live":
                sys.exit(1)
    if mode == "paper":
        poly = xmarket_poly_exec.make(paper=True)

    if a.check:
        print("KALSHI balance: $", kalshi_trade.get_balance(key, kid))
        print("POLY usdc:      $", poly.usdc_balance() if poly else "n/a")
        c = detect(a.min_edge, a.game)
        print(f"opportunities now: {len(c)}")
        if c:
            print("  best:", fmt(c[0], size))
        return

    print(f"=== xmarket arb executor — MODE={mode}  size={size}  min_edge={a.min_edge*100:.0f}c "
          f"game={a.game or 'any'} ===")
    if mode != "paper":
        print(f"!! LIVE EXECUTION. caps: <= {MAX_CONTRACTS} contracts, "
              f"<= ${MAX_OPEN_EXPOSURE_USD} exposure. kill: touch {KILL_SWITCH}")
    exposure = 0.0
    cooldown = {}
    while True:
        if os.path.exists(KILL_SWITCH):
            print("KILL SWITCH active — stopping."); break
        cs = detect(a.min_edge, a.game)
        now = time.time()
        acted = False
        for c in cs:
            if now - cooldown.get(c["game"], 0) < PER_GAME_COOLDOWN_S:
                continue
            ok, why = risk_ok(c, size, exposure)
            if not ok:
                print(f"  skip {c['game']}: {why}")
                continue
            print(f"  >>> {fmt(c, size)}")
            r = execute(c, size, mode, key, kid, poly)
            cooldown[c["game"]] = now
            if mode != "paper" and r.get("k_filled"):
                exposure += size * (c["k_ask"] + c["hedge_ask"])
            print(f"      result: {r.get('note')}  (k_filled={r.get('k_filled')})")
            acted = True
            break  # one trade per loop pass
        if not acted:
            print(f"  {time.strftime('%H:%M:%S')} no actionable opportunity "
                  f"(have {len(cs)} raw, blocked by risk/cooldown)")
        if a.once:
            break
        time.sleep(8)


if __name__ == "__main__":
    main()
