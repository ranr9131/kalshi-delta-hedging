"""
Forensic A/B: for one live game, compare Kalshi WS book vs Kalshi REST vs
Polymarket /book at the same instant. Reveals whether a flagged cross-venue
"edge" is a real two-sided market state or a data-collection artifact
(stale WS book, non-simultaneous quotes, wrong token, empty side, etc.).
"""
import sys, time, requests
from dotenv import dotenv_values
import xmarket_logger as x
import kalshi_auth, kalshi_orderbook as ob

LABEL = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. "WSH@SF"

# 1) find the live game
game = None
for lg, cfg in x.LEAGUES.items():
    for e in x._espn_scoreboard_any(cfg["espn"]):
        g = x._parse_espn_game(e)
        if not g:
            continue
        lab = f"{g['away_abbr']}@{g['home_abbr']}"
        if LABEL is None or lab == LABEL:
            game, league, gcfg = g, lg, cfg
            break
    if game:
        break
if not game:
    print("no matching live game"); sys.exit(1)

print(f"GAME {league} {game['away_name']} @ {game['home_name']}  {game['status']}  "
      f"score {game['away_score']}-{game['home_score']}")

# 2) match both venues
outs = x._build_outcomes(game, gcfg["draw"])
x._match_kalshi(gcfg["kalshi"], game, outs)
slug = x._match_poly(game, outs)
print("poly_slug:", slug)
for o in outs:
    print(f"  {o['role']:5} {o['label'][:20]:20} k={o['k']}  p={(o['p'] or '')[:14]}")

# 3) start a FRESH WS book for these tickers
env = dotenv_values(".env")
key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
ob.start(key, env["KALSHI_API_KEY_ID"], [o["k"] for o in outs if o["k"]])
time.sleep(4)

def rest_quote(t):
    try:
        r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}", timeout=6)
        m = r.json().get("market", {})
        return m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    except Exception as ex:
        return None, ex

def poly_raw(tok):
    try:
        b = requests.get("https://clob.polymarket.com/book", params={"token_id": tok}, timeout=6).json()
        asks = b.get("asks") or []; bids = b.get("bids") or []
        return (bids[-1] if bids else None), (asks[-1] if asks else None), len(asks), b.get("timestamp")
    except Exception as ex:
        return None, ex, 0, None

# 4) three synchronized snapshots
for it in range(3):
    print(f"\n===== snapshot {it+1}  t={time.strftime('%H:%M:%S')} =====")
    for o in outs:
        print(f"-- {o['role']} {o['label'][:18]} --")
        if o["k"]:
            b = ob.get_book(o["k"])
            if b and b.snapshot_seen:
                wa = b.yes_asks_sorted()[:2]; wb = b.yes_bids_sorted()[:2]
                print(f"   Kalshi WS  : yes_bid={b.yes_bid()} yes_ask={b.yes_ask()} "
                      f"age={b.age():.1f}s  ask_ladder={wa} bid_ladder={wb}")
            else:
                print("   Kalshi WS  : (no snapshot yet)")
            rb, ra = rest_quote(o["k"])
            print(f"   Kalshi REST: yes_bid={rb} yes_ask={ra}")
        if o["p"]:
            pb, pa, na, pts = poly_raw(o["p"])
            print(f"   Poly /book : best_bid={pb} best_ask={pa} n_asks={na} ts={pts}")
    # internal consistency: sum of yes_asks across the 2 teams on each venue
    ks = [ob.get_book(o["k"]).yes_ask() if (o["k"] and ob.get_book(o["k"]) and ob.get_book(o["k"]).snapshot_seen) else None for o in outs[:2]]
    if all(v is not None for v in ks):
        print(f"   >> Kalshi WS yes_ask(home)+yes_ask(away) = {sum(ks):.3f}  (real book should be >~1.0)")
    time.sleep(2)
