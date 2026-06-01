"""Parameterized last-24h backtest. Set ASSET env var (sol|xrp|eth|btc)."""
import os, sys, time, json, math
sys.path.insert(0, "/root/kalshi-delta-hedging/live")
sys.path.insert(0, "/root/kalshi-delta-hedging")
ASSET = os.environ.get("ASSET", "sol").upper()
os.environ["ASSET"] = ASSET
os.environ["FAIR_PRICE_SOURCE"] = "nn"
os.environ["NN_CHECKPOINT"] = (
    "../nn/checkpoints/best_v2_small.pt"
    if ASSET == "BTC"
    else f"../nn/checkpoints/best_v2_small_{ASSET.lower()}.pt"
)
os.environ["PAPER_MODE"] = "true"
os.chdir("/root/kalshi-delta-hedging/live")

import trader
trader._load_nn()
import kalshi_client, strategy
import requests
from datetime import datetime, timezone

if ASSET == "ETH":
    import eth_data as data_mod
    fetch_prices = data_mod.fetch_eth_prices
elif ASSET == "BTC":
    import btc_data as data_mod
    fetch_prices = data_mod.fetch_btc_prices
elif ASSET == "SOL":
    import sol_data as data_mod
    fetch_prices = data_mod.fetch_sol_prices
elif ASSET == "XRP":
    import xrp_data as data_mod
    fetch_prices = data_mod.fetch_xrp_prices
else:
    sys.exit(f"Unknown asset {ASSET}")

SERIES = f"KX{ASSET}15M"
url = "https://api.elections.kalshi.com/trade-api/v2/markets"
DAYS = int(os.environ.get("DAYS", "1"))
cutoff = int(time.time()) - DAYS * 86400
print(f"Backtest range: last {DAYS} day(s)")
mkts = []
cursor = None
for page in range(500):
    params = {"series_ticker": SERIES, "status": "settled",
              "min_close_ts": cutoff, "limit": 200}
    if cursor: params["cursor"] = cursor
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    d = r.json()
    mkts.extend(d.get("markets", []))
    cursor = d.get("cursor")
    if not cursor or not d.get("markets"):
        break
    time.sleep(0.2)

print(f"Settled {ASSET} markets last 24h: {len(mkts)}")
if not mkts:
    sys.exit(0)
mkts.sort(key=lambda m: m["close_time"])

t_min = min(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
t_max = max(int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
prices = fetch_prices(t_min - 600, t_max + 60)
print(f"{ASSET} prices loaded: {len(prices)}")

BASE_STAKE = 10.0
FILL_BUFFER = 0.03
FEE = 0.07
MAX_FILL_PRICE = 0.95


def time_decay_mult(m):
    if m < 7: return 0.4
    if m < 10: return 0.8
    return 1.2


def simulate(market, invert=False, min_edge=3.0, min_price=0.0, nn_max=13):
    ot = market["open_time"]; ct = market["close_time"]
    tk = market["ticker"]; result = market["result"]
    if result not in ("yes", "no"): return []
    t0 = int(datetime.fromisoformat(ot.replace("Z", "+00:00")).timestamp())
    asset_t0 = data_mod.lookup(prices, t0)
    if asset_t0 is None: return []
    trader._V2_WINDOW_CACHE.update({"open_iso": None})
    try:
        candles = kalshi_client.fetch_candlesticks(tk, ot, ct)
    except Exception:
        return []
    if not candles: return []
    bets = []
    hour = datetime.fromisoformat(ot.replace("Z", "+00:00")).hour
    kt0_candle = next((c for c in candles if c["ts"] >= t0), None)
    if kt0_candle is None: return []
    kt0 = float(kt0_candle["yes_close"])
    if not (0.01 < kt0 < 0.99): return []
    for minute in range(10, nn_max + 1):
        t = t0 + minute * 60
        asset_now = data_mod.lookup(prices, t)
        if asset_now is None: continue
        cand_at_m = next((c for c in candles if c["ts"] >= t), None)
        if cand_at_m is None: continue
        yes_bid = float(cand_at_m.get("yes_bid_close", cand_at_m["yes_close"]))
        yes_ask = float(cand_at_m.get("yes_ask_close", cand_at_m["yes_close"]))
        if yes_bid <= 0 or yes_ask >= 1: continue
        p_yes = trader._nn_p_yes(None, asset_t0, kt0, hour, minute,
                                 ticker=tk, open_iso=ot, close_iso=ct)
        direction_up = asset_now > asset_t0
        fair = p_yes if direction_up else (1.0 - p_yes)
        abs_pct_move = abs(asset_now - asset_t0) / asset_t0 * 100
        f_btc = strategy.sigmoid_btc(abs_pct_move)
        td_mult = time_decay_mult(minute)
        if direction_up:
            cost_aligned = yes_ask + FILL_BUFFER
            mis = fair - cost_aligned
        else:
            cost_aligned = (1 - yes_bid) + FILL_BUFFER
            mis = fair - cost_aligned
        edge_cents = mis * 100
        if edge_cents < min_edge: continue
        g = strategy.sigmoid_mispricing(mis)
        target = BASE_STAKE * f_btc * g * td_mult
        side = "yes" if direction_up else "no"
        if invert:
            side = "no" if side == "yes" else "yes"
        if side == "yes":
            fill = yes_ask + FILL_BUFFER
        else:
            fill = (1 - yes_bid) + FILL_BUFFER
        if fill < min_price: continue
        if fill > MAX_FILL_PRICE: continue
        contracts = target / fill
        win = (side == "yes" and result == "yes") or (side == "no" and result == "no")
        if win:
            pnl = contracts * (1.0 - fill) * (1 - FEE)
        else:
            pnl = -target
        bets.append({"window": ot, "minute": minute, "side": side, "fair": fair,
                     "fill": fill, "stake": target, "win": win, "pnl": pnl,
                     "p_yes": p_yes, "result": result})
    return bets


def report(name, bets):
    print(f"\n=== {name} ===")
    print(f"Bets: {len(bets)}")
    if not bets: return
    wins = sum(1 for b in bets if b["win"])
    pnl = sum(b["pnl"] for b in bets)
    wager = sum(b["stake"] for b in bets)
    print(f"Wins: {wins}/{len(bets)} ({100*wins/len(bets):.1f}%)")
    print(f"Net P&L: ${pnl:+.2f}")
    print(f"Wagered: ${wager:.2f}")
    print(f"ROI: {100*pnl/wager:+.2f}%")
    fills = [b["fill"] for b in bets]
    print(f"Fills: min={min(fills):.3f} max={max(fills):.3f} med={sorted(fills)[len(fills)//2]:.3f}")
    # Day-by-day breakdown
    from collections import defaultdict
    by_day = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0, "wager": 0.0})
    for b in bets:
        day = b["window"][:10]  # "2026-05-31"
        by_day[day]["bets"] += 1
        by_day[day]["wins"] += int(b["win"])
        by_day[day]["pnl"] += b["pnl"]
        by_day[day]["wager"] += b["stake"]
    print(f"Day-by-day:")
    for day in sorted(by_day):
        d = by_day[day]
        wr = 100 * d["wins"] / d["bets"] if d["bets"] else 0
        roi = 100 * d["pnl"] / d["wager"] if d["wager"] else 0
        print(f"  {day}  bets={d['bets']:3d}  wr={wr:5.1f}%  wagered=${d['wager']:7.2f}  pnl=${d['pnl']:+8.2f}  roi={roi:+6.1f}%")


raw = []
inverted = []
print(f"\nSimulating {len(mkts)} {ASSET} markets...")
for i, m in enumerate(mkts):
    raw.extend(simulate(m, invert=False, min_edge=3.0, nn_max=13))
    inverted.extend(simulate(m, invert=True, min_edge=3.0, nn_max=13))
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(mkts)}] raw={len(raw)} inv={len(inverted)}")

report(f"{ASSET} RAW (edge>=3c, T+10..T+13)", raw)
report(f"{ASSET} INVERTED", inverted)
