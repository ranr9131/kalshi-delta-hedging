"""Replay last night's settled ETH 15-min windows with the working NN and
compute hypothetical P&L for Arm A (aligned, filtered) and Arm B (inverted)."""
import os, sys, time, json, math
sys.path.insert(0, "/root/kalshi-delta-hedging/live")
sys.path.insert(0, "/root/kalshi-delta-hedging")
os.environ["ASSET"] = "ETH"
os.environ["FAIR_PRICE_SOURCE"] = "nn"
os.environ["NN_CHECKPOINT"] = "../nn/checkpoints/best_v2_small_eth.pt"
os.environ["PAPER_MODE"] = "true"
os.chdir("/root/kalshi-delta-hedging/live")

import trader
trader._load_nn()  # explicit load — main() not called when imported
import kalshi_client, eth_data, strategy
import requests
from datetime import datetime, timezone

# Fetch settled markets from past 24h directly (no cache)
url = "https://api.elections.kalshi.com/trade-api/v2/markets"
cutoff = int(time.time()) - 24 * 3600
mkts = []
cursor = None
for page in range(50):
    params = {"series_ticker": "KXETH15M", "status": "settled",
              "min_close_ts": cutoff, "limit": 200}
    if cursor:
        params["cursor"] = cursor
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    d = r.json()
    mkts.extend(d.get("markets", []))
    cursor = d.get("cursor")
    if not cursor or not d.get("markets"):
        break
    time.sleep(0.2)

print("Settled markets last 24h:", len(mkts))
if not mkts:
    sys.exit(0)
mkts.sort(key=lambda m: m["close_time"])
print("First:", mkts[0]["close_time"], " Last:", mkts[-1]["close_time"])

# Load ETH price data for full range
t_min = min(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
t_max = max(int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
print("Range:", datetime.fromtimestamp(t_min, tz=timezone.utc).isoformat(),
      "->", datetime.fromtimestamp(t_max, tz=timezone.utc).isoformat())
prices = eth_data.fetch_eth_prices(t_min - 600, t_max + 60)
print("ETH prices loaded:", len(prices))

# Strategy parameters mirroring shadow services
BASE_STAKE = 10.0
MIN_EDGE_CENTS = 8.0
MIN_BET_PRICE = 0.40
MAX_FILL_PRICE = 0.95  # default in trader.py
FILL_BUFFER = 0.03
FEE = 0.07  # Kalshi fee on winnings
NN_MIN, NN_MAX = 10, 11


def time_decay_mult(m):
    if m < 7: return 0.4
    if m < 10: return 0.8
    return 1.2


def simulate(market, invert=False):
    ot = market["open_time"]
    ct = market["close_time"]
    tk = market["ticker"]
    result = market["result"]
    if result not in ("yes", "no"):
        return []
    t0 = int(datetime.fromisoformat(ot.replace("Z", "+00:00")).timestamp())
    eth_t0 = eth_data.lookup(prices, t0)
    if eth_t0 is None:
        return []
    trader._V2_WINDOW_CACHE.update({"open_iso": None})  # force refresh
    try:
        candles = kalshi_client.fetch_candlesticks(tk, ot, ct)
    except Exception:
        return []
    if not candles:
        return []
    bets = []
    hour = datetime.fromisoformat(ot.replace("Z", "+00:00")).hour
    kt0_candle = next((c for c in candles if c["ts"] >= t0), None)
    if kt0_candle is None:
        return []
    kt0 = float(kt0_candle["yes_close"])
    if not (0.01 < kt0 < 0.99):
        return []
    for minute in range(NN_MIN, NN_MAX + 1):
        t = t0 + minute * 60
        eth_now = eth_data.lookup(prices, t)
        if eth_now is None:
            continue
        cand_at_m = next((c for c in candles if c["ts"] >= t), None)
        if cand_at_m is None:
            continue
        yes_bid = float(cand_at_m.get("yes_bid_close", cand_at_m["yes_close"]))
        yes_ask = float(cand_at_m.get("yes_ask_close", cand_at_m["yes_close"]))
        if yes_bid <= 0 or yes_ask >= 1:
            continue
        # NN inference
        p_yes = trader._nn_p_yes(None, eth_t0, kt0, hour, minute,
                                 ticker=tk, open_iso=ot, close_iso=ct)
        direction_up = eth_now > eth_t0
        fair = p_yes if direction_up else (1.0 - p_yes)
        abs_pct_move = abs(eth_now - eth_t0) / eth_t0 * 100
        f_btc = strategy.sigmoid_btc(abs_pct_move)
        td_mult = time_decay_mult(minute)
        if direction_up:
            cost_aligned = yes_ask + FILL_BUFFER
            mis = fair - cost_aligned
        else:
            cost_aligned = (1 - yes_bid) + FILL_BUFFER
            mis = fair - cost_aligned
        edge_cents = mis * 100
        if edge_cents < MIN_EDGE_CENTS:
            continue
        g = strategy.sigmoid_mispricing(mis)
        target = BASE_STAKE * f_btc * g * td_mult
        side = "yes" if direction_up else "no"
        if invert:
            side = "no" if side == "yes" else "yes"
        # Recompute fill for the actual side we're buying
        if side == "yes":
            fill = yes_ask + FILL_BUFFER
        else:
            fill = (1 - yes_bid) + FILL_BUFFER
        if fill < MIN_BET_PRICE:
            continue
        if fill > MAX_FILL_PRICE:
            continue
        contracts = target / fill
        win = (side == "yes" and result == "yes") or (side == "no" and result == "no")
        if win:
            pnl = contracts * (1.0 - fill) * (1 - FEE)
        else:
            pnl = -target
        bets.append({
            "window": ot, "minute": minute, "side": side, "fair": fair,
            "fill": fill, "stake": target, "win": win, "pnl": pnl,
            "p_yes": p_yes, "result": result,
        })
    return bets


def report(name, bets):
    print(f"\n=== {name} ===")
    print(f"Bets:         {len(bets)}")
    if not bets:
        return
    wins = sum(1 for b in bets if b["win"])
    pnl = sum(b["pnl"] for b in bets)
    wager = sum(b["stake"] for b in bets)
    print(f"Wins:         {wins}/{len(bets)} ({100*wins/len(bets):.1f}%)")
    print(f"Net P&L:      ${pnl:+.2f}")
    print(f"Total wagered:${wager:.2f}")
    print(f"ROI:          {100*pnl/wager:+.2f}%")
    fills = [b["fill"] for b in bets]
    print(f"Fill prices:  min={min(fills):.3f} max={max(fills):.3f} median={sorted(fills)[len(fills)//2]:.3f}")
    print(f"Sample bets:")
    for b in bets[:10]:
        marker = "WIN " if b["win"] else "LOSS"
        print(f"  {b['window']} m{b['minute']}: {b['side'].upper()} @ {b['fill']:.3f} stake=${b['stake']:5.2f} → {marker} pnl=${b['pnl']:+6.2f} (NN p_yes={b['p_yes']:.3f}, settled={b['result']})")


arm_a = []
arm_b = []
print(f"\nSimulating {len(mkts)} markets...")
for i, m in enumerate(mkts):
    arm_a.extend(simulate(m, invert=False))
    arm_b.extend(simulate(m, invert=True))
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(mkts)}] A_bets={len(arm_a)} B_bets={len(arm_b)}")

report("ARM A (aligned, filtered: edge>=8c, price>=0.40)", arm_a)
report("ARM B (INVERTED, filtered: edge>=8c, price>=0.40)", arm_b)

# Also report what the unfiltered (raw) strategy would have done
print("\n--- Also: what raw config would have done (no MIN_BET_PRICE, edge>=3c, NN T+10..T+13) ---")
MIN_EDGE_CENTS = 3.0
MIN_BET_PRICE = 0.0
NN_MAX = 13
arm_a_raw = []
for m in mkts:
    arm_a_raw.extend(simulate(m, invert=False))
report("ARM A RAW (edge>=3, no price floor, T+10..T+13)", arm_a_raw)
