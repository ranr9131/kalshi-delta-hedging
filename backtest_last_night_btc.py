"""Replay last 24h settled BTC 15-min windows with the v2_small BTC NN
and compute hypothetical P&L for filtered + raw configs, plus inverted."""
import os, sys, time, json, math
sys.path.insert(0, "/root/kalshi-delta-hedging/live")
sys.path.insert(0, "/root/kalshi-delta-hedging")
os.environ["ASSET"] = "BTC"
os.environ["FAIR_PRICE_SOURCE"] = "nn"
os.environ["NN_CHECKPOINT"] = "../nn/checkpoints/best_v2_small.pt"
os.environ["PAPER_MODE"] = "true"
os.chdir("/root/kalshi-delta-hedging/live")

import trader
trader._load_nn()
import kalshi_client, btc_data, strategy
import requests
from datetime import datetime, timezone

url = "https://api.elections.kalshi.com/trade-api/v2/markets"
cutoff = int(time.time()) - 24 * 3600
mkts = []
cursor = None
for page in range(50):
    params = {"series_ticker": "KXBTC15M", "status": "settled",
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

print("Settled BTC markets last 24h:", len(mkts))
if not mkts:
    sys.exit(0)
mkts.sort(key=lambda m: m["close_time"])
print("First:", mkts[0]["close_time"], " Last:", mkts[-1]["close_time"])

t_min = min(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
t_max = max(int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
print("Range:", datetime.fromtimestamp(t_min, tz=timezone.utc).isoformat(),
      "->", datetime.fromtimestamp(t_max, tz=timezone.utc).isoformat())
prices = btc_data.fetch_btc_prices(t_min - 600, t_max + 60)
print("BTC prices loaded:", len(prices))

BASE_STAKE = 10.0
MIN_EDGE_CENTS = 3.0
MIN_BET_PRICE = 0.0
MAX_FILL_PRICE = 0.95
FILL_BUFFER = 0.03
FEE = 0.07
NN_MIN, NN_MAX = 10, 13


def time_decay_mult(m):
    if m < 7: return 0.4
    if m < 10: return 0.8
    return 1.2


def simulate(market, invert=False, min_edge=3.0, min_price=0.0, nn_max=13):
    ot = market["open_time"]
    ct = market["close_time"]
    tk = market["ticker"]
    result = market["result"]
    if result not in ("yes", "no"):
        return []
    t0 = int(datetime.fromisoformat(ot.replace("Z", "+00:00")).timestamp())
    btc_t0 = btc_data.lookup(prices, t0)
    if btc_t0 is None:
        return []
    trader._V2_WINDOW_CACHE.update({"open_iso": None})
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
    for minute in range(NN_MIN, nn_max + 1):
        t = t0 + minute * 60
        btc_now = btc_data.lookup(prices, t)
        if btc_now is None:
            continue
        cand_at_m = next((c for c in candles if c["ts"] >= t), None)
        if cand_at_m is None:
            continue
        yes_bid = float(cand_at_m.get("yes_bid_close", cand_at_m["yes_close"]))
        yes_ask = float(cand_at_m.get("yes_ask_close", cand_at_m["yes_close"]))
        if yes_bid <= 0 or yes_ask >= 1:
            continue
        p_yes = trader._nn_p_yes(None, btc_t0, kt0, hour, minute,
                                 ticker=tk, open_iso=ot, close_iso=ct)
        direction_up = btc_now > btc_t0
        fair = p_yes if direction_up else (1.0 - p_yes)
        abs_pct_move = abs(btc_now - btc_t0) / btc_t0 * 100
        f_btc = strategy.sigmoid_btc(abs_pct_move)
        td_mult = time_decay_mult(minute)
        if direction_up:
            cost_aligned = yes_ask + FILL_BUFFER
            mis = fair - cost_aligned
        else:
            cost_aligned = (1 - yes_bid) + FILL_BUFFER
            mis = fair - cost_aligned
        edge_cents = mis * 100
        if edge_cents < min_edge:
            continue
        g = strategy.sigmoid_mispricing(mis)
        target = BASE_STAKE * f_btc * g * td_mult
        side = "yes" if direction_up else "no"
        if invert:
            side = "no" if side == "yes" else "yes"
        if side == "yes":
            fill = yes_ask + FILL_BUFFER
        else:
            fill = (1 - yes_bid) + FILL_BUFFER
        if fill < min_price:
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


print(f"\nSimulating {len(mkts)} BTC markets — 3 configs...")

raw_aligned = []
raw_inverted = []
filtered_aligned = []

for i, m in enumerate(mkts):
    raw_aligned.extend(simulate(m, invert=False, min_edge=3.0, min_price=0.0, nn_max=13))
    raw_inverted.extend(simulate(m, invert=True, min_edge=3.0, min_price=0.0, nn_max=13))
    filtered_aligned.extend(simulate(m, invert=False, min_edge=8.0, min_price=0.40, nn_max=11))
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(mkts)}] raw={len(raw_aligned)} inv={len(raw_inverted)} filt={len(filtered_aligned)}")

report("BTC RAW (edge>=3c, no price floor, T+10..T+13) — same config as ETH live", raw_aligned)
report("BTC INVERTED (raw, flipped)", raw_inverted)
report("BTC FILTERED (edge>=8c, price>=0.40, T+10..T+11)", filtered_aligned)

print("\n--- Hourly P&L breakdown (raw aligned) ---")
from collections import defaultdict
by_hour = defaultdict(lambda: [0, 0.0])
for b in raw_aligned:
    h = b["window"][:13]
    by_hour[h][0] += 1
    by_hour[h][1] += b["pnl"]
for h in sorted(by_hour):
    n, p = by_hour[h]
    print(f"  {h}: {n} bets, pnl=${p:+.2f}")
