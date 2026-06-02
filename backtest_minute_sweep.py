"""Sweep MAX_FILL_PRICE × MINUTE_RANGE. Tests strategy at different parts of
the window. NN_MIN/NN_MAX configurable via env vars.

Run: ASSET=BTC DAYS=7 NN_MIN=4 NN_MAX=8 python backtest_minute_sweep.py
"""
import os, sys, time, json, math
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging")
ASSET = os.environ.get("ASSET", "BTC").upper()
NN_MIN = int(os.environ.get("NN_MIN", "10"))
NN_MAX = int(os.environ.get("NN_MAX", "13"))
os.environ["ASSET"] = ASSET
os.environ["FAIR_PRICE_SOURCE"] = "nn"
os.environ["NN_CHECKPOINT"] = (
    "../nn/checkpoints/best_v2_small.pt"
    if ASSET == "BTC"
    else f"../nn/checkpoints/best_v2_small_{ASSET.lower()}.pt"
)
os.environ["PAPER_MODE"] = "true"
os.chdir("/home/ec2-user/kalshi-delta-hedging/live")

import trader
trader._load_nn()
import kalshi_client, strategy
import requests
from datetime import datetime, timezone

if ASSET == "ETH":
    import eth_data as data_mod
elif ASSET == "BTC":
    import btc_data as data_mod
elif ASSET == "SOL":
    import sol_data as data_mod
elif ASSET == "XRP":
    import xrp_data as data_mod
else:
    sys.exit(f"Unknown asset {ASSET}")

SERIES = f"KX{ASSET}15M"
DAYS = int(os.environ.get("DAYS", "7"))
cutoff = int(time.time()) - DAYS * 86400
print(f"Backtest: {ASSET} | {DAYS} days | minutes T+{NN_MIN}..T+{NN_MAX}")

url = "https://api.elections.kalshi.com/trade-api/v2/markets"
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
    if not cursor or not d.get("markets"): break
    time.sleep(0.2)

print(f"Settled markets: {len(mkts)}")
if not mkts: sys.exit(0)
mkts.sort(key=lambda m: m["close_time"])

t_min = min(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
t_max = max(int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)

if ASSET == "BTC":
    prices = data_mod.fetch_btc_prices(t_min - 600, t_max + 60)
elif ASSET == "ETH":
    prices = data_mod.fetch_eth_prices(t_min - 600, t_max + 60)
elif ASSET == "SOL":
    prices = data_mod.fetch_sol_prices(t_min - 600, t_max + 60)
elif ASSET == "XRP":
    prices = data_mod.fetch_xrp_prices(t_min - 600, t_max + 60)
print(f"Prices loaded: {len(prices)}")

BASE_STAKE = 10.0
FILL_BUFFER = 0.05
FEE = 0.07


def time_decay_mult(m):
    if m < 7: return 0.4
    if m < 10: return 0.8
    return 1.2


def find_candle(candles, ts):
    for c in candles:
        if c["ts"] >= ts:
            return c
    return None


def collect(market):
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
    kt0_candle = find_candle(candles, t0)
    if kt0_candle is None: return []
    kt0 = float(kt0_candle["yes_close"])
    if not (0.01 < kt0 < 0.99): return []
    for minute in range(NN_MIN, NN_MAX + 1):
        t_dec = t0 + minute * 60
        prev_candle = find_candle(candles, t_dec - 60)
        if prev_candle is None or prev_candle["ts"] >= t_dec: continue
        ref_yes_bid = float(prev_candle.get("yes_bid_close", prev_candle["yes_close"]))
        ref_yes_ask = float(prev_candle.get("yes_ask_close", prev_candle["yes_close"]))
        if ref_yes_bid <= 0 or ref_yes_ask >= 1: continue
        asset_now = data_mod.lookup(prices, t_dec)
        if asset_now is None: continue
        p_yes = trader._nn_p_yes(None, asset_t0, kt0, hour, minute,
                                 ticker=tk, open_iso=ot, close_iso=ct)
        direction_up = asset_now > asset_t0
        fair = p_yes if direction_up else (1.0 - p_yes)
        abs_pct_move = abs(asset_now - asset_t0) / asset_t0 * 100
        f_btc = strategy.sigmoid_btc(abs_pct_move)
        td_mult = time_decay_mult(minute)
        if direction_up:
            cost_aligned = ref_yes_ask + FILL_BUFFER
            mis = fair - cost_aligned
        else:
            cost_aligned = (1 - ref_yes_bid) + FILL_BUFFER
            mis = fair - cost_aligned
        edge_cents = mis * 100
        if edge_cents < 3.0: continue
        g = strategy.sigmoid_mispricing(mis)
        target = BASE_STAKE * f_btc * g * td_mult
        side = "yes" if direction_up else "no"
        if side == "yes":
            our_limit = ref_yes_ask + FILL_BUFFER
        else:
            our_limit = (1 - ref_yes_bid) + FILL_BUFFER
        cur_candle = find_candle(candles, t_dec)
        if cur_candle is None or cur_candle["ts"] >= t_dec + 60: continue
        yes_low = float(cur_candle.get("yes_low", cur_candle["yes_close"]))
        yes_high = float(cur_candle.get("yes_high", cur_candle["yes_close"]))
        if side == "yes":
            filled = yes_low <= our_limit
        else:
            short_price = 1 - our_limit
            filled = yes_high >= short_price
        if not filled: continue
        contracts = target / our_limit
        win = (side == "yes" and result == "yes") or (side == "no" and result == "no")
        if win:
            pnl = contracts * (1.0 - our_limit) * (1 - FEE)
        else:
            pnl = -target
        bets.append({"window": ot, "minute": minute, "side": side,
                     "fill": our_limit, "stake": target, "win": win, "pnl": pnl,
                     "p_yes": p_yes, "result": result})
    return bets


print(f"\nCollecting bets...")
all_bets = []
for i, m in enumerate(mkts):
    all_bets.extend(collect(m))
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(mkts)}] bets={len(all_bets)}")
print(f"\nTotal bets: {len(all_bets)}")
if not all_bets: sys.exit(0)

# Fill distribution
fills = sorted([b["fill"] for b in all_bets])
print(f"Fill range: min={fills[0]:.3f} med={fills[len(fills)//2]:.3f} max={fills[-1]:.3f}")

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.95]
print(f"\n{'max_fill':>8} {'bets':>6} {'wins':>6} {'wr':>5} {'wagered':>9} {'P&L':>9} {'ROI':>7} {'$/day':>8}")
print("-" * 72)
for thr in THRESHOLDS:
    filtered = [b for b in all_bets if b["fill"] <= thr]
    n = len(filtered)
    if n == 0:
        print(f"{thr:>8.2f}    0      0    0%      $0     $0      n/a      $0")
        continue
    wins = sum(1 for b in filtered if b["win"])
    wager = sum(b["stake"] for b in filtered)
    pnl = sum(b["pnl"] for b in filtered)
    wr = 100 * wins / n
    roi = 100 * pnl / wager if wager > 0 else 0
    per_day = pnl / DAYS
    print(f"{thr:>8.2f}{n:>6}{wins:>6}  {wr:>4.1f}% ${wager:>7.0f}  ${pnl:>+7.0f}  {roi:>+5.1f}%  ${per_day:>+5.0f}")
