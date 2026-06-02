"""Realistic backtest — no future info, honest fill model.

Key differences vs backtest_last_night_asset.py:
1. DECISION price uses PREVIOUS minute's yes_bid_close/yes_ask_close
   (observable at decision time). Current minute's close is FUTURE info.
2. FILL PRICE uses our limit (prev_close + buffer), conditional on
   the market actually touching it during the current minute:
     - BUY YES: filled iff current_minute.yes_low <= our_yes_limit
     - BUY NO  (= short YES): filled iff current_minute.yes_high >= 1 - our_no_limit
3. Failed orders (limit never reached) → no bet, no P&L.
4. Uses FILL_BUFFER_CENTS = 5 (matches current live config).

Run: ASSET=BTC DAYS=7 python backtest_realistic.py
"""
import os, sys, time, json, math
sys.path.insert(0, "/root/kalshi-delta-hedging/live")
sys.path.insert(0, "/root/kalshi-delta-hedging")
ASSET = os.environ.get("ASSET", "BTC").upper()
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
DAYS = int(os.environ.get("DAYS", "7"))
cutoff = int(time.time()) - DAYS * 86400
print(f"Backtest: {ASSET} | last {DAYS} day(s) | REALISTIC fill model")

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

print(f"Settled {ASSET} markets: {len(mkts)}")
if not mkts: sys.exit(0)
mkts.sort(key=lambda m: m["close_time"])

t_min = min(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
t_max = max(int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
prices = fetch_prices(t_min - 600, t_max + 60)
print(f"{ASSET} prices loaded: {len(prices)}")

BASE_STAKE = 10.0
FILL_BUFFER = 0.05  # matches live (bumped from 0.03 → 0.05 on 2026-06-01)
FEE = 0.07
MAX_FILL_PRICE = 0.95


def time_decay_mult(m):
    if m < 7: return 0.4
    if m < 10: return 0.8
    return 1.2


def find_candle(candles, ts):
    """First candle at or after ts."""
    for c in candles:
        if c["ts"] >= ts:
            return c
    return None


def simulate(market, invert=False, min_edge=3.0, min_price=0.0, max_price=0.95, nn_max=13):
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
    for minute in range(10, nn_max + 1):
        t_dec = t0 + minute * 60  # decision time = start of minute m
        # DECISION REFERENCE: previous minute's close (= what we'd see at decision time)
        prev_candle = find_candle(candles, t_dec - 60)  # minute m-1
        if prev_candle is None or prev_candle["ts"] >= t_dec:
            continue
        ref_yes_bid = float(prev_candle.get("yes_bid_close", prev_candle["yes_close"]))
        ref_yes_ask = float(prev_candle.get("yes_ask_close", prev_candle["yes_close"]))
        if ref_yes_bid <= 0 or ref_yes_ask >= 1: continue
        # NN inference (uses past data, no future info)
        asset_now = data_mod.lookup(prices, t_dec)
        if asset_now is None: continue
        p_yes = trader._nn_p_yes(None, asset_t0, kt0, hour, minute,
                                 ticker=tk, open_iso=ot, close_iso=ct)
        direction_up = asset_now > asset_t0
        fair = p_yes if direction_up else (1.0 - p_yes)
        abs_pct_move = abs(asset_now - asset_t0) / asset_t0 * 100
        f_btc = strategy.sigmoid_btc(abs_pct_move)
        td_mult = time_decay_mult(minute)
        # Our LIMIT prices (based on observed prev_close)
        if direction_up:
            cost_aligned = ref_yes_ask + FILL_BUFFER
            mis = fair - cost_aligned
        else:
            cost_aligned = (1 - ref_yes_bid) + FILL_BUFFER
            mis = fair - cost_aligned
        edge_cents = mis * 100
        if edge_cents < min_edge: continue
        g = strategy.sigmoid_mispricing(mis)
        target = BASE_STAKE * f_btc * g * td_mult
        side = "yes" if direction_up else "no"
        if invert:
            side = "no" if side == "yes" else "yes"
        # Our limit price (what we'd send)
        if side == "yes":
            our_limit = ref_yes_ask + FILL_BUFFER
        else:
            our_limit = (1 - ref_yes_bid) + FILL_BUFFER
        if our_limit < min_price or our_limit > max_price: continue
        if our_limit > MAX_FILL_PRICE: continue
        # FILL CHECK: did market reach our limit during minute m?
        cur_candle = find_candle(candles, t_dec)
        if cur_candle is None or cur_candle["ts"] >= t_dec + 60:
            continue  # no current-minute data, skip
        yes_low = float(cur_candle.get("yes_low", cur_candle["yes_close"]))
        yes_high = float(cur_candle.get("yes_high", cur_candle["yes_close"]))
        if side == "yes":
            # Buying YES at our_limit: filled iff yes_low <= our_limit
            filled = yes_low <= our_limit
        else:
            # Buying NO at our_limit = shorting YES at (1 - our_limit)
            short_price = 1 - our_limit
            filled = yes_high >= short_price
        if not filled:
            bets.append({"window": ot, "minute": minute, "side": side, "fair": fair,
                         "fill": our_limit, "stake": target, "win": False, "pnl": 0.0,
                         "p_yes": p_yes, "result": result, "failed": True})
            continue
        # We filled at our_limit
        contracts = target / our_limit
        win = (side == "yes" and result == "yes") or (side == "no" and result == "no")
        if win:
            pnl = contracts * (1.0 - our_limit) * (1 - FEE)
        else:
            pnl = -target
        bets.append({"window": ot, "minute": minute, "side": side, "fair": fair,
                     "fill": our_limit, "stake": target, "win": win, "pnl": pnl,
                     "p_yes": p_yes, "result": result, "failed": False})
    return bets


def report(name, bets):
    print(f"\n=== {name} ===")
    print(f"Attempts:   {len(bets)}")
    if not bets: return
    failed = [b for b in bets if b.get("failed")]
    filled = [b for b in bets if not b.get("failed")]
    print(f"Failed:     {len(failed)} ({100*len(failed)/len(bets):.1f}%)")
    print(f"Filled:     {len(filled)} ({100*len(filled)/len(bets):.1f}%)")
    if not filled: return
    wins = sum(1 for b in filled if b["win"])
    pnl = sum(b["pnl"] for b in filled)
    wager = sum(b["stake"] for b in filled)
    print(f"Wins:       {wins}/{len(filled)} ({100*wins/len(filled):.1f}%)")
    print(f"Net P&L:    ${pnl:+.2f}")
    print(f"Wagered:    ${wager:.2f}")
    print(f"ROI:        {100*pnl/wager:+.2f}%")
    fills = [b["fill"] for b in filled]
    print(f"Fills:      min={min(fills):.3f} max={max(fills):.3f} med={sorted(fills)[len(fills)//2]:.3f}")
    print(f"$/day:      ${pnl/DAYS:+.2f}")


raw = []
inverted = []
print(f"\nSimulating {len(mkts)} {ASSET} markets...")
for i, m in enumerate(mkts):
    raw.extend(simulate(m, invert=False, min_edge=3.0, nn_max=13))
    inverted.extend(simulate(m, invert=True, min_edge=3.0, nn_max=13))
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(mkts)}] raw={len(raw)} inv={len(inverted)}")

report(f"{ASSET} RAW (realistic fill, edge>=3c, T+10..T+13)", raw)
report(f"{ASSET} INVERTED", inverted)
