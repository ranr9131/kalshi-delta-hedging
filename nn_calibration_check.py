"""NN calibration check on recent settled markets.

Asks: when NN says 90% YES, does YES actually win 90% of the time?
Bins predictions by 5% buckets, computes realized win rate per bucket.

A well-calibrated model has predicted ≈ realized in every bucket.
A drifted/overconfident model has realized << predicted in extreme bins.

Run: ASSET=BTC DAYS=7 python nn_calibration_check.py
"""
import os, sys, time, math
from collections import defaultdict
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging")
ASSET = os.environ.get("ASSET", "BTC").upper()
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
import kalshi_client
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

SERIES = f"KX{ASSET}15M"
DAYS = int(os.environ.get("DAYS", "7"))
cutoff = int(time.time()) - DAYS * 86400
print(f"Calibration check: {ASSET} | last {DAYS} days")

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
print(f"Markets: {len(mkts)}")
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
print(f"Prices: {len(prices)}")


def find_candle(candles, ts):
    for c in candles:
        if c["ts"] >= ts:
            return c
    return None


# For each market+minute: record (predicted_p_yes, actual_yes_won)
records = []

for i, market in enumerate(mkts):
    ot = market["open_time"]; ct = market["close_time"]
    tk = market["ticker"]; result = market["result"]
    if result not in ("yes", "no"): continue
    actual_yes = 1 if result == "yes" else 0
    t0 = int(datetime.fromisoformat(ot.replace("Z", "+00:00")).timestamp())
    asset_t0 = data_mod.lookup(prices, t0)
    if asset_t0 is None: continue
    trader._V2_WINDOW_CACHE.update({"open_iso": None})
    try:
        candles = kalshi_client.fetch_candlesticks(tk, ot, ct)
    except Exception:
        continue
    if not candles: continue
    hour = datetime.fromisoformat(ot.replace("Z", "+00:00")).hour
    kt0_candle = find_candle(candles, t0)
    if kt0_candle is None: continue
    kt0 = float(kt0_candle["yes_close"])
    if not (0.01 < kt0 < 0.99): continue
    for minute in range(10, 14):
        t_dec = t0 + minute * 60
        asset_now = data_mod.lookup(prices, t_dec)
        if asset_now is None: continue
        p_yes = trader._nn_p_yes(None, asset_t0, kt0, hour, minute,
                                 ticker=tk, open_iso=ot, close_iso=ct)
        records.append((p_yes, actual_yes))
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(mkts)}] records={len(records)}")

print(f"\nTotal predictions: {len(records)}")
if not records: sys.exit(0)

# Bin by 5% buckets
buckets = defaultdict(lambda: [0, 0])  # bucket -> [yes_wins, total]
for p_yes, actual_yes in records:
    bucket = int(p_yes * 20) / 20  # round down to nearest 5%
    buckets[bucket][0] += actual_yes
    buckets[bucket][1] += 1

print(f"\n{'Predicted':>12} {'N':>6} {'Realized':>10} {'Calibration':>14}")
print("-" * 50)
brier = 0
for bucket in sorted(buckets):
    wins, total = buckets[bucket]
    realized = wins / total if total > 0 else 0
    err = realized - bucket - 0.025  # center of bucket
    mark = ""
    if abs(err) > 0.10: mark = " ⚠"
    if abs(err) > 0.20: mark = " 🚨"
    print(f"  {bucket:.2f}-{bucket+0.05:.2f} {total:>6} {realized:>9.1%}  {err:>+10.1%}{mark}")

# Brier score
brier = sum((p - a) ** 2 for p, a in records) / len(records)
print(f"\nBrier score: {brier:.4f} (lower = better; 0 = perfect, 0.25 = random)")

# Overall calibration
avg_pred = sum(p for p, _ in records) / len(records)
avg_actual = sum(a for _, a in records) / len(records)
print(f"Mean predicted p_yes: {avg_pred:.3f}")
print(f"Actual yes rate:      {avg_actual:.3f}")
print(f"Bias:                 {avg_pred - avg_actual:+.3f} (positive = overestimating YES)")

# Predictions strongly favoring YES or NO — are those reliable?
strong_yes = [(p, a) for p, a in records if p > 0.80]
strong_no = [(p, a) for p, a in records if p < 0.20]
if strong_yes:
    wr = sum(a for _, a in strong_yes) / len(strong_yes)
    print(f"\n'High confidence YES' (p>80%): {len(strong_yes)} predictions, actual YES rate = {wr:.1%}")
if strong_no:
    wr = sum(a for _, a in strong_no) / len(strong_no)
    print(f"'High confidence NO'  (p<20%): {len(strong_no)} predictions, actual NO rate = {1-wr:.1%}")
