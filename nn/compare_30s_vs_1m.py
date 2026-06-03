"""Head-to-head accuracy comparison: 1m model vs 30s model.

For each settled market in last N days, at each decision minute (T+10..T+13):
- Run 1m model (existing) → get p_yes_1m
- Run 30s model (new) → get p_yes_30s
- Compare to actual outcome

Reports calibration + accuracy for each model.

Run: ASSET=btc DAYS=7 python compare_30s_vs_1m.py
"""
import os, sys, time, json, math
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging")
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/nn")

ASSET = os.environ.get("ASSET", "btc").lower()
ASSET_U = ASSET.upper()
DAYS = int(os.environ.get("DAYS", "7"))

os.environ["ASSET"] = ASSET_U
os.environ["FAIR_PRICE_SOURCE"] = "nn"
os.environ["NN_CHECKPOINT"] = (
    "../nn/checkpoints/best_v2_small.pt"
    if ASSET_U == "BTC"
    else f"../nn/checkpoints/best_v2_small_{ASSET}.pt"
)
os.environ["PAPER_MODE"] = "true"
os.chdir("/home/ec2-user/kalshi-delta-hedging/live")

import trader
trader._load_nn()  # this loads the 1m model

# Now load the 30s model separately
from model import TSWinPredictor
CKPT_30S = f"/home/ec2-user/kalshi-delta-hedging/nn/checkpoints/best_v2_small_30s_{ASSET}.pt"
ckpt_30s = torch.load(CKPT_30S, map_location="cpu", weights_only=False)
NN_30S = TSWinPredictor(
    n_features=ckpt_30s["n_features"],
    d_model=ckpt_30s.get("d_model", 32),
    n_heads=ckpt_30s.get("n_heads", 4),
    n_layers=ckpt_30s.get("n_layers", 2),
    dim_feedforward=ckpt_30s.get("dim_feedforward", 64),
    dropout=ckpt_30s.get("dropout", 0.1),
    max_len=ckpt_30s.get("max_len", 30),
)
NN_30S.load_state_dict(ckpt_30s["model_state"])
NN_30S.eval()
MEAN_30S = np.array(ckpt_30s["feature_mean"], dtype=np.float32)
STD_30S = np.array(ckpt_30s["feature_std"], dtype=np.float32)

import kalshi_client
import requests
from datetime import datetime, timezone

if ASSET_U == "BTC":
    import btc_data as data_mod
elif ASSET_U == "ETH":
    import eth_data as data_mod
elif ASSET_U == "SOL":
    import sol_data as data_mod
elif ASSET_U == "XRP":
    import xrp_data as data_mod

# Load 30s binance buckets
from config import CACHE_DIR
import glob

def load_binance_30s_range(start_ts, end_ts):
    out = {}
    start_day = (start_ts // 86400) * 86400
    end_day = (end_ts // 86400) * 86400 + 86400
    cur = start_day
    while cur < end_day:
        day_str = datetime.utcfromtimestamp(cur).strftime("%Y%m%d")
        path = os.path.join(CACHE_DIR, f"binance_30s_{ASSET_U}_{day_str}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            for k, v in d.items():
                out[int(k)] = v
        cur += 86400
    return out


def lookup_30s(buckets, ts):
    bucket_ts = (ts // 30) * 30
    for offset in range(0, 300, 30):
        if (bucket_ts - offset) in buckets:
            return buckets[bucket_ts - offset]
    return None


def find_candle(candles, ts):
    for c in candles:
        if c["ts"] >= ts:
            return c
    return None


# Build 30s feature vector at decision point (step = T+minute * 2)
def build_30s_features(t0, decision_minute, candles, binance_buckets, asset_t0, kalshi_t0, hour, dow):
    """Build (30, 14) feature tensor + mask."""
    X = np.zeros((30, 14), dtype=np.float32)
    mask = np.zeros(30, dtype=bool)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin  = math.sin(2 * math.pi * dow / 7)
    decision_step = decision_minute * 2  # T+10 = step 20

    asset_history = []
    abs_max = 0.0
    for step in range(30):
        if step > decision_step:
            break
        t = t0 + step * 30
        bucket = lookup_30s(binance_buckets, t)
        cand = find_candle(candles, t)
        if bucket is None or cand is None:
            continue
        asset_now = bucket["close"]
        yc = float(cand["yes_close"])
        if not (0.01 < yc < 0.99):
            continue
        asset_history.append(asset_now)
        ret_t0 = (asset_now / asset_t0) - 1.0
        ret_1m = (asset_now / asset_history[-3]) - 1.0 if len(asset_history) >= 3 else 0.0
        abs_max = max(abs_max, abs(ret_t0))
        ret_5m = (asset_now / asset_history[-11]) - 1.0 if len(asset_history) >= 11 else 0.0
        yo = float(cand.get("yes_open", yc))
        yh = float(cand.get("yes_high", yc))
        yl = float(cand.get("yes_low", yc))
        bc = float(cand.get("yes_bid_close", yc))
        ac = float(cand.get("yes_ask_close", yc))
        vol = float(cand.get("volume", 0.0))
        intramin = yc - yo
        rng_norm = (yh - yl) / max(yc, 0.05)
        spread = max(0.0, min(0.20, ac - bc))
        vol_log = math.log1p(vol) / 10.0
        X[step, 0] = ret_t0
        X[step, 1] = ret_1m
        X[step, 2] = ret_5m
        X[step, 3] = abs_max
        X[step, 4] = yc
        X[step, 5] = yc - kalshi_t0
        X[step, 6] = intramin
        X[step, 7] = rng_norm
        X[step, 8] = spread
        X[step, 9] = vol_log
        X[step, 10] = step / 29.0
        X[step, 11] = hour_sin
        X[step, 12] = hour_cos
        X[step, 13] = dow_sin
        mask[step] = True
    return X, mask


def run_30s_nn(X, mask):
    if mask.sum() == 0:
        return 0.5
    Xn = ((X - MEAN_30S) / STD_30S).astype(np.float32)
    with torch.no_grad():
        logit = NN_30S(torch.from_numpy(Xn[None]), torch.from_numpy(mask[None]))
        return float(torch.sigmoid(logit).item())


# Fetch markets
url = "https://api.elections.kalshi.com/trade-api/v2/markets"
cutoff = int(time.time()) - DAYS * 86400
mkts = []
cursor = None
SERIES = f"KX{ASSET_U}15M"
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

t_min = min(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)
t_max = max(int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) for m in mkts)

# Crypto 1s data (already cached)
if ASSET_U == "BTC":
    prices_1m = data_mod.fetch_btc_prices(t_min - 600, t_max + 60)
elif ASSET_U == "ETH":
    prices_1m = data_mod.fetch_eth_prices(t_min - 600, t_max + 60)
elif ASSET_U == "SOL":
    prices_1m = data_mod.fetch_sol_prices(t_min - 600, t_max + 60)
elif ASSET_U == "XRP":
    prices_1m = data_mod.fetch_xrp_prices(t_min - 600, t_max + 60)
print(f"1m prices: {len(prices_1m)}")

# 30s data from binance
binance_30s = load_binance_30s_range(t_min - 600, t_max + 60)
print(f"30s buckets: {len(binance_30s)}")

# Collect predictions
preds_1m = []
preds_30s = []
print("Collecting predictions...")
for i, m in enumerate(mkts):
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(mkts)}]")
    ot = m["open_time"]; ct = m["close_time"]
    tk = m["ticker"]; result = m["result"]
    if result not in ("yes", "no"): continue
    actual = 1 if result == "yes" else 0
    t0 = int(datetime.fromisoformat(ot.replace("Z", "+00:00")).timestamp())
    asset_t0_1m = data_mod.lookup(prices_1m, t0)
    bucket_t0 = lookup_30s(binance_30s, t0)
    if asset_t0_1m is None or bucket_t0 is None: continue
    asset_t0_30s = bucket_t0["close"]
    try:
        candles = kalshi_client.fetch_candlesticks(tk, ot, ct)
    except Exception:
        continue
    if not candles: continue
    hour = datetime.fromisoformat(ot.replace("Z", "+00:00")).hour
    dow = datetime.fromisoformat(ot.replace("Z", "+00:00")).weekday()
    kt0_candle = find_candle(candles, t0)
    if kt0_candle is None: continue
    kt0 = float(kt0_candle["yes_close"])
    if not (0.01 < kt0 < 0.99): continue
    for minute in range(10, 14):
        # 1m model
        p_1m = trader._nn_p_yes(None, asset_t0_1m, kt0, hour, minute,
                                ticker=tk, open_iso=ot, close_iso=ct)
        # 30s model
        X30, mask30 = build_30s_features(t0, minute, candles, binance_30s,
                                          asset_t0_30s, kt0, hour, dow)
        p_30s = run_30s_nn(X30, mask30)
        preds_1m.append((p_1m, actual))
        preds_30s.append((p_30s, actual))

print(f"\nTotal predictions: 1m={len(preds_1m)}, 30s={len(preds_30s)}")
if not preds_1m: sys.exit(0)


def brier(preds):
    return sum((p - a) ** 2 for p, a in preds) / len(preds)


def log_loss(preds):
    eps = 1e-9
    return -sum(a * math.log(max(p, eps)) + (1-a) * math.log(max(1-p, eps)) for p, a in preds) / len(preds)


def accuracy(preds):
    return sum(1 for p, a in preds if (p > 0.5) == bool(a)) / len(preds)


def high_conf_accuracy(preds, thr=0.8):
    """For high-confidence predictions (p > thr or p < 1-thr), how accurate?"""
    confident = [(p, a) for p, a in preds if p > thr or p < 1 - thr]
    if not confident:
        return None, 0
    correct = sum(1 for p, a in confident if (p > 0.5) == bool(a))
    return correct / len(confident), len(confident)


print(f"\n{'METRIC':<28} {'1m model':<15} {'30s model':<15} {'Winner':<10}")
print("=" * 70)
brier_1m = brier(preds_1m); brier_30s = brier(preds_30s)
print(f"{'Brier (lower=better)':<28} {brier_1m:<15.4f} {brier_30s:<15.4f} {'30s' if brier_30s < brier_1m else '1m':<10}")
ll_1m = log_loss(preds_1m); ll_30s = log_loss(preds_30s)
print(f"{'Log loss (lower=better)':<28} {ll_1m:<15.4f} {ll_30s:<15.4f} {'30s' if ll_30s < ll_1m else '1m':<10}")
acc_1m = accuracy(preds_1m); acc_30s = accuracy(preds_30s)
print(f"{'Accuracy (higher=better)':<28} {acc_1m:<15.4f} {acc_30s:<15.4f} {'30s' if acc_30s > acc_1m else '1m':<10}")
hc_acc_1m, hc_n_1m = high_conf_accuracy(preds_1m, 0.8)
hc_acc_30s, hc_n_30s = high_conf_accuracy(preds_30s, 0.8)
if hc_acc_1m and hc_acc_30s:
    print(f"{'High conf accuracy (p>80%)':<28} {hc_acc_1m:.4f} (n={hc_n_1m:>4d}) {hc_acc_30s:.4f} (n={hc_n_30s:>4d}) {'30s' if hc_acc_30s > hc_acc_1m else '1m':<10}")

# Bin calibration
print(f"\nCalibration comparison (5% bins):")
print(f"{'Bin':<12} {'1m N':<8} {'1m Real':<10} {'30s N':<8} {'30s Real':<10}")
buckets_1m = defaultdict(lambda: [0, 0])
buckets_30s = defaultdict(lambda: [0, 0])
for (p, a) in preds_1m:
    b = int(p * 20) / 20
    buckets_1m[b][0] += a; buckets_1m[b][1] += 1
for (p, a) in preds_30s:
    b = int(p * 20) / 20
    buckets_30s[b][0] += a; buckets_30s[b][1] += 1
all_bins = sorted(set(list(buckets_1m) + list(buckets_30s)))
for b in all_bins:
    w_1m, n_1m = buckets_1m[b]
    w_30s, n_30s = buckets_30s[b]
    r_1m = w_1m/n_1m if n_1m else 0
    r_30s = w_30s/n_30s if n_30s else 0
    print(f"{b:.2f}-{b+0.05:.2f} {n_1m:<8} {r_1m:.3f}     {n_30s:<8} {r_30s:.3f}")
