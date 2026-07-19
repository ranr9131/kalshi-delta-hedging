"""Diagnose recent losses — by hour, by asset, by direction."""
import sys, csv, os
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
from dotenv import dotenv_values
import requests, kalshi_auth, kalshi_trade
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")

def bal(label):
    k = kalshi_auth.load_private_key(env[label + "_PRIVATE_KEY"])
    return kalshi_trade.get_balance(k, env[label + "_KEY_ID"])

leo_bal = bal("LEO")
fr_bal  = bal("FRIEND")
total   = leo_bal + fr_bal
LEO_AT_MULTI_START   = 761.33
FR_AT_MULTI_START    = 79.06
LIVE_TOT_AT_START    = 675.62  # original start of live trading yesterday

now = datetime.now(timezone.utc)

print("=" * 60)
print("BALANCE TIMELINE")
print("=" * 60)
print("  Original live start (yesterday 20:36 UTC): $%.2f (LEO only)" % LIVE_TOT_AT_START)
print("  At multi switch (today 15:30 UTC):         LEO=$%.2f FRIEND=$%.2f  TOT=$%.2f" % (
    LEO_AT_MULTI_START, FR_AT_MULTI_START, LEO_AT_MULTI_START+FR_AT_MULTI_START))
print("  NOW (%s):                  LEO=$%.2f FRIEND=$%.2f  TOT=$%.2f" % (
    now.strftime("%H:%M"), leo_bal, fr_bal, total))
print()
print("  Delta since live start (~24h ago):  $%+.2f" % (total - LIVE_TOT_AT_START))
print("  Delta since multi switch (~%.1fh):  $%+.2f" % (
    (now - datetime(2026,6,5,15,30,tzinfo=timezone.utc)).total_seconds()/3600,
    total - (LEO_AT_MULTI_START + FR_AT_MULTI_START)))
print()

# Settlement analysis — last 6 hours bucketed hourly
settlements = {r["ticker"]: r for r in csv.DictReader(
    open("/home/ec2-user/kalshi-delta-hedging/live/settlements.csv"))}

# Pull settled snipes from all csvs
all_snipes = []
for path in ("snipes.csv", "snipes_leo.csv", "snipes_friend.csv"):
    p = "/home/ec2-user/kalshi-delta-hedging/live/" + path
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        if r.get("mode") != "live": continue
        all_snipes.append((path, r))

# Bucket by hour
buckets = defaultdict(lambda: {"w":0,"l":0,"open":0,"pnl":0.0,"stake":0.0,
                                "by_asset":defaultdict(lambda:{"w":0,"l":0,"pnl":0.0})})
for path, r in all_snipes:
    try:
        ts = datetime.fromisoformat(r["ts_iso"].replace("Z","+00:00"))
    except Exception:
        continue
    if (now - ts).total_seconds() > 6*3600: continue  # last 6h only
    hour = ts.strftime("%H:00")
    b = buckets[hour]
    try:
        qty = float(r["qty"]); stake = float(r["stake_dollars"])
    except: continue
    b["stake"] += stake
    a = r["asset"]
    s = settlements.get(r["ticker"])
    if not s:
        b["open"] += 1
        continue
    result = (s.get("result") or "").lower()
    if result not in ("yes","no"): continue
    if r["side"] == result:
        b["w"] += 1; b["pnl"] += qty - stake
        b["by_asset"][a]["w"] += 1; b["by_asset"][a]["pnl"] += qty - stake
    else:
        b["l"] += 1; b["pnl"] -= stake
        b["by_asset"][a]["l"] += 1; b["by_asset"][a]["pnl"] -= stake

print("=" * 60)
print("LAST 6 HOURS (settled snipes only — open ones not in PnL yet)")
print("=" * 60)
print(f"  {'hour':<8} {'W':>4} {'L':>4} {'WR':>5} {'open':>5} {'stake':>9} {'pnl':>10} {'ROI':>7}")
for hour in sorted(buckets.keys()):
    b = buckets[hour]
    set_ = b["w"]+b["l"]
    wr = b["w"]/max(1,set_)*100
    roi = b["pnl"]/max(1,b["stake"])*100
    print("  %-8s %4d %4d %4.0f%% %5d  $%7.2f  $%+8.2f %+6.1f%%" % (
        hour, b["w"], b["l"], wr, b["open"], b["stake"], b["pnl"], roi))

# Top losing assets in last 3h
print()
print("=" * 60)
print("ASSETS BY PNL — last 3 hours")
print("=" * 60)
recent_by_asset = defaultdict(lambda: {"w":0,"l":0,"open":0,"pnl":0.0,"stake":0.0,
                                        "yes_pnl":0.0,"no_pnl":0.0})
for path, r in all_snipes:
    try:
        ts = datetime.fromisoformat(r["ts_iso"].replace("Z","+00:00"))
    except: continue
    if (now - ts).total_seconds() > 3*3600: continue
    try:
        qty = float(r["qty"]); stake = float(r["stake_dollars"])
    except: continue
    a = r["asset"]
    d = recent_by_asset[a]
    d["stake"] += stake
    s = settlements.get(r["ticker"])
    if not s:
        d["open"] += 1
        continue
    result = (s.get("result") or "").lower()
    if result not in ("yes","no"): continue
    if r["side"] == result:
        p = qty - stake; d["pnl"] += p; d["w"] += 1
        if r["side"] == "yes": d["yes_pnl"] += p
        else: d["no_pnl"] += p
    else:
        p = -stake; d["pnl"] += p; d["l"] += 1
        if r["side"] == "yes": d["yes_pnl"] += p
        else: d["no_pnl"] += p

for a in sorted(recent_by_asset.keys(), key=lambda x: recent_by_asset[x]["pnl"]):
    d = recent_by_asset[a]
    set_ = d["w"]+d["l"]
    wr = d["w"]/max(1,set_)*100 if set_ else 0
    print("  %s: W/L=%d/%d (%.0f%%) open=%d  stake=$%.2f  pnl=$%+.2f  (yes=$%+.2f no=$%+.2f)" % (
        a, d["w"], d["l"], wr, d["open"], d["stake"], d["pnl"], d["yes_pnl"], d["no_pnl"]))

# Check if crypto moved a lot recently
print()
print("=" * 60)
print("CRYPTO PRICE CHANGES IN LAST 3 HOURS (from Coinbase 1h candles)")
print("=" * 60)
for coin, pid in [("BTC","BTC-USD"),("ETH","ETH-USD"),("SOL","SOL-USD"),
                  ("XRP","XRP-USD"),("HYPE","HYPE-USD")]:
    end_ms = int(now.timestamp())
    start_ms = end_ms - 4*3600
    from datetime import datetime as dt
    start_iso = dt.fromtimestamp(start_ms, tz=timezone.utc).isoformat()
    end_iso   = dt.fromtimestamp(end_ms, tz=timezone.utc).isoformat()
    try:
        r = requests.get("https://api.exchange.coinbase.com/products/"+pid+"/candles",
                         params={"granularity":3600,"start":start_iso,"end":end_iso}, timeout=10)
        rows = r.json()
        if len(rows) < 2:
            print("  %s: insufficient data" % coin)
            continue
        # rows = [[time, low, high, open, close, vol], ...] descending
        rows.sort(key=lambda x: x[0])
        first_open = float(rows[0][3])
        last_close = float(rows[-1][4])
        change = (last_close - first_open) / first_open * 100
        high = max(float(r[2]) for r in rows)
        low  = min(float(r[1]) for r in rows)
        range_pct = (high - low) / first_open * 100
        print("  %s: $%.4f → $%.4f  (%+.2f%%)  range %.2f%%" % (
            coin, first_open, last_close, change, range_pct))
    except Exception as e:
        print("  %s: err %s" % (coin, e))
