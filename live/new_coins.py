"""Show V2 PAPER performance for the 4 newer coins (BNB/TON/DOGE/ADA)."""
import csv
from datetime import datetime, timezone

settlements = {r["ticker"]: r for r in csv.DictReader(
    open("/home/ec2-user/kalshi-delta-hedging/live/settlements.csv"))}

NEW = ("BNB", "TON", "DOGE", "ADA", "HYPE")

print("=" * 90)
print("V2 PAPER — newer coins (added in the last ~12h)")
print("=" * 90)
print()
print("%-5s  %-7s  %-7s  %-9s  %-9s  %-9s  %-8s  %s" % (
    "asset", "fires", "settled", "W/L", "WR%", "stake", "pnl", "ROI"))
print("-" * 80)

totals = {"fires":0,"settled":0,"w":0,"l":0,"stake":0.0,"pnl":0.0}
per_asset = {}

for r in csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/snipes_v2.csv")):
    a = r.get("asset", "").upper()
    if a not in NEW: continue
    if a not in per_asset:
        per_asset[a] = {"fires":0,"settled":0,"w":0,"l":0,"stake":0.0,"pnl":0.0,
                        "first_ts":"","last_ts":""}
    p = per_asset[a]
    p["fires"] += 1
    p["stake"] += float(r.get("stake_dollars") or 0)
    if not p["first_ts"]: p["first_ts"] = r["ts_iso"]
    p["last_ts"] = r["ts_iso"]
    s = settlements.get(r["ticker"])
    if not s: continue
    result = (s.get("result") or "").lower()
    if result not in ("yes","no"): continue
    p["settled"] += 1
    try:
        qty = float(r["qty"]); stake = float(r["stake_dollars"])
    except: continue
    if r["side"] == result:
        p["w"] += 1
        p["pnl"] += qty - stake
    else:
        p["l"] += 1
        p["pnl"] -= stake

for a in NEW:
    if a not in per_asset:
        print("%-5s  %-7s  %s" % (a, "0", "no fires yet"))
        continue
    p = per_asset[a]
    settled = p["w"] + p["l"]
    wr = p["w"]/max(1,settled)*100
    roi = p["pnl"]/max(1,p["stake"])*100
    wl = "%d/%d" % (p["w"], p["l"])
    print("%-5s  %-7d  %-7d  %-9s  %-9s  $%-7.2f  $%+7.2f  %+.1f%%" % (
        a, p["fires"], settled, wl,
        ("%.0f%%" % wr) if settled else "—",
        p["stake"], p["pnl"], roi))
    totals["fires"] += p["fires"]
    totals["settled"] += settled
    totals["w"] += p["w"]; totals["l"] += p["l"]
    totals["stake"] += p["stake"]; totals["pnl"] += p["pnl"]

print("-" * 80)
tot_set = totals["w"] + totals["l"]
print("%-5s  %-7d  %-7d  %d/%d       %-9s  $%-7.2f  $%+7.2f  %+.1f%%" % (
    "TOTAL", totals["fires"], totals["settled"], totals["w"], totals["l"],
    ("%.0f%%" % (totals["w"]/max(1,tot_set)*100)) if tot_set else "—",
    totals["stake"], totals["pnl"], totals["pnl"]/max(1,totals["stake"])*100))

print()
print("=" * 90)
print("first fires by asset")
print("=" * 90)
for a in NEW:
    if a in per_asset:
        p = per_asset[a]
        print("  %s:  first=%s   last=%s" % (a, p["first_ts"][:19], p["last_ts"][:19]))

# Show calibration status for each
print()
print("=" * 90)
print("calibration status (need 20+ settled to get asset-specific fit)")
print("=" * 90)
import json
cal = json.load(open("/home/ec2-user/kalshi-delta-hedging/live/calibration_v2.json"))
for a in NEW:
    if a in cal:
        e = cal[a]
        print("  %-5s ASSET-SPECIFIC  a=%+.4f b=%+.4f n=%s" % (a, e["a"], e["b"], e.get("n","?")))
    else:
        need = max(0, 20 - per_asset.get(a, {"w":0,"l":0})["w"] - per_asset.get(a, {"w":0,"l":0})["l"])
        print("  %-5s using _global fallback  (need %d more settled to refit)" % (a, need))
