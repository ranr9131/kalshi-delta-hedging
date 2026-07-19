import sys, csv
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
from dotenv import dotenv_values
import requests, kalshi_auth
from datetime import datetime, timezone

env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")
key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
keyid = env["KALSHI_API_KEY_ID"]

def get(path, params=None):
    h = kalshi_auth.make_auth_headers(key, keyid, "GET", path)
    return requests.get("https://api.elections.kalshi.com" + path, headers=h, params=params, timeout=15).json()

bal = get("/trade-api/v2/portfolio/balance").get("balance", 0) / 100.0

# Compute today-only PnL from snipes.csv (LIVE only) + settlements.csv
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
settlements = {r["ticker"]: r for r in csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/settlements.csv"))}

today_pnl = 0.0
today_wins = today_losses = today_open = 0
today_stake = 0.0
today_by_asset = {}
for r in csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/snipes.csv")):
    if r.get("mode") != "live": continue
    if not r["ts_iso"].startswith(TODAY): continue
    a = r["asset"]
    if a not in today_by_asset: today_by_asset[a] = {"n":0,"w":0,"l":0,"pnl":0.0,"stake":0.0}
    today_by_asset[a]["n"] += 1
    today_stake += float(r["stake_dollars"] or 0)
    today_by_asset[a]["stake"] += float(r["stake_dollars"] or 0)
    s = settlements.get(r["ticker"])
    if not s:
        today_open += 1
        continue
    result = (s.get("result") or "").lower()
    if result not in ("yes","no"): continue
    if r["side"] == result:
        p = float(r["qty"]) - float(r["stake_dollars"])
        today_pnl += p
        today_wins += 1
        today_by_asset[a]["w"] += 1
        today_by_asset[a]["pnl"] += p
    else:
        p = -float(r["stake_dollars"])
        today_pnl += p
        today_losses += 1
        today_by_asset[a]["l"] += 1
        today_by_asset[a]["pnl"] += p

# Last hour
from datetime import timedelta
ONE_HR_AGO = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()[:19]
hr_pnl = 0.0; hr_n = hr_w = hr_l = 0
for r in csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/snipes.csv")):
    if r.get("mode") != "live": continue
    if r["ts_iso"][:19] < ONE_HR_AGO: continue
    hr_n += 1
    s = settlements.get(r["ticker"])
    if not s: continue
    result = (s.get("result") or "").lower()
    if result not in ("yes","no"): continue
    if r["side"] == result:
        hr_pnl += float(r["qty"]) - float(r["stake_dollars"])
        hr_w += 1
    else:
        hr_pnl -= float(r["stake_dollars"])
        hr_l += 1

# Lifetime since live start
START_BAL = 675.62
lifetime_delta = bal - START_BAL
START_DT = datetime(2026,6,4,20,36,tzinfo=timezone.utc)
hrs_live = (datetime.now(timezone.utc) - START_DT).total_seconds() / 3600

print("=" * 50)
print("LIVE PNL DASHBOARD")
print("=" * 50)
print("  Kalshi balance now:  $%.2f" % bal)
print("  Started live at:     $675.62  (20:36 UTC yesterday)")
print()
print("  LIFETIME DELTA:      $%+.2f  (%+.1f%%)  over %.1fh = $%+.2f/hr" % (
    lifetime_delta, lifetime_delta/675.62*100, hrs_live, lifetime_delta/max(1,hrs_live)))
print()
print("  TODAY UTC (%s):      $%+.2f" % (TODAY, today_pnl))
print("    fires:    %d (W=%d L=%d, %d still open)" % (today_wins+today_losses+today_open, today_wins, today_losses, today_open))
print("    stake:    $%.2f" % today_stake)
print("    ROI:      %.1f%%" % (today_pnl/max(1,today_stake)*100))
print()
print("  LAST HOUR:          $%+.2f  (%d fires, W=%d L=%d)" % (hr_pnl, hr_n, hr_w, hr_l))
print()
print("  per asset today:")
for a in sorted(today_by_asset):
    d = today_by_asset[a]
    wr = d["w"]/max(1,d["w"]+d["l"])*100
    print("    %s: n=%d  W/L=%d/%d (%.0f%%)  pnl=$%+.2f" % (a, d["n"], d["w"], d["l"], wr, d["pnl"]))
