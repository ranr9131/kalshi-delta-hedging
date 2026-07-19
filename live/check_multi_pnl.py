"""Quick health check on the multi-account sniper."""
import sys, csv, os
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
from dotenv import dotenv_values
import requests, kalshi_auth, kalshi_trade
from datetime import datetime, timezone, timedelta

env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")

# Pull both account balances
def bal(label):
    k = kalshi_auth.load_private_key(env[f"{label}_PRIVATE_KEY"])
    return kalshi_trade.get_balance(k, env[f"{label}_KEY_ID"])

leo_bal = bal("LEO")
fr_bal  = bal("FRIEND")
total   = leo_bal + fr_bal

# Initial balances when multi started (from earlier setup output)
LEO_START = 761.33
FR_START  = 79.06
TOT_START = 840.39

print("=" * 50)
print("BALANCE NOW vs MULTI START (~15:30 UTC)")
print("=" * 50)
print(f"  LEO:    ${leo_bal:.2f}  (was ${LEO_START:.2f},  delta $%+.2f)" % (leo_bal - LEO_START))
print(f"  FRIEND: ${fr_bal:.2f}   (was ${FR_START:.2f},   delta $%+.2f)" % (fr_bal - FR_START))
print(f"  TOTAL:  ${total:.2f}  (was ${TOT_START:.2f},  delta $%+.2f)" % (total - TOT_START))
print()

# Settled snipes per account since multi start
settlements = {r["ticker"]: r for r in csv.DictReader(
    open("/home/ec2-user/kalshi-delta-hedging/live/settlements.csv"))}

START_TS = "2026-06-05T15:30:00"
for label, fname in [("LEO", "snipes_leo.csv"), ("FRIEND", "snipes_friend.csv")]:
    p = "/home/ec2-user/kalshi-delta-hedging/live/" + fname
    if not os.path.exists(p):
        print(f"{label}: no snipes file yet")
        continue
    rows = [r for r in csv.DictReader(open(p)) if r["ts_iso"] >= START_TS]
    won = lost = open_ = 0
    pnl = 0.0; stake = 0.0
    by_asset = {}
    last_5 = []
    for r in rows:
        stake += float(r["stake_dollars"] or 0)
        a = r["asset"]
        if a not in by_asset: by_asset[a] = {"w":0,"l":0,"open":0,"pnl":0.0}
        s = settlements.get(r["ticker"])
        last_5.append(r)
        if not s:
            open_ += 1; by_asset[a]["open"] += 1; continue
        result = (s.get("result") or "").lower()
        if result not in ("yes","no"): continue
        if r["side"] == result:
            p = float(r["qty"]) - float(r["stake_dollars"])
            pnl += p; won += 1; by_asset[a]["w"] += 1; by_asset[a]["pnl"] += p
        else:
            p = -float(r["stake_dollars"])
            pnl += p; lost += 1; by_asset[a]["l"] += 1; by_asset[a]["pnl"] += p
    set_ = won + lost
    wr = won/max(1,set_)*100
    roi = pnl/max(1,stake)*100
    print(f"{label}: fires={len(rows)}  settled={set_}  W={won} L={lost} WR={wr:.0f}%  "
          f"open={open_}  stake=${stake:.2f}  pnl=${pnl:+.2f}  ROI={roi:.1f}%")
    for a, d in sorted(by_asset.items()):
        if d["w"] + d["l"] == 0 and d["open"] == 0: continue
        wr_a = d["w"]/max(1,d["w"]+d["l"])*100
        print(f"    {a}: W/L={d['w']}/{d['l']}  open={d['open']}  pnl=${d['pnl']:+.2f}  WR={wr_a:.0f}%")
    # Last 5 fires summary
    print(f"    last 5 fires:")
    for r in rows[-5:]:
        s = settlements.get(r["ticker"], {})
        result = (s.get("result") or "?").lower() or "OPEN"
        won_str = "✓W" if r["side"] == result else ("✗L" if result in ("yes","no") else "—")
        print(f"      {r['ts_iso'][11:19]}  {r['asset']:4s} {r['side']:3s} qty={float(r['qty']):>4.1f}  "
              f"stake=${float(r['stake_dollars']):.2f}  result={result}  {won_str}")
    print()
