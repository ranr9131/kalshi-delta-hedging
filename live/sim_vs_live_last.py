"""Compare the most recent settled snipes on live vs paper for matching tickers."""
import csv, os
from datetime import datetime, timezone

ROOT = "/home/ec2-user/kalshi-delta-hedging/live"
settlements = {r["ticker"]: r for r in csv.DictReader(open(ROOT + "/settlements.csv"))}

def load(fname):
    rows = list(csv.DictReader(open(ROOT + "/" + fname)))
    rows.sort(key=lambda r: r["ts_iso"])
    return rows

leo  = load("snipes_leo.csv")
fr   = load("snipes_friend.csv")
v2   = load("snipes_v2.csv")

# Most recent settled LIVE fire on LEO
print("=== most recent settled fires on each strategy ===")
print()

def pnl_for(r, settlement):
    try:
        qty = float(r["qty"]); stake = float(r["stake_dollars"])
    except: return None
    result = (settlement.get("result") or "").lower()
    if result not in ("yes","no"): return None
    if r["side"] == result:
        return qty - stake, "WIN"
    return -stake, "LOSS"

# Most recent 5 settled LEO fires
leo_settled = [r for r in leo if r["ticker"] in settlements
               and (settlements[r["ticker"]].get("result","").lower() in ("yes","no"))]
v2_settled  = [r for r in v2 if r["ticker"] in settlements
               and (settlements[r["ticker"]].get("result","").lower() in ("yes","no"))
               and r.get("mode") == "paper"]

print("LAST 5 SETTLED LEO (live) FIRES:")
for r in leo_settled[-5:]:
    s = settlements[r["ticker"]]
    p, wl = pnl_for(r, s)
    print("  %s  %s  %s @ %s¢  stake=$%.2f  qty=%s  →  %s  result=%s  pnl=$%+.2f" % (
        r["ts_iso"][11:19], r["ticker"][-30:], r["side"],
        r["fill_cents_est"], float(r["stake_dollars"]), r["qty"],
        wl, s["result"].upper(), p))
print()

print("LAST 5 SETTLED V2 (paper) FIRES:")
for r in v2_settled[-5:]:
    s = settlements[r["ticker"]]
    p, wl = pnl_for(r, s)
    print("  %s  %s  %s @ %s¢  stake=$%.2f  qty=%s  →  %s  result=%s  pnl=$%+.2f" % (
        r["ts_iso"][11:19], r["ticker"][-30:], r["side"],
        r["fill_cents_est"], float(r["stake_dollars"]), r["qty"],
        wl, s["result"].upper(), p))
print()

# Find tickers where BOTH live and paper fired (since multi switch ~19:39)
live_tickers = {r["ticker"]: r for r in leo if r["ts_iso"] >= "2026-06-05T19:39"}
paper_tickers = {}
for r in v2:
    if r.get("mode")=="paper" and r["ts_iso"] >= "2026-06-05T19:39":
        paper_tickers[r["ticker"]] = r

overlap = sorted(set(live_tickers.keys()) & set(paper_tickers.keys()))
print("=== OVERLAP: same ticker fired by both live & paper (since 19:39) ===")
print("  total overlapping tickers: %d" % len(overlap))
print()
for t in overlap[-10:]:
    lr = live_tickers[t]
    pr = paper_tickers[t]
    s  = settlements.get(t)
    if not s:
        # Both open
        print("  %s" % t)
        print("    LIVE:   %s @ %s¢  stake=$%.2f  qty=%s  OPEN" % (
            lr["side"], lr["fill_cents_est"], float(lr["stake_dollars"]), lr["qty"]))
        print("    PAPER:  %s @ %s¢  stake=$%.2f  qty=%s  OPEN" % (
            pr["side"], pr["fill_cents_est"], float(pr["stake_dollars"]), pr["qty"]))
        continue
    result = (s.get("result") or "").lower()
    lp, lwl = pnl_for(lr, s)
    pp, pwl = pnl_for(pr, s)
    print("  %s  result=%s" % (t, result.upper()))
    print("    LIVE:   %s @ %s¢  stake=$%.2f  qty=%s  →  %s  pnl=$%+.2f" % (
        lr["side"], lr["fill_cents_est"], float(lr["stake_dollars"]), lr["qty"], lwl, lp))
    print("    PAPER:  %s @ %s¢  stake=$%.2f  qty=%s  →  %s  pnl=$%+.2f" % (
        pr["side"], pr["fill_cents_est"], float(pr["stake_dollars"]), pr["qty"], pwl, pp))
    if lwl != pwl:
        print("    *** DIVERGENCE: live=%s paper=%s ***" % (lwl, pwl))
