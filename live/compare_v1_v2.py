"""Compare V1 LIVE vs V2 PAPER over the same time window."""
import csv
from datetime import datetime, timezone

LIVE_START = "2026-06-04T20:36"

settlements = {r["ticker"]: r for r in csv.DictReader(
    open("/home/ec2-user/kalshi-delta-hedging/live/settlements.csv"))}


def pnl_breakdown(path, mode_filter=None):
    """Returns (overall_dict, per_asset_dict)."""
    by_asset = {}
    overall = {"fires": 0, "settled": 0, "wins": 0, "losses": 0,
               "open": 0, "stake": 0.0, "pnl": 0.0}
    for r in csv.DictReader(open(path)):
        if mode_filter and r.get("mode") != mode_filter:
            continue
        ts = r.get("ts_iso", "")
        if ts < LIVE_START:
            continue
        a = r.get("asset", "?")
        if a not in by_asset:
            by_asset[a] = {"fires": 0, "wins": 0, "losses": 0,
                           "open": 0, "stake": 0.0, "pnl": 0.0,
                           "fills_cents_sum": 0.0, "fills_n": 0}
        try:
            qty = float(r.get("qty") or 0)
            stake = float(r.get("stake_dollars") or 0)
            fc = float(r.get("fill_cents_est") or 0)
        except Exception:
            continue
        overall["fires"] += 1
        by_asset[a]["fires"] += 1
        overall["stake"] += stake
        by_asset[a]["stake"] += stake
        by_asset[a]["fills_cents_sum"] += fc
        by_asset[a]["fills_n"] += 1

        s = settlements.get(r["ticker"])
        if not s:
            overall["open"] += 1
            by_asset[a]["open"] += 1
            continue
        result = (s.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue
        overall["settled"] += 1
        if r["side"] == result:
            p = qty - stake
            overall["wins"] += 1
            overall["pnl"] += p
            by_asset[a]["wins"] += 1
            by_asset[a]["pnl"] += p
        else:
            p = -stake
            overall["losses"] += 1
            overall["pnl"] += p
            by_asset[a]["losses"] += 1
            by_asset[a]["pnl"] += p
    return overall, by_asset


print("=" * 80)
print("V1 LIVE vs V2 PAPER — same time window (since live flip 20:36 UTC Jun 4)")
print("=" * 80)

v1, v1a = pnl_breakdown("/home/ec2-user/kalshi-delta-hedging/live/snipes.csv",
                       mode_filter="live")
v2, v2a = pnl_breakdown("/home/ec2-user/kalshi-delta-hedging/live/snipes_v2.csv",
                       mode_filter="paper")


def fmt(d):
    set_ = d["wins"] + d["losses"]
    wr = d["wins"] / max(1, set_) * 100
    roi = d["pnl"] / max(1, d["stake"]) * 100
    return ("fires=%d  settled=%d  W=%d L=%d  open=%d  "
            "WR=%.0f%%  stake=$%.2f  pnl=$%+.2f  ROI=%.1f%%" % (
        d["fires"], set_, d["wins"], d["losses"], d["open"],
        wr, d["stake"], d["pnl"], roi))


print()
print("V1 LIVE :", fmt(v1))
print("V2 PAPER:", fmt(v2))
print()
print("-" * 80)
print("PER-ASSET HEAD-TO-HEAD")
print("-" * 80)
print()
print("%-6s  %-30s  %-30s  %s" % ("asset", "V1 LIVE", "V2 PAPER", "Δ ROI"))
print("-" * 100)
all_assets = sorted(set(v1a.keys()) | set(v2a.keys()))
for a in all_assets:
    v1d = v1a.get(a, {"fires":0,"wins":0,"losses":0,"stake":0,"pnl":0,"open":0})
    v2d = v2a.get(a, {"fires":0,"wins":0,"losses":0,"stake":0,"pnl":0,"open":0})
    v1_str = "n=%d W=%d L=%d $%+.2f" % (v1d["fires"], v1d["wins"], v1d["losses"], v1d["pnl"])
    v2_str = "n=%d W=%d L=%d $%+.2f" % (v2d["fires"], v2d["wins"], v2d["losses"], v2d["pnl"])
    v1_roi = v1d["pnl"]/max(1,v1d["stake"])*100 if v1d["stake"] > 0 else 0
    v2_roi = v2d["pnl"]/max(1,v2d["stake"])*100 if v2d["stake"] > 0 else 0
    print("%-6s  %-30s  %-30s  V1=%+.1f%%  V2=%+.1f%%" % (
        a, v1_str, v2_str, v1_roi, v2_roi))

# Find tickers both fired on for direct overlap analysis
v1_tickers = set()
v2_tickers = set()
v1_rows = {}
v2_rows = {}
for r in csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/snipes.csv")):
    if r.get("mode") == "live" and r["ts_iso"] >= LIVE_START:
        v1_tickers.add(r["ticker"])
        v1_rows.setdefault(r["ticker"], []).append(r)
for r in csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/snipes_v2.csv")):
    if r["ts_iso"] >= LIVE_START:
        v2_tickers.add(r["ticker"])
        v2_rows.setdefault(r["ticker"], []).append(r)

print()
print("-" * 80)
print("MARKET OVERLAP")
print("-" * 80)
both = v1_tickers & v2_tickers
only_v1 = v1_tickers - v2_tickers
only_v2 = v2_tickers - v1_tickers
print("  fired by BOTH:    %d markets" % len(both))
print("  V1 only:          %d markets" % len(only_v1))
print("  V2 only:          %d markets" % len(only_v2))
print("  → V2 fires more often (more aggressive gates / less filtering)")
