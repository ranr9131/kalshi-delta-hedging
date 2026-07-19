import sys
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
from dotenv import dotenv_values
import requests, kalshi_auth
from datetime import datetime, timezone
from collections import Counter

env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")
key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
keyid = env["KALSHI_API_KEY_ID"]
BASE = "https://api.elections.kalshi.com"


def get(path, params=None):
    h = kalshi_auth.make_auth_headers(key, keyid, "GET", path)
    return requests.get(BASE + path, headers=h, params=params, timeout=15).json()


bal = get("/trade-api/v2/portfolio/balance").get("balance", 0) / 100.0

ppd = get("/trade-api/v2/portfolio/positions")
open_positions = []
for p in ppd.get("market_positions", []):
    pos = p.get("position", 0)
    if pos != 0:
        open_positions.append((p.get("ticker"), pos, p.get("market_exposure", 0) / 100.0))

START_BAL = 675.62
START_TS = "2026-06-04T20:36"
delta = bal - START_BAL

all_orders = []
cursor = None
for _ in range(20):
    params = {"limit": 200}
    if cursor:
        params["cursor"] = cursor
    r = get("/trade-api/v2/portfolio/orders", params=params)
    all_orders.extend(r.get("orders", []))
    cursor = r.get("cursor")
    if not cursor:
        break

live_orders = [o for o in all_orders if o.get("created_time", "") >= START_TS]
by_status = Counter(o.get("status") for o in live_orders)
total_fills = sum(float(o.get("fill_count_fp", 0) or 0) for o in live_orders)
total_cost = sum(float(o.get("taker_fill_cost_dollars", 0) or 0) +
                 float(o.get("maker_fill_cost_dollars", 0) or 0) for o in live_orders)
total_fees = sum(float(o.get("taker_fees_dollars", 0) or 0) +
                 float(o.get("maker_fees_dollars", 0) or 0) for o in live_orders)

now = datetime.now(timezone.utc)
start = datetime(2026, 6, 4, 20, 36, tzinfo=timezone.utc)
hrs = (now - start).total_seconds() / 3600

now_str = now.strftime("%Y-%m-%d %H:%M")

print("=" * 50)
print("LIVE TRADING SUMMARY")
print("=" * 50)
print("  Started:       2026-06-04 20:36 UTC")
print("  Now:           %s UTC" % now_str)
print("  Duration:      %.1f hours" % hrs)
print()
print("  Start balance: $%.2f" % START_BAL)
print("  Now balance:   $%.2f" % bal)
print("  DELTA:         $%+.2f   (%+.1f%%)" % (delta, delta / START_BAL * 100))
print("  $/hour:        $%+.2f" % (delta / max(1, hrs)))
print("  $/day pace:    $%+.2f" % (delta * 24 / max(1, hrs)))
print()
print("  Total fires:   %d" % len(live_orders))
print("  By status:     %s" % dict(by_status))
print("  Total cost:    $%.2f" % total_cost)
print("  Total fees:    $%.4f" % total_fees)
print("  Total fills:   %.0f contracts" % total_fills)
print()
if open_positions:
    total_open = sum(e for _, _, e in open_positions)
    print("  Open positions: %d markets, $%.2f at risk (will settle within 15 min)" % (
        len(open_positions), total_open))
    for t, pos, exp in open_positions[:5]:
        print("    %s  position=%d  exposure=$%.2f" % (t, pos, exp))
else:
    print("  No open positions right now")
