"""Compare V1 LIVE snipes (CSV) vs Kalshi reality.

v2 of the script — uses correct Kalshi v2 field names:
  fill_count_fp, initial_count_fp, remaining_count_fp
  taker_fill_cost_dollars, maker_fill_cost_dollars
  yes_price_dollars, no_price_dollars
All "_fp" / "_dollars" are STRINGS that must be float()-ed.
"""
import sys, csv
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
from dotenv import dotenv_values
import requests
import kalshi_auth
from datetime import datetime, timezone
from collections import Counter

env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")
key = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
keyid = env["KALSHI_API_KEY_ID"]
BASE = "https://api.elections.kalshi.com"
LIVE_START = "2026-06-04T20:36:00"

def _f(v, default=0.0):
    try: return float(v)
    except: return default

rows = list(csv.DictReader(open("/home/ec2-user/kalshi-delta-hedging/live/snipes.csv")))
live = [r for r in rows if r["mode"] == "live" and r["ts_iso"] >= LIVE_START]

def kalshi_get(path, params=None):
    hdr = kalshi_auth.make_auth_headers(key, keyid, "GET", path)
    return requests.get(BASE + path, headers=hdr, params=params, timeout=15).json()

# Pull all orders since live mode
all_orders = []
cursor = None
for _ in range(20):
    params = {"limit": 200}
    if cursor: params["cursor"] = cursor
    d = kalshi_get("/trade-api/v2/portfolio/orders", params=params)
    batch = d.get("orders", [])
    all_orders.extend(batch)
    cursor = d.get("cursor")
    if not cursor or not batch: break

orders_recent = [o for o in all_orders if o.get("created_time", "") >= LIVE_START]
status_counts = Counter(o.get("status") for o in orders_recent)

# Balance now
bp = kalshi_get("/trade-api/v2/portfolio/balance")
bal = bp.get("balance", 0) / 100.0

print("=" * 120)
print("LIVE vs CSV verification  (V1 sniper since %sZ)" % LIVE_START)
print("=" * 120)
print(f"CSV rows in window:       {len(live)}")
print(f"Kalshi orders in window:  {len(orders_recent)}")
print(f"Status distribution:      {dict(status_counts)}")
print(f"Current Kalshi balance:   ${bal:.2f}   (started at $675.62)")
print()
print(f"  {'ts':<19} {'ticker':<28} {'sd':<3} | "
      f"{'csv_lim':>7} {'csv_fill':>8} {'csv_qty':>8} | "
      f"{'kal_status':<10} {'kal_qty':>7} {'kal_fill_$':>10} {'slip_c':>6}")
print("-" * 120)

matched = unmatched = no_fill = 0
total_expected_stake = total_actual_stake = total_fees = 0.0
slip_sum = 0.0; slip_n = 0

for r in live:
    snipe_dt = datetime.fromisoformat(r["ts_iso"].replace("Z","+00:00"))
    candidates = []
    for o in orders_recent:
        if o.get("ticker") != r["ticker"] or o.get("side") != r["side"]: continue
        try:
            o_dt = datetime.fromisoformat(o["created_time"].replace("Z","+00:00"))
        except Exception:
            continue
        if abs((o_dt - snipe_dt).total_seconds()) < 10:
            candidates.append(o)

    csv_lim  = _f(r["limit_cents"])
    csv_fill = _f(r["fill_cents_est"])
    csv_qty  = _f(r["qty"])
    csv_stake = _f(r["stake_dollars"])
    total_expected_stake += csv_stake
    ts = r["ts_iso"][:19]

    if not candidates:
        unmatched += 1
        print(f"  {ts:<19} {r['ticker']:<28} {r['side']:<3} | "
              f"{csv_lim:>7.1f} {csv_fill:>8.1f} {csv_qty:>8.1f} | NOT FOUND on Kalshi")
        continue

    matched += 1
    o = candidates[0]
    status = o.get("status", "?")
    fill_qty = _f(o.get("fill_count_fp", 0))
    initial_qty = _f(o.get("initial_count_fp", 0))
    cost = _f(o.get("taker_fill_cost_dollars", 0)) + _f(o.get("maker_fill_cost_dollars", 0))
    fees = _f(o.get("taker_fees_dollars", 0)) + _f(o.get("maker_fees_dollars", 0))
    total_fees += fees

    if fill_qty <= 0:
        no_fill += 1
        slip_str = "-"
        kal_fill_str = "-"
    else:
        actual_per_contract_dollars = cost / fill_qty
        if r["side"] == "yes":
            actual_cents = actual_per_contract_dollars * 100
        else:
            # Cost is what we paid (NO side; cost = no_price * qty)
            actual_cents = actual_per_contract_dollars * 100
        slip = actual_cents - csv_fill
        slip_sum += slip * fill_qty  # weighted by qty
        slip_n += fill_qty
        total_actual_stake += cost
        slip_str = f"{slip:+.1f}"
        kal_fill_str = f"{actual_cents:.1f}"

    print(f"  {ts:<19} {r['ticker']:<28} {r['side']:<3} | "
          f"{csv_lim:>7.1f} {csv_fill:>8.1f} {csv_qty:>8.1f} | "
          f"{status:<10} {fill_qty:>7.1f} {kal_fill_str:>10} {slip_str:>6}")

print("-" * 120)
print()
print("SUMMARY")
print(f"  matched (CSV → Kalshi order):  {matched}/{len(live)}")
print(f"  unmatched:                     {unmatched}  (should be 0)")
print(f"  placed but no fill:            {no_fill}  (IOC missed the level)")
print(f"  expected total stake:          ${total_expected_stake:.2f}")
print(f"  actual cash spent on fills:    ${total_actual_stake:.2f}")
print(f"  fees paid to Kalshi:           ${total_fees:.4f}")
if slip_n > 0:
    print(f"  qty-weighted avg slippage:     {slip_sum/slip_n:+.2f} cents/contract")
    print(f"                                 (positive = paid MORE than expected)")
print(f"  balance delta:                 ${bal - 675.62:+.2f}")
