"""Diagnose execution gap: compare LIMIT prices vs ACTUAL fill prices."""
import time, base64, json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dotenv import dotenv_values
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

env = dotenv_values("/home/ec2-user/kalshi-delta-hedging/live/.env")
key_pem = env["KALSHI_PRIVATE_KEY"].replace("\\n", "\n").encode()


def sig_hdrs(method, path):
    ts = str(int(time.time() * 1000))
    msg = (ts + method + path).encode()
    k = serialization.load_pem_private_key(key_pem, password=None)
    s = k.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                 salt_length=padding.PSS.DIGEST_LENGTH),
               hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": env["KALSHI_API_KEY_ID"],
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(s).decode()}


# Pull fills
all_fills = []
cursor = None
for page in range(20):
    params = {"limit": 200}
    if cursor:
        params["cursor"] = cursor
    path = "/trade-api/v2/portfolio/fills"
    r = requests.get(f"https://api.elections.kalshi.com{path}",
                     headers=sig_hdrs("GET", path), params=params, timeout=10)
    if not r.ok:
        print(f"Error: {r.status_code} {r.text}")
        break
    d = r.json()
    batch = d.get("fills", [])
    if not batch:
        break
    all_fills.extend(batch)
    cursor = d.get("cursor")
    if not cursor:
        break
    time.sleep(0.2)

print(f"Total fills: {len(all_fills)}")

# Filter to last 2 days
cutoff = datetime.now(timezone.utc) - timedelta(days=2)
recent = []
for f in all_fills:
    ct = f.get("created_time", "")
    if not ct:
        continue
    try:
        ts = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        if ts >= cutoff:
            recent.append(f)
    except Exception:
        continue
print(f"Last 2 days: {len(recent)}")
if not recent:
    exit()

# Group by order_id to handle partial fills
order_fills = defaultdict(list)
for f in recent:
    order_fills[f["order_id"]].append(f)
print(f"Unique orders: {len(order_fills)}")

# Fetch order details for each to get the LIMIT we sent
order_data = {}
print("\nFetching order details...")
for oid in list(order_fills):
    path = f"/trade-api/v2/portfolio/orders/{oid}"
    r = requests.get(f"https://api.elections.kalshi.com{path}",
                     headers=sig_hdrs("GET", path), timeout=10)
    if r.ok:
        order_data[oid] = r.json().get("order", {})
    time.sleep(0.05)
print(f"  fetched {len(order_data)} orders")

# Analyze per order
print(f"\n{'Time':<19} {'Side':<4} {'Ticker':<28} {'Limit':>6} {'AvgFill':>8} {'Slip':>6} {'Filled':>7}")
print("-" * 110)

slips = []
total_stake = 0
total_slip_dollars = 0
buy_orders = 0
for oid, fills in sorted(order_fills.items(), key=lambda kv: kv[1][0]["created_time"]):
    order = order_data.get(oid, {})
    if not fills:
        continue
    side = fills[0]["side"]
    ticker = fills[0]["ticker"][:28]
    ts_str = fills[0]["created_time"][:19]
    # Get LIMIT price from order
    if side == "yes":
        limit_c = order.get("yes_price", 0)
    else:
        limit_c = order.get("no_price", 0)
    if limit_c <= 0:
        continue
    # Weighted avg fill price (in cents)
    total_count = 0
    total_cost_c = 0
    for f in fills:
        cnt = float(f.get("count_fp", "0"))
        if side == "yes":
            p_dollars = float(f.get("yes_price_dollars", "0"))
        else:
            p_dollars = float(f.get("no_price_dollars", "0"))
        total_count += cnt
        total_cost_c += cnt * p_dollars * 100
    if total_count <= 0:
        continue
    avg_fill_c = total_cost_c / total_count
    slip = avg_fill_c - limit_c
    stake = total_cost_c / 100
    total_stake += stake
    total_slip_dollars += (slip / 100) * total_count
    slips.append(slip)
    buy_orders += 1
    print(f"{ts_str:<19} {side:<4} {ticker:<28} {limit_c:>4}¢  {avg_fill_c:>5.1f}¢  {slip:>+5.1f}¢  {total_count:>6.1f}")

print(f"\n{'='*110}")
print(f"Orders analyzed: {buy_orders}")
print(f"Total $$ wagered: ${total_stake:.2f}")
print(f"Total slippage $$ (paid above limit): ${total_slip_dollars:.2f}")
if slips:
    avg = sum(slips) / len(slips)
    print(f"Avg slippage per order: {avg:+.2f}¢")
    over = sum(1 for s in slips if s > 0.5)
    eq = sum(1 for s in slips if -0.5 <= s <= 0.5)
    under = sum(1 for s in slips if s < -0.5)
    print(f"Orders filled ABOVE limit (slip > 0.5¢): {over}")
    print(f"Orders filled AT limit (±0.5¢):           {eq}")
    print(f"Orders filled BELOW limit (slip < -0.5¢): {under}")
    s2 = sorted(slips)
    n = len(s2)
    print(f"\nSlippage percentiles:")
    print(f"  min:    {s2[0]:+.2f}¢")
    print(f"  25%:    {s2[n//4]:+.2f}¢")
    print(f"  median: {s2[n//2]:+.2f}¢")
    print(f"  75%:    {s2[3*n//4]:+.2f}¢")
    print(f"  max:    {s2[-1]:+.2f}¢")
