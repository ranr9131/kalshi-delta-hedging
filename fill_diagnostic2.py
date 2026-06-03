"""Compare trader log LIMITS to actual Kalshi FILLS.

Parses trader log lines like:
  -> BET NO  $10.32 @ 0.450 (45.0c/contract) | 23 contracts
     NO order placed: e2aa5cc8-...  (filled $10.32)

Then queries Kalshi for actual fill prices.
"""
import re, time, base64, json
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


# Parse trader logs
log_files = [
    "/home/ec2-user/kalshi-delta-hedging/live/trader-nn.log",
    "/home/ec2-user/kalshi-delta-hedging/live/trader-nn-eth.log",
    "/home/ec2-user/kalshi-delta-hedging/live/trader-nn-sol.log",
    "/home/ec2-user/kalshi-delta-hedging/live/trader-nn-xrp.log",
]

# Pattern: -> BET NO $10.32 @ 0.450 (45.0c/contract) | 23 contracts
bet_pat = re.compile(r"BET (YES|NO)\s+\$?([\d.]+)\s+@\s+([\d.]+)\s+\(([\d.]+)c/contract\)\s+\|\s+(\d+)\s+contracts")
order_pat = re.compile(r"(YES|NO|RH-YES|RH-NO) order placed: ([a-f0-9-]+)")

# We need to match BET lines with subsequent order_id lines
bet_entries = []
for lf in log_files:
    asset = lf.split("trader-nn")[-1].replace(".log","").replace("-","") or "btc"
    with open(lf) as f:
        lines = f.readlines()
    last_bet = None
    for line in lines:
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            continue
        ts = ts_match.group(1)
        bet_m = bet_pat.search(line)
        if bet_m:
            side, stake, limit_dollars, limit_cents, count = bet_m.groups()
            last_bet = {
                "asset": asset, "ts": ts, "side": side.lower(),
                "limit_dollars": float(limit_dollars),
                "limit_cents": float(limit_cents),
                "expected_count": int(count),
                "expected_stake": float(stake),
            }
        order_m = order_pat.search(line)
        if order_m and last_bet:
            kind, order_id = order_m.groups()
            last_bet["order_id"] = order_id
            last_bet["order_kind"] = kind
            bet_entries.append(last_bet)
            last_bet = None

print(f"Parsed bet+order pairs: {len(bet_entries)}")

# Filter to last 2 days
cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
recent = [b for b in bet_entries if b["ts"] >= cutoff]
print(f"Last 2 days: {len(recent)}")

# Fetch Kalshi orders to get actual fills
print("\nFetching actual fills from Kalshi...")
slippages = []
underfills = []
print(f"\n{'Time':<20} {'Asset':<5} {'Side':<6} {'OurLimit':>10} {'KalshiFill':>11} {'Slip':>6} {'OurCount':>9} {'ActualCount':>12}")
print("-" * 110)
for b in sorted(recent, key=lambda x: x["ts"]):
    oid = b.get("order_id")
    if not oid: continue
    path = f"/trade-api/v2/portfolio/orders/{oid}"
    r = requests.get(f"https://api.elections.kalshi.com{path}",
                     headers=sig_hdrs("GET", path), timeout=10)
    if not r.ok: continue
    order = r.json().get("order", {})
    fill_count = float(order.get("fill_count_fp", "0"))
    fill_cost = float(order.get("taker_fill_cost_dollars", "0"))
    if fill_count <= 0:
        # Order didn't fill
        continue
    avg_fill_cents = (fill_cost / fill_count) * 100
    slip = avg_fill_cents - b["limit_cents"]
    slippages.append(slip)
    if fill_count < b["expected_count"]:
        underfills.append((b, fill_count))
    print(f"{b['ts']:<20} {b['asset']:<5} {b['side']:<6} {b['limit_cents']:>8.1f}¢ {avg_fill_cents:>9.1f}¢ {slip:>+5.1f}¢ {b['expected_count']:>9d} {fill_count:>11.0f}")
    time.sleep(0.05)

if slippages:
    print(f"\n{'='*110}")
    avg_slip = sum(slippages) / len(slippages)
    over = sum(1 for s in slippages if s > 0.5)
    eq = sum(1 for s in slippages if -0.5 <= s <= 0.5)
    under = sum(1 for s in slippages if s < -0.5)
    print(f"Orders: {len(slippages)}")
    print(f"Avg slippage: {avg_slip:+.2f}¢ per contract")
    print(f"")
    print(f"Filled ABOVE our limit (slip > 0.5¢): {over}")
    print(f"Filled AT our limit (±0.5¢):           {eq}")
    print(f"Filled BELOW our limit (slip < -0.5¢): {under}")
    print(f"")
    print(f"Total $$ slippage = avg_slip × total_contracts:")
    print(f"  {sum(slippages)/100:.2f}¢ × 1 contract = ${sum(slippages)/100/100:.2f}")
    s2 = sorted(slippages); n = len(s2)
    print(f"\nSlippage percentiles:")
    print(f"  min:    {s2[0]:+.1f}¢")
    print(f"  25%:    {s2[n//4]:+.1f}¢")
    print(f"  median: {s2[n//2]:+.1f}¢")
    print(f"  75%:    {s2[3*n//4]:+.1f}¢")
    print(f"  max:    {s2[-1]:+.1f}¢")
    if underfills:
        print(f"\nPartial fills: {len(underfills)} orders got fewer contracts than expected")
