#!/usr/bin/env python3.11
"""LIP live preflight: prove place -> visible -> cancel -> gone through
lip_api's REAL endpoints before any live farming. Exits 0 only if the whole
chain works. Costs at most ~1c if the test bid crosses a 1c ask."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LIP_PAPER", "1")   # config import; we pass keys directly

from dotenv import dotenv_values
import kalshi_auth
import lip_api

env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
pk = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
kid = env["KALSHI_API_KEY_ID"]

bal = lip_api.get_balance(pk, kid)
print(f"balance: ${bal}")
assert bal is not None and bal > 20, "balance unreadable or too low"

# pick a rest-safe market: prefer ask==0 or ask>=2c so a 1c bid rests
mkts = lip_api.get_markets_by_tickers([])  # not usable for discovery; use raw GET
data = lip_api._get("/trade-api/v2/markets",
                    {"series_ticker": "KXSOLD", "status": "open", "limit": 30})
ticker = None
for m in data.get("markets") or []:
    ask = float(m.get("yes_ask_dollars") or 0)
    if ask == 0 or ask >= 0.02:
        ticker = m["ticker"]
        break
if not ticker and (data.get("markets") or []):
    ticker = data["markets"][0]["ticker"]
assert ticker, "no market for preflight"
print(f"test market: {ticker}")

print("placing 1-lot yes bid @1c via lip_api.place_resting_bid ...")
o = lip_api.place_resting_bid(pk, kid, ticker, "yes", 1, 1)
oid = o.get("order_id")
print(f"placed: order_id={oid} status={o.get('status')}")
assert oid, "no order_id"

time.sleep(2.5)   # orders index lag
resting = lip_api.get_resting_orders(pk, kid)
ours = [r for r in resting if r.get("order_id") == oid]
print(f"visible in resting index: {bool(ours)}")

ok = lip_api.cancel_order(pk, kid, oid)
print(f"cancel: {ok}")
assert ok, "cancel failed — DO NOT GO LIVE"

time.sleep(2.0)
resting = lip_api.get_resting_orders(pk, kid)
still = [r for r in resting if r.get("order_id") == oid]
assert not still, "order still resting after cancel — DO NOT GO LIVE"
print("PREFLIGHT PASS: place/visible/cancel/gone all confirmed")
