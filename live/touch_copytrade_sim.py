"""
Copytrade feasibility simulator for @sharky6999 (or any Polymarket wallet).

The replication backtests showed his *model* has no edge — but his *actual*
trades are +EV (93% win, +$8.6k/wk settled). So the edge is his SELECTION, and
copytrading rents it directly (his holds are multi-day → no speed needed).

The one thing that can kill copytrading is SLIPPAGE: his touch-market edge is
only 3-5c, so if we fill a couple cents worse than he did (his own fill + other
copiers move the book), the edge evaporates. This replays his real fills,
charges `slip` cents on every buy (we pay more) and sell (we get less), holds to
ACTUAL resolution, and sweeps slip to find the breakeven.

PnL per market = Σ trades[ BUY: -size*(price+slip) | SELL: +size*(price-slip) ]
               + Σ redeems[ +size ]  + remaining_shares * (1 if our outcome won)

Run:  python3 touch_copytrade_sim.py
      python3 touch_copytrade_sim.py --wallet 0x... --slips 0,0.5,1,2,3
"""
from __future__ import annotations

import sys
import os
import json
import time
import argparse

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from touch_shadow_logger import market_resolution

DATA = "https://data-api.polymarket.com"
WALLET = "0x751a2b86cab503496efd325c8344e10159349ea1"


def fetch_activity(wallet, pages=8):
    recs = []
    for off in range(0, pages * 500, 500):
        try:
            r = requests.get(f"{DATA}/activity", params={
                "user": wallet, "limit": 500, "offset": off,
                "sortBy": "TIMESTAMP", "sortDirection": "DESC"}, timeout=20)
            r.raise_for_status()
            chunk = r.json()
        except Exception:
            break
        if not chunk:
            break
        recs += chunk
    # dedup
    seen, out = set(), []
    for x in recs:
        k = (x.get("transactionHash"), x.get("timestamp"), x.get("asset"),
             x.get("size"), x.get("type"))
        if k in seen:
            continue
        seen.add(k); out.append(x)
    return out


def build_books(recs):
    """token -> {trades:[(side,size,price,ts)], redeem, cond, out, title}"""
    books = {}
    for x in recs:
        if x.get("type") == "TRADE":
            t = x["asset"]
            b = books.setdefault(t, {"trades": [], "redeem": 0.0,
                                     "cond": x.get("conditionId"),
                                     "out": x.get("outcome"), "title": x.get("title", "")})
            b["trades"].append((x["side"], x["size"], x.get("price", 0.0), x["timestamp"]))
        elif x.get("type") == "REDEEM":
            t = x["asset"]
            if t in books:
                books[t]["redeem"] += x.get("size", 0.0)
    return books


def market_pnl(b, win, slip):
    """settled copy-PnL for one market at slippage `slip` (price units, e.g. 0.01=1c)."""
    cash = 0.0; shares = 0.0
    for side, size, price, _ in b["trades"]:
        if side == "BUY":
            cash -= size * (price + slip); shares += size
        else:
            cash += size * max(0.0, price - slip); shares -= size
    cash += b["redeem"]; shares -= b["redeem"]
    settle = max(0.0, shares) * (1.0 if b["out"] == win else 0.0)
    return cash + settle


def simulate(wallet, slips):
    recs = fetch_activity(wallet)
    tr = [x for x in recs if x.get("type") == "TRADE"]
    ts = [x["timestamp"] for x in tr]
    span_d = (max(ts) - min(ts)) / 86400 if ts else 0
    books = build_books(recs)
    print(f"wallet {wallet[:10]}…  {len(tr)} trades / {len(books)} markets / ~{span_d:.1f}d")

    # resolve once (cached)
    cache = {}
    resolved = []
    for t, b in books.items():
        win = market_resolution(b["cond"], cache) if b["cond"] else None
        if win is not None:
            b["win"] = win
            last_ts = max(tt[3] for tt in b["trades"])
            resolved.append((last_ts, b))
    resolved.sort(key=lambda x: x[0])
    print(f"resolved markets: {len(resolved)} (of {len(books)})\n")

    print(f"{'slip':>6} {'copyPnL':>10} {'winRate':>8} {'avg/mkt':>8} {'maxDD':>9} {'capDeployed':>12}")
    print("-" * 60)
    for slip in slips:
        s = slip / 100.0
        pnls = []
        cap = 0.0
        for _, b in resolved:
            cap += sum(sz * (pr + s) for sd, sz, pr, _ in b["trades"] if sd == "BUY")
            pnls.append(market_pnl(b, b["win"], s))
        tot = sum(pnls)
        wins = sum(1 for p in pnls if p > 0.5)
        # equity curve / max drawdown (markets in resolution order)
        eq = 0.0; peak = 0.0; dd = 0.0
        for p in pnls:
            eq += p; peak = max(peak, eq); dd = min(dd, eq - peak)
        print(f"{slip:>5.1f}c ${tot:>9,.0f} {100*wins/max(1,len(pnls)):>7.0f}% "
              f"${tot/max(1,len(pnls)):>7,.1f} ${dd:>8,.0f} ${cap:>11,.0f}")
    print("\nslip = cents worse than his fill on every buy AND sell (the copy-cost).")
    print("Find where copyPnL crosses zero — that's the max slippage the edge tolerates.")
    print("maxDD = worst peak-to-trough on the settled equity curve (the pain you'd inherit).")
    print("capDeployed = total buy notional (capital recycled across the window).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", default=WALLET)
    ap.add_argument("--slips", default="0,0.5,1,2,3")
    args = ap.parse_args()
    simulate(args.wallet, [float(x) for x in args.slips.split(",")])


if __name__ == "__main__":
    main()
