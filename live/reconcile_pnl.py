"""
True PnL reconciliation.

Pulls /portfolio/fills for each account (real prices + actual fees Kalshi
charged) and joins them with settlements.csv to compute wallet-truth PnL.

This replaces the CSV-based estimate, which was overstating PnL because:
  - `fill_cents_est` in snipes_<acct>.csv is the price we EXPECTED to walk
    the ladder to, not the price we actually got.  Real fills are often
    a cent or two worse.
  - Kalshi charges per-contract fees (0.07 × P × (1-P) × N).  CSV ignored them.
  - Some fires fail / partially fill — CSV records full intended qty.

Usage:
  python3.11 reconcile_pnl.py                  # since session start
  python3.11 reconcile_pnl.py --since 08:05    # since HH:MM UTC today
  python3.11 reconcile_pnl.py --since 2026-06-06T08:05:00
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
from dotenv import dotenv_values

import kalshi_trade
import kalshi_auth

BASE = os.path.dirname(os.path.abspath(__file__))
ENV  = dotenv_values(os.path.join(BASE, ".env"))
SETTLE_PATH = os.path.join(BASE, "settlements.csv")


def fee_for(yes_price: float, count: float) -> float:
    """Kalshi fee formula: 0.07 × yes_price × (1 - yes_price) × N."""
    return 0.07 * yes_price * (1.0 - yes_price) * count


def pull_fills(label: str, since_ts: float) -> list:
    """Pull all fills for an account since `since_ts` (unix seconds)."""
    kid = ENV.get(f"{label}_KEY_ID")
    pem = (ENV.get(f"{label}_PRIVATE_KEY") or "").replace("\\n", "\n")
    if not (kid and pem):
        print(f"[{label}] missing credentials in .env", file=sys.stderr)
        return []
    pk = kalshi_auth.load_private_key(pem)
    fills = []
    cursor = None
    pages = 0
    while True:
        pages += 1
        path = "/trade-api/v2/portfolio/fills"
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        headers = kalshi_auth.make_auth_headers(pk, kid, "GET", path)
        headers["Accept"] = "application/json"
        r = requests.get(kalshi_trade.BASE_URL + path,
                         params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"[{label}] page {pages}: HTTP {r.status_code} {r.text[:200]}",
                  file=sys.stderr)
            break
        data = r.json()
        page_fills = data.get("fills", [])
        # Stop once we cross since_ts (fills come newest-first)
        keep_going = True
        for f in page_fills:
            if int(f.get("ts", 0)) < since_ts:
                keep_going = False
                break
            fills.append(f)
        if not keep_going:
            break
        cursor = data.get("cursor")
        if not cursor or not page_fills:
            break
        if pages > 50:  # hard safety
            print(f"[{label}] hit 50-page limit", file=sys.stderr)
            break
    return fills


def load_settlements() -> dict:
    out = {}
    if not os.path.exists(SETTLE_PATH):
        return out
    with open(SETTLE_PATH) as f:
        for r in csv.DictReader(f):
            t = (r.get("ticker") or "").strip()
            res = (r.get("result") or "").strip().lower()
            if t and res in ("yes", "no"):
                out[t] = res
    return out


def reconcile(label: str, since_ts: float, settle: dict) -> dict:
    fills = pull_fills(label, since_ts)
    n_fills = len(fills)
    by_status = {"settled_win": 0, "settled_loss": 0, "open": 0}
    by_asset_side = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])
    # ^ [n_fills, n_wins, total_qty, total_realized_pnl, total_fees]
    total_realized = 0.0
    total_fees = 0.0
    total_open_cost = 0.0  # cost paid for contracts not yet settled

    for f in fills:
        ticker = f.get("market_ticker") or f.get("ticker", "")
        side   = (f.get("side") or "").lower()
        count  = float(f.get("count_fp") or f.get("count") or 0)
        yes_px = float(f.get("yes_price_dollars") or 0)
        # Cost per contract on OUR side
        our_px = yes_px if side == "yes" else (1.0 - yes_px)
        cost   = our_px * count
        fee    = float(f.get("fee_cost") or fee_for(yes_px, count))

        # Asset = strip the ticker suffix
        # e.g. "KXSOL15M-26JUN060515-15" -> "SOL"
        asset = ""
        for prefix, sym in (("KXBTC", "BTC"), ("KXETH", "ETH"), ("KXSOL", "SOL"),
                            ("KXXRP", "XRP"), ("KXHYPE", "HYPE"), ("KXBNB", "BNB"),
                            ("KXDOGE", "DOGE"), ("KXTON", "TON"), ("KXADA", "ADA")):
            if ticker.startswith(prefix):
                asset = sym; break

        key = (asset, side)
        by_asset_side[key][0] += 1
        by_asset_side[key][2] += count
        by_asset_side[key][4] += fee
        total_fees += fee

        res = settle.get(ticker)
        if res not in ("yes", "no"):
            by_status["open"] += 1
            total_open_cost += cost
            continue

        if side == res:
            # Win: each contract pays $1, we already paid `our_px` per contract
            pnl = count * (1.0 - our_px) - fee
            by_status["settled_win"] += 1
            by_asset_side[key][1] += 1
        else:
            # Loss: each contract → 0, we paid `our_px` × count + fee
            pnl = -count * our_px - fee
            by_status["settled_loss"] += 1
        total_realized += pnl
        by_asset_side[key][3] += pnl

    return {
        "label": label,
        "n_fills": n_fills,
        "by_status": by_status,
        "total_realized": total_realized,
        "total_fees": total_fees,
        "total_open_cost": total_open_cost,
        "by_asset_side": dict(by_asset_side),
    }


def fmt_dollars(d): return f"${d:+8.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-06-06T05:20:00",
                    help="ISO timestamp or HH:MM today (UTC)")
    args = ap.parse_args()

    since = args.since.strip()
    if "T" not in since:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        since = f"{today}T{since}:00"
    since_ts = datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp()

    settle = load_settlements()
    print(f"=== reconcile since {since} (UTC) ===")
    print(f"(settlements joined: {len(settle)} known tickers)\n")

    grand = {"realized": 0.0, "fees": 0.0, "open_cost": 0.0, "n": 0,
             "asset_side": defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])}

    for label in ("LEO", "FRIEND"):
        r = reconcile(label, since_ts, settle)
        s = r["by_status"]
        print(f"-- {label} --")
        print(f"  fills:            {r['n_fills']}")
        print(f"    settled wins:   {s['settled_win']}")
        print(f"    settled losses: {s['settled_loss']}")
        print(f"    open:           {s['open']}")
        print(f"  realized PnL:     {fmt_dollars(r['total_realized'])}")
        print(f"  fees paid:        {fmt_dollars(-r['total_fees'])}  (already inside realized)")
        print(f"  open cost basis:  ${r['total_open_cost']:.2f}  (could go either way)")
        print()
        grand["realized"]  += r["total_realized"]
        grand["fees"]      += r["total_fees"]
        grand["open_cost"] += r["total_open_cost"]
        grand["n"]         += r["n_fills"]
        for k, v in r["by_asset_side"].items():
            for i in range(5):
                grand["asset_side"][k][i] += v[i]

    print("=== combined ===")
    print(f"  total fills:        {grand['n']}")
    print(f"  realized PnL:       {fmt_dollars(grand['realized'])}")
    print(f"  fees paid:          {fmt_dollars(-grand['fees'])}  ({grand['fees'] / max(grand['n'], 1) * 100:.2f}¢/fill avg)")
    print(f"  open cost basis:    ${grand['open_cost']:.2f}")
    print()
    print("  per (asset,side):")
    print(f"  {'asset/side':<12} {'fills':>6} {'wins':>6} {'qty':>7}  {'realized':>10}  {'fees':>8}")
    for k in sorted(grand["asset_side"], key=lambda x: -grand["asset_side"][x][3]):
        a, s = k
        n_, w_, q_, p_, fe_ = grand["asset_side"][k]
        wr = f"{w_ / n_ * 100:.0f}%" if n_ else "--"
        print(f"  {a + '/' + s.upper():<12} {n_:>6} {w_:>6} {q_:>7.0f}  {fmt_dollars(p_):>10}  {fmt_dollars(-fe_):>8}")


if __name__ == "__main__":
    main()
