"""
Live copytrade monitor for @sharky6999 (paper-only — places NO orders).

Evidence so far (2026-06-10):
  * His settled fills are +EV: copytrade sim = +$15.4k/6.2d at slip 0,
    +$7.9k at 1c slip, ~$0 at 2c  (touch_copytrade_sim.py).
  * Empirical copy-slippage from our own book snapshots: median +1.0c if you
    copy within ~10 min, +2.0c after 1h  (join of his fills vs
    touch_shadow_snapshots.csv).
This monitor closes the remaining unknowns with zero risk:
  1. SIGNAL LATENCY — how many seconds after his on-chain fill does the trade
     appear on data-api /activity?  (logged as `lag_s`)
  2. COPYABLE SIZE — at the moment we see his fill, how many shares rest at
     ≤ his_price + SLIP_BUDGET on the same book?  (logged as `avail@+1c`)
  3. PAPER P&L — what would our copy (capped size, +1c limit) have done?

Exit mirroring: paper positions are opened on his copyable BUYs (filled at
his_price + SLIP_BUDGET) and closed when he SELLs the same market (filled at
his_price - SLIP_BUDGET, capped by bid-side depth within budget and by our
held size). Realized P&L per exit goes in `paper_pnl_usd`; open positions
persist in copy_positions.json. Fills he makes while the monitor is down
(>1h old on restart) are skipped on both sides, so positions can strand —
check --report for open inventory. Settlement redemptions (non-TRADE) are
not tracked; positions he holds to settlement stay open here.

Run:  python3 copy_monitor.py                 # poll every 20s, log forever
      python3 copy_monitor.py --report        # summarize copy_monitor.csv
Output: copy_monitor.csv (one row per copyable fill-event)
"""
from __future__ import annotations

import os
import csv
import time
import json
import argparse
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "copy_monitor.csv")
SEEN = os.path.join(ROOT, "copy_monitor_seen.json")
POS = os.path.join(ROOT, "copy_positions.json")

DATA = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"

WALLET = "0x751a2b86cab503496efd325c8344e10159349ea1"
SLIP_BUDGET_C = 1.0      # max cents above his fill we'd pay
COPY_FRACTION = 0.15     # of his size
MIN_HIS_NOTIONAL = 50.0  # ignore dust fills < $50
POLL_S = 20

COLS = ["ts_seen", "ts_fill", "lag_s", "slug", "outcome", "side", "his_price_c",
        "his_size", "book_best_ask_c", "book_best_bid_c", "avail_within_budget",
        "copy_size", "copy_limit_c", "copyable", "paper_pos", "paper_avg_cost_c",
        "paper_pnl_usd"]


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def token_for(slug, outcome):
    """conditionId->clobTokenIds lookup via gamma, cached."""
    try:
        r = requests.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=10).json()
        if isinstance(r, list) and r:
            m = r[0]
            toks = json.loads(m.get("clobTokenIds") or "[]")
            outs = json.loads(m.get("outcomes") or '["Yes","No"]')
            for t, o in zip(toks, outs):
                if o == outcome:
                    return t
    except Exception:
        pass
    return None


def book_depth(token, max_price_c):
    """(best_ask_c, shares available at <= max_price_c) on the ask side."""
    try:
        b = requests.get(f"{CLOB}/book", params={"token_id": token}, timeout=10).json()
        asks = b.get("asks") or []
        if not asks:
            return None, 0.0
        best = min(_f(a["price"]) for a in asks) * 100
        avail = sum(_f(a["size"]) for a in asks if _f(a["price"]) * 100 <= max_price_c + 1e-9)
        return round(best, 1), round(avail, 1)
    except Exception:
        return None, 0.0


def book_depth_bids(token, min_price_c):
    """(best_bid_c, shares bid at >= min_price_c) — what we could sell into."""
    try:
        b = requests.get(f"{CLOB}/book", params={"token_id": token}, timeout=10).json()
        bids = b.get("bids") or []
        if not bids:
            return None, 0.0
        best = max(_f(x["price"]) for x in bids) * 100
        avail = sum(_f(x["size"]) for x in bids if _f(x["price"]) * 100 >= min_price_c - 1e-9)
        return round(best, 1), round(avail, 1)
    except Exception:
        return None, 0.0


def load_positions():
    if os.path.exists(POS):
        try:
            return json.load(open(POS))
        except Exception:
            pass
    return {}


def save_positions(positions):
    try:
        json.dump(positions, open(POS, "w"), indent=1)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--poll", type=int, default=POLL_S)
    args = ap.parse_args()

    if args.report:
        report()
        return

    seen = set()
    if os.path.exists(SEEN):
        try:
            seen = set(json.load(open(SEEN)))
        except Exception:
            pass
    if not os.path.exists(CSV):
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(COLS)
    print(f"watching {WALLET[:10]}… poll={args.poll}s  budget=+{SLIP_BUDGET_C}c  "
          f"fraction={COPY_FRACTION}  (paper only, exit-mirroring on)")

    tok_cache = {}
    positions = load_positions()
    while True:
        try:
            acts = requests.get(f"{DATA}/activity",
                                params={"user": WALLET, "limit": 100}, timeout=15).json()
        except Exception as e:
            print("poll err:", e)
            time.sleep(args.poll)
            continue
        now = time.time()
        new = []
        for a in acts if isinstance(acts, list) else []:
            if not isinstance(a, dict) or a.get("type") != "TRADE":
                continue
            key = a.get("transactionHash") or f'{a.get("timestamp")}/{a.get("slug")}/{a.get("size")}'
            if key in seen:
                continue
            seen.add(key)
            new.append(a)
        for a in sorted(new, key=lambda x: x.get("timestamp", 0)):
            tfill = a.get("timestamp", 0)
            if now - tfill > 3600:      # backfill on first loop — log but mark stale
                continue
            his_p = _f(a.get("price")) * 100
            his_sz = _f(a.get("size"))
            if his_p * his_sz / 100 < MIN_HIS_NOTIONAL:
                continue
            slug, outcome, side = a.get("slug", ""), a.get("outcome", ""), a.get("side", "")
            row = {"ts_seen": round(now, 1), "ts_fill": tfill,
                   "lag_s": round(now - tfill, 1), "slug": slug, "outcome": outcome,
                   "side": side, "his_price_c": round(his_p, 1), "his_size": his_sz,
                   "book_best_ask_c": "", "book_best_bid_c": "",
                   "avail_within_budget": "", "copy_size": "",
                   "copy_limit_c": "", "copyable": 0,
                   "paper_pos": "", "paper_avg_cost_c": "", "paper_pnl_usd": ""}
            pkey = f"{slug}|{outcome}"
            if side == "BUY":
                tok = tok_cache.get((slug, outcome)) or token_for(slug, outcome)
                if tok:
                    tok_cache[(slug, outcome)] = tok
                    limit = his_p + SLIP_BUDGET_C
                    best, avail = book_depth(tok, limit)
                    copy_sz = min(his_sz * COPY_FRACTION, avail)
                    row.update(book_best_ask_c=best if best is not None else "",
                               avail_within_budget=avail,
                               copy_size=round(copy_sz, 1),
                               copy_limit_c=round(limit, 1),
                               copyable=1 if copy_sz > 0 else 0)
                    if copy_sz > 0:
                        # paper fill at the slip-budget limit (conservative)
                        pos = positions.get(pkey, {"size": 0.0, "cost_c": 0.0})
                        new_size = pos["size"] + copy_sz
                        pos["cost_c"] = (pos["cost_c"] * pos["size"] + limit * copy_sz) / new_size
                        pos["size"] = new_size
                        positions[pkey] = pos
                        row.update(paper_pos=round(new_size, 1),
                                   paper_avg_cost_c=round(pos["cost_c"], 2))
            elif side == "SELL":
                pos = positions.get(pkey)
                tok = tok_cache.get((slug, outcome)) or token_for(slug, outcome)
                if tok:
                    tok_cache[(slug, outcome)] = tok
                    limit = his_p - SLIP_BUDGET_C
                    best_bid, bid_avail = book_depth_bids(tok, limit)
                    row.update(book_best_bid_c=best_bid if best_bid is not None else "",
                               avail_within_budget=bid_avail,
                               copy_limit_c=round(limit, 1))
                    if pos and pos["size"] > 0:
                        exit_sz = min(pos["size"], his_sz * COPY_FRACTION, bid_avail)
                        if exit_sz > 0:
                            # paper exit at the slip-budget limit (conservative)
                            pnl = exit_sz * (limit - pos["cost_c"]) / 100.0
                            pos["size"] = round(pos["size"] - exit_sz, 4)
                            if pos["size"] <= 0:
                                positions.pop(pkey, None)
                            row.update(copy_size=round(exit_sz, 1), copyable=1,
                                       paper_pos=round(max(pos["size"], 0.0), 1),
                                       paper_avg_cost_c=round(pos["cost_c"], 2),
                                       paper_pnl_usd=round(pnl, 2))
            with open(CSV, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in COLS])
            save_positions(positions)
            pnl_str = f' pnl=${row["paper_pnl_usd"]}' if row["paper_pnl_usd"] != "" else ""
            print(f'{time.strftime("%H:%M:%S")} fill lag={row["lag_s"]:6.1f}s '
                  f'{side:4s} {outcome:3s} {slug[:44]:44s} @{his_p:5.1f}c x{his_sz:8.0f} '
                  f'-> avail@±{SLIP_BUDGET_C:.0f}c={row["avail_within_budget"]} '
                  f'copyable={row["copyable"]}{pnl_str}')
        try:
            json.dump(list(seen)[-5000:], open(SEEN, "w"))
        except Exception:
            pass
        time.sleep(args.poll)


def report():
    if not os.path.exists(CSV):
        print("no data yet")
        return
    rows = list(csv.DictReader(open(CSV)))
    fills = [r for r in rows if r["side"] == "BUY"]
    if not fills:
        print(f"{len(rows)} rows, no BUY fills yet")
        return
    lags = sorted(_f(r["lag_s"]) for r in fills)
    cop = [r for r in fills if r["copyable"] == "1"]
    print(f"BUY fills seen: {len(fills)}  copyable at +{SLIP_BUDGET_C}c: {len(cop)} "
          f"({100*len(cop)/len(fills):.0f}%)")
    print(f"signal lag: median {lags[len(lags)//2]:.0f}s  p90 {lags[int(.9*len(lags))]:.0f}s")
    if cop:
        fr = [min(_f(r["avail_within_budget"]) / _f(r["his_size"]), 1.0) for r in cop
              if _f(r["his_size"]) > 0]
        fr.sort()
        print(f"copyable size as fraction of his: median {fr[len(fr)//2]:.2f}  "
              f"p25 {fr[len(fr)//4]:.2f}")
    sells = [r for r in rows if r["side"] == "SELL"]
    ex = [r for r in sells if r.get("copyable") == "1"]
    if sells:
        print(f"SELL fills seen: {len(sells)}  exits mirrored: {len(ex)}")
    realized = sum(_f(r.get("paper_pnl_usd")) for r in ex)
    if ex:
        print(f"realized paper P&L (mirrored exits): ${realized:+.2f}")
    pos = load_positions()
    if pos:
        open_cost = sum(p["size"] * p["cost_c"] / 100.0 for p in pos.values())
        print(f"open paper positions: {len(pos)}  (cost basis ${open_cost:.2f})")
        for k, p in sorted(pos.items()):
            print(f"  {k:60s} {p['size']:>8.1f} @ {p['cost_c']:.1f}c")


if __name__ == "__main__":
    main()
