"""
Backfill actual executed fills into a trade log.

The live trader records `actual_fill` best-effort at submit time, but Kalshi's
/portfolio/fills endpoint can lag the order by a moment, so some rows land blank.
This script rereads a trade log, and for every row with a real order_id but a
blank actual_fill, pulls the true executed VWAP/count from /fills and rewrites
the CSV in place. Safe to run repeatedly.

Usage:
    python reconcile_fills.py [trade_log.csv]
"""

import csv
import os
import sys

from dotenv import dotenv_values

import kalshi_auth
import kalshi_trade

_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))
PRIVATE_KEY = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")

SKIP_IDS = {"", "none", "paper"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_dir, "trade_log.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if "actual_fill" not in fields:
        print(f"{path} has no actual_fill column (old schema) — nothing to reconcile.")
        return

    updated = 0
    for r in rows:
        oid = (r.get("order_id") or "").strip()
        if oid in SKIP_IDS:
            continue
        if r.get("order_result") != "ok":
            continue
        if (r.get("actual_fill") or "").strip():
            continue  # already have it
        fill = kalshi_trade.get_order_fills(PRIVATE_KEY, API_KEY_ID, oid)
        if fill["filled_count"] > 0:
            r["actual_fill"] = round(fill["vwap"], 4) if fill["vwap"] is not None else ""
            r["filled_count"] = fill["filled_count"]
            r["n_fills"] = fill["n_fills"]
            updated += 1
            print(f"  {oid[:8]}  {r.get('bet_side'):>3}  intended={r.get('fill_price')}  actual={r['actual_fill']}  n={fill['n_fills']}")

    if updated:
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    print(f"\nReconciled {updated} row(s) in {os.path.basename(path)}.")


if __name__ == "__main__":
    main()
