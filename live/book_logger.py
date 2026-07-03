"""
Passive order-book depth logger — places NO orders.

Subscribes to the live Kalshi order book for the current BTC market and writes a
full-depth snapshot to a JSONL file every few seconds. This captures the
LIQUIDITY component of slippage (how far a given size walks the book) at scale
and for free, with zero capital risk. Run it 24/7 alongside (or instead of) the
trader — ideally on the same AWS host so book age reflects real feed latency.

Analyze the output with depth_slippage.py.

Usage:
    python book_logger.py                      # default 2s cadence
    SNAPSHOT_SECS=1 python book_logger.py      # faster cadence
"""

import json
import os
import time
from datetime import datetime, timezone

from dotenv import dotenv_values

import kalshi_auth
import kalshi_feed
import kalshi_trade

_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))
PRIVATE_KEY = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")

SNAPSHOT_SECS = float(os.environ.get("SNAPSHOT_SECS", "2"))
OUT_PATH = os.path.join(_dir, "book_log.jsonl")
DEPTH_LEVELS = 20   # cap ladder length written per side


def current_ticker():
    try:
        m = kalshi_trade.get_open_market()
        return m["ticker"] if m else None
    except Exception:
        return None


def main():
    tk = current_ticker()
    print(f"Starting book logger -> {OUT_PATH}")
    print(f"Cadence {SNAPSHOT_SECS}s | initial ticker: {tk}")
    kalshi_feed.start(PRIVATE_KEY, API_KEY_ID, tk)

    last_ticker_check = 0.0
    n = 0
    with open(OUT_PATH, "a", buffering=1) as out:
        while True:
            now = time.time()
            # Re-check the open market every ~20s and roll the subscription.
            if now - last_ticker_check > 20:
                nt = current_ticker()
                if nt and nt != tk:
                    tk = nt
                    kalshi_feed.set_ticker(tk)
                    print(f"[{datetime.now(timezone.utc).isoformat()}] rolled to {tk}")
                last_ticker_check = now

            book = kalshi_feed.get_book()
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "ticker": book["ticker"],
                "book_age": None if book["age"] == float("inf") else round(book["age"], 3),
                "yes_bid": kalshi_feed.get_bid(),
                "yes_ask": kalshi_feed.get_ask(),
                # ladders: [price_dollars, size], cheapest-first, to buy that side
                "yes_asks": book["yes_asks"][:DEPTH_LEVELS],
                "no_asks": book["no_asks"][:DEPTH_LEVELS],
            }
            # Only bother writing once we actually have a book.
            if rec["yes_asks"] or rec["no_asks"]:
                out.write(json.dumps(rec) + "\n")
                n += 1
                if n % 30 == 0:
                    ya = rec["yes_asks"][0] if rec["yes_asks"] else None
                    print(f"  {n} snaps | {rec['ticker']} | best yes_ask {ya} | age {rec['book_age']}s")

            time.sleep(SNAPSHOT_SECS)


if __name__ == "__main__":
    main()
