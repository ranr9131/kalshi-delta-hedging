"""
Rate-limited candle re-fetcher with resume.

Loops over all settled KXBTC15M markets and re-fetches Kalshi candles with
the richer field set (high/low/bid/ask/volume). Skips any cache file already
in the new format. Pauses on 429s with exponential backoff. Designed to be
safe to interrupt + resume.

Settings:
  REQ_DELAY        — min seconds between API calls
  BACKOFF_SECS_429 — base wait on rate-limit; doubled per consecutive 429

Run:
  python3 nn/refetch_candles.py
"""

import os, sys, json, time, math, random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kalshi_client
from config import DATA_DAYS, CACHE_DIR

REQ_DELAY        = 0.6     # base sleep between requests (safe under most limits)
BACKOFF_SECS_429 = 15      # initial backoff on rate-limit
BACKOFF_MAX      = 300     # cap backoff at 5 minutes


def has_new_format(path):
    """Quick check: does cache file already contain yes_high field?"""
    try:
        with open(path) as f:
            head = f.read(800)
        return '"yes_high"' in head
    except Exception:
        return False


def main():
    print("Loading settled markets…", flush=True)
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    print(f"  {len(markets)} markets to consider", flush=True)

    # Pre-screen: skip markets whose cache already has the new format.
    to_fetch = []
    skipped_have_fresh = 0
    for mk in markets:
        ticker = mk["ticker"]
        path = os.path.join(CACHE_DIR, f"candles_{ticker}.json")
        if os.path.exists(path) and has_new_format(path):
            skipped_have_fresh += 1
        else:
            to_fetch.append(mk)
    print(f"  Already fresh: {skipped_have_fresh}")
    print(f"  Need to fetch: {len(to_fetch)}", flush=True)
    print(f"  ETA at {REQ_DELAY}s/req: ~{len(to_fetch)*REQ_DELAY/60:.0f} min "
          f"(plus retries)\n", flush=True)

    consecutive_429 = 0
    backoff = BACKOFF_SECS_429
    start_time = time.time()
    success = 0
    failed = 0

    for i, mk in enumerate(to_fetch):
        ticker    = mk["ticker"]
        open_iso  = mk["open_time"]
        close_iso = mk["close_time"]
        path      = os.path.join(CACHE_DIR, f"candles_{ticker}.json")

        # Delete old-format file so fetch_candlesticks re-fetches
        if os.path.exists(path):
            try: os.remove(path)
            except Exception: pass

        try:
            candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
            if candles:
                success += 1
                consecutive_429 = 0
                backoff = BACKOFF_SECS_429
            else:
                failed += 1
        except Exception as e:
            es = str(e)
            if "429" in es:
                consecutive_429 += 1
                wait = min(BACKOFF_MAX, backoff * (2 ** (consecutive_429 - 1)))
                print(f"  [{i+1}/{len(to_fetch)}] 429 ({consecutive_429} in a row) "
                      f"— sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            else:
                failed += 1
                print(f"  [{i+1}/{len(to_fetch)}] error on {ticker}: {e}", flush=True)

        time.sleep(REQ_DELAY)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(to_fetch) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(to_fetch)}] success={success} failed={failed} "
                  f"rate={rate:.2f}/s ETA={eta:.0f}min", flush=True)

    print(f"\nDone. success={success} failed={failed}")
    # Final count of files with new format
    new_count = 0; total = 0
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith("candles_KXBTC15M-") and fname.endswith(".json"):
            total += 1
            if has_new_format(os.path.join(CACHE_DIR, fname)):
                new_count += 1
    print(f"Cache: {new_count} new-format / {total} total candle files")


if __name__ == "__main__":
    main()
