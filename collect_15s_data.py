"""
collect_15s_data.py

Collects 15-second-resolution Kalshi trade snapshots for simulation.
BTC prices use linear interpolation of existing 1-minute Coinbase cache.

Designed for KXBTC15M first; extend to other series (SOL, ETH) via --series.
Each market's 15s Kalshi snapshots are saved to:
  data/cache/trades_15s_{ticker}.json  →  {str(unix_ts): yes_price}

Usage:
  python collect_15s_data.py                          # BTC default
  python collect_15s_data.py --series KXSOL15M
  python collect_15s_data.py --workers 20             # more threads
  python collect_15s_data.py --limit 500              # resume partial run
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from config import CACHE_DIR, DATA_DAYS, KALSHI_BASE_URL

SNAPSHOT_INTERVAL = 15  # seconds between each price snapshot
MAX_LIMIT         = 1000
RATE_SLEEP        = 0.05  # seconds between pages within one market (20 req/s per thread)


# ── Kalshi trades fetch ───────────────────────────────────────────────────────

def _fetch_trades_for_ticker(ticker: str) -> list:
    """Return all trades for ticker, sorted oldest-first."""
    all_trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": MAX_LIMIT}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(4):
            try:
                resp = requests.get(
                    f"{KALSHI_BASE_URL}/markets/trades",
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                time.sleep(1.5 ** attempt)
        else:
            return all_trades  # partial — still usable

        batch = data.get("trades", [])
        all_trades.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(RATE_SLEEP)

    all_trades.sort(key=lambda t: t["created_time"])
    return all_trades


def build_15s_snapshots(trades: list, open_ts: int, close_ts: int) -> dict:
    """
    Given trades sorted oldest-first, build {str(ts): yes_price} for every
    15-second boundary in [open_ts, close_ts].  Uses last-observed price.
    """
    snapshots = {}
    price = None
    trade_idx = 0
    n_trades = len(trades)

    ts = (open_ts // SNAPSHOT_INTERVAL) * SNAPSHOT_INTERVAL
    while ts <= close_ts:
        # Advance through trades up to this timestamp
        while trade_idx < n_trades:
            t_ts = _parse_ts(trades[trade_idx]["created_time"])
            if t_ts <= ts:
                price = float(trades[trade_idx]["yes_price_dollars"])
                trade_idx += 1
            else:
                break
        if price is not None:
            snapshots[str(ts)] = price
        ts += SNAPSHOT_INTERVAL

    return snapshots


def _parse_ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


# ── Per-market worker (runs in thread pool) ───────────────────────────────────

def process_market(market: dict) -> tuple[str, str]:
    """
    Fetch trades for one market and save 15s snapshot to cache.
    Returns (ticker, status) where status is 'ok', 'cached', or 'skip'.
    """
    ticker    = market.get("ticker", "")
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")

    if not ticker or not open_iso or result not in ("yes", "no"):
        return ticker, "skip"

    cache_path = os.path.join(CACHE_DIR, f"trades_15s_{ticker}.json")
    if os.path.exists(cache_path):
        return ticker, "cached"

    open_ts  = _parse_ts(open_iso)
    close_ts = _parse_ts(close_iso)

    trades    = _fetch_trades_for_ticker(ticker)
    snapshots = build_15s_snapshots(trades, open_ts, close_ts)

    if snapshots:
        with open(cache_path, "w") as f:
            json.dump(snapshots, f)
        return ticker, "ok"
    return ticker, "skip"


# ── BTC interpolation helper (no new data needed) ────────────────────────────

def load_btc_1min(series: str) -> dict:
    """
    Load existing 1-minute Coinbase BTC price cache.
    For other series, the caller supplies the right coin cache files.
    Returns {ts: price} with integer keys.
    """
    coin = _coin_from_series(series)
    prices = {}
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith(f"{coin}_cb_") and fname.endswith(".json"):
            with open(os.path.join(CACHE_DIR, fname)) as f:
                day = json.load(f)
            prices.update({int(k): v for k, v in day.items()})
    return prices


def interp_btc(prices_1min: dict, ts: int) -> float | None:
    """
    Return BTC price at `ts` via linear interpolation of 1-minute data.
    Falls back to nearest neighbour if interpolation is impossible.
    """
    base = (ts // 60) * 60
    p0 = prices_1min.get(base)
    p1 = prices_1min.get(base + 60)
    if p0 is not None and p1 is not None:
        frac = (ts - base) / 60.0
        return p0 + frac * (p1 - p0)
    for off in [0, 60, -60, 120, -120]:
        v = prices_1min.get(base + off)
        if v is not None:
            return v
    return None


def _coin_from_series(series: str) -> str:
    # KXBTC15M → btc, KXSOL15M → sol, KXETH15M → eth
    return series[2:5].lower()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series",  default="KXBTC15M",
                        help="Kalshi series ticker (default: KXBTC15M)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel threads for Kalshi fetching (default: 10)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Only process first N markets (0 = all, for testing)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load markets from existing cache (analyze_coin.py must have run first)
    cache_path = os.path.join(CACHE_DIR, f"markets_{args.series}_{DATA_DAYS}d.json")
    if not os.path.exists(cache_path):
        # Fall back to the BTC-specific markets cache used by the older scripts
        cache_path = os.path.join(CACHE_DIR, f"markets_{DATA_DAYS}d.json")
    if not os.path.exists(cache_path):
        print(f"No market cache found. Run analyze_coin.py --series {args.series} first.")
        return

    with open(cache_path) as f:
        markets = json.load(f)

    valid = [m for m in markets if m.get("result") in ("yes", "no") and m.get("open_time")]
    if args.limit:
        valid = valid[:args.limit]

    already_cached = sum(
        1 for m in valid
        if os.path.exists(os.path.join(CACHE_DIR, f"trades_15s_{m.get('ticker','')}.json"))
    )
    to_fetch = len(valid) - already_cached

    print(f"\nSeries: {args.series}")
    print(f"Total valid markets : {len(valid)}")
    print(f"Already cached      : {already_cached}")
    print(f"To fetch            : {to_fetch}")
    print(f"Workers             : {args.workers}")
    if to_fetch == 0:
        print("Nothing to fetch — all markets already cached.")
        return
    print()

    done = 0
    errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_market, m): m for m in valid}
        for future in as_completed(futures):
            ticker, status = future.result()
            done += 1
            if status == "ok":
                elapsed = time.time() - t_start
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (to_fetch - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(valid)}] {ticker} — {status}"
                    f"  |  {rate:.1f}/s  ETA {eta/60:.1f}m",
                    end="\r",
                )
            elif status not in ("cached", "skip"):
                errors += 1

    elapsed = time.time() - t_start
    print(f"\n\nDone in {elapsed/60:.1f} min.  Errors: {errors}")
    print(f"15s snapshot files in: {CACHE_DIR}/trades_15s_*.json")

    # Quick sanity check on one random market
    snap_files = [
        f for f in os.listdir(CACHE_DIR)
        if f.startswith("trades_15s_") and args.series.split("M")[0] in f
    ]
    if snap_files:
        sample = snap_files[0]
        with open(os.path.join(CACHE_DIR, sample)) as f:
            snap = json.load(f)
        print(f"\nSample ({sample}): {len(snap)} snapshots")
        items = sorted(snap.items())
        if items:
            print(f"  First: ts={items[0][0]}  yes={items[0][1]:.3f}")
            print(f"  Last:  ts={items[-1][0]}  yes={items[-1][1]:.3f}")


if __name__ == "__main__":
    main()
