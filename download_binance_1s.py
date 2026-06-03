"""Download 1-second historical klines from Binance Vision and aggregate to 30s.

Source: https://data.binance.vision/ (free, public)

For each asset (BTC, ETH, SOL, XRP), downloads daily zip files of 1s klines,
extracts, aggregates to 30s, saves as JSON per day.

Output: data/cache/binance_30s_{ASSET}_{YYYYMMDD}.json
Each file maps {minute_ts_string: {open, high, low, close, volume}} for 30s buckets.

Run: python download_binance_1s.py 2026-05-03 2026-06-02
"""
import os
import sys
import json
import time
import zipfile
import io
import csv
from datetime import datetime, timezone, timedelta
import requests

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def daily_url(symbol: str, date_str: str) -> str:
    # https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2026-06-01.zip
    return f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1s/{symbol}-1s-{date_str}.zip"


def fetch_1s_klines(symbol: str, date_str: str) -> list[list]:
    """Returns list of [open_time_ms, open, high, low, close, volume, ...]"""
    url = daily_url(symbol, date_str)
    print(f"  GET {url}", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            print("(404 — not available)")
            return []
        r.raise_for_status()
    except Exception as e:
        print(f"(error: {e})")
        return []
    klines = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            with zf.open(name) as fh:
                reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
                for row in reader:
                    if not row or not row[0].isdigit():
                        continue
                    klines.append(row)
    print(f"({len(klines)} 1s klines)")
    return klines


def aggregate_30s(klines_1s: list[list]) -> dict:
    """Aggregate 1s klines into 30s candles, keyed by start-of-bucket unix seconds."""
    buckets: dict[int, dict] = {}
    for row in klines_1s:
        ts_ms = int(row[0])
        ts_s = ts_ms // 1000
        bucket_start = (ts_s // 30) * 30  # round down to 30s boundary
        o = float(row[1]); h = float(row[2]); l = float(row[3])
        c = float(row[4]); v = float(row[5])
        if bucket_start not in buckets:
            buckets[bucket_start] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        else:
            b = buckets[bucket_start]
            # open stays from first; high/low updated; close = last; volume sums
            b["high"] = max(b["high"], h)
            b["low"]  = min(b["low"], l)
            b["close"] = c
            b["volume"] += v
    return buckets


def download_range(asset: str, start_date: str, end_date: str):
    symbol = ASSETS[asset]
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = start
    while cur <= end:
        date_str = cur.strftime("%Y-%m-%d")
        cache_path = os.path.join(CACHE_DIR, f"binance_30s_{asset}_{cur.strftime('%Y%m%d')}.json")
        if os.path.exists(cache_path):
            print(f"  [{asset}] {date_str}: cached, skipping")
        else:
            klines = fetch_1s_klines(symbol, date_str)
            if klines:
                buckets = aggregate_30s(klines)
                with open(cache_path, "w") as f:
                    json.dump({str(k): v for k, v in sorted(buckets.items())}, f)
                print(f"  [{asset}] {date_str}: saved {len(buckets)} 30s candles -> {cache_path}")
            else:
                print(f"  [{asset}] {date_str}: no data, skipping")
            time.sleep(0.5)  # be polite
        cur += timedelta(days=1)


def main():
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        end_dt = datetime.now(timezone.utc) - timedelta(days=1)
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")
    print(f"Range: {start_date} -> {end_date}")
    for asset in ASSETS:
        print(f"\n=== {asset} ===")
        download_range(asset, start_date, end_date)
    print("\nDone.")


if __name__ == "__main__":
    main()
