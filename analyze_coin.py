"""
Run the full DH simulation + 2D win-rate analysis for any Kalshi 15-minute series.

Usage:
  python analyze_coin.py --series KXSOL15M --symbol SOL-USD
  python analyze_coin.py --series KXETH15M --symbol ETH-USD
  python analyze_coin.py --series KXXRP15M  --symbol XRP-USD
  python analyze_coin.py --series KXBTC15M --symbol BTC-USD   # baseline check
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from config import CACHE_DIR, LOGS_DIR, DATA_DAYS, STAKE, FEE_RATE

# ── Constants (same as simulate_dh / analyze_minutes_2d) ─────────────────────
SIGMOID_CENTER   = 0.10
SIGMOID_K        = 20.0
SIGMOID_MAX_MULT = 3.0
MISPRICING_K     = 8.0
MISPRICING_MAX   = 2.0
MIN_BET          = 5.0
DH_MINUTES       = list(range(4, 14))
_2D_MIN_N        = 30

_2D_BUCKETS = [
    (0.000, 0.05), (0.050, 0.10), (0.100, 0.20), (0.200, 0.50), (0.500, float("inf")),
]
_2D_BUCKET_LABELS = ["0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]

COINBASE_BASE = "https://api.exchange.coinbase.com/products"
KALSHI_BASE   = "https://api.elections.kalshi.com/trade-api/v2"


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_prices(symbol: str, start_ts: int, end_ts: int) -> dict:
    coin = symbol.split("-")[0].lower()
    start_day = int(start_ts // 86400) * 86400
    end_day   = int(end_ts   // 86400) * 86400 + 86400
    prices = {}
    current = start_day
    total_days = (end_day - start_day) // 86400
    day_num = 0

    while current < end_day:
        day_num += 1
        day_str    = datetime.fromtimestamp(current, tz=timezone.utc).strftime("%Y%m%d")
        cache_path = os.path.join(CACHE_DIR, f"{coin}_cb_{day_str}.json")

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                day_prices = json.load(f)
            if not day_prices:
                os.remove(cache_path)
                day_prices = _fetch_price_day(symbol, current, day_str, day_num, total_days)
                if day_prices:
                    with open(cache_path, "w") as f:
                        json.dump(day_prices, f)
        else:
            day_prices = _fetch_price_day(symbol, current, day_str, day_num, total_days)
            if day_prices:
                with open(cache_path, "w") as f:
                    json.dump(day_prices, f)

        prices.update(day_prices)
        current += 86400

    return prices


def _fetch_price_day(symbol: str, day_start: int, day_str: str, day_num: int, total: int) -> dict:
    url    = f"{COINBASE_BASE}/{symbol}/candles"
    prices = {}
    chunk  = 300 * 60
    cs     = day_start

    while cs < day_start + 86400:
        ce  = min(cs + chunk, day_start + 86400)
        s   = datetime.fromtimestamp(cs, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        e   = datetime.fromtimestamp(ce, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        for attempt in range(4):
            try:
                resp = requests.get(url, params={"granularity": 60, "start": s, "end": e}, timeout=15)
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                for row in resp.json():
                    ts = int(row[0])
                    if day_start <= ts < day_start + 86400:
                        prices[str(ts)] = float(row[4])
                break
            except Exception as exc:
                time.sleep(2 * (attempt + 1))
        cs += chunk
        time.sleep(0.15)

    if prices:
        print(f"  [{day_num}/{total}] {day_str}: {len(prices)} {symbol} prices")
    return prices


def lookup_price(prices: dict, ts: int):
    base = (ts // 60) * 60
    for off in [0, 60, -60, 120, -120]:
        v = prices.get(str(base + off))
        if v is not None:
            return v
    return None


# ── Kalshi market data ────────────────────────────────────────────────────────

def fetch_markets(series: str, days: int = DATA_DAYS) -> list:
    cache_path = os.path.join(CACHE_DIR, f"markets_{series}_{days}d.json")
    if os.path.exists(cache_path):
        print(f"Loading {series} markets from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    markets, cursor = [], None

    print(f"Fetching settled {series} markets (last {days} days)...")
    while True:
        params = {"series_ticker": series, "status": "settled", "min_close_ts": cutoff_ts, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(3):
            try:
                resp = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                time.sleep(2 ** attempt)
        else:
            break

        batch  = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        print(f"  {len(batch)} markets (total: {len(markets)})")
        if not cursor or not batch:
            break
        time.sleep(0.3)

    with open(cache_path, "w") as f:
        json.dump(markets, f)
    return markets


def fetch_candles(series: str, ticker: str, open_iso: str, close_iso: str) -> list:
    cache_path = os.path.join(CACHE_DIR, f"candles_{ticker}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    open_dt  = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    close_dt = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    s_ts = int((open_dt  - timedelta(minutes=1)).timestamp())
    e_ts = int((close_dt + timedelta(minutes=1)).timestamp())

    try:
        resp = requests.get(
            f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks",
            params={"start_ts": s_ts, "end_ts": e_ts, "period_interval": 1},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  candles error {ticker}: {e}")
        return []

    result = []
    for c in resp.json().get("candlesticks", []):
        p = c.get("price", {})
        yo = p.get("open_dollars") or p.get("open")
        yc = p.get("close_dollars") or p.get("close")
        if yo is None or yc is None:
            continue
        result.append({"ts": c["end_period_ts"], "yes_open": float(yo), "yes_close": float(yc)})
    result.sort(key=lambda x: x["ts"])

    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


def get_price_at(candles: list, target_ts: int):
    price = None
    for c in candles:
        if c["ts"] <= target_ts:
            price = c["yes_close"]
        else:
            break
    return price


# ── Sigmoid functions ─────────────────────────────────────────────────────────

def sig_btc(x):
    return SIGMOID_MAX_MULT / (1 + np.exp(-SIGMOID_K * (x - SIGMOID_CENTER)))


def sig_mis(x):
    return MISPRICING_MAX / (1 + np.exp(-MISPRICING_K * x))


def bucket_idx(pct: float) -> int:
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= pct < hi:
            return i
    return len(_2D_BUCKETS) - 1


# ── P&L helpers ───────────────────────────────────────────────────────────────

def yes_pnl(stake, price, won):
    return stake * (1 - price) / price * (1 - FEE_RATE) if won else -stake


def no_pnl(stake, price, won):
    np_ = 1 - price
    return stake * (1 - np_) / np_ * (1 - FEE_RATE) if not won else -stake


# ── Phase 1: build 2D win-rate table from raw market data ────────────────────

def build_2d_table(markets: list, prices: dict, series: str) -> dict:
    """
    Returns table: (minute, bucket_idx) → {n, wins, sum_fill}
    """
    table = {}

    for market in markets:
        open_iso = market.get("open_time", "")
        result   = market.get("result", "")
        if not open_iso or result not in ("yes", "no"):
            continue

        open_dt    = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
        t0         = int(open_dt.timestamp())
        resolved   = result == "yes"
        coin_t0    = lookup_price(prices, t0)
        if coin_t0 is None:
            continue

        ticker     = market.get("ticker", "")
        close_iso  = market.get("close_time", "")
        candles    = fetch_candles(series, ticker, open_iso, close_iso)
        if not candles:
            continue

        for minute in range(1, 15):
            t       = t0 + minute * 60
            coin_t  = lookup_price(prices, t)
            k_yes   = get_price_at(candles, t)
            if coin_t is None or k_yes is None or not (0.01 < k_yes < 0.99):
                continue

            pct          = abs(coin_t - coin_t0) / coin_t0 * 100
            direction_up = coin_t > coin_t0
            sig_correct  = (direction_up == resolved)
            fill         = k_yes if direction_up else (1.0 - k_yes)

            bi  = bucket_idx(pct)
            key = (minute, bi)
            if key not in table:
                table[key] = {"n": 0, "wins": 0, "sum_fill": 0.0}
            table[key]["n"]        += 1
            table[key]["wins"]     += int(sig_correct)
            table[key]["sum_fill"] += fill

    return table


# ── Phase 2: simulate DH-target with 2D fair prices ──────────────────────────

def simulate_dh(markets: list, prices: dict, series: str, table_2d: dict,
                kal_filter: bool = False) -> list:
    """
    kal_filter: if True, skip any minute where Kalshi direction disagrees with
    BTC position (BTC above floor but YES < 0.50, or BTC below floor but YES > 0.50).
    """
    results = []

    for market in markets:
        open_iso = market.get("open_time", "")
        result   = market.get("result", "")
        if not open_iso or result not in ("yes", "no"):
            continue

        open_dt  = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
        t0       = int(open_dt.timestamp())
        resolved = result == "yes"
        coin_t0  = lookup_price(prices, t0)
        if coin_t0 is None:
            continue

        ticker    = market.get("ticker", "")
        close_iso = market.get("close_time", "")
        candles   = fetch_candles(series, ticker, open_iso, close_iso)
        if not candles:
            continue

        k_t0 = candles[0]["yes_open"] if candles else None
        if k_t0 is None or not (0.01 < k_t0 < 0.99):
            continue

        yes_exp, no_exp = 0.0, 0.0
        yes_bets, no_bets = [], []

        for minute in DH_MINUTES:
            t       = t0 + minute * 60
            coin_t  = lookup_price(prices, t)
            k_yes   = get_price_at(candles, t)
            if coin_t is None or k_yes is None or not (0.01 < k_yes < 0.99):
                continue

            direction_up = coin_t > coin_t0

            # Kalshi disagreement filter
            if kal_filter and direction_up and k_yes < 0.5:
                continue
            if kal_filter and not direction_up and k_yes > 0.5:
                continue

            pct  = abs(coin_t - coin_t0) / coin_t0 * 100
            f    = sig_btc(pct)
            bi   = bucket_idx(pct)
            cell = table_2d.get((minute, bi))
            if cell and cell["n"] >= _2D_MIN_N:
                fair = cell["wins"] / cell["n"]
            else:
                fair = 0.698  # global fallback

            if direction_up:
                mis      = fair - k_yes
                g        = sig_mis(mis)
                tgt_yes  = STAKE * f * g
                tgt_no   = 0.0
            else:
                mis      = fair - (1.0 - k_yes)
                g        = sig_mis(mis)
                tgt_yes  = 0.0
                tgt_no   = STAKE * f * g

            gap_yes = max(0.0, tgt_yes - yes_exp)
            gap_no  = max(0.0, tgt_no  - no_exp)

            if gap_yes >= MIN_BET:
                yes_bets.append((gap_yes, k_yes))
                yes_exp += gap_yes
            if gap_no >= MIN_BET:
                no_bets.append((gap_no, k_yes))
                no_exp += gap_no

        pnl_yes = sum(yes_pnl(s, p, resolved)  for s, p in yes_bets)
        pnl_no  = sum(no_pnl(s, p, resolved)   for s, p in no_bets)
        total_wagered = yes_exp + no_exp

        if total_wagered == 0:
            continue

        results.append({
            "total_pnl":    pnl_yes + pnl_no,
            "total_wagered": total_wagered,
            "n_yes_bets":   len(yes_bets),
            "n_no_bets":    len(no_bets),
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="KXSOL15M")
    parser.add_argument("--symbol", default="SOL-USD")
    parser.add_argument("--days",   type=int, default=DATA_DAYS)
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Series: {args.series}  |  Price feed: {args.symbol}  |  Days: {args.days}")
    print(f"{'='*55}\n")

    markets = fetch_markets(args.series, args.days)
    print(f"Markets loaded: {len(markets)}\n")

    timestamps = []
    for m in markets:
        try:
            dt = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            timestamps.append(int(dt.timestamp()))
        except Exception:
            pass

    if not timestamps:
        print("No valid market timestamps found.")
        return

    print(f"Fetching {args.symbol} price data...")
    prices = fetch_prices(args.symbol, min(timestamps) - 600, max(timestamps) + 1800)
    print(f"Price data loaded: {len(prices)} minute-prices\n")

    print("Building 2D win-rate table...")
    table_2d = build_2d_table(markets, prices, args.series)

    # Print 2D table
    win_rates_all = []
    print(f"\n{'Minute':>7}  {'Bucket':<14}  {'N':>5}  {'WinRate':>8}  {'AvgFill':>8}")
    print("-" * 55)
    for minute in range(4, 14):
        for bi, label in enumerate(_2D_BUCKET_LABELS):
            cell = table_2d.get((minute, bi))
            if cell and cell["n"] >= _2D_MIN_N:
                wr   = cell["wins"] / cell["n"]
                af   = cell["sum_fill"] / cell["n"]
                win_rates_all.append(wr)
                print(f"  T+{minute:<4}  {label:<14}  {cell['n']:>5}  {wr:>8.3f}  {af:>8.3f}")
    print()

    def summarise(label, results):
        if not results:
            print(f"  {label}: no results")
            return
        total_pnl     = sum(r["total_pnl"]     for r in results)
        total_wagered = sum(r["total_wagered"] for r in results)
        avg_bets      = sum(r["n_yes_bets"] + r["n_no_bets"] for r in results) / len(results)
        wins          = sum(1 for r in results if r["total_pnl"] > 0)
        win_rate      = wins / len(results)
        roi           = total_pnl / total_wagered * 100
        print(f"\n{'='*57}")
        print(f"RESULTS: {args.series} — {label}")
        print(f"{'='*57}")
        print(f"  Windows simulated:  {len(results)}")
        print(f"  Win rate:           {win_rate:.1%}")
        print(f"  Total P&L:          ${total_pnl:+,.2f}")
        print(f"  Total wagered:      ${total_wagered:,.2f}")
        print(f"  ROI:                {roi:+.2f}%")
        print(f"  Avg bets/window:    {avg_bets:.1f}")
        print(f"{'='*57}")
        return roi, avg_bets

    print("Running DH-target simulation (baseline)...")
    r_base   = simulate_dh(markets, prices, args.series, table_2d, kal_filter=False)
    base_out = summarise("baseline (no filter)", r_base)

    print("\nRunning DH-target simulation (Kalshi disagreement filter)...")
    r_filt   = simulate_dh(markets, prices, args.series, table_2d, kal_filter=True)
    filt_out = summarise("kal-filter ON (skip when Kalshi disagrees)", r_filt)

    if base_out and filt_out:
        delta_roi  = filt_out[0] - base_out[0]
        delta_bets = filt_out[1] - base_out[1]
        print(f"\n  Filter impact: ROI {delta_roi:+.2f}pp | avg bets/window {delta_bets:+.1f}")

    # Save 2D table to CSV
    tag      = args.series.lower()
    out_path = os.path.join(LOGS_DIR, f"minute_analysis_2d_{tag}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute", "bucket", "n", "win_rate", "avg_fill"])
        for minute in range(1, 15):
            for bi, label in enumerate(_2D_BUCKET_LABELS):
                cell = table_2d.get((minute, bi))
                if cell and cell["n"] > 0:
                    w.writerow([
                        minute, label, cell["n"],
                        round(cell["wins"] / cell["n"], 6),
                        round(cell["sum_fill"] / cell["n"], 6),
                    ])
    print(f"2D table saved: {out_path}")


if __name__ == "__main__":
    main()
