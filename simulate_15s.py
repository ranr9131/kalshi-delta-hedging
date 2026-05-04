"""
simulate_15s.py

Compares four strategy variants on markets that have 15s Kalshi trade data:

  A) 1-min  + mag-f   (current baseline)
  B) 15-sec + mag-f   (frequency only)
  C) 1-min  + wr-f    (option-3 f only)
  D) 15-sec + wr-f    (both improvements)

BTC price at 15s intervals: linear interpolation of 1-minute Coinbase cache.
Kalshi price at 15s: trades_15s_{ticker}.json snapshots from collect_15s_data.py.

Usage:
  python simulate_15s.py
"""

import csv
import json
import math
import os

import numpy as np

from config import CACHE_DIR, LOGS_DIR, DATA_DAYS, STAKE, FEE_RATE
from analyze_coin import (
    fetch_markets, fetch_prices, build_2d_table,
    lookup_price, sig_mis,
    _2D_MIN_N, MIN_BET,
    _2D_BUCKETS,
)
from collect_15s_data import interp_btc, load_btc_1min, _parse_ts

SERIES  = "KXBTC15M"
SYMBOL  = "BTC-USD"

SNAP_INTERVAL   = 15          # seconds between 15s snapshots
DH_START_SECS   = 4 * 60      # T+4:00
DH_END_SECS     = 14 * 60     # T+14:00 (last full minute before close)
DH_END_15S_SECS = 14 * 60 + 45  # T+14:45 (last 15s mark before T+15:00 close)

WR_FALLBACK   = 0.698
WR_CENTER     = 0.65
WR_K          = 20.0
MAG_CENTER    = 0.10
MAG_K         = 20.0
SIGMOID_MAX   = 3.0


# ── f-functions ───────────────────────────────────────────────────────────────

def f_mag(pct, wr=None):
    return SIGMOID_MAX / (1 + math.exp(-MAG_K * (pct - MAG_CENTER)))


def f_wr(pct, wr):
    if wr is None:
        wr = WR_FALLBACK
    return SIGMOID_MAX / (1 + math.exp(-WR_K * (wr - WR_CENTER)))


# ── helpers ───────────────────────────────────────────────────────────────────

def bucket_idx(pct):
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= pct < hi:
            return i
    return len(_2D_BUCKETS) - 1


def get_fair(table_2d, minute, pct):
    """1-minute table lookup: key is (minute, bucket_idx)."""
    bi   = bucket_idx(pct)
    cell = table_2d.get((minute, bi))
    if cell and cell["n"] >= _2D_MIN_N:
        return cell["wins"] / cell["n"]
    return WR_FALLBACK


def get_fair_15s(table_15s, offset_secs, pct):
    """15-second table lookup: key is (offset_secs, bucket_idx)."""
    bi   = bucket_idx(pct)
    cell = table_15s.get((offset_secs, bi))
    if cell and cell["n"] >= _2D_MIN_N:
        return cell["wins"] / cell["n"]
    # fall back to nearest minute entry
    minute = offset_secs // 60
    cell   = table_15s.get((minute * 60, bi))
    if cell and cell["n"] >= _2D_MIN_N:
        return cell["wins"] / cell["n"]
    return WR_FALLBACK


def build_2d_table_15s(markets, prices_1min, series):
    """
    Build 2D win-rate table at 15-second resolution.
    Key: (offset_secs_from_t0, bucket_idx)  e.g. (240, 2) = T+4:00, 0.10-0.20%
    Needs 15s Kalshi snapshots (trades_15s_{ticker}.json) for Kalshi prices.
    BTC prices come from interpolated 1-minute data.
    """
    table = {}
    offsets = range(DH_START_SECS, DH_END_15S_SECS + 1, SNAP_INTERVAL)

    for market in markets:
        open_iso = market.get("open_time", "")
        result   = market.get("result", "")
        if not open_iso or result not in ("yes", "no"):
            continue

        t0       = _parse_ts(open_iso)
        resolved = result == "yes"
        coin_t0  = interp_btc(prices_1min, t0)
        if coin_t0 is None:
            continue

        ticker = market.get("ticker", "")
        snap   = load_15s_snap(ticker)
        if snap is None:
            continue

        for offset in offsets:
            ts     = t0 + offset
            coin_t = interp_btc(prices_1min, ts)
            k_yes  = get_kalshi_at(snap, ts)
            if coin_t is None or k_yes is None or not (0.01 < k_yes < 0.99):
                continue

            pct          = abs(coin_t - coin_t0) / coin_t0 * 100
            direction_up = coin_t > coin_t0
            sig_correct  = (direction_up == resolved)
            fill         = k_yes if direction_up else (1.0 - k_yes)

            bi  = bucket_idx(pct)
            key = (offset, bi)
            if key not in table:
                table[key] = {"n": 0, "wins": 0, "sum_fill": 0.0}
            table[key]["n"]        += 1
            table[key]["wins"]     += int(sig_correct)
            table[key]["sum_fill"] += fill

    return table


def yes_pnl(stake, price, won):
    return stake * (1 - price) / price * (1 - FEE_RATE) if won else -stake


def no_pnl(stake, fill_no, won):
    return stake * (1 - fill_no) / fill_no * (1 - FEE_RATE) if not won else -stake


def load_15s_snap(ticker):
    path = os.path.join(CACHE_DIR, f"trades_15s_{ticker}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_kalshi_at(snap, ts):
    """Last-known YES price at or before ts (15s snap dict with int keys)."""
    price = None
    for t in sorted(snap):
        if t <= ts:
            price = snap[t]
        else:
            break
    return price


# ── core simulation ───────────────────────────────────────────────────────────

FILL_SIM_BUFFER = 0.025  # half typical spread (0.005) + live fill buffer (0.02)


def simulate_one(markets, prices_1min, table_1min, table_15s, f_fn, use_15s,
                 skip_neg_mis=False, additive=False, fill_buf=FILL_SIM_BUFFER,
                 vel_filter=False, min_pct=0.0, vel_soft_k=None, confirm_intervals=0):
    """
    Simulate DH. use_15s=True checks every 15s using the 15s table;
    False checks every 60s using the 1-min table.
    skip_neg_mis=True zeroes out the target whenever mispricing < 0.
    additive=True bets the full target each interval (not just the gap vs current exposure).
    fill_buf: cents added to k_yes to approximate ask+buffer vs last-trade price.
    vel_filter=True hard-skips bets when 15s BTC velocity disagrees with direction.
    min_pct: minimum abs % BTC move from T+0 required to place any bet.
    vel_soft_k: if set, applies soft velocity multiplier f_vel = 2.0/(1+exp(-k*vel_aligned))
                where vel_aligned is % velocity in the direction of the bet.
                Neutral at vel=0 (f_vel=1.0), boosts agreeing vel, shrinks opposing vel.
    """
    results = []

    for market in markets:
        open_iso = market.get("open_time", "")
        result   = market.get("result", "")
        if not open_iso or result not in ("yes", "no"):
            continue

        t0       = _parse_ts(open_iso)
        resolved = result == "yes"
        coin_t0  = interp_btc(prices_1min, t0)
        if coin_t0 is None:
            continue

        ticker = market.get("ticker", "")
        snap   = load_15s_snap(ticker)
        if snap is None:
            continue

        step    = SNAP_INTERVAL if use_15s else 60
        t_start = t0 + DH_START_SECS
        t_end   = t0 + (DH_END_15S_SECS if use_15s else DH_END_SECS)

        yes_exp, no_exp = 0.0, 0.0
        yes_bets, no_bets = [], []
        opp_streak = 0

        ts = t_start
        while ts <= t_end:
            offset = ts - t0

            coin_t = interp_btc(prices_1min, ts)
            k_yes  = get_kalshi_at(snap, ts)
            if coin_t is None or k_yes is None or not (0.01 < k_yes < 0.99):
                ts += step
                continue

            direction_up = coin_t > coin_t0
            pct  = abs(coin_t - coin_t0) / coin_t0 * 100

            if min_pct > 0.0 and pct < min_pct:
                ts += step
                continue

            if vel_filter:
                btc_15s_ago = interp_btc(prices_1min, ts - 15)
                if btc_15s_ago is not None:
                    vel_agrees = (coin_t > btc_15s_ago) == direction_up
                else:
                    vel_agrees = True
                if not vel_agrees:
                    ts += step
                    continue

            if use_15s:
                fair = get_fair_15s(table_15s, offset, pct)
            else:
                fair = get_fair(table_1min, offset // 60, pct)
            f    = f_fn(pct, fair)

            # Soft velocity multiplier: f_vel = 2.0/(1+exp(-k*vel_aligned))
            # vel_aligned > 0 means BTC is moving in the direction of the bet
            if vel_soft_k is not None:
                btc_15s_ago = interp_btc(prices_1min, ts - 15)
                if btc_15s_ago is not None:
                    raw_vel     = (coin_t - btc_15s_ago) / btc_15s_ago * 100
                    vel_aligned = raw_vel if direction_up else -raw_vel
                else:
                    vel_aligned = 0.0
                f_vel = 2.0 / (1.0 + math.exp(-vel_soft_k * vel_aligned))
            else:
                f_vel = 1.0

            fill_yes = min(k_yes + fill_buf, 0.99)
            fill_no  = min((1.0 - k_yes) + fill_buf, 0.99)

            if direction_up:
                mis      = fair - fill_yes
                g        = sig_mis(mis)
                tgt_yes  = 0.0 if (skip_neg_mis and mis < 0) else STAKE * f * f_vel * g
                tgt_no   = 0.0
            else:
                mis      = fair - fill_no
                g        = sig_mis(mis)
                tgt_yes  = 0.0
                tgt_no   = 0.0 if (skip_neg_mis and mis < 0) else STAKE * f * f_vel * g

            if additive:
                gap_yes = tgt_yes
                gap_no  = tgt_no
            else:
                gap_yes = max(0.0, tgt_yes - yes_exp)
                gap_no  = max(0.0, tgt_no  - no_exp)

            # Consecutive confirmation gate
            if confirm_intervals > 0:
                has_exp = (yes_exp + no_exp) > 0
                dom_up = None
                if has_exp:
                    if yes_exp > no_exp:
                        dom_up = True
                    elif no_exp > yes_exp:
                        dom_up = False
                if dom_up is not None and direction_up != dom_up:
                    opp_streak += 1
                else:
                    opp_streak = 0
                if dom_up is not None and direction_up != dom_up and opp_streak < confirm_intervals:
                    gap_yes = 0.0
                    gap_no  = 0.0

            if gap_yes >= MIN_BET:
                yes_bets.append((gap_yes, fill_yes))
                yes_exp += gap_yes
            if gap_no >= MIN_BET:
                no_bets.append((gap_no, fill_no))
                no_exp += gap_no

            ts += step

        pnl_yes = sum(yes_pnl(s, p, resolved) for s, p in yes_bets)
        pnl_no  = sum(no_pnl(s, p, resolved)  for s, p in no_bets)
        total_w = yes_exp + no_exp
        if total_w == 0:
            continue

        results.append({
            "pnl":    pnl_yes + pnl_no,
            "wagered": total_w,
            "n_bets": len(yes_bets) + len(no_bets),
            "win":    int((pnl_yes + pnl_no) > 0),
        })

    return results


def summarise(label, results):
    if not results:
        return None
    pnl      = sum(r["pnl"]     for r in results)
    wagered  = sum(r["wagered"] for r in results)
    win_rate = sum(r["win"]     for r in results) / len(results)
    roi      = pnl / wagered * 100
    avg_bets = sum(r["n_bets"]  for r in results) / len(results)
    return dict(label=label, windows=len(results), win_rate=win_rate,
                pnl=pnl, wagered=wagered, roi=roi, avg_bets=avg_bets)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime, timezone

    markets = fetch_markets(SERIES, DATA_DAYS)
    print(f"Total markets: {len(markets)}")

    # Filter to only markets that have 15s data
    markets_15s = [
        m for m in markets
        if m.get("result") in ("yes", "no")
        and os.path.exists(os.path.join(CACHE_DIR, f"trades_15s_{m.get('ticker','')}.json"))
    ]
    print(f"Markets with 15s data: {len(markets_15s)}\n")

    timestamps = [_parse_ts(m["open_time"]) for m in markets_15s if m.get("open_time")]
    print(f"Loading BTC price data...")
    prices = fetch_prices(SYMBOL, min(timestamps) - 600, max(timestamps) + 1800)
    # Also load as int-keyed dict for interpolation
    prices_1min = {int(k): v for k, v in prices.items()}
    print(f"  {len(prices_1min)} minute-prices\n")

    print("Building 1-min 2D win-rate table...")
    table_1min = build_2d_table(markets_15s, prices, SERIES)
    print(f"  {len(table_1min)} cells\n")

    print("Building 15-sec 2D win-rate table...")
    table_15s = build_2d_table_15s(markets_15s, prices_1min, SERIES)
    print(f"  {len(table_15s)} cells\n")

    # Quick check: show how much win rates vary within a single minute
    from collections import defaultdict
    bucket_label = "0.10-0.20%"
    bi_check = 2
    print("  Win rate within minute 5 (0.10-0.20% bucket) at each 15s mark:")
    for off in range(300, 360, 15):
        cell = table_15s.get((off, bi_check))
        if cell and cell["n"] >= _2D_MIN_N:
            wr = cell["wins"] / cell["n"]
            print(f"    T+{off//60}:{off%60:02d}  wr={wr:.3f}  n={cell['n']}")
    print()

    # Save 15s table to CSV for the live trader
    _2D_BUCKET_LABELS = ["0.00-0.01%", "0.01-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]
    out_path = os.path.join(LOGS_DIR, f"minute_analysis_2d_15s_{SERIES.lower()}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset_secs", "bucket", "n", "win_rate", "avg_fill"])
        for (offset, bi), cell in sorted(table_15s.items()):
            if cell["n"] > 0:
                w.writerow([offset, _2D_BUCKET_LABELS[bi], cell["n"],
                            round(cell["wins"] / cell["n"], 6),
                            round(cell["sum_fill"] / cell["n"], 6)])
    print(f"15s table saved: {out_path}\n")

    # (label, f_fn, use_15s, skip_neg_mis, additive, vel_filter)
    variants = [
        ("A) 1-min  + mag-f  (baseline)",         f_mag, False, False, False, False),
        ("B) 15-sec + mag-f            ",          f_mag, True,  False, False, False),
        ("C) 1-min  + wr-f   (opt-3)   ",          f_wr,  False, False, False, False),
        ("D) 15-sec + wr-f             ",          f_wr,  True,  False, False, False),
        ("E) 15-sec + wr-f  + no-neg-mis",         f_wr,  True,  True,  False, False),
        ("F) 15-sec + wr-f  + additive ",          f_wr,  True,  False, True,  False),
        ("G) 15-sec + wr-f  + add+no-neg-mis",     f_wr,  True,  True,  True,  False),
        ("H) 15-sec + wr-f  + vel-filter",         f_wr,  True,  False, False, True),
        ("I) 15-sec + wr-f  + no-neg-mis + vel",   f_wr,  True,  True,  False, True),
    ]

    rows = []
    for label, f_fn, use_15s, skip_neg_mis, additive, vel_filter in variants:
        print(f"  Simulating {label} ...", end="\r")
        res = simulate_one(markets_15s, prices_1min, table_1min, table_15s, f_fn, use_15s, skip_neg_mis, additive, vel_filter=vel_filter)
        row = summarise(label, res)
        rows.append(row)
        print(f"  {label}  ROI {row['roi']:+.2f}%  win {row['win_rate']:.1%}  bets/w {row['avg_bets']:.1f}  P&L ${row['pnl']:+,.0f}")

    base_roi = rows[0]["roi"]
    print(f"\n{'='*78}")
    print(f"{'Variant':<40}  {'ROI':>8}  {'vs A':>7}  {'Win%':>6}  {'Bets/w':>7}  {'P&L':>12}")
    print("-" * 78)
    for r in rows:
        delta  = r["roi"] - base_roi
        marker = " <--" if r["roi"] == max(x["roi"] for x in rows) else ""
        print(
            f"  {r['label']:<38}  {r['roi']:>+7.2f}%  {delta:>+6.2f}pp"
            f"  {r['win_rate']:>5.1%}  {r['avg_bets']:>6.1f}  ${r['pnl']:>+11,.0f}{marker}"
        )
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
