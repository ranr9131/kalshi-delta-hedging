"""
Delta hedging simulation on Kalshi KXBTC15M markets.

Strategy:
  T+0       : Bet $STAKE on Yes at Kalshi opening price.
  T+5..T+10 : Every minute, compute a target position based on:
                 f(cumulative BTC move from T+0)  x  g(Kalshi mispricing)
               - Additive:  place the full computed stake each interval.
               - Target:    only place the gap between current exposure and target.
               Both track Yes and No exposure independently (can hold both).
  T+15      : Kalshi settles.

Outputs:
  data/logs/simulation_results_dh_additive.csv
  data/logs/simulation_results_dh_target.csv
"""

import os
import csv
import time
import argparse
import numpy as np
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import STAKE, FEE_RATE, DATA_DAYS, LOGS_DIR, CACHE_DIR

# ── Scaling parameters (same as analyze_scaled.py combined mode) ──────────────
FAIR_PRICE       = 0.698
SIGMOID_CENTER   = 0.10
SIGMOID_K        = 20.0
SIGMOID_MAX_MULT = 3.0
MISPRICING_K     = 8.0
MISPRICING_MAX   = 2.0
MIN_BET          = 5.0   # ignore bets smaller than this (fee drag not worth it)

DEFAULT_MINUTES = "5-10"
DH_MINUTES = list(range(5, 11))   # overridden by --minutes arg at runtime

# Empirical win rate per minute from analyze_minutes.py (30d, 2820 markets).
# Replaces fixed FAIR_PRICE when --dynamic-fair-price is set.
FAIR_PRICE_BY_MINUTE = {
    1: 0.582, 2: 0.617, 3: 0.636, 4: 0.670,
    5: 0.698, 6: 0.728, 7: 0.751, 8: 0.759,
    9: 0.783, 10: 0.798, 11: 0.806, 12: 0.815,
    13: 0.826, 14: 0.704,
}

# 2D fair price table: loaded from minute_analysis_2d.csv when --fair-price-2d is set.
# Key: (minute, bucket_index)  Value: (win_rate, avg_fill, n)
_FAIR_PRICE_2D: dict[tuple[int, int], tuple[float, float, int]] = {}

# Magnitude buckets — must match analyze_minutes_2d.py exactly
_2D_BUCKETS = [
    (0.000, 0.05),
    (0.050, 0.10),
    (0.100, 0.20),
    (0.200, 0.50),
    (0.500, float("inf")),
]
_2D_BUCKET_LABELS = [
    "0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+",
]
_2D_MIN_N = 30   # cells below this sample count fall back to 1D table


def _load_2d_table(csv_path: str) -> None:
    label_to_idx = {lbl: i for i, lbl in enumerate(_2D_BUCKET_LABELS)}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            minute = int(row["minute"])
            bi     = label_to_idx.get(row["bucket"])
            if bi is None:
                continue
            _FAIR_PRICE_2D[(minute, bi)] = (
                float(row["win_rate"]),
                float(row["avg_fill"]),
                int(row["n"]),
            )


def _get_bucket_idx(abs_pct: float) -> int:
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= abs_pct < hi:
            return i
    return len(_2D_BUCKETS) - 1


def get_fair_price_2d(minute: int, abs_pct_move: float) -> float:
    """
    Return the 2D empirical win rate for (minute, magnitude_bucket).
    Falls back to FAIR_PRICE_BY_MINUTE if cell has too few samples.
    """
    bi = _get_bucket_idx(abs_pct_move)
    entry = _FAIR_PRICE_2D.get((minute, bi))
    if entry is not None and entry[2] >= _2D_MIN_N:
        return entry[0]
    return FAIR_PRICE_BY_MINUTE.get(minute, FAIR_PRICE)


# ── Scaling functions ─────────────────────────────────────────────────────────

def sigmoid_btc(abs_pct_move):
    x = SIGMOID_K * (abs_pct_move - SIGMOID_CENTER)
    return SIGMOID_MAX_MULT / (1.0 + np.exp(-x))


def sigmoid_mispricing(mispricing):
    x = MISPRICING_K * mispricing
    return MISPRICING_MAX / (1.0 + np.exp(-x))


# ── P&L helpers ───────────────────────────────────────────────────────────────

def yes_pnl(stake, yes_price, resolved_yes):
    if resolved_yes:
        return stake * (1 - yes_price) / yes_price * (1 - FEE_RATE)
    return -stake


def no_pnl(stake, yes_price, resolved_yes):
    no_price = 1 - yes_price
    if not resolved_yes:
        return stake * (1 - no_price) / no_price * (1 - FEE_RATE)
    return -stake


# ── Core: simulate one window in both modes simultaneously ────────────────────

def simulate_market_dh(market, btc_prices, dh_minutes=None, dynamic_fair_price=False, dead_zone=0.0, fair_price_2d=False):
    """
    Returns (additive_row, target_row) or (None, None) if data is missing.
    Both modes run on the same fetched candle + BTC data.
    """
    ticker     = market["ticker"]
    open_iso   = market.get("open_time", "")
    close_iso  = market.get("close_time", "")
    result_field = market.get("result", "")

    if not open_iso or not close_iso or result_field not in ("yes", "no"):
        return None, None

    open_dt    = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0         = int(open_dt.timestamp())
    resolved_yes = result_field == "yes"

    btc_t0 = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None:
        return None, None

    candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
    if not candles:
        return None, None

    kalshi_t0 = candles[0]["yes_open"]
    if kalshi_t0 is None or not (0.01 < kalshi_t0 < 0.99):
        return None, None

    pnl_initial = yes_pnl(STAKE, kalshi_t0, resolved_yes)

    # T+5 snapshot for backward-compat columns
    t5          = t0 + 5 * 60
    btc_t5      = btc_data.lookup(btc_prices, t5)
    kalshi_t5   = kalshi_client.get_yes_price_at(candles, t5)
    btc_dir_up  = (btc_t5 > btc_t0) if btc_t5 is not None else None
    sig_correct = (btc_dir_up == resolved_yes) if btc_dir_up is not None else None

    # ── Per-mode state ────────────────────────────────────────────────────────
    # additive: accumulates bets unrestricted each interval
    # target:   only bets the gap between current exposure and target
    add_yes_exp, add_no_exp = 0.0, 0.0
    tgt_yes_exp, tgt_no_exp = 0.0, 0.0

    add_yes_bets, add_no_bets = [], []   # list of (stake, kalshi_yes_price)
    tgt_yes_bets, tgt_no_bets = [], []

    minutes = dh_minutes if dh_minutes is not None else DH_MINUTES

    # ── Minute-by-minute loop ─────────────────────────────────────────────────
    for minute in minutes:
        t          = t0 + minute * 60
        btc_t      = btc_data.lookup(btc_prices, t)
        kalshi_yes = kalshi_client.get_yes_price_at(candles, t)

        if btc_t is None or kalshi_yes is None:
            continue
        if not (0.01 < kalshi_yes < 0.99):
            continue

        cumulative_pct = abs(btc_t - btc_t0) / btc_t0 * 100

        if dead_zone > 0 and cumulative_pct < dead_zone:
            continue

        f = sigmoid_btc(cumulative_pct)

        if fair_price_2d:
            fair = get_fair_price_2d(minute, cumulative_pct)
        elif dynamic_fair_price:
            fair = FAIR_PRICE_BY_MINUTE.get(minute, FAIR_PRICE)
        else:
            fair = FAIR_PRICE

        if btc_t > btc_t0:
            mispricing_yes = fair - kalshi_yes
            mispricing_no  = 0.0
            g_yes = sigmoid_mispricing(mispricing_yes)
            computed_yes = STAKE * f * g_yes
            computed_no  = 0.0
        else:
            mispricing_no  = kalshi_yes - (1.0 - fair)
            g_no = sigmoid_mispricing(mispricing_no)
            computed_yes = 0.0
            computed_no  = STAKE * f * g_no

        # Additive: bet the full computed amount each interval
        if computed_yes >= MIN_BET:
            add_yes_bets.append((computed_yes, kalshi_yes))
            add_yes_exp += computed_yes
        if computed_no >= MIN_BET:
            add_no_bets.append((computed_no, kalshi_yes))
            add_no_exp += computed_no

        # Target: only bet the gap to target
        gap_yes = max(0.0, computed_yes - tgt_yes_exp)
        gap_no  = max(0.0, computed_no  - tgt_no_exp)
        if gap_yes >= MIN_BET:
            tgt_yes_bets.append((gap_yes, kalshi_yes))
            tgt_yes_exp += gap_yes
        if gap_no >= MIN_BET:
            tgt_no_bets.append((gap_no, kalshi_yes))
            tgt_no_exp += gap_no

    # ── P&L ──────────────────────────────────────────────────────────────────
    def build_row(yes_bets, no_bets, yes_exp, no_exp):
        pnl_yes = sum(yes_pnl(s, p, resolved_yes) for s, p in yes_bets)
        pnl_no  = sum(no_pnl(s, p, resolved_yes)  for s, p in no_bets)
        pnl_dec = pnl_yes + pnl_no
        total_pnl     = pnl_initial + pnl_dec
        total_wagered = STAKE + yes_exp + no_exp

        # first decision direction: whichever side we bet more on
        if yes_exp > no_exp:
            first_decision = "size_up"
        elif no_exp > yes_exp:
            first_decision = "hedge"
        else:
            first_decision = "none"

        return {
            "timestamp_t0":         open_iso,
            "ticker":               ticker,
            "resolved_yes":         resolved_yes,
            "btc_t0":               round(btc_t0, 2),
            "btc_t5":               round(btc_t5, 2) if btc_t5 else None,
            "btc_direction_up":     btc_dir_up,
            "signal_correct":       sig_correct,
            "kalshi_yes_t0":        round(kalshi_t0, 4),
            "kalshi_yes_t5":        round(kalshi_t5, 4) if kalshi_t5 else None,
            "kalshi_t5_vs_t0_shift": round(kalshi_t5 - kalshi_t0, 4) if kalshi_t5 else None,
            "decision":             first_decision,
            "n_yes_bets":           len(yes_bets),
            "n_no_bets":            len(no_bets),
            "yes_stake":            round(yes_exp, 4),
            "no_stake":             round(no_exp, 4),
            "pnl_initial":          round(pnl_initial, 4),
            "pnl_decision":         round(pnl_dec, 4),
            "total_pnl":            round(total_pnl, 4),
            "total_wagered":        round(total_wagered, 4),
            "roi_pct":              round(total_pnl / total_wagered * 100, 4) if total_wagered else 0,
        }

    additive_row = build_row(add_yes_bets, add_no_bets, add_yes_exp, add_no_exp)
    target_row   = build_row(tgt_yes_bets, tgt_no_bets, tgt_yes_exp, tgt_no_exp)
    return additive_row, target_row


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser(description="Delta hedging simulation")
    parser.add_argument(
        "--minutes", default=DEFAULT_MINUTES,
        help="Minute range to trade, e.g. 5-10 or 3-13 or 1-14 (default: 5-10)"
    )
    parser.add_argument(
        "--dynamic-fair-price", action="store_true",
        help="Use per-minute empirical fair prices instead of fixed FAIR_PRICE=0.698"
    )
    parser.add_argument(
        "--dead-zone", type=float, default=0.0,
        help="Skip intervals where BTC is within ±X%% of cutoff (e.g. 0.05)"
    )
    parser.add_argument(
        "--fair-price-2d", action="store_true",
        help="Use 2D (minute × magnitude bucket) empirical win rates as fair price. "
             "Loads data/logs/minute_analysis_2d.csv. Supersedes --dynamic-fair-price."
    )
    args = parser.parse_args()

    start_min, end_min = map(int, args.minutes.split("-"))
    dh_minutes  = list(range(start_min, end_min + 1))
    is_default  = (args.minutes == DEFAULT_MINUTES)
    dynamic_fp  = args.dynamic_fair_price
    dead_zone   = args.dead_zone
    use_2d      = args.fair_price_2d

    if use_2d:
        csv_2d = os.path.join(LOGS_DIR, "minute_analysis_2d.csv")
        if not os.path.exists(csv_2d):
            print(f"ERROR: {csv_2d} not found. Run analyze_minutes_2d.py first.")
            return
        _load_2d_table(csv_2d)
        print(f"Loaded 2D fair price table: {len(_FAIR_PRICE_2D)} cells from {csv_2d}")

    print(f"DH minutes: T+{start_min} through T+{end_min} ({len(dh_minutes)} intervals)")
    if use_2d:
        print(f"Fair price: 2D empirical (minute × magnitude bucket)")
    elif dynamic_fp:
        print(f"Fair price: dynamic (per-minute empirical)")
    else:
        print(f"Fair price: fixed ({FAIR_PRICE})")
    print(f"Dead zone:  {'none' if dead_zone == 0 else f'±{dead_zone}%'}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    if not markets:
        print("No Kalshi markets found.")
        return

    timestamps = []
    for m in markets:
        try:
            dt = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            timestamps.append(int(dt.timestamp()))
        except Exception:
            pass

    range_start = min(timestamps) - 600
    range_end   = max(timestamps) + 1800

    print(f"Fetching BTC price data for {DATA_DAYS} days from Coinbase...")
    btc_prices = btc_data.fetch_btc_prices(range_start, range_end)
    print(f"Loaded {len(btc_prices)} BTC minute-prices.\n")

    add_results, tgt_results = [], []
    skipped = 0

    print(f"Simulating {len(markets)} windows (both modes simultaneously)...\n")

    for i, market in enumerate(markets):
        add_row, tgt_row = simulate_market_dh(market, btc_prices, dh_minutes, dynamic_fp, dead_zone, use_2d)
        if add_row is None:
            skipped += 1
            continue
        add_results.append(add_row)
        tgt_results.append(tgt_row)

        if (i + 1) % 200 == 0 or i < 3:
            print(f"[{i+1}/{len(markets)}] {market.get('ticker','?')}: "
                  f"add={add_row['total_pnl']:+.2f}  "
                  f"tgt={tgt_row['total_pnl']:+.2f}")

    if not add_results:
        print(f"No results (skipped {skipped}).")
        return

    def write_csv(rows, filename):
        path = os.path.join(LOGS_DIR, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")

    range_part = "" if is_default else f"_{args.minutes.replace('-', '_')}"
    fp_part    = "_2d" if use_2d else ("_dynamic" if dynamic_fp else "")
    dz_part    = f"_dz{str(dead_zone).replace('.', 'p')}" if dead_zone > 0 else ""
    suffix     = range_part + fp_part + dz_part
    write_csv(add_results, f"simulation_results_dh_additive{suffix}.csv")
    write_csv(tgt_results, f"simulation_results_dh_target{suffix}.csv")

    print(f"\nSimulated: {len(add_results)}  Skipped: {skipped}  Minutes: T+{start_min}..T+{end_min}")

    for label, rows in [("Additive", add_results), ("Target", tgt_results)]:
        total_pnl     = sum(r["total_pnl"] for r in rows)
        total_wagered = sum(r["total_wagered"] for r in rows)
        avg_bets      = sum(r["n_yes_bets"] + r["n_no_bets"] for r in rows) / len(rows)
        print(f"\n[{label}]")
        print(f"  Total P&L:      ${total_pnl:+.2f}")
        print(f"  Total wagered:  ${total_wagered:.2f}")
        print(f"  ROI:            {total_pnl/total_wagered*100:+.2f}%")
        print(f"  Avg bets/window: {avg_bets:.1f}")

    print(f"\nRun analyze_dh.py for charts.")


if __name__ == "__main__":
    run()
