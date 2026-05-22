"""
80-20 train-test split sanity check for the 2D fair-price strategy.

Splits all settled KXBTC15M windows chronologically:
  - First 80% (oldest) → build the 2D fair-price table
  - Last 20% (newest)  → run the strategy on these UNSEEN windows using
                         the table built from train-only data

Compares the test-set ROI to the in-sample ROI reported by simulate_dh.
A large drop suggests look-ahead bias in the fair-price table.

Run:  python train_test_split.py
"""
import os
import csv
from datetime import datetime
import numpy as np

import kalshi_client
import btc_data
import simulate_dh
from config import STAKE, FEE_RATE, DATA_DAYS, LOGS_DIR

ALL_MINUTES = list(range(1, 15))
BUCKETS = [
    (0.000, 0.05),
    (0.050, 0.10),
    (0.100, 0.20),
    (0.200, 0.50),
    (0.500, float("inf")),
]
N_BUCKETS  = len(BUCKETS)
DH_MINUTES = list(range(4, 14))   # T+4..T+13


def get_bucket(pct):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= pct < hi:
            return i
    return N_BUCKETS - 1


def build_fair_table(train_markets, btc_prices, min_n=20):
    """Build a 2D fair-price table from train_markets only.
    Returns {(minute, bucket_idx): (win_rate, avg_fill, n)}."""
    stats = {
        m: [{"n": 0, "correct": 0, "fills": []}
            for _ in range(N_BUCKETS)]
        for m in ALL_MINUTES
    }
    for market in train_markets:
        open_iso = market.get("open_time", "")
        result   = market.get("result", "")
        if result not in ("yes", "no") or not open_iso:
            continue
        try:
            t0 = int(datetime.fromisoformat(open_iso.replace("Z", "+00:00")).timestamp())
        except Exception:
            continue
        btc_t0 = btc_data.lookup(btc_prices, t0)
        if btc_t0 is None:
            continue
        candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, market.get("close_time",""))
        if not candles:
            continue
        resolved_yes = result == "yes"
        for minute in ALL_MINUTES:
            t = t0 + minute * 60
            btc_t = btc_data.lookup(btc_prices, t)
            kalshi_yes = kalshi_client.get_yes_price_at(candles, t)
            if btc_t is None or kalshi_yes is None:
                continue
            if not (0.01 < kalshi_yes < 0.99):
                continue
            abs_pct = abs(btc_t - btc_t0) / btc_t0 * 100
            direction_up = btc_t > btc_t0
            bi = get_bucket(abs_pct)
            fill = kalshi_yes if direction_up else (1.0 - kalshi_yes)
            s = stats[minute][bi]
            s["n"] += 1
            s["correct"] += int(direction_up == resolved_yes)
            s["fills"].append(fill)

    table = {}
    for m in ALL_MINUTES:
        for bi in range(N_BUCKETS):
            s = stats[m][bi]
            if s["n"] < min_n:
                continue
            table[(m, bi)] = (
                s["correct"] / s["n"],
                float(np.mean(s["fills"])),
                s["n"],
            )
    return table


def main():
    print("Loading markets and BTC prices…")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    print(f"  {len(markets)} settled markets")

    # Sort chronologically (oldest → newest)
    markets_dated = []
    for m in markets:
        try:
            dt = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            markets_dated.append((dt, m))
        except Exception:
            pass
    markets_dated.sort(key=lambda x: x[0])
    markets_sorted = [m for _, m in markets_dated]

    timestamps = [int(d.timestamp()) for d, _ in markets_dated]
    btc_prices = btc_data.fetch_btc_prices(min(timestamps) - 600, max(timestamps) + 1800)
    print(f"  {len(btc_prices)} BTC minute-prices")

    # 80/20 split (oldest 80% = train, newest 20% = test)
    split = int(len(markets_sorted) * 0.8)
    train, test = markets_sorted[:split], markets_sorted[split:]
    print(f"\nTrain: {len(train)} windows ({markets_dated[0][0].date()} → {markets_dated[split-1][0].date()})")
    print(f"Test:  {len(test)} windows ({markets_dated[split][0].date()} → {markets_dated[-1][0].date()})")

    # Build TRAIN-ONLY 2D table
    print("\nBuilding fair-price table from TRAIN windows only…")
    train_table = build_fair_table(train, btc_prices)
    print(f"  {len(train_table)} cells")

    # Patch simulate_dh to use this in-memory table
    simulate_dh._FAIR_PRICE_2D.clear()
    simulate_dh._FAIR_PRICE_2D.update(train_table)
    # Force the 2D path even though we didn't load from disk
    print("\nRunning sim on TEST windows (out-of-sample)…")

    rh_minute   = 10
    rh_trigger  = 1.0
    time_decay  = True
    early_skip_minute, early_skip_pct = 5, 0.025

    rows = []
    for i, market in enumerate(test):
        _add, tgt = simulate_dh.simulate_market_dh(
            market, btc_prices, DH_MINUTES,
            dynamic_fair_price=False,
            dead_zone=0.0,
            fair_price_2d=True,
            ncs_minute=None, ncs_threshold_pct=0.0,
            rh_minute=rh_minute, rh_min_trigger=rh_trigger,
            time_decay=time_decay,
            early_skip_minute=early_skip_minute,
            early_skip_pct=early_skip_pct,
        )
        if tgt is None:
            continue
        rows.append(tgt)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(test)}] …")

    # ── Compare ─────────────────────────────────────────────────────────────
    total_pnl = sum(r["total_pnl"] for r in rows)
    total_wag = sum(r["total_wagered"] for r in rows)
    pnls = sorted(r["total_pnl"] for r in rows)
    losers = sum(1 for p in pnls if p < 0)

    print()
    print("=" * 60)
    print(f"OUT-OF-SAMPLE TEST RESULTS  (n={len(rows)} windows)")
    print("=" * 60)
    print(f"  Total P&L:        ${total_pnl:+,.0f}")
    print(f"  Total wagered:    ${total_wag:,.0f}")
    print(f"  ROI on wagered:   {total_pnl/total_wag*100:+.2f}%")
    print(f"  Losing windows:   {losers}/{len(rows)} ({losers/len(rows)*100:.1f}%)")
    print(f"  Avg per window:   ${total_pnl/len(rows):+.2f}")
    print(f"  Worst window:     ${pnls[0]:,.0f}")
    print(f"  Best window:      ${pnls[-1]:,.0f}")
    print()

    # Load the in-sample (full-data) result for comparison
    insample_path = os.path.join(LOGS_DIR, "simulation_results_dh_target_4_13_2d_td_es5-0p025.csv")
    if os.path.exists(insample_path):
        ins = list(csv.DictReader(open(insample_path)))
        ins_p = sum(float(r["total_pnl"]) for r in ins)
        ins_w = sum(float(r["total_wagered"]) for r in ins)
        ins_n = len(ins)
        print("=" * 60)
        print(f"IN-SAMPLE BASELINE  (full data, n={ins_n})")
        print("=" * 60)
        print(f"  Total P&L:        ${ins_p:+,.0f}")
        print(f"  ROI on wagered:   {ins_p/ins_w*100:+.2f}%")
        print(f"  Avg per window:   ${ins_p/ins_n:+.2f}")
        print()
        print("=" * 60)
        print("EDGE EROSION (in-sample → out-of-sample)")
        print("=" * 60)
        roi_in  = ins_p/ins_w*100
        roi_out = total_pnl/total_wag*100
        delta   = roi_out - roi_in
        print(f"  ROI delta:        {delta:+.2f}pp")
        if delta < -10:
            print(f"  → SIGNIFICANT look-ahead bias suggested.")
        elif delta < -3:
            print(f"  → Modest edge erosion. Real but smaller than sim.")
        else:
            print(f"  → Edge holds out-of-sample. Strategy has genuine signal.")


if __name__ == "__main__":
    main()
