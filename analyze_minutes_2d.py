"""
2D per-minute signal analysis for KXBTC15M.

For each (minute T+1..T+13, BTC move magnitude bucket) cell, computes:
  - Win rate and 95% CI
  - Average fill price
  - Edge = win rate - fill price
  - Flat $1 ROI

This replaces the 1D FAIR_PRICE_BY_MINUTE table with a 2D lookup that
accounts for both how far into the window we are AND how much BTC has moved.
Cells with tiny moves naturally show ~50% win rate, making the dead zone
implicit in the fair price rather than a hard filter.

Run: python analyze_minutes_2d.py
Outputs:
  data/logs/minute_analysis_2d.csv
  data/logs/minute_analysis_2d.png
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import FEE_RATE, DATA_DAYS, LOGS_DIR, CACHE_DIR

ALL_MINUTES = list(range(1, 15))

BUCKETS = [
    ("0.00-0.05%", 0.000, 0.05),
    ("0.05-0.10%", 0.050, 0.10),
    ("0.10-0.20%", 0.100, 0.20),
    ("0.20-0.50%", 0.200, 0.50),
    ("0.50%+",     0.500, float("inf")),
]
BUCKET_LABELS = [b[0] for b in BUCKETS]
N_BUCKETS = len(BUCKETS)


def get_bucket(pct):
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo <= pct < hi:
            return i
    return N_BUCKETS - 1


def yes_pnl(stake, yes_price, resolved_yes):
    if resolved_yes:
        return stake * (1 - yes_price) / yes_price * (1 - FEE_RATE)
    return -stake


def no_pnl(stake, yes_price, resolved_yes):
    no_price = 1 - yes_price
    if not resolved_yes:
        return stake * (1 - no_price) / no_price * (1 - FEE_RATE)
    return -stake


def run():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    if not markets:
        print("No markets found.")
        return
    print(f"Loaded {len(markets)} settled markets.")

    timestamps = []
    for m in markets:
        try:
            dt = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            timestamps.append(int(dt.timestamp()))
        except Exception:
            pass

    btc_prices = btc_data.fetch_btc_prices(min(timestamps) - 600, max(timestamps) + 1800)
    print(f"Loaded {len(btc_prices)} BTC minute-prices.\n")

    # stats[minute][bucket] = {n, correct, fill_prices, flat_pnl}
    stats = {
        m: [{"n": 0, "correct": 0, "fill_prices": [], "flat_pnl": []}
            for _ in range(N_BUCKETS)]
        for m in ALL_MINUTES
    }

    skipped = 0

    for i, market in enumerate(markets):
        open_iso  = market.get("open_time", "")
        close_iso = market.get("close_time", "")
        result    = market.get("result", "")

        if not open_iso or not close_iso or result not in ("yes", "no"):
            skipped += 1
            continue

        open_dt      = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
        t0           = int(open_dt.timestamp())
        resolved_yes = result == "yes"

        btc_t0 = btc_data.lookup(btc_prices, t0)
        if btc_t0 is None:
            skipped += 1
            continue

        candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, close_iso)
        if not candles:
            skipped += 1
            continue

        for minute in ALL_MINUTES:
            t          = t0 + minute * 60
            btc_t      = btc_data.lookup(btc_prices, t)
            kalshi_yes = kalshi_client.get_yes_price_at(candles, t)

            if btc_t is None or kalshi_yes is None:
                continue
            if not (0.01 < kalshi_yes < 0.99):
                continue

            abs_pct_move   = abs(btc_t - btc_t0) / btc_t0 * 100
            direction_up   = btc_t > btc_t0
            signal_correct = (direction_up == resolved_yes)
            bi             = get_bucket(abs_pct_move)

            fill_price = kalshi_yes if direction_up else (1.0 - kalshi_yes)
            flat = yes_pnl(1.0, kalshi_yes, resolved_yes) if direction_up else \
                   no_pnl(1.0, kalshi_yes, resolved_yes)

            s = stats[minute][bi]
            s["n"]             += 1
            s["correct"]       += int(signal_correct)
            s["fill_prices"].append(fill_price)
            s["flat_pnl"].append(flat)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(markets)} markets...")

    print(f"\nDone. Skipped {skipped} markets.\n")

    # ── Build grids ────────────────────────────────────────────────────────────
    nrows = len(ALL_MINUTES)
    win_rate_grid = np.full((nrows, N_BUCKETS), np.nan)
    edge_grid     = np.full((nrows, N_BUCKETS), np.nan)
    n_grid        = np.zeros((nrows, N_BUCKETS), dtype=int)

    csv_rows = []

    print(f"\n{'Min':>4}", end="")
    for label in BUCKET_LABELS:
        print(f"  {label:>14}", end="")
    print()
    print("-" * (4 + 16 * N_BUCKETS))

    for mi, m in enumerate(ALL_MINUTES):
        print(f"T+{m:2d}", end="")
        for bi in range(N_BUCKETS):
            s = stats[m][bi]
            n = s["n"]
            n_grid[mi, bi] = n

            if n < 20:
                print(f"  {'—':>14}", end="")
                continue

            wr       = s["correct"] / n
            avg_fill = np.mean(s["fill_prices"])
            edge     = wr - avg_fill
            flat_roi = sum(s["flat_pnl"]) / n * 100

            z  = 1.96
            lo = (wr + z**2/(2*n) - z * np.sqrt(wr*(1-wr)/n + z**2/(4*n**2))) / (1 + z**2/n)
            hi = (wr + z**2/(2*n) + z * np.sqrt(wr*(1-wr)/n + z**2/(4*n**2))) / (1 + z**2/n)

            win_rate_grid[mi, bi] = wr
            edge_grid[mi, bi]     = edge

            csv_rows.append({
                "minute":        m,
                "bucket":        BUCKET_LABELS[bi],
                "n":             n,
                "win_rate":      round(wr, 4),
                "win_rate_lo95": round(lo, 4),
                "win_rate_hi95": round(hi, 4),
                "avg_fill":      round(avg_fill, 4),
                "edge":          round(edge, 4),
                "flat_roi_pct":  round(flat_roi, 2),
            })

            print(f"  {wr:.3f}(n={n:5d})", end="")
        print()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if csv_rows:
        csv_path = os.path.join(LOGS_DIR, "minute_analysis_2d.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved: {csv_path}")

    # ── Heatmap plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"2D Signal Analysis — KXBTC15M ({DATA_DAYS}d, {len(markets)} markets)\n"
        "Win Rate, Edge, and Sample Count by (Minute × BTC Move Magnitude)",
        fontsize=13, fontweight="bold"
    )

    y_labels = [f"T+{m}" for m in ALL_MINUTES]

    def annotate(ax, grid, fmt):
        for mi in range(nrows):
            for bi in range(N_BUCKETS):
                val = grid[mi, bi]
                if not np.isnan(val):
                    ax.text(bi, mi, fmt(val), ha="center", va="center",
                            fontsize=7, color="black")

    # Win rate
    ax = axes[0]
    masked = np.ma.array(win_rate_grid, mask=np.isnan(win_rate_grid))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.92)
    ax.set_xticks(range(N_BUCKETS))
    ax.set_xticklabels(BUCKET_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Win Rate")
    plt.colorbar(im, ax=ax, format="%.2f")
    annotate(ax, win_rate_grid, lambda v: f"{v:.3f}")

    # Edge
    ax = axes[1]
    masked_e = np.ma.array(edge_grid, mask=np.isnan(edge_grid))
    im2 = ax.imshow(masked_e, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.18)
    ax.set_xticks(range(N_BUCKETS))
    ax.set_xticklabels(BUCKET_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Edge (Win Rate − Fill Price)")
    plt.colorbar(im2, ax=ax, format="+%.3f")
    annotate(ax, edge_grid, lambda v: f"{v:+.3f}")

    # Sample count
    ax = axes[2]
    im3 = ax.imshow(n_grid, aspect="auto", cmap="Blues")
    ax.set_xticks(range(N_BUCKETS))
    ax.set_xticklabels(BUCKET_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Sample Count (n)")
    plt.colorbar(im3, ax=ax)
    for mi in range(nrows):
        for bi in range(N_BUCKETS):
            n = n_grid[mi, bi]
            ax.text(bi, mi, str(n), ha="center", va="center", fontsize=7,
                    color="white" if n > n_grid.max() * 0.65 else "black")

    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, "minute_analysis_2d.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # ── Summary: which cells have real edge ───────────────────────────────────
    print("\nCells with edge > +0.03 and n >= 100:")
    print(f"  {'Minute':>6}  {'Bucket':>14}  {'n':>6}  {'WinRate':>8}  {'Edge':>8}  {'FlatROI':>8}")
    print("  " + "-" * 60)
    for row in sorted(csv_rows, key=lambda r: -r["edge"]):
        if row["edge"] > 0.03 and row["n"] >= 100:
            print(f"  T+{row['minute']:2d}    {row['bucket']:>14}  {row['n']:>6}  "
                  f"{row['win_rate']:>8.3f}  {row['edge']:>+8.3f}  {row['flat_roi_pct']:>+7.2f}%")


if __name__ == "__main__":
    run()
