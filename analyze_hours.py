"""
Hour-of-day profitability analysis for KXBTC15M delta hedging strategy.

For each UTC hour (0-23), computes across all windows that opened in that hour:
  - Window count
  - Win rate (fraction with total_pnl > 0)
  - Average & total P&L  (DH loop bets only, matches live trader DH mode)
  - Average ROI
  - Average bets per window
  - Average abs BTC move at T+5

Outputs:
  data/logs/hour_analysis.csv
  data/logs/hour_analysis.png

Run: python analyze_hours.py [--csv path/to/simulation.csv]
Default CSV: data/logs/simulation_results_dh_target_4_13_2d.csv
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from collections import defaultdict

from config import LOGS_DIR

DEFAULT_CSV = os.path.join(LOGS_DIR, "simulation_results_dh_target_4_13_2d.csv")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Simulation results CSV to analyse")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run simulate_dh.py first.")
        return

    # ── Load rows ──────────────────────────────────────────────────────────────
    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Loaded {len(rows)} windows from {args.csv}")

    # ── Bucket by UTC hour ─────────────────────────────────────────────────────
    # Use DH-only P&L (pnl_decision / yes_stake+no_stake) — matches live trader.
    by_hour = defaultdict(list)
    for r in rows:
        try:
            dt   = datetime.fromisoformat(r["timestamp_t0"].replace("Z", "+00:00"))
            hour = dt.hour
        except Exception:
            continue

        pnl_dh   = float(r["pnl_decision"])
        wagered  = float(r["yes_stake"]) + float(r["no_stake"])
        n_bets   = int(r["n_yes_bets"]) + int(r["n_no_bets"])

        # BTC magnitude: use btc_t5 vs btc_t0 as a proxy (available in all rows)
        btc_t0 = float(r["btc_t0"])
        btc_t5 = r["btc_t5"]
        if btc_t5:
            abs_pct = abs(float(btc_t5) - btc_t0) / btc_t0 * 100
        else:
            abs_pct = 0.0

        by_hour[hour].append({
            "pnl":     pnl_dh,
            "wagered": wagered,
            "n_bets":  n_bets,
            "abs_pct": abs_pct,
            "win":     pnl_dh > 0,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    hours = list(range(24))
    stats = []
    for h in hours:
        ws = by_hour[h]
        n  = len(ws)
        if n == 0:
            stats.append(None)
            continue

        total_pnl   = sum(w["pnl"] for w in ws)
        total_wag   = sum(w["wagered"] for w in ws)
        win_rate    = sum(w["win"] for w in ws) / n
        avg_pnl     = total_pnl / n
        avg_roi     = (total_pnl / total_wag * 100) if total_wag > 0 else 0
        avg_bets    = sum(w["n_bets"] for w in ws) / n
        avg_abs_pct = sum(w["abs_pct"] for w in ws) / n

        stats.append({
            "hour":       h,
            "n":          n,
            "total_pnl":  round(total_pnl, 2),
            "total_wagered": round(total_wag, 2),
            "win_rate":   round(win_rate, 4),
            "avg_pnl":    round(avg_pnl, 2),
            "avg_roi_pct": round(avg_roi, 2),
            "avg_bets":   round(avg_bets, 2),
            "avg_abs_pct_move": round(avg_abs_pct, 4),
        })

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'Hour (UTC)':>10}  {'N':>5}  {'WinRate':>8}  {'AvgPnL':>8}  "
          f"{'AvgROI%':>8}  {'AvgBets':>8}  {'AvgBTCMov':>10}")
    print("-" * 72)
    for s in stats:
        if s is None:
            continue
        flag = " *" if s["avg_roi_pct"] >= 30 else ""
        print(f"  {s['hour']:02d}:00      {s['n']:>5}  {s['win_rate']:>8.3f}  "
              f"{s['avg_pnl']:>+8.2f}  {s['avg_roi_pct']:>+8.2f}%  "
              f"{s['avg_bets']:>8.2f}  {s['avg_abs_pct_move']:>9.4f}%{flag}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, "hour_analysis.csv")
    valid = [s for s in stats if s is not None]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(valid[0].keys()))
        writer.writeheader()
        writer.writerows(valid)
    print(f"\nSaved: {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Hour-of-Day Analysis — KXBTC15M DH Target (2D fair price, T+4-13)\n"
        f"Based on {len(rows)} windows | P&L = DH loop only (matches live trader)",
        fontsize=12, fontweight="bold"
    )

    hs       = [s["hour"] for s in stats if s]
    ns       = [s["n"] for s in stats if s]
    win_rs   = [s["win_rate"] for s in stats if s]
    avg_rois = [s["avg_roi_pct"] for s in stats if s]
    avg_pnls = [s["avg_pnl"] for s in stats if s]
    avg_bets = [s["avg_bets"] for s in stats if s]
    avg_movs = [s["avg_abs_pct_move"] for s in stats if s]

    def bar_color(values, threshold, above="steelblue", below="salmon"):
        return [above if v >= threshold else below for v in values]

    # Win rate by hour
    ax = axes[0, 0]
    colors = bar_color(win_rs, 0.50)
    ax.bar(hs, win_rs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.50, color="black", linewidth=0.8, linestyle="--", label="50% break-even")
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Hour")
    ax.set_xticks(hs)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for h, v in zip(hs, win_rs):
        ax.text(h, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    # Avg ROI by hour
    ax = axes[0, 1]
    colors = bar_color(avg_rois, 0)
    ax.bar(hs, avg_rois, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Avg ROI %")
    ax.set_title("Average ROI % per Window by Hour")
    ax.set_xticks(hs)
    for h, v in zip(hs, avg_rois):
        ax.text(h, v + (1 if v >= 0 else -2), f"{v:+.0f}%", ha="center", va="bottom", fontsize=6)

    # Avg abs BTC move by hour
    ax = axes[1, 0]
    ax.bar(hs, avg_movs, color="mediumseagreen", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Avg |BTC move| %")
    ax.set_title("Average BTC Volatility by Hour (|T+5 move|)")
    ax.set_xticks(hs)
    for h, v in zip(hs, avg_movs):
        ax.text(h, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    # Window count + avg bets
    ax = axes[1, 1]
    x = np.array(hs)
    w = 0.4
    ax.bar(x - w/2, ns, width=w, label="Windows", color="steelblue", edgecolor="white")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, avg_bets, width=w, label="Avg bets/window", color="darkorange", edgecolor="white")
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Window count", color="steelblue")
    ax2.set_ylabel("Avg bets/window", color="darkorange")
    ax.set_title("Window Count & Avg Bets per Window by Hour")
    ax.set_xticks(hs)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, "hour_analysis.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # ── Best hours summary ─────────────────────────────────────────────────────
    print("\nHours sorted by avg ROI (n >= 200 windows):")
    print(f"  {'Hour':>6}  {'N':>5}  {'WinRate':>8}  {'AvgROI':>8}  {'AvgBets':>8}  {'AvgBTC%':>8}")
    print("  " + "-" * 55)
    for s in sorted(valid, key=lambda x: -x["avg_roi_pct"]):
        if s["n"] >= 200:
            print(f"  {s['hour']:02d}:00   {s['n']:>5}  {s['win_rate']:>8.3f}  "
                  f"{s['avg_roi_pct']:>+8.2f}%  {s['avg_bets']:>8.2f}  "
                  f"{s['avg_abs_pct_move']:>8.4f}%")

    # Active hours filter: what if we only bet during top-N hours by avg ROI?
    print("\nCumulative impact of restricting to best-ROI hours (n >= 200):")
    eligible = sorted([s for s in valid if s["n"] >= 200], key=lambda x: -x["avg_roi_pct"])
    cum_pnl, cum_wag, cum_n = 0, 0, 0
    print(f"  {'Hours kept':>11}  {'Windows':>8}  {'CumPnL':>10}  {'CumROI':>8}")
    print("  " + "-" * 46)
    for i, s in enumerate(eligible):
        cum_pnl += s["total_pnl"]
        cum_wag += s["total_wagered"]
        cum_n   += s["n"]
        cum_roi  = cum_pnl / cum_wag * 100 if cum_wag > 0 else 0
        hours_str = ", ".join(f"{e['hour']:02d}" for e in eligible[:i+1])
        print(f"  top {i+1:2d} hours  {cum_n:>8}  {cum_pnl:>+10.0f}  {cum_roi:>+7.2f}%  [{hours_str}]")


if __name__ == "__main__":
    run()
