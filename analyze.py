"""
Reads simulation_results.csv and produces stats + charts.
Run after simulate.py.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import LOGS_DIR

CSV_PATH = os.path.join(LOGS_DIR, "simulation_results.csv")


def load():
    if not os.path.exists(CSV_PATH):
        print(f"No results found at {CSV_PATH}. Run simulate.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp_t0"])
    df = df.sort_values("timestamp_t0").reset_index(drop=True)
    df["cumulative_pnl"] = df["total_pnl"].cumsum()
    df["cumulative_wagered"] = df["total_wagered"].cumsum()
    df["running_roi"] = df["cumulative_pnl"] / df["cumulative_wagered"] * 100
    return df


def print_stats(df):
    n = len(df)
    total_pnl = df["total_pnl"].sum()
    total_wagered = df["total_wagered"].sum()
    signal_acc = df["signal_correct"].mean() * 100
    win_rate = (df["total_pnl"] > 0).mean() * 100

    size_ups = df[df["decision"] == "size_up"]
    hedges = df[df["decision"] == "hedge"]

    print(f"\n{'='*55}")
    print(f"  DETAILED ANALYSIS  ({n} windows over ~{n//96} days)")
    print(f"{'='*55}")
    print(f"\n[Overall]")
    print(f"  Total P&L:            ${total_pnl:+.2f}")
    print(f"  Total wagered:        ${total_wagered:.2f}")
    print(f"  ROI:                  {total_pnl/total_wagered*100:+.2f}%")
    print(f"  Winning windows:      {win_rate:.1f}%")
    print(f"  Avg P&L per window:   ${df['total_pnl'].mean():+.2f}")

    print(f"\n[Signal quality -- BTC price T+5 vs T+0]")
    print(f"  BTC dir predicts Kalshi 15-min: {signal_acc:.1f}%")
    print(f"  (50% = random, ~70% = theoretical max under random walk)")
    btc_up_pct = df["btc_direction_up"].mean() * 100
    print(f"  BTC rose at T+5:   {btc_up_pct:.1f}%  (size-up cases)")
    print(f"  BTC fell at T+5:   {100-btc_up_pct:.1f}%  (hedge cases)")

    # Kalshi odds movement
    shifted_up = (df["kalshi_t5_vs_t0_shift"] > 0.02).mean() * 100
    shifted_down = (df["kalshi_t5_vs_t0_shift"] < -0.02).mean() * 100
    no_shift = 100 - shifted_up - shifted_down
    print(f"\n[Kalshi odds movement T+0 -> T+5]")
    print(f"  Moved up >2%:    {shifted_up:.1f}%  (market noticed BTC up)")
    print(f"  Moved down >2%:  {shifted_down:.1f}%  (market noticed BTC down)")
    print(f"  Stayed flat:     {no_shift:.1f}%  (potential lag window)")

    # Lag analysis: when Kalshi didn't move, was there still an edge?
    flat = df[df["kalshi_t5_vs_t0_shift"].abs() < 0.02]
    if len(flat) > 0:
        flat_pnl = flat["total_pnl"].sum()
        flat_wagered = flat["total_wagered"].sum()
        print(f"  Flat-odds P&L:   ${flat_pnl:+.2f}  ROI: {flat_pnl/flat_wagered*100:+.2f}%")

    # Agreement vs disagreement between BTC signal and Kalshi odds movement
    if "btc_direction_up" in df.columns and "kalshi_t5_vs_t0_shift" in df.columns:
        agree = df[
            (df["btc_direction_up"] & (df["kalshi_t5_vs_t0_shift"] > 0.02)) |
            (~df["btc_direction_up"] & (df["kalshi_t5_vs_t0_shift"] < -0.02))
        ]
        disagree = df[
            (df["btc_direction_up"] & (df["kalshi_t5_vs_t0_shift"] < -0.02)) |
            (~df["btc_direction_up"] & (df["kalshi_t5_vs_t0_shift"] > 0.02))
        ]
        print(f"\n[BTC signal vs Kalshi odds alignment]")
        if len(agree):
            a_pnl = agree["total_pnl"].sum()
            a_roi = a_pnl / agree["total_wagered"].sum() * 100
            print(f"  BTC+Kalshi agree ({len(agree)}):    P&L=${a_pnl:+.2f}  ROI={a_roi:+.2f}%")
        if len(disagree):
            d_pnl = disagree["total_pnl"].sum()
            d_roi = d_pnl / disagree["total_wagered"].sum() * 100
            print(f"  BTC+Kalshi disagree ({len(disagree)}):  P&L=${d_pnl:+.2f}  ROI={d_roi:+.2f}%")

    # Fade Kalshi overreaction analysis
    print(f"\n[Fade Kalshi overreaction (no BTC signal needed)]")
    extreme_up = df[df["kalshi_yes_t5"] > 0.65]
    extreme_dn = df[df["kalshi_yes_t5"] < 0.35]
    neutral    = df[(df["kalshi_yes_t5"] >= 0.35) & (df["kalshi_yes_t5"] <= 0.65)]
    if len(extreme_up):
        eu_no_pnl = sum(no_pnl_calc(100, p, r) for p, r in zip(extreme_up["kalshi_yes_t5"], extreme_up["resolved_yes"]))
        print(f"  Kalshi >0.65 Yes ({len(extreme_up)}): fade-No ROI = {eu_no_pnl/len(extreme_up)/100*100:+.2f}%")
    if len(extreme_dn):
        ed_yes_pnl = sum(yes_pnl_calc(100, p, r) for p, r in zip(extreme_dn["kalshi_yes_t5"], extreme_dn["resolved_yes"]))
        print(f"  Kalshi <0.35 Yes ({len(extreme_dn)}): fade-Yes ROI = {ed_yes_pnl/len(extreme_dn)/100*100:+.2f}%")
    if len(neutral):
        print(f"  Kalshi 35-65% ({len(neutral)}): neutral zone")

    print(f"\n[Break-even check]")
    initial_pnl = df["pnl_initial"].sum()
    decision_pnl = df["pnl_decision"].sum()
    print(f"  Initial bet leg:  ${initial_pnl:+.2f}")
    print(f"  Decision leg:     ${decision_pnl:+.2f}")
    if decision_pnl > 0:
        print(f"  [YES] Decision leg is profitable -- strategy has edge at T+5")
    else:
        print(f"  [NO]  Decision leg is losing -- Kalshi odds already reflect BTC move")

    print(f"\n{'='*55}\n")


def yes_pnl_calc(stake, yes_price, resolved_yes):
    fee = 0.07
    if resolved_yes:
        return stake * (1 - yes_price) / yes_price * (1 - fee)
    return -stake


def no_pnl_calc(stake, yes_price, resolved_yes):
    fee = 0.07
    no_price = 1 - yes_price
    if not resolved_yes:
        return stake * (1 - no_price) / no_price * (1 - fee)
    return -stake


def plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"BTC Kalshi 15-Min Strategy Simulation ({len(df)} windows)", fontsize=14)

    # 1. Cumulative P&L over time
    ax = axes[0, 0]
    ax.plot(df["timestamp_t0"], df["cumulative_pnl"], color="steelblue", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", rotation=30)

    # 2. Signal accuracy: was T+5 direction right?
    ax = axes[0, 1]
    # Rolling 50-window accuracy
    rolling_acc = df["signal_correct"].rolling(50, min_periods=10).mean() * 100
    ax.plot(df["timestamp_t0"], rolling_acc, color="darkorange", linewidth=1.5)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, label="Random (50%)")
    ax.axhline(70, color="green", linestyle="--", linewidth=0.8, label="Theoretical max (70%)")
    ax.set_title("T+5 Signal Accuracy (rolling 50)")
    ax.set_ylabel("% correct")
    ax.set_ylim(30, 90)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", rotation=30)

    # 3. Kalshi Yes price at T+5 vs decision
    ax = axes[1, 0]
    su = df[df["decision"] == "size_up"]["kalshi_yes_t5"]
    hd = df[df["decision"] == "hedge"]["kalshi_yes_t5"]
    ax.hist(su, bins=30, alpha=0.6, color="green", label=f"Size-up (n={len(su)})")
    ax.hist(hd, bins=30, alpha=0.6, color="red", label=f"Hedge (n={len(hd)})")
    ax.axvline(0.70, color="black", linestyle="--", linewidth=1, label="Fair T+5 price (≈0.70)")
    ax.set_title("Kalshi Yes Price at T+5")
    ax.set_xlabel("Yes price")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # 4. P&L per window distribution
    ax = axes[1, 1]
    ax.hist(df["total_pnl"], bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(df["total_pnl"].mean(), color="green", linestyle="--", linewidth=1,
               label=f"Mean: ${df['total_pnl'].mean():+.2f}")
    ax.set_title("P&L Distribution per Window")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, "simulation_charts.png")
    plt.savefig(out_path, dpi=150)
    print(f"Charts saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    df = load()
    print_stats(df)
    plot(df)
