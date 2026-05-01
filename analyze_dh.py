"""
Analysis and plots for the delta hedging simulation.
Reads simulation_results_dh_additive.csv and simulation_results_dh_target.csv,
plus the baseline simulation_results.csv for comparison.

Run after simulate_dh.py.

Examples:
  python analyze_dh.py
  python analyze_dh.py --minutes 1-13
  python analyze_dh.py --minutes 1-13 --dynamic-fair-price
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import LOGS_DIR

BASELINE_CSV = os.path.join(LOGS_DIR, "simulation_results.csv")

DEFAULT_MINUTES = "5-10"


def csv_paths(minutes_arg, dynamic):
    if minutes_arg == DEFAULT_MINUTES:
        range_part = ""
    else:
        range_part = f"_{minutes_arg.replace('-', '_')}"
    fp_part = "_dynamic" if dynamic else ""
    suffix  = range_part + fp_part
    add = os.path.join(LOGS_DIR, f"simulation_results_dh_additive{suffix}.csv")
    tgt = os.path.join(LOGS_DIR, f"simulation_results_dh_target{suffix}.csv")
    return add, tgt


def load(path):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["timestamp_t0"])
    df = df.sort_values("timestamp_t0").reset_index(drop=True)
    df["cumulative_pnl"]     = df["total_pnl"].cumsum()
    df["cumulative_wagered"] = df["total_wagered"].cumsum()
    df["running_roi"]        = df["cumulative_pnl"] / df["cumulative_wagered"] * 100
    return df


def print_stats(df, label):
    n             = len(df)
    total_pnl     = df["total_pnl"].sum()
    total_wagered = df["total_wagered"].sum()
    win_rate      = (df["total_pnl"] > 0).mean() * 100
    signal_acc    = df["signal_correct"].mean() * 100 if "signal_correct" in df else float("nan")

    print(f"\n{'='*55}")
    print(f"  {label}  ({n} windows)")
    print(f"{'='*55}")
    print(f"  Total P&L:         ${total_pnl:+.2f}")
    print(f"  Total wagered:     ${total_wagered:.2f}")
    print(f"  ROI:               {total_pnl/total_wagered*100:+.2f}%")
    print(f"  Winning windows:   {win_rate:.1f}%")
    print(f"  Avg P&L/window:    ${df['total_pnl'].mean():+.2f}")
    print(f"  Signal accuracy:   {signal_acc:.1f}%")

    if "n_yes_bets" in df.columns:
        avg_bets = (df["n_yes_bets"] + df["n_no_bets"]).mean()
        avg_yes  = df["yes_stake"].mean()
        avg_no   = df["no_stake"].mean()
        print(f"  Avg bets/window:   {avg_bets:.2f}")
        print(f"  Avg Yes stake:     ${avg_yes:.2f}")
        print(f"  Avg No stake:      ${avg_no:.2f}")
        print(f"  Avg total wagered: ${df['total_wagered'].mean():.2f}")

        zero_bets = (df["n_yes_bets"] + df["n_no_bets"] == 0).sum()
        print(f"  Windows w/ 0 DH bets: {zero_bets} ({zero_bets/n*100:.1f}%)")

    print(f"  Initial leg P&L:   ${df['pnl_initial'].sum():+.2f}")
    if "pnl_decision" in df.columns:
        print(f"  Decision leg P&L:  ${df['pnl_decision'].sum():+.2f}")


def plot_comparison(base_df, add_df, tgt_df):
    """Single chart comparing all three strategies' cumulative P&L."""
    fig, ax = plt.subplots(figsize=(13, 6))

    ts_base = pd.to_datetime(base_df["timestamp_t0"])
    ts_add  = pd.to_datetime(add_df["timestamp_t0"])
    ts_tgt  = pd.to_datetime(tgt_df["timestamp_t0"])

    ax.plot(ts_base, base_df["cumulative_pnl"], color="gray",      linewidth=1.2, label="Baseline (single T+5 decision)")
    ax.plot(ts_add,  add_df["cumulative_pnl"],  color="steelblue", linewidth=1.5, label="DH Additive")
    ax.plot(ts_tgt,  tgt_df["cumulative_pnl"],  color="darkorange",linewidth=1.5, label="DH Target Position")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)

    ax.set_title("Cumulative P&L — Strategy Comparison", fontsize=13)
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(LOGS_DIR, "dh_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_detail(df, label, filename):
    """4-panel detail plot for one DH mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Delta Hedging — {label} ({len(df)} windows)", fontsize=13)

    ts = pd.to_datetime(df["timestamp_t0"])

    # 1. Cumulative P&L
    ax = axes[0, 0]
    ax.plot(ts, df["cumulative_pnl"], linewidth=1.5, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", rotation=30)

    # 2. Rolling signal accuracy
    ax = axes[0, 1]
    rolling_acc = df["signal_correct"].rolling(50, min_periods=10).mean() * 100
    ax.plot(ts, rolling_acc, color="darkorange", linewidth=1.5)
    ax.axhline(50, color="red",   linestyle="--", linewidth=0.8, label="Random (50%)")
    ax.axhline(70, color="green", linestyle="--", linewidth=0.8, label="Theoretical max (70%)")
    ax.set_title("T+5 Signal Accuracy (rolling 50)")
    ax.set_ylabel("% correct")
    ax.set_ylim(30, 90)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", rotation=30)

    # 3. Bets per window distribution
    ax = axes[1, 0]
    total_bets = df["n_yes_bets"] + df["n_no_bets"]
    max_bets = int(total_bets.max()) + 1
    ax.hist(total_bets, bins=range(0, max_bets + 1), color="steelblue",
            alpha=0.8, edgecolor="white", align="left")
    ax.set_title("Number of DH Bets per Window")
    ax.set_xlabel("Bets placed")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, max_bets))

    # 4. P&L distribution
    ax = axes[1, 1]
    ax.hist(df["total_pnl"], bins=60, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red",   linestyle="--", linewidth=1)
    ax.axvline(df["total_pnl"].mean(), color="green", linestyle="--", linewidth=1,
               label=f"Mean: ${df['total_pnl'].mean():+.2f}")
    ax.set_title("P&L Distribution per Window")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(LOGS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DH simulation results")
    parser.add_argument("--minutes", default=DEFAULT_MINUTES,
                        help="Minute range used in simulate_dh.py, e.g. 1-13 (default: 5-10)")
    parser.add_argument("--dynamic-fair-price", action="store_true",
                        help="Load the _dynamic variant files")
    args = parser.parse_args()

    add_csv, tgt_csv = csv_paths(args.minutes, args.dynamic_fair_price)
    fp_label = "dynamic fair price" if args.dynamic_fair_price else f"fixed fair price"
    range_label = f"T+{args.minutes.replace('-', '..T+')} | {fp_label}"

    # Output filenames include suffix so they don't overwrite baseline charts
    if args.minutes == DEFAULT_MINUTES:
        range_part = ""
    else:
        range_part = f"_{args.minutes.replace('-', '_')}"
    fp_part  = "_dynamic" if args.dynamic_fair_price else ""
    out_sfx  = range_part + fp_part

    base_df = load(BASELINE_CSV)
    add_df  = load(add_csv)
    tgt_df  = load(tgt_csv)

    print_stats(base_df, "BASELINE (single T+5 decision)")
    print_stats(add_df,  f"DH ADDITIVE  [{range_label}]")
    print_stats(tgt_df,  f"DH TARGET    [{range_label}]")

    plot_comparison(base_df, add_df, tgt_df)
    plot_detail(add_df, f"Additive — {range_label}",        f"dh_detail_additive{out_sfx}.png")
    plot_detail(tgt_df, f"Target Position — {range_label}", f"dh_detail_target{out_sfx}.png")
