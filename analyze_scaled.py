"""
Magnitude-scaled decision leg analysis.

Re-uses simulation_results.csv but scales the decision leg stake
by two independent signals multiplied together:
  f(BTC magnitude)  -- how much did BTC move at T+5?
  g(mispricing)     -- how far is Kalshi's price from fair value?

Scaling modes:
  none     -- flat $100 per leg (baseline)
  linear   -- f only: linear in |btc_pct_move|, capped at 3x
  sigmoid  -- f only: sigmoid in |btc_pct_move|, asymptotes to 3x
  combined -- f * g: sigmoid magnitude x sigmoid mispricing

Run:
  python analyze_scaled.py --scale combined
  python analyze_scaled.py --scale all
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import LOGS_DIR, STAKE, FEE_RATE

CSV_PATH = os.path.join(LOGS_DIR, "simulation_results.csv")

# ── f(BTC magnitude) parameters ───────────────────────────────────────────────
# Empirical: median move=0.056%, 90th=0.18%
LINEAR_REF_PCT   = 0.056
LINEAR_MIN_MULT  = 0.1
LINEAR_MAX_MULT  = 3.0

SIGMOID_CENTER   = 0.10
SIGMOID_K        = 20.0
SIGMOID_MAX_MULT = 3.0

# ── g(mispricing) parameters ──────────────────────────────────────────────────
# Empirical: median mispricing=+0.028, range roughly -0.30 to +0.63
# Fair Yes price given 69.8% signal accuracy
FAIR_PRICE       = 0.698
MISPRICING_K     = 8.0   # steepness: at ±0.15 mispricing → multiplier ~1.6x or ~0.4x
MISPRICING_MAX   = 2.0   # g ranges (0, 2.0); equals 1.0 at mispricing=0 (fair price)


# ── Scaling functions ─────────────────────────────────────────────────────────

def linear_multiplier(abs_pct_move):
    mult = abs_pct_move / LINEAR_REF_PCT
    return float(np.clip(mult, LINEAR_MIN_MULT, LINEAR_MAX_MULT))


def sigmoid_btc(abs_pct_move):
    x = SIGMOID_K * (abs_pct_move - SIGMOID_CENTER)
    return SIGMOID_MAX_MULT / (1.0 + np.exp(-x))


def sigmoid_mispricing(mispricing):
    """
    Centered sigmoid: returns MISPRICING_MAX/2 = 1.0 at mispricing=0 (fair price).
    Approaches MISPRICING_MAX for large positive mispricing (great price for us).
    Approaches 0 for large negative mispricing (bad price for us).
    """
    x = MISPRICING_K * mispricing
    return MISPRICING_MAX / (1.0 + np.exp(-x))


def compute_mispricing(row):
    """
    Positive = Kalshi price is favorable relative to our fair value estimate.
    Size-up: we buy Yes, so good when Yes is cheap (kalshi_yes_t5 < FAIR_PRICE).
    Hedge:   we buy No,  so good when No  is cheap (kalshi_yes_t5 > 1-FAIR_PRICE).
    """
    if row["decision"] == "size_up":
        return FAIR_PRICE - row["kalshi_yes_t5"]
    else:
        return row["kalshi_yes_t5"] - (1.0 - FAIR_PRICE)


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


# ── Core: recompute P&L with scaled stakes ────────────────────────────────────

def apply_scaling(df, scale_mode):
    df = df.copy()
    df["abs_pct_move"] = (df["btc_t5"] - df["btc_t0"]).abs() / df["btc_t0"] * 100
    df["mispricing"]   = df.apply(compute_mispricing, axis=1)

    if scale_mode == "none":
        df["decision_stake"] = STAKE

    elif scale_mode == "linear":
        df["decision_stake"] = df["abs_pct_move"].apply(
            lambda x: STAKE * linear_multiplier(x)
        )

    elif scale_mode == "sigmoid":
        df["decision_stake"] = df["abs_pct_move"].apply(
            lambda x: STAKE * sigmoid_btc(x)
        )

    elif scale_mode == "combined":
        df["f_btc"]       = df["abs_pct_move"].apply(sigmoid_btc)
        df["g_misprice"]  = df["mispricing"].apply(sigmoid_mispricing)
        df["decision_stake"] = STAKE * df["f_btc"] * df["g_misprice"]

    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    def recompute_row(row):
        stake = row["decision_stake"]
        yes_p = row["kalshi_yes_t5"]
        res   = bool(row["resolved_yes"])
        if row["decision"] == "size_up":
            return yes_pnl(stake, yes_p, res)
        else:
            return no_pnl(stake, yes_p, res)

    df["pnl_decision"]  = df.apply(recompute_row, axis=1)
    df["total_pnl"]     = df["pnl_initial"] + df["pnl_decision"]
    df["total_wagered"] = STAKE + df["decision_stake"]

    df = df.sort_values("timestamp_t0").reset_index(drop=True)
    df["cumulative_pnl"]     = df["total_pnl"].cumsum()
    df["cumulative_wagered"] = df["total_wagered"].cumsum()
    df["running_roi"]        = df["cumulative_pnl"] / df["cumulative_wagered"] * 100

    return df


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(df, scale_mode):
    n             = len(df)
    total_pnl     = df["total_pnl"].sum()
    total_wagered = df["total_wagered"].sum()
    signal_acc    = df["signal_correct"].mean() * 100
    win_rate      = (df["total_pnl"] > 0).mean() * 100
    avg_stake     = df["decision_stake"].mean()

    size_ups = df[df["decision"] == "size_up"]
    hedges   = df[df["decision"] == "hedge"]

    print(f"\n{'='*55}")
    print(f"  SCALING MODE: {scale_mode.upper()}  ({n} windows)")
    print(f"{'='*55}")
    print(f"\n[Overall]")
    print(f"  Total P&L:              ${total_pnl:+.2f}")
    print(f"  Total wagered:          ${total_wagered:.2f}")
    print(f"  ROI:                    {total_pnl/total_wagered*100:+.2f}%")
    print(f"  Winning windows:        {win_rate:.1f}%")
    print(f"  Avg P&L per window:     ${df['total_pnl'].mean():+.2f}")
    print(f"  Avg decision stake:     ${avg_stake:.2f}  (baseline=${STAKE:.2f})")

    print(f"\n[Signal accuracy]")
    print(f"  BTC dir correct:        {signal_acc:.1f}%")

    print(f"\n[Decision stake distribution]")
    print(f"  Min:  ${df['decision_stake'].min():.2f}")
    print(f"  25th: ${df['decision_stake'].quantile(0.25):.2f}")
    print(f"  Med:  ${df['decision_stake'].median():.2f}")
    print(f"  75th: ${df['decision_stake'].quantile(0.75):.2f}")
    print(f"  Max:  ${df['decision_stake'].max():.2f}")

    if scale_mode == "combined":
        print(f"\n[Mispricing breakdown]")
        good = df[df["mispricing"] > 0]
        bad  = df[df["mispricing"] <= 0]
        print(f"  Good price windows ({len(good)}): avg stake ${good['decision_stake'].mean():.2f}  "
              f"P&L ${good['total_pnl'].sum():+.2f}")
        print(f"  Bad  price windows ({len(bad)}):  avg stake ${bad['decision_stake'].mean():.2f}  "
              f"P&L ${bad['total_pnl'].sum():+.2f}")

    print(f"\n[Size-up windows: {len(size_ups)}]")
    if len(size_ups):
        su_pnl     = size_ups["pnl_decision"].sum()
        su_wagered = size_ups["decision_stake"].sum()
        print(f"  Decision P&L:  ${su_pnl:+.2f}  ROI: {su_pnl/su_wagered*100:+.2f}%")

    print(f"\n[Hedge windows: {len(hedges)}]")
    if len(hedges):
        h_pnl     = hedges["pnl_decision"].sum()
        h_wagered = hedges["decision_stake"].sum()
        print(f"  Decision P&L:  ${h_pnl:+.2f}  ROI: {h_pnl/h_wagered*100:+.2f}%")

    print(f"\n[Break-even check]")
    print(f"  Initial leg:   ${df['pnl_initial'].sum():+.2f}")
    print(f"  Decision leg:  ${df['pnl_decision'].sum():+.2f}")
    print(f"{'='*55}\n")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(df, scale_mode):
    label = scale_mode.capitalize()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"BTC Kalshi 15-Min Strategy — Scaling: {label} ({len(df)} windows)",
        fontsize=13
    )

    ts = pd.to_datetime(df["timestamp_t0"])

    # 1. Cumulative P&L
    ax = axes[0, 0]
    ax.plot(ts, df["cumulative_pnl"], color="steelblue", linewidth=1.5)
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

    # 3. Bottom-left: depends on mode
    ax = axes[1, 0]
    if scale_mode == "combined":
        # 2D scatter: x=btc_move, y=mispricing, color=stake size
        sc = ax.scatter(
            df["abs_pct_move"], df["mispricing"],
            c=df["decision_stake"], cmap="RdYlGn",
            alpha=0.4, s=10, vmin=0, vmax=df["decision_stake"].max()
        )
        plt.colorbar(sc, ax=ax, label="Decision stake ($)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Stake Surface: BTC Move vs Mispricing")
        ax.set_xlabel("|BTC % move| at T+5")
        ax.set_ylabel("Mispricing (positive = good price)")
    else:
        # 1D: stake vs BTC move
        su = df[df["decision"] == "size_up"]
        hd = df[df["decision"] == "hedge"]
        ax.scatter(su["abs_pct_move"], su["decision_stake"],
                   alpha=0.3, s=8, color="green", label="Size-up")
        ax.scatter(hd["abs_pct_move"], hd["decision_stake"],
                   alpha=0.3, s=8, color="red",   label="Hedge")
        x_range = np.linspace(0, df["abs_pct_move"].max(), 300)
        if scale_mode == "linear":
            y_curve = [STAKE * linear_multiplier(x) for x in x_range]
        elif scale_mode == "sigmoid":
            y_curve = [STAKE * sigmoid_btc(x) for x in x_range]
        else:
            y_curve = [STAKE] * len(x_range)
        ax.plot(x_range, y_curve, color="black", linewidth=1.5, label="Scaling fn")
        ax.set_title("Decision Stake vs BTC % Move")
        ax.set_xlabel("|BTC % move| at T+5")
        ax.set_ylabel("Decision stake ($)")
        ax.legend(fontsize=8)

    # 4. P&L distribution
    ax = axes[1, 1]
    ax.hist(df["total_pnl"], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(df["total_pnl"].mean(), color="green", linestyle="--", linewidth=1,
               label=f"Mean: ${df['total_pnl'].mean():+.2f}")
    ax.set_title("P&L Distribution per Window")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, f"simulation_charts_{scale_mode}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Chart saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def load():
    if not os.path.exists(CSV_PATH):
        print(f"No results at {CSV_PATH}. Run simulate.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp_t0"])
    return df


def run_mode(scale_mode):
    df = load()
    df = apply_scaling(df, scale_mode)
    print_stats(df, scale_mode)
    plot(df, scale_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        choices=["none", "linear", "sigmoid", "combined", "all"],
        default="all",
        help="Scaling mode (default: all)"
    )
    args = parser.parse_args()

    modes = ["none", "linear", "sigmoid", "combined"] if args.scale == "all" else [args.scale]
    for mode in modes:
        run_mode(mode)
