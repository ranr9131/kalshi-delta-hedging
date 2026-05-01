"""
Per-minute signal analysis for KXBTC15M.

For every minute T+1 through T+14 across all historical markets, computes:
  - Win rate (signal accuracy) at each minute
  - Average Kalshi fill price at each minute
  - Edge = win rate - fill price (where the money actually is)
  - Flat $1 ROI and sigmoid-scaled ROI by minute

Also computes the empirical fair price per minute, which replaces the fixed
FAIR_PRICE=0.698 constant with a time-varying one calibrated to the actual data.

Run: python analyze_minutes.py
Outputs:
  data/logs/minute_analysis.png
  data/logs/minute_analysis.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import STAKE, FEE_RATE, DATA_DAYS, LOGS_DIR, CACHE_DIR

# ── Strategy params (same as simulate_dh.py) ──────────────────────────────────
FAIR_PRICE       = 0.698
SIGMOID_CENTER   = 0.10
SIGMOID_K        = 20.0
SIGMOID_MAX_MULT = 3.0
MISPRICING_K     = 8.0
MISPRICING_MAX   = 2.0
MIN_BET          = 5.0

ALL_MINUTES = list(range(1, 15))   # T+1 through T+14


# ── Helpers ───────────────────────────────────────────────────────────────────

def sigmoid_btc(x):
    return SIGMOID_MAX_MULT / (1.0 + np.exp(-SIGMOID_K * (x - SIGMOID_CENTER)))


def sigmoid_mispricing(m):
    return MISPRICING_MAX / (1.0 + np.exp(-MISPRICING_K * m))


def yes_pnl(stake, yes_price, resolved_yes):
    if resolved_yes:
        return stake * (1 - yes_price) / yes_price * (1 - FEE_RATE)
    return -stake


def no_pnl(stake, yes_price, resolved_yes):
    no_price = 1 - yes_price
    if not resolved_yes:
        return stake * (1 - no_price) / no_price * (1 - FEE_RATE)
    return -stake


# ── Core ──────────────────────────────────────────────────────────────────────

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

    # Per-minute accumulators
    stats = {m: {
        "n":               0,
        "correct":         0,
        "fill_prices":     [],   # effective fill price (yes price for YES bets, 1-yes for NO)
        "flat_pnl":        [],
        "scaled_pnl":      [],
        "scaled_stakes":   [],
        "btc_moves":       [],
    } for m in ALL_MINUTES}

    skipped = 0

    for i, market in enumerate(markets):
        open_iso  = market.get("open_time", "")
        close_iso = market.get("close_time", "")
        result    = market.get("result", "")
        ticker    = market["ticker"]

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

        candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
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

            direction_up   = btc_t > btc_t0
            signal_correct = (direction_up == resolved_yes)
            abs_pct_move   = abs(btc_t - btc_t0) / btc_t0 * 100

            # Effective fill price (fraction of $1 we spend per contract)
            fill_price = kalshi_yes if direction_up else (1.0 - kalshi_yes)

            # Flat $1 bet P&L
            if direction_up:
                flat = yes_pnl(1.0, kalshi_yes, resolved_yes)
            else:
                flat = no_pnl(1.0, kalshi_yes, resolved_yes)

            # Sigmoid-scaled bet P&L (using FAIR_PRICE — intentionally fixed for comparison)
            f = sigmoid_btc(abs_pct_move)
            if direction_up:
                mispricing = FAIR_PRICE - kalshi_yes
            else:
                mispricing = kalshi_yes - (1.0 - FAIR_PRICE)
            g     = sigmoid_mispricing(mispricing)
            stake = STAKE * f * g

            if direction_up:
                scaled = yes_pnl(stake, kalshi_yes, resolved_yes)
            else:
                scaled = no_pnl(stake, kalshi_yes, resolved_yes)

            s = stats[minute]
            s["n"]             += 1
            s["correct"]       += int(signal_correct)
            s["fill_prices"].append(fill_price)
            s["flat_pnl"].append(flat)
            s["scaled_pnl"].append(scaled)
            s["scaled_stakes"].append(stake)
            s["btc_moves"].append(abs_pct_move)

        if (i + 1) % 300 == 0:
            print(f"  Processed {i+1}/{len(markets)} markets...")

    print(f"\nDone. Skipped {skipped} markets.\n")

    # ── Compute summary stats ─────────────────────────────────────────────────
    minutes     = ALL_MINUTES
    win_rates   = []
    avg_fills   = []
    edges       = []
    flat_rois   = []
    scaled_rois = []
    avg_moves   = []
    sample_ns   = []
    # 95% CI on win rate (Wilson interval approx)
    win_rate_lo = []
    win_rate_hi = []

    csv_rows = []

    for m in minutes:
        s = stats[m]
        n = s["n"]
        if n == 0:
            for lst in [win_rates, avg_fills, edges, flat_rois, scaled_rois, avg_moves, sample_ns, win_rate_lo, win_rate_hi]:
                lst.append(np.nan)
            continue

        wr        = s["correct"] / n
        avg_fill  = np.mean(s["fill_prices"])
        edge      = wr - avg_fill
        flat_roi  = sum(s["flat_pnl"]) / n * 100   # % per bet at $1
        total_scaled_pnl    = sum(s["scaled_pnl"])
        total_scaled_wagered = sum(s["scaled_stakes"])
        scaled_roi = (total_scaled_pnl / total_scaled_wagered * 100) if total_scaled_wagered > 0 else 0
        avg_move  = np.mean(s["btc_moves"])

        # Wilson 95% CI
        z   = 1.96
        lo  = (wr + z**2/(2*n) - z * np.sqrt(wr*(1-wr)/n + z**2/(4*n**2))) / (1 + z**2/n)
        hi  = (wr + z**2/(2*n) + z * np.sqrt(wr*(1-wr)/n + z**2/(4*n**2))) / (1 + z**2/n)

        win_rates.append(wr)
        avg_fills.append(avg_fill)
        edges.append(edge)
        flat_rois.append(flat_roi)
        scaled_rois.append(scaled_roi)
        avg_moves.append(avg_move)
        sample_ns.append(n)
        win_rate_lo.append(lo)
        win_rate_hi.append(hi)

        csv_rows.append({
            "minute":        m,
            "n":             n,
            "win_rate":      round(wr, 4),
            "win_rate_lo95": round(lo, 4),
            "win_rate_hi95": round(hi, 4),
            "avg_fill":      round(avg_fill, 4),
            "edge":          round(edge, 4),
            "flat_roi_pct":  round(flat_roi, 2),
            "scaled_roi_pct": round(scaled_roi, 2),
            "avg_btc_move":  round(avg_move, 4),
        })

        print(f"T+{m:2d}: n={n:4d}  win={wr:.3f} [{lo:.3f},{hi:.3f}]  "
              f"fill={avg_fill:.3f}  edge={edge:+.3f}  "
              f"flat_roi={flat_roi:+.2f}%  scaled_roi={scaled_roi:+.2f}%")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(LOGS_DIR, "minute_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Per-Minute Signal Analysis — KXBTC15M ({DATA_DAYS}d, {len(markets)} markets)",
        fontsize=14, fontweight="bold"
    )

    x = np.array(minutes)
    wr  = np.array(win_rates)
    lo  = np.array(win_rate_lo)
    hi  = np.array(win_rate_hi)
    af  = np.array(avg_fills)
    ed  = np.array(edges)
    fr  = np.array(flat_rois)
    sr  = np.array(scaled_rois)
    ns  = np.array(sample_ns, dtype=float)

    # ── Panel 1: Win rate + fill price ────────────────────────────────────────
    ax = axes[0, 0]
    ax.fill_between(x, lo, hi, alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(x, wr, "o-", color="steelblue", lw=2, label="Win rate")
    ax.plot(x, af, "s--", color="darkorange", lw=1.5, label="Avg fill price")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.axvspan(5, 10, alpha=0.07, color="green", label="Current DH range")
    ax.set_title("Win Rate vs Fill Price by Minute")
    ax.set_xlabel("Minutes into window (T+m)")
    ax.set_ylabel("Probability / Price")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Edge (win rate − fill price) ─────────────────────────────────
    ax = axes[0, 1]
    colors = ["forestgreen" if e > 0 else "tomato" for e in ed]
    ax.bar(x, ed * 100, color=colors, alpha=0.8, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(5, 10, alpha=0.07, color="green")
    ax.set_title("Edge per Minute (Win Rate − Fill Price)")
    ax.set_xlabel("Minutes into window (T+m)")
    ax.set_ylabel("Edge (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Flat $1 ROI by minute ───────────────────────────────────────
    ax = axes[1, 0]
    colors = ["forestgreen" if r > 0 else "tomato" for r in fr]
    ax.bar(x, fr, color=colors, alpha=0.8, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(5, 10, alpha=0.07, color="green", label="Current DH range")
    ax.set_title("Flat $1 Bet ROI by Minute")
    ax.set_xlabel("Minutes into window (T+m)")
    ax.set_ylabel("ROI per $1 bet (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Sigmoid-scaled ROI by minute ────────────────────────────────
    ax = axes[1, 1]
    colors = ["forestgreen" if r > 0 else "tomato" for r in sr]
    ax.bar(x, sr, color=colors, alpha=0.8, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(5, 10, alpha=0.07, color="green", label="Current DH range")
    ax.set_title("Sigmoid-Scaled ROI by Minute (current strategy params)")
    ax.set_xlabel("Minutes into window (T+m)")
    ax.set_ylabel("ROI (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    # Sample size annotation on panel 4
    for xi, ni in zip(x, ns):
        if not np.isnan(ni):
            ax.annotate(f"n={int(ni)}", (xi, 0), textcoords="offset points",
                        xytext=(0, -18), ha="center", fontsize=6, color="gray")

    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, "minute_analysis.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

    # ── Print empirical fair prices (time-varying FAIR_PRICE) ─────────────────
    print("\nEmpirical fair price by minute (replace fixed FAIR_PRICE=0.698):")
    print(f"{'Minute':>8}  {'Win Rate':>10}  {'95% CI':>18}  {'Edge':>8}  {'Flat ROI':>10}")
    print("-" * 62)
    for row in csv_rows:
        m = row["minute"]
        print(f"  T+{m:2d}    {row['win_rate']:>8.3f}    "
              f"[{row['win_rate_lo95']:.3f}, {row['win_rate_hi95']:.3f}]    "
              f"{row['edge']:>+7.3f}    {row['flat_roi_pct']:>+7.2f}%")


if __name__ == "__main__":
    run()
