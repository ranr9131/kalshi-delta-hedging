"""
Plot the historical-sim comparison across all reversal-handling variants.

Outputs:
  data/logs/variants_cumulative.png       — cumulative P&L over time per variant
  data/logs/variants_tradeoff.png         — P&L vs wagered (capital-efficiency view)
  data/logs/variants_distribution.png     — per-window P&L histograms
  data/logs/variants_loss_recovery.png    — on baseline-losing windows, P&L per variant
"""
import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

LOGS = "data/logs"
VARIANTS = [
    ("baseline",   "simulation_results_dh_target_4_13_2d.csv",                  "#666666",  2.0),
    ("rh-12",      "simulation_results_dh_target_4_13_2d_rh12.csv",             "#1f77b4",  1.8),
    ("rh-11",      "simulation_results_dh_target_4_13_2d_rh11.csv",             "#2ca02c",  1.3),
    ("rh-10",      "simulation_results_dh_target_4_13_2d_rh10.csv",             "#17becf",  1.3),
    ("ncs-11-.08", "simulation_results_dh_target_4_13_2d_ncs11-0p08.csv",       "#ff7f0e",  1.0),
    ("ncs-10-.10", "simulation_results_dh_target_4_13_2d_ncs10-0p1.csv",        "#ffbb78",  1.0),
    ("ncs11+rh12", "simulation_results_dh_target_4_13_2d_ncs11-0p08_rh12.csv",  "#d62728",  1.3),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["ts"]            = datetime.fromisoformat(r["timestamp_t0"].replace("Z", "+00:00"))
        r["total_pnl"]     = float(r["total_pnl"])
        r["total_wagered"] = float(r["total_wagered"])
    rows.sort(key=lambda r: r["ts"])
    return rows


def main():
    data = {label: load(os.path.join(LOGS, fname)) for label, fname, _, _ in VARIANTS}
    colors = {label: c for label, _, c, _ in VARIANTS}
    widths = {label: w for label, _, _, w in VARIANTS}

    base = data["baseline"]
    base_pnl_by_ticker = {r["ticker"]: r["total_pnl"] for r in base}

    # 1. Cumulative P&L over time
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, _, _, _ in VARIANTS:
        rows = data[label]
        ts   = [r["ts"] for r in rows]
        cum  = np.cumsum([r["total_pnl"] for r in rows])
        ax.plot(ts, cum, color=colors[label], linewidth=widths[label],
                label=f"{label}: ${cum[-1]:+,.0f}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("Cumulative P&L — 6,275 historical windows (target mode, T+4..T+13, 2D fair price)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(LOGS, "variants_cumulative.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"Saved: {out}")

    # 2. P&L vs Wagered
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, _, _, _ in VARIANTS:
        rows = data[label]
        tot_pnl = sum(r["total_pnl"] for r in rows)
        tot_wag = sum(r["total_wagered"] for r in rows)
        roi     = tot_pnl / tot_wag * 100
        ax.scatter([tot_wag/1e6], [tot_pnl/1e3], s=180,
                   color=colors[label], edgecolor="black", zorder=3)
        ax.annotate(f"  {label}\n  {roi:+.1f}% ROI",
                    (tot_wag/1e6, tot_pnl/1e3), fontsize=10,
                    va="center", ha="left")
    wag_grid = np.linspace(2, 6, 50)
    for roi_pct in [20, 25, 30, 35, 40]:
        ax.plot(wag_grid, wag_grid * roi_pct * 10, color="gray",
                linestyle=":", linewidth=0.6, alpha=0.5)
        ax.text(wag_grid[-5], wag_grid[-5] * roi_pct * 10, f"{roi_pct}% ROI",
                fontsize=8, color="gray", alpha=0.7)
    ax.set_xlabel("Total Wagered ($M)")
    ax.set_ylabel("Total P&L ($k)")
    ax.set_title("P&L vs. Capital Wagered — the absolute-vs-ROI tradeoff",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(LOGS, "variants_tradeoff.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"Saved: {out}")

    # 3. Per-window P&L distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.linspace(-1500, 1500, 80)
    for i, (label, _, color, _) in enumerate(VARIANTS):
        if i >= len(axes): break
        ax = axes[i]
        pnls = [r["total_pnl"] for r in data[label]]
        ax.hist(np.clip(pnls, bins[0], bins[-1]), bins=bins,
                color=color, alpha=0.8, edgecolor="white")
        mean = np.mean(pnls)
        worst = min(pnls); best = max(pnls)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(mean, color="red", linestyle="-", linewidth=1.2,
                   label=f"mean ${mean:+.0f}")
        ax.set_title(f"{label}  (worst -${abs(worst):,.0f} / best +${best:,.0f})",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    for i in range(len(VARIANTS), len(axes)):
        axes[i].axis("off")
    fig.suptitle("Per-window P&L distributions (clipped to ±$1,500 for readability)",
                 fontsize=13, fontweight="bold", y=1.00)
    for ax in axes[-3:]:
        ax.set_xlabel("Window P&L ($)")
    for ax in [axes[0], axes[3], axes[6]]:
        ax.set_ylabel("Count")
    plt.tight_layout()
    out = os.path.join(LOGS, "variants_distribution.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"Saved: {out}")

    # 4. Loss recovery on baseline-losing windows
    losing_tickers = {t for t, p in base_pnl_by_ticker.items() if p < 0}
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_labels, bar_values, bar_colors = [], [], []
    base_loss_total = sum(base_pnl_by_ticker[t] for t in losing_tickers)
    for label, _, color, _ in VARIANTS:
        rows_by_t = {r["ticker"]: r["total_pnl"] for r in data[label]}
        s = sum(rows_by_t.get(t, 0) for t in losing_tickers)
        bar_labels.append(label)
        bar_values.append(s)
        bar_colors.append(color)
    bars = ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor="black")
    ax.axhline(base_loss_total, color="black", linestyle="--", linewidth=1,
               label=f"baseline floor: ${base_loss_total:+,.0f}")
    for bar, v in zip(bars, bar_values):
        delta = v - base_loss_total
        ax.text(bar.get_x() + bar.get_width()/2, v,
                f"${v:+,.0f}\n(Δ {'+' if delta>=0 else ''}{delta:,.0f})",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title(f"Variant P&L on the {len(losing_tickers)} baseline-losing windows\n"
                 f"(positive = recovered money the baseline gave up)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sum of P&L on baseline-losing windows ($)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    out = os.path.join(LOGS, "variants_loss_recovery.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
