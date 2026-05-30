"""
Cumulative + daily P&L plot for the live paper trader.

Reads live/window_log.csv. Only counts windows on/after START_FROM_ISO.

Two modes:
    python3 plot_pnl.py
        One-shot. Saves data/logs/pnl_plot.png and exits.

    python3 plot_pnl.py --watch [SECS]
        Opens a live window that re-renders every SECS seconds
        (default 60). Pulls fresh data from window_log.csv each tick.
        Close the window to stop.
"""

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT       = Path(__file__).resolve().parent
LOG_PATH   = ROOT / "live" / "window_log.csv"                  # DH trader log
NN_LOG     = ROOT / "nn"   / "shadow_log.csv"                  # NN v1 multi shadow
V2S_LOG    = ROOT / "nn"   / "shadow_log_v2_small.csv"         # NN v2_small shadow
OUT_PATH   = ROOT / "data" / "logs" / "pnl_plot.png"

# Droplet sync — pulls both DH window_log.csv and NN shadow_log.csv from
# the droplet so the local plot reflects what production is logging.
# No-op if --sync is not passed.
DROPLET_HOST = "root@167.172.154.205"
DROPLET_DH   = "/root/kalshi-delta-hedging/live/window_log.csv"
DROPLET_NN   = "/root/kalshi-delta-hedging/nn/shadow_log.csv"


def _rsync(remote, local):
    try:
        subprocess.run(
            ["rsync", "-q", "-e", "ssh -o ConnectTimeout=5",
             f"{DROPLET_HOST}:{remote}", str(local)],
            check=True, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [sync {local.name}] {type(e).__name__}; using local copy")


def sync_from_droplet():
    """Pull both DH window_log and NN shadow_log from the droplet."""
    _rsync(DROPLET_DH, LOG_PATH)
    _rsync(DROPLET_NN, NN_LOG)

BACKTEST_ROI = 0.2286  # honest baseline matching live risk controls (max-hedge-fill 0.80, max-legs 2)

# Track only windows from this point forward — set to the trader restart that
# established the canonical config (SIDE_FILTER off, RH=10, edge filter on,
# fresh 2D table). Pre-restart data is from inferior strategy versions.
START_FROM_ISO = "2026-05-29T03:29:57+00:00"


def load():
    start = datetime.fromisoformat(START_FROM_ISO)
    rows = []
    if not LOG_PATH.exists():
        return rows
    with open(LOG_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(r["window_ts"].replace("Z", "+00:00"))
                if ts < start:
                    continue
                rows.append((ts, float(r["total_wagered"]), float(r["total_pnl"])))
            except (ValueError, KeyError):
                continue
    rows.sort()
    return rows


def _load_shadow_csv(path):
    """Return list of (ts, wagered, pnl) tuples from any shadow CSV schema."""
    start = datetime.fromisoformat(START_FROM_ISO)
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(r["window_ts"].replace("Z", "+00:00"))
                if ts < start:
                    continue
                wag = float(r.get("total_wagered") or r.get("stake") or 0)
                pnl = float(r.get("total_pnl")     or r.get("pnl")   or 0)
                rows.append((ts, wag, pnl))
            except (ValueError, KeyError):
                continue
    rows.sort()
    return rows


def load_nn():
    """NN v1 multi shadow (post-restart)."""
    return _load_shadow_csv(NN_LOG)


def load_v2s():
    """NN v2_small shadow."""
    return _load_shadow_csv(V2S_LOG)


def render(fig, ax1, ax2, rows, nn_rows=None, v2s_rows=None):
    ax1.clear()
    ax2.clear()
    start = datetime.fromisoformat(START_FROM_ISO)
    now   = datetime.now(timezone.utc)
    nn_rows = nn_rows or []
    v2s_rows = v2s_rows or []

    if not rows:
        ax1.axhline(0, color="black", linewidth=0.6)
        ax1.set_xlim(start, now)
        ax1.set_ylim(-10, 10)
        ax1.set_title(f"Live paper P&L — no windows yet since "
                      f"{start.strftime('%Y-%m-%d %H:%M UTC')}\n"
                      f"Last refresh: {now.strftime('%H:%M:%S UTC')}",
                      fontsize=11)
        ax1.set_ylabel("Cumulative P&L ($)")
        ax1.text(0.5, 0.5, "Waiting for first window to settle…",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=14, color="gray")
        ax1.grid(True, alpha=0.3)
        ax2.set_visible(False)
        return

    ax2.set_visible(True)
    times = [r[0] for r in rows]
    cum_pnl = []
    cum_exp = []
    rp = 0.0; re = 0.0
    dh_acted = 0; dh_wins = 0; dh_wagered = 0.0
    for _, w, p in rows:
        rp += p; re += w * BACKTEST_ROI
        cum_pnl.append(rp)
        cum_exp.append(re)
        if w > 0:
            dh_acted += 1
            dh_wagered += w
            if p > 0: dh_wins += 1
    dh_winpct = (dh_wins / dh_acted * 100) if dh_acted else 0
    dh_roi    = (cum_pnl[-1] / dh_wagered * 100) if dh_wagered else 0

    # ── Top: cumulative ────────────────────────────────────────────────────
    ax1.plot(times, cum_pnl, color="tab:blue", linewidth=2,
             label="DH trader (live paper)", zorder=4)
    ax1.plot(times, cum_exp, color="tab:gray", linewidth=1.3, linestyle="--",
             label=f"DH backtest expectation ({BACKTEST_ROI*100:+.1f}% × wagered)",
             zorder=2)

    def _line_summary(rows_list, color, label_prefix, zorder):
        if not rows_list:
            return "", 0, 0, 0.0
        times = [r[0] for r in rows_list]
        cum, rp = [], 0.0
        acted = wins = 0; wag = 0.0
        for _, stake, p in rows_list:
            rp += p; cum.append(rp)
            if stake > 0:
                acted += 1; wag += stake
                if p > 0: wins += 1
        last = cum[-1]
        wp = wins/acted*100 if acted else 0
        roi = last/wag*100 if wag else 0
        ax1.plot(times, cum, color=color, linewidth=2,
                 label=f"{label_prefix} ({len(rows_list)} windows)", zorder=zorder)
        summ = (f"\n{label_prefix:<5s}: ${last:+7.2f}  on {len(rows_list):>3}w  "
                f"({acted} bets, {wp:.0f}% win, ROI {roi:+.1f}%)")
        return summ, acted, wins, wag

    nn_summary,  _, _, _ = _line_summary(nn_rows,  "tab:purple", "NN v1",      zorder=5)
    v2s_summary, _, _, _ = _line_summary(v2s_rows, "tab:orange", "NN v2_small", zorder=6)
    nn_summary += v2s_summary

    ax1.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax1.fill_between(times, cum_pnl, 0,
                     where=[v >= 0 for v in cum_pnl],
                     color="tab:green", alpha=0.08, zorder=1)
    ax1.fill_between(times, cum_pnl, 0,
                     where=[v < 0 for v in cum_pnl],
                     color="tab:red", alpha=0.08, zorder=1)

    ax1.set_ylabel("Cumulative P&L ($)".replace("$", r"\$"))
    title = (
        f"DH trader vs NN shadow — KXBTC15M paper  "
        f"(tracking from {start.strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"refresh {now.strftime('%H:%M:%S UTC')})\n"
        f"DH:   ${cum_pnl[-1]:+7.2f}  on {len(rows):>3}w  "
        f"({dh_acted} bets, {dh_winpct:.0f}% win, ROI {dh_roi:+.1f}%, "
        f"backtest-expected ${cum_exp[-1]:+.2f})"
        f"{nn_summary}"
    )
    ax1.set_title(title.replace("$", r"\$"),
                  fontsize=10, loc="left", family="monospace")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # ── Bottom: daily bars ─────────────────────────────────────────────────
    day_pnl = {}
    for ts, _, p in rows:
        d = ts.date()
        day_pnl[d] = day_pnl.get(d, 0.0) + p
    days       = sorted(day_pnl.keys())
    day_values = [day_pnl[d] for d in days]
    day_dts    = [datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in days]

    colors = ["tab:green" if v >= 0 else "tab:red" for v in day_values]
    ax2.bar(day_dts, day_values, color=colors, alpha=0.8, width=0.7)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Daily P&L ($)")
    ax2.set_xlabel("Date (UTC)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for dt, v in zip(day_dts, day_values):
        ax2.text(dt, v + (0.5 if v >= 0 else -1.5), f"{v:+.1f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", nargs="?", const="60", default=None,
                   help="Live-refresh mode. Optional SECS interval (default 60).")
    p.add_argument("--sync", action="store_true",
                   help="Pull window_log.csv from the droplet before rendering "
                        "(only if you're running the trader on the droplet).")
    args = p.parse_args()

    watch = args.watch is not None
    interval = float(args.watch) if watch else 0
    do_sync = args.sync

    # Watch mode is now headless — saves PNG on a timer, no GUI window to close.
    matplotlib.use("Agg")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.35},
    )

    if not watch:
        if do_sync:
            sync_from_droplet()
        render(fig, ax1, ax2, load(), load_nn(), load_v2s())
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
        rows = load()
        msg = f"Saved: {OUT_PATH}  windows={len(rows)}"
        if rows:
            cum = sum(r[2] for r in rows)
            exp = sum(r[1] * BACKTEST_ROI for r in rows)
            msg += f"  cum=${cum:+.2f}  expected=${exp:+.2f}"
        print(msg)
        return

    # Headless watch loop — re-render and save PNG every `interval` seconds.
    # No interactive window; user views the PNG file directly. Stops on ctrl-C.
    print(f"Watch mode (headless) — saving PNG every {interval:.0f}s to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import time as _t
    try:
        while True:
            if do_sync:
                sync_from_droplet()
            rows = load()
            render(fig, ax1, ax2, rows, load_nn(), load_v2s())
            try:
                fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
            except Exception as e:
                print(f"  [save] {type(e).__name__}: {e}")
            _t.sleep(interval)
    except KeyboardInterrupt:
        pass
    print("Watch stopped.")


if __name__ == "__main__":
    main()
