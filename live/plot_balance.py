"""
Live paper-balance + real Kalshi balance chart in Pacific time.
- Solid line: trader's tracked balance (from balance_log.csv)
- Dashed line: live Kalshi cash balance (polled directly from Kalshi API every 30s)

Usage (in a separate terminal, with the trader already running):
    cd live && ./venv/bin/python plot_balance.py
"""

import csv
import os
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from dotenv import dotenv_values

import kalshi_auth
import kalshi_trade

KALSHI_BASE = "https://api.elections.kalshi.com"

_dir = os.path.dirname(os.path.abspath(__file__))
WINDOW_LOG_PATH = os.path.join(_dir, "window_log.csv")
ENV_PATH        = os.path.join(_dir, ".env")

env = dotenv_values(ENV_PATH)
STARTING_BALANCE = float(env.get("STARTING_BALANCE", "1000.0"))
REFRESH_SECS     = 5
KALSHI_POLL_SECS = 5
PT = ZoneInfo("America/Los_Angeles")

START_TIME = datetime.now(PT)

API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")
PRIV_KEY   = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"]) if env.get("KALSHI_PRIVATE_KEY") else None

kalshi_times: list = []
kalshi_balances: list = []   # total portfolio value = cash + open position MtM
last_kalshi_poll: float = 0.0


def fetch_portfolio_value() -> float | None:
    """Cash + sum of market_exposure across all open positions, in dollars."""
    if PRIV_KEY is None or API_KEY_ID == "":
        return None
    cash = kalshi_trade.get_balance(PRIV_KEY, API_KEY_ID)
    if cash is None:
        return None
    try:
        path    = "/trade-api/v2/portfolio/positions"
        headers = kalshi_auth.make_auth_headers(PRIV_KEY, API_KEY_ID, "GET", path)
        resp    = requests.get(KALSHI_BASE + path, headers=headers,
                                params={"limit": 50}, timeout=10)
        resp.raise_for_status()
        positions = resp.json().get("market_positions", [])
        pos_value = sum(p.get("market_exposure", 0) / 100
                        for p in positions if p.get("position", 0) != 0)
    except Exception:
        pos_value = 0.0
    return cash + pos_value


def read_tracked():
    """Read settled windows from window_log.csv. balance = STARTING_BALANCE + cumulative_pnl."""
    times    = [START_TIME]
    balances = [STARTING_BALANCE]
    triggers = ["start"]
    if not os.path.exists(WINDOW_LOG_PATH):
        return times, balances, triggers
    with open(WINDOW_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts  = datetime.fromisoformat(row["settlement_ts"].replace("Z", "+00:00"))
                bal = STARTING_BALANCE + float(row["cumulative_pnl"])
            except Exception:
                continue
            times.append(ts.astimezone(PT))
            balances.append(bal)
            triggers.append("settle")
    return times, balances, triggers


def maybe_poll_kalshi():
    global last_kalshi_poll
    now = time.time()
    if PRIV_KEY is None or API_KEY_ID == "":
        return
    if now - last_kalshi_poll < KALSHI_POLL_SECS:
        return
    val = fetch_portfolio_value()
    last_kalshi_poll = now
    if val is None:
        return
    kalshi_times.append(datetime.now(PT))
    kalshi_balances.append(val)


def main():
    plt.ion()
    fig, ax = plt.subplots(figsize=(13, 7))

    while plt.fignum_exists(fig.number):
        maybe_poll_kalshi()
        times, balances, triggers = read_tracked()
        current   = balances[-1]
        change    = current - STARTING_BALANCE
        pct       = change / STARTING_BALANCE * 100
        sign      = "+" if change >= 0 else ""
        n_settles = sum(1 for t in triggers if t == "settle")
        color     = "tab:green" if change >= 0 else "tab:red"

        ax.clear()
        ax.plot(times, balances, color=color, linewidth=2, marker="o", markersize=4,
                label="trader tracked")
        if kalshi_times:
            ax.plot(kalshi_times, kalshi_balances, color="tab:blue",
                    linestyle="--", linewidth=1.6, marker="s", markersize=3,
                    label="Kalshi portfolio = cash + open positions (live)")
        ax.axhline(STARTING_BALANCE, color="gray", linestyle=":", linewidth=1)

        settle_t = [t for t, tr in zip(times, triggers) if tr == "settle"]
        settle_b = [b for b, tr in zip(balances, triggers) if tr == "settle"]
        if settle_t:
            ax.scatter(settle_t, settle_b, color="black", zorder=5, s=70,
                       marker="D", label="settlement")
            for t, b in zip(settle_t, settle_b):
                ax.annotate(f"${b:,.2f}", (t, b),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=9, fontweight="bold")

        kalshi_str = f"   |   Portfolio: ${kalshi_balances[-1]:,.2f}" if kalshi_balances else ""
        ax.set_title(
            f"${current:,.2f}   {sign}${change:,.2f} ({sign}{pct:.2f}%)   "
            f"·  {n_settles} settled  ·  start ${STARTING_BALANCE:,.0f}{kalshi_str}",
            fontsize=13, color=color, loc="left", pad=12,
        )
        ax.set_xlabel("Time (Pacific)")
        ax.set_ylabel("Balance ($)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=PT))
        fig.autofmt_xdate()

        x_lo = times[0]
        x_hi = max(times[-1], datetime.now(PT)) + timedelta(minutes=2)
        if (x_hi - x_lo) < timedelta(minutes=30):
            x_hi = x_lo + timedelta(minutes=30)
        ax.set_xlim(x_lo, x_hi)

        all_y = balances + kalshi_balances + [STARTING_BALANCE]
        y_min, y_max = min(all_y), max(all_y)
        pad = max(20.0, (y_max - y_min) * 0.20)
        ax.set_ylim(y_min - pad, y_max + pad)

        fig.tight_layout()
        plt.pause(REFRESH_SECS)


if __name__ == "__main__":
    main()
