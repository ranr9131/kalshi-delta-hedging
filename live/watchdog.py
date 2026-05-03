"""
Watchdog supervisor for trader.py.

Starts the trader, monitors balance_log.csv freshness, and auto-restarts
the trader if it freezes or dies. Preserves the running balance across
restarts by updating STARTING_BALANCE in .env to the latest balance row.

Usage (replaces running trader.py directly):
    cd live && ./venv/bin/python watchdog.py
"""

import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

_dir         = os.path.dirname(os.path.abspath(__file__))
TRADER_PATH  = os.path.join(_dir, "trader.py")
ENV_PATH     = os.path.join(_dir, ".env")
BALANCE_LOG  = os.path.join(_dir, "balance_log.csv")
LOG_PATH     = "/tmp/kalshi_trader.log"
PYTHON       = os.path.join(_dir, "venv", "bin", "python")

STALE_SECS     = 30 * 60   # restart if no balance row written for 30 min
CHECK_INTERVAL = 60

# Safe restart window: minutes within each 15-min market cycle when no bets
# are placed. Bets are placed at T+4..T+13. T+15 = settle = T+0 of next window.
# So safe = T+13 (last bet done) through T+3 of next window (no bets yet).
# In UTC clock minutes: 13, 14, 0, 1, 2, 3 of every 15-min cycle.
SAFE_MINUTES_IN_CYCLE = {13, 14, 0, 1, 2, 3}
MAX_WAIT_FOR_SAFE     = 12 * 60   # cap wait at 12 min (worst case ~9 min)

SSL_CERT_FILE = subprocess.check_output(
    [PYTHON, "-c", "import certifi; print(certifi.where())"], text=True,
).strip()

trader_proc: subprocess.Popen | None = None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def watchdog_log(msg):
    line = f"[{now_iso()}] [watchdog] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_latest_balance() -> float | None:
    if not os.path.exists(BALANCE_LOG):
        return None
    last = None
    with open(BALANCE_LOG, newline="") as f:
        for row in csv.DictReader(f):
            try:
                last = float(row["balance"])
            except (ValueError, KeyError, TypeError):
                pass
    return last


def update_starting_balance(new_balance: float) -> None:
    with open(ENV_PATH) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("STARTING_BALANCE="):
            lines[i] = f"STARTING_BALANCE={new_balance:.4f}\n"
            break
    else:
        lines.append(f"STARTING_BALANCE={new_balance:.4f}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


def start_trader() -> None:
    global trader_proc
    env = os.environ.copy()
    env["SSL_CERT_FILE"] = SSL_CERT_FILE
    log_f = open(LOG_PATH, "a")
    trader_proc = subprocess.Popen(
        [PYTHON, TRADER_PATH],
        cwd=_dir, env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    watchdog_log(f"trader started PID={trader_proc.pid}")


def stop_trader() -> None:
    global trader_proc
    if trader_proc is None:
        return
    if trader_proc.poll() is None:
        watchdog_log(f"stopping trader PID={trader_proc.pid}")
        trader_proc.terminate()
        try:
            trader_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            watchdog_log("trader did not exit in 15s, force-killing")
            trader_proc.kill()
            trader_proc.wait()
    trader_proc = None


def trader_alive() -> bool:
    return trader_proc is not None and trader_proc.poll() is None


def freshness_age_secs() -> float | None:
    """Age (s) of the trader's most recent stdout write.
    The trader writes a log line at least once per minute while healthy."""
    if not os.path.exists(LOG_PATH):
        return None
    return time.time() - os.path.getmtime(LOG_PATH)


def latest_balance_from_window_log() -> float | None:
    """Read the most recent balance from window_log.csv (Raymond's schema includes
    cumulative_pnl; balance = STARTING_BALANCE + cumulative_pnl)."""
    window_log = os.path.join(_dir, "window_log.csv")
    if not os.path.exists(window_log):
        return None
    last_pnl = None
    with open(window_log, newline="") as f:
        for row in csv.DictReader(f):
            try:
                last_pnl = float(row["cumulative_pnl"])
            except (ValueError, KeyError, TypeError):
                pass
    if last_pnl is None:
        return None
    env_now = {}
    try:
        with open(ENV_PATH) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    env_now[k.strip()] = v.strip().strip('"')
    except Exception:
        pass
    start = float(env_now.get("STARTING_BALANCE", "0"))
    return start + last_pnl


def is_safe_restart_minute() -> bool:
    return (datetime.now(timezone.utc).minute % 15) in SAFE_MINUTES_IN_CYCLE


def wait_for_safe_window(reason: str) -> None:
    if is_safe_restart_minute():
        return
    deadline = time.time() + MAX_WAIT_FOR_SAFE
    watchdog_log(f"restart deferred ({reason}) — waiting for safe window (T+13..T+3, no open bets)")
    while time.time() < deadline:
        if is_safe_restart_minute():
            watchdog_log("safe window reached — proceeding with restart")
            return
        time.sleep(10)
    watchdog_log(f"max wait {MAX_WAIT_FOR_SAFE}s exceeded — restarting anyway")


def restart(reason: str) -> None:
    wait_for_safe_window(reason)
    bal = latest_balance_from_window_log()
    if bal is None:
        bal = get_latest_balance()
    if bal is not None:
        update_starting_balance(bal)
        watchdog_log(f"preserving balance ${bal:.2f}")
    stop_trader()
    time.sleep(2)
    start_trader()


def handle_signal(sig, frame):
    watchdog_log(f"received signal {sig}, shutting down trader and exiting")
    stop_trader()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    watchdog_log(f"starting (stale threshold={STALE_SECS}s, check interval={CHECK_INTERVAL}s)")
    start_trader()
    grace_until = time.time() + 90  # don't check freshness for first 90s after start

    while True:
        time.sleep(CHECK_INTERVAL)

        if not trader_alive():
            code = trader_proc.returncode if trader_proc else "?"
            watchdog_log(f"trader died (exit={code})")
            restart(f"process exit {code}")
            grace_until = time.time() + 90
            continue

        if time.time() < grace_until:
            continue

        age = freshness_age_secs()
        if age is not None and age > STALE_SECS:
            watchdog_log(f"trader log stale {age:.0f}s > {STALE_SECS}s — frozen trader detected")
            restart(f"frozen, log stale {age:.0f}s")
            grace_until = time.time() + 90


if __name__ == "__main__":
    main()
