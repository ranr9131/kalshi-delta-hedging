"""
Stop-loss watchdog for the live trader.

Polls Kalshi balance every POLL_SECS. If balance drops below THRESHOLD for
CONFIRM_HITS consecutive checks, sends SIGINT to the trader PID (triggers
the trader's graceful shutdown handler — finishes current window cleanly).

CONFIRM_HITS guards against transient API errors or very brief mid-window
dips while bets are in-flight before settlement returns cash.
"""
import os
import signal
import sys
import time
from datetime import datetime, timezone

# Reuse live infra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import kalshi_auth
import kalshi_trade


# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD     = float(os.environ.get("STOPLOSS_THRESHOLD", "50.0"))
TRADER_PID    = int(os.environ["TRADER_PID"])   # required, set on launch
POLL_SECS     = 30
CONFIRM_HITS  = 2          # require N consecutive low readings before firing
ERROR_BACKOFF = 60         # if API errors, wait this many seconds before retry

_dir = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_dir, "stoploss.log")


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} [stoploss] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    env = dotenv_values(os.path.join(_dir, ".env"))
    api_key_id = env.get("KALSHI_API_KEY_ID", "")
    raw_pem    = env.get("KALSHI_PRIVATE_KEY", "")
    if not api_key_id or not raw_pem:
        log("ERROR: Kalshi credentials missing from .env. Exiting.")
        sys.exit(1)
    key = kalshi_auth.load_private_key(raw_pem)

    # Verify trader process exists at launch
    try:
        os.kill(TRADER_PID, 0)
    except ProcessLookupError:
        log(f"ERROR: trader PID {TRADER_PID} not running. Exiting.")
        sys.exit(1)

    log(f"Stop-loss watchdog starting | trader PID={TRADER_PID} | "
        f"threshold=${THRESHOLD:.2f} | poll={POLL_SECS}s | confirm={CONFIRM_HITS} hits")

    consecutive_low = 0
    while True:
        try:
            bal = kalshi_trade.get_balance(key, api_key_id)
        except Exception as e:
            log(f"balance fetch error: {e} — backing off {ERROR_BACKOFF}s")
            time.sleep(ERROR_BACKOFF)
            continue

        if bal is None:
            log(f"balance unavailable (auth?) — backing off {ERROR_BACKOFF}s")
            time.sleep(ERROR_BACKOFF)
            continue

        # Is the trader still alive?
        try:
            os.kill(TRADER_PID, 0)
        except ProcessLookupError:
            log(f"Trader PID {TRADER_PID} no longer running. Exiting watchdog.")
            return

        if bal < THRESHOLD:
            consecutive_low += 1
            log(f"BALANCE LOW: ${bal:.2f} < ${THRESHOLD:.2f}  ({consecutive_low}/{CONFIRM_HITS} confirmations)")
            if consecutive_low >= CONFIRM_HITS:
                log(f"STOP-LOSS TRIPPED. Sending SIGINT to trader PID {TRADER_PID} (graceful shutdown).")
                try:
                    os.kill(TRADER_PID, signal.SIGINT)
                    log("SIGINT sent. Trader will exit after current window.")
                except Exception as e:
                    log(f"Failed to signal trader: {e}")
                return
        else:
            if consecutive_low > 0:
                log(f"balance recovered: ${bal:.2f} >= ${THRESHOLD:.2f}, reset counter")
            consecutive_low = 0
            # Periodic heartbeat
            if int(time.time()) % 300 < POLL_SECS:
                log(f"balance OK: ${bal:.2f}")

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
