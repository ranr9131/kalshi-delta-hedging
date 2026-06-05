"""
Poll Kalshi for each sniped market's outcome and write settlements.csv.

Settlement logic per snipe row:
  - reconstruct each market's close_time from any snipe row (close_iso = ts + minutes_left)
  - wait SETTLEMENT_DELAY_SEC after close to give Kalshi time to finalize
  - GET /trade-api/v2/markets/{ticker}
      - if status in {"finalized", "settled"} → record result + write row
      - if status == "active" but close_time should have passed → likely
        ticker-recycle by Kalshi: mark UNKNOWN so we don't keep retrying

Output columns: ticker, result, settled_at_iso, raw_status, expiration_value

We never modify snipes.csv (the sniper writes to it).  The dashboard joins
on ticker at read time.
"""
from __future__ import annotations
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.environ.get("LOG_PATH",    os.path.join(ROOT, "snipes.csv"))
RESULT_PATH = os.environ.get("RESULT_PATH", os.path.join(ROOT, "settlements.csv"))
# Extra snipe-log files to pull tickers from (e.g. v2 sniper).  Comma-separated.
EXTRA_LOGS  = [p for p in os.environ.get("EXTRA_LOGS", "").split(",") if p]
# Sensible default: include snipes_v2.csv if it exists.
_default_v2 = os.path.join(ROOT, "snipes_v2.csv")
if not EXTRA_LOGS and os.path.exists(_default_v2):
    EXTRA_LOGS.append(_default_v2)

KALSHI_BASE = "https://api.elections.kalshi.com"
POLL_INTERVAL_SEC      = float(os.environ.get("POLL_INTERVAL_SEC", "30"))
SETTLEMENT_DELAY_SEC   = float(os.environ.get("SETTLEMENT_DELAY_SEC", "60"))
RECYCLE_GRACE_SEC      = float(os.environ.get("RECYCLE_GRACE_SEC", "900"))

FIELDS = ["ticker", "result", "settled_at_iso", "raw_status", "expiration_value", "close_iso"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  settler  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("settler")


def _read_snipes() -> list[dict]:
    """Read snipes from LOG_PATH plus any EXTRA_LOGS (e.g. snipes_v2.csv).
    Filter out malformed rows (missing ticker or None keys from extra columns)."""
    rows: list[dict] = []
    for path in [LOG_PATH] + EXTRA_LOGS:
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    r.pop(None, None)
                    if r.get("ticker"):
                        rows.append(r)
        except Exception:
            pass
    return rows


def _read_settlements() -> Dict[str, dict]:
    if not os.path.exists(RESULT_PATH):
        return {}
    with open(RESULT_PATH, newline="") as f:
        return {r["ticker"]: r for r in csv.DictReader(f)}


def _derive_close(ts_iso: str, minutes_left: str | float) -> datetime | None:
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return t + timedelta(minutes=float(minutes_left))
    except Exception:
        return None


def _append(rows: list[dict]):
    need_header = not os.path.exists(RESULT_PATH)
    with open(RESULT_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _query(ticker: str) -> dict | None:
    try:
        r = requests.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}", timeout=10)
        if not r.ok:
            return None
        return r.json().get("market", {})
    except Exception as e:
        log.warning(f"query {ticker} failed: {e}")
        return None


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def loop():
    log.info(f"settler start  LOG={LOG_PATH}  OUT={RESULT_PATH}")
    while True:
        try:
            snipes = _read_snipes()
            settled = _read_settlements()
            ticker_close: Dict[str, datetime] = {}
            for r in snipes:
                t = r.get("ticker")
                if not t or t in settled:
                    continue
                close = _derive_close(r.get("ts_iso", ""), r.get("minutes_left", "0"))
                if close is None:
                    continue
                # keep the LATEST close estimate across all snipes of this ticker
                cur = ticker_close.get(t)
                if cur is None or close > cur:
                    ticker_close[t] = close

            now = datetime.now(timezone.utc)
            ready = [(t, c) for t, c in ticker_close.items()
                     if (now - c).total_seconds() >= SETTLEMENT_DELAY_SEC]
            if ready:
                log.info(f"checking {len(ready)} markets (of {len(ticker_close)} unsettled)")

            new = []
            for ticker, close_t in ready:
                m = _query(ticker)
                if m is None:
                    continue
                status = (m.get("status") or "").lower()
                result = (m.get("result") or "").lower()
                exp    = m.get("expiration_value") or m.get("settled_to_yes_price")

                # Kalshi uses three terminal statuses: "determined" appears as
                # soon as the outcome is known, "settled" / "finalized" come
                # later when the cash is distributed.  Any of the three with a
                # valid result is good enough for our PnL accounting.
                if status in ("determined", "finalized", "settled") and result in ("yes", "no"):
                    new.append({
                        "ticker": ticker,
                        "result": result,
                        "settled_at_iso": _iso(now),
                        "raw_status": status,
                        "expiration_value": str(exp) if exp is not None else "",
                        "close_iso": _iso(close_t),
                    })
                    log.info(f"settled {ticker} → {result.upper()}  (status={status})")
                elif (now - close_t).total_seconds() > RECYCLE_GRACE_SEC:
                    # Kalshi appears to have rotated the ticker; mark unknown so
                    # we stop retrying.  We record empty result so dashboard PnL
                    # treats these as "unsettleable".
                    new.append({
                        "ticker": ticker,
                        "result": "unknown",
                        "settled_at_iso": _iso(now),
                        "raw_status": status or "missing",
                        "expiration_value": "",
                        "close_iso": _iso(close_t),
                    })
                    log.warning(f"timeout {ticker} after {RECYCLE_GRACE_SEC}s → unknown")

            if new:
                _append(new)
        except Exception as e:
            log.warning(f"settler iter err: {e}")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    loop()
