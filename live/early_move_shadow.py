#!/usr/bin/env python3
"""Read-only Kalshi paper trader for the surviving 15-minute early-move rule.

The rule deliberately mirrors the backtest and does not scan for the best
intraminute quote:

* Observe the 2:00, 3:00, and 4:00 elapsed marks of aligned BTC/ETH/SOL/XRP
  15-minute contracts.
* Let BTC choose the side.  Its favored-side midpoint must be at least 90c.
* The same side on an alt must be executable at least 5c below BTC's midpoint.
* Paper-buy 10 contracts using full visible-book VWAP, once per alt/window.
* Charge the order-rounded Kalshi taker fee and settle against the final result.

This module imports the market-data WebSocket only.  It has no POST/DELETE
requests and does not import any order-routing module.

Collect:  python3 early_move_shadow.py
Report:   python3 early_move_shadow.py --report
Test:     python3 early_move_shadow.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

import requests
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_auth
import kalshi_orderbook


ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://api.elections.kalshi.com"

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}
ALTS = ("ETH", "SOL", "XRP")


def _minutes_from_env() -> tuple[int, ...]:
    raw = os.environ.get("EARLY_SAMPLE_MINUTES", "2,3,4")
    vals = sorted({int(v.strip()) for v in raw.split(",") if v.strip()})
    if not vals or any(v <= 0 or v >= 15 for v in vals):
        raise ValueError("EARLY_SAMPLE_MINUTES must contain elapsed minutes 1..14")
    return tuple(vals)


SAMPLE_MINUTES = _minutes_from_env()
BTC_THRESHOLD = float(os.environ.get("EARLY_BTC_THRESHOLD", "0.90"))
MIN_GAP = float(os.environ.get("EARLY_MIN_GAP", "0.05"))
PAPER_SIZE = float(os.environ.get("EARLY_PAPER_SIZE", "10"))
SAMPLE_GRACE_SEC = float(os.environ.get("EARLY_SAMPLE_GRACE_SEC", "5"))
REFRESH_SEC = float(os.environ.get("EARLY_REFRESH_SEC", "10"))
TICK_SEC = float(os.environ.get("EARLY_TICK_SEC", "0.20"))
# A quiet but valid book can have an old last delta.  Zero means do not reject
# it; age is always logged so stricter freshness can be analyzed afterward.
MAX_BOOK_AGE = float(os.environ.get("EARLY_MAX_BOOK_AGE", "0"))
SETTLE_POLL_SEC = float(os.environ.get("EARLY_SETTLE_POLL_SEC", "30"))

SAMPLE_CSV = os.environ.get(
    "EARLY_SAMPLE_CSV", os.path.join(ROOT, "early_move_samples.csv")
)
SIGNAL_CSV = os.environ.get(
    "EARLY_SIGNAL_CSV", os.path.join(ROOT, "early_move_signals.csv")
)
SETTLEMENT_CSV = os.environ.get(
    "EARLY_SETTLEMENT_CSV", os.path.join(ROOT, "early_move_settlements.csv")
)

SAMPLE_COLS = [
    "sample_ts", "sample_iso", "open_ts", "close_ts", "btc_ticker",
    "alt_ticker", "asset", "minute", "elapsed_sec", "sample_delay_sec",
    "side", "btc_favored_mid", "btc_yes_bid", "btc_yes_ask", "alt_ask",
    "gap", "btc_book_age", "alt_book_age", "requested_size", "visible_size",
    "paper_vwap", "decision",
]
SIGNAL_COLS = [
    "signal_id", "signal_ts", "signal_iso", "open_ts", "close_ts",
    "btc_ticker", "alt_ticker", "asset", "minute", "elapsed_sec",
    "sample_delay_sec", "side", "btc_favored_mid", "btc_yes_bid",
    "btc_yes_ask", "alt_top_ask", "gap", "btc_book_age", "alt_book_age",
    "requested_size", "fill_size", "fill_vwap", "fee_dollars",
    "total_cost_dollars",
]
SETTLEMENT_COLS = [
    "signal_id", "settled_ts", "settled_iso", "alt_ticker", "asset", "side",
    "result", "win", "fill_vwap", "fill_size", "fee_dollars",
    "payout_dollars", "pnl_dollars", "pnl_c_per_contract", "roi",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s early-move %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("early_move_shadow")


def _f(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _parse_iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _fmt(value, digits=4):
    return "" if value is None or not math.isfinite(float(value)) else round(float(value), digits)


def _append_csv(path: str, columns: list[str], row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})
        fh.flush()
        os.fsync(fh.fileno())


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def favored_side(yes_bid: float, yes_ask: float) -> tuple[str, float]:
    """Return BTC's favored side and that side's midpoint."""
    if not (0 <= yes_bid <= yes_ask <= 1):
        raise ValueError("invalid YES quote")
    yes_mid = (yes_bid + yes_ask) / 2.0
    return ("yes", yes_mid) if yes_mid >= 0.5 else ("no", 1.0 - yes_mid)


def book_vwap(levels: Iterable[tuple[float, float]], size: float):
    """Return (VWAP, fills, visible size), requiring the entire requested size."""
    if size <= 0:
        raise ValueError("size must be positive")
    remaining = size
    notional = 0.0
    fills = []
    visible = 0.0
    clean = []
    for price, qty in levels:
        price, qty = float(price), float(qty)
        if 0 <= price <= 1 and qty > 0:
            clean.append((price, qty))
            visible += qty
    for price, qty in sorted(clean):
        take = min(remaining, qty)
        if take > 0:
            fills.append((price, take))
            notional += price * take
            remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        return None, fills, visible
    return notional / size, fills, visible


def taker_fee(fills: Iterable[tuple[float, float]]) -> float:
    """Order-level fee in dollars, rounded up to the next cent."""
    raw = sum(0.07 * qty * price * (1.0 - price) for price, qty in fills)
    return math.ceil(max(0.0, raw) * 100.0 - 1e-12) / 100.0


def qualifies(btc_favored_mid: float, alt_ask: float) -> bool:
    return (
        btc_favored_mid + 1e-12 >= BTC_THRESHOLD
        and btc_favored_mid - alt_ask + 1e-12 >= MIN_GAP
    )


def settlement_pnl(side: str, result: str, size: float, vwap: float, fee: float):
    win = side == result
    payout = size if win else 0.0
    cost = size * vwap + fee
    pnl = payout - cost
    return win, payout, cost, pnl


def _public_get(path: str, params=None) -> dict:
    response = requests.get(BASE_URL + path, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def _current_market(series: str, now: float) -> dict | None:
    data = _public_get(
        "/trade-api/v2/markets",
        {"series_ticker": series, "status": "open", "limit": 20},
    )
    choices = []
    for market in data.get("markets", []):
        try:
            opened = _parse_iso(market["open_time"])
            closes = _parse_iso(market["close_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if opened - 2 <= now < closes:
            choices.append((closes, market, opened))
    if not choices:
        return None
    _, market, opened = min(choices, key=lambda item: item[0])
    return {
        "ticker": market["ticker"],
        "open_ts": opened,
        "close_ts": _parse_iso(market["close_time"]),
    }


def discover_aligned_window(now: float | None = None) -> dict | None:
    """Find one live BTC window and same-timestamp alt contracts."""
    now = time.time() if now is None else now
    markets = {}
    for asset, series in SERIES.items():
        market = _current_market(series, now)
        if market is not None:
            markets[asset] = market
    btc = markets.get("BTC")
    if not btc:
        return None
    aligned = {"BTC": btc}
    for asset in ALTS:
        market = markets.get(asset)
        if (
            market
            and abs(market["open_ts"] - btc["open_ts"]) <= 1
            and abs(market["close_ts"] - btc["close_ts"]) <= 1
        ):
            aligned[asset] = market
    return aligned if len(aligned) > 1 else None


def _book_quotes(book, side: str | None = None):
    if not book or not book.snapshot_seen:
        return None
    yes_bid, yes_ask = book.yes_bid(), book.yes_ask()
    if yes_bid is None or yes_ask is None or not (0 <= yes_bid <= yes_ask <= 1):
        return None
    if side == "yes":
        levels = book.yes_asks_sorted()
    elif side == "no":
        levels = book.no_asks_sorted()
    else:
        levels = None
    if MAX_BOOK_AGE > 0 and book.age() > MAX_BOOK_AGE:
        return None
    return yes_bid, yes_ask, levels, book.age()


class ShadowTrader:
    def __init__(self):
        self.window = None
        self.sampled = set()
        self.fired = set()
        self.signals = {}
        self.settled = set()
        self._tickers = tuple()
        self._load_state()

    def _load_state(self):
        for row in _read_csv(SAMPLE_CSV):
            try:
                self.sampled.add((row["btc_ticker"], int(float(row["minute"])), row["asset"]))
            except (KeyError, TypeError, ValueError):
                continue
        for row in _read_csv(SIGNAL_CSV):
            signal_id = row.get("signal_id")
            if not signal_id:
                continue
            self.signals[signal_id] = row
            self.fired.add((row.get("btc_ticker"), row.get("asset")))
        for row in _read_csv(SETTLEMENT_CSV):
            if row.get("signal_id"):
                self.settled.add(row["signal_id"])
        log.info(
            "restored %d samples, %d signals, %d settlements",
            len(self.sampled), len(self.signals), len(self.settled),
        )

    def set_window(self, window: dict):
        tickers = tuple(sorted(m["ticker"] for m in window.values()))
        if tickers == self._tickers:
            self.window = window
            return
        self.window = window
        self._tickers = tickers
        kalshi_orderbook.set_tickers(list(tickers))
        btc = window["BTC"]
        log.info(
            "window %s -> %s aligned=%s",
            _iso(btc["open_ts"]), _iso(btc["close_ts"]),
            ",".join(sorted(window)),
        )

    def _sample_row(self, now, minute, asset, decision, values=None):
        values = values or {}
        btc = self.window["BTC"]
        alt = self.window[asset]
        target = btc["open_ts"] + 60 * minute
        row = {
            "sample_ts": round(now, 3),
            "sample_iso": _iso(now),
            "open_ts": round(btc["open_ts"], 3),
            "close_ts": round(btc["close_ts"], 3),
            "btc_ticker": btc["ticker"],
            "alt_ticker": alt["ticker"],
            "asset": asset,
            "minute": minute,
            "elapsed_sec": round(now - btc["open_ts"], 3),
            "sample_delay_sec": round(now - target, 3),
            "requested_size": PAPER_SIZE,
            "decision": decision,
            **values,
        }
        _append_csv(SAMPLE_CSV, SAMPLE_COLS, row)
        self.sampled.add((btc["ticker"], minute, asset))
        return row

    def _record_signal(self, sample_row, fills, vwap):
        btc_ticker = sample_row["btc_ticker"]
        asset = sample_row["asset"]
        signal_id = f"{btc_ticker}|{asset}"
        fee = taker_fee(fills)
        size = PAPER_SIZE
        row = {
            "signal_id": signal_id,
            "signal_ts": sample_row["sample_ts"],
            "signal_iso": sample_row["sample_iso"],
            "open_ts": sample_row["open_ts"],
            "close_ts": sample_row["close_ts"],
            "btc_ticker": btc_ticker,
            "alt_ticker": sample_row["alt_ticker"],
            "asset": asset,
            "minute": sample_row["minute"],
            "elapsed_sec": sample_row["elapsed_sec"],
            "sample_delay_sec": sample_row["sample_delay_sec"],
            "side": sample_row["side"],
            "btc_favored_mid": sample_row["btc_favored_mid"],
            "btc_yes_bid": sample_row["btc_yes_bid"],
            "btc_yes_ask": sample_row["btc_yes_ask"],
            "alt_top_ask": sample_row["alt_ask"],
            "gap": sample_row["gap"],
            "btc_book_age": sample_row["btc_book_age"],
            "alt_book_age": sample_row["alt_book_age"],
            "requested_size": size,
            "fill_size": size,
            "fill_vwap": round(vwap, 6),
            "fee_dollars": round(fee, 2),
            "total_cost_dollars": round(size * vwap + fee, 6),
        }
        _append_csv(SIGNAL_CSV, SIGNAL_COLS, row)
        self.signals[signal_id] = row
        self.fired.add((btc_ticker, asset))
        log.info(
            "PAPER BUY %-3s %-3s size=%.2f vwap=%.2fc BTCmid=%.2fc gap=%.2fc minute=%s fee=$%.2f",
            row["side"].upper(), asset, size, vwap * 100,
            float(row["btc_favored_mid"]) * 100, float(row["gap"]) * 100,
            row["minute"], fee,
        )

    def sample_due(self, now: float):
        if not self.window or "BTC" not in self.window:
            return
        btc_meta = self.window["BTC"]
        btc_ticker = btc_meta["ticker"]
        for minute in SAMPLE_MINUTES:
            target = btc_meta["open_ts"] + 60 * minute
            if now < target:
                continue
            due_assets = [
                asset for asset in ALTS
                if asset in self.window and (btc_ticker, minute, asset) not in self.sampled
            ]
            if not due_assets:
                continue
            if now > target + SAMPLE_GRACE_SEC:
                for asset in due_assets:
                    self._sample_row(now, minute, asset, "missed_target")
                continue

            btc_book = kalshi_orderbook.get_book(btc_ticker)
            btc_quote = _book_quotes(btc_book)
            if btc_quote is None:
                continue
            btc_yes_bid, btc_yes_ask, _, btc_age = btc_quote
            side, btc_mid = favored_side(btc_yes_bid, btc_yes_ask)

            for asset in due_assets:
                alt_meta = self.window[asset]
                alt_book = kalshi_orderbook.get_book(alt_meta["ticker"])
                alt_quote = _book_quotes(alt_book, side)
                if alt_quote is None:
                    continue
                _, _, levels, alt_age = alt_quote
                if not levels:
                    continue
                alt_ask = levels[0][0]
                vwap, fills, visible = book_vwap(levels, PAPER_SIZE)
                gap = btc_mid - alt_ask
                already_fired = (btc_ticker, asset) in self.fired
                if already_fired:
                    decision = "already_fired"
                elif btc_mid + 1e-12 < BTC_THRESHOLD:
                    decision = "btc_below_threshold"
                elif gap + 1e-12 < MIN_GAP:
                    decision = "gap_below_threshold"
                elif vwap is None:
                    decision = "insufficient_depth"
                else:
                    decision = "signal"
                values = {
                    "side": side,
                    "btc_favored_mid": round(btc_mid, 6),
                    "btc_yes_bid": round(btc_yes_bid, 6),
                    "btc_yes_ask": round(btc_yes_ask, 6),
                    "alt_ask": round(alt_ask, 6),
                    "gap": round(gap, 6),
                    "btc_book_age": _fmt(btc_age, 3),
                    "alt_book_age": _fmt(alt_age, 3),
                    "visible_size": round(visible, 4),
                    "paper_vwap": _fmt(vwap, 6),
                }
                sample_row = self._sample_row(now, minute, asset, decision, values)
                if decision == "signal":
                    self._record_signal(sample_row, fills, vwap)

    def settle_pending(self):
        for signal_id, row in list(self.signals.items()):
            if signal_id in self.settled:
                continue
            close_ts = _f(row.get("close_ts"), 0)
            if time.time() < close_ts + 10:
                continue
            ticker = row["alt_ticker"]
            try:
                market = _public_get(f"/trade-api/v2/markets/{ticker}").get("market", {})
            except Exception as exc:
                log.warning("settlement fetch %s: %s", ticker, exc)
                continue
            result = str(market.get("result") or "").lower()
            if result not in ("yes", "no"):
                continue
            side = row["side"].lower()
            size = float(row["fill_size"])
            vwap = float(row["fill_vwap"])
            fee = float(row["fee_dollars"])
            win, payout, cost, pnl = settlement_pnl(side, result, size, vwap, fee)
            now = time.time()
            out = {
                "signal_id": signal_id,
                "settled_ts": round(now, 3),
                "settled_iso": _iso(now),
                "alt_ticker": ticker,
                "asset": row["asset"],
                "side": side,
                "result": result,
                "win": int(win),
                "fill_vwap": round(vwap, 6),
                "fill_size": size,
                "fee_dollars": round(fee, 2),
                "payout_dollars": round(payout, 6),
                "pnl_dollars": round(pnl, 6),
                "pnl_c_per_contract": round(100 * pnl / size, 4),
                "roi": round(pnl / cost, 6) if cost else "",
            }
            _append_csv(SETTLEMENT_CSV, SETTLEMENT_COLS, out)
            self.settled.add(signal_id)
            log.info(
                "SETTLED %-3s %-3s result=%s pnl=$%+.2f (%.2fc/contract)",
                side.upper(), row["asset"], result.upper(), pnl, 100 * pnl / size,
            )


def load_credentials():
    env = dotenv_values(os.path.join(ROOT, ".env"))
    key_id = env.get("KALSHI_API_KEY_ID", "")
    pem = env.get("KALSHI_PRIVATE_KEY", "")
    if not key_id or not pem:
        raise RuntimeError("KALSHI_API_KEY_ID/KALSHI_PRIVATE_KEY missing from live/.env")
    return kalshi_auth.load_private_key(pem), key_id


def collect():
    private_key, key_id = load_credentials()
    trader = ShadowTrader()
    initial = discover_aligned_window()
    initial_tickers = [m["ticker"] for m in initial.values()] if initial else []
    kalshi_orderbook.start(private_key, key_id, initial_tickers)
    if initial:
        trader.set_window(initial)
    log.info(
        "SHADOW ONLY minutes=%s BTC>=%.2fc gap>=%.2fc size=%.2f; no order route loaded",
        SAMPLE_MINUTES, BTC_THRESHOLD * 100, MIN_GAP * 100, PAPER_SIZE,
    )

    next_refresh = 0.0
    next_settle = 0.0
    last_refresh_error = None
    while True:
        loop_start = time.time()
        # Sampling comes first so a slow REST refresh cannot move the observation.
        trader.sample_due(loop_start)
        if loop_start >= next_settle:
            trader.settle_pending()
            next_settle = loop_start + SETTLE_POLL_SEC
        if loop_start >= next_refresh:
            try:
                window = discover_aligned_window(loop_start)
                if window:
                    trader.set_window(window)
                last_refresh_error = None
            except Exception as exc:
                msg = str(exc)
                if msg != last_refresh_error:
                    log.warning("market discovery: %s", exc)
                last_refresh_error = msg
            next_refresh = loop_start + REFRESH_SEC
        time.sleep(max(0.02, TICK_SEC - (time.time() - loop_start)))


def _settle_for_report(signals: list[dict]):
    """Use the same durable settlement path before printing a report."""
    trader = ShadowTrader()
    trader.settle_pending()


def report():
    signals = _read_csv(SIGNAL_CSV)
    samples = _read_csv(SAMPLE_CSV)
    if signals:
        _settle_for_report(signals)
    settlements = _read_csv(SETTLEMENT_CSV)

    print("EARLY-MOVE SHADOW REPORT")
    print(
        f"Rule: minutes={SAMPLE_MINUTES}, BTC favored midpoint >= {BTC_THRESHOLD:.0%}, "
        f"alt ask gap >= {MIN_GAP:.0%}, size={PAPER_SIZE:g}"
    )
    print(f"Samples: {len(samples)}  Signals: {len(signals)}  Settled: {len(settlements)}")
    if samples:
        counts = Counter(row.get("decision", "unknown") for row in samples)
        print("Sample decisions: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not settlements:
        print("No settled paper signals yet.")
        return

    settled_by_id = {row["signal_id"]: row for row in settlements}
    trades = []
    for signal in signals:
        outcome = settled_by_id.get(signal.get("signal_id"))
        if outcome:
            trades.append({**signal, **outcome})

    def aggregate(label, rows):
        if not rows:
            return
        n = len(rows)
        total = sum(float(r["pnl_dollars"]) for r in rows)
        contracts = sum(float(r["fill_size"]) for r in rows)
        wins = sum(int(float(r["win"])) for r in rows)
        avg_gap = sum(float(r["gap"]) for r in rows) / n
        avg_price = sum(float(r["fill_vwap"]) for r in rows) / n
        print(
            f"  {label:>12}: n={n:>3} win={wins/n:>6.1%} pnl=${total:>+8.2f} "
            f"pnl/contract={100*total/contracts:>+7.2f}c "
            f"avg_vwap={100*avg_price:>5.2f}c avg_gap={100*avg_gap:>5.2f}c"
        )

    print("\nSettled performance (fees and displayed-depth VWAP included):")
    aggregate("ALL", trades)
    for asset in ALTS:
        aggregate(asset, [r for r in trades if r["asset"] == asset])
    print("By entry minute:")
    for minute in SAMPLE_MINUTES:
        aggregate(f"minute {minute}", [r for r in trades if int(float(r["minute"])) == minute])
    print("By BTC side:")
    for side in ("yes", "no"):
        aggregate(side.upper(), [r for r in trades if r["side"] == side])
    unsettled = len(signals) - len(trades)
    if unsettled:
        print(f"Unsettled signals: {unsettled}")


def selftest():
    assert favored_side(0.89, 0.91) == ("yes", 0.9)
    side, mid = favored_side(0.09, 0.11)
    assert side == "no" and abs(mid - 0.9) < 1e-12
    vwap, fills, visible = book_vwap([(0.70, 4), (0.72, 10)], 10)
    assert abs(vwap - 0.712) < 1e-12 and len(fills) == 2 and visible == 14
    assert book_vwap([(0.70, 4)], 10)[0] is None
    assert taker_fee([(0.90, 10)]) == 0.07
    assert qualifies(0.90, 0.85)
    assert not qualifies(0.8999, 0.84)
    assert not qualifies(0.90, 0.8501)
    win, payout, cost, pnl = settlement_pnl("yes", "yes", 10, 0.85, 0.09)
    assert win and payout == 10 and abs(cost - 8.59) < 1e-12 and abs(pnl - 1.41) < 1e-12
    print("early_move_shadow self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="settle pending signals and summarize")
    parser.add_argument("--selftest", action="store_true", help="run offline accounting/decision checks")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.report:
        report()
    else:
        try:
            collect()
        except KeyboardInterrupt:
            log.info("stopped")


if __name__ == "__main__":
    main()
