#!/usr/bin/env python3
"""Polymarket 15-minute BTC-led early-move shadow paper trader.

Frozen rule: sample elapsed minutes 2/3/4; require BTC favored-token midpoint
>= 90c and the same-side alt 10-share displayed ask VWAP >= 5c cheaper.  At
most one paper entry per alt/window.  A qualifying FOK is simulated only after
Polymarket's 250ms taker delay, with its initial worst fill price as the limit.

Public Gamma/CLOB GETs and the public market WebSocket only.  No wallet, keys,
order client, POST order endpoint, or cancellation endpoint is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polymarket_orderbook as polybook


ROOT = os.path.dirname(os.path.abspath(__file__))
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
ASSETS = ("BTC", "ETH", "SOL", "XRP")
ALTS = ("ETH", "SOL", "XRP")
SLUG_ASSET = {"BTC": "btc", "ETH": "eth", "SOL": "sol", "XRP": "xrp"}


def _minutes():
    values = sorted({
        int(value.strip())
        for value in os.environ.get("POLY_EARLY_MINUTES", "2,3,4").split(",")
        if value.strip()
    })
    if not values or any(value <= 0 or value >= 15 for value in values):
        raise ValueError("POLY_EARLY_MINUTES must contain elapsed minutes 1..14")
    return tuple(values)


SAMPLE_MINUTES = _minutes()
BTC_THRESHOLD = float(os.environ.get("POLY_EARLY_BTC_THRESHOLD", "0.90"))
MIN_GAP = float(os.environ.get("POLY_EARLY_MIN_GAP", "0.05"))
PAPER_SIZE = float(os.environ.get("POLY_EARLY_SIZE", "10"))
TAKER_DELAY_SEC = float(os.environ.get("POLY_EARLY_TAKER_DELAY", "0.25"))
SAMPLE_GRACE_SEC = float(os.environ.get("POLY_EARLY_GRACE", "5"))
REFRESH_SEC = float(os.environ.get("POLY_EARLY_REFRESH", "10"))
TICK_SEC = float(os.environ.get("POLY_EARLY_TICK", "0.05"))
SETTLE_POLL_SEC = float(os.environ.get("POLY_EARLY_SETTLE_POLL", "30"))

SAMPLE_CSV = os.environ.get("POLY_EARLY_SAMPLE_CSV", os.path.join(ROOT, "poly_early_move_samples.csv"))
SIGNAL_CSV = os.environ.get("POLY_EARLY_SIGNAL_CSV", os.path.join(ROOT, "poly_early_move_signals.csv"))
SETTLE_CSV = os.environ.get("POLY_EARLY_SETTLE_CSV", os.path.join(ROOT, "poly_early_move_settlements.csv"))

SAMPLE_COLS = [
    "sample_ts", "sample_iso", "start_ts", "close_ts", "btc_slug", "alt_slug",
    "asset", "minute", "elapsed_sec", "sample_delay_sec", "side",
    "btc_mid", "btc_up_bid", "btc_up_ask", "btc_down_bid", "btc_down_ask",
    "initial_top_ask", "initial_vwap", "initial_worst", "initial_visible",
    "initial_gap", "btc_book_age", "alt_book_age", "delay_target_sec",
    "delay_actual_sec", "limit_price", "delayed_top_ask", "delayed_vwap",
    "delayed_worst", "delayed_visible_at_limit", "delayed_book_age", "fee_rate",
    "fee_exponent", "fee_dollars", "decision",
]
SIGNAL_COLS = [
    "signal_id", "signal_ts", "signal_iso", "start_ts", "close_ts", "btc_slug",
    "alt_slug", "condition_id", "asset", "minute", "side", "btc_mid",
    "initial_vwap", "initial_gap", "limit_price", "fill_size", "fill_vwap",
    "fill_worst", "fee_rate", "fee_exponent", "fee_dollars", "total_cost_dollars",
    "delay_actual_sec",
]
SETTLE_COLS = [
    "signal_id", "settled_ts", "settled_iso", "condition_id", "asset", "side",
    "result", "win", "fill_size", "fill_vwap", "fee_dollars", "payout_dollars",
    "pnl_dollars", "pnl_c_per_share", "roi",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s poly-early %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("poly_early_move_shadow")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "kalshi-delta-hedging-shadow/1.0"})


def _float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _append(path, columns, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def valid_book(book):
    return (
        book is not None and book.snapshot_seen and book.best_bid() is not None
        and book.best_ask() is not None and 0 <= book.best_bid() <= book.best_ask() <= 1
    )


def favored_side(up_book, down_book):
    if not valid_book(up_book) or not valid_book(down_book):
        raise ValueError("both outcome books must be valid")
    up_mid, down_mid = up_book.midpoint(), down_book.midpoint()
    return ("up", up_mid) if up_mid >= down_mid else ("down", down_mid)


def book_vwap(asks, size, limit=None):
    """Return VWAP/fills/visible/worst, requiring all shares at or below limit."""
    levels = sorted(
        (float(price), float(qty)) for price, qty in asks
        if 0 < float(price) < 1 and float(qty) > 0 and (limit is None or float(price) <= limit + 1e-12)
    )
    visible = sum(qty for _, qty in levels)
    remaining, cost, fills, worst = size, 0.0, [], None
    for price, qty in levels:
        take = min(remaining, qty)
        if take > 0:
            fills.append((price, take))
            cost += price * take
            worst = price
            remaining -= take
        if remaining <= 1e-9:
            break
    return (
        None if remaining > 1e-9 else cost / size,
        fills,
        visible,
        worst if remaining <= 1e-9 else None,
    )


def taker_fee(fills, rate, exponent=1.0):
    if exponent != 1.0:
        raise ValueError("unsupported fee exponent")
    raw = sum(qty * rate * price * (1.0 - price) for price, qty in fills)
    return round(raw + 1e-12, 5)


def settlement_pnl(side, result, size, vwap, fee):
    win = side == result
    payout = size if win else 0.0
    cost = size * vwap + fee
    return win, payout, cost, payout - cost


def _json_get(base, path, params=None):
    response = SESSION.get(base + path, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def _one_market(asset, start_ts):
    slug = f"{SLUG_ASSET[asset]}-updown-15m-{start_ts}"
    events = _json_get(GAMMA, "/events", {"slug": slug})
    if not events or not events[0].get("markets"):
        return None
    market = events[0]["markets"][0]
    if market.get("closed") or not market.get("active") or not market.get("acceptingOrders"):
        return None
    try:
        outcomes = json.loads(market["outcomes"])
        tokens = json.loads(market["clobTokenIds"])
        token_map = {str(outcome).lower(): str(token) for outcome, token in zip(outcomes, tokens)}
        condition = str(market["conditionId"])
        info = _json_get(CLOB, f"/clob-markets/{condition}")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if "up" not in token_map or "down" not in token_map:
        return None
    fee = info.get("fd") or {}
    return {
        "asset": asset,
        "slug": slug,
        "condition_id": condition,
        "tokens": token_map,
        "start_ts": float(start_ts),
        "close_ts": float(start_ts + 900),
        "fee_rate": _float(fee.get("r"), -1.0),
        "fee_exponent": _float(fee.get("e"), -1.0),
        "taker_only": bool(fee.get("to")),
        "taker_delay_enabled": bool(info.get("itode")),
        "min_order_size": _float(info.get("mos"), 5.0),
    }


def discover_window(now=None):
    now = time.time() if now is None else now
    start = int(now) // 900 * 900
    window = {}
    for asset in ASSETS:
        market = _one_market(asset, start)
        if market:
            window[asset] = market
    return window if "BTC" in window and len(window) > 1 else None


class ShadowTrader:
    def __init__(self):
        self.window = None
        self.sampled = set()
        self.fired = set()
        self.signals = {}
        self.settled = set()
        self.token_tuple = tuple()
        self._restore()

    def _restore(self):
        for row in _read(SAMPLE_CSV):
            try:
                self.sampled.add((row["btc_slug"], int(float(row["minute"])), row["asset"]))
            except (KeyError, TypeError, ValueError):
                pass
        for row in _read(SIGNAL_CSV):
            signal_id = row.get("signal_id")
            if signal_id:
                self.signals[signal_id] = row
                self.fired.add((row.get("btc_slug"), row.get("asset")))
        for row in _read(SETTLE_CSV):
            if row.get("signal_id"):
                self.settled.add(row["signal_id"])
        log.info("restored %d samples, %d signals, %d settlements", len(self.sampled), len(self.signals), len(self.settled))

    def set_window(self, window):
        tokens = tuple(sorted(token for market in window.values() for token in market["tokens"].values()))
        self.window = window
        if tokens == self.token_tuple:
            return
        self.token_tuple = tokens
        polybook.set_tokens(tokens)
        btc = window["BTC"]
        log.info("window %s -> %s aligned=%s tokens=%d", _iso(btc["start_ts"]), _iso(btc["close_ts"]), ",".join(sorted(window)), len(tokens))

    def _base_row(self, now, minute, asset, decision, values=None):
        values = values or {}
        btc, alt = self.window["BTC"], self.window[asset]
        target = btc["start_ts"] + minute * 60
        row = {
            "sample_ts": round(now, 3), "sample_iso": _iso(now),
            "start_ts": btc["start_ts"], "close_ts": btc["close_ts"],
            "btc_slug": btc["slug"], "alt_slug": alt["slug"], "asset": asset,
            "minute": minute, "elapsed_sec": round(now - btc["start_ts"], 3),
            "sample_delay_sec": round(now - target, 3), "delay_target_sec": TAKER_DELAY_SEC,
            "fee_rate": alt["fee_rate"], "fee_exponent": alt["fee_exponent"],
            "decision": decision, **values,
        }
        _append(SAMPLE_CSV, SAMPLE_COLS, row)
        self.sampled.add((btc["slug"], minute, asset))
        return row

    def _record_signal(self, sample, fills):
        asset = sample["asset"]
        btc, alt = self.window["BTC"], self.window[asset]
        signal_id = f"{btc['slug']}|{asset}"
        size, vwap = PAPER_SIZE, float(sample["delayed_vwap"])
        fee = taker_fee(fills, alt["fee_rate"], alt["fee_exponent"])
        row = {
            "signal_id": signal_id, "signal_ts": sample["sample_ts"],
            "signal_iso": sample["sample_iso"], "start_ts": btc["start_ts"],
            "close_ts": btc["close_ts"], "btc_slug": btc["slug"],
            "alt_slug": alt["slug"], "condition_id": alt["condition_id"],
            "asset": asset, "minute": sample["minute"], "side": sample["side"],
            "btc_mid": sample["btc_mid"], "initial_vwap": sample["initial_vwap"],
            "initial_gap": sample["initial_gap"], "limit_price": sample["limit_price"],
            "fill_size": size, "fill_vwap": vwap, "fill_worst": sample["delayed_worst"],
            "fee_rate": alt["fee_rate"], "fee_exponent": alt["fee_exponent"],
            "fee_dollars": fee, "total_cost_dollars": round(size * vwap + fee, 6),
            "delay_actual_sec": sample["delay_actual_sec"],
        }
        _append(SIGNAL_CSV, SIGNAL_COLS, row)
        self.signals[signal_id] = row
        self.fired.add((btc["slug"], asset))
        log.info("PAPER FOK BUY %s %s size=%.0f vwap=%.2fc BTCmid=%.2fc gap=%.2fc delay=%.0fms fee=$%.4f", row["side"].upper(), asset, size, vwap * 100, float(row["btc_mid"]) * 100, float(row["initial_gap"]) * 100, float(row["delay_actual_sec"]) * 1000, fee)

    def sample_due(self, now):
        if not self.window or "BTC" not in self.window:
            return
        btc = self.window["BTC"]
        for minute in SAMPLE_MINUTES:
            target = btc["start_ts"] + minute * 60
            if now < target:
                continue
            due = [asset for asset in ALTS if asset in self.window and (btc["slug"], minute, asset) not in self.sampled]
            if not due:
                continue
            if now > target + SAMPLE_GRACE_SEC:
                for asset in due:
                    self._base_row(now, minute, asset, "missed_target_or_book")
                continue

            up_book = polybook.get_book(btc["tokens"]["up"])
            down_book = polybook.get_book(btc["tokens"]["down"])
            try:
                side, btc_mid = favored_side(up_book, down_book)
            except ValueError:
                continue
            detected_at = time.time()
            common_values = {
                "side": side, "btc_mid": round(btc_mid, 6),
                "btc_up_bid": up_book.best_bid(), "btc_up_ask": up_book.best_ask(),
                "btc_down_bid": down_book.best_bid(), "btc_down_ask": down_book.best_ask(),
                "btc_book_age": round(max(up_book.age(), down_book.age()), 4),
            }
            candidates = {}
            for asset in due:
                alt = self.window[asset]
                book = polybook.get_book(alt["tokens"][side])
                if not valid_book(book):
                    continue
                vwap, fills, visible, worst = book_vwap(book.asks, PAPER_SIZE)
                gap = None if vwap is None else btc_mid - vwap
                values = {
                    **common_values, "initial_top_ask": book.best_ask(),
                    "initial_vwap": "" if vwap is None else round(vwap, 6),
                    "initial_worst": "" if worst is None else worst,
                    "initial_visible": round(visible, 4),
                    "initial_gap": "" if gap is None else round(gap, 6),
                    "alt_book_age": round(book.age(), 4),
                }
                if (btc["slug"], asset) in self.fired:
                    self._base_row(detected_at, minute, asset, "already_fired", values)
                elif btc_mid + 1e-12 < BTC_THRESHOLD:
                    self._base_row(detected_at, minute, asset, "btc_below_threshold", values)
                elif vwap is None:
                    self._base_row(detected_at, minute, asset, "insufficient_initial_depth", values)
                elif gap + 1e-12 < MIN_GAP:
                    self._base_row(detected_at, minute, asset, "gap_below_threshold", values)
                elif alt["fee_rate"] < 0 or alt["fee_exponent"] != 1 or not alt["taker_only"]:
                    self._base_row(detected_at, minute, asset, "unknown_fee_curve", values)
                elif PAPER_SIZE + 1e-12 < alt["min_order_size"]:
                    self._base_row(detected_at, minute, asset, "below_min_order", values)
                else:
                    candidates[asset] = (values, worst)

            if not candidates:
                continue
            delay_deadline = detected_at + TAKER_DELAY_SEC
            time.sleep(max(0.0, delay_deadline - time.time()))
            delayed_at = time.time()
            for asset, (values, limit_price) in candidates.items():
                alt = self.window[asset]
                book = polybook.get_book(alt["tokens"][side])
                delayed_vwap = delayed_worst = None
                delayed_visible = 0.0
                delayed_fills = []
                if valid_book(book):
                    delayed_vwap, delayed_fills, delayed_visible, delayed_worst = book_vwap(
                        book.asks, PAPER_SIZE, limit=limit_price
                    )
                values.update({
                    "delay_actual_sec": round(delayed_at - detected_at, 6),
                    "limit_price": limit_price,
                    "delayed_top_ask": book.best_ask() if valid_book(book) else "",
                    "delayed_vwap": "" if delayed_vwap is None else round(delayed_vwap, 6),
                    "delayed_worst": "" if delayed_worst is None else delayed_worst,
                    "delayed_visible_at_limit": round(delayed_visible, 4),
                    "delayed_book_age": round(book.age(), 4) if valid_book(book) else "",
                    "fee_dollars": "" if delayed_vwap is None else taker_fee(delayed_fills, alt["fee_rate"], alt["fee_exponent"]),
                })
                decision = "signal" if delayed_vwap is not None else "delayed_fok_no_fill"
                sample = self._base_row(delayed_at, minute, asset, decision, values)
                if decision == "signal":
                    self._record_signal(sample, delayed_fills)

    def settle_pending(self):
        for signal_id, row in list(self.signals.items()):
            if signal_id in self.settled or time.time() < float(row["close_ts"]) + 20:
                continue
            try:
                market = _json_get(CLOB, f"/markets/{row['condition_id']}")
                winners = [str(token.get("outcome") or "").lower() for token in market.get("tokens", []) if token.get("winner")]
            except Exception as exc:
                log.warning("settlement fetch %s: %s", row["condition_id"], exc)
                continue
            if len(winners) != 1 or winners[0] not in ("up", "down"):
                continue
            result = winners[0]
            size, vwap, fee = float(row["fill_size"]), float(row["fill_vwap"]), float(row["fee_dollars"])
            win, payout, cost, pnl = settlement_pnl(row["side"], result, size, vwap, fee)
            stamp = time.time()
            out = {
                "signal_id": signal_id, "settled_ts": round(stamp, 3), "settled_iso": _iso(stamp),
                "condition_id": row["condition_id"], "asset": row["asset"], "side": row["side"],
                "result": result, "win": int(win), "fill_size": size, "fill_vwap": vwap,
                "fee_dollars": fee, "payout_dollars": payout, "pnl_dollars": round(pnl, 6),
                "pnl_c_per_share": round(100 * pnl / size, 4), "roi": round(pnl / cost, 6) if cost else "",
            }
            _append(SETTLE_CSV, SETTLE_COLS, out)
            self.settled.add(signal_id)
            log.info("SETTLED %s %s result=%s pnl=$%+.2f", row["side"].upper(), row["asset"], result.upper(), pnl)


def collect():
    trader = ShadowTrader()
    try:
        window = discover_window()
    except Exception as exc:
        log.warning("initial discovery: %s", exc)
        window = None
    if window:
        trader.set_window(window)
    polybook.start(trader.token_tuple)
    log.info("SHADOW ONLY minutes=%s BTC>=%.0fc VWAP gap>=%.0fc size=%.0f delay=%.0fms; no order client", SAMPLE_MINUTES, BTC_THRESHOLD * 100, MIN_GAP * 100, PAPER_SIZE, TAKER_DELAY_SEC * 1000)
    next_refresh = next_settle = 0.0
    last_error = None
    while True:
        started = time.time()
        trader.sample_due(started)
        if started >= next_settle:
            trader.settle_pending()
            next_settle = started + SETTLE_POLL_SEC
        if started >= next_refresh:
            try:
                window = discover_window(started)
                if window:
                    trader.set_window(window)
                last_error = None
            except Exception as exc:
                if str(exc) != last_error:
                    log.warning("discovery: %s", exc)
                last_error = str(exc)
            next_refresh = started + REFRESH_SEC
        time.sleep(max(0.01, TICK_SEC - (time.time() - started)))


def report():
    trader = ShadowTrader()
    trader.settle_pending()
    samples, signals, settlements = _read(SAMPLE_CSV), _read(SIGNAL_CSV), _read(SETTLE_CSV)
    print("POLYMARKET 15M EARLY-MOVE SHADOW")
    print(f"Rule: minutes={SAMPLE_MINUTES}, BTC>={BTC_THRESHOLD:.0%}, 10-share VWAP gap>={MIN_GAP:.0%}, delay={TAKER_DELAY_SEC*1000:.0f}ms")
    print(f"Samples: {len(samples)}  Signals: {len(signals)}  Settled: {len(settlements)}")
    if samples:
        decisions = Counter(row.get("decision", "unknown") for row in samples)
        print("Decisions: " + ", ".join(f"{key}={value}" for key, value in sorted(decisions.items())))
    if not settlements:
        print("No settled paper signals yet.")
        return
    settled = {row["signal_id"]: row for row in settlements}
    trades = [{**signal, **settled[signal["signal_id"]]} for signal in signals if signal["signal_id"] in settled]

    def aggregate(label, rows):
        if not rows:
            return
        pnl = sum(float(row["pnl_dollars"]) for row in rows)
        shares = sum(float(row["fill_size"]) for row in rows)
        wins = sum(int(float(row["win"])) for row in rows)
        print(f"  {label:>10}: n={len(rows):3d} win={wins/len(rows):6.1%} pnl=${pnl:+8.2f} pnl/share={100*pnl/shares:+6.2f}c")

    aggregate("ALL", trades)
    for asset in ALTS:
        aggregate(asset, [row for row in trades if row["asset"] == asset])
    for minute in SAMPLE_MINUTES:
        aggregate(f"minute {minute}", [row for row in trades if int(float(row["minute"])) == minute])


def selftest():
    view = polybook.BookView("x", ((0.88, 10),), ((0.90, 10),), time.time(), time.time(), True)
    down = polybook.BookView("y", ((0.10, 10),), ((0.12, 10),), time.time(), time.time(), True)
    assert favored_side(view, down) == ("up", 0.89)
    vwap, fills, visible, worst = book_vwap(((0.80, 4), (0.82, 10)), 10)
    assert abs(vwap - 0.812) < 1e-12 and visible == 14 and worst == 0.82
    assert book_vwap(((0.80, 4), (0.82, 10)), 10, limit=0.80)[0] is None
    assert taker_fee([(0.80, 10)], 0.07) == 0.112
    win, payout, cost, pnl = settlement_pnl("up", "up", 10, 0.80, 0.112)
    assert win and payout == 10 and abs(cost - 8.112) < 1e-12 and abs(pnl - 1.888) < 1e-12
    print("poly_early_move_shadow self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--selftest", action="store_true")
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
