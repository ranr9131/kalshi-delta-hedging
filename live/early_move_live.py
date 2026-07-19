#!/usr/bin/env python3
"""Live Kalshi executor for the frozen 15-minute early-move rule.

This deliberately reuses the shadow trader's discovery, exact-minute sampling,
order-book, qualification, and settlement functions.  The only changed piece
is execution: a qualifying 10-contract signal is submitted as a price-bounded
V2 fill-or-kill order.

Safety properties:
* a durable intent is fsynced before every POST;
* one attempted order per alt/window, including after ambiguous timeouts;
* deterministic client_order_id for exchange-side idempotency;
* FOK only -- no resting or partially intended exposure;
* per-order, per-window, daily-notional, daily-loss, and balance guards;
* STOP_EARLY_MOVE_LIVE kill file prevents new orders while settlement continues.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import requests


ROOT = os.path.dirname(os.path.abspath(__file__))

# The shared shadow module reads its configuration at import time.  Give this
# process an isolated state namespace so it cannot replay or corrupt paper data.
os.environ.setdefault("EARLY_SAMPLE_CSV", os.path.join(ROOT, "early_move_live_samples.csv"))
os.environ.setdefault("EARLY_SIGNAL_CSV", os.path.join(ROOT, "early_move_live_fills.csv"))
os.environ.setdefault(
    "EARLY_SETTLEMENT_CSV", os.path.join(ROOT, "early_move_live_settlements.csv")
)
os.environ.setdefault("EARLY_PAPER_SIZE", os.environ.get("EARLY_LIVE_SIZE", "10"))

sys.path.insert(0, ROOT)
import early_move_shadow as shadow  # noqa: E402
import kalshi_auth  # noqa: E402
import kalshi_orderbook  # noqa: E402


LIVE_SIZE = float(os.environ.get("EARLY_LIVE_SIZE", "10"))
MAX_ORDER_COST = float(os.environ.get("EARLY_LIVE_MAX_ORDER_COST", "10"))
MAX_WINDOW_COST = float(os.environ.get("EARLY_LIVE_MAX_WINDOW_COST", "30"))
MAX_DAILY_NOTIONAL = float(os.environ.get("EARLY_LIVE_MAX_DAILY_NOTIONAL", "100"))
DAILY_LOSS_LIMIT = float(os.environ.get("EARLY_LIVE_DAILY_LOSS", "25"))
BALANCE_FLOOR = float(os.environ.get("EARLY_LIVE_BALANCE_FLOOR", "100"))
LIVE_ENABLED = os.environ.get("EARLY_LIVE_ENABLED", "0") == "1"

ORDER_EVENT_CSV = os.environ.get(
    "EARLY_LIVE_ORDER_EVENT_CSV", os.path.join(ROOT, "early_move_live_orders.csv")
)
FILL_CSV = shadow.SIGNAL_CSV
SETTLEMENT_CSV = shadow.SETTLEMENT_CSV
KILL_FILE = os.environ.get(
    "EARLY_LIVE_KILL_FILE", os.path.join(ROOT, "STOP_EARLY_MOVE_LIVE")
)
LOCK_FILE = os.environ.get(
    "EARLY_LIVE_LOCK_FILE", os.path.join(ROOT, ".early_move_live.lock")
)

V2_ORDERS_PATH = "/trade-api/v2/portfolio/events/orders"
ORDERS_PATH = "/trade-api/v2/portfolio/orders"
FILLS_PATH = "/trade-api/v2/portfolio/fills"

ORDER_EVENT_COLS = [
    "event_ts", "event_iso", "signal_id", "status", "client_order_id",
    "order_id", "signal_ts", "signal_iso", "open_ts", "close_ts",
    "btc_ticker", "alt_ticker", "asset", "minute", "elapsed_sec",
    "sample_delay_sec", "side", "btc_favored_mid", "btc_yes_bid",
    "btc_yes_ask", "alt_top_ask", "gap", "btc_book_age", "alt_book_age",
    "requested_size", "limit_price", "yes_book_price", "estimated_cost",
    "http_status", "message",
]
FILL_COLS = [
    "signal_id", "signal_ts", "signal_iso", "open_ts", "close_ts",
    "btc_ticker", "alt_ticker", "asset", "minute", "elapsed_sec",
    "sample_delay_sec", "side", "btc_favored_mid", "btc_yes_bid",
    "btc_yes_ask", "alt_top_ask", "gap", "btc_book_age", "alt_book_age",
    "requested_size", "limit_price", "client_order_id", "order_id",
    "fill_size", "fill_vwap", "fee_dollars", "total_cost_dollars",
    "submit_latency_sec",
]
SETTLEMENT_COLS = shadow.SETTLEMENT_COLS + ["client_order_id", "order_id"]

log = logging.getLogger("early_move_live")
_session = requests.Session()


def _num(row: dict, *names: str, default=0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return default


def _utc_day(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(time.time() if ts is None else ts, tz=timezone.utc)
    return dt.date().isoformat()


def deterministic_coid(signal_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kalshi-early-move-live:{signal_id}"))


def yes_book_price(side: str, outcome_limit: float) -> float:
    """Translate an outcome-side buy limit to Kalshi's single YES book."""
    if side not in ("yes", "no") or not 0 < outcome_limit < 1:
        raise ValueError("invalid outcome-side limit")
    return outcome_limit if side == "yes" else 1.0 - outcome_limit


def actual_from_order(order: dict, side: str) -> dict:
    """Extract executed size/cost from the fixed-point V2 order record.

    The fills index can trail matching.  Current V2 order records expose total
    taker/maker fill cost and fees, which are sufficient to recover exact VWAP
    without guessing from the submitted limit.
    """
    count = _num(order, "fill_count_fp", "fill_count")
    cost = (
        _num(order, "taker_fill_cost_dollars")
        + _num(order, "maker_fill_cost_dollars")
    )
    fees = _num(order, "taker_fees_dollars") + _num(order, "maker_fees_dollars")
    if count <= 0:
        return {"fill_size": 0.0, "fill_vwap": None, "fee_dollars": 0.0}
    if cost > 0:
        vwap = cost / count
    else:
        yes_price = _num(order, "yes_price_dollars", "average_fill_price", default=-1)
        if side == "yes":
            vwap = yes_price
        else:
            vwap = _num(order, "no_price_dollars", default=-1)
            if vwap < 0 and yes_price >= 0:
                vwap = 1.0 - yes_price
    if not 0 <= vwap <= 1:
        return {"fill_size": 0.0, "fill_vwap": None, "fee_dollars": 0.0}
    return {"fill_size": count, "fill_vwap": vwap, "fee_dollars": fees}


def risk_decision(
    *, estimated_cost: float, window_cost: float, daily_notional: float,
    daily_pnl: float, balance: float | None, killed: bool,
) -> str | None:
    if killed:
        return "kill_file_present"
    if not LIVE_ENABLED:
        return "live_disabled"
    if balance is None:
        return "balance_unreadable"
    if balance < BALANCE_FLOOR:
        return "balance_below_floor"
    if estimated_cost > MAX_ORDER_COST + 1e-9:
        return "max_order_cost"
    if window_cost + estimated_cost > MAX_WINDOW_COST + 1e-9:
        return "max_window_cost"
    if daily_notional + estimated_cost > MAX_DAILY_NOTIONAL + 1e-9:
        return "max_daily_notional"
    if daily_pnl <= -DAILY_LOSS_LIMIT + 1e-9:
        return "daily_loss_limit"
    return None


class KalshiLiveAPI:
    def __init__(self, private_key, key_id: str):
        self.private_key = private_key
        self.key_id = key_id

    def _headers(self, method: str, path: str) -> dict:
        return kalshi_auth.make_auth_headers(self.private_key, self.key_id, method, path)

    def balance(self) -> float | None:
        path = "/trade-api/v2/portfolio/balance"
        try:
            response = _session.get(
                shadow.BASE_URL + path, headers=self._headers("GET", path), timeout=10
            )
            response.raise_for_status()
            return _num(response.json(), "balance") / 100.0
        except Exception as exc:
            log.error("balance check failed: %s", exc)
            return None

    def find_order_by_coid(self, ticker: str, coid: str) -> dict | None:
        try:
            response = _session.get(
                shadow.BASE_URL + ORDERS_PATH,
                params={"ticker": ticker, "limit": 100},
                headers=self._headers("GET", ORDERS_PATH),
                timeout=10,
            )
            response.raise_for_status()
            for order in response.json().get("orders") or []:
                if order.get("client_order_id") == coid:
                    return order
        except Exception as exc:
            log.error("client-order reconciliation failed: %s", exc)
        return None

    def get_order(self, order_id: str) -> dict | None:
        path = f"{ORDERS_PATH}/{order_id}"
        try:
            response = _session.get(
                shadow.BASE_URL + path, headers=self._headers("GET", path), timeout=10
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("order", payload)
        except Exception as exc:
            log.error("order reconciliation failed: %s", exc)
            return None

    def place_fok(
        self, *, ticker: str, side: str, count: float,
        outcome_limit: float, coid: str,
    ) -> tuple[dict, int]:
        book_price = yes_book_price(side, outcome_limit)
        body = {
            "ticker": ticker,
            "client_order_id": coid,
            "side": "bid" if side == "yes" else "ask",
            "count": f"{count:.2f}",
            "price": f"{book_price:.4f}",
            "time_in_force": "fill_or_kill",
            "self_trade_prevention_type": "taker_at_cross",
            "cancel_order_on_pause": True,
        }
        try:
            response = _session.post(
                shadow.BASE_URL + V2_ORDERS_PATH,
                json=body,
                headers=self._headers("POST", V2_ORDERS_PATH),
                timeout=10,
            )
        except (requests.Timeout, requests.ConnectionError):
            # Never blindly retry an order write.  The deterministic coid lets
            # us adopt an accepted order without creating duplicate exposure.
            time.sleep(0.5)
            adopted = self.find_order_by_coid(ticker, coid)
            if adopted is not None:
                return adopted, 0
            raise
        if not response.ok:
            message = response.text[:500].replace("\n", " ")
            raise requests.HTTPError(
                f"{response.status_code} {response.reason}: {message}", response=response
            )
        payload = response.json()
        return payload.get("order", payload), response.status_code

    def actual_fills(self, order_id: str, side: str) -> dict:
        response = _session.get(
            shadow.BASE_URL + FILLS_PATH,
            params={"order_id": order_id, "limit": 100},
            headers=self._headers("GET", FILLS_PATH),
            timeout=10,
        )
        response.raise_for_status()
        count = notional = fees = 0.0
        records = []
        for fill in response.json().get("fills") or []:
            if fill.get("order_id") not in (None, "", order_id):
                continue
            qty = _num(fill, "count_fp", "count")
            if qty <= 0:
                continue
            yes_price = _num(fill, "yes_price_dollars", default=-1)
            if side == "yes":
                price = yes_price
            else:
                price = _num(fill, "no_price_dollars", default=-1)
                if price < 0 and yes_price >= 0:
                    price = 1.0 - yes_price
            if not 0 <= price <= 1:
                continue
            count += qty
            notional += qty * price
            fees += _num(fill, "fee_cost", "fee_cost_dollars")
            records.append(fill)
        return {
            "fill_size": count,
            "fill_vwap": notional / count if count else None,
            "fee_dollars": fees,
            "records": records,
        }


class LiveTrader(shadow.ShadowTrader):
    def __init__(self, api: KalshiLiveAPI):
        self.api = api
        super().__init__()
        self.attempted = set()
        latest_events = {}
        for row in shadow._read_csv(ORDER_EVENT_CSV):
            signal_id = row.get("signal_id")
            if signal_id:
                self.attempted.add(signal_id)
                self.fired.add((row.get("btc_ticker"), row.get("asset")))
                latest_events[signal_id] = row
        log.info("restored %d durable live intents", len(self.attempted))
        self._recover_unfinished(latest_events)

    def _event(self, status: str, sample: dict, **extra) -> None:
        now = time.time()
        signal_id = f"{sample['btc_ticker']}|{sample['asset']}"
        row = {
            "event_ts": round(now, 3),
            "event_iso": shadow._iso(now),
            "signal_id": signal_id,
            "status": status,
            "signal_ts": sample.get("sample_ts", sample.get("signal_ts")),
            "signal_iso": sample.get("sample_iso", sample.get("signal_iso")),
            "open_ts": sample.get("open_ts"),
            "close_ts": sample.get("close_ts"),
            "btc_ticker": sample.get("btc_ticker"),
            "alt_ticker": sample.get("alt_ticker"),
            "asset": sample.get("asset"),
            "minute": sample.get("minute"),
            "elapsed_sec": sample.get("elapsed_sec"),
            "sample_delay_sec": sample.get("sample_delay_sec"),
            "side": sample.get("side"),
            "btc_favored_mid": sample.get("btc_favored_mid"),
            "btc_yes_bid": sample.get("btc_yes_bid"),
            "btc_yes_ask": sample.get("btc_yes_ask"),
            "alt_top_ask": sample.get("alt_ask", sample.get("alt_top_ask")),
            "gap": sample.get("gap"),
            "btc_book_age": sample.get("btc_book_age"),
            "alt_book_age": sample.get("alt_book_age"),
            "requested_size": LIVE_SIZE,
            **extra,
        }
        shadow._append_csv(ORDER_EVENT_CSV, ORDER_EVENT_COLS, row)

    def _store_fill(
        self, sample_row: dict, *, coid: str, order_id: str,
        outcome_limit: float, actual: dict, submit_latency: float,
    ) -> dict:
        signal_id = f"{sample_row['btc_ticker']}|{sample_row['asset']}"
        fill_size = float(actual["fill_size"])
        vwap = float(actual["fill_vwap"])
        fee = float(actual["fee_dollars"])
        total_cost = fill_size * vwap + fee
        row = {
            "signal_id": signal_id,
            "signal_ts": sample_row.get("sample_ts", sample_row.get("signal_ts")),
            "signal_iso": sample_row.get("sample_iso", sample_row.get("signal_iso")),
            "open_ts": sample_row.get("open_ts"),
            "close_ts": sample_row.get("close_ts"),
            "btc_ticker": sample_row["btc_ticker"],
            "alt_ticker": sample_row["alt_ticker"],
            "asset": sample_row["asset"],
            "minute": sample_row.get("minute"),
            "elapsed_sec": sample_row.get("elapsed_sec"),
            "sample_delay_sec": sample_row.get("sample_delay_sec"),
            "side": sample_row["side"],
            "btc_favored_mid": sample_row.get("btc_favored_mid"),
            "btc_yes_bid": sample_row.get("btc_yes_bid"),
            "btc_yes_ask": sample_row.get("btc_yes_ask"),
            "alt_top_ask": sample_row.get("alt_ask", sample_row.get("alt_top_ask")),
            "gap": sample_row.get("gap"),
            "btc_book_age": sample_row.get("btc_book_age"),
            "alt_book_age": sample_row.get("alt_book_age"),
            "requested_size": LIVE_SIZE,
            "limit_price": round(outcome_limit, 4),
            "client_order_id": coid,
            "order_id": order_id,
            "fill_size": round(fill_size, 4),
            "fill_vwap": round(vwap, 6),
            "fee_dollars": round(fee, 6),
            "total_cost_dollars": round(total_cost, 6),
            "submit_latency_sec": round(submit_latency, 4),
        }
        shadow._append_csv(FILL_CSV, FILL_COLS, row)
        self.signals[signal_id] = row
        return row

    def _recover_unfinished(self, latest_events: dict) -> None:
        """Adopt a FOK accepted just before a crash and restore its accounting."""
        for signal_id, event in latest_events.items():
            if signal_id in self.signals:
                continue
            if event.get("status") not in ("intent", "submit_unknown"):
                continue
            coid = event.get("client_order_id") or deterministic_coid(signal_id)
            order = self.api.find_order_by_coid(event.get("alt_ticker", ""), coid)
            if not order:
                continue
            order_id = str(order.get("order_id") or "")
            actual = actual_from_order(order, event.get("side", ""))
            if order_id and actual["fill_size"] <= 0:
                try:
                    actual = self.api.actual_fills(order_id, event.get("side", ""))
                except Exception:
                    pass
            common = {
                "client_order_id": coid,
                "order_id": order_id,
                "limit_price": event.get("limit_price"),
                "yes_book_price": event.get("yes_book_price"),
                "estimated_cost": event.get("estimated_cost"),
            }
            if actual["fill_size"] > 0:
                self._store_fill(
                    event, coid=coid, order_id=order_id,
                    outcome_limit=float(event["limit_price"]), actual=actual,
                    submit_latency=0.0,
                )
                self._event("recovered_fill", event, message="startup adoption", **common)
                log.warning("recovered live fill after restart: %s", signal_id)
            else:
                self._event("recovered_no_fill", event, message="FOK unfilled", **common)

    def _risk_totals(self, btc_ticker: str) -> tuple[float, float, float]:
        day = _utc_day()
        daily_notional = 0.0
        window_cost = 0.0
        for row in self.signals.values():
            if str(row.get("signal_iso", ""))[:10] == day:
                daily_notional += _num(row, "total_cost_dollars")
            if row.get("btc_ticker") == btc_ticker:
                window_cost += _num(row, "total_cost_dollars")
        daily_pnl = sum(
            _num(row, "pnl_dollars")
            for row in shadow._read_csv(SETTLEMENT_CSV)
            if str(row.get("settled_iso", ""))[:10] == day
        )
        return window_cost, daily_notional, daily_pnl

    def _read_fills_with_retry(
        self, order_id: str, side: str, initial_order: dict | None = None,
    ) -> dict:
        last = {"fill_size": 0.0, "fill_vwap": None, "fee_dollars": 0.0}
        for _ in range(8):
            try:
                last = self.api.actual_fills(order_id, side)
                if last["fill_size"] > 0:
                    return last
            except Exception as exc:
                log.warning("fill reconciliation %s: %s", order_id, exc)
            time.sleep(0.25)
        order = self.api.get_order(order_id) or initial_order or {}
        recovered = actual_from_order(order, side)
        if recovered["fill_size"] > 0:
            log.warning("recovered fill accounting from V2 order record %s", order_id)
            return recovered
        return last

    def _record_signal(self, sample_row, fills, _paper_vwap):
        signal_id = f"{sample_row['btc_ticker']}|{sample_row['asset']}"
        if signal_id in self.attempted:
            return

        # Mark in memory immediately.  The durable intent below is written
        # before the exchange call, preventing duplicate exposure after crash.
        self.attempted.add(signal_id)
        self.fired.add((sample_row["btc_ticker"], sample_row["asset"]))
        outcome_limit = max(float(price) for price, qty in fills if float(qty) > 0)
        expected_fee = shadow.taker_fee(fills)
        estimated_cost = LIVE_SIZE * outcome_limit + expected_fee
        book_price = yes_book_price(sample_row["side"], outcome_limit)
        coid = deterministic_coid(signal_id)
        common = {
            "client_order_id": coid,
            "limit_price": round(outcome_limit, 4),
            "yes_book_price": round(book_price, 4),
            "estimated_cost": round(estimated_cost, 6),
        }

        window_cost, daily_notional, daily_pnl = self._risk_totals(
            sample_row["btc_ticker"]
        )
        balance = self.api.balance()
        reason = risk_decision(
            estimated_cost=estimated_cost,
            window_cost=window_cost,
            daily_notional=daily_notional,
            daily_pnl=daily_pnl,
            balance=balance,
            killed=os.path.exists(KILL_FILE),
        )
        if reason:
            self._event("risk_rejected", sample_row, message=reason, **common)
            log.error("SKIP LIVE %s %s: %s", sample_row["asset"], signal_id, reason)
            return

        self._event("intent", sample_row, message="fsynced_before_post", **common)
        started = time.time()
        try:
            order, http_status = self.api.place_fok(
                ticker=sample_row["alt_ticker"],
                side=sample_row["side"],
                count=LIVE_SIZE,
                outcome_limit=outcome_limit,
                coid=coid,
            )
            order_id = str(order.get("order_id") or "")
            if not order_id:
                raise RuntimeError(f"order response missing order_id: {str(order)[:300]}")
            actual = self._read_fills_with_retry(
                order_id, sample_row["side"], initial_order=order
            )
            fill_size = float(actual.get("fill_size") or 0)
            if fill_size <= 0:
                self._event(
                    "no_fill", sample_row, order_id=order_id,
                    http_status=http_status, message="FOK did not fill", **common,
                )
                log.info(
                    "LIVE FOK NO-FILL %-3s %-3s limit=%.2fc",
                    sample_row["side"].upper(), sample_row["asset"], outcome_limit * 100,
                )
                return

            row = self._store_fill(
                sample_row, coid=coid, order_id=order_id,
                outcome_limit=outcome_limit, actual=actual,
                submit_latency=time.time() - started,
            )
            vwap = float(row["fill_vwap"])
            fee = float(row["fee_dollars"])
            total_cost = float(row["total_cost_dollars"])
            status = "filled" if abs(fill_size - LIVE_SIZE) <= 1e-6 else "partial_violation"
            self._event(
                status, sample_row, order_id=order_id, http_status=http_status,
                message=f"fill={fill_size:g} vwap={vwap:.6f} fee={fee:.6f}", **common,
            )
            log.warning(
                "LIVE FILLED %-3s %-3s size=%.2f vwap=%.2fc fee=$%.4f cost=$%.2f",
                sample_row["side"].upper(), sample_row["asset"], fill_size,
                vwap * 100, fee, total_cost,
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", "") if response is not None else ""
            self._event(
                "submit_unknown", sample_row, http_status=status,
                message=str(exc)[:500], **common,
            )
            log.exception("LIVE SUBMISSION UNKNOWN %s; will not retry", signal_id)

    def settle_pending(self):
        for signal_id, row in list(self.signals.items()):
            if signal_id in self.settled:
                continue
            close_ts = _num(row, "close_ts")
            if time.time() < close_ts + 10:
                continue
            ticker = row["alt_ticker"]
            try:
                market = shadow._public_get(f"/trade-api/v2/markets/{ticker}").get(
                    "market", {}
                )
            except Exception as exc:
                log.warning("settlement fetch %s: %s", ticker, exc)
                continue
            result = str(market.get("result") or "").lower()
            if result not in ("yes", "no"):
                continue
            side = row["side"].lower()
            size = _num(row, "fill_size")
            vwap = _num(row, "fill_vwap")
            fee = _num(row, "fee_dollars")
            win, payout, cost, pnl = shadow.settlement_pnl(side, result, size, vwap, fee)
            now = time.time()
            out = {
                "signal_id": signal_id,
                "settled_ts": round(now, 3),
                "settled_iso": shadow._iso(now),
                "alt_ticker": ticker,
                "asset": row["asset"],
                "side": side,
                "result": result,
                "win": int(win),
                "fill_vwap": round(vwap, 6),
                "fill_size": size,
                "fee_dollars": round(fee, 6),
                "payout_dollars": round(payout, 6),
                "pnl_dollars": round(pnl, 6),
                "pnl_c_per_contract": round(100 * pnl / size, 4),
                "roi": round(pnl / cost, 6) if cost else "",
                "client_order_id": row.get("client_order_id", ""),
                "order_id": row.get("order_id", ""),
            }
            shadow._append_csv(SETTLEMENT_CSV, SETTLEMENT_COLS, out)
            self.settled.add(signal_id)
            log.warning(
                "LIVE SETTLED %s %s result=%s pnl=$%+.2f",
                side.upper(), row["asset"], result.upper(), pnl,
            )


def load_api() -> KalshiLiveAPI:
    private_key, key_id = shadow.load_credentials()
    return KalshiLiveAPI(private_key, key_id)


def acquire_lock():
    handle = open(LOCK_FILE, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("another early-move live process holds the lock") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def preflight() -> bool:
    api = load_api()
    balance = api.balance()
    window = shadow.discover_aligned_window()
    print(f"credentials: {'PASS' if balance is not None else 'FAIL'}")
    print(f"balance: ${balance:.2f}" if balance is not None else "balance: unreadable")
    print(f"balance floor: ${BALANCE_FLOOR:.2f}")
    print(f"aligned window: {','.join(sorted(window)) if window else 'none'}")
    print(f"live enabled: {LIVE_ENABLED}")
    print(f"kill file present: {os.path.exists(KILL_FILE)}")
    return (
        balance is not None
        and balance >= BALANCE_FLOOR
        and window is not None
        and not os.path.exists(KILL_FILE)
    )


_LOCK_HANDLE = None


def run_live():
    global _LOCK_HANDLE
    _LOCK_HANDLE = acquire_lock()
    api = load_api()
    balance = api.balance()
    if balance is None or balance < BALANCE_FLOOR:
        raise RuntimeError(f"balance preflight failed: {balance}")
    if not LIVE_ENABLED:
        raise RuntimeError("EARLY_LIVE_ENABLED is not 1")
    if os.path.exists(KILL_FILE):
        raise RuntimeError(f"kill file present: {KILL_FILE}")

    trader = LiveTrader(api)
    initial = shadow.discover_aligned_window()
    initial_tickers = [m["ticker"] for m in initial.values()] if initial else []
    kalshi_orderbook.start(api.private_key, api.key_id, initial_tickers)
    if initial:
        trader.set_window(initial)
    log.warning(
        "LIVE ENABLED minutes=%s BTC>=%.2fc gap>=%.2fc size=%.0f "
        "order=$%.2f window=$%.2f daily=$%.2f loss=$%.2f floor=$%.2f",
        shadow.SAMPLE_MINUTES, shadow.BTC_THRESHOLD * 100, shadow.MIN_GAP * 100,
        LIVE_SIZE, MAX_ORDER_COST, MAX_WINDOW_COST, MAX_DAILY_NOTIONAL,
        DAILY_LOSS_LIMIT, BALANCE_FLOOR,
    )

    next_refresh = next_settle = 0.0
    last_refresh_error = None
    while True:
        loop_start = time.time()
        trader.sample_due(loop_start)
        if loop_start >= next_settle:
            trader.settle_pending()
            next_settle = loop_start + shadow.SETTLE_POLL_SEC
        if loop_start >= next_refresh:
            try:
                window = shadow.discover_aligned_window(loop_start)
                if window:
                    trader.set_window(window)
                last_refresh_error = None
            except Exception as exc:
                msg = str(exc)
                if msg != last_refresh_error:
                    log.warning("market discovery: %s", exc)
                last_refresh_error = msg
            next_refresh = loop_start + shadow.REFRESH_SEC
        time.sleep(max(0.02, shadow.TICK_SEC - (time.time() - loop_start)))


def selftest():
    assert yes_book_price("yes", 0.81) == 0.81
    assert abs(yes_book_price("no", 0.81) - 0.19) < 1e-12
    assert deterministic_coid("a") == deterministic_coid("a")
    assert deterministic_coid("a") != deterministic_coid("b")
    sample = {"count_fp": "10.00", "yes_price_dollars": "0.1900",
              "no_price_dollars": "0.8100", "fee_cost": "0.110000"}
    # Exercise the fixed-point fields used by the live Kalshi API.
    assert _num(sample, "count_fp") == 10
    assert _num(sample, "no_price_dollars") == 0.81
    recovered = actual_from_order({
        "fill_count_fp": "10.00",
        "taker_fill_cost_dollars": "8.100000",
        "taker_fees_dollars": "0.110000",
    }, "no")
    assert recovered["fill_size"] == 10.0
    assert abs(recovered["fill_vwap"] - 0.81) < 1e-12
    assert abs(recovered["fee_dollars"] - 0.11) < 1e-12
    print("early_move_live self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="authenticated GETs only; no order")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.preflight:
        raise SystemExit(0 if preflight() else 1)
    else:
        run_live()


if __name__ == "__main__":
    main()
