"""
Order state machine: converge our live resting orders to the desired
QuotePlan per market, with churn damping (don't cancel/replace for moves
that don't change our score).

Paper mode: orders are simulated in-memory with fake order ids; the same
state machine runs so paper behavior == live behavior minus the API writes.
"""

import logging
import time
from datetime import datetime

import lip_api
import lip_config as cfg

log = logging.getLogger("lip_quoter")


class Quoter:
    def __init__(self, private_key=None, api_key_id=None, paper: bool = True):
        self.paper = paper
        self.pk = private_key
        self.kid = api_key_id
        # live state: {ticker: {"yes": order, "no": order}}
        # order = {order_id, price, size, placed_ts}
        self.orders = {}
        self._paper_seq = 0
        self.pending_cancels = []
        self.reject_cooldown = {}

    # ------------------------------------------------------------ primitives

    def _place(self, ticker: str, side: str, price: int, size: int,
               expiration_ts: int = None):
        if self.paper:
            self._paper_seq += 1
            return {"order_id": f"paper-{self._paper_seq}", "price": price,
                    "size": size, "placed_ts": time.time(),
                    "expiration_ts": expiration_ts}
        o = lip_api.place_resting_bid(
            self.pk, self.kid, ticker, side, price, size,
            expiration_ts=expiration_ts,
        )
        return {"order_id": o.get("order_id"), "price": price, "size": size,
                "placed_ts": time.time(), "expiration_ts": expiration_ts}

    def _cancel(self, order) -> bool:
        if self.paper or not order.get("order_id"):
            return True
        return lip_api.cancel_order(self.pk, self.kid, order["order_id"])

    # ------------------------------------------------------------ public api

    def converge(self, ticker: str, plan, expiration_ts: int = None) -> int:
        """
        Make live orders match plan (QuotePlan). Returns number of API writes.
        Two-phase: cancel every stale side first, THEN place, so a reprice of
        both sides can never transiently rest a crossing pair of our own
        orders (old no bid vs new yes bid).
        """
        writes = 0
        if self.reject_cooldown.get(ticker, 0) > time.time():
            return 0
        cur = self.orders.setdefault(ticker, {})
        desired = {"yes": (plan.yes_price, plan.yes_size),
                   "no": (plan.no_price, plan.no_size)}
        to_place = []
        for side, (want_price, want_size) in desired.items():
            have = cur.get(side)
            if want_price is None:
                if have and self._cancel(have):
                    cur.pop(side, None)
                    writes += 1
                continue
            expiry_changed = bool(have) and expiration_ts is not None and \
                abs(int(have.get("expiration_ts") or 0) - int(expiration_ts)) > 1
            if have and abs(have["price"] - want_price) <= cfg.REPRICE_TOLERANCE \
                    and have["size"] >= 0.6 * want_size and not expiry_changed:
                continue  # close enough; don't churn
            if have:
                if not self._cancel(have):
                    continue  # retry next cycle rather than risk doubling up
                cur.pop(side, None)
                writes += 1
            to_place.append((side, want_price, want_size))
        for side, want_price, want_size in to_place:
            try:
                cur[side] = self._place(
                    ticker, side, want_price, want_size,
                    expiration_ts=expiration_ts,
                )
                writes += 1
            except Exception as e:
                msg = str(e).lower()
                if "insufficient" in msg or "balance" in msg:
                    self.reject_cooldown[ticker] = time.time() + 1800
                    log.warning("place %s rejected for balance - cooling 30min",
                                ticker)
                else:
                    log.warning("place %s %s %dc x%d failed: %s",
                                ticker, side, want_price, want_size, e)
        return writes

    def retry_pending_cancels(self) -> None:
        for item in list(self.pending_cancels):
            if self._cancel(item["order"]):
                self.pending_cancels.remove(item)
                log.info("pending cancel confirmed: %s %s",
                         item["ticker"], item["order"].get("order_id"))

    def withdraw(self, ticker: str) -> None:
        """Cancel both sides; failed cancels remain queued until confirmed."""
        cur = self.orders.get(ticker) or {}
        for side in ("yes", "no"):
            order = cur.get(side)
            if not order:
                continue
            if not self._cancel(order):
                self.pending_cancels.append(
                    {"ticker": ticker, "side": side, "order": order})
                log.warning("cancel failed for %s %s - queued for retry",
                            ticker, side)
            cur.pop(side, None)
        self.orders.pop(ticker, None)

    def withdraw_all(self) -> None:
        for ticker in list(self.orders.keys()):
            self.withdraw(ticker)
        self.retry_pending_cancels()

    def our_ladders(self, ticker: str) -> dict:
        """
        Our resting orders as ladders for lip_scoring.strip_our_orders.
        Paper mode returns {} - simulated orders are NOT in the fetched book,
        so stripping them would delete other participants' size at our price.
        """
        if self.paper:
            return {}
        cur = self.orders.get(ticker) or {}
        out = {}
        for side in ("yes", "no"):
            if cur.get(side):
                o = cur[side]
                out[side] = [(float(o["price"]), float(o["size"]))]
        return out

    def sync_from_exchange(self) -> None:
        """
        Heal drift: adopt our resting lip- orders from the exchange and drop
        local records whose orders no longer rest (filled or cancelled
        out-of-band). Live mode only.
        """
        if self.paper:
            return
        try:
            resting = lip_api.get_resting_orders(self.pk, self.kid)
        except Exception as e:
            log.warning("order sync failed: %s", e)
            return
        seen = {}
        for o in resting:
            client_id = o.get("client_order_id") or ""
            if not client_id.startswith("lip-"):
                continue
            ticker = o.get("ticker")
            parts = client_id.split("-")
            side = parts[1] if len(parts) > 2 and parts[1] in ("yes", "no") \
                else o.get("side")
            pd = o.get("yes_price_dollars")
            if pd is not None:
                yes_c = int(round(float(pd) * 100))
                price = yes_c if side == "yes" else 100 - yes_c
            else:
                price = o.get("yes_price") if side == "yes" else o.get("no_price")
            rem = o.get("remaining_count_fp")
            remaining = float(rem) if rem is not None else \
                o.get("remaining_count", o.get("count", 0))
            if not ticker or price is None:
                continue
            expiration_ts = None
            if o.get("expiration_time"):
                try:
                    expiration_ts = int(datetime.fromisoformat(
                        o["expiration_time"].replace("Z", "+00:00")
                    ).timestamp())
                except (TypeError, ValueError):
                    pass
            rec = {"order_id": o.get("order_id"), "price": int(price),
                   "size": int(remaining), "placed_ts": time.time(),
                   "expiration_ts": expiration_ts}
            prev = seen.setdefault(ticker, {}).get(side)
            if prev:
                # duplicate lip- order on the same side: keep one, cancel the other
                log.warning("duplicate %s %s orders, cancelling %s",
                            ticker, side, rec["order_id"])
                self._cancel(rec)
            else:
                seen[ticker][side] = rec
        # the exchange's resting index lags placements by ~1-2s (verified
        # live): keep local orders placed in the last 30s that sync can't
        # see yet, else we'd orphan them and double-place on next converge
        now = time.time()
        for ticker, cur in self.orders.items():
            for side, o in cur.items():
                if now - o.get("placed_ts", 0) < 30 \
                        and side not in seen.get(ticker, {}):
                    seen.setdefault(ticker, {})[side] = o
        self.orders = seen
