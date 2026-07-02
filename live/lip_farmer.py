"""
LIP farmer daemon - earns Kalshi Liquidity Incentive Program rewards by
maintaining resting two-sided depth in program markets, sized and priced by
an exact replica of Kalshi's snapshot scoring.

Run modes:
  PAPER (default, LIP_PAPER=1): full pipeline against live books, no orders
    placed; accrues estimated $/day to lip_accrual.csv for go/no-go.
  LIVE (LIP_PAPER=0): places real resting post-only bids. Requires
    KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY in live/.env.

  python3 lip_farmer.py            # daemon
  python3 lip_farmer.py --once     # one discovery+alloc+quote cycle, then exit

Kill switch: `touch live/lip_kill` -> cancels all LIP orders and exits.

Economics recap (see lip_scoring.py docstring for the scoring math):
revenue needs NO fills - only resting depth. The only loss path is adverse
fills, bounded per market by LIP_MAX_LOSS_PER_MARKET via the price caps in
the optimizer. Fills trigger a cooldown (quotes pulled, position held to
settlement by default since bounded cheap fills are usually lottery tickets,
not liabilities).
"""

import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lip_allocator
import lip_api
import lip_config as cfg
import lip_quoter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    handlers=[logging.FileHandler(cfg.LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("lip_farmer")


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_csv(path, fieldnames, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


ACCRUAL_FIELDS = ["ts", "ticker", "share", "reward_per_day", "accrued_dollars",
                  "yes_price", "no_price", "size", "paper"]
ALLOC_FIELDS = ["ts", "n_chosen", "capital", "worst_loss", "est_per_day", "tickers"]
FILL_FIELDS = ["ts", "ticker", "side", "action", "count", "yes_price", "no_price",
               "is_taker", "order_id"]


class Farmer:
    def __init__(self):
        self.paper = cfg.PAPER
        self.pk, self.kid = None, None
        if not self.paper:
            from dotenv import dotenv_values
            env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
            import kalshi_auth
            self.pk = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
            self.kid = env["KALSHI_API_KEY_ID"]
        self.quoter = lip_quoter.Quoter(self.pk, self.kid, paper=self.paper)
        self.programs = {}
        self.held = {}          # ticker -> Candidate
        self.state = self._load_state()
        self.fills_today = 0
        self.fills_day = datetime.now(timezone.utc).date().isoformat()
        self.paused_until = 0.0
        self._timers = {}
        self._stop = False

    # ------------------------------------------------------------ state

    def _load_state(self):
        try:
            with open(cfg.STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {"fill_cursor_ts": int(time.time()), "cooldowns": {},
                    "positions": {}, "accrual_total": 0.0}

    def _save_state(self):
        self.state["held"] = {
            t: {"share": round(c.plan.est.period_share, 4),
                "est_per_day": round(c.est_dollars_per_day, 2),
                "yes_price": c.plan.yes_price, "no_price": c.plan.no_price,
                "size": c.plan.yes_size or c.plan.no_size,
                "reward_per_day": round(c.program.reward_per_day, 2),
                "capital": round(c.plan.capital_dollars, 2),
                "worst_loss": round(c.plan.worst_loss_dollars, 2),
                "end_date": datetime.fromtimestamp(
                    c.program.end_ts, timezone.utc).isoformat(timespec="seconds")}
            for t, c in self.held.items()}
        self.state["paper"] = self.paper
        self.state["updated"] = _now_iso()
        tmp = cfg.STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, cfg.STATE_FILE)

    def _due(self, name, interval):
        now = time.monotonic()
        if now - self._timers.get(name, 0) >= interval:
            self._timers[name] = now
            return True
        return False

    # ------------------------------------------------------------ cycles

    def refresh_programs(self):
        try:
            self.programs = lip_allocator.fetch_programs()
            log.info("programs: %d active liquidity tickers, $%.0f/day total",
                     len(self.programs),
                     sum(p.reward_per_day for p in self.programs.values()))
        except Exception as e:
            log.warning("program refresh failed: %s", e)

    def reallocate(self):
        if not self.programs:
            return
        if not self.paper:
            try:
                bal = lip_api.get_balance(self.pk, self.kid)
                if bal is not None and bal < cfg.MIN_BALANCE:
                    log.warning("balance $%.2f below floor $%.2f - withdrawing all",
                                bal, cfg.MIN_BALANCE)
                    self.quoter.withdraw_all()
                    self.held = {}
                    return
            except Exception as e:
                log.warning("balance check failed: %s", e)
        chosen = lip_allocator.select_portfolio(
            self.programs,
            held_tickers=set(self.held.keys()),
            our_resting_by_ticker={t: self.quoter.our_ladders(t) for t in self.held},
            cooldowns=self.state.get("cooldowns", {}),
            series_cooldowns=self.state.get("series_cooldowns", {}),
        )
        new_held = {c.program.ticker: c for c in chosen}
        for ticker in set(self.held) - set(new_held):
            log.info("dropping %s", ticker)
            self.quoter.withdraw(ticker)
        self.held = new_held
        _append_csv(cfg.ALLOC_CSV, ALLOC_FIELDS, {
            "ts": _now_iso(), "n_chosen": len(chosen),
            "capital": round(sum(c.plan.capital_dollars for c in chosen), 2),
            "worst_loss": round(sum(c.plan.worst_loss_dollars for c in chosen), 2),
            "est_per_day": round(sum(c.est_dollars_per_day for c in chosen), 2),
            "tickers": " ".join(sorted(new_held)),
        })

    def refresh_quotes(self):
        """Re-plan and converge quotes for held markets; accrue estimates."""
        now = time.time()
        dt = min(now - self.state.get("last_accrual_ts", now), 300.0)
        self.state["last_accrual_ts"] = now
        for ticker, cand in list(self.held.items()):
            if self.state.get("cooldowns", {}).get(ticker, 0) > now:
                self.quoter.withdraw(ticker)
                continue
            fresh = lip_allocator.evaluate_market(
                cand.program, self.quoter.our_ladders(ticker))
            if fresh is None:
                continue
            fresh.market = cand.market
            self.held[ticker] = fresh
            if now < self.paused_until:
                continue
            self.quoter.converge(ticker, fresh.plan)
            self._record_order_ids()
            accrued = fresh.plan.est.period_share * cand.program.reward_per_day * dt / 86400.0
            self.state["accrual_total"] = self.state.get("accrual_total", 0.0) + accrued
            _append_csv(cfg.ACCRUAL_CSV, ACCRUAL_FIELDS, {
                "ts": _now_iso(), "ticker": ticker,
                "share": round(fresh.plan.est.period_share, 4),
                "reward_per_day": round(cand.program.reward_per_day, 2),
                "accrued_dollars": round(accrued, 4),
                "yes_price": fresh.plan.yes_price, "no_price": fresh.plan.no_price,
                "size": fresh.plan.yes_size or fresh.plan.no_size,
                "paper": int(self.paper),
            })

    def _record_order_ids(self):
        """Persist every order id we've placed so fills can be attributed
        even after the order was cancel-replaced or the daemon restarted."""
        if self.paper:
            return
        ids = set(self.state.get("placed_order_ids", []))
        for cur in self.quoter.orders.values():
            for o in cur.values():
                if o.get("order_id"):
                    ids.add(o["order_id"])
        self.state["placed_order_ids"] = list(ids)[-5000:]

    def poll_fills(self):
        if self.paper:
            return
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.fills_day:
            self.fills_day, self.fills_today = today, 0
        try:
            fills = lip_api.get_fills(self.pk, self.kid,
                                      min_ts=self.state.get("fill_cursor_ts"))
        except Exception as e:
            log.warning("fill poll failed: %s", e)
            return
        our_ids = set(self.state.get("placed_order_ids", []))
        our_ids |= {o["order_id"] for cur in self.quoter.orders.values()
                    for o in cur.values() if o.get("order_id")}
        seen = set(self.state.get("seen_fill_ids", []))
        for f in fills:
            fid = f.get("trade_id") or f.get("fill_id") or json.dumps(f, sort_keys=True)
            if fid in seen:
                continue
            seen.add(fid)
            if f.get("order_id") not in our_ids:
                continue
            ticker = f.get("ticker", "")
            side = f.get("side", "yes")
            # API may return integer-cent fields or fixed-point/_dollars only
            cnt = float(f.get("count") or f.get("count_fp") or 0)
            def _px(key):
                if f.get(key) is not None:
                    return float(f[key])
                d = f.get(key + "_dollars")
                return float(d) * 100.0 if d is not None else 0.0
            ypx, npx = _px("yes_price"), _px("no_price")
            px = ypx if side == "yes" else npx
            log.info("FILL %s %s x%s @ y%.0f/n%.0f", ticker, side, cnt, ypx, npx)
            _append_csv(cfg.FILLS_CSV, FILL_FIELDS, {
                "ts": f.get("created_time") or _now_iso(), "ticker": ticker,
                "side": side, "action": f.get("action"),
                "count": cnt, "yes_price": ypx,
                "no_price": npx, "is_taker": f.get("is_taker"),
                "order_id": f.get("order_id"),
            })
            pos = self.state.setdefault("positions", {}).setdefault(
                ticker, {"yes": 0, "no": 0, "cost": 0.0})
            pos[side] = pos.get(side, 0) + cnt
            pos["cost"] = pos.get("cost", 0.0) + cnt * px / 100.0
            # fill reaction: pull quotes, cool the market AND its whole series
            # down (informed flow is correlated across an event's markets)
            self.state.setdefault("cooldowns", {})[ticker] = \
                time.time() + cfg.FILL_COOLDOWN_SEC
            series = ticker.split("-")[0]
            self.state.setdefault("series_cooldowns", {})[series] = \
                time.time() + cfg.SERIES_COOLDOWN_SEC
            self.quoter.withdraw(ticker)
            for held_t in list(self.held):
                if held_t.split("-")[0] == series:
                    log.warning("series quarantine: withdrawing %s", held_t)
                    self.quoter.withdraw(held_t)
                    self.held.pop(held_t, None)
            self.fills_today += 1
        self.state["seen_fill_ids"] = list(seen)[-2000:]
        self.state["fill_cursor_ts"] = int(time.time()) - 120
        if self.fills_today >= cfg.MAX_FILLS_PER_DAY:
            log.warning("circuit breaker: %d fills today >= %d - pausing 6h",
                        self.fills_today, cfg.MAX_FILLS_PER_DAY)
            self.quoter.withdraw_all()
            self.paused_until = time.time() + 6 * 3600

    # ------------------------------------------------------------ lifecycle

    def shutdown(self, *_):
        if self._stop:
            return
        self._stop = True
        log.info("shutting down%s", " - cancelling all LIP orders"
                 if (cfg.CANCEL_ON_EXIT and not self.paper) else "")
        if cfg.CANCEL_ON_EXIT:
            self.quoter.withdraw_all()
        self._save_state()
        sys.exit(0)

    def run(self, once=False):
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        log.info("LIP farmer starting (%s mode), caps: %d markets / $%.0f capital / $%.0f worst-loss",
                 "PAPER" if self.paper else "LIVE", cfg.MAX_MARKETS,
                 cfg.MAX_TOTAL_CAPITAL, cfg.MAX_TOTAL_WORST_LOSS)
        if not self.paper:
            self.quoter.sync_from_exchange()
            log.info("adopted %d markets with resting lip- orders",
                     len(self.quoter.orders))
        self.refresh_programs()
        self.reallocate()
        self.refresh_quotes()
        self._save_state()
        for name, iv in (("programs", 0), ("alloc", 0), ("quotes", 0), ("state", 0)):
            self._timers[name] = time.monotonic()  # startup pass counts; don't re-fire at once
        if once:
            log.info("--once complete: holding %d markets, est $%.2f/day",
                     len(self.held),
                     sum(c.est_dollars_per_day for c in self.held.values()))
            if cfg.CANCEL_ON_EXIT and not self.paper:
                self.quoter.withdraw_all()
            return
        while not self._stop:
            if os.path.exists(cfg.KILL_FILE):
                log.warning("kill file found")
                self.shutdown()
            if self._due("programs", cfg.PROGRAM_REFRESH_SEC):
                self.refresh_programs()
            if self._due("alloc", cfg.ALLOC_INTERVAL_SEC):
                self.reallocate()
            if self._due("quotes", cfg.QUOTE_REFRESH_SEC):
                self.refresh_quotes()
            if self._due("fills", cfg.FILL_POLL_SEC):
                self.poll_fills()
            if self._due("sync", 300):
                self.quoter.sync_from_exchange()
            if self._due("state", cfg.STATE_FLUSH_SEC):
                self._save_state()
            time.sleep(1)


if __name__ == "__main__":
    Farmer().run(once="--once" in sys.argv)
