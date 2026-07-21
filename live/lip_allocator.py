"""
LIP portfolio allocator.

Pulls all active liquidity programs, scans orderbooks for the most promising
ones (top-N by reward rate + a random exploration slice + everything we
currently hold), plans optimal quotes per market via lip_scoring, and greedily
builds the target portfolio under the global caps.

Ranking metric: estimated $/day = period_share x period_reward / period_days,
with a hysteresis bonus for incumbents to avoid churn.
"""

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import lip_api
import lip_config as cfg
import lip_safe_window
import lip_scoring as scoring

log = logging.getLogger("lip_alloc")


def _parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


@dataclass
class Program:
    ticker: str
    series_ticker: str
    program_id: str
    reward_dollars: float          # full period reward
    reward_per_day: float
    target_size: float
    df: float
    start_ts: float
    end_ts: float
    description: str = ""

    @property
    def remaining_frac(self) -> float:
        now = time.time()
        total = max(1.0, self.end_ts - self.start_ts)
        return max(0.0, min(1.0, (self.end_ts - now) / total))


@dataclass
class Candidate:
    program: Program
    plan: scoring.QuotePlan
    market: dict = field(default_factory=dict)
    book_ts: float = 0.0
    safe_deadline_ts: float = 0.0
    safe_policy: str = ""
    safe_reason: str = ""

    @property
    def est_dollars_per_day(self) -> float:
        return self.plan.est.period_share * self.program.reward_per_day

    @property
    def est_remaining_payout(self) -> float:
        # joining now only earns share of remaining snapshots in the period
        return (self.plan.est.period_share * self.program.reward_dollars
                * self.program.remaining_frac)


def fetch_programs() -> dict:
    """Active liquidity programs -> {ticker: Program}. If a ticker has several
    concurrent programs, rewards add; we keep the dominant one's mechanics
    (they all use the same df/target in practice) and sum reward rates."""
    raw = lip_api.get_liquidity_programs(status="active")
    out = {}
    now = time.time()
    for p in raw:
        try:
            start, end = _parse_ts(p["start_date"]), _parse_ts(p["end_date"])
            if end <= now:
                continue
            days = max(1.0 / 24.0, (end - start) / 86400.0)
            reward = (p.get("period_reward") or 0) / 10000.0  # centi-cents -> $
            prog = Program(
                ticker=p["market_ticker"],
                series_ticker=p.get("series_ticker") or p["market_ticker"].split("-")[0],
                program_id=p["id"],
                reward_dollars=reward,
                reward_per_day=reward / days,
                target_size=float(p.get("target_size_fp") or 0) or 100.0,
                df=(p.get("discount_factor_bps") or 5000) / 10000.0,
                start_ts=start, end_ts=end,
                description=p.get("incentive_description") or "",
            )
        except Exception as e:
            log.warning("bad program record %s: %s", p.get("id"), e)
            continue
        prev = out.get(prog.ticker)
        if prev:
            prev.reward_dollars += prog.reward_dollars
            prev.reward_per_day += prog.reward_per_day
        else:
            out[prog.ticker] = prog
    return out


def _series_of(ticker: str) -> str:
    return ticker.split("-")[0]


def _eligible(prog: Program) -> bool:
    if prog.reward_per_day < cfg.MIN_REWARD_PER_DAY:
        return False
    if _series_of(prog.ticker) in cfg.SERIES_BLACKLIST:
        return False
    if not lip_safe_window.series_supported(prog):
        return False
    return True


def evaluate_market(prog: Program, our_resting: dict = None) -> Optional[Candidate]:
    """Fetch the book and plan optimal quotes. our_resting: {'yes': [(p,s)],
    'no': [(p,s)]} of OUR current orders so they're stripped before scoring."""
    try:
        raw = lip_api.get_orderbook(prog.ticker)
    except Exception as e:
        log.warning("orderbook %s failed: %s", prog.ticker, e)
        return None
    if raw is None:
        log.warning("orderbook %s returned None (rate limit?)", prog.ticker)
        return None
    yes, no = scoring.parse_orderbook(raw)
    if our_resting:
        yes = scoring.strip_our_orders(yes, our_resting.get("yes", []))
        no = scoring.strip_our_orders(no, our_resting.get("no", []))
    plan = scoring.plan_quotes(
        yes, no,
        target_size=prog.target_size, df=prog.df,
        size_mult=cfg.SIZE_MULT,
        max_loss_dollars=cfg.MAX_LOSS_PER_MARKET,
        empty_price=cfg.EMPTY_BOOK_PRICE,
        max_behind=cfg.MAX_BEHIND_TICKS,
        allow_improve=cfg.ALLOW_IMPROVE,
        mid_margin=cfg.MID_MARGIN_CENTS,
        max_credible_spread=cfg.MAX_CREDIBLE_SPREAD,
        self_cross_gap=cfg.SELF_CROSS_GAP,
    )
    return Candidate(program=prog, plan=plan, book_ts=time.time())


def select_portfolio(programs: dict, held_tickers: set,
                     our_resting_by_ticker: dict = None,
                     cooldowns: dict = None,
                     series_cooldowns: dict = None) -> list:
    """
    Returns the chosen list of Candidates (sorted by est $/day desc).
    Scans: all held tickers + top SCAN_TOP_N by reward rate + EXPLORE_N random.
    """
    now = time.time()
    cooldowns = cooldowns or {}
    series_cooldowns = series_cooldowns or {}
    eligible = [p for p in programs.values()
                if _eligible(p) and cooldowns.get(p.ticker, 0) < now
                and series_cooldowns.get(_series_of(p.ticker), 0) < now]

    # filter on market metadata (status / close time) in batch
    by_reward = sorted(eligible, key=lambda p: -p.reward_per_day)
    scan = list({p.ticker: p for p in (
        [programs[t] for t in held_tickers
         if t in programs and _eligible(programs[t])
         and cooldowns.get(t, 0) < now
         and series_cooldowns.get(_series_of(t), 0) < now]
        + by_reward[:cfg.SCAN_TOP_N]
        + random.sample(by_reward[cfg.SCAN_TOP_N:],
                        min(cfg.EXPLORE_N, max(0, len(by_reward) - cfg.SCAN_TOP_N)))
    )}.values())

    markets = lip_api.get_markets_by_tickers([p.ticker for p in scan])
    min_close = now + cfg.MIN_HOURS_TO_CLOSE * 3600
    candidates = []
    safe_rejects = Counter()
    for prog in scan:
        m = markets.get(prog.ticker)
        if not m or m.get("status") not in ("active", "open"):
            continue
        try:
            if _parse_ts(m["close_time"]) < min_close:
                continue
        except Exception:
            pass
        safe = lip_safe_window.assess(prog, m, now_ts=now)
        if not safe.allowed:
            safe_rejects[safe.policy] += 1
            continue
        cand = evaluate_market(prog, (our_resting_by_ticker or {}).get(prog.ticker))
        if cand is None:
            continue
        cand.market = m
        cand.safe_deadline_ts = safe.deadline_ts or 0.0
        cand.safe_policy = safe.policy
        cand.safe_reason = safe.reason
        if cand.est_remaining_payout < cfg.MIN_EXPECTED_PAYOUT:
            continue
        if cand.plan.est.period_share <= 0:
            continue
        candidates.append(cand)

    # greedy fill under caps, with hysteresis for incumbents
    def rank_key(c: Candidate):
        bonus = cfg.HYSTERESIS if c.program.ticker in held_tickers else 1.0
        return -c.est_dollars_per_day * bonus

    candidates.sort(key=rank_key)
    chosen, cap_used, loss_used = [], 0.0, 0.0
    event_counts = Counter()
    for c in candidates:
        if len(chosen) >= cfg.MAX_MARKETS:
            break
        cap = c.plan.capital_dollars
        loss = c.plan.worst_loss_dollars
        if cap_used + cap > cfg.MAX_TOTAL_CAPITAL:
            continue
        if loss_used + loss > cfg.MAX_TOTAL_WORST_LOSS:
            continue
        event_key = lip_safe_window.risk_group(c.program, c.market)
        if event_counts[event_key] >= cfg.MAX_MARKETS_PER_EVENT:
            continue
        chosen.append(c)
        cap_used += cap
        loss_used += loss
        event_counts[event_key] += 1
    log.info("allocation: %d candidates -> %d chosen, capital $%.0f, worst-loss $%.0f, est $%.2f/day",
             len(candidates), len(chosen), cap_used, loss_used,
             sum(c.est_dollars_per_day for c in chosen))
    if safe_rejects:
        log.info("safe-window rejects: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(safe_rejects.items())))
    return chosen
