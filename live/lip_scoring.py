"""
Exact replica of Kalshi's LIP snapshot scoring (CFTC filing Sept 2025,
amended Feb 2026), plus the quote optimizer built on top of it.

Scoring, per snapshot, per side (yes bids and no bids are scored
independently; a resting yes ask IS a no bid in Kalshi's book model):

  1. Reference price = best bid on the side. Must exist and be < 99c
     (filing: "less than the highest possible price"), else the side has
     no qualifying set.
  2. Walk down levels from the reference accumulating size until cumulative
     size >= target_size. Levels visited are the qualifying set; deeper
     levels score zero. If the whole side can't reach target_size, the
     qualifying set is cleared - nobody scores.
     (Assumption: the level that crosses the threshold qualifies in full.
     The filing language is per-bid, not pro-rata; if Kalshi pro-rates the
     boundary level our share estimates are slightly optimistic there.)
  3. Score(level) = discount_factor^(cents behind reference) * size,
     normalized per side so each side's scores sum to 1.
  4. Post-2026-02-28: the snapshot is excluded entirely (pays nobody)
     unless BOTH sides reach target_size.

All prices are float cents (books can be fractional), sizes float contracts.
A "ladder" is a list of (price_cents, size) in any order.

Everything here is pure - no I/O - so it's unit-testable and reusable by
the allocator, the quoter, and paper-mode accrual.
"""

from dataclasses import dataclass, field
from typing import Optional

MAX_BID_CENTS = 99.0  # highest possible bid; a reference AT this disqualifies the side


# --------------------------------------------------------------------------
# Core scoring replica
# --------------------------------------------------------------------------

def qualifying_window(ladder: list, target_size: float):
    """
    Return (levels, reference_price) where levels is the qualifying set as
    [(price, size), ...] sorted best-first, or ([], None) if the side fails
    to qualify (no bids, reference at price cap, or can't reach target).
    """
    if not ladder or target_size <= 0:
        return [], None
    levels = sorted(((p, s) for p, s in ladder if s > 0), key=lambda x: -x[0])
    if not levels:
        return [], None
    ref = levels[0][0]
    if ref >= MAX_BID_CENTS:
        return [], None
    out, cum = [], 0.0
    for p, s in levels:
        out.append((p, s))
        cum += s
        if cum >= target_size:
            return out, ref
    return [], None  # whole side too thin: qualifying set cleared


def side_share(others: list, ours: list, target_size: float, df: float):
    """
    Our normalized score share on one side.
    others/ours: ladders [(price_cents, size), ...]. Our orders merge into
    the combined book (same level as others = same discount, score is
    size-pro-rata within the level automatically since entries are kept
    separate).
    Returns (our_share in [0,1], side_qualifies: bool).
    """
    combined = [(p, s, False) for p, s in others if s > 0] + \
               [(p, s, True) for p, s in ours if s > 0]
    window, ref = qualifying_window([(p, s) for p, s, _ in combined], target_size)
    if ref is None:
        return 0.0, False
    cutoff = window[-1][0]  # worst qualifying price
    total, mine = 0.0, 0.0
    for p, s, is_ours in combined:
        if p < cutoff:
            continue
        score = (df ** max(0.0, ref - p)) * s
        total += score
        if is_ours:
            mine += score
    if total <= 0:
        return 0.0, False
    return mine / total, True


@dataclass
class SnapshotEstimate:
    share_yes: float = 0.0
    share_no: float = 0.0
    yes_qualifies: bool = False
    no_qualifies: bool = False

    @property
    def counted(self) -> bool:
        """Snapshot pays anyone at all (two-sided gate)."""
        return self.yes_qualifies and self.no_qualifies

    @property
    def period_share(self) -> float:
        """
        Our fraction of this snapshot's total payout weight. Each counted
        snapshot distributes 2.0 normalized points (1 per side); the period
        reward is split by sum-of-points, so our fraction is the mean of
        our two side shares.
        """
        if not self.counted:
            return 0.0
        return (self.share_yes + self.share_no) / 2.0


def estimate_snapshot(yes_others: list, no_others: list,
                      yes_ours: list, no_ours: list,
                      target_size: float, df: float) -> SnapshotEstimate:
    sy, qy = side_share(yes_others, yes_ours, target_size, df)
    sn, qn = side_share(no_others, no_ours, target_size, df)
    return SnapshotEstimate(share_yes=sy, share_no=sn,
                            yes_qualifies=qy, no_qualifies=qn)


# --------------------------------------------------------------------------
# Quote optimizer
# --------------------------------------------------------------------------

@dataclass
class QuotePlan:
    """Desired resting bids for one market. price=None means don't quote side."""
    yes_price: Optional[int] = None   # integer cents (orders are placed in cents)
    yes_size: int = 0
    no_price: Optional[int] = None
    no_size: int = 0
    est: SnapshotEstimate = field(default_factory=SnapshotEstimate)

    @property
    def capital_dollars(self) -> float:
        cap = 0.0
        if self.yes_price:
            cap += self.yes_price * self.yes_size / 100.0
        if self.no_price:
            cap += self.no_price * self.no_size / 100.0
        return cap

    @property
    def worst_loss_dollars(self) -> float:
        """
        Worst case is ONE side adversely filled and losing at settlement
        (if both bids fill, the paired contracts lock in 100 - py - pn > 0,
        so two-sided fills are not the bad case).
        """
        wy = self.yes_price * self.yes_size / 100.0 if self.yes_price else 0.0
        wn = self.no_price * self.no_size / 100.0 if self.no_price else 0.0
        return max(wy, wn)


def _best(ladder: list) -> Optional[float]:
    prices = [p for p, s in ladder if s > 0]
    return max(prices) if prices else None


def _credible_mid(yes_others: list, no_others: list, max_spread: int) -> Optional[float]:
    """Mid in yes-cents if both sides have real quotes and spread is sane."""
    yb, nb = _best(yes_others), _best(no_others)
    if yb is None or nb is None:
        return None
    ya = 100.0 - nb  # best no bid IS the yes ask
    if ya <= yb:     # crossed/garbage book
        return None
    if ya - yb > max_spread:
        return None
    return (yb + ya) / 2.0


def _side_candidates(others: list, opposite_others: list, *,
                     empty_price: int, max_behind: int, allow_improve: bool,
                     price_cap: int) -> list:
    """
    Candidate integer-cent prices for our bid on one side, best-first.
    Constraints applied here:
      - never cross others' ask on this side  (our bid < 100 - opposite_ref)
      - never bid above price_cap (risk budget / mid margin)
      - join-only unless allow_improve
    """
    ref = _best(others)
    opp_ref = _best(opposite_others)
    no_cross_max = int(100 - opp_ref - 1) if opp_ref is not None else 98
    hard_max = min(price_cap, no_cross_max, 98)
    if hard_max < 1:
        return []
    cands = []
    if ref is None:
        cands.append(min(empty_price, hard_max))
    else:
        top = int(ref) + 1 if allow_improve else int(ref)
        for p in range(min(top, hard_max), max(1, int(ref) - max_behind) - 1, -1):
            if 1 <= p <= hard_max:
                cands.append(p)
        if not cands:
            # everything near the reference is above our cap: bid the cap
            cands.append(hard_max)
    return cands


def plan_quotes(yes_others: list, no_others: list, *,
                target_size: float, df: float,
                size_mult: float = 1.0,
                max_loss_dollars: float = 30.0,
                empty_price: int = 2,
                max_behind: int = 5,
                allow_improve: bool = False,
                mid_margin: int = 3,
                max_credible_spread: int = 15,
                self_cross_gap: int = 2) -> QuotePlan:
    """
    Choose our (yes_price, no_price) to maximize estimated period share
    subject to the risk constraints. Greedy over the small candidate grid -
    candidates are ordered best-share-first per side, and share is monotone
    in price proximity to the reference, but the self-cross constraint
    couples the sides, so we evaluate the cross product (tiny: <= ~50 pairs).
    """
    size = max(1, int(round(target_size * size_mult)))

    # price cap from the per-market loss budget: a bid at p filled and lost
    # costs p*size cents
    loss_cap = max(1, int((max_loss_dollars * 100) // size))

    # price cap from credible mid (don't rest near/above fair value)
    mid = _credible_mid(yes_others, no_others, max_credible_spread)
    yes_cap = loss_cap if mid is None else min(loss_cap, int(mid - mid_margin))
    no_cap = loss_cap if mid is None else min(loss_cap, int((100 - mid) - mid_margin))

    yes_cands = _side_candidates(yes_others, no_others,
                                 empty_price=empty_price, max_behind=max_behind,
                                 allow_improve=allow_improve, price_cap=yes_cap)
    no_cands = _side_candidates(no_others, yes_others,
                                empty_price=empty_price, max_behind=max_behind,
                                allow_improve=allow_improve, price_cap=no_cap)

    best_plan = QuotePlan()
    # also consider one-sided quoting (other side None) in case the cross
    # constraint forbids any pair
    for py in yes_cands + [None]:
        for pn in no_cands + [None]:
            if py is None and pn is None:
                continue
            if py is not None and pn is not None and py + pn > 100 - self_cross_gap:
                continue
            est = estimate_snapshot(
                yes_others, no_others,
                [(float(py), float(size))] if py is not None else [],
                [(float(pn), float(size))] if pn is not None else [],
                target_size, df)
            plan = QuotePlan(yes_price=py, yes_size=size if py else 0,
                             no_price=pn, no_size=size if pn else 0, est=est)
            if plan.est.period_share > best_plan.est.period_share + 1e-12:
                best_plan = plan
    return best_plan


# --------------------------------------------------------------------------
# Book parsing (REST /markets/{t}/orderbook -> ladders in cents)
# --------------------------------------------------------------------------

def parse_orderbook(raw: dict):
    """
    Accepts the v2 REST response (either `orderbook_fp` dollar-string form
    or legacy `orderbook` integer-cents form) and returns
    (yes_bids, no_bids) ladders as [(price_cents, size), ...].
    """
    if "orderbook_fp" in raw:
        ob = raw["orderbook_fp"] or {}
        yes = [(float(p) * 100.0, float(s)) for p, s in (ob.get("yes_dollars") or [])]
        no = [(float(p) * 100.0, float(s)) for p, s in (ob.get("no_dollars") or [])]
        return yes, no
    ob = raw.get("orderbook") or {}
    yes = [(float(p), float(s)) for p, s in (ob.get("yes") or [])]
    no = [(float(p), float(s)) for p, s in (ob.get("no") or [])]
    return yes, no


def strip_our_orders(ladder: list, ours: list) -> list:
    """
    Remove our own resting size from a fetched ladder so share estimation
    doesn't double-count us. ours: [(price_cents, size), ...].
    """
    out = []
    remaining = {}
    for p, s in ours:
        remaining[round(p, 4)] = remaining.get(round(p, 4), 0.0) + s
    for p, s in ladder:
        key = round(p, 4)
        if key in remaining and remaining[key] > 0:
            take = min(s, remaining[key])
            remaining[key] -= take
            s -= take
        if s > 1e-9:
            out.append((p, s))
    return out
