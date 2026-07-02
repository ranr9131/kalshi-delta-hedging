"""
Unit tests for lip_scoring - hand-computed cases from the CFTC filing
mechanics (DF^N discount, target-size qualifying window, two-sided gate,
99c disqualification, per-side normalization).

  python3 test_lip_scoring.py
"""

import math

import lip_scoring as s


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_qualifying_window():
    # empty / zero-size books never qualify
    assert s.qualifying_window([], 100) == ([], None)
    assert s.qualifying_window([(50, 0)], 100) == ([], None)
    # reference at the 99c cap disqualifies the whole side
    assert s.qualifying_window([(99, 5000)], 100) == ([], None)
    # too thin to reach target: qualifying set cleared
    assert s.qualifying_window([(50, 999)], 1000) == ([], None)
    # exact target at one level qualifies
    win, ref = s.qualifying_window([(50, 1000)], 1000)
    assert ref == 50 and win == [(50, 1000)]
    # walk stops at the level that crosses the threshold
    win, ref = s.qualifying_window([(50, 600), (49, 600), (48, 600)], 1000)
    assert ref == 50 and win == [(50, 600), (49, 600)]


def test_side_share_discounting():
    # competitor 600 @ 50c (reference), us 600 @ 49c, target 1000, df 0.5:
    # scores 600 vs 0.5*600=300 -> our share 1/3
    share, ok = s.side_share([(50, 600)], [(49, 600)], 1000, 0.5)
    assert ok
    approx(share, 300.0 / 900.0)
    # same level as competitor: pure size pro-rata
    share, ok = s.side_share([(50, 600)], [(50, 200)], 800, 0.5)
    assert ok
    approx(share, 200.0 / 800.0)
    # below the cutoff: window already full above us -> we score zero
    share, ok = s.side_share([(50, 1000)], [(49, 500)], 1000, 0.5)
    assert ok
    approx(share, 0.0)
    # we alone fill the whole window
    share, ok = s.side_share([], [(2, 1000)], 1000, 0.5)
    assert ok
    approx(share, 1.0)
    # side too thin even with us -> nobody scores
    share, ok = s.side_share([], [(2, 500)], 1000, 0.5)
    assert not ok
    approx(share, 0.0)


def test_two_sided_gate():
    est = s.estimate_snapshot([], [], [(2, 1000)], [], 1000, 0.5)
    assert est.yes_qualifies and not est.no_qualifies
    assert not est.counted
    approx(est.period_share, 0.0)
    est = s.estimate_snapshot([], [], [(2, 1000)], [(2, 1000)], 1000, 0.5)
    assert est.counted
    approx(est.period_share, 1.0)


def test_plan_quotes_empty_book():
    plan = s.plan_quotes([], [], target_size=1000, df=0.5,
                         max_loss_dollars=30.0, empty_price=2)
    assert plan.yes_price == 2 and plan.no_price == 2
    assert plan.yes_size == 1000 and plan.no_size == 1000
    approx(plan.est.period_share, 1.0)
    approx(plan.capital_dollars, 40.0)
    approx(plan.worst_loss_dollars, 20.0)  # one-sided adverse fill


def test_plan_quotes_real_book():
    # the KXNHLPRICE book observed live 2026-06-10: a farmer already rests
    # 1000/1050 @ 1c on both sides, real quotes at 12/13
    yes = [(1, 1000), (3, 237), (12, 8)]
    no = [(1, 1050), (76, 4), (87, 60)]
    plan = s.plan_quotes(yes, no, target_size=1000, df=0.5,
                         max_loss_dollars=30.0)
    # loss budget $30 at size 1000 caps bids at 3c on both sides
    assert plan.yes_price == 3 and plan.no_price == 3
    assert plan.worst_loss_dollars <= 30.0
    assert 0.0 < plan.est.period_share < 0.5
    # we should beat the 1c farmer: at 3c we sit 9 ticks behind ref vs their 11
    yes_share, _ = s.side_share(yes, [(3, 1000)], 1000, 0.5)
    farmer_share, _ = s.side_share(
        [(3, 237), (12, 8)], [(1, 1000)], 1000, 0.5)
    assert yes_share > farmer_share


def test_plan_quotes_never_crosses():
    # moderately competitive cheap book: yes 200@10, no 200@85 (yes ask 15)
    plan = s.plan_quotes([(10, 200)], [(85, 200)], target_size=250,
                         df=0.5, max_loss_dollars=30.0)
    assert plan.yes_price is not None and plan.no_price is not None
    assert plan.yes_price + plan.no_price <= 98     # self-cross guard
    assert plan.yes_price < 15                      # never cross the ask
    assert plan.yes_price * plan.yes_size / 100.0 <= 30.0   # loss budget
    # credible mid 12.5 caps yes bid at mid - 3 = 9 (binds before loss cap 12)
    assert plan.yes_price == 9


def test_plan_quotes_refuses_saturated_book():
    # 2000 resting at the touch with target 250: the qualifying window is
    # saturated at the reference, and joining 60c blows the $30 loss budget,
    # so every affordable price earns zero share -> don't quote at all
    plan = s.plan_quotes([(60, 2000)], [(39, 2000)], target_size=250,
                         df=0.5, max_loss_dollars=30.0)
    approx(plan.est.period_share, 0.0)
    assert plan.yes_price is None and plan.no_price is None


def test_disqualified_side_skipped():
    # someone bids 99 on yes: the yes side can never qualify, snapshot never
    # counts, optimizer should find zero share everywhere
    plan = s.plan_quotes([(99, 5000)], [(2, 50)], target_size=1000, df=0.5,
                         max_loss_dollars=30.0)
    approx(plan.est.period_share, 0.0)


def test_parse_and_strip():
    raw = {"orderbook_fp": {
        "yes_dollars": [["0.0100", "1000.00"], ["0.1200", "8.00"]],
        "no_dollars": [["0.8700", "60.00"]]}}
    yes, no = s.parse_orderbook(raw)
    assert yes == [(1.0, 1000.0), (12.0, 8.0)]
    assert no == [(87.0, 60.0)]
    stripped = s.strip_our_orders(yes, [(1.0, 1000.0)])
    assert stripped == [(12.0, 8.0)]
    # partial overlap
    stripped = s.strip_our_orders(yes, [(1.0, 400.0)])
    assert stripped == [(1.0, 600.0), (12.0, 8.0)]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
