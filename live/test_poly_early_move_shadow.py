import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polymarket_orderbook as pob
import poly_early_move_shadow as em


def view(token, bid, ask, asks=None):
    asks = asks or ((ask, 10.0),)
    return pob.BookView(token, ((bid, 10.0),), tuple(asks), time.time(), time.time(), True)


class PolyEarlyMoveTests(unittest.TestCase):
    def test_favorite_uses_both_token_books(self):
        self.assertEqual(em.favored_side(view("u", .88, .90), view("d", .10, .12)), ("up", .89))
        side, mid = em.favored_side(view("u", .08, .10), view("d", .90, .92))
        self.assertEqual(side, "down")
        self.assertAlmostEqual(mid, .91)

    def test_vwap_walks_depth(self):
        vwap, fills, visible, worst = em.book_vwap(((.80, 4), (.82, 10)), 10)
        self.assertAlmostEqual(vwap, .812)
        self.assertEqual(fills, [(.8, 4.0), (.82, 6.0)])
        self.assertEqual(visible, 14)
        self.assertEqual(worst, .82)

    def test_fok_limit_rejects_disappeared_depth(self):
        vwap, _, visible, worst = em.book_vwap(((.80, 4), (.82, 10)), 10, limit=.80)
        self.assertIsNone(vwap)
        self.assertEqual(visible, 4)
        self.assertIsNone(worst)

    def test_documented_fee_curve(self):
        self.assertEqual(em.taker_fee([(.80, 10)], .07), .112)
        with self.assertRaises(ValueError):
            em.taker_fee([(.50, 10)], .07, 2)

    def test_settlement_accounting(self):
        win, payout, cost, pnl = em.settlement_pnl("down", "down", 10, .80, .112)
        self.assertTrue(win)
        self.assertEqual(payout, 10)
        self.assertAlmostEqual(cost, 8.112)
        self.assertAlmostEqual(pnl, 1.888)
        self.assertEqual(em.settlement_pnl("down", "up", 10, .80, .112)[3], -8.112)


if __name__ == "__main__":
    unittest.main()
