import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_move_shadow as em


class EarlyMoveShadowTests(unittest.TestCase):
    def test_favored_yes(self):
        self.assertEqual(em.favored_side(0.88, 0.92), ("yes", 0.9))

    def test_favored_no(self):
        side, midpoint = em.favored_side(0.08, 0.12)
        self.assertEqual(side, "no")
        self.assertAlmostEqual(midpoint, 0.9)

    def test_invalid_crossed_quote(self):
        with self.assertRaises(ValueError):
            em.favored_side(0.6, 0.5)

    def test_full_book_vwap(self):
        vwap, fills, visible = em.book_vwap([(0.72, 10), (0.70, 4)], 10)
        self.assertAlmostEqual(vwap, 0.712)
        self.assertEqual(fills, [(0.7, 4.0), (0.72, 6.0)])
        self.assertEqual(visible, 14)

    def test_rejects_partial_depth(self):
        vwap, fills, visible = em.book_vwap([(0.70, 4)], 10)
        self.assertIsNone(vwap)
        self.assertEqual(fills, [(0.7, 4.0)])
        self.assertEqual(visible, 4)

    def test_order_rounded_fee(self):
        self.assertEqual(em.taker_fee([(0.90, 10)]), 0.07)
        self.assertEqual(em.taker_fee([(0.50, 1)]), 0.02)

    def test_exact_thresholds_qualify(self):
        self.assertTrue(em.qualifies(0.90, 0.85))
        self.assertFalse(em.qualifies(0.8999, 0.84))
        self.assertFalse(em.qualifies(0.90, 0.8501))

    def test_yes_settlement_accounting(self):
        win, payout, cost, pnl = em.settlement_pnl("yes", "yes", 10, 0.85, 0.09)
        self.assertTrue(win)
        self.assertEqual(payout, 10)
        self.assertAlmostEqual(cost, 8.59)
        self.assertAlmostEqual(pnl, 1.41)

    def test_no_losing_settlement_accounting(self):
        win, payout, cost, pnl = em.settlement_pnl("no", "yes", 10, 0.80, 0.12)
        self.assertFalse(win)
        self.assertEqual(payout, 0)
        self.assertAlmostEqual(cost, 8.12)
        self.assertAlmostEqual(pnl, -8.12)


if __name__ == "__main__":
    unittest.main()
