import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tennis_lock_shadow import (
    family_name,
    kalshi_fee,
    poly_fee,
    strict_name_match,
    walk_book,
)


class TennisLockShadowTest(unittest.TestCase):
    def test_family_name_is_trailing_token(self):
        self.assertEqual(family_name("Alejandro Davidovich Fokina"), "fokina")
        self.assertEqual(family_name("Juan Carlos Prado Angelo"), "angelo")

    def test_rejects_shared_first_name_false_pair(self):
        self.assertFalse(strict_name_match("Lukas Neumayer", "Luka Talan Lopatic"))
        self.assertFalse(strict_name_match("Juan Carlos Prado Angelo", "Carlos Lopez Montagud"))

    def test_accepts_surname_and_truncation(self):
        self.assertTrue(strict_name_match("Tamara Zidansek", "T. Zidanse"))
        self.assertTrue(strict_name_match("Anna Bondar", "Anna Bondar"))
        self.assertTrue(
            strict_name_match("Alejandro Davidovich", "Alejandro Davidovich Fokina")
        )

    def test_walk_book_requires_full_depth(self):
        vwap, fills, visible, worst = walk_book([(0.2, 2), (0.3, 4)], 5)
        self.assertAlmostEqual(vwap, 0.26)
        self.assertEqual(fills, [(0.2, 2.0), (0.3, 3.0)])
        self.assertEqual(visible, 6.0)
        self.assertEqual(worst, 0.3)
        missing, _, visible, _ = walk_book([(0.2, 2)], 5)
        self.assertIsNone(missing)
        self.assertEqual(visible, 2.0)

    def test_walk_book_respects_hedge_limit(self):
        vwap, _, visible, _ = walk_book([(0.2, 3), (0.25, 3)], 5, limit=0.22)
        self.assertIsNone(vwap)
        self.assertEqual(visible, 3.0)

    def test_fee_models(self):
        self.assertEqual(kalshi_fee([(0.6, 5)]), 0.09)
        self.assertAlmostEqual(poly_fee([(0.6, 5)], 0.02, 1.0), 0.024)
        with self.assertRaises(ValueError):
            poly_fee([(0.6, 5)], 0.02, 2.0)


if __name__ == "__main__":
    unittest.main()
