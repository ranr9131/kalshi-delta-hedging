import os
import sys
import unittest

os.environ["EARLY_LIVE_ENABLED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_move_live as live


class EarlyMoveLiveTests(unittest.TestCase):
    def test_yes_book_price(self):
        self.assertEqual(live.yes_book_price("yes", 0.81), 0.81)
        self.assertAlmostEqual(live.yes_book_price("no", 0.81), 0.19)

    def test_invalid_price(self):
        with self.assertRaises(ValueError):
            live.yes_book_price("no", 1.0)

    def test_deterministic_coid(self):
        self.assertEqual(live.deterministic_coid("x"), live.deterministic_coid("x"))
        self.assertNotEqual(live.deterministic_coid("x"), live.deterministic_coid("y"))

    def test_actual_from_fixed_point_order(self):
        actual = live.actual_from_order({
            "fill_count_fp": "10.00",
            "taker_fill_cost_dollars": "8.100000",
            "taker_fees_dollars": "0.110000",
        }, "no")
        self.assertEqual(actual["fill_size"], 10)
        self.assertAlmostEqual(actual["fill_vwap"], 0.81)
        self.assertAlmostEqual(actual["fee_dollars"], 0.11)

    def test_risk_allows_bounded_order(self):
        self.assertIsNone(live.risk_decision(
            estimated_cost=8.5, window_cost=0, daily_notional=0,
            daily_pnl=0, balance=219, killed=False,
        ))

    def test_risk_fails_closed(self):
        common = dict(
            estimated_cost=8.5, window_cost=0, daily_notional=0,
            daily_pnl=0, balance=219, killed=False,
        )
        self.assertEqual(live.risk_decision(**{**common, "killed": True}), "kill_file_present")
        self.assertEqual(live.risk_decision(**{**common, "balance": None}), "balance_unreadable")
        self.assertEqual(live.risk_decision(**{**common, "estimated_cost": 10.01}), "max_order_cost")
        self.assertEqual(live.risk_decision(**{**common, "daily_notional": 95}), "max_daily_notional")
        self.assertEqual(live.risk_decision(**{**common, "daily_pnl": -25}), "daily_loss_limit")


if __name__ == "__main__":
    unittest.main()
