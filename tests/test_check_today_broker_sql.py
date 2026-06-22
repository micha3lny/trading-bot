from __future__ import annotations

import unittest

import pandas as pd

from scripts.check_today_broker_sql import (
    closed_execution_symbol_set,
    execution_closed_summary,
    execution_pnl_by_symbol,
    pnl_formula_comparison,
    pnl_formula_totals,
    pnl_symbol_diffs,
)


class CheckTodayBrokerSqlTests(unittest.TestCase):
    def test_execution_pnl_formulas_show_commission_semantics(self) -> None:
        executions = pd.DataFrame(
            [
                {
                    "execution_time": "2026-06-18T13:30:00+00:00",
                    "symbol": "AKTX",
                    "side": "BUY",
                    "quantity": 5,
                    "price": 10.0,
                    "commission": 1.0,
                    "realized_pnl": 0.0,
                },
                {
                    "execution_time": "2026-06-18T14:00:00+00:00",
                    "symbol": "AKTX",
                    "side": "SELL",
                    "quantity": 5,
                    "price": 12.0,
                    "commission": 2.0,
                    "realized_pnl": 10.0,
                },
            ]
        )

        by_symbol = execution_pnl_by_symbol(executions)
        totals = pnl_formula_totals(by_symbol)

        self.assertEqual(closed_execution_symbol_set(executions), {"AKTX"})
        self.assertAlmostEqual(by_symbol["AKTX"]["gross_realized"], 10.0)
        self.assertAlmostEqual(by_symbol["AKTX"]["sell_commission"], 2.0)
        self.assertAlmostEqual(by_symbol["AKTX"]["all_commission"], 3.0)
        self.assertAlmostEqual(totals["net_if_realized_only"], 10.0)
        self.assertAlmostEqual(totals["net_if_realized_minus_sell_commission"], 8.0)
        self.assertAlmostEqual(totals["net_if_realized_minus_all_commission"], 7.0)

    def test_execution_closed_summary_uses_realized_minus_sell_commission(self) -> None:
        by_symbol = {
            "AKTX": {
                "closed_qty": 5.0,
                "gross_realized": 10.0,
                "sell_commission": 2.0,
                "all_commission": 3.0,
                "net_if_realized_only": 10.0,
                "net_if_realized_minus_sell_commission": 8.0,
                "net_if_realized_minus_all_commission": 7.0,
            }
        }

        summary = execution_closed_summary(by_symbol)

        self.assertEqual(summary["closed_symbols"], 1)
        self.assertAlmostEqual(summary["closed_net"], 8.0)
        self.assertEqual(summary["closed_pnl_source"], "executions_realized_pnl_minus_sell_commission")

    def test_symbol_diff_reports_each_formula(self) -> None:
        broker = {
            "AKTX": {
                "closed_qty": 5.0,
                "gross_realized": 10.0,
                "sell_commission": 2.0,
                "all_commission": 3.0,
                "net_if_realized_only": 10.0,
                "net_if_realized_minus_sell_commission": 8.0,
                "net_if_realized_minus_all_commission": 7.0,
            }
        }
        sqlite = {
            "AKTX": {
                "closed_qty": 5.0,
                "gross_realized": 9.0,
                "sell_commission": 1.5,
                "all_commission": 2.5,
                "net_if_realized_only": 9.0,
                "net_if_realized_minus_sell_commission": 7.5,
                "net_if_realized_minus_all_commission": 6.5,
            }
        }

        comparison = pnl_formula_comparison(broker, sqlite)
        diffs = pnl_symbol_diffs(broker, sqlite)

        self.assertAlmostEqual(comparison["gross_realized"]["diff"], 1.0)
        self.assertAlmostEqual(comparison["realized_minus_sell_commission"]["diff"], 0.5)
        self.assertAlmostEqual(comparison["realized_minus_all_commission"]["diff"], 0.5)
        self.assertEqual(diffs[0]["symbol"], "AKTX")
        self.assertAlmostEqual(diffs[0]["diff_realized_only"], 1.0)
        self.assertAlmostEqual(diffs[0]["diff_minus_sell_commission"], 0.5)
        self.assertAlmostEqual(diffs[0]["diff_minus_all_commission"], 0.5)


if __name__ == "__main__":
    unittest.main()
