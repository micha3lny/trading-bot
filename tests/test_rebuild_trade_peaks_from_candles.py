from __future__ import annotations

import unittest

import pandas as pd

from scripts.rebuild_trade_peaks_from_candles import calculate_peak_metrics


def candles(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([row[0] for row in rows], utc=True),
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
        }
    )


class TradePeakRebuildTests(unittest.TestCase):
    def test_frmm_profitable_trade_gets_peak_above_gross_return(self) -> None:
        stats = calculate_peak_metrics(
            candles(
                [
                    ("2026-07-16T13:30:00+00:00", 10.20, 9.95),
                    ("2026-07-16T13:35:00+00:00", 10.85, 10.10),
                    ("2026-07-16T13:40:00+00:00", 10.65, 10.30),
                ]
            ),
            entry_time="2026-07-16T13:30:00+00:00",
            exit_time="2026-07-16T13:40:00+00:00",
            entry_price=10.0,
            exit_price=10.62,
            quantity=100,
            net_pnl=61.986,
        )

        self.assertEqual(stats.validator_status, "OK")
        self.assertGreaterEqual(stats.mfe_pct or -1, 6.1986)
        self.assertAlmostEqual(stats.peak_price or 0, 10.85)
        self.assertLess(stats.drop_from_peak_pct or 0, 0)

    def test_uctt_uses_exit_price_when_candle_high_misses_profitable_exit(self) -> None:
        stats = calculate_peak_metrics(
            candles(
                [
                    ("2026-07-16T14:00:00+00:00", 20.10, 19.90),
                    ("2026-07-16T14:01:00+00:00", 20.20, 20.00),
                ]
            ),
            entry_time="2026-07-16T14:00:00+00:00",
            exit_time="2026-07-16T14:02:00+00:00",
            entry_price=20.0,
            exit_price=20.35324,
            quantity=50,
            net_pnl=17.662,
        )

        self.assertEqual(stats.peak_data_quality, "INCOMPLETE")
        self.assertEqual(stats.validator_status, "OK")
        self.assertAlmostEqual(stats.mfe_pct or 0, 1.7662, places=3)
        self.assertIn("peak_from_exit_price", stats.notes)

    def test_blze_small_winner_does_not_get_zero_peak(self) -> None:
        stats = calculate_peak_metrics(
            candles(
                [
                    ("2026-07-16T15:00:00+00:00", 8.05, 7.90),
                    ("2026-07-16T15:04:00+00:00", 8.11, 7.98),
                ]
            ),
            entry_time="2026-07-16T15:00:00+00:00",
            exit_time="2026-07-16T15:04:00+00:00",
            entry_price=8.0,
            exit_price=8.10116,
            quantity=100,
            net_pnl=10.116,
        )

        self.assertIsNotNone(stats.mfe_pct)
        self.assertGreater(stats.mfe_pct or 0, 0)
        self.assertEqual(stats.validator_status, "OK")

    def test_grrr_large_drop_from_peak_uses_same_peak_source(self) -> None:
        stats = calculate_peak_metrics(
            candles(
                [
                    ("2026-07-16T16:00:00+00:00", 10.20, 9.90),
                    ("2026-07-16T16:10:00+00:00", 17.80, 15.00),
                    ("2026-07-16T16:20:00+00:00", 10.04, 9.95),
                ]
            ),
            entry_time="2026-07-16T16:00:00+00:00",
            exit_time="2026-07-16T16:20:00+00:00",
            entry_price=10.0,
            exit_price=10.04,
            quantity=10,
            net_pnl=4.0,
        )

        self.assertAlmostEqual(stats.peak_price or 0, 17.8)
        self.assertAlmostEqual(stats.mfe_pct or 0, 78.0)
        self.assertLess(stats.drop_from_peak_pct or 0, -40.0)
        self.assertEqual(stats.validator_status, "OK")

    def test_missing_candles_are_missing_not_zero(self) -> None:
        stats = calculate_peak_metrics(
            pd.DataFrame(),
            entry_time="2026-07-16T13:30:00+00:00",
            exit_time="2026-07-16T13:40:00+00:00",
            entry_price=10.0,
            exit_price=11.0,
            quantity=1,
            net_pnl=1.0,
        )

        self.assertEqual(stats.peak_data_quality, "MISSING")
        self.assertIsNone(stats.peak_price)
        self.assertIsNone(stats.mfe_pct)


if __name__ == "__main__":
    unittest.main()
