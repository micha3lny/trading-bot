from __future__ import annotations

import unittest

import pandas as pd

from scripts.rebuild_trade_peaks import calculate_peak


def candle_rows(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([row[0] for row in rows], utc=True),
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
        }
    )


def trade(symbol: str, entry: str, exit_: str, entry_price: float, exit_price: float, qty: float = 100) -> dict[str, object]:
    return {
        "trade_id": f"canonical:{symbol}:{entry}:{exit_}",
        "symbol": symbol,
        "entry_fill_time": entry,
        "exit_fill_time": exit_,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": qty,
    }


class RebuildTradePeaksTests(unittest.TestCase):
    def test_profitable_trade_peak_comes_from_window_high(self) -> None:
        result = calculate_peak(
            candle_rows(
                [
                    ("2026-07-16T13:30:00+00:00", 10.10, 9.95),
                    ("2026-07-16T13:31:00+00:00", 10.80, 10.05),
                    ("2026-07-16T13:32:00+00:00", 10.60, 10.20),
                ]
            ),
            trade=trade("FRMM", "2026-07-16T13:30:00+00:00", "2026-07-16T13:32:00+00:00", 10.0, 10.61986),
        )

        self.assertEqual(result.peak_data_quality, "EXACT")
        self.assertEqual(result.validation_status, "OK")
        self.assertAlmostEqual(result.peak_price or 0, 10.8)
        self.assertGreaterEqual(result.peak_pct or 0, 6.1986)
        self.assertAlmostEqual(result.giveback_usd or 0, (10.8 - 10.61986) * 100)

    def test_losing_trade_that_was_briefly_profitable(self) -> None:
        result = calculate_peak(
            candle_rows(
                [
                    ("2026-07-16T14:00:00+00:00", 20.40, 19.95),
                    ("2026-07-16T14:01:00+00:00", 20.10, 19.50),
                    ("2026-07-16T14:02:00+00:00", 19.80, 19.20),
                ]
            ),
            trade=trade("FBYD", "2026-07-16T14:00:00+00:00", "2026-07-16T14:02:00+00:00", 20.0, 19.4, qty=10),
        )

        self.assertEqual(result.validation_status, "OK")
        self.assertGreater(result.peak_pct or 0, 0)
        self.assertLess(result.drop_from_peak_pct or 0, 0)

    def test_losing_trade_that_never_crossed_entry_can_have_negative_peak_pct(self) -> None:
        result = calculate_peak(
            candle_rows(
                [
                    ("2026-07-16T15:00:00+00:00", 9.95, 9.80),
                    ("2026-07-16T15:01:00+00:00", 9.90, 9.60),
                ]
            ),
            trade=trade("VELO", "2026-07-16T15:00:00+00:00", "2026-07-16T15:01:00+00:00", 10.0, 9.7),
        )

        self.assertEqual(result.validation_status, "OK")
        self.assertLess(result.peak_pct or 0, 0)

    def test_partial_exit_trade_uses_canonical_quantity_for_giveback(self) -> None:
        result = calculate_peak(
            candle_rows(
                [
                    ("2026-07-16T13:30:00+00:00", 6.90, 6.60),
                    ("2026-07-16T13:31:00+00:00", 7.10, 6.80),
                ]
            ),
            trade=trade("UCTT", "2026-07-16T13:30:00+00:00", "2026-07-16T13:31:00+00:00", 6.68, 6.85, qty=156),
        )

        self.assertAlmostEqual(result.giveback_usd or 0, (7.10 - 6.85) * 156)
        self.assertEqual(result.validation_status, "OK")

    def test_multiple_trades_same_symbol_same_day_have_separate_windows(self) -> None:
        candles = candle_rows(
            [
                ("2026-07-16T13:30:00+00:00", 5.50, 5.00),
                ("2026-07-16T13:31:00+00:00", 5.30, 5.10),
                ("2026-07-16T14:30:00+00:00", 8.00, 7.00),
                ("2026-07-16T14:31:00+00:00", 7.50, 7.10),
            ]
        )

        first = calculate_peak(candles, trade=trade("BLZE", "2026-07-16T13:30:00+00:00", "2026-07-16T13:31:00+00:00", 5.0, 5.1))
        second = calculate_peak(candles, trade=trade("BLZE", "2026-07-16T14:30:00+00:00", "2026-07-16T14:31:00+00:00", 7.0, 7.2))

        self.assertAlmostEqual(first.peak_price or 0, 5.5)
        self.assertAlmostEqual(second.peak_price or 0, 8.0)

    def test_missing_candles_return_null_peak_values(self) -> None:
        result = calculate_peak(
            pd.DataFrame(),
            trade=trade("MISS", "2026-07-16T13:30:00+00:00", "2026-07-16T13:31:00+00:00", 10.0, 11.0),
        )

        self.assertEqual(result.peak_data_quality, "MISSING_CANDLES")
        self.assertIsNone(result.peak_price)
        self.assertIsNone(result.peak_pct)
        self.assertIsNone(result.drop_from_peak_pct)
        self.assertIsNone(result.giveback_usd)

    def test_mixed_sqlite_timestamp_formats_are_normalized(self) -> None:
        result = calculate_peak(
            candle_rows(
                [
                    ("2026-07-16T13:30:00Z", 10.2, 10.0),
                    ("2026-07-16T13:31:00Z", 10.4, 10.1),
                ]
            ),
            trade=trade("TIME", "2026-07-16 13:30:00+00:00", "2026-07-16T13:31:00+00:00", 10.0, 10.2),
        )

        self.assertEqual(result.peak_data_quality, "EXACT")
        self.assertEqual(result.validation_status, "OK")

    def test_grrr_drop_from_peak_uses_same_window_peak(self) -> None:
        result = calculate_peak(
            candle_rows(
                [
                    ("2026-07-16T16:00:00+00:00", 10.2, 9.9),
                    ("2026-07-16T16:01:00+00:00", 17.8, 15.0),
                    ("2026-07-16T16:02:00+00:00", 10.04, 9.95),
                ]
            ),
            trade=trade("GRRR", "2026-07-16T16:00:00+00:00", "2026-07-16T16:02:00+00:00", 10.0, 10.04, qty=10),
        )

        self.assertAlmostEqual(result.peak_price or 0, 17.8)
        self.assertLess(result.drop_from_peak_pct or 0, -40.0)
        self.assertEqual(result.validation_status, "OK")


if __name__ == "__main__":
    unittest.main()
