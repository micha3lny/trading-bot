from __future__ import annotations

import unittest
from datetime import date

from src.live_trading.market_calendar import (
    get_us_equity_session,
    is_us_equity_trading_day,
    previous_us_equity_trading_day,
)


class MarketCalendarTests(unittest.TestCase):
    def test_memorial_day_2026_is_market_closed(self) -> None:
        session = get_us_equity_session(date(2026, 5, 25))

        self.assertFalse(is_us_equity_trading_day(date(2026, 5, 25)))
        self.assertFalse(session.is_trading_day)
        self.assertEqual(session.reason, "memorial_day")
        self.assertIsNone(session.open_utc)
        self.assertIsNone(session.close_utc)

    def test_next_day_after_memorial_day_2026_is_regular_session(self) -> None:
        session = get_us_equity_session(date(2026, 5, 26))

        self.assertTrue(is_us_equity_trading_day(date(2026, 5, 26)))
        self.assertTrue(session.is_trading_day)
        self.assertFalse(session.is_early_close)
        self.assertEqual(session.open_utc.isoformat(), "2026-05-26T13:30:00+00:00")
        self.assertEqual(session.close_utc.isoformat(), "2026-05-26T20:00:00+00:00")

    def test_previous_trading_day_skips_memorial_day_weekend(self) -> None:
        self.assertEqual(previous_us_equity_trading_day(date(2026, 5, 26)), date(2026, 5, 22))


if __name__ == "__main__":
    unittest.main()
