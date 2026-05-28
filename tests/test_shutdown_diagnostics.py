from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from src.live_trading.unified_logger import daily_log_path
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ShutdownDiagnostics,
    is_us_equity_session_active_now,
)


class ShutdownDiagnosticsTests(unittest.TestCase):
    def test_us_equity_session_active_detection(self) -> None:
        args = SimpleNamespace(market_open_utc="13:30", market_close_utc="20:00")

        self.assertTrue(
            is_us_equity_session_active_now(
                args,
                now=datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc),
            )
        )
        self.assertFalse(
            is_us_equity_session_active_now(
                args,
                now=datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc),
            )
        )

    def test_shutdown_logs_exit_and_unexpected_clean_session_exit(self) -> None:
        args = SimpleNamespace(market_open_utc="13:30", market_close_utc="20:00")
        with tempfile.TemporaryDirectory() as tmp:
            shutdown = ShutdownDiagnostics(log_dir=tmp, args=args)
            shutdown.log_main_loop_exit(reason="duration_elapsed", exit_code=0)
            shutdown.log_exit(reason="duration_elapsed", exit_code=0, recorder_dir="data/live/recorder/2026-05-28")

            content = daily_log_path(tmp).read_text(encoding="utf-8")
            self.assertIn("MAIN_LOOP_EXIT", content)
            self.assertIn("BOT_EXIT", content)
            self.assertIn("BOT_STOP", content)


if __name__ == "__main__":
    unittest.main()
