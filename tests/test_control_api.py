from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.live_trading.control.control_api import (
    _bool_value,
    _build_history_collector_args,
    _collector_allowed,
    _history_collector_status,
    _in_utc_window,
    _queue_history_collector,
    ControlApiContext,
)


class ControlApiGuardTests(unittest.TestCase):
    def test_utc_window_handles_overnight_ranges(self) -> None:
        self.assertTrue(_in_utc_window("20:15", "15:00", datetime(2026, 5, 17, 21, 0, tzinfo=timezone.utc)))
        self.assertTrue(_in_utc_window("20:15", "15:00", datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)))
        self.assertFalse(_in_utc_window("20:15", "15:00", datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)))

    def test_collector_allowed_on_weekend_overnight_window(self) -> None:
        runtime_state = {
            "market_open_utc": "15:00",
            "market_close_utc": "20:00",
            "history_collector_start_utc": "20:15",
            "history_collector_end_utc": "15:00",
        }
        allowed, reason = _collector_allowed(
            runtime_state,
            now=datetime(2026, 5, 17, 21, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "allowed")

    def test_collector_rejects_market_session_without_force(self) -> None:
        runtime_state = {
            "market_open_utc": "15:00",
            "market_close_utc": "20:00",
            "history_collector_start_utc": "20:15",
            "history_collector_end_utc": "15:00",
        }
        allowed, reason = _collector_allowed(
            runtime_state,
            now=datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "market_session_active")

    def test_collector_force_bypasses_session_guard(self) -> None:
        allowed, reason = _collector_allowed({}, force=True, now=datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc))
        self.assertTrue(allowed)
        self.assertEqual(reason, "forced")


class ControlApiHelperTests(unittest.TestCase):
    def test_bool_value_accepts_common_truthy_strings(self) -> None:
        for value in ["1", "true", "yes", "y", "on", True]:
            self.assertTrue(_bool_value(value))
        for value in ["0", "false", "no", "", False]:
            self.assertFalse(_bool_value(value))

    def test_history_collector_status_reports_queue_and_running_state(self) -> None:
        status = _history_collector_status(
            {
                "history_collector_commands": [{"id": "a"}, {"id": "b"}],
                "history_collector_process": None,
                "history_collector_last_run_key": "2026-05-18_RTH",
            }
        )
        self.assertTrue(status["ok"])
        self.assertFalse(status["running"])
        self.assertEqual(status["pending_commands"], 2)
        self.assertEqual(status["last_run_key"], "2026-05-18_RTH")

    def test_build_history_collector_args_includes_limit_symbols(self) -> None:
        args = _build_history_collector_args(
            {
                "start_date": "2026-05-18",
                "end_date": "2026-05-18",
                "session_type": "RTH",
                "client_id": 168,
                "max_tasks": 300,
                "limit_symbols": 10,
            }
        )
        self.assertIn("--limit-symbols", args)
        self.assertIn("10", args)
        self.assertIn("--allow-outside-window", args)

    def test_build_history_collector_args_includes_plan_flags(self) -> None:
        args = _build_history_collector_args(
            {
                "start_date": "2026-01-01",
                "end_date": "2026-05-15",
                "session_type": "RTH",
                "plan_only": True,
                "include_weekends": True,
            }
        )
        self.assertIn("--plan-only", args)
        self.assertIn("--include-weekends", args)

    def test_queue_history_collector_date_sets_single_day_range(self) -> None:
        ctx = ControlApiContext(
            ib=None,
            recorder=None,
            managed_positions={},
            runtime_state={},
            record_lifecycle_fn=lambda *args, **kwargs: None,
        )
        payload = _queue_history_collector(
            ctx,
            {
                "date": "2026-05-15",
                "session_type": "RTH",
                "max_tasks": 3000,
                "force": True,
            },
            force=True,
        )

        self.assertTrue(payload["ok"])
        command = payload["command"]
        self.assertEqual(command["start_date"], "2026-05-15")
        self.assertEqual(command["end_date"], "2026-05-15")
        self.assertEqual(command["max_tasks"], 3000)


if __name__ == "__main__":
    unittest.main()
