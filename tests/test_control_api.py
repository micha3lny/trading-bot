from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from src.live_trading.v67_live_top100_expansion_paper_trader import enqueue_overnight_collector_if_due
from src.live_trading.control.control_api import (
    _bool_value,
    _build_history_collector_args,
    _collector_allowed,
    _flatten_request,
    _history_collector_status,
    _in_utc_window,
    _queue_history_collector,
    ControlApiContext,
    process_control_api_commands,
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
                "max_attempts": 1,
                "limit_symbols": 10,
            }
        )
        self.assertIn("--limit-symbols", args)
        self.assertIn("10", args)
        self.assertIn("--max-attempts", args)
        self.assertIn("1", args)
        self.assertIn("--allow-outside-window", args)

    def test_build_history_collector_args_includes_plan_flags(self) -> None:
        args = _build_history_collector_args(
            {
                "start_date": "2026-01-01",
                "end_date": "2026-05-15",
                "session_type": "RTH",
                "plan_only": True,
                "include_weekends": True,
                "retry_failed": True,
            }
        )
        self.assertIn("--plan-only", args)
        self.assertIn("--include-weekends", args)
        self.assertIn("--retry-failed", args)

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
                "max_attempts": 1,
                "force": True,
                "allow_live_session": True,
            },
            force=True,
        )

        self.assertTrue(payload["ok"])
        command = payload["command"]
        self.assertEqual(command["start_date"], "2026-05-15")
        self.assertEqual(command["end_date"], "2026-05-15")
        self.assertEqual(command["max_tasks"], 3000)
        self.assertEqual(command["max_attempts"], 1)

    def test_queue_history_collector_rejects_force_during_live_session_by_default(self) -> None:
        ctx = ControlApiContext(
            ib=None,
            recorder=None,
            managed_positions={},
            runtime_state={
                "market_open_utc": "00:00",
                "market_close_utc": "23:59",
                "history_collector_start_utc": "20:15",
                "history_collector_end_utc": "15:00",
            },
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

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "market_session_active")

    def test_queue_history_collector_allows_explicit_live_session_override(self) -> None:
        ctx = ControlApiContext(
            ib=None,
            recorder=None,
            managed_positions={},
            runtime_state={
                "market_open_utc": "00:00",
                "market_close_utc": "23:59",
                "history_collector_start_utc": "20:15",
                "history_collector_end_utc": "15:00",
            },
            record_lifecycle_fn=lambda *args, **kwargs: None,
        )
        payload = _queue_history_collector(
            ctx,
            {
                "date": "2026-05-15",
                "session_type": "RTH",
                "max_tasks": 3000,
                "force": True,
                "allow_live_session": True,
            },
            force=True,
        )

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["command"]["allow_live_session"])

    def test_flatten_request_can_queue_ibkr_portfolio_orphan(self) -> None:
        class Contract:
            symbol = "AVLN"
            currency = "USD"

        class Item:
            contract = Contract()
            position = 3
            averageCost = 10
            marketPrice = 10.5

        class IB:
            def portfolio(self):
                return [Item()]

        events = []
        ctx = ControlApiContext(
            ib=IB(),
            recorder=None,
            managed_positions={},
            runtime_state={},
            record_lifecycle_fn=lambda *args, **kwargs: events.append((args, kwargs)),
        )

        payload = _flatten_request(ctx, "AVLN", dry_run=False)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["source"], "ibkr_portfolio")
        self.assertEqual(len(ctx.runtime_state["control_api_commands"]), 1)
        self.assertEqual(ctx.runtime_state["control_api_commands"][0]["ibkr_quantity"], 3.0)

    def test_fractional_portfolio_flatten_records_failure_without_order(self) -> None:
        class Contract:
            symbol = "ASST"
            currency = "USD"

        class IB:
            def qualifyContracts(self, contract):
                return [contract]

            def placeOrder(self, contract, order):  # pragma: no cover - must not be called
                raise AssertionError("fractional order should not be placed")

        events = []
        runtime_state = {
            "control_api_commands": [
                {
                    "id": "cmd1",
                    "type": "flatten_symbol",
                    "symbol": "ASST",
                    "action": "SELL",
                    "quantity": 0.2,
                    "ibkr_quantity": 0.2,
                    "source": "ibkr_portfolio",
                    "contract": Contract(),
                }
            ]
        }

        processed = process_control_api_commands(
            ib=IB(),
            recorder=None,
            managed_positions={},
            runtime_state=runtime_state,
            record_lifecycle_fn=lambda *args, **kwargs: events.append((args, kwargs)),
        )

        self.assertEqual(processed, 1)
        self.assertEqual(events[0][0][1], "MANUAL_FLATTEN_FAILED")
        self.assertEqual(events[0][1]["reason"], "fractional_quantity_api_unsupported")
        self.assertEqual(runtime_state["control_api_commands"], [])

    def test_overnight_scheduler_prioritizes_previous_trading_day(self) -> None:
        runtime_state = {}
        args = SimpleNamespace(
            enable_overnight_automation=True,
            overnight_collector_times_utc="20:15,07:00",
            overnight_backlog_collector_times_utc="07:00",
            overnight_prioritize_previous_day=True,
            market_close_utc="20:00",
            overnight_collector_start_date="2026-01-01",
            overnight_backlog_lookback_days=30,
            overnight_daily_collector_max_tasks=3000,
            overnight_collector_max_tasks=3000,
            overnight_collector_max_attempts=5,
            history_collector_client_id=168,
            overnight_collector_retry_failed=False,
        )

        enqueue_overnight_collector_if_due(
            runtime_state,
            args,
            now=datetime(2026, 5, 20, 20, 16, tzinfo=timezone.utc),
        )

        queue = runtime_state["history_collector_commands"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["collector_mode"], "daily")
        self.assertEqual(queue[0]["start_date"], "2026-05-20")
        self.assertEqual(queue[0]["end_date"], "2026-05-20")

    def test_overnight_backlog_slot_runs_daily_before_catchup(self) -> None:
        runtime_state = {}
        args = SimpleNamespace(
            enable_overnight_automation=True,
            overnight_collector_times_utc="20:15,07:00",
            overnight_backlog_collector_times_utc="07:00",
            overnight_prioritize_previous_day=True,
            market_close_utc="20:00",
            overnight_collector_start_date="2026-01-01",
            overnight_backlog_lookback_days=30,
            overnight_daily_collector_max_tasks=3000,
            overnight_collector_max_tasks=3000,
            overnight_collector_max_attempts=5,
            history_collector_client_id=168,
            overnight_collector_retry_failed=False,
        )

        enqueue_overnight_collector_if_due(
            runtime_state,
            args,
            now=datetime(2026, 5, 21, 7, 1, tzinfo=timezone.utc),
        )

        queue = runtime_state["history_collector_commands"]
        self.assertEqual([cmd["collector_mode"] for cmd in queue], ["daily", "backlog"])
        self.assertEqual(queue[0]["start_date"], "2026-05-20")
        self.assertEqual(queue[0]["end_date"], "2026-05-20")
        self.assertEqual(queue[1]["start_date"], "2026-04-20")
        self.assertEqual(queue[1]["end_date"], "2026-05-20")

    def test_overnight_backlog_can_fall_back_to_configured_start_date(self) -> None:
        runtime_state = {}
        args = SimpleNamespace(
            enable_overnight_automation=True,
            overnight_collector_times_utc="07:00",
            overnight_backlog_collector_times_utc="07:00",
            overnight_prioritize_previous_day=False,
            market_close_utc="20:00",
            overnight_collector_start_date="2026-02-01",
            overnight_backlog_lookback_days=0,
            overnight_daily_collector_max_tasks=3000,
            overnight_collector_max_tasks=3000,
            overnight_collector_max_attempts=5,
            history_collector_client_id=168,
            overnight_collector_retry_failed=False,
        )

        enqueue_overnight_collector_if_due(
            runtime_state,
            args,
            now=datetime(2026, 5, 21, 7, 1, tzinfo=timezone.utc),
        )

        queue = runtime_state["history_collector_commands"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["collector_mode"], "backlog")
        self.assertEqual(queue[0]["start_date"], "2026-02-01")
        self.assertEqual(queue[0]["end_date"], "2026-05-20")


if __name__ == "__main__":
    unittest.main()
