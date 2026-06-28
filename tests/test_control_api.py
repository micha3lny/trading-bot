from __future__ import annotations

import unittest
from datetime import datetime, timezone
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.live_trading.v67_live_top100_expansion_paper_trader import (
    apply_top100_freshness_gate,
    enqueue_overnight_collector_if_due,
    enqueue_startup_history_repair_if_needed,
    history_parquet_path,
    history_task_key,
    process_daily_top100_build,
)
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

    def test_build_history_collector_args_includes_priority_recent_catchup(self) -> None:
        args = _build_history_collector_args(
            {
                "start_date": "2026-01-01",
                "end_date": "2026-06-22",
                "priority_recent_catchup": True,
                "recent_sessions": 5,
            }
        )
        self.assertIn("--priority-recent-catchup", args)
        self.assertIn("--recent-sessions", args)
        self.assertIn("5", args)

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

    def test_queue_history_collector_defaults_to_runtime_latest_session_not_year_start(self) -> None:
        ctx = ControlApiContext(
            ib=None,
            recorder=None,
            managed_positions={},
            runtime_state={"history_collector_default_date": "2026-06-22"},
            record_lifecycle_fn=lambda *args, **kwargs: None,
        )
        payload = _queue_history_collector(
            ctx,
            {
                "session_type": "RTH",
                "force": True,
                "allow_live_session": True,
            },
            force=True,
        )

        self.assertTrue(payload["ok"])
        command = payload["command"]
        self.assertEqual(command["start_date"], "2026-06-22")
        self.assertEqual(command["end_date"], "2026-06-22")
        self.assertNotEqual(command["start_date"], "2026-01-01")

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
        self.assertEqual([cmd["collector_mode"] for cmd in queue], ["daily"])
        self.assertEqual(queue[0]["start_date"], "2026-05-20")
        self.assertEqual(queue[0]["end_date"], "2026-05-20")

    def test_overnight_backlog_runs_only_after_latest_day_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            status = {
                history_task_key("AAA", datetime(2026, 5, 20, tzinfo=timezone.utc).date()): {"status": "complete"},
                history_task_key("BBB", datetime(2026, 5, 20, tzinfo=timezone.utc).date()): {"status": "no_data"},
            }
            (root / "history").mkdir(parents=True, exist_ok=True)
            (root / "history" / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")
            runtime_state = {}
            args = SimpleNamespace(
                enable_overnight_automation=True,
                overnight_collector_times_utc="07:00",
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
                startup_history_repair_min_completion_pct=100.0,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
            )

            enqueue_overnight_collector_if_due(
                runtime_state,
                args,
                now=datetime(2026, 5, 21, 7, 1, tzinfo=timezone.utc),
            )

            queue = runtime_state["history_collector_commands"]
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["collector_mode"], "backlog")
            self.assertEqual(queue[0]["start_date"], "2026-04-20")
            self.assertEqual(queue[0]["end_date"], "2026-05-20")
            self.assertTrue(queue[0]["priority_recent_catchup"])
            self.assertEqual(queue[0]["recent_sessions"], 5)

    def test_overnight_scheduler_skips_holiday_monday_to_previous_friday(self) -> None:
        runtime_state = {}
        args = SimpleNamespace(
            enable_overnight_automation=True,
            overnight_collector_times_utc="07:00",
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
            now=datetime(2026, 5, 25, 7, 1, tzinfo=timezone.utc),
        )

        queue = runtime_state["history_collector_commands"]
        self.assertEqual([cmd["collector_mode"] for cmd in queue], ["daily"])
        self.assertEqual(queue[0]["start_date"], "2026-05-22")
        self.assertEqual(queue[0]["end_date"], "2026-05-22")

    def test_overnight_backlog_without_wide_flag_stays_on_latest_day(self) -> None:
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
        self.assertEqual(queue[0]["start_date"], "2026-05-20")
        self.assertEqual(queue[0]["end_date"], "2026-05-20")

    def test_overnight_backlog_uses_configured_start_date_only_with_wide_flag(self) -> None:
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
        previous = os.environ.get("TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR")
        os.environ["TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR"] = "1"
        try:
            enqueue_overnight_collector_if_due(
                runtime_state,
                args,
                now=datetime(2026, 5, 21, 7, 1, tzinfo=timezone.utc),
            )
        finally:
            if previous is None:
                os.environ.pop("TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR", None)
            else:
                os.environ["TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR"] = previous

        queue = runtime_state["history_collector_commands"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["collector_mode"], "backlog")
        self.assertEqual(queue[0]["start_date"], "2026-02-01")
        self.assertEqual(queue[0]["end_date"], "2026-05-20")

    def test_overnight_backlog_does_not_start_from_year_start_without_wide_flag(self) -> None:
        runtime_state = {}
        args = SimpleNamespace(
            enable_overnight_automation=True,
            overnight_collector_times_utc="07:00",
            overnight_backlog_collector_times_utc="07:00",
            overnight_prioritize_previous_day=False,
            market_close_utc="20:00",
            overnight_collector_start_date="2026-01-01",
            overnight_backlog_lookback_days=0,
            overnight_daily_collector_max_tasks=3000,
            overnight_collector_max_tasks=3000,
            overnight_collector_max_attempts=5,
            history_collector_client_id=168,
            overnight_collector_retry_failed=False,
        )
        previous = os.environ.get("TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR")
        os.environ.pop("TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR", None)
        try:
            enqueue_overnight_collector_if_due(
                runtime_state,
                args,
                now=datetime(2026, 5, 21, 7, 1, tzinfo=timezone.utc),
            )
        finally:
            if previous is not None:
                os.environ["TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR"] = previous

        queue = runtime_state["history_collector_commands"]
        self.assertEqual(len(queue), 1)
        self.assertNotEqual(queue[0]["start_date"], "2026-01-01")
        self.assertEqual(queue[0]["start_date"], "2026-05-20")

    def test_startup_history_repair_queues_when_previous_day_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            history_parquet_path(history_dir, "AAA", datetime(2026, 5, 27, tzinfo=timezone.utc).date()).parent.mkdir(parents=True, exist_ok=True)
            history_parquet_path(history_dir, "AAA", datetime(2026, 5, 27, tzinfo=timezone.utc).date()).write_bytes(b"x")
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=100.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=1,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )

            result = enqueue_startup_history_repair_if_needed(
                runtime_state,
                args,
                now=datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(result["queued"])
            queue = runtime_state["history_collector_commands"]
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["collector_mode"], "startup_repair")
            self.assertEqual(queue[0]["start_date"], "2026-05-27")
            self.assertEqual(queue[0]["end_date"], "2026-05-27")
            self.assertTrue(queue[0]["force"])

    def test_startup_history_repair_defers_near_market_open(self) -> None:
        runtime_state = {}
        args = SimpleNamespace(
            startup_history_repair=True,
            market_open_utc="13:30",
            market_close_utc="20:00",
        )

        result = enqueue_startup_history_repair_if_needed(
            runtime_state,
            args,
            now=datetime(2026, 6, 24, 13, 14, tzinfo=timezone.utc),
        )

        self.assertFalse(result["queued"])
        self.assertEqual(result["reason"], "market_session_active_or_near_open")
        self.assertNotIn("history_collector_commands", runtime_state)

    def test_startup_history_repair_defers_when_eod_active(self) -> None:
        runtime_state = {"eod_active": True}
        args = SimpleNamespace(
            startup_history_repair=True,
            market_open_utc="13:30",
            market_close_utc="20:00",
        )

        result = enqueue_startup_history_repair_if_needed(
            runtime_state,
            args,
            now=datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(result["queued"])
        self.assertEqual(result["reason"], "eod_reconciliation_active")
        self.assertNotIn("history_collector_commands", runtime_state)

    def test_startup_history_repair_defaults_to_latest_completed_session_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=100.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=5,
                startup_history_repair_lookback_days=1,
                market_close_utc="20:00",
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )

            result = enqueue_startup_history_repair_if_needed(
                runtime_state,
                args,
                now=datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(result["queued"])
            queue = runtime_state["history_collector_commands"]
            self.assertEqual(queue[0]["collector_mode"], "startup_repair")
            self.assertEqual(queue[0]["start_date"], "2026-06-23")
            self.assertEqual(queue[0]["end_date"], "2026-06-23")
            self.assertEqual(queue[0]["id"], "startup_history_repair_20260623")
            self.assertEqual(result["repair_range_sessions"], ["2026-06-23"])

    def test_startup_history_repair_wide_range_requires_env_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=100.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=5,
                startup_history_repair_lookback_days=1,
                market_close_utc="20:00",
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )
            previous = os.environ.get("TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR")
            os.environ["TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR"] = "1"
            try:
                result = enqueue_startup_history_repair_if_needed(
                    runtime_state,
                    args,
                    now=datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc),
                )
            finally:
                if previous is None:
                    os.environ.pop("TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR", None)
                else:
                    os.environ["TRADING_BOT_ALLOW_WIDE_HISTORY_REPAIR"] = previous

            self.assertTrue(result["queued"])
            queue = runtime_state["history_collector_commands"]
            self.assertEqual(queue[0]["end_date"], "2026-06-23")
            self.assertEqual(len(result["repair_range_sessions"]), 5)
            self.assertEqual(queue[0]["start_date"], result["repair_range_sessions"][0])
            self.assertRegex(queue[0]["id"], r"startup_history_repair_\d{8}_\d{8}")

    def test_startup_history_repair_skips_when_status_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            status = {
                history_task_key("AAA", datetime(2026, 5, 27, tzinfo=timezone.utc).date()): {"status": "complete"},
                history_task_key("BBB", datetime(2026, 5, 27, tzinfo=timezone.utc).date()): {"status": "no_data"},
            }
            (root / "history").mkdir(parents=True, exist_ok=True)
            (root / "history" / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=100.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=1,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )

            result = enqueue_startup_history_repair_if_needed(
                runtime_state,
                args,
                now=datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(result["queued"])
            self.assertEqual(result["reason"], "readiness_ok")
            self.assertNotIn("history_collector_commands", runtime_state)

    def test_startup_history_repair_no_data_counts_as_ready_without_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            session = datetime(2026, 5, 27, tzinfo=timezone.utc).date()
            history_parquet_path(history_dir, "AAA", session).parent.mkdir(parents=True, exist_ok=True)
            history_parquet_path(history_dir, "AAA", session).write_bytes(b"x")
            status = {
                history_task_key("AAA", session): {"status": "complete"},
                history_task_key("BBB", session): {"status": "no_data"},
            }
            (root / "history").mkdir(parents=True, exist_ok=True)
            (root / "history" / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=100.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=1,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )

            result = enqueue_startup_history_repair_if_needed(
                runtime_state,
                args,
                now=datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(result["queued"])
            self.assertEqual(result["reason"], "readiness_ok")
            self.assertEqual(result["effective_completion_pct"], 100.0)
            self.assertEqual(result["parquet_completion_pct"], 50.0)
            self.assertNotIn("history_collector_commands", runtime_state)

    def test_startup_history_repair_skips_acceptable_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\nCCC\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            session = datetime(2026, 5, 27, tzinfo=timezone.utc).date()
            status = {
                history_task_key("AAA", session): {"status": "complete"},
                history_task_key("BBB", session): {"status": "no_data"},
                history_task_key("CCC", session): {"status": "partial"},
            }
            (root / "history").mkdir(parents=True, exist_ok=True)
            (root / "history" / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=60.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=1,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )

            result = enqueue_startup_history_repair_if_needed(
                runtime_state,
                args,
                now=datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(result["queued"])
            self.assertEqual(result["reason"], "acceptable_partial")
            self.assertEqual(result["readiness_status"], "PARTIAL")
            self.assertNotIn("history_collector_commands", runtime_state)

    def test_startup_history_repair_default_operational_threshold_skips_partial_above_95(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols = [f"S{i:03d}" for i in range(100)]
            universe = root / "universe.csv"
            universe.write_text("symbol\n" + "\n".join(symbols) + "\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            session = datetime(2026, 6, 25, tzinfo=timezone.utc).date()
            status = {}
            for symbol in symbols[:94]:
                status[history_task_key(symbol, session)] = {"status": "complete"}
            for symbol in symbols[94:98]:
                status[history_task_key(symbol, session)] = {"status": "no_data"}
            for symbol in symbols[98:]:
                status[history_task_key(symbol, session)] = {"status": "partial"}
            (root / "history").mkdir(parents=True, exist_ok=True)
            (root / "history" / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")
            runtime_state = {}
            args = SimpleNamespace(
                startup_history_repair=True,
                startup_history_repair_min_completion_pct=95.0,
                startup_history_repair_retry_failed=True,
                startup_history_repair_max_tasks=3000,
                startup_history_repair_lookback_sessions=1,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                overnight_collector_max_attempts=5,
                history_collector_client_id=168,
            )

            result = enqueue_startup_history_repair_if_needed(
                runtime_state,
                args,
                now=datetime(2026, 6, 26, 8, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(result["queued"])
            self.assertEqual(result["reason"], "acceptable_partial")
            self.assertEqual(result["effective_completion_pct"], 98.0)
            self.assertEqual(result["partial_symbols"], 2)
            self.assertNotIn("history_collector_commands", runtime_state)

    def test_daily_top100_build_waits_for_latest_history_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            history_parquet_path(history_dir, "AAA", datetime(2026, 5, 27, tzinfo=timezone.utc).date()).parent.mkdir(parents=True, exist_ok=True)
            history_parquet_path(history_dir, "AAA", datetime(2026, 5, 27, tzinfo=timezone.utc).date()).write_bytes(b"x")
            runtime_state = {}
            args = SimpleNamespace(
                enable_overnight_automation=True,
                daily_top100_build_utc="12:45",
                market_close_utc="20:00",
                startup_history_repair_min_completion_pct=100.0,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                daily_top100_output_dir=str(root / "universe_out"),
                daily_top100_latest_output=str(root / "daily_top100_latest.csv"),
                daily_top100_top_n=100,
                daily_top100_sqlite_path=str(root / "rankings.sqlite"),
            )

            process_daily_top100_build(
                runtime_state,
                args,
                now=datetime(2026, 5, 28, 12, 46, tzinfo=timezone.utc),
            )

            self.assertIsNone(runtime_state.get("daily_top100_process"))
            self.assertEqual(runtime_state.get("daily_top100_build_run_keys"), set())
            self.assertIn("daily_top100_build_wait_logged_keys", runtime_state)

    def test_daily_top100_auto_build_queues_when_acceptable_partial_and_dated_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols = [f"S{i:03d}" for i in range(100)]
            universe = root / "universe.csv"
            universe.write_text("symbol\n" + "\n".join(symbols) + "\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            session = datetime(2026, 6, 25, tzinfo=timezone.utc).date()
            status = {}
            for symbol in symbols[:94]:
                status[history_task_key(symbol, session)] = {"status": "complete"}
            for symbol in symbols[94:98]:
                status[history_task_key(symbol, session)] = {"status": "no_data"}
            for symbol in symbols[98:]:
                status[history_task_key(symbol, session)] = {"status": "partial"}
            (root / "history").mkdir(parents=True, exist_ok=True)
            (root / "history" / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")

            class Proc:
                def poll(self):
                    return None

            runtime_state = {}
            args = SimpleNamespace(
                enable_overnight_automation=True,
                daily_top100_build_utc="12:45",
                market_close_utc="20:00",
                startup_history_repair_min_completion_pct=95.0,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                daily_top100_output_dir=str(root / "universe_out"),
                daily_top100_latest_output=str(root / "daily_top100_latest.csv"),
                daily_top100_top_n=100,
                daily_top100_sqlite_path=str(root / "rankings.sqlite"),
            )

            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.subprocess.Popen", return_value=Proc()) as popen:
                process_daily_top100_build(
                    runtime_state,
                    args,
                    now=datetime(2026, 6, 26, 8, 0, tzinfo=timezone.utc),
                )

            self.assertTrue(popen.called)
            self.assertIsNotNone(runtime_state.get("daily_top100_process"))
            command = runtime_state["daily_top100_running_command"]
            self.assertEqual(command["ranking_date"], "2026-06-25")
            self.assertTrue(any("daily_top100_2026-06-25.csv" in str(part) for part in command["command"]))
            self.assertIn("2026-06-25_auto_missing_dated", runtime_state["daily_top100_build_run_keys"])

    def test_daily_top100_auto_build_skips_when_dated_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\n", encoding="utf-8")
            output_dir = root / "universe_out"
            output_dir.mkdir(parents=True)
            (output_dir / "daily_top100_2026-06-25.csv").write_text("rank,symbol,score\n1,AAA,1\n", encoding="utf-8")
            runtime_state = {}
            args = SimpleNamespace(
                enable_overnight_automation=True,
                daily_top100_build_utc="12:45",
                market_close_utc="20:00",
                startup_history_repair_min_completion_pct=95.0,
                daily_top100_history_dir=str(root / "history" / "universe_1m"),
                daily_top100_universe=str(universe),
                daily_top100_output_dir=str(output_dir),
                daily_top100_latest_output=str(root / "daily_top100_latest.csv"),
                daily_top100_top_n=100,
                daily_top100_sqlite_path=str(root / "rankings.sqlite"),
            )

            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.subprocess.Popen") as popen:
                process_daily_top100_build(
                    runtime_state,
                    args,
                    now=datetime(2026, 6, 26, 8, 0, tzinfo=timezone.utc),
                )

            self.assertFalse(popen.called)
            self.assertIsNone(runtime_state.get("daily_top100_process"))
            self.assertIn("2026-06-25_dated_exists", runtime_state["daily_top100_auto_build_skip_logged_keys"])

    def test_daily_top100_auto_build_does_not_queue_when_history_not_acceptable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
            history_dir = root / "history" / "universe_1m"
            history_parquet_path(history_dir, "AAA", datetime(2026, 6, 25, tzinfo=timezone.utc).date()).parent.mkdir(parents=True, exist_ok=True)
            history_parquet_path(history_dir, "AAA", datetime(2026, 6, 25, tzinfo=timezone.utc).date()).write_bytes(b"x")
            runtime_state = {}
            args = SimpleNamespace(
                enable_overnight_automation=True,
                daily_top100_build_utc="12:45",
                market_close_utc="20:00",
                startup_history_repair_min_completion_pct=95.0,
                daily_top100_history_dir=str(history_dir),
                daily_top100_universe=str(universe),
                daily_top100_output_dir=str(root / "universe_out"),
                daily_top100_latest_output=str(root / "daily_top100_latest.csv"),
                daily_top100_top_n=100,
                daily_top100_sqlite_path=str(root / "rankings.sqlite"),
            )

            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.subprocess.Popen") as popen:
                process_daily_top100_build(
                    runtime_state,
                    args,
                    now=datetime(2026, 6, 26, 8, 0, tzinfo=timezone.utc),
                )

            self.assertFalse(popen.called)
            self.assertIsNone(runtime_state.get("daily_top100_process"))
            self.assertEqual(runtime_state.get("daily_top100_build_run_keys"), set())

    def test_stale_top100_blocks_entries_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            latest.write_text("rank,symbol,score\n1,OLD,1\n", encoding="utf-8")
            runtime_state = {"entries_blocked_reason": ""}
            args = SimpleNamespace(
                daily_top100_latest_output=str(latest),
                daily_top100_output_dir=str(root),
                daily_top100_top_n=1,
                allow_stale_top100=False,
            )

            state = apply_top100_freshness_gate(
                runtime_state,
                args,
                datetime(2026, 5, 27, tzinfo=timezone.utc).date(),
            )

            self.assertFalse(state["ready"])
            self.assertTrue(runtime_state["top100_entries_blocked"])
            self.assertEqual(runtime_state["entries_blocked_reason"], "stale_top100")


if __name__ == "__main__":
    unittest.main()
