from __future__ import annotations

import argparse
import json
import unittest

from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    SymbolState,
    runtime_state_growth_current_metrics,
    runtime_state_growth_delta_json,
    runtime_state_growth_numeric_baseline,
)


class FakeEvent:
    def __init__(self, count: int) -> None:
        self.handlers = [object() for _ in range(count)]


class FakeIB:
    pendingTickersEvent = FakeEvent(1)
    orderStatusEvent = FakeEvent(1)
    execDetailsEvent = FakeEvent(1)
    commissionReportEvent = FakeEvent(1)
    errorEvent = FakeEvent(1)
    disconnectedEvent = FakeEvent(1)

    def isConnected(self) -> bool:
        return False


class RuntimeStateGrowthDiagnosticsTests(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(max_market_data_subscriptions=100)

    def base_runtime_state(self) -> dict:
        return {
            "entry_symbols": {"AAA", "BBB"},
            "top100_reload_symbols": ["AAA", "BBB"],
            "reqMktData_total_count": 2,
            "cancelMktData_total_count": 0,
            "symbols_added_since_start": {"AAA", "BBB"},
            "symbols_removed_since_start": set(),
            "entry_order_by_order_id": {},
            "entry_rejection_processed": set(),
            "fill_diagnostic_execution_ids": set(),
            "rate_limited_log_state": {},
            "sqlite_writer_status": {"ack_timeouts_total": 0},
            "process_start_monotonic": 0.0,
        }

    def test_two_sessions_without_restart_detects_old_symbol_state(self) -> None:
        states = {
            "AAA": SymbolState(symbol="AAA", first_seen_utc="2026-07-09T13:30:00+00:00", first_price=10.0, first_5m_high=10.5, first_15m_high=11.0),
            "BBB": SymbolState(symbol="BBB", first_seen_utc="2026-07-10T13:30:00+00:00", first_price=20.0),
        }
        metrics = runtime_state_growth_current_metrics(
            ib=None,
            runtime_state=self.base_runtime_state(),
            states=states,
            contracts=[],
            contract_by_symbol={},
            tickers={},
            managed_positions={},
            seen_fills=set(),
            latest_snapshots={},
            args=self.args(),
            current_session_date="2026-07-10",
            previous_session_date="2026-07-09",
        )
        self.assertEqual(metrics["state_with_old_first_seen_count"], 1)
        self.assertEqual(metrics["state_with_first5_from_previous_session_count"], 1)
        self.assertEqual(metrics["state_with_first15_from_previous_session_count"], 1)

    def test_top100_change_detects_removed_subscribed_and_added_missing_state(self) -> None:
        runtime_state = self.base_runtime_state()
        runtime_state["entry_symbols"] = {"BBB", "CCC"}
        states = {"AAA": SymbolState(symbol="AAA"), "BBB": SymbolState(symbol="BBB")}
        metrics = runtime_state_growth_current_metrics(
            ib=None,
            runtime_state=runtime_state,
            states=states,
            contracts=[("AAA", object()), ("BBB", object())],
            contract_by_symbol={"AAA": object(), "BBB": object()},
            tickers={"AAA": object(), "BBB": object()},
            managed_positions={},
            seen_fills=set(),
            latest_snapshots={},
            args=self.args(),
            current_session_date="2026-07-10",
        )
        self.assertEqual(metrics["symbols_removed_from_top100_but_still_subscribed"], 1)
        self.assertEqual(metrics["symbols_in_top100_without_subscription"], 1)
        self.assertEqual(metrics["symbols_in_top100_without_ticker"], 1)
        self.assertEqual(metrics["symbols_in_top100_without_state"], 1)

    def test_signal_sent_pending_maps_and_callbacks_are_counted(self) -> None:
        runtime_state = self.base_runtime_state()
        runtime_state["entry_order_by_order_id"] = {"123": {"symbol": "AAA"}}
        states = {"AAA": SymbolState(symbol="AAA", signal_sent=True)}
        managed = {
            "AAA": ManagedPosition(symbol="AAA", contract=object(), quantity=1, entry_price=10.0, entry_time="", peak_price=10.0, active=True, exit_sent=True, exit_order_id=456)
        }
        metrics = runtime_state_growth_current_metrics(
            ib=FakeIB(),
            runtime_state=runtime_state,
            states=states,
            contracts=[("AAA", object())],
            contract_by_symbol={"AAA": object()},
            tickers={"AAA": object()},
            managed_positions=managed,
            seen_fills={"exec1", "exec2"},
            latest_snapshots={"AAA": {"price": 10.0}},
            args=self.args(),
            current_session_date="2026-07-10",
        )
        self.assertEqual(metrics["signal_sent_count"], 1)
        self.assertEqual(metrics["runtime_entry_order_map_count"], 1)
        self.assertEqual(metrics["runtime_exit_order_map_count"], 1)
        self.assertEqual(metrics["pending_exit_orders_count"], 1)
        self.assertEqual(metrics["executions_seen_cache_count"], 2)
        self.assertEqual(metrics["callback_count_execDetailsEvent"], 1)

    def test_state_growth_snapshot_deltas_are_reported(self) -> None:
        metrics = {"total_state_count": 5, "sqlite_ack_timeouts_total": 3, "rss_memory_mb": 100.0}
        startup = runtime_state_growth_numeric_baseline({"total_state_count": 2, "sqlite_ack_timeouts_total": 1, "rss_memory_mb": 80.0})
        session = runtime_state_growth_numeric_baseline({"total_state_count": 4, "sqlite_ack_timeouts_total": 2, "rss_memory_mb": 90.0})
        parsed = json.loads(runtime_state_growth_delta_json(metrics, startup, session))
        self.assertEqual(parsed["total_state_count"]["current"], 5)
        self.assertEqual(parsed["total_state_count"]["delta_startup"], 3.0)
        self.assertEqual(parsed["total_state_count"]["delta_session"], 1.0)
        self.assertEqual(parsed["sqlite_ack_timeouts_total"]["delta_session"], 1.0)


if __name__ == "__main__":
    unittest.main()
