from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    SymbolState,
    build_entry_feature_snapshot,
    emit_pre_signal_runtime_snapshot,
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


    def test_entry_feature_snapshot_includes_buy_time_required_features(self) -> None:
        snapshot = build_entry_feature_snapshot(
            top100_meta={"top100_rank": 4, "top100_score": 91.2, "top100_source_date": "2026-07-17"},
            signal_payload={
                "spread_bps": 18.5,
                "first_5m_high_pct": 1.1,
                "first_15m_high_pct": 2.2,
                "or_range_pct": 0.8,
                "premarket_range_pct": 7.7,
                "premarket_change_pct": 3.3,
                "premarket_volume": 12345,
                "premarket_vwap": 10.42,
                "distance_from_premarket_high_pct": -1.2,
                "distance_from_premarket_low_pct": 4.4,
                "distance_from_premarket_vwap_pct": 0.9,
                "gap_from_previous_close_pct": 5.5,
            },
            diagnostics={"candidate_age_seconds": 12.3, "signal_time": "2026-07-17T13:45:00+00:00"},
            features={"score": 88.8, "reason": "breakout_ready"},
            live_entry_score=88.8,
            ranking_position=2,
            order_id=12345,
            perm_id=67890,
        )
        self.assertEqual(snapshot["spread_bps_at_entry"], 18.5)
        self.assertEqual(snapshot["top100_rank"], 4)
        self.assertEqual(snapshot["top100_score"], 91.2)
        self.assertEqual(snapshot["live_entry_score"], 88.8)
        self.assertEqual(snapshot["live_entry_rank"], 2)
        self.assertEqual(snapshot["premarket_range_pct"], 7.7)
        self.assertEqual(snapshot["premarket_change_pct"], 3.3)
        self.assertEqual(snapshot["premarket_volume"], 12345)
        self.assertEqual(snapshot["premarket_vwap"], 10.42)
        self.assertEqual(snapshot["distance_from_premarket_high_pct"], -1.2)
        self.assertEqual(snapshot["distance_from_premarket_vwap_pct"], 0.9)
        self.assertEqual(snapshot["gap_from_previous_close_pct"], 5.5)
        self.assertEqual(snapshot["entry_feature_snapshot_source"], "buy_decision_runtime")
        self.assertIn("feature_snapshot_time", snapshot)

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
        runtime_state["pre_signal_snapshots_emitted_total"] = 7
        runtime_state["pre_signal_snapshots_suppressed_total"] = 11
        runtime_state["sqlite_writer_status"] = {"ack_timeouts_total": 0, "write_count_by_method": {"record_runtime_event": 13}}
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
        self.assertEqual(metrics["pre_signal_snapshots_emitted_total"], 7)
        self.assertEqual(metrics["pre_signal_snapshots_suppressed_total"], 11)
        self.assertEqual(metrics["runtime_event_writes_total"], 13)

    def test_state_growth_snapshot_deltas_are_reported(self) -> None:
        metrics = {"total_state_count": 5, "sqlite_ack_timeouts_total": 3, "rss_memory_mb": 100.0}
        startup = runtime_state_growth_numeric_baseline({"total_state_count": 2, "sqlite_ack_timeouts_total": 1, "rss_memory_mb": 80.0})
        session = runtime_state_growth_numeric_baseline({"total_state_count": 4, "sqlite_ack_timeouts_total": 2, "rss_memory_mb": 90.0})
        parsed = json.loads(runtime_state_growth_delta_json(metrics, startup, session))
        self.assertEqual(parsed["total_state_count"]["current"], 5)
        self.assertEqual(parsed["total_state_count"]["delta_startup"], 3.0)
        self.assertEqual(parsed["total_state_count"]["delta_session"], 1.0)
        self.assertEqual(parsed["sqlite_ack_timeouts_total"]["delta_session"], 1.0)

    def snapshot_recorder(self, tmpdir: str):
        class Recorder:
            session_date = "2026-07-13"
            sqlite_store = None

            def __init__(self, root: str) -> None:
                self.root = Path(root)

            def path(self, name: str) -> Path:
                return self.root / name

        return Recorder(tmpdir)

    def base_pre_signal_snapshot(self, **overrides) -> dict:
        snapshot = {
            "symbol": "AAA",
            "session_date": "2026-07-13",
            "scan_id": 1,
            "ranking_position": 1,
            "candidate_age_seconds": None,
            "in_top100": 1,
            "top100_rank": 1,
            "top100_score": 90.0,
            "contract_present": 1,
            "ticker_present": 1,
            "usable_price": 1,
            "current_price": 10.0,
            "bid": 9.99,
            "ask": 10.01,
            "spread_bps": 20.0,
            "state_present": 1,
            "signal_sent": 0,
            "ready": 0,
            "ready_since": "",
            "first_seen": "2026-07-13T13:30:00+00:00",
            "last_live_update": "2026-07-13T13:31:00+00:00",
            "first_price_initialized": 1,
            "first5_initialized": 1,
            "first15_initialized": 1,
            "first_5m_high_pct": 0.1,
            "first_15m_high_pct": 0.2,
            "or_range_pct": 0.3,
            "live_entry_score": 42.0,
            "rejection_reason": "first_5m_high_too_low",
            "entries_blocked": 0,
            "entries_blocked_reason": "",
            "stale_reason": "",
            "already_open": 0,
            "quantity": 100,
            "quantity_reason": "ok",
            "would_emit_signal_ready": 0,
            "signal_ready_reason": "not_ready:first_5m_high_too_low",
            "entry_symbol_allowed": 1,
            "symbol_ineligible": 0,
        }
        snapshot.update(overrides)
        return snapshot

    def lifecycle_rows(self, tmpdir: str) -> list[str]:
        path = Path(tmpdir) / "trade_lifecycle.csv"
        if not path.exists():
            return []
        return [line for line in path.read_text(encoding="utf-8").splitlines() if "PRE_SIGNAL_RUNTIME_SNAPSHOT" in line]

    def test_pre_signal_first_decision_snapshot_is_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_state: dict = {}
            emit_pre_signal_runtime_snapshot(
                self.snapshot_recorder(tmpdir),
                runtime_state,
                self.base_pre_signal_snapshot(),
                now_ts=100.0,
            )
            self.assertEqual(len(self.lifecycle_rows(tmpdir)), 1)
            self.assertEqual(runtime_state["pre_signal_snapshots_emitted_total"], 1)
            self.assertEqual(runtime_state.get("pre_signal_snapshots_suppressed_total", 0), 0)

    def test_pre_signal_unchanged_non_ready_snapshot_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_state: dict = {}
            recorder = self.snapshot_recorder(tmpdir)
            snapshot = self.base_pre_signal_snapshot()
            emit_pre_signal_runtime_snapshot(recorder, runtime_state, snapshot, now_ts=100.0)
            emit_pre_signal_runtime_snapshot(recorder, runtime_state, snapshot, now_ts=120.0)
            self.assertEqual(len(self.lifecycle_rows(tmpdir)), 1)
            self.assertEqual(runtime_state["pre_signal_snapshots_emitted_total"], 1)
            self.assertEqual(runtime_state["pre_signal_snapshots_suppressed_total"], 1)

    def test_pre_signal_changed_reason_emits_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_state: dict = {}
            recorder = self.snapshot_recorder(tmpdir)
            emit_pre_signal_runtime_snapshot(recorder, runtime_state, self.base_pre_signal_snapshot(), now_ts=100.0)
            emit_pre_signal_runtime_snapshot(
                recorder,
                runtime_state,
                self.base_pre_signal_snapshot(signal_ready_reason="not_ready:or_range_too_low"),
                now_ts=120.0,
            )
            self.assertEqual(len(self.lifecycle_rows(tmpdir)), 2)
            self.assertEqual(runtime_state["pre_signal_snapshots_emitted_total"], 2)

    def test_pre_signal_would_emit_ready_is_always_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_state: dict = {}
            recorder = self.snapshot_recorder(tmpdir)
            snapshot = self.base_pre_signal_snapshot(
                ready=1,
                rejection_reason="live_safe_expansion_ready",
                would_emit_signal_ready=1,
                signal_ready_reason="would_emit_signal_ready",
            )
            emit_pre_signal_runtime_snapshot(recorder, runtime_state, snapshot, now_ts=100.0)
            emit_pre_signal_runtime_snapshot(recorder, runtime_state, snapshot, now_ts=101.0)
            self.assertEqual(len(self.lifecycle_rows(tmpdir)), 2)
            self.assertEqual(runtime_state["pre_signal_snapshots_emitted_total"], 2)


if __name__ == "__main__":
    unittest.main()
