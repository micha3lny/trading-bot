from __future__ import annotations

import argparse
import copy
import unittest
from datetime import datetime, timezone

from src.live_trading.candidate_snapshot_telemetry import CandidateScanCollector, FullSnapshotTracker
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    SymbolState,
    build_pre_signal_runtime_snapshot,
    candidate_feature_state_checkpoint,
    compute_live_safe_features,
    finalize_candidate_scan_telemetry,
    maybe_add_candidate_full_snapshot,
    start_candidate_scan_collector,
    update_candidate_scan_from_pre_signal,
    update_state,
)


class _IB:
    def openTrades(self):
        return []

    def openOrders(self):
        return []


class _Writer:
    def __init__(self) -> None:
        self.expected = []
        self.batches = []

    def note_expected_batch(self, *args, **kwargs):
        self.expected.append((args, kwargs))

    def enqueue(self, batch):
        self.batches.append(batch)
        return True


def args() -> argparse.Namespace:
    return argparse.Namespace(
        min_first_5m_high_pct=4.0,
        min_first_15m_high_pct=6.5,
        min_or_range_pct=5.0,
        min_price=5.0,
        max_spread_bps=50.0,
        max_open_positions=5,
        max_entries_per_cycle=5,
        max_entries_per_minute=5,
        alpha_rank_csv="daily_top100.csv",
    )


class CandidateSnapshotRuntimeIntegrationTests(unittest.TestCase):
    def test_completed_bar_full_uses_state_before_first_tick_of_next_bar(self) -> None:
        state = SymbolState("AAA")
        state.first_price = 9.0
        state.open_price = 9.0
        state.first_5m_high = 10.0
        state.first_15m_high = 10.0
        state.or_high = 10.0
        state.or_low = 9.0
        completed_state = candidate_feature_state_checkpoint(state)
        state.first_5m_high = 12.0
        state.first_15m_high = 12.0
        state.or_high = 12.0
        collector = CandidateScanCollector(
            session_date="2026-08-05", run_id="run", process_start_id="process",
            scan_id=1, scan_started_at="2026-08-05T13:35:01+00:00", expected_symbols=("AAA",),
        )
        maybe_add_candidate_full_snapshot(
            collector,
            FullSnapshotTracker(),
            symbol="AAA",
            state=state,
            features={"first_5m_complete": 1, "first_15m_complete": 0},
            snap={"volume": 1234},
            completed_bar={
                "bucket_utc": "2026-08-05T13:34:00+00:00", "open": 9.8,
                "high": 10.0, "low": 9.7, "close": 9.9, "volume": 500,
                "samples": 12, "source": "live_ticker_snapshot", "session_phase": "RTH",
            },
            completed_feature_state=completed_state,
            checkpoint_seconds=60.0,
            now_monotonic=100.0,
        )
        row = collector.full_rows[0]
        self.assertEqual(row["candle_high"], 10.0)
        self.assertEqual(row["first_5m_high"], 10.0)
        self.assertEqual(row["or_high"], 10.0)

    def test_expected_watchlist_not_contracts_controls_light_coverage(self) -> None:
        writer = _Writer()
        runtime_state = {
            "top100_pipeline_run_id": "run-1",
            "candidate_snapshot_process_start_id": "process-1",
            "entry_symbols": {"AAA", "BBB", "CCC"},
            "top100_reload_symbols": ["AAA", "BBB", "CCC"],
            "first_tick_received_symbols": set(),
            "ineligible_symbols": set(),
            "top100_entry_metadata_by_symbol": {},
            "contract_source_by_symbol": {"AAA": "qualified"},
        }
        collector = start_candidate_scan_collector(
            writer=writer,
            runtime_state=runtime_state,
            session_date="2026-08-05",
            scan_id=1,
            scan_started_at="2026-08-05T13:30:00+00:00",
            expected_symbols=["AAA", "BBB", "CCC"],
            contract_by_symbol={"AAA": object()},
            tickers={},
            states={"AAA": SymbolState("AAA")},
            managed_positions={},
            ib=_IB(),
            args=args(),
            entries_blocked=False,
            manual_block=False,
            entry_delay_block=False,
            restart_block=False,
            reconnect_block=False,
            disk_block=False,
            top100_block=False,
            observed_at=datetime(2026, 8, 5, 13, 30, tzinfo=timezone.utc),
        )
        finalize_candidate_scan_telemetry(writer, collector, scan_started_monotonic=0.0)
        rows = {row["symbol"]: row for row in writer.batches[0].light_rows}
        self.assertEqual(set(rows), {"AAA", "BBB", "CCC"})
        self.assertEqual(rows["BBB"]["contract_present"], 0)
        self.assertEqual(rows["BBB"]["state_present"], 0)

    def test_observation_does_not_change_scores_order_ready_or_signal_state(self) -> None:
        runtime_state = {
            "top100_pipeline_run_id": "run-1",
            "candidate_snapshot_process_start_id": "process-1",
            "entry_symbols": {"AAA", "BBB"},
            "top100_reload_symbols": ["AAA", "BBB"],
            "first_tick_received_symbols": {"AAA", "BBB"},
            "ineligible_symbols": set(),
            "top100_entry_metadata_by_symbol": {},
            "contract_source_by_symbol": {"AAA": "qualified", "BBB": "qualified"},
        }
        states = {"AAA": SymbolState("AAA"), "BBB": SymbolState("BBB")}
        snapshots = {
            "AAA": {"price": 11.0, "bid": 10.99, "ask": 11.01, "spread_bps": 18.18, "volume": 1000},
            "BBB": {"price": 10.0, "bid": 9.99, "ask": 10.01, "spread_bps": 20.0, "volume": 500},
        }
        observed = datetime(2026, 8, 5, 13, 46, tzinfo=timezone.utc)
        for symbol, state in states.items():
            update_state(state, snapshots[symbol], 16 * 60, 15 * 60, observed_at=observed)
            state.first_price = 9.0
            state.first_5m_high = 10.0
            state.first_15m_high = 10.5
            state.or_high = 10.5
            state.or_low = 9.0
        before_features = {symbol: compute_live_safe_features(state, snapshots[symbol], args()) for symbol, state in states.items()}
        before_order = sorted(before_features, key=lambda symbol: before_features[symbol]["score"], reverse=True)
        before_state = copy.deepcopy(states)

        writer = _Writer()
        collector = start_candidate_scan_collector(
            writer=writer, runtime_state=runtime_state, session_date="2026-08-05", scan_id=1,
            scan_started_at=observed.isoformat(), expected_symbols=["AAA", "BBB"],
            contract_by_symbol={"AAA": object(), "BBB": object()}, tickers={}, states=states,
            managed_positions={}, ib=_IB(), args=args(), entries_blocked=False,
            manual_block=False, entry_delay_block=False, restart_block=False,
            reconnect_block=False, disk_block=False, top100_block=False, observed_at=observed,
        )
        tracker = FullSnapshotTracker()
        for symbol, state in states.items():
            features = before_features[symbol]
            pre = build_pre_signal_runtime_snapshot(
                symbol=symbol, session_date="2026-08-05", scan_id=1, contract=object(),
                ticker_present=True, snap=snapshots[symbol], state=state, features=features,
                runtime_state=runtime_state, ranking_position=before_order.index(symbol) + 1,
                has_active_position=False, entry_symbol_allowed=True, symbol_ineligible=False,
                entries_blocked=False, entries_blocked_reason="", stale_reason="", position_usd=1000,
                now_ts=observed.timestamp(), observed_at=observed,
            )
            update_candidate_scan_from_pre_signal(collector, pre, snap=snapshots[symbol], observed_at=observed)
            maybe_add_candidate_full_snapshot(
                collector, tracker, symbol=symbol, state=state, features=features,
                snap=snapshots[symbol], completed_bar=None, completed_feature_state=None,
                checkpoint_seconds=60,
                now_monotonic=100.0,
            )

        after_features = {symbol: compute_live_safe_features(state, snapshots[symbol], args()) for symbol, state in states.items()}
        after_order = sorted(after_features, key=lambda symbol: after_features[symbol]["score"], reverse=True)
        self.assertEqual(before_order, after_order)
        self.assertEqual(before_features, after_features)
        self.assertEqual(
            {symbol: state.signal_sent for symbol, state in before_state.items()},
            {symbol: state.signal_sent for symbol, state in states.items()},
        )


if __name__ == "__main__":
    unittest.main()
