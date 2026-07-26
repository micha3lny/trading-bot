from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    SymbolState,
    compute_live_safe_features,
    entry_delay_active_for_time,
    record_completed_live_state_bar,
    reset_session_candle_state,
    reset_symbol_rth_state_preserve_premarket,
    runtime_session_timing,
    session_phase_for_time,
    update_state,
)


class LiveFeatureSessionAnchorTests(unittest.TestCase):
    def test_runtime_session_timing_uses_new_york_dst_and_entry_delay(self) -> None:
        args = argparse.Namespace(entry_delay_after_open_minutes=5.0, premarket_collection_minutes=30.0, market_open_utc="13:30")

        july = runtime_session_timing(args, datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(july.market_open, datetime(2026, 7, 13, 13, 30, tzinfo=timezone.utc))
        self.assertEqual(july.premarket_start, datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc))
        self.assertEqual(july.earliest_entry_time, datetime(2026, 7, 13, 13, 35, tzinfo=timezone.utc))
        self.assertEqual(session_phase_for_time(datetime(2026, 7, 13, 13, 10, tzinfo=timezone.utc), july), "PREMARKET")
        self.assertTrue(entry_delay_active_for_time(datetime(2026, 7, 13, 13, 34, 59, tzinfo=timezone.utc), july))
        self.assertFalse(entry_delay_active_for_time(datetime(2026, 7, 13, 13, 35, tzinfo=timezone.utc), july))

        january = runtime_session_timing(args, datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(january.market_open, datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc))
        self.assertEqual(january.premarket_start, datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc))
        self.assertEqual(january.earliest_entry_time, datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc))

    def test_premarket_ticks_do_not_consume_opening_feature_windows(self) -> None:
        state = SymbolState("AAA")
        args = argparse.Namespace(
            min_first_5m_high_pct=0.5,
            min_first_15m_high_pct=1.0,
            min_or_range_pct=0.5,
            min_price=5.0,
            max_spread_bps=50.0,
        )

        update_state(
            state,
            {"price": 10.0},
            session_elapsed=-1800.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 5, 25, 13, 0, tzinfo=timezone.utc),
            session_phase="PREMARKET",
        )
        self.assertIsNone(state.open_price)
        self.assertIsNone(state.first_5m_high)
        self.assertEqual(state.premarket_candle_count, 1)
        self.assertEqual(len(state.bars), 1)
        self.assertEqual(state.bars[0]["session_phase"], "PREMARKET")

        reset_symbol_rth_state_preserve_premarket(state, session_date="2026-05-25")
        self.assertEqual(state.premarket_candle_count, 1)
        self.assertIsNone(state.open_price)

        update_state(
            state,
            {"price": 10.0},
            session_elapsed=0.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 5, 25, 13, 30, tzinfo=timezone.utc),
            session_phase="RTH",
        )
        update_state(
            state,
            {"price": 10.8},
            session_elapsed=240.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 5, 25, 13, 34, tzinfo=timezone.utc),
            session_phase="RTH",
        )

        features = compute_live_safe_features(state, {"price": 10.8}, args)
        self.assertAlmostEqual(features["first_5m_high_pct"], 8.0)
        self.assertAlmostEqual(features["first_15m_high_pct"], 8.0)
        self.assertGreater(features["or_range_pct"], 0.0)
        self.assertEqual(features["premarket_data_quality"], "OK")

    def test_completed_live_bars_are_recorded_with_session_phase(self) -> None:
        state = SymbolState("AAA")
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(output_dir=tmp, session_date="2026-07-13")
            self.assertIsNone(
                update_state(
                    state,
                    {"price": 10.0, "volume": 100},
                    session_elapsed=-1800.0,
                    opening_range_seconds=15 * 60,
                    observed_at=datetime(2026, 7, 13, 13, 0, 1, tzinfo=timezone.utc),
                    session_phase="PREMARKET",
                )
            )
            completed = update_state(
                state,
                {"price": 10.5, "volume": 200},
                session_elapsed=-1740.0,
                opening_range_seconds=15 * 60,
                observed_at=datetime(2026, 7, 13, 13, 1, 1, tzinfo=timezone.utc),
                session_phase="PREMARKET",
            )
            self.assertEqual(record_completed_live_state_bar(recorder, "AAA", completed), 1)

            completed = update_state(
                state,
                {"price": 11.0, "volume": 300},
                session_elapsed=0.0,
                opening_range_seconds=15 * 60,
                observed_at=datetime(2026, 7, 13, 13, 30, 1, tzinfo=timezone.utc),
                session_phase="RTH",
            )
            self.assertEqual(record_completed_live_state_bar(recorder, "AAA", completed), 1)
            completed = update_state(
                state,
                {"price": 11.2, "volume": 400},
                session_elapsed=60.0,
                opening_range_seconds=15 * 60,
                observed_at=datetime(2026, 7, 13, 13, 31, 1, tzinfo=timezone.utc),
                session_phase="RTH",
            )
            self.assertEqual(record_completed_live_state_bar(recorder, "AAA", completed), 1)

            with recorder.path("candles_1m.csv").open(newline="", encoding="utf-8") as f:
                candle_rows = list(csv.DictReader(f))
            with recorder.path("premarket_1m.csv").open(newline="", encoding="utf-8") as f:
                premarket_rows = list(csv.DictReader(f))

            self.assertEqual([row["session_phase"] for row in candle_rows], ["PREMARKET", "PREMARKET", "RTH"])
            self.assertEqual(len(premarket_rows), 2)

    def test_duplicate_minute_snapshots_update_compact_bar_without_changing_features(self) -> None:
        state = SymbolState("AAA")
        runtime_state: dict = {}
        args = argparse.Namespace(
            min_first_5m_high_pct=0.5,
            min_first_15m_high_pct=1.0,
            min_or_range_pct=0.5,
            min_price=5.0,
            max_spread_bps=50.0,
        )
        update_state(
            state,
            {"price": 10.0},
            session_elapsed=0.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 7, 13, 13, 30, 1, tzinfo=timezone.utc),
            runtime_state=runtime_state,
        )
        update_state(
            state,
            {"price": 10.8},
            session_elapsed=20.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 7, 13, 13, 30, 20, tzinfo=timezone.utc),
            runtime_state=runtime_state,
        )
        self.assertEqual(len(state.bars), 1)
        self.assertEqual(runtime_state["symbol_state_duplicate_bar_suppressed_total"], 1)
        self.assertAlmostEqual(state.bars[0]["high"], 10.8)
        features = compute_live_safe_features(state, {"price": 10.8}, args)
        self.assertAlmostEqual(features["first_5m_high_pct"], 8.0)
        self.assertAlmostEqual(features["first_15m_high_pct"], 8.0)

    def test_symbol_bars_are_bounded_without_changing_feature_scalars(self) -> None:
        state = SymbolState("AAA")
        runtime_state: dict = {}
        start = datetime(2026, 7, 13, 13, 30, tzinfo=timezone.utc)
        for minute in range(65):
            update_state(
                state,
                {"price": 10.0 + minute},
                session_elapsed=float(minute * 60),
                opening_range_seconds=15 * 60,
                observed_at=start + timedelta(minutes=minute),
                runtime_state=runtime_state,
            )
        self.assertEqual(len(state.bars), 60)
        self.assertEqual(runtime_state["symbol_state_bars_trimmed_total"], 5)
        self.assertAlmostEqual(state.first_5m_high, 14.0)

    def test_session_reset_clears_candle_and_signal_state(self) -> None:
        state = SymbolState("AAA", signal_sent=True)
        runtime_state: dict = {}
        update_state(
            state,
            {"price": 10.0},
            session_elapsed=0.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 7, 13, 13, 30, tzinfo=timezone.utc),
            runtime_state=runtime_state,
        )
        reset_session_candle_state(
            {"AAA": state},
            runtime_state,
            previous_session_date="2026-07-13",
            current_session_date="2026-07-14",
        )
        self.assertEqual(state.bars, [])
        self.assertIsNone(state.first_price)
        self.assertIsNone(state.first_5m_high)
        self.assertFalse(state.signal_sent)
        self.assertEqual(runtime_state["symbol_state_session_reset_count"], 1)
        self.assertEqual(runtime_state["symbol_state_bars_cleared_total"], 1)


if __name__ == "__main__":
    unittest.main()
