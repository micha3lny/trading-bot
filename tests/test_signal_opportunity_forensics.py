from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.signal_opportunity_forensics import (
    build_parity_rows,
    classify_case,
    simulate_v67_exit,
    write_csv,
)
from src.live_trading.analysis.common import live_signal_replay


class SignalOpportunityForensicsTests(unittest.TestCase):
    def test_bar_start_does_not_use_1345_unfinished_bar(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-07-20T13:30:00Z"), "open": 10.0, "high": 10.8, "low": 9.9, "close": 10.5},
            {"timestamp": pd.Timestamp("2026-07-20T13:34:00Z"), "open": 10.5, "high": 10.9, "low": 10.1, "close": 10.8},
            {"timestamp": pd.Timestamp("2026-07-20T13:44:00Z"), "open": 10.8, "high": 11.0, "low": 10.3, "close": 10.9},
            {"timestamp": pd.Timestamp("2026-07-20T13:45:00Z"), "open": 10.9, "high": 12.5, "low": 10.8, "close": 12.0},
        ])
        replay = live_signal_replay(candles, bar_timestamp_semantics="bar_start")
        self.assertEqual(replay.possible_signal_time, pd.Timestamp("2026-07-20T13:45:00Z"))
        self.assertEqual(replay.candle_timestamp, pd.Timestamp("2026-07-20T13:44:00Z"))
        self.assertEqual(replay.current_price, 10.9)

    def test_candle_high_breakout_is_not_live_gate(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-07-20T13:30:00Z"), "open": 10.0, "high": 10.8, "low": 9.9, "close": 10.5},
            {"timestamp": pd.Timestamp("2026-07-20T13:34:00Z"), "open": 10.5, "high": 10.9, "low": 10.1, "close": 10.8},
            {"timestamp": pd.Timestamp("2026-07-20T13:44:00Z"), "open": 10.8, "high": 11.0, "low": 4.8, "close": 4.9},
            {"timestamp": pd.Timestamp("2026-07-20T13:45:00Z"), "open": 4.9, "high": 12.5, "low": 4.8, "close": 4.9},
        ])
        replay = live_signal_replay(candles, bar_timestamp_semantics="bar_start")
        self.assertEqual(replay.did_break_or_high, 1)
        self.assertEqual(replay.breakout_gate_used, 0)
        self.assertIsNone(replay.possible_signal_time)
        self.assertEqual(replay.reason, "price_too_low")

    def test_parity_rows_mark_legacy_high_breakout_divergence(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-07-20T13:30:00Z"), "open": 10.0, "high": 10.8, "low": 9.9, "close": 10.5},
            {"timestamp": pd.Timestamp("2026-07-20T13:34:00Z"), "open": 10.5, "high": 10.9, "low": 10.1, "close": 10.8},
            {"timestamp": pd.Timestamp("2026-07-20T13:44:00Z"), "open": 10.8, "high": 11.0, "low": 10.3, "close": 10.9},
            {"timestamp": pd.Timestamp("2026-07-20T13:45:00Z"), "open": 10.9, "high": 12.5, "low": 10.8, "close": 10.7},
        ])
        rows = build_parity_rows(session_date="2026-07-20", symbol="TEST", candles=candles, center=pd.Timestamp("2026-07-20T13:45:00Z"))
        at_1345 = [row for row in rows if row["timestamp"] == "2026-07-20T13:45:00+00:00"][0]
        self.assertEqual(at_1345["live_equivalent_decision"], 1)
        self.assertEqual(at_1345["offline_decision"], 1)
        self.assertEqual(at_1345["first_divergence"], "")

    def test_v67_exit_simulation_trailing_stop(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-07-20T13:45:00Z"), "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0},
            {"timestamp": pd.Timestamp("2026-07-20T13:46:00Z"), "open": 10.0, "high": 10.5, "low": 10.0, "close": 10.4},
            {"timestamp": pd.Timestamp("2026-07-20T13:47:00Z"), "open": 10.4, "high": 10.5, "low": 10.1, "close": 10.2},
        ])
        sim = simulate_v67_exit(candles, entry_time=pd.Timestamp("2026-07-20T13:45:00Z"), entry_price=10.0, notional=1000.0, slippage_bps=0.0)
        self.assertEqual(sim.exit_reason, "v46_wide_trail_trailing_stop")
        self.assertEqual(sim.exit_time, pd.Timestamp("2026-07-20T13:48:00Z"))
        self.assertGreater(sim.net_pnl or 0.0, 0.0)

    def test_classify_offline_lookahead_false_positive(self) -> None:
        classification, opportunity = classify_case(
            replay_time=None,
            source_time=pd.Timestamp("2026-07-20T13:45:00Z"),
            mfe_pct=5.0,
            net_pnl=None,
            divergence="",
        )
        self.assertEqual(classification, "OFFLINE_LOOKAHEAD_FALSE_POSITIVE")
        self.assertEqual(opportunity, "")

    def test_write_csv_keeps_declared_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_csv(path, [{"a": 1, "b": 2, "extra": 3}], ["a", "b"])
            self.assertEqual(path.read_text().splitlines()[0], "a,b")


if __name__ == "__main__":
    unittest.main()
