from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.stop_loss_strategy_analyzer import (
    data_quality,
    simulate_stop,
    build_segment_analysis,
    add_segment_buckets,
    STOP_LOSS_PCTS,
)


class StopLossStrategyAnalyzerTests(unittest.TestCase):
    def test_simulate_stop_saved_loser_after_activation_delay(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": "2026-07-17T13:30:00+00:00", "open": 10.0, "high": 10.1, "low": 9.8, "close": 9.9},
            {"timestamp": "2026-07-17T13:31:00+00:00", "open": 9.9, "high": 10.0, "low": 9.3, "close": 9.4},
            {"timestamp": "2026-07-17T13:32:00+00:00", "open": 9.4, "high": 9.5, "low": 9.1, "close": 9.2},
        ])
        row = {"trade_id": "T1", "symbol": "AAA", "entry_price": 10.0, "exit_price": 9.2, "quantity": 10, "net_pnl": -8.0, "entry_time": "2026-07-17T13:30:00+00:00", "exit_time": "2026-07-17T13:32:00+00:00"}
        result = simulate_stop(row, candles, stop_pct=5.0, activation_delay_min=0, slippage_bps=0)
        self.assertEqual(result["stop_hit"], 1)
        self.assertEqual(result["stop_outcome"], "saved_loser")
        delayed = simulate_stop(row, candles, stop_pct=5.0, activation_delay_min=5, slippage_bps=0)
        self.assertEqual(delayed["stop_hit"], 0)

    def test_false_stop_when_trade_later_recovers(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": "2026-07-17T13:30:00+00:00", "open": 10.0, "high": 10.1, "low": 9.8, "close": 9.9},
            {"timestamp": "2026-07-17T13:31:00+00:00", "open": 9.9, "high": 10.8, "low": 9.4, "close": 10.7},
        ])
        row = {"trade_id": "T2", "symbol": "BBB", "entry_price": 10.0, "exit_price": 10.7, "quantity": 10, "net_pnl": 7.0, "entry_time": "2026-07-17T13:30:00+00:00", "exit_time": "2026-07-17T13:31:00+00:00"}
        result = simulate_stop(row, candles, stop_pct=5.0, activation_delay_min=0, slippage_bps=0)
        self.assertEqual(result["stop_hit"], 1)
        self.assertEqual(result["stop_outcome"], "false_stop")
        self.assertGreater(result["later_mfe_pct"], 0)

    def test_premarket_coverage_unavailable_when_missing(self) -> None:
        paths = pd.DataFrame([
            {"trade_id": "T1", "symbol": "AAA", "stop_pct": STOP_LOSS_PCTS[0], "actual_net_pnl": 1.0, "simulated_net_pnl": 0.5, "premarket_range_pct": None},
            {"trade_id": "T2", "symbol": "BBB", "stop_pct": STOP_LOSS_PCTS[0], "actual_net_pnl": -1.0, "simulated_net_pnl": -0.5, "premarket_range_pct": None},
        ])
        quality = data_quality(paths)
        self.assertEqual(quality["premarket_feature_coverage"], "unavailable_for_session")
        segments = build_segment_analysis(paths)
        premarket = segments[segments["segment_feature"] == "premarket_range_pct"]
        self.assertTrue(premarket.empty)

    def test_segment_buckets_use_dynamic_premarket_when_available(self) -> None:
        paths = pd.DataFrame([
            {"trade_id": "T1", "symbol": "AAA", "stop_loss_pct": 1.0, "actual_net_pnl": 1.0, "simulated_net_pnl": 0.5, "premarket_range_pct": 2.0},
            {"trade_id": "T2", "symbol": "BBB", "stop_loss_pct": 1.0, "actual_net_pnl": -1.0, "simulated_net_pnl": -0.5, "premarket_range_pct": 8.0},
        ])
        bucketed = add_segment_buckets(paths)
        self.assertIn("premarket_range_pct_bucket", bucketed.columns)
        self.assertTrue(bucketed["premarket_range_pct_bucket"].notna().all())
        segments = build_segment_analysis(paths)
        self.assertIn("premarket_range_pct", set(segments["segment_feature"].tolist()))


if __name__ == "__main__":
    unittest.main()
