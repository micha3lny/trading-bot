from __future__ import annotations

import argparse
import unittest
from datetime import datetime, timezone

from src.live_trading.v67_live_top100_expansion_paper_trader import (
    SymbolState,
    compute_live_safe_features,
    update_state,
)


class LiveFeatureSessionAnchorTests(unittest.TestCase):
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
        )
        self.assertIsNone(state.open_price)
        self.assertIsNone(state.first_5m_high)

        update_state(
            state,
            {"price": 10.0},
            session_elapsed=0.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 5, 25, 13, 30, tzinfo=timezone.utc),
        )
        update_state(
            state,
            {"price": 10.8},
            session_elapsed=240.0,
            opening_range_seconds=15 * 60,
            observed_at=datetime(2026, 5, 25, 13, 34, tzinfo=timezone.utc),
        )

        features = compute_live_safe_features(state, {"price": 10.8}, args)
        self.assertAlmostEqual(features["first_5m_high_pct"], 8.0)
        self.assertAlmostEqual(features["first_15m_high_pct"], 8.0)
        self.assertGreater(features["or_range_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
