from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.multiday_top100_ranking_analyzer import BaselineSettings, analyze_range, compare_baseline, reproduce_baseline, score_variants
from src.live_trading.analysis.top100_analysis_common import session_dates
from src.live_trading.ranking.daily_top100_builder import parquet_path


class MultidayTop100RankingAnalyzerTests(unittest.TestCase):
    def test_date_range_skips_non_trading_days(self) -> None:
        self.assertEqual(
            session_dates(None, "2026-07-02", "2026-07-06"),
            ["2026-07-02", "2026-07-06"],
        )

    def test_explicit_non_trading_date_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a US equity trading session"):
            session_dates("2026-07-04", None, None)

    def fixture(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["AAA", "BBB", "CCC"], "production_score": [90.0, 80.0, 70.0],
            "return_3d": [1.0, 8.0, -2.0], "return_5d": [2.0, 7.0, -3.0], "return_10d": [3.0, 6.0, -4.0],
            "return_20d": [4.0, 5.0, -5.0], "return_60d": [5.0, 4.0, -6.0],
            "volume_acceleration": [1.0, 2.0, 0.5], "drawdown_from_recent_high_pct": [-1.0, -2.0, -10.0],
            "trend_agreement_short_medium_long": [1, 1, 0], "consecutive_days_in_top100": [5, 1, 0],
        })

    def test_baseline_exact_reproduction_check(self) -> None:
        reproduced = pd.DataFrame({"rank": [1, 2], "symbol": ["AAA", "BBB"]})
        exact = compare_baseline(reproduced, pd.DataFrame({"symbol": ["AAA", "BBB"]}))
        mismatch = compare_baseline(reproduced, pd.DataFrame({"symbol": ["BBB", "AAA"]}))
        self.assertTrue(exact["baseline_match"])
        self.assertFalse(mismatch["baseline_match"])
        self.assertEqual(mismatch["rank_mismatch_count"], 2)

    def test_variant_scoring_is_deterministic_and_distinct(self) -> None:
        first = score_variants(self.fixture())
        second = score_variants(self.fixture())
        pd.testing.assert_series_equal(first["rank_hybrid_70_30"], second["rank_hybrid_70_30"])
        self.assertFalse(first["score_production_baseline"].equals(first["score_reversal"]))

    def test_feature_dates_must_precede_trading_session(self) -> None:
        frame = pd.DataFrame({"feature_max_date": ["2026-07-30", "2026-07-31"], "trading_session_date": ["2026-07-31", "2026-07-31"]})
        passed = pd.to_datetime(frame["feature_max_date"]) < pd.to_datetime(frame["trading_session_date"])
        self.assertEqual(passed.tolist(), [True, False])

    def test_identical_inputs_keep_symbol_tie_order_deterministic(self) -> None:
        frame = pd.concat([self.fixture().iloc[[0]], self.fixture().iloc[[0]]], ignore_index=True)
        frame["symbol"] = ["BBB", "AAA"]
        ranked = score_variants(frame).sort_values(["rank_production_baseline", "symbol"])
        self.assertEqual(set(ranked["symbol"]), {"AAA", "BBB"})

    def test_end_to_end_writes_causal_matrix_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            top100_dir = root / "top100"
            output = root / "analysis"
            top100_dir.mkdir()
            universe = root / "universe.csv"
            pd.DataFrame({"symbol": ["AAA", "BBB"]}).to_csv(universe, index=False)
            for day in pd.date_range("2026-07-27", "2026-07-31", freq="D"):
                for offset, symbol in enumerate(("AAA", "BBB")):
                    path = parquet_path(history, symbol, day.date(), "RTH")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    base = 10 + offset + (day.day - 27) * 0.1
                    pd.DataFrame({
                        "timestamp": pd.date_range(f"{day.date()}T13:30:00Z", periods=5, freq="min"),
                        "open": [base] * 5, "high": [base + 0.5] * 5, "low": [base - 0.1] * 5,
                        "close": [base + value * 0.05 for value in range(5)], "volume": [1000] * 5,
                    }).to_parquet(path, index=False)
            settings = BaselineSettings(top_n=2, min_price=1, min_bars=1, min_volume=1, min_dollar_volume=1, prior_sessions=2)
            reproduced = reproduce_baseline(pd.Timestamp("2026-07-30").date(), universe_path=universe, history_dir=history, settings=settings)
            reproduced.to_csv(top100_dir / "daily_top100_2026-07-30.csv", index=False)
            paths = analyze_range(["2026-07-31"], history_dir=history, top100_dir=top100_dir, universe_path=universe, output_dir=output, settings=settings)
            matrix = pd.read_parquet(paths["feature_matrix"])
            self.assertTrue(matrix["leakage_check_passed"].eq(1).all())
            quality = paths["data_quality"].read_text(encoding="utf-8")
            self.assertIn('"all_baselines_match": true', quality)


if __name__ == "__main__":
    unittest.main()
