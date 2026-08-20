from __future__ import annotations

import unittest
import tempfile
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.live_trading.analysis.multiday_top100_ranking_analyzer import (
    ELIGIBLE,
    VARIANTS,
    BaselineSettings,
    SessionHistoryCache,
    analyze_range,
    attach_outcomes,
    build_production_populations_from_cache,
    compare_baseline,
    portfolio_replays,
    reproduce_baseline,
    reproduce_baselines_from_cache,
    score_variants,
)
from src.live_trading.analysis.top100_analysis_common import session_dates
from src.live_trading.ranking.daily_top100_builder import parquet_path


class MultidayTop100RankingAnalyzerTests(unittest.TestCase):
    @staticmethod
    def write_candles(history: Path, symbol: str, current: date, *, base: float = 10.0) -> None:
        path = parquet_path(history, symbol, current, "RTH")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": pd.date_range(f"{current.isoformat()}T13:30:00Z", periods=5, freq="min"),
            "open": [base] * 5,
            "high": [base + 0.5] * 5,
            "low": [base - 0.1] * 5,
            "close": [base + value * 0.05 for value in range(5)],
            "volume": [1000] * 5,
        }).to_parquet(path, index=False)

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
                    base = 10 + offset + (day.day - 27) * 0.1
                    self.write_candles(history, symbol, day.date(), base=base)
            settings = BaselineSettings(top_n=2, min_price=1, min_bars=1, min_volume=1, min_dollar_volume=1, prior_sessions=2)
            reproduced = reproduce_baseline(pd.Timestamp("2026-07-30").date(), universe_path=universe, history_dir=history, settings=settings)
            reproduced.to_csv(top100_dir / "daily_top100_2026-07-30.csv", index=False)
            stdout = StringIO()
            with redirect_stdout(stdout):
                paths = analyze_range(["2026-07-31"], history_dir=history, top100_dir=top100_dir, universe_path=universe, output_dir=output, settings=settings)
            matrix = pd.read_parquet(paths["feature_matrix"])
            self.assertTrue(matrix["leakage_check_passed"].eq(1).all())
            quality = paths["data_quality"].read_text(encoding="utf-8")
            self.assertIn('"all_baselines_match": true', quality)
            self.assertIn('"performance_diagnostics"', quality)
            for event in (
                "P2_SESSION_START", "BASELINE_START", "BASELINE_DONE", "FEATURE_MATRIX_START",
                "FEATURE_MATRIX_DONE", "VARIANT_COMPARISON_START", "VARIANT_COMPARISON_DONE",
                "PORTFOLIO_REPLAY_START", "PORTFOLIO_REPLAY_DONE", "P2_SESSION_DONE",
            ):
                self.assertIn(event, stdout.getvalue())

    def test_cached_baseline_matches_production_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            universe = root / "universe.csv"
            pd.DataFrame({"symbol": ["AAA", "BBB"]}).to_csv(universe, index=False)
            for current in (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)):
                self.write_candles(history, "AAA", current, base=10.0)
                self.write_candles(history, "BBB", current, base=11.0)
            settings = BaselineSettings(top_n=2, min_price=1, min_bars=1, min_volume=1, min_dollar_volume=1, prior_sessions=2)
            expected = reproduce_baseline(date(2026, 7, 29), universe_path=universe, history_dir=history, settings=settings)
            cache = SessionHistoryCache(
                history,
                max_feature_date=date(2026, 7, 29),
                feature_dates={date(2026, 7, 29)},
                outcome_dates=set(),
                baseline_prior_sessions=2,
            )
            actual = reproduce_baselines_from_cache(
                [date(2026, 7, 29)], symbols=["AAA", "BBB"], cache=cache, settings=settings
            )[date(2026, 7, 29)]
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)

    def test_history_cache_reads_symbol_partitions_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history"
            for current in (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)):
                self.write_candles(history, "AAA", current)
            cache = SessionHistoryCache(
                history,
                max_feature_date=date(2026, 7, 28),
                feature_dates={date(2026, 7, 28)},
                outcome_dates={date(2026, 7, 29)},
                baseline_prior_sessions=1,
            )
            cache.prepare_symbol("AAA")
            cache.get_daily_history("AAA")
            cache.get_daily_metrics("AAA", date(2026, 7, 29))
            cache.get_session("AAA", date(2026, 7, 29))
            # D outcome is deliberately loaded after the D-1 feature batch.
            self.assertEqual(cache.diagnostics.parquet_read_operations, 2)
            self.assertEqual(cache.diagnostics.parquet_files_read, 3)
            self.assertGreaterEqual(cache.diagnostics.cache_hits, 2)

    def test_population_api_retains_full_universe_and_global_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history"
            current = date(2026, 7, 29)
            self.write_candles(history, "AAA", current, base=10.0)
            settings = BaselineSettings(
                top_n=1, min_price=1, min_bars=1, min_volume=1,
                min_dollar_volume=1, prior_sessions=0,
            )
            cache = SessionHistoryCache(
                history,
                max_feature_date=current,
                feature_dates={current},
                outcome_dates=set(),
                baseline_prior_sessions=0,
            )
            population = build_production_populations_from_cache(
                [current],
                symbols=["AAA", "MISSING"],
                cache=cache,
                settings=settings,
            )[current]
            self.assertEqual(set(population["symbol"]), {"AAA", "MISSING"})
            eligible = population[population["ranking_eligibility_status"].eq(ELIGIBLE)]
            self.assertEqual(eligible.iloc[0]["production_rank_global"], 1)
            self.assertTrue(
                population.loc[population["symbol"].eq("MISSING"), "production_rank_global"].isna().all()
            )

    def test_session_outcome_is_loaded_only_after_feature_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history"
            feature_date = date(2026, 7, 29)
            outcome_date = date(2026, 7, 30)
            self.write_candles(history, "AAA", feature_date, base=10.0)
            self.write_candles(history, "AAA", outcome_date, base=20.0)
            cache = SessionHistoryCache(
                history,
                max_feature_date=feature_date,
                feature_dates={feature_date},
                outcome_dates={outcome_date},
                baseline_prior_sessions=0,
            )
            _daily, frames = cache.prepare_symbol("AAA")
            self.assertNotIn(outcome_date, frames)
            self.assertNotIn(("AAA", outcome_date), cache.daily_metrics)
            matrix = pd.DataFrame({"symbol": ["AAA"]})
            attached = attach_outcomes(
                matrix, history, outcome_date.isoformat(), history_cache=cache
            )
            self.assertEqual(attached.iloc[0]["outcome_available"], 1)

    def test_cached_baseline_preserves_missing_and_invalid_history_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            universe = root / "universe.csv"
            pd.DataFrame({"symbol": ["AAA", "MISSING", "INVALID"]}).to_csv(universe, index=False)
            self.write_candles(history, "AAA", date(2026, 7, 29))
            invalid = parquet_path(history, "INVALID", date(2026, 7, 29), "RTH")
            invalid.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"timestamp": ["2026-07-29T13:30:00Z"], "close": [10.0]}).to_parquet(invalid, index=False)
            settings = BaselineSettings(top_n=3, min_price=1, min_bars=1, min_volume=1, min_dollar_volume=1, prior_sessions=0)
            expected = reproduce_baseline(date(2026, 7, 29), universe_path=universe, history_dir=history, settings=settings)
            cache = SessionHistoryCache(
                history,
                max_feature_date=date(2026, 7, 29),
                feature_dates={date(2026, 7, 29)},
                outcome_dates=set(),
                baseline_prior_sessions=0,
            )
            actual = reproduce_baselines_from_cache(
                [date(2026, 7, 29)], symbols=["AAA", "MISSING", "INVALID"], cache=cache, settings=settings
            )[date(2026, 7, 29)]
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)

    def test_portfolio_variants_share_prepared_feature_lookups(self) -> None:
        matrix = self.fixture()
        matrix["production_rank"] = [1, 2, 3]
        matrix = score_variants(matrix)
        cache = SimpleNamespace(
            diagnostics=SimpleNamespace(replay_session_calls=0),
            get_session=lambda symbol, current: pd.DataFrame({
                "timestamp": pd.to_datetime(["2026-07-31T13:30:00Z"]),
                "open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0], "volume": [1000],
            }),
        )
        replay = SimpleNamespace(trades=[], equity_curve=[], max_concurrent_positions=0, events=[], skipped={})
        with patch("src.live_trading.analysis.multiday_top100_ranking_analyzer.replay_session", return_value=replay) as mocked:
            portfolio_replays(
                matrix,
                session_date="2026-07-31",
                history_dir=Path("unused"),
                output_dir=Path("unused"),
                top_n=3,
                baseline_comparable=True,
                history_cache=cache,
            )
        self.assertEqual(mocked.call_count, len(VARIANTS))
        prepared_ids = {id(call.kwargs["prepared_sessions_by_symbol"]) for call in mocked.call_args_list}
        self.assertEqual(len(prepared_ids), 1)
        self.assertEqual(cache.diagnostics.replay_session_calls, len(VARIANTS))


if __name__ == "__main__":
    unittest.main()
