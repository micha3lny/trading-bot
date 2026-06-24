from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.live_trading.ranking.daily_top100_builder import (
    build_daily_top,
    parquet_path,
    recent_prior_closes_with_diagnostics,
    update_latest_output,
    write_diagnostics_csv,
    write_output_csv,
)
from src.live_trading.ranking.ranking_store import RankingStore


def write_universe(path: Path, symbols: list[str]) -> None:
    pd.DataFrame({"symbol": symbols}).to_csv(path, index=False)


def session_frame(symbol: str, start_price: float, close_price: float, volume: int = 1_000) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 5, 15, 13, 30, tzinfo=timezone.utc)
    steps = 210
    for idx in range(steps):
        frac = idx / (steps - 1)
        price = start_price + (close_price - start_price) * frac
        rows.append(
            {
                "symbol": symbol,
                "bar_time_utc": (start + timedelta(minutes=idx)).isoformat(),
                "open": price,
                "high": price * (1.005 + 0.0005 * (idx % 5)),
                "low": price * 0.997,
                "close": price,
                "volume": volume,
                "wap": price,
                "trade_count": 10,
                "session_type": "RTH",
            }
        )
    return pd.DataFrame(rows)


def write_session(history_dir: Path, symbol: str, session_date: date, df: pd.DataFrame) -> None:
    path = parquet_path(history_dir, symbol, session_date, "RTH")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise unittest.SkipTest("pyarrow/fastparquet is required for parquet integration test") from exc


class DailyTop100BuilderTests(unittest.TestCase):
    def test_build_daily_top_outputs_compatible_ranked_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            history = root / "history"
            output = root / "daily_top.csv"
            write_universe(universe, ["AAA", "BBB", "MISS"])
            write_session(history, "AAA", date(2026, 5, 15), session_frame("AAA", 10.0, 13.0, 5_000))
            write_session(history, "BBB", date(2026, 5, 15), session_frame("BBB", 20.0, 20.5, 2_000))
            write_session(history, "AAA", date(2026, 5, 14), session_frame("AAA", 9.0, 9.5, 3_000))

            rows, stats = build_daily_top(
                ranking_date=date(2026, 5, 15),
                universe_path=universe,
                history_dir=history,
                top_n=2,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
            )
            write_output_csv(output, rows)

            self.assertEqual(len(rows), 2)
            self.assertEqual(stats["missing"], 1)
            self.assertEqual(rows[0]["rank"], 1)
            self.assertEqual(rows[0]["symbol"], "AAA")
            self.assertIn("alpha_score", rows[0])
            self.assertIn("final_score", rows[0])
            self.assertIn("close_open_pct", rows[0])
            self.assertIn("momentum_score", rows[0])
            self.assertIn("liquidity_score", rows[0])
            self.assertEqual(rows[0]["score"], rows[0]["alpha_score"])
            self.assertGreaterEqual(float(stats["elapsed_seconds"]), 0.0)
            self.assertGreater(float(stats["symbols_per_second"]), 0.0)
            self.assertGreaterEqual(float(stats["current_read_seconds"]), 0.0)
            self.assertGreaterEqual(float(stats["prior_read_seconds"]), 0.0)
            self.assertGreaterEqual(float(stats["analyze_seconds"]), 0.0)
            self.assertEqual(stats["current_day_read_seconds"], stats["current_read_seconds"])
            self.assertEqual(stats["prior_sessions_read_seconds"], stats["prior_read_seconds"])
            self.assertEqual(stats["analysis_seconds"], stats["analyze_seconds"])
            self.assertEqual(stats["total_seconds"], stats["elapsed_seconds"])
            loaded = pd.read_csv(output)
            self.assertIn("components_json", loaded.columns)
            self.assertEqual(loaded["symbol"].tolist(), ["AAA", "BBB"])

    def test_denylisted_symbol_is_excluded_from_top100_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            history = root / "history"
            denylist = root / "denylist.csv"
            runtime_ineligible = root / "ineligible.json"
            write_universe(universe, ["CONL", "AAA", "BBB"])
            write_session(history, "CONL", date(2026, 5, 15), session_frame("CONL", 10.0, 20.0, 10_000))
            write_session(history, "AAA", date(2026, 5, 15), session_frame("AAA", 10.0, 13.0, 5_000))
            write_session(history, "BBB", date(2026, 5, 15), session_frame("BBB", 20.0, 21.0, 5_000))
            denylist.write_text(
                "symbol,reason,source,first_seen_at,last_seen_at,notes\n"
                "CONL,kid_priip_ineligible,ibkr_error_201,2026-05-29T00:00:00+00:00,2026-05-29T00:00:00+00:00,\n",
                encoding="utf-8",
            )

            rows, stats = build_daily_top(
                ranking_date=date(2026, 5, 15),
                universe_path=universe,
                history_dir=history,
                top_n=2,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
                symbol_denylist_path=denylist,
                runtime_ineligible_path=runtime_ineligible,
            )

            self.assertEqual([row["symbol"] for row in rows], ["AAA", "BBB"])
            self.assertEqual(stats["excluded_ineligible"], 1)
            self.assertEqual(stats["_excluded_ineligible_rows"][0]["symbol"], "CONL")

    def test_recent_prior_closes_reads_limited_prior_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            ranking_date = date(2026, 5, 15)
            write_session(history, "AAA", date(2026, 5, 14), session_frame("AAA", 10.0, 11.0, 5_000))
            write_session(history, "AAA", date(2026, 5, 13), session_frame("AAA", 10.0, 12.0, 5_000))
            write_session(history, "AAA", date(2026, 5, 12), session_frame("AAA", 10.0, 13.0, 5_000))

            result = recent_prior_closes_with_diagnostics(
                history,
                "AAA",
                ranking_date,
                limit=2,
                session_type="RTH",
                slow_seconds=10.0,
            )

            self.assertEqual(result.closes, [11.0, 12.0])
            self.assertEqual(result.paths_checked, 2)
            self.assertEqual(result.paths_found, 2)
            self.assertFalse(result.degraded)

    def test_prior_read_slow_guard_degrades_symbol_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            ranking_date = date(2026, 5, 15)
            write_universe(root / "universe.csv", ["AAA"])
            write_session(history, "AAA", ranking_date, session_frame("AAA", 10.0, 13.0, 5_000))
            write_session(history, "AAA", date(2026, 5, 14), session_frame("AAA", 10.0, 11.0, 5_000))
            write_session(history, "AAA", date(2026, 5, 13), session_frame("AAA", 10.0, 12.0, 5_000))

            rows, stats = build_daily_top(
                ranking_date=ranking_date,
                universe_path=root / "universe.csv",
                history_dir=history,
                top_n=1,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
                prior_read_slow_seconds=0.000001,
            )

            self.assertEqual(len(rows), 1)
            self.assertGreaterEqual(stats["prior_slow_symbols"], 1)
            self.assertGreaterEqual(stats["prior_degraded_symbols"], 1)
            self.assertGreaterEqual(stats["prior_partial_symbols"], 1)

    def test_missing_prior_session_data_degrades_without_rejecting_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            ranking_date = date(2026, 5, 15)
            write_universe(root / "universe.csv", ["AAA"])
            write_session(history, "AAA", ranking_date, session_frame("AAA", 10.0, 13.0, 5_000))

            rows, stats = build_daily_top(
                ranking_date=ranking_date,
                universe_path=root / "universe.csv",
                history_dir=history,
                top_n=1,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
                max_partial_history_log=0,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(stats["valid"], 1)
            self.assertEqual(stats["prior_partial_symbols"], 1)
            self.assertEqual(stats["prior_paths_found"], 0)

    def test_diagnostics_report_contains_missing_and_rejected_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            history = root / "history"
            diagnostics = root / "diagnostics.csv"
            write_universe(universe, ["AAA", "MISS", "BADWW"])
            write_session(history, "AAA", date(2026, 5, 15), session_frame("AAA", 10.0, 13.0, 5_000))
            write_session(history, "BADWW", date(2026, 5, 15), session_frame("BADWW", 10.0, 13.0, 5_000))

            _, stats = build_daily_top(
                ranking_date=date(2026, 5, 15),
                universe_path=universe,
                history_dir=history,
                top_n=2,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
                max_missing_log=0,
                max_reject_log=0,
            )
            written = write_diagnostics_csv(diagnostics, date(2026, 5, 15), stats)
            report = pd.read_csv(diagnostics)

            self.assertEqual(written, 2)
            self.assertEqual(set(report["status"]), {"missing", "rejected"})
            self.assertIn("MISS", set(report["symbol"]))
            self.assertIn("BADWW", set(report["symbol"]))

    def test_runner_potential_beats_raw_mega_cap_liquidity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            history = root / "history"
            write_universe(universe, ["MEGA", "RUNR"])
            write_session(history, "MEGA", date(2026, 5, 15), session_frame("MEGA", 500.0, 501.0, 2_000_000))
            write_session(history, "RUNR", date(2026, 5, 15), session_frame("RUNR", 12.0, 14.4, 20_000))

            rows, _ = build_daily_top(
                ranking_date=date(2026, 5, 15),
                universe_path=universe,
                history_dir=history,
                top_n=2,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
            )

            self.assertEqual(rows[0]["symbol"], "RUNR")
            self.assertGreater(rows[0]["momentum_score"], rows[1]["momentum_score"])
            self.assertLessEqual(rows[0]["liquidity_score"], rows[1]["liquidity_score"])

    def test_latest_output_only_updates_for_valid_minimum_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dated = root / "daily_top100_2026-05-15.csv"
            latest = root / "daily_top100_latest.csv"
            latest.write_text("rank,symbol,score,alpha_score\n1,OLD,1,1\n", encoding="utf-8")

            short_rows = [{"rank": 1, "symbol": "AAA", "score": 90.0, "alpha_score": 90.0}]
            write_output_csv(dated, short_rows)
            self.assertFalse(update_latest_output(dated, latest, short_rows))
            self.assertIn("OLD", latest.read_text(encoding="utf-8"))

            valid_rows = [
                {
                    "rank": idx,
                    "symbol": f"S{idx:03d}",
                    "score": 100.0 - idx / 1000,
                    "alpha_score": 100.0 - idx / 1000,
                }
                for idx in range(1, 101)
            ]
            write_output_csv(dated, valid_rows)
            self.assertTrue(update_latest_output(dated, latest, valid_rows))
            loaded = pd.read_csv(latest)
            self.assertEqual(len(loaded), 100)
            self.assertEqual(loaded["symbol"].iloc[0], "S001")

    def test_latest_output_blocks_when_missing_history_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dated = root / "daily_top100_2026-06-22.csv"
            latest = root / "daily_top100_latest.csv"
            latest.write_text("rank,symbol,score,alpha_score\n1,OLD,1,1\n", encoding="utf-8")
            rows = [
                {
                    "rank": idx,
                    "symbol": f"S{idx:03d}",
                    "score": 100.0 - idx / 1000,
                    "alpha_score": 100.0 - idx / 1000,
                }
                for idx in range(1, 101)
            ]
            write_output_csv(dated, rows)

            self.assertFalse(
                update_latest_output(
                    dated,
                    latest,
                    rows,
                    missing_history_count=1,
                    max_missing_history_for_latest=0,
                )
            )
            self.assertIn("OLD", latest.read_text(encoding="utf-8"))

    def test_ranking_store_replaces_one_day_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RankingStore(Path(tmp) / "rankings.sqlite")
            store.replace_daily_rankings(
                "2026-05-15",
                [
                    {"rank": 1, "symbol": "AAA", "score": 80.0, "components_json": "{}"},
                    {"rank": 2, "symbol": "BBB", "score": 70.0, "components_json": "{}"},
                ],
            )
            store.replace_daily_rankings(
                "2026-05-15",
                [{"rank": 1, "symbol": "CCC", "score": 90.0, "components_json": "{}"}],
            )

            rows = store.load_daily_rankings("2026-05-15")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "CCC")
            self.assertEqual(rows[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
