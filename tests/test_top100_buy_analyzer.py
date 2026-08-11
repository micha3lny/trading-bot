from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.live_trading.analysis.full_session_replay_v67 import (
    _rows,
    profile_config,
    replay_performance_counters,
    reset_replay_performance_counters,
)
from src.live_trading.analysis.top100_buy_analyzer import (
    PreparedCompletedBarFeatures,
    analyze_session,
    completed_bar_features,
    enrich_light_snapshots,
    portfolio_filter_simulation,
    replay_snapshots,
)
from src.live_trading.ranking.daily_top100_builder import parquet_path


def candles(start: str, closes: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps, "open": closes, "high": [value + 0.2 for value in closes],
        "low": [value - 0.1 for value in closes], "close": closes, "volume": [1000] * len(closes),
    })


class Top100BuyAnalyzerTests(unittest.TestCase):
    def test_completed_bar_features_exclude_unfinished_bar(self) -> None:
        frame = candles("2026-07-31T13:30:00Z", [10, 11, 50])
        result = completed_bar_features(frame, "2026-07-31T13:32:00Z")
        self.assertAlmostEqual(result["return_1m"], 10.0)
        self.assertLess(result["pullback_from_recent_high_pct"], 0)
        self.assertNotEqual(result["return_1m"], (50 / 11 - 1) * 100)

    def test_prepared_completed_features_match_reference(self) -> None:
        frame = candles("2026-07-31T13:30:00Z", [10, 11, 10.5, 12, 11.5, 13, 12.5])
        prepared = PreparedCompletedBarFeatures(_rows(frame, "bar_start"))
        for timestamp in pd.date_range("2026-07-31T13:29:00Z", "2026-07-31T13:40:00Z", freq="min"):
            expected = completed_bar_features(frame, timestamp)
            actual = prepared.at(timestamp)
            self.assertEqual(actual.keys(), expected.keys())
            for key, expected_value in expected.items():
                actual_value = actual[key]
                if isinstance(expected_value, float):
                    self.assertAlmostEqual(actual_value, expected_value, places=12, msg=f"{timestamp=} {key=}")
                else:
                    self.assertEqual(actual_value, expected_value, f"{timestamp=} {key=}")

    def test_replay_snapshots_avoids_repeated_full_frame_processing(self) -> None:
        frame = candles("2026-07-31T13:30:00Z", [10 + index * 0.1 for index in range(30)])
        rows = _rows(frame, "bar_start")
        top100 = pd.DataFrame({"symbol": ["AAA"], "rank": [1], "score": [100]})
        reset_replay_performance_counters()
        with patch("src.live_trading.analysis.top100_buy_analyzer._feature_at", side_effect=AssertionError("full-frame replay feature path used")), patch(
            "src.live_trading.analysis.top100_buy_analyzer.completed_bar_features",
            side_effect=AssertionError("full-frame completed-bar path used"),
        ):
            snapshots = replay_snapshots(
                "2026-07-31",
                top100,
                Path("unused"),
                profile_config("live"),
                prepared_rows_by_symbol={"AAA": rows},
            )
        self.assertEqual(len(snapshots), len(rows))
        counters = replay_performance_counters()
        self.assertEqual(counters["legacy_full_frame_feature_calls"], 0)
        self.assertEqual(counters["completed_bar_full_frame_calls"], 0)
        self.assertEqual(counters["replay_snapshots_calls"], 1)
        self.assertEqual(counters["causal_cursor_advances"], len(rows))

    def test_optimized_snapshots_match_full_frame_reference_exactly(self) -> None:
        frame = candles("2026-07-31T13:30:00Z", [10 + index * 0.1 for index in range(30)])
        rows = _rows(frame, "bar_start")
        top100 = pd.DataFrame({"symbol": ["AAA"], "rank": [1], "score": [100]})
        kwargs = {
            "session_date": "2026-07-31",
            "top100": top100,
            "history_dir": Path("unused"),
            "config": profile_config("live"),
            "prepared_rows_by_symbol": {"AAA": rows},
        }
        optimized = replay_snapshots(**kwargs)

        class ReferenceReplayFeatures:
            def __init__(self, prepared, config):
                self.rows = prepared
                self.config = config

            def at(self, timestamp):
                from src.live_trading.analysis.full_session_replay_v67 import _feature_at
                return _feature_at(self.rows, timestamp, self.config)

        class ReferenceCompletedFeatures:
            def __init__(self, prepared):
                self.rows = prepared

            def at(self, timestamp):
                return completed_bar_features(self.rows, timestamp)

        class ReferenceSession:
            def __init__(self, symbol, session_date, prepared, config):
                self.symbol = symbol
                self.session_date = session_date
                self.rows = prepared
                self.config = config
                self.replay_features = ReferenceReplayFeatures(prepared, config)
                self.completed_bar_features = ReferenceCompletedFeatures(prepared)

            def iter_features(self):
                for timestamp in sorted(self.rows["available_at"].dropna().unique()):
                    when = pd.Timestamp(timestamp)
                    yield when, self.replay_features.at(when), self.completed_bar_features.at(when)

        with patch("src.live_trading.analysis.top100_buy_analyzer.PreparedCausalSession", ReferenceSession):
            reference = replay_snapshots(**kwargs)
        pd.testing.assert_frame_equal(optimized, reference, check_exact=True)

    def test_end_to_end_outputs_match_full_frame_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top100_dir = root / "top100"
            history = root / "history"
            recorder = root / "recorder"
            optimized_output = root / "optimized"
            reference_output = root / "reference"
            top100_dir.mkdir()
            pd.DataFrame({"rank": [1, 2, 3], "symbol": ["READY", "NEVER", "MISSING"], "score": [100, 80, 60]}).to_csv(
                top100_dir / "daily_top100_2026-07-30.csv", index=False
            )
            for symbol, closes in {
                "READY": [10 + index * 0.1 for index in range(30)],
                "NEVER": [10 + index * 0.005 for index in range(30)],
            }.items():
                path = parquet_path(history, symbol, pd.Timestamp("2026-07-31").date(), "RTH")
                path.parent.mkdir(parents=True, exist_ok=True)
                candles("2026-07-31T13:30:00Z", closes).to_parquet(path, index=False)
            db = root / "runtime.sqlite"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE trades (trade_id TEXT, session_date TEXT, symbol TEXT, status TEXT, entry_fill_time TEXT, exit_fill_time TEXT, closed_at TEXT, entry_price REAL, exit_price REAL, quantity REAL, net_pnl REAL, exit_reason TEXT)")

            kwargs = {
                "session_date": "2026-07-31",
                "sqlite_path": db,
                "history_dir": history,
                "recorder_dir": recorder,
                "top100_dir": top100_dir,
            }
            optimized = analyze_session(output_dir=optimized_output, **kwargs)

            class ReferenceReplayFeatures:
                def __init__(self, prepared, config):
                    self.rows = prepared
                    self.config = config

                def latest_row(self, timestamp):
                    visible = self.rows[self.rows["available_at"] <= timestamp]
                    return None if visible.empty else visible.iloc[-1]

                def at(self, timestamp):
                    from src.live_trading.analysis.full_session_replay_v67 import _feature_at
                    return _feature_at(self.rows, timestamp, self.config)

            class ReferenceCompletedFeatures:
                def __init__(self, prepared):
                    self.rows = prepared

                def at(self, timestamp):
                    return completed_bar_features(self.rows, timestamp)

            class ReferenceSession:
                def __init__(self, symbol, session_date, prepared, config):
                    self.symbol = symbol
                    self.session_date = session_date
                    self.rows = prepared
                    self.config = config
                    self.replay_features = ReferenceReplayFeatures(prepared, config)
                    self.completed_bar_features = ReferenceCompletedFeatures(prepared)

                def iter_features(self):
                    for timestamp in sorted(self.rows["available_at"].dropna().unique()):
                        when = pd.Timestamp(timestamp)
                        yield when, self.replay_features.at(when), self.completed_bar_features.at(when)

            class ReferenceCache:
                def __init__(self, **_kwargs):
                    pass

                def get_or_build(self, symbol, session_date, prepared, config, **_kwargs):
                    return ReferenceSession(symbol, session_date, prepared, config)

            with patch("src.live_trading.analysis.top100_buy_analyzer.PreparedSessionCache", ReferenceCache):
                reference = analyze_session(output_dir=reference_output, **kwargs)

            for key in ("symbol_day", "snapshots", "feature_analysis", "filter_simulation", "portfolio_replay"):
                optimized_frame = pd.read_parquet(optimized[key]) if optimized[key].suffix == ".parquet" else pd.read_csv(optimized[key])
                reference_frame = pd.read_parquet(reference[key]) if reference[key].suffix == ".parquet" else pd.read_csv(reference[key])
                pd.testing.assert_frame_equal(optimized_frame, reference_frame, check_exact=True)
            self.assertEqual(json.loads(optimized["data_quality"].read_text()), json.loads(reference["data_quality"].read_text()))
            self.assertEqual(optimized["summary"].read_text(), reference["summary"].read_text())

    def test_light_dedupe_and_restart_identity(self) -> None:
        frame = pd.DataFrame([
            {"session_date": "2026-07-31", "process_start_id": "a", "scan_id": 1, "symbol": "AAA", "timestamp": "2026-07-31T13:31:00Z", "live_rank": 2, "live_entry_score": 10, "ready": 1},
            {"session_date": "2026-07-31", "process_start_id": "b", "scan_id": 1, "symbol": "AAA", "timestamp": "2026-07-31T13:32:00Z", "live_rank": 1, "live_entry_score": 12, "ready": 1},
        ])
        out = enrich_light_snapshots(frame)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[-1]["live_rank_delta_1_scan"], -1)
        self.assertEqual(out.iloc[-1]["consecutive_scans_ready"], 2)

    def test_symbol_day_keeps_every_dated_top100_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top100_dir = root / "top100"
            history = root / "history"
            recorder = root / "recorder"
            output = root / "analysis"
            top100_dir.mkdir()
            pd.DataFrame({"rank": [1, 2, 3], "symbol": ["AAA", "BBB", "CCC"], "score": [90, 80, 70]}).to_csv(top100_dir / "daily_top100_2026-07-30.csv", index=False)
            for symbol in ("AAA", "BBB"):
                path = parquet_path(history, symbol, pd.Timestamp("2026-07-31").date(), "RTH")
                path.parent.mkdir(parents=True, exist_ok=True)
                candles("2026-07-31T13:30:00Z", [10 + i * 0.1 for i in range(30)]).to_parquet(path, index=False)
            light_dir = recorder / "2026-07-31" / "top100_candidate_snapshots" / "light"
            light_dir.mkdir(parents=True)
            pd.DataFrame([
                {"session_date": "2026-07-31", "process_start_id": "p", "scan_id": 1, "scan_uid": "p:1", "symbol": "AAA", "timestamp": "2026-07-31T13:46:00Z", "live_rank": 1, "live_entry_score": 20, "ready": 1, "would_emit_signal_ready": 1, "current_price": 11.5},
            ]).to_parquet(light_dir / "chunk.parquet", index=False)
            db = root / "runtime.sqlite"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE trades (trade_id TEXT, session_date TEXT, symbol TEXT, status TEXT, entry_fill_time TEXT, exit_fill_time TEXT, closed_at TEXT, entry_price REAL, exit_price REAL, quantity REAL, net_pnl REAL, exit_reason TEXT)")
                conn.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("T1", "2026-07-31", "AAA", "CLOSED", "2026-07-31T13:46:00Z", "2026-07-31T14:00:00Z", "2026-07-31T14:00:00Z", 11.5, 12.0, 10, 4.0, "trailing_stop"))
            paths = analyze_session("2026-07-31", sqlite_path=db, history_dir=history, recorder_dir=recorder, top100_dir=top100_dir, output_dir=output)
            rows = pd.read_csv(paths["symbol_day"])
            self.assertEqual(set(rows["symbol"]), {"AAA", "BBB", "CCC"})
            self.assertEqual(len(rows), 3)
            self.assertEqual(int(rows.loc[rows["symbol"].eq("AAA"), "actually_bought"].iloc[0]), 1)
            self.assertEqual(rows.loc[rows["symbol"].eq("AAA"), "canonical_trade_id"].iloc[0], "T1")
            quality = json.loads(paths["data_quality"].read_text())
            self.assertEqual(quality["expected_top100_symbols"], 3)
            self.assertEqual(quality["symbols_loaded"], 3)
            filters = pd.read_csv(paths["filter_simulation"])
            self.assertIn("full_session_v67_baseline", set(filters["variant"]))

    def test_portfolio_simulation_respects_slots(self) -> None:
        rows = pd.DataFrame([
            {"symbol": "AAA", "potential_entry_eligible": 1, "hypothetical_entry_time": "2026-07-31T13:45:00Z", "hypothetical_exit_time": "2026-07-31T14:00:00Z", "hypothetical_net_pnl": 10, "hypothetical_gross_pnl": 12, "live_rank": 1, "top100_rank": 1},
            {"symbol": "BBB", "potential_entry_eligible": 1, "hypothetical_entry_time": "2026-07-31T13:46:00Z", "hypothetical_exit_time": "2026-07-31T13:55:00Z", "hypothetical_net_pnl": 20, "hypothetical_gross_pnl": 22, "live_rank": 2, "top100_rank": 2},
            {"symbol": "CCC", "potential_entry_eligible": 1, "hypothetical_entry_time": "2026-07-31T14:01:00Z", "hypothetical_exit_time": "2026-07-31T14:20:00Z", "hypothetical_net_pnl": -5, "hypothetical_gross_pnl": -3, "live_rank": 3, "top100_rank": 3},
        ])
        summary, replay = portfolio_filter_simulation(rows, max_positions=1)
        baseline = summary[summary["variant"].eq("baseline")].iloc[0]
        self.assertEqual(baseline["entries_selected"], 2)
        self.assertEqual(set(replay[replay["variant"].eq("baseline")]["symbol"]), {"AAA", "CCC"})

    def test_non_ready_future_winner_is_not_selected_by_baseline(self) -> None:
        rows = pd.DataFrame([
            {"symbol": "READY", "potential_entry_eligible": 1, "hypothetical_entry_time": "2026-07-31T13:45:00Z", "hypothetical_exit_time": "2026-07-31T14:00:00Z", "hypothetical_net_pnl": -1, "live_rank": 1},
            {"symbol": "FUTURE", "potential_entry_eligible": 0, "hypothetical_entry_time": "2026-07-31T13:45:00Z", "hypothetical_exit_time": "2026-07-31T14:00:00Z", "hypothetical_net_pnl": 100, "live_rank": 2},
        ])
        _summary, replay = portfolio_filter_simulation(rows, max_positions=5)
        self.assertEqual(replay.loc[replay["variant"].eq("baseline"), "symbol"].tolist(), ["READY"])


if __name__ == "__main__":
    unittest.main()
