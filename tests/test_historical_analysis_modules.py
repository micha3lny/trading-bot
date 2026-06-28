from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.bad_entries_analyzer import (
    analyze_bad_entries,
    first_green_seconds,
    load_closed_trades,
    rank_bucket,
    signal_age,
    signal_age_bucket,
)
from src.live_trading.analysis.common import (
    calculate_path_stats,
    calculate_runner_stats,
    entry_time_bucket,
    load_recorder_candles,
    min_after_pct,
    simulate_tp_sl,
)
from src.live_trading.analysis.missed_runners_analyzer import classify_missed_reason
from src.live_trading.analysis.missed_runners_analyzer import add_multiday_ranks, previous_session_context
from src.live_trading.analysis.overnight_hold_ranker import analyze_trade_overnight, ensure_overnight_columns, overnight_score, score_bucket
from src.live_trading.ranking.daily_top100_builder import parquet_path
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class HistoricalAnalysisModuleTests(unittest.TestCase):
    def candles(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-06-18T13:30:00Z"), "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
            {"timestamp": pd.Timestamp("2026-06-18T13:31:00Z"), "open": 10.1, "high": 11.0, "low": 10.0, "close": 10.8, "volume": 1000},
            {"timestamp": pd.Timestamp("2026-06-18T13:32:00Z"), "open": 10.8, "high": 10.9, "low": 9.6, "close": 9.8, "volume": 1000},
            {"timestamp": pd.Timestamp("2026-06-18T13:33:00Z"), "open": 9.8, "high": 10.4, "low": 9.7, "close": 10.2, "volume": 1000},
        ])

    def test_mfe_mae_and_min_after_entry(self) -> None:
        entry_time = pd.Timestamp("2026-06-18T13:30:00Z")
        stats = calculate_path_stats(self.candles(), 10.0, entry_time)
        self.assertAlmostEqual(stats.mfe_pct or 0.0, 10.0)
        self.assertAlmostEqual(stats.mae_pct or 0.0, -4.0)
        self.assertAlmostEqual(min_after_pct(self.candles(), 10.0, entry_time, 3) or 0.0, -4.0)
        self.assertEqual(stats.time_to_peak_seconds, 60.0)
        self.assertEqual(stats.time_to_low_seconds, 120.0)

    def test_tp_sl_simulation_uses_conservative_intraminute_order(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-06-18T13:30:00Z"), "open": 10.0, "high": 10.4, "low": 9.7, "close": 10.2, "volume": 1000},
        ])
        sim = simulate_tp_sl(
            candles,
            entry_price=10.0,
            tp_pct=3.0,
            sl_pct=-2.0,
            fallback_exit_time=pd.Timestamp("2026-06-18T13:35:00Z"),
            fallback_exit_price=10.1,
        )
        self.assertEqual(sim.exit_reason, "SL -2%")
        self.assertAlmostEqual(sim.exit_price or 0.0, 9.8)
        self.assertAlmostEqual(sim.pnl_pct or 0.0, -2.0)

    def test_tp_sl_simulation_handles_missing_candles(self) -> None:
        sim = simulate_tp_sl(
            pd.DataFrame(),
            entry_price=10.0,
            tp_pct=3.0,
            sl_pct=-2.0,
            fallback_exit_time=pd.Timestamp("2026-06-18T13:35:00Z"),
            fallback_exit_price=10.1,
        )
        self.assertEqual(sim.exit_reason, "actual_exit")
        self.assertAlmostEqual(sim.pnl_pct or 0.0, 1.0)

    def test_entry_time_buckets(self) -> None:
        self.assertEqual(entry_time_bucket(pd.Timestamp("2026-06-18T13:35:00Z"), "2026-06-18"), "0-15m")
        self.assertEqual(entry_time_bucket(pd.Timestamp("2026-06-18T13:50:00Z"), "2026-06-18"), "15-30m")
        self.assertEqual(entry_time_bucket(pd.Timestamp("2026-06-18T14:10:00Z"), "2026-06-18"), "30-60m")
        self.assertEqual(entry_time_bucket(pd.Timestamp("2026-06-18T15:00:00Z"), "2026-06-18"), "60m+")

    def test_missed_runner_threshold_detection(self) -> None:
        stats = calculate_runner_stats(self.candles())
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertGreaterEqual(stats.open_to_high_pct, 8.0)
        self.assertAlmostEqual(stats.open_to_high_pct, 10.0)

    def test_missed_reason_classification(self) -> None:
        reason = classify_missed_reason(
            source_bucket="top100",
            was_bought=False,
            entry_time=None,
            high_time=pd.Timestamp("2026-06-18T13:40:00Z"),
            signal_row={"blocked_reason": "spread_too_wide"},
            order_row={},
        )
        self.assertEqual(reason, "spread_too_wide")
        outside = classify_missed_reason(
            source_bucket="outside_top100",
            was_bought=False,
            entry_time=None,
            high_time=None,
            signal_row={},
            order_row={},
        )
        self.assertEqual(outside, "not_in_top100")

    def test_bad_entries_filters_primarily_by_session_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            conn = sqlite3.connect(db)
            try:
                conn.execute(
                    """
                    CREATE TABLE trades (
                        trade_id TEXT,
                        status TEXT,
                        session_date TEXT,
                        symbol TEXT,
                        entry_fill_time TEXT,
                        exit_fill_time TEXT,
                        closed_at TEXT
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        ("wanted", "CLOSED", "2026-06-26", "AAA", "2026-06-26T13:31:00+00:00", "2026-06-26T13:40:00+00:00", None),
                        ("old_session_closed_today", "CLOSED", "2026-05-20", "OLD", "2026-05-20T13:31:00+00:00", "2026-06-26T13:40:00+00:00", None),
                        ("fallback_empty_session", "CLOSED", "", "BBB", "2026-06-26T13:32:00+00:00", "2026-06-26T13:50:00+00:00", None),
                    ],
                )
                conn.commit()
            finally:
                conn.close()
            trades = load_closed_trades(db, "2026-06-26", "2026-06-26")
            self.assertEqual(set(trades["trade_id"]), {"wanted", "fallback_empty_session"})

    def test_signal_age_negative_is_warning_and_null(self) -> None:
        age, warning = signal_age(
            {"ready_since": "2026-06-26T13:35:00+00:00"},
            pd.Timestamp("2026-06-26T13:34:00Z"),
        )
        self.assertIsNone(age)
        self.assertEqual(warning, "negative_age")
        self.assertEqual(signal_age_bucket(age, warning), "invalid_negative")

    def test_rank_and_signal_age_buckets(self) -> None:
        self.assertEqual(rank_bucket(7), "1-10")
        self.assertEqual(rank_bucket(25), "11-25")
        self.assertEqual(rank_bucket(80), "76-100")
        self.assertEqual(signal_age_bucket(29), "0-30s")
        self.assertEqual(signal_age_bucket(90), "1-3m")
        self.assertEqual(signal_age_bucket(None), "missing")

    def test_first_green_and_never_green_inputs(self) -> None:
        entry_time = pd.Timestamp("2026-06-18T13:30:00Z")
        self.assertEqual(first_green_seconds(self.candles(), 10.0, entry_time), 0.0)
        red = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-06-18T13:30:00Z"), "open": 10.0, "high": 10.0, "low": 9.8, "close": 9.9, "volume": 1000},
            {"timestamp": pd.Timestamp("2026-06-18T13:31:00Z"), "open": 9.9, "high": 9.95, "low": 9.7, "close": 9.8, "volume": 1000},
        ])
        self.assertIsNone(first_green_seconds(red, 10.0, entry_time))

    def test_missed_runner_detectability_from_previous_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history"
            # 2026-06-18 previous trading days are 2026-06-17, 16, 15, 12, 11.
            for day, open_price, close_price, high_price in [
                ("2026-06-17", 10.0, 10.5, 10.8),
                ("2026-06-16", 9.7, 10.0, 10.2),
                ("2026-06-15", 9.4, 9.7, 10.0),
            ]:
                path = parquet_path(history, "AAA", pd.Timestamp(day).date(), "RTH")
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([
                    {"timestamp": f"{day}T13:30:00+00:00", "open": open_price, "high": high_price, "low": open_price * 0.99, "close": close_price, "volume": 1000},
                ]).to_parquet(path, index=False)
            ctx = previous_session_context(history, "AAA", "2026-06-18")
            self.assertEqual(ctx["was_detectable_from_history"], 1)
            self.assertIn("prev_1d_return>=3", ctx["detectability_reason"])

    def test_missed_runner_adds_hypothetical_multiday_rank(self) -> None:
        df = pd.DataFrame([
            {"symbol": "AAA", "prev_1d_return_pct": 5, "prev_3d_return_pct": 8, "prev_5d_return_pct": 10, "prev_3d_max_intraday_high_pct": 9, "prev_5d_max_intraday_high_pct": 12, "prev_3d_relative_volume_like": 2, "prev_5d_relative_volume_like": 2},
            {"symbol": "BBB", "prev_1d_return_pct": 0, "prev_3d_return_pct": 0, "prev_5d_return_pct": 0, "prev_3d_max_intraday_high_pct": 0, "prev_5d_max_intraday_high_pct": 0},
        ])
        ranked = add_multiday_ranks(df)
        self.assertEqual(ranked.loc[ranked["symbol"] == "AAA", "hypothetical_multiday_rank"].iloc[0], 1)
        self.assertEqual(ranked.loc[ranked["symbol"] == "AAA", "would_enter_multiday_top100"].iloc[0], 1)

    def test_recorder_candles_bar_time_timezone_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recorder" / "2026-06-26"
            root.mkdir(parents=True)
            pd.DataFrame([
                {"symbol": "AAA", "bar_time": "2026-06-26 09:30:00-04:00", "open": 10, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100},
            ]).to_csv(root / "candles_1m.csv", index=False)
            candles = load_recorder_candles(Path(tmp) / "recorder", "2026-06-26", "AAA")
            self.assertFalse(candles.empty)
            self.assertEqual(candles.iloc[0]["timestamp"], pd.Timestamp("2026-06-26T13:30:00Z"))

    def test_bad_entries_uses_parquet_when_recorder_candles_do_not_cover_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "runtime.sqlite"
            history = root / "history"
            recorder = root / "recorder"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "T_FCEL",
                    "strategy_name": "v67",
                    "session_date": "2026-06-26",
                    "symbol": "FCEL",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-06-26T14:02:47+00:00",
                    "exit_fill_time": "2026-06-26T14:12:47+00:00",
                    "entry_price": 10.0,
                    "exit_price": 10.4,
                    "quantity": 10,
                    "net_pnl": 4.0,
                    "raw_json": {"live_entry_score": 70},
                })
                store.upsert_execution({
                    "execution_id": "B_FCEL",
                    "trade_id": "T_FCEL",
                    "strategy_name": "v67",
                    "session_date": "2026-06-26",
                    "symbol": "FCEL",
                    "side": "BOT",
                    "quantity": 10,
                    "price": 10.0,
                    "executed_at": "2026-06-26T14:02:47+00:00",
                })
            finally:
                store.close()
            rec_root = recorder / "2026-06-26"
            rec_root.mkdir(parents=True)
            pd.DataFrame([
                {"symbol": "FCEL", "bar_time": "2026-06-26 09:30:00-04:00", "open": 9.8, "high": 10.0, "low": 9.7, "close": 9.9, "volume": 100},
                {"symbol": "FCEL", "bar_time": "2026-06-26 09:58:00-04:00", "open": 9.9, "high": 10.1, "low": 9.8, "close": 10.0, "volume": 100},
            ]).to_csv(rec_root / "candles_1m.csv", index=False)
            path = parquet_path(history, "FCEL", pd.Timestamp("2026-06-26").date(), "RTH")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"timestamp": "2026-06-26T14:02:00+00:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
                {"timestamp": "2026-06-26T14:03:00+00:00", "open": 10.1, "high": 10.8, "low": 10.0, "close": 10.7, "volume": 1000},
                {"timestamp": "2026-06-26T14:12:00+00:00", "open": 10.4, "high": 10.5, "low": 10.2, "close": 10.4, "volume": 1000},
                {"timestamp": "2026-06-26T14:13:00+00:00", "open": 10.4, "high": 10.5, "low": 10.2, "close": 10.4, "volume": 1000},
            ]).to_parquet(path, index=False)

            out = analyze_bad_entries(
                start_date="2026-06-26",
                end_date="2026-06-26",
                sqlite_path=db,
                history_dir=history,
                recorder_dir=recorder,
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["candle_source"], "parquet")
            self.assertEqual(out.iloc[0]["candle_coverage_warning"], "recorder_incomplete")
            self.assertGreater(float(out.iloc[0]["mfe_pct"]), 0)

    def test_bad_entries_keeps_execution_fifo_trade_when_trades_table_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "runtime.sqlite"
            history = root / "history"
            recorder = root / "recorder"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_execution({
                    "execution_id": "B_UNCY",
                    "strategy_name": "v67",
                    "session_date": "2026-06-26",
                    "symbol": "UNCY",
                    "side": "BOT",
                    "quantity": 10,
                    "price": 10.0,
                    "executed_at": "2026-06-26T14:08:53+00:00",
                    "commission": 0.5,
                    "commission_source": "ibkr",
                })
                store.upsert_execution({
                    "execution_id": "S_UNCY",
                    "strategy_name": "v67",
                    "session_date": "2026-06-26",
                    "symbol": "UNCY",
                    "side": "SLD",
                    "quantity": 10,
                    "price": 10.5,
                    "executed_at": "2026-06-26T14:18:53+00:00",
                    "commission": 0.5,
                    "commission_source": "ibkr",
                    "realized_pnl": 5.0,
                })
                store.execute("DELETE FROM trades")
            finally:
                store.close()
            path = parquet_path(history, "UNCY", pd.Timestamp("2026-06-26").date(), "RTH")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"timestamp": "2026-06-26T14:08:00+00:00", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1000},
                {"timestamp": "2026-06-26T14:09:00+00:00", "open": 10.0, "high": 10.7, "low": 10.0, "close": 10.6, "volume": 1000},
                {"timestamp": "2026-06-26T14:18:00+00:00", "open": 10.4, "high": 10.5, "low": 10.3, "close": 10.5, "volume": 1000},
                {"timestamp": "2026-06-26T14:19:00+00:00", "open": 10.5, "high": 10.5, "low": 10.3, "close": 10.5, "volume": 1000},
            ]).to_parquet(path, index=False)

            out = analyze_bad_entries(
                start_date="2026-06-26",
                end_date="2026-06-26",
                sqlite_path=db,
                history_dir=history,
                recorder_dir=recorder,
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["symbol"], "UNCY")
            self.assertEqual(out.iloc[0]["candle_source"], "parquet")
            self.assertGreater(float(out.iloc[0]["mfe_pct"]), 0)

    def test_overnight_score_and_bucket(self) -> None:
        score, bucket, reason = overnight_score({
            "next_session_high_from_entry_pct": 6.0,
            "next_session_close_from_entry_pct": 3.0,
            "next_session_open_gap_pct": 1.2,
            "mfe_pct": 4.0,
            "live_entry_score": 40.0,
            "top100_rank": 5,
            "next_session_max_drawdown_from_entry_pct": -1.0,
            "mae_pct": -1.0,
            "never_green": 0,
            "immediate_drop": 0,
        })
        self.assertEqual(score, 80.0)
        self.assertEqual(bucket, "strong_hold_candidate")
        self.assertIn("next_high>=5", reason)
        self.assertEqual(score_bucket(20), "avoid_overnight")

    def test_overnight_missing_next_session_data_has_no_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = analyze_trade_overnight(
                {
                    "trade_id": "T1",
                    "symbol": "AAA",
                    "session_date": "2026-06-26",
                    "entry_fill_time": "2026-06-26T13:31:00+00:00",
                    "exit_fill_time": "2026-06-26T14:00:00+00:00",
                    "entry_price": 10.0,
                    "exit_price": 10.2,
                    "quantity": 1,
                    "net_pnl": 0.2,
                },
                history_dir=Path(tmp) / "history",
                recorder_dir=Path(tmp) / "recorder",
                session_type="RTH",
            )
            self.assertIsNone(row["overnight_hold_score"])
            self.assertEqual(row["overnight_hold_bucket"], "missing_next_session_data")

    def test_overnight_migration_adds_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE trades (trade_id TEXT PRIMARY KEY, status TEXT)")
                conn.commit()
            finally:
                conn.close()
            added = ensure_overnight_columns(db)
            self.assertIn("overnight_hold_score", added)
            conn = sqlite3.connect(db)
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
            finally:
                conn.close()
            self.assertIn("overnight_hold_features_json", cols)

    def test_runtime_store_schema_includes_overnight_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.close()
            conn = sqlite3.connect(db)
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
            finally:
                conn.close()
            self.assertIn("overnight_hold_score", cols)
            self.assertIn("next_session_high_from_entry_pct", cols)


if __name__ == "__main__":
    unittest.main()
