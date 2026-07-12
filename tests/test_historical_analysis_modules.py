from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.bad_entries_analyzer import (
    analyze_bad_entries,
    dedupe_logical_trades_for_analysis,
    first_green_seconds,
    load_closed_trades,
    rank_bucket,
    signal_age,
    signal_age_bucket,
)
from src.live_trading.analysis.buy_decision_trace import (
    build_parser as build_buy_decision_trace_parser,
    classify_verdict as classify_buy_decision_verdict,
    heartbeat_block_state,
)
from src.live_trading.analysis.entry_timing_analyzer import (
    build_parser as build_entry_timing_parser,
    entry_chasing_score,
    pullback_from_recent_high,
    simulate_pullback_entry,
)
from src.live_trading.analysis.common import (
    calculate_path_stats,
    calculate_runner_stats,
    entry_time_bucket,
    load_recorder_candles,
    min_after_pct,
    simulate_tp_sl,
)
from src.live_trading.analysis.missed_runners_analyzer import build_parser as build_missed_runners_parser, classify_missed_reason
from src.live_trading.analysis.missed_runners_analyzer import add_multiday_ranks, no_signal_diagnostics, previous_session_context
from src.live_trading.analysis.no_buy_after_signal_investigator import (
    build_parser as build_no_buy_after_signal_parser,
    classify_no_buy_reason,
    first_record_time,
    lifecycle_stage_trace,
    records_in_window,
    select_signal_ready_time,
    post_signal_terminal_evidence,
    summary_for_cases as no_buy_after_signal_summary_for_cases,
)
from src.live_trading.analysis.overnight_hold_ranker import analyze_trade_overnight, ensure_overnight_columns, overnight_score, score_bucket
from src.live_trading.analysis.signal_replay_analyzer import (
    build_parser as build_signal_replay_parser,
    classify_replay_reason,
    filter_should_have_signaled_targets,
    merge_timeline_events,
)
from src.live_trading.analysis.signal_case_trace import (
    build_parser as build_signal_case_trace_parser,
    decision_classification,
    pass_fail,
)
from src.live_trading.analysis.should_have_signaled_investigator import (
    build_parser as build_shs_investigator_parser,
    summary_for_cases,
)
from src.live_trading.analysis.strategy_coverage_report import build_runner_rows, summarize_coverage_from_missed
from src.live_trading.analysis.symbol_subscription_inspector import (
    build_parser as build_subscription_inspector_parser,
    extract_last_restart_unblock_time,
    infer_verdict,
    parse_key_values,
    stale_or_backfill_skip_symbols,
    symbol_journal_lines,
)
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

    def test_strategy_coverage_summary_buckets_runners_and_misses(self) -> None:
        missed = pd.DataFrame([
            {
                "symbol": "AAA",
                "open_to_high_pct": 12.0,
                "source_bucket": "top100",
                "was_bought": 1,
                "was_detectable_from_history": 1,
                "top100_no_signal_reason": "",
                "signal_time": "2026-06-26T14:00:00Z",
                "ready_since": "2026-06-26T14:00:00Z",
                "blocked_reason": "",
                "rejection_reason": "",
            },
            {
                "symbol": "BBB",
                "open_to_high_pct": 18.0,
                "source_bucket": "outside_top100",
                "was_bought": 0,
                "was_detectable_from_history": 1,
                "top100_no_signal_reason": "",
                "signal_time": "",
                "ready_since": "",
                "blocked_reason": "",
                "rejection_reason": "",
            },
            {
                "symbol": "CCC",
                "open_to_high_pct": 6.0,
                "source_bucket": "top100",
                "was_bought": 0,
                "was_detectable_from_history": 1,
                "top100_no_signal_reason": "should_have_signaled",
                "signal_time": "",
                "ready_since": "",
                "blocked_reason": "",
                "rejection_reason": "",
            },
            {
                "symbol": "DDD",
                "open_to_high_pct": 22.0,
                "source_bucket": "outside_top100",
                "was_bought": 0,
                "was_detectable_from_history": 0,
                "top100_no_signal_reason": "",
                "signal_time": "",
                "ready_since": "",
                "blocked_reason": "",
                "rejection_reason": "",
            },
        ])
        summary = summarize_coverage_from_missed(missed, session_date="2026-06-26")
        self.assertEqual(summary["universe_gt_5"], 4)
        self.assertEqual(summary["top100_gt_5"], 2)
        self.assertEqual(summary["top100_runner_count_gt_5"], 2)
        self.assertEqual(summary["bought_gt_5"], 1)
        self.assertEqual(summary["bought_runner_count_gt_5"], 1)
        self.assertEqual(summary["missed_gt_5"], 3)
        self.assertAlmostEqual(float(summary["coverage_gt_5_pct"]), 50.0)
        self.assertAlmostEqual(float(summary["capture_gt_5_pct"]), 50.0)
        self.assertEqual(summary["universe_gt_10"], 3)
        self.assertEqual(summary["top100_gt_10"], 1)
        self.assertEqual(summary["bought_gt_10"], 1)
        self.assertEqual(summary["missed_detectable"], 2)
        self.assertEqual(summary["missed_undetectable"], 1)
        self.assertEqual(summary["missed_should_have_signaled"], 1)
        self.assertEqual(summary["missed_runtime_missing"], 1)

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

    def test_bad_entries_groups_partial_exit_fills_by_logical_trade_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "runtime.sqlite"
            history = root / "history"
            recorder = root / "recorder"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_execution({
                    "execution_id": "B_CAST",
                    "order_id": 65046,
                    "strategy_name": "v67",
                    "session_date": "2026-06-26",
                    "symbol": "CAST",
                    "side": "BOT",
                    "quantity": 113,
                    "price": 8.81,
                    "executed_at": "2026-06-26T14:08:00+00:00",
                    "commission": 1.0,
                    "commission_source": "ibkr",
                })
                for exec_id, qty, ts in [
                    ("S_CAST_1", 100, "2026-06-26T14:18:00+00:00"),
                    ("S_CAST_2", 13, "2026-06-26T14:18:05+00:00"),
                ]:
                    store.upsert_execution({
                        "execution_id": exec_id,
                        "order_id": 65047,
                        "strategy_name": "v67",
                        "session_date": "2026-06-26",
                        "symbol": "CAST",
                        "side": "SLD",
                        "quantity": qty,
                        "price": 8.79,
                        "executed_at": ts,
                        "commission": 0.5,
                        "commission_source": "ibkr",
                        "realized_pnl": -0.02 * qty,
                    })
                store.execute("DELETE FROM trades")
            finally:
                store.close()
            path = parquet_path(history, "CAST", pd.Timestamp("2026-06-26").date(), "RTH")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"timestamp": "2026-06-26T14:08:00+00:00", "open": 8.81, "high": 8.85, "low": 8.78, "close": 8.82, "volume": 1000},
                {"timestamp": "2026-06-26T14:09:00+00:00", "open": 8.82, "high": 8.90, "low": 8.80, "close": 8.87, "volume": 1000},
                {"timestamp": "2026-06-26T14:18:00+00:00", "open": 8.80, "high": 8.81, "low": 8.78, "close": 8.79, "volume": 1000},
                {"timestamp": "2026-06-26T14:19:00+00:00", "open": 8.79, "high": 8.81, "low": 8.78, "close": 8.79, "volume": 1000},
            ]).to_parquet(path, index=False)

            grouped = analyze_bad_entries(
                start_date="2026-06-26",
                end_date="2026-06-26",
                sqlite_path=db,
                history_dir=history,
                recorder_dir=recorder,
            )
            per_fill = analyze_bad_entries(
                start_date="2026-06-26",
                end_date="2026-06-26",
                sqlite_path=db,
                history_dir=history,
                recorder_dir=recorder,
                per_fill=True,
            )
            self.assertEqual(len(grouped), 1)
            self.assertEqual(grouped.iloc[0]["analysis_source"], "reconstructed_execution_fifo")
            self.assertAlmostEqual(float(grouped.iloc[0]["quantity"]), 113.0)
            self.assertEqual(len(per_fill), 2)
            self.assertTrue((per_fill["analysis_source"] == "reconstructed_execution_fifo_fill").all())

    def test_bad_entries_dedupe_collapses_duplicate_partial_fill_rows(self) -> None:
        rows = pd.DataFrame([
            {
                "trade_id": "CAST_100",
                "analysis_source": "sqlite_trades",
                "session_date": "2026-07-08",
                "symbol": "CAST",
                "entry_order_id": 65046,
                "entry_fill_time": "2026-07-08T14:08:00+00:00",
                "exit_fill_time": "2026-07-08T14:18:00+00:00",
                "entry_price": 8.81,
                "exit_price": 8.79,
                "quantity": 100,
                "net_pnl": -2.0,
                "raw_json": {"entry_order_id": 65046},
            },
            {
                "trade_id": "CAST_13",
                "analysis_source": "sqlite_trades",
                "session_date": "2026-07-08",
                "symbol": "CAST",
                "entry_order_id": 65046,
                "entry_fill_time": "2026-07-08T14:08:00+00:00",
                "exit_fill_time": "2026-07-08T14:18:05+00:00",
                "entry_price": 8.81,
                "exit_price": 8.79,
                "quantity": 13,
                "net_pnl": -0.26,
                "raw_json": {"entry_order_id": 65046},
            },
        ])
        out = dedupe_logical_trades_for_analysis(rows)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["quantity"]), 113.0)
        self.assertAlmostEqual(float(out.iloc[0]["net_pnl"]), -2.26)
        self.assertEqual(out.attrs["dedupe_diagnostics"]["dedupe_removed_rows"], 1)

    def test_bad_entries_dedupe_keeps_independent_same_symbol_trades(self) -> None:
        rows = pd.DataFrame([
            {
                "trade_id": "RXT_1",
                "analysis_source": "sqlite_trades",
                "session_date": "2026-07-08",
                "symbol": "RXT",
                "entry_order_id": 70001,
                "entry_fill_time": "2026-07-08T14:08:00+00:00",
                "quantity": 50,
                "net_pnl": 1.0,
                "raw_json": {"entry_order_id": 70001},
            },
            {
                "trade_id": "RXT_2",
                "analysis_source": "sqlite_trades",
                "session_date": "2026-07-08",
                "symbol": "RXT",
                "entry_order_id": 70002,
                "entry_fill_time": "2026-07-08T15:08:00+00:00",
                "quantity": 50,
                "net_pnl": 2.0,
                "raw_json": {"entry_order_id": 70002},
            },
        ])
        out = dedupe_logical_trades_for_analysis(rows)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.attrs["dedupe_diagnostics"]["dedupe_removed_rows"], 0)

    def test_bad_entries_dedupe_prefers_sqlite_over_reconstructed_duplicate(self) -> None:
        rows = pd.DataFrame([
            {
                "trade_id": "SQL_AMPG",
                "analysis_source": "sqlite_trades",
                "session_date": "2026-07-08",
                "symbol": "AMPG",
                "entry_order_id": 71001,
                "entry_fill_time": "2026-07-08T14:08:00+00:00",
                "exit_fill_time": "2026-07-08T14:18:00+00:00",
                "entry_price": 10.0,
                "exit_price": 10.5,
                "quantity": 100,
                "net_pnl": 50.0,
                "live_entry_score": 72.0,
                "raw_json": {"entry_order_id": 71001, "live_entry_score": 72.0},
            },
            {
                "trade_id": "exec_fifo:2026-07-08:AMPG:B1:S1:100",
                "analysis_source": "reconstructed_execution_fifo",
                "session_date": "2026-07-08",
                "symbol": "AMPG",
                "entry_order_id": 71001,
                "entry_fill_time": "2026-07-08T14:08:00+00:00",
                "exit_fill_time": "2026-07-08T14:18:00+00:00",
                "entry_price": 10.0,
                "exit_price": 10.5,
                "quantity": 100,
                "net_pnl": 50.0,
                "raw_json": {"entry_order_id": 71001, "reconstruction_source": "bad_entries_execution_fifo_grouped"},
            },
        ])
        out = dedupe_logical_trades_for_analysis(rows)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["quantity"]), 100.0)
        self.assertAlmostEqual(float(out.iloc[0]["net_pnl"]), 50.0)
        self.assertAlmostEqual(float(out.iloc[0]["live_entry_score"]), 72.0)
        raw = out.iloc[0]["raw_json"]
        self.assertEqual(raw["dedupe_dropped_reconstructed_row_count"], 1)

    def test_bad_entries_dedupe_does_not_double_count_exact_duplicate_rows(self) -> None:
        row = {
            "trade_id": "AXTI_1",
            "analysis_source": "sqlite_trades",
            "session_date": "2026-07-08",
            "symbol": "AXTI",
            "entry_order_id": 72001,
            "entry_fill_time": "2026-07-08T14:08:00+00:00",
            "exit_fill_time": "2026-07-08T14:18:00+00:00",
            "entry_price": 10.0,
            "exit_price": 9.9,
            "quantity": 100,
            "net_pnl": -10.0,
            "raw_json": {"entry_order_id": 72001},
        }
        duplicate = dict(row)
        duplicate["trade_id"] = "AXTI_1_COPY"
        rows = pd.DataFrame([row, duplicate])
        out = dedupe_logical_trades_for_analysis(rows)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["quantity"]), 100.0)
        self.assertAlmostEqual(float(out.iloc[0]["net_pnl"]), -10.0)
        raw = out.iloc[0]["raw_json"]
        self.assertEqual(raw["dedupe_dropped_exact_row_count"], 1)

    def test_entry_timing_pullback_from_recent_high(self) -> None:
        self.assertAlmostEqual(pullback_from_recent_high(9.9, 10.0) or 0.0, -1.0)
        self.assertIsNone(pullback_from_recent_high(10.0, None))

    def test_entry_timing_chasing_score(self) -> None:
        score = entry_chasing_score(
            pullback_5m_pct=-0.1,
            momentum_3m_pct=2.5,
            entry_vs_prev_high_pct=-0.05,
            green_count_3m=2,
            min_after_5m_pct=-1.2,
        )
        self.assertEqual(score, 100.0)
        self.assertEqual(entry_chasing_score(pullback_5m_pct=-2.0, momentum_3m_pct=0.0, entry_vs_prev_high_pct=-1.0, green_count_3m=0, min_after_5m_pct=0.0), 0.0)

    def test_entry_timing_pullback_simulation_enter_and_no_enter(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-06-26T14:00:00Z"), "open": 10.0, "high": 10.2, "low": 10.0, "close": 10.1, "volume": 100},
            {"timestamp": pd.Timestamp("2026-06-26T14:02:00Z"), "open": 10.1, "high": 10.3, "low": 9.9, "close": 10.2, "volume": 100},
            {"timestamp": pd.Timestamp("2026-06-26T14:05:00Z"), "open": 10.2, "high": 10.6, "low": 10.1, "close": 10.5, "volume": 100},
        ])
        hit = simulate_pullback_entry(candles, original_entry_time=pd.Timestamp("2026-06-26T14:00:00Z"), original_entry_price=10.0, exit_time=None, pullback_pct=1.0)
        miss = simulate_pullback_entry(candles, original_entry_time=pd.Timestamp("2026-06-26T14:00:00Z"), original_entry_price=10.0, exit_time=None, pullback_pct=2.0)
        self.assertEqual(hit["would_enter"], 1)
        self.assertEqual(hit["entry_price"], 9.9)
        self.assertEqual(miss["would_enter"], 0)

    def test_missed_runner_no_signal_classification_should_have_signaled(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-06-26T13:30:00Z"), "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100},
            {"timestamp": pd.Timestamp("2026-06-26T13:34:00Z"), "open": 10.1, "high": 10.7, "low": 10.0, "close": 10.6, "volume": 100},
            {"timestamp": pd.Timestamp("2026-06-26T13:44:00Z"), "open": 10.6, "high": 11.2, "low": 10.5, "close": 11.0, "volume": 100},
            {"timestamp": pd.Timestamp("2026-06-26T13:46:00Z"), "open": 11.0, "high": 11.5, "low": 10.9, "close": 11.4, "volume": 100},
        ])
        diag = no_signal_diagnostics(candles, min_first_5m_high_pct=0.5, min_first_15m_high_pct=1.0, min_or_range_pct=0.5)
        self.assertEqual(diag["top100_no_signal_reason"], "should_have_signaled")
        self.assertEqual(diag["had_required_first5"], 1)
        self.assertEqual(diag["had_required_first15"], 1)

    def test_missed_runner_no_signal_classification_failed_first5(self) -> None:
        candles = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-06-26T13:30:00Z"), "open": 10.0, "high": 10.01, "low": 9.9, "close": 10.0, "volume": 100},
            {"timestamp": pd.Timestamp("2026-06-26T13:46:00Z"), "open": 10.0, "high": 11.0, "low": 9.9, "close": 10.8, "volume": 100},
        ])
        diag = no_signal_diagnostics(candles, min_first_5m_high_pct=0.5, min_first_15m_high_pct=1.0, min_or_range_pct=0.5)
        self.assertEqual(diag["top100_no_signal_reason"], "failed_first5")

    def test_missed_runners_cli_has_performance_flags(self) -> None:
        help_text = build_missed_runners_parser().format_help()
        self.assertIn("--max-symbols", help_text)
        self.assertIn("--force", help_text)

    def test_should_have_signaled_investigator_parser_and_summary(self) -> None:
        help_text = build_shs_investigator_parser().format_help()
        self.assertIn("--max-cases", help_text)
        self.assertIn("--start-date", help_text)
        cases = pd.DataFrame([
            {"final_classification": "runtime_signal_ready_but_no_buy"},
            {"final_classification": "risk_guard_blocked"},
            {"final_classification": "runtime_never_processed_symbol"},
        ])
        summary = summary_for_cases(cases, "2026-06-26").iloc[0]
        self.assertEqual(int(summary["total_should_have_signaled"]), 3)
        self.assertEqual(int(summary["runtime_signal_ready_but_no_buy"]), 1)
        self.assertEqual(int(summary["risk_guard_blocked"]), 1)
        self.assertEqual(int(summary["runtime_never_processed_symbol"]), 1)

    def test_no_buy_after_signal_parser_and_summary(self) -> None:
        help_text = build_no_buy_after_signal_parser().format_help()
        self.assertIn("--max-cases", help_text)
        self.assertIn("--start-date", help_text)
        cases = pd.DataFrame([
            {"final_no_buy_reason": "unexplained_after_signal_before_dispatch"},
            {"final_no_buy_reason": "entries_blocked"},
            {"final_no_buy_reason": "lower_rank_candidate_not_selected"},
        ])
        summary = no_buy_after_signal_summary_for_cases(cases, "2026-06-26").iloc[0]
        self.assertEqual(int(summary["total_runtime_signal_ready_but_no_buy"]), 3)
        self.assertEqual(int(summary["unexplained_after_signal_before_dispatch"]), 1)
        self.assertEqual(int(summary["entries_blocked"]), 1)
        self.assertEqual(int(summary["lower_rank_candidate_not_selected"]), 1)

    def test_no_buy_after_signal_classification_priority(self) -> None:
        self.assertEqual(
            classify_no_buy_reason(
                {
                    "entries_blocked_at_scan": 1,
                    "failed_final_entry_filter_reason": "",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 0,
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                }
            ),
            "entries_blocked",
        )
        self.assertEqual(
            classify_no_buy_reason(
                {
                    "entries_blocked_at_scan": 0,
                    "failed_final_entry_filter_reason": "spread_too_wide",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 0,
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                }
            ),
            "final_filter_failed_spread",
        )
        self.assertEqual(
            classify_no_buy_reason(
                {
                    "entries_blocked_at_scan": 0,
                    "failed_final_entry_filter_reason": "",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 2,
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                }
            ),
            "lower_rank_candidate_not_selected",
        )

    def test_no_buy_after_signal_lifecycle_traces_dispatch_gap_after_signal_ready(self) -> None:
        trace = lifecycle_stage_trace(
            symbol="OMER",
            signal_ready_seen=True,
            dispatch=0,
            ack=0,
            skip_reason="",
            failed_reason="",
            entries_blocked=0,
            max_entries_reached=0,
            better=[],
            same_attempt_confirmed=1,
        )
        self.assertEqual(trace["ready_list_stage"], "seen_in_entry_candidates_inferred_from_SIGNAL_READY")
        self.assertEqual(trace["ranking_stage"], "ranked_in_ordered_entry_candidates_inferred_from_SIGNAL_READY")
        self.assertEqual(trace["selection_stage"], "selected_for_entry_evaluation_SIGNAL_READY")
        self.assertEqual(trace["candidate_disappeared_stage"], "unexplained_after_SIGNAL_READY_before_dispatch_attempt")
        self.assertIn("SIGNAL_READY", trace["candidate_lifecycle_trace"])

    def test_no_buy_after_signal_detects_post_signal_stale_skip(self) -> None:
        terminal = post_signal_terminal_evidence(
            "2026-06-26T14:02:42+00:00 STALE_OR_BACKFILL_READY_SKIPPED symbol=AOUT reason=signal_before_last_unblock",
            "AOUT",
        )
        self.assertEqual(terminal["post_signal_terminal_event"], "STALE_OR_BACKFILL_READY_SKIPPED")
        self.assertEqual(terminal["stale_or_backfill_reason"], "signal_before_last_unblock")
        self.assertEqual(terminal["post_signal_continue_detected"], 1)
        self.assertEqual(
            classify_no_buy_reason(
                {
                    **terminal,
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                    "entries_blocked_at_scan": 0,
                    "failed_final_entry_filter_reason": "",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 0,
                    "same_attempt_match_confirmed": 1,
                }
            ),
            "post_signal_stale_or_backfill_skip",
        )

    def test_no_buy_after_signal_detects_post_signal_already_open_skip(self) -> None:
        terminal = post_signal_terminal_evidence(
            "2026-06-26T14:03:00+00:00 ENTRY_REJECTED symbol=OMER reason=already_open_position",
            "OMER",
        )
        self.assertEqual(terminal["post_signal_terminal_event"], "already_open_position")
        self.assertEqual(terminal["already_open_after_signal"], 1)
        self.assertEqual(
            classify_no_buy_reason(
                {
                    **terminal,
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                    "entries_blocked_at_scan": 0,
                    "failed_final_entry_filter_reason": "",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 0,
                    "same_attempt_match_confirmed": 1,
                }
            ),
            "post_signal_already_open_skip",
        )

    def test_no_buy_after_signal_unexplained_only_after_known_paths_excluded(self) -> None:
        self.assertEqual(
            classify_no_buy_reason(
                {
                    "post_signal_terminal_event": "",
                    "post_signal_terminal_reason": "",
                    "already_open_after_signal": 0,
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                    "entries_blocked_at_scan": 0,
                    "failed_final_entry_filter_reason": "",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 0,
                    "same_attempt_match_confirmed": 1,
                }
            ),
            "unexplained_after_signal_before_dispatch",
        )

    def test_no_buy_after_signal_dispatch_attempt_is_not_unexplained(self) -> None:
        self.assertEqual(
            classify_no_buy_reason(
                {
                    "post_signal_terminal_event": "",
                    "post_signal_terminal_reason": "",
                    "already_open_after_signal": 0,
                    "order_dispatch_attempted": 1,
                    "signal_ready_time": "2026-06-26T14:00:00Z",
                    "entries_blocked_at_scan": 0,
                    "failed_final_entry_filter_reason": "",
                    "order_dispatch_skip_reason": "",
                    "max_entries_per_scan_reached": 0,
                    "candidates_ahead_count": 0,
                }
            ),
            "unknown_no_buy_after_signal",
        )

    def test_no_buy_after_signal_selects_second_signal_when_second_matches_target(self) -> None:
        timeline = [
            {"time": "2026-07-09T14:00:00+00:00", "event": "SIGNAL_READY", "reason": "", "details": "symbol=ABCD", "symbol": "ABCD"},
            {"time": "2026-07-09T14:05:00+00:00", "event": "SIGNAL_READY", "reason": "", "details": "symbol=ABCD", "symbol": "ABCD"},
        ]
        selected, candidates, reason = select_signal_ready_time(
            timeline,
            [],
            "ABCD",
            pd.Timestamp("2026-07-09T14:04:59Z"),
        )
        self.assertEqual(str(selected), "2026-07-09 14:05:00+00:00")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(reason, "first_signal_ready_at_or_after_target_time")

    def test_no_buy_after_signal_first_signal_blocked_second_dispatches(self) -> None:
        start = pd.Timestamp("2026-07-09T14:00:00Z")
        second = pd.Timestamp("2026-07-09T14:01:00Z")
        records = [
            {"time": start, "text": "2026-07-09T14:00:00Z SIGNAL_READY symbol=ABCD"},
            {"time": pd.Timestamp("2026-07-09T14:00:01Z"), "text": "2026-07-09T14:00:01Z RISK_GUARD_BLOCK_ENTRY symbol=ABCD reason=max_positions"},
            {"time": second, "text": "2026-07-09T14:01:00Z SIGNAL_READY symbol=ABCD"},
            {"time": pd.Timestamp("2026-07-09T14:01:01Z"), "text": "2026-07-09T14:01:01Z ENTRY_ORDER_DISPATCH_ATTEMPT symbol=ABCD"},
        ]
        first_window = records_in_window(records, start, second)
        self.assertIsNone(first_record_time(first_window, "ABCD", ["ENTRY_ORDER_DISPATCH_ATTEMPT"]))
        self.assertIsNotNone(first_record_time(first_window, "ABCD", ["RISK_GUARD_BLOCK_ENTRY"]))
        second_window = records_in_window(records, second, second + pd.Timedelta(minutes=2))
        self.assertIsNotNone(first_record_time(second_window, "ABCD", ["ENTRY_ORDER_DISPATCH_ATTEMPT"]))

    def test_no_buy_after_signal_duplicate_signal_ready_rows_same_timestamp_are_one_attempt(self) -> None:
        timeline = [
            {"time": "2026-07-09T14:00:00.123456+00:00", "event": "SIGNAL_READY", "reason": "", "details": "symbol=ABCD", "symbol": "ABCD"},
            {"time": "2026-07-09T14:00:00.123456+00:00", "event": "SIGNAL_READY", "reason": "", "details": "symbol=ABCD duplicate", "symbol": "ABCD"},
        ]
        selected, candidates, _reason = select_signal_ready_time(
            timeline,
            [],
            "ABCD",
            pd.Timestamp("2026-07-09T14:00:00.123456Z"),
        )
        same_ts_count = sum(1 for ts in candidates if selected is not None and abs((ts - selected).total_seconds()) <= 0.001)
        self.assertEqual(same_ts_count, 1)

    def test_no_buy_after_signal_microsecond_and_timezone_equivalent_selection(self) -> None:
        timeline = [
            {"time": "2026-07-09T10:00:00.500000-04:00", "event": "SIGNAL_READY", "reason": "", "details": "symbol=ABCD", "symbol": "ABCD"},
        ]
        selected, _candidates, _reason = select_signal_ready_time(
            timeline,
            [],
            "ABCD",
            pd.Timestamp("2026-07-09T14:00:00.499999Z"),
        )
        self.assertEqual(str(selected), "2026-07-09 14:00:00.500000+00:00")

    def test_no_buy_after_signal_event_after_next_signal_not_matched_to_prior_attempt(self) -> None:
        first = pd.Timestamp("2026-07-09T14:00:00Z")
        second = pd.Timestamp("2026-07-09T14:00:30Z")
        records = [
            {"time": first, "text": "2026-07-09T14:00:00Z SIGNAL_READY symbol=ABCD"},
            {"time": second, "text": "2026-07-09T14:00:30Z SIGNAL_READY symbol=ABCD"},
            {"time": pd.Timestamp("2026-07-09T14:00:31Z"), "text": "2026-07-09T14:00:31Z ENTRY_ORDER_DISPATCH_ATTEMPT symbol=ABCD"},
        ]
        first_window = records_in_window(records, first, second)
        self.assertIsNone(first_record_time(first_window, "ABCD", ["ENTRY_ORDER_DISPATCH_ATTEMPT"]))

    def test_no_buy_after_signal_ambiguous_correlation_not_unexplained_bug(self) -> None:
        self.assertEqual(
            classify_no_buy_reason(
                {
                    "correlation_issue_reason": "missing_observed_signal_ready_event",
                    "order_dispatch_attempted": 0,
                    "signal_ready_time": "2026-07-09T14:00:00Z",
                }
            ),
            "ambiguous_event_correlation",
        )

    def test_strategy_coverage_runner_rows_uses_default_signal_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            session = "2026-06-26"
            symbol_dir = history / "session_type=RTH" / "symbol=AAA" / "year=2026" / "month=06"
            symbol_dir.mkdir(parents=True)
            pd.DataFrame([
                {"timestamp": "2026-06-26T13:30:00Z", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100},
                {"timestamp": "2026-06-26T13:34:00Z", "open": 10.1, "high": 10.7, "low": 10.0, "close": 10.6, "volume": 100},
                {"timestamp": "2026-06-26T13:44:00Z", "open": 10.6, "high": 11.2, "low": 10.5, "close": 11.0, "volume": 100},
                {"timestamp": "2026-06-26T13:46:00Z", "open": 11.0, "high": 11.5, "low": 10.9, "close": 11.4, "volume": 100},
            ]).to_parquet(symbol_dir / "day=26.parquet")
            universe = root / "universe.csv"
            top100 = root / "top100.csv"
            sqlite_path = root / "runtime.sqlite"
            pd.DataFrame([{"symbol": "AAA"}]).to_csv(universe, index=False)
            pd.DataFrame([{"symbol": "AAA", "top100_rank": 1, "top100_score": 90.0}]).to_csv(top100, index=False)
            rows, diagnostics = build_runner_rows(
                session_date=session,
                history_dir=history,
                universe_path=universe,
                top100_path=top100,
                sqlite_path=sqlite_path,
            )
            self.assertEqual(diagnostics["processed_symbols"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.iloc[0]["top100_no_signal_reason"], "should_have_signaled")

    def test_entry_timing_cli_help(self) -> None:
        help_text = build_entry_timing_parser().format_help()
        self.assertIn("Analyze whether live entries chase spikes", help_text)
        self.assertIn("--date", help_text)

    def test_signal_replay_filters_should_have_signaled_targets(self) -> None:
        missed = pd.DataFrame([
            {"symbol": "AAA", "source_bucket": "top100", "was_bought": 0, "top100_no_signal_reason": "should_have_signaled"},
            {"symbol": "BBB", "source_bucket": "top100", "was_bought": 1, "top100_no_signal_reason": "should_have_signaled"},
            {"symbol": "CCC", "source_bucket": "outside_top100", "was_bought": 0, "top100_no_signal_reason": "should_have_signaled"},
            {"symbol": "DDD", "source_bucket": "top100", "was_bought": 0, "top100_no_signal_reason": "failed_first5"},
        ])
        out = filter_should_have_signaled_targets(missed)
        self.assertEqual(out["symbol"].tolist(), ["AAA"])

    def test_signal_replay_timeline_merge_sort(self) -> None:
        events = [
            {"time": "2026-06-26T14:05:00+00:00", "source": "orders", "event": "ORDER_SUBMITTED"},
            {"time": "2026-06-26T14:01:00+00:00", "source": "candle", "event": "possible_signal"},
        ]
        merged = merge_timeline_events(events)
        self.assertEqual([event["event"] for event in merged], ["possible_signal", "ORDER_SUBMITTED"])

    def test_signal_replay_classifies_risk_guard(self) -> None:
        reason = classify_replay_reason(
            [{"event": "RISK_GUARD_BLOCK_ENTRY", "reason": "daily_loss", "details": ""}],
            {"trades_count": 0, "executions_count": 0},
        )
        self.assertEqual(reason, "risk_guard_blocked")

    def test_signal_replay_classifies_max_positions(self) -> None:
        reason = classify_replay_reason(
            [{"event": "BUY_BLOCKED", "reason": "max_positions", "details": ""}],
            {"trades_count": 0, "executions_count": 0},
        )
        self.assertEqual(reason, "max_positions_blocked")

    def test_signal_replay_classifies_signal_ready_no_buy(self) -> None:
        reason = classify_replay_reason(
            [{"event": "SIGNAL_READY", "reason": "", "details": ""}],
            {"trades_count": 0, "executions_count": 0},
        )
        self.assertEqual(reason, "signal_ready_but_no_buy_attempt")

    def test_signal_replay_classifies_no_runtime_evidence(self) -> None:
        self.assertEqual(classify_replay_reason([], {"trades_count": 0, "executions_count": 0}), "no_runtime_evidence")

    def test_signal_replay_cli_help(self) -> None:
        help_text = build_signal_replay_parser().format_help()
        self.assertIn("Replay should-have-signaled", help_text)
        self.assertIn("--missed-runners-csv", help_text)

    def test_signal_case_trace_decision_offline_not_ready(self) -> None:
        self.assertEqual(decision_classification(False, [], {}), "offline_signal_not_ready")

    def test_signal_case_trace_decision_risk_guard(self) -> None:
        decision = decision_classification(
            True,
            [{"event": "RISK_GUARD_BLOCK_ENTRY", "reason": "daily_loss", "details": ""}],
            {"trades_count": 0, "executions_count": 0},
        )
        self.assertEqual(decision, "buy_blocked_risk_guard")

    def test_signal_case_trace_cli_help(self) -> None:
        help_text = build_signal_case_trace_parser().format_help()
        self.assertIn("Trace one should-have-signaled symbol", help_text)
        self.assertIn("--symbol", help_text)

    def test_signal_case_trace_pass_fail(self) -> None:
        self.assertEqual(pass_fail(True), "PASS")
        self.assertEqual(pass_fail(False), "FAIL")

    def test_buy_decision_trace_cli_help(self) -> None:
        help_text = build_buy_decision_trace_parser().format_help()
        self.assertIn("Trace why a should-have-signaled", help_text)
        self.assertIn("--symbol", help_text)

    def test_buy_decision_trace_parses_heartbeat_block_reason(self) -> None:
        state = heartbeat_block_state(
            "2026-06-26T15:02:00+00:00 heartbeat entries_blocked=1 "
            "risk_guard_block=1 risk_guard_reason=max_single_position managed_open=62"
        )
        self.assertEqual(state["entries_blocked"], 1)
        self.assertEqual(state["risk_guard_block"], 1)
        self.assertEqual(state["risk_guard_reason"], "max_single_position")
        self.assertIn("risk_guard_reason:max_single_position", state["active_reasons"])

    def test_buy_decision_trace_global_risk_guard_is_not_symbol_verdict(self) -> None:
        verdict = classify_buy_decision_verdict(
            offline_ready=True,
            center=pd.Timestamp("2026-06-26T15:02:00Z"),
            last_unblock=pd.Timestamp("2026-06-26T14:02:39Z"),
            block_state={
                "entries_blocked": 1,
                "active_reasons": ["risk_guard_reason:max_single_position"],
                "risk_guard_block": 1,
                "risk_guard_reason": "max_single_position",
            },
            runtime_events=[],
            ready_candidate_seen=False,
            competing_buys=[{"symbol": "ABSI"}],
        )
        self.assertEqual(verdict, "runtime_never_processed_symbol")

    def test_buy_decision_trace_classifies_symbol_specific_risk_guard(self) -> None:
        verdict = classify_buy_decision_verdict(
            offline_ready=True,
            center=pd.Timestamp("2026-06-26T15:02:00Z"),
            last_unblock=pd.Timestamp("2026-06-26T14:02:39Z"),
            block_state={"entries_blocked": 1, "active_reasons": [], "risk_guard_block": 1},
            runtime_events=[{"event": "RISK_GUARD_BLOCK_ENTRY", "reason": "max_single_position", "details": "symbol=OMER"}],
            ready_candidate_seen=True,
            competing_buys=[],
        )
        self.assertEqual(verdict, "missed_due_to_risk_guard")

    def test_buy_decision_trace_classifies_missing_ready_candidate(self) -> None:
        verdict = classify_buy_decision_verdict(
            offline_ready=True,
            center=pd.Timestamp("2026-06-26T15:02:00Z"),
            last_unblock=pd.Timestamp("2026-06-26T14:02:39Z"),
            block_state={"entries_blocked": 0, "active_reasons": [], "risk_guard_block": 0},
            runtime_events=[{"event": "HEARTBEAT_CONTEXT", "reason": "", "details": ""}],
            ready_candidate_seen=False,
            competing_buys=[{"symbol": "ABSI"}],
        )
        self.assertEqual(verdict, "missed_due_to_not_in_ready_candidates")

    def test_subscription_inspector_symbol_journal_lines_exact_symbol(self) -> None:
        lines = [
            "2026-06-26T13:31:00+00:00 TOP100_RELOAD_REQUESTED symbol=AOUT conId=1",
            "2026-06-26T13:31:00+00:00 TOP100_RELOAD_REQUESTED symbol=AOUTX conId=2",
        ]
        self.assertEqual(len(symbol_journal_lines(lines, "AOUT")), 1)

    def test_subscription_inspector_parse_key_values(self) -> None:
        parsed = parse_key_values("TOP100_RELOAD_DONE subscribed_top100=86 active_position_symbols_count=14 max_subscriptions=100")
        self.assertEqual(parsed["subscribed_top100"], "86")
        self.assertEqual(parsed["max_subscriptions"], "100")

    def test_subscription_inspector_infers_subscription_cap_root_cause(self) -> None:
        verdict = infer_verdict(
            in_top100=True,
            journal_symbol=[],
            journal_terms=["TOP100_RELOAD_DONE top100_requested=100 subscribed_top100=86 max_subscriptions=100"],
            contract_rows=0,
            candles_rows=0,
            sqlite_count_total=0,
            center_lines=[],
            appears_anywhere_in_journal=False,
        )
        self.assertEqual(verdict["likely_root_cause"], "subscription_cap_or_not_subscribed")

    def test_subscription_inspector_extracts_last_unblock_and_stale_symbols(self) -> None:
        lines = [
            "2026-06-26T14:02:40+00:00 heartbeat last_restart_unblock_time=2026-06-26T14:02:39.831068+00:00",
            "2026-06-26T14:02:42+00:00 STALE_OR_BACKFILL_READY_SKIPPED symbol=AOUT reason=signal_before_last_unblock",
            "2026-06-26T14:02:43+00:00 STALE_OR_BACKFILL_READY_SKIPPED symbol=COAG reason=signal_before_last_unblock",
        ]
        self.assertEqual(
            str(extract_last_restart_unblock_time(lines)),
            "2026-06-26 14:02:39.831068+00:00",
        )
        self.assertEqual(stale_or_backfill_skip_symbols(lines), ["AOUT", "COAG"])

    def test_subscription_inspector_signal_before_last_unblock_root_cause(self) -> None:
        verdict = infer_verdict(
            in_top100=True,
            journal_symbol=[],
            journal_terms=[],
            contract_rows=0,
            candles_rows=0,
            sqlite_count_total=0,
            center_lines=["2026-06-26T13:55:00+00:00 heartbeat top100_block=1 entries_blocked=1"],
            possible_signal_ts=pd.Timestamp("2026-06-26T13:55:00Z"),
            last_restart_unblock_ts=pd.Timestamp("2026-06-26T14:02:39Z"),
            appears_anywhere_in_journal=False,
        )
        self.assertEqual(verdict["likely_root_cause"], "signal_before_last_unblock")
        self.assertEqual(verdict["signal_before_last_unblock"], 1)

    def test_subscription_inspector_infers_subscribed_but_no_market_data(self) -> None:
        verdict = infer_verdict(
            in_top100=True,
            journal_symbol=[
                "TOP100_RELOAD_REQUESTED symbol=AOUT conId=1",
                "TOP100_RELOAD_SUBSCRIBED symbol=AOUT conId=1",
            ],
            journal_terms=[],
            contract_rows=1,
            candles_rows=0,
            sqlite_count_total=0,
            center_lines=[],
        )
        self.assertEqual(verdict["likely_root_cause"], "subscribed_but_no_market_data_seen")

    def test_subscription_inspector_cli_help(self) -> None:
        help_text = build_subscription_inspector_parser().format_help()
        self.assertIn("Inspect whether one or more Top100 symbols", help_text)
        self.assertIn("--symbols", help_text)

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
        self.assertEqual(score_bucket(50), "hold_candidate")

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
