from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.full_session_replay_v67 import profile_config
from src.live_trading.analysis.top100_buy_analyzer import (
    analyze_session,
    completed_bar_features,
    enrich_light_snapshots,
    portfolio_filter_simulation,
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
