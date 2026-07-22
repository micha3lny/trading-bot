from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.live_trading.analysis.shs_pipeline_root_cause import (
    EvidenceItem,
    _iter_recorder_rows,
    _iter_sqlite_rows,
    classify_root_cause,
    run_symbol,
    row_matches_session,
    stage_presence,
)


def item(event: str, payload: str, stage: str = "symbol_registration") -> EvidenceItem:
    return EvidenceItem(pd.Timestamp("2026-07-20T13:45:00Z"), "test", stage, event, "NUAI", payload)


class SHSPipelineRootCauseTests(unittest.TestCase):
    def test_no_daily_top100_is_top100_not_loaded_runtime(self) -> None:
        root, reason, confidence = classify_root_cause({}, [], False)
        self.assertEqual(root, "TOP100_NOT_LOADED_RUNTIME")
        self.assertEqual(confidence, "high")
        self.assertIn("Top100", reason)

    def test_no_symbol_specific_evidence_is_insufficient_telemetry(self) -> None:
        root, reason, confidence = classify_root_cause({"symbol": "NUAI"}, [], True)
        self.assertEqual(root, "INSUFFICIENT_SYMBOL_SPECIFIC_RUNTIME_TELEMETRY")
        self.assertEqual(confidence, "medium")
        self.assertIn("no symbol-specific", reason)

    def test_watchlist_only_is_insufficient_telemetry_not_contract_missing(self) -> None:
        root, reason, confidence = classify_root_cause({"symbol": "NUAI"}, [item("TOP100_RELOAD", "TOP100_RELOAD_START symbol=NUAI")], True)
        self.assertEqual(root, "INSUFFICIENT_TELEMETRY")
        self.assertEqual(confidence, "low")
        self.assertIn("not comprehensive", reason)

    def test_positive_contract_absence_is_contract_request_not_sent(self) -> None:
        root, _, confidence = classify_root_cause({"symbol": "NUAI"}, [item("PRE_SIGNAL_RUNTIME_SNAPSHOT", "symbol=NUAI contract_present=0 ticker_present=0 state_present=0")], True)
        self.assertEqual(root, "CONTRACT_REQUEST_NOT_SENT")
        self.assertEqual(confidence, "high")

    def test_downstream_subscription_without_contract_record_is_not_missing_proof(self) -> None:
        root, _, confidence = classify_root_cause({"symbol": "NUAI"}, [item("TOP100_RELOAD_SUBSCRIBED", "symbol=NUAI subscribed reqMktData")], True)
        self.assertEqual(root, "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED")
        self.assertEqual(confidence, "low")

    def test_no_ticker_after_subscription(self) -> None:
        evidence = [
            item("CONTRACT_METADATA", "symbol=NUAI conId=123 contract resolved"),
            item("TOP100_RELOAD_SUBSCRIBED", "symbol=NUAI subscribed reqMktData"),
        ]
        root, _, _ = classify_root_cause({"symbol": "NUAI"}, evidence, True)
        self.assertEqual(root, "NO_TICKER_RECEIVED")

    def test_state_missing_after_ticker(self) -> None:
        evidence = [
            item("CONTRACT_METADATA", "symbol=NUAI conId=123 contract resolved"),
            item("TOP100_RELOAD_SUBSCRIBED", "symbol=NUAI subscribed reqMktData"),
            item("TICKER", "symbol=NUAI price=10.0 ticker"),
        ]
        root, _, _ = classify_root_cause({"symbol": "NUAI"}, evidence, True)
        self.assertEqual(root, "SYMBOL_STATE_NOT_CREATED")

    def test_state_flags_mark_pre_signal_features(self) -> None:
        evidence = [item("PRE_SIGNAL_RUNTIME_SNAPSHOT", "symbol=NUAI state_present=1 first_5m_high_pct=5 first_15m_high_pct=7 or_range_pct=6 ranking_position=2")]
        stages = stage_presence(evidence)
        self.assertTrue(stages["candidate_state_creation"])
        self.assertTrue(stages["first5_first15_or_state"])
        self.assertTrue(stages["ranking_replacement"])

    def test_row_matches_session_ignores_top100_source_date(self) -> None:
        row = {
            "timestamp": "2026-07-21T13:45:00Z",
            "session_date": "2026-07-21",
            "symbol": "ADVB",
            "raw_json": json.dumps({"top100_source_date": "2026-07-20", "event_type": "ENTRY_SIGNAL"}),
        }
        self.assertFalse(row_matches_session(row, "2026-07-20"))
        self.assertTrue(row_matches_session(row, "2026-07-21"))

    def test_recorder_reader_rejects_next_day_event_in_previous_day_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "recorder"
            day = root / "2026-07-20"
            day.mkdir(parents=True)
            path = day / "trade_lifecycle.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "session_date", "symbol", "event_type", "raw_json"])
                writer.writeheader()
                writer.writerow({
                    "timestamp": "2026-07-21T13:45:00Z",
                    "session_date": "2026-07-21",
                    "symbol": "ADVB",
                    "event_type": "ENTRY_SIGNAL",
                    "raw_json": json.dumps({"top100_source_date": "2026-07-20"}),
                })
            self.assertEqual(_iter_recorder_rows(root, "2026-07-20", "ADVB"), [])


    def test_historical_market_data_sessions_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "runtime.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE market_data_sessions (date TEXT, symbol TEXT, collection_status TEXT, parquet_path TEXT, rows INTEGER)")
            conn.execute(
                "INSERT INTO market_data_sessions VALUES (?, ?, ?, ?, ?)",
                ("2026-02-10", "NUAI", "complete", "data/history/universe_1m/session_type=RTH/symbol=NUAI/year=2026/month=02/day=10.parquet", 390),
            )
            conn.commit()
            conn.close()
            self.assertEqual(_iter_sqlite_rows(db, "2026-07-20", "NUAI"), [])

    def test_runtime_event_counters_are_ignored_for_symbol_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "runtime.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE runtime_event_counters (date TEXT, event_type TEXT, symbol TEXT, reason TEXT, count INTEGER, first_seen_at TEXT, last_seen_at TEXT)")
            conn.execute(
                "INSERT INTO runtime_event_counters VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-07-20", "SIGNAL_READY", "NUAI", "global aggregate", 9, "2026-07-20T13:35:00Z", "2026-07-20T14:00:00Z"),
            )
            conn.commit()
            conn.close()
            self.assertEqual(_iter_sqlite_rows(db, "2026-07-20", "NUAI"), [])

    def test_sqlite_rows_for_other_date_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "runtime.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE runtime_events (session_date TEXT, event_time TEXT, event_type TEXT, symbol TEXT, raw_json TEXT)")
            conn.execute(
                "INSERT INTO runtime_events VALUES (?, ?, ?, ?, ?)",
                ("2026-07-21", "2026-07-21T13:45:00Z", "SIGNAL_READY", "NUAI", json.dumps({"symbol": "NUAI"})),
            )
            conn.commit()
            conn.close()
            self.assertEqual(_iter_sqlite_rows(db, "2026-07-20", "NUAI"), [])

    def test_sqlite_rows_for_other_symbol_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "runtime.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE runtime_events (session_date TEXT, event_time TEXT, event_type TEXT, symbol TEXT, raw_json TEXT)")
            conn.execute(
                "INSERT INTO runtime_events VALUES (?, ?, ?, ?, ?)",
                ("2026-07-20", "2026-07-20T13:45:00Z", "SIGNAL_READY", "IREN", json.dumps({"symbol": "IREN"})),
            )
            conn.commit()
            conn.close()
            self.assertEqual(_iter_sqlite_rows(db, "2026-07-20", "NUAI"), [])

    def test_missing_logs_do_not_fabricate_contract_root_cause(self) -> None:
        root, reason, confidence = classify_root_cause({"symbol": "NUAI"}, [], True)
        self.assertEqual(root, "INSUFFICIENT_SYMBOL_SPECIFIC_RUNTIME_TELEMETRY")
        self.assertEqual(confidence, "medium")
        self.assertNotIn("contract", reason.lower().split("proves", 1)[0])

    def test_run_symbol_writes_specific_root_cause_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = root / "daily_top100_2026-07-20.csv"
            pd.DataFrame([{"symbol": "NUAI", "top100_rank": 2, "top100_score": 98.0}]).to_csv(top, index=False)
            cases = root / "should_have_signaled_cases_2026-07-20.csv"
            pd.DataFrame([{"symbol": "NUAI", "possible_signal_time": "2026-07-20T13:45:00Z"}]).to_csv(cases, index=False)
            args = Namespace(
                date="2026-07-20",
                top100=top,
                cases_csv=cases,
                sqlite_path=root / "missing.sqlite",
                recorder_dir=root / "recorder",
                history_dir=root / "history",
                log_dir=root / "logs",
                output_dir=root / "analysis",
            )
            with patch("src.live_trading.analysis.shs_pipeline_root_cause.load_session_candles", return_value=pd.DataFrame()):
                row = run_symbol(args, "NUAI")
            self.assertEqual(row["symbol"], "NUAI")
            self.assertEqual(row["final_root_cause"], "INSUFFICIENT_SYMBOL_SPECIFIC_RUNTIME_TELEMETRY")
            self.assertEqual(row["contract_telemetry_assessment"], "INSUFFICIENT_TELEMETRY")
            self.assertTrue((root / "analysis" / "shs_root_cause_NUAI_2026-07-20.md").exists())


if __name__ == "__main__":
    unittest.main()
