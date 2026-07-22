from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.live_trading.analysis.shs_pipeline_root_cause import (
    EvidenceItem,
    _iter_recorder_rows,
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

    def test_no_symbol_specific_evidence_is_data_retention(self) -> None:
        root, reason, confidence = classify_root_cause({"symbol": "NUAI"}, [], True)
        self.assertEqual(root, "DATA_RETENTION_PREVENTS_ROOT_CAUSE")
        self.assertEqual(confidence, "medium")
        self.assertIn("no symbol-specific", reason)

    def test_contract_request_missing_after_runtime_watchlist(self) -> None:
        root, _, _ = classify_root_cause({"symbol": "NUAI"}, [item("TOP100_RELOAD", "TOP100_RELOAD_SUBSCRIBED symbol=NUAI")], True)
        self.assertEqual(root, "CONTRACT_REQUEST_MISSING")

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
            self.assertEqual(row["final_root_cause"], "DATA_RETENTION_PREVENTS_ROOT_CAUSE")
            self.assertTrue((root / "analysis" / "shs_root_cause_NUAI_2026-07-20.md").exists())


if __name__ == "__main__":
    unittest.main()
