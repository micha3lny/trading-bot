from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.live_trading.v62_live_data_recorder import (
    FillEvent,
    LiveCandle1m,
    LiveDataRecorder,
    OrderIntent,
    PortfolioSnapshot,
    SelectionEvent,
    SignalSnapshot,
    append_csv_row,
    resolved_record_session_date,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class RecorderSessionRotationTests(unittest.TestCase):
    def test_explicit_session_date_takes_precedence_over_top100_source_date(self) -> None:
        row = {
            "symbol": "ADVB",
            "event": "PRE_SIGNAL_RUNTIME_SNAPSHOT",
            "recorded_at": "2026-07-21T13:45:01+00:00",
            "session_date": "2026-07-21",
            "raw_json": json.dumps({"top100_source_date": "2026-07-20"}),
        }
        self.assertEqual(resolved_record_session_date(row, fallback_session_date="2026-07-20"), "2026-07-21")

    def test_timestamp_session_ignores_top100_source_date(self) -> None:
        row = {
            "symbol": "ADVB",
            "event": "PRE_SIGNAL_RUNTIME_SNAPSHOT",
            "recorded_at": "2026-07-21T13:45:01+00:00",
            "raw_json": json.dumps({"top100_source_date": "2026-07-20"}),
        }
        self.assertEqual(resolved_record_session_date(row, fallback_session_date="2026-07-20"), "2026-07-21")

    def test_trade_lifecycle_path_rotates_on_next_session_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-07-20")
            first = {"recorded_at": "2026-07-20T13:45:00+00:00", "event": "PRE_SIGNAL_RUNTIME_SNAPSHOT", "symbol": "AAA", "raw_json": ""}
            append_csv_row(recorder.path("trade_lifecycle.csv", row=first, event_type="PRE_SIGNAL_RUNTIME_SNAPSHOT", symbol="AAA"), first, list(first.keys()))

            second = {
                "recorded_at": "2026-07-21T13:45:01+00:00",
                "event": "PRE_SIGNAL_RUNTIME_SNAPSHOT",
                "symbol": "ADVB",
                "raw_json": json.dumps({"top100_source_date": "2026-07-20"}),
            }
            append_csv_row(recorder.path("trade_lifecycle.csv", row=second, event_type="PRE_SIGNAL_RUNTIME_SNAPSHOT", symbol="ADVB"), second, list(second.keys()))

            old_rows = read_csv(Path(tmp) / "2026-07-20" / "trade_lifecycle.csv")
            new_rows = read_csv(Path(tmp) / "2026-07-21" / "trade_lifecycle.csv")
            self.assertEqual([row["symbol"] for row in old_rows], ["AAA"])
            self.assertEqual([row["symbol"] for row in new_rows], ["ADVB"])
            self.assertEqual(recorder.session_date, "2026-07-21")

    def test_cached_writer_target_is_replaced_for_standard_record_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-07-20")
            recorder.record_signal(SignalSnapshot(symbol="AAA", signal_name="x", action="BUY", recorded_at="2026-07-20T13:40:00+00:00"))
            recorder.record_signal(SignalSnapshot(symbol="ADVB", signal_name="x", action="BUY", recorded_at="2026-07-21T13:45:00+00:00", features_json={"top100_source_date": "2026-07-20"}))
            self.assertEqual([r["symbol"] for r in read_csv(Path(tmp) / "2026-07-20" / "signal_snapshots.csv")], ["AAA"])
            self.assertEqual([r["symbol"] for r in read_csv(Path(tmp) / "2026-07-21" / "signal_snapshots.csv")], ["ADVB"])

    def test_candle_batches_are_split_by_resolved_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-07-20")
            recorder.record_candles_1m([
                LiveCandle1m(symbol="AAA", bar_time="2026-07-20T13:30:00+00:00"),
                LiveCandle1m(symbol="ADVB", bar_time="2026-07-21T13:30:00+00:00"),
            ])
            self.assertEqual([r["symbol"] for r in read_csv(Path(tmp) / "2026-07-20" / "candles_1m.csv")], ["AAA"])
            self.assertEqual([r["symbol"] for r in read_csv(Path(tmp) / "2026-07-21" / "candles_1m.csv")], ["ADVB"])

    def test_representative_recorder_files_share_session_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-07-20")
            recorder.record_selection(SelectionEvent(symbol="ADVB", stage="scan", decision="accepted", recorded_at="2026-07-21T13:45:00+00:00"))
            recorder.record_order_intent(OrderIntent(symbol="ADVB", action="BUY", recorded_at="2026-07-21T13:45:01+00:00"))
            recorder.record_fill(FillEvent(execution_id="e1", symbol="ADVB", action="BUY", recorded_at="2026-07-21T13:45:02+00:00"))
            recorder.record_portfolio(PortfolioSnapshot(recorded_at="2026-07-21T13:45:03+00:00"))
            recorder.record_run_metadata({"recorded_at": "2026-07-21T13:45:04+00:00"})
            recorder.write_manifest()

            day21 = Path(tmp) / "2026-07-21"
            self.assertTrue((day21 / "selection_events.csv").exists())
            self.assertTrue((day21 / "order_intents.csv").exists())
            self.assertTrue((day21 / "fills.csv").exists())
            self.assertTrue((day21 / "portfolio_snapshots.csv").exists())
            self.assertTrue((day21 / "run_metadata.csv").exists())
            self.assertTrue((day21 / "manifest.json").exists())

    def test_audit_detects_next_day_row_in_prior_session_directory(self) -> None:
        from scripts.audit_recorder_session_consistency import audit_date

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "2026-07-20"
            session_dir.mkdir(parents=True)
            row = {
                "recorded_at": "2026-07-21T13:45:01+00:00",
                "event": "PRE_SIGNAL_RUNTIME_SNAPSHOT",
                "symbol": "ADVB",
                "raw_json": json.dumps({"top100_source_date": "2026-07-20", "session_date": "2026-07-21"}),
            }
            append_csv_row(session_dir / "trade_lifecycle.csv", row, list(row.keys()))
            mismatches = audit_date(session_dir)
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0]["symbol"], "ADVB")
            self.assertEqual(mismatches[0]["actual_session_date"], "2026-07-21")
            self.assertEqual(mismatches[0]["file"], "trade_lifecycle.csv")


if __name__ == "__main__":
    unittest.main()
