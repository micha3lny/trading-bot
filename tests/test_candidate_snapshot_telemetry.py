from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from src.live_trading.candidate_snapshot_telemetry import (
    CandidateSnapshotBatch,
    CandidateSnapshotWriter,
    CandidateScanCollector,
    FULL_SCHEMA_VERSION,
    FullSnapshotTracker,
    LIGHT_SCHEMA_VERSION,
    export_snapshot_csv,
    feature_state_hash,
)


def light_row(symbol: str, scan_id: int, process_id: str = "p1", session_date: str = "2026-08-05") -> dict:
    return {
        "session_date": session_date,
        "trading_session_date": session_date,
        "timestamp": f"{session_date}T13:30:05+00:00",
        "process_start_id": process_id,
        "run_id": f"run:{process_id}",
        "scan_id": scan_id,
        "scan_uid": f"{process_id}:{scan_id}",
        "symbol": symbol,
        "source": "live_runtime",
        "expected_in_runtime_top100": 1,
    }


def full_row(symbol: str, scan_id: int, revision: int = 1, process_id: str = "p1") -> dict:
    return {
        **light_row(symbol, scan_id, process_id),
        "emit_reason": "new_completed_1m_bar",
        "candle_timestamp": "2026-08-05T13:30:00+00:00",
        "feature_state_revision": revision,
        "feature_state_hash": feature_state_hash((symbol, revision)),
    }


def batch(scan_id: int, symbols: list[str], *, process_id: str = "p1", full_rows: tuple[dict, ...] = ()) -> CandidateSnapshotBatch:
    return CandidateSnapshotBatch(
        session_date="2026-08-05",
        process_start_id=process_id,
        scan_id=scan_id,
        scan_uid=f"{process_id}:{scan_id}",
        expected_symbols=len(symbols),
        light_rows=tuple(light_row(symbol, scan_id, process_id) for symbol in symbols),
        full_rows=full_rows,
    )


class CandidateSnapshotTelemetryTests(unittest.TestCase):
    def test_writes_one_light_row_per_scan_and_symbol_with_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=2)
            self.assertTrue(writer.enqueue(batch(1, ["AAA", "BBB"])))
            writer.stop()

            path = next((Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "light").glob("*.parquet"))
            rows = pq.read_table(path).to_pylist()
            self.assertEqual({row["symbol"] for row in rows}, {"AAA", "BBB"})
            self.assertEqual({row["schema_version"] for row in rows}, {LIGHT_SCHEMA_VERSION})
            self.assertEqual({row["scan_uid"] for row in rows}, {"p1:1"})

    def test_full_dedupes_same_completed_candle_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            duplicate = full_row("AAA", 1)
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1_000)
            writer.enqueue(batch(1, ["AAA"], full_rows=(duplicate, duplicate)))
            writer.stop()

            path = next((Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "full").glob("*.parquet"))
            rows = pq.read_table(path).to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["schema_version"], FULL_SCHEMA_VERSION)

    def test_full_tracker_emits_state_once_then_waits_for_checkpoint_or_candle(self) -> None:
        tracker = FullSnapshotTracker()
        first = tracker.consider(
            "AAA", feature_values=(1, 2), initialization_flags=(1, 1, 1),
            completion_flags=(0, 0), now_monotonic=100.0, checkpoint_seconds=60.0,
        )
        unchanged = tracker.consider(
            "AAA", feature_values=(1, 2), initialization_flags=(1, 1, 1),
            completion_flags=(0, 0), now_monotonic=105.0, checkpoint_seconds=60.0,
        )
        changed_high_only = tracker.consider(
            "AAA", feature_values=(1, 3), initialization_flags=(1, 1, 1),
            completion_flags=(0, 0), now_monotonic=110.0, checkpoint_seconds=60.0,
        )
        completed = tracker.consider(
            "AAA", feature_values=(1, 3), initialization_flags=(1, 1, 1),
            completion_flags=(0, 0), candle_timestamp="2026-08-05T13:30:00Z",
            now_monotonic=115.0, checkpoint_seconds=60.0,
        )

        self.assertEqual(first["emit_reason"], "state_created")
        self.assertIsNone(unchanged)
        self.assertIsNone(changed_high_only)
        self.assertEqual(completed["emit_reason"], "new_completed_1m_bar")

    def test_completion_flag_change_emits_full_without_waiting_for_checkpoint(self) -> None:
        tracker = FullSnapshotTracker()
        tracker.consider(
            "AAA", feature_values=(0,), initialization_flags=(1, 1, 1),
            completion_flags=(0, 0), now_monotonic=100.0, checkpoint_seconds=60.0,
        )
        changed = tracker.consider(
            "AAA", feature_values=(1,), initialization_flags=(1, 1, 1),
            completion_flags=(1, 0), now_monotonic=101.0, checkpoint_seconds=60.0,
        )
        self.assertEqual(changed["emit_reason"], "completion_state_change")

    def test_completed_candle_emits_at_most_once_per_process_tracker(self) -> None:
        tracker = FullSnapshotTracker()
        first = tracker.consider(
            "AAA", feature_values=(1,), initialization_flags=(1, 1, 1),
            completion_flags=(1, 1), candle_timestamp="2026-08-05T13:30:00Z",
            now_monotonic=100.0, checkpoint_seconds=60.0,
        )
        duplicate = tracker.consider(
            "AAA", feature_values=(1,), initialization_flags=(1, 1, 1),
            completion_flags=(1, 1), candle_timestamp="2026-08-05T13:30:00Z",
            now_monotonic=101.0, checkpoint_seconds=60.0,
        )
        self.assertEqual(first["emit_reason"], "new_completed_1m_bar")
        self.assertIsNone(duplicate)

    def test_collector_preserves_rows_for_missing_pipeline_objects(self) -> None:
        collector = CandidateScanCollector(
            session_date="2026-08-05", run_id="run", process_start_id="p1",
            scan_id=7, scan_started_at="2026-08-05T13:30:00Z",
            expected_symbols=("NO_CONTRACT", "NO_TICKER", "NO_PRICE", "NO_STATE"),
        )
        collector.update("NO_CONTRACT", contract_present=0)
        collector.update("NO_TICKER", contract_present=1, ticker_present=0)
        collector.update("NO_PRICE", contract_present=1, ticker_present=1, usable_price=0)
        collector.update("NO_STATE", contract_present=1, ticker_present=1, usable_price=1, state_present=0)
        result = collector.finalize("2026-08-05T13:30:01Z", 10.0)

        self.assertEqual(len(result.light_rows), 4)
        self.assertEqual({row["symbol"] for row in result.light_rows}, {"NO_CONTRACT", "NO_TICKER", "NO_PRICE", "NO_STATE"})
        self.assertTrue(all(row["scan_uid"] == "p1:7" for row in result.light_rows))

    def test_queue_full_is_non_blocking_and_counts_whole_batch_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", queue_size=1, start_thread=False)
            self.assertTrue(writer.enqueue(batch(1, ["AAA", "BBB"])))
            started = time.perf_counter()
            self.assertFalse(writer.enqueue(batch(2, ["AAA", "BBB"])))
            elapsed = time.perf_counter() - started
            health = writer.health("2026-08-05")

            self.assertLess(elapsed, 0.05)
            self.assertEqual(health["dropped_scan_batches"], 1)
            self.assertEqual(health["dropped_rows"], 2)
            self.assertEqual(health["session_completeness"], "PARTIAL_QUEUE_DROP")

    def test_restart_process_ids_make_scan_uid_unique(self) -> None:
        first = batch(1, ["AAA"], process_id="process-a")
        second = batch(1, ["AAA"], process_id="process-b")
        self.assertNotEqual(first.scan_uid, second.scan_uid)
        self.assertNotEqual(first.light_rows[0]["scan_uid"], second.light_rows[0]["scan_uid"])

    def test_session_rollover_writes_each_batch_to_its_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1_000)
            writer.enqueue(batch(1, ["AAA"]))
            next_day_row = light_row("BBB", 2, session_date="2026-08-06")
            writer.enqueue(CandidateSnapshotBatch(
                session_date="2026-08-06",
                process_start_id="p1",
                scan_id=2,
                scan_uid="p1:2",
                expected_symbols=1,
                light_rows=(next_day_row,),
            ))
            writer.stop()

            self.assertTrue(list((Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "light").glob("*.parquet")))
            self.assertTrue(list((Path(tmp) / "2026-08-06" / "top100_candidate_snapshots" / "light").glob("*.parquet")))

    def test_manifest_reports_complete_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1_000)
            writer.enqueue(batch(1, ["AAA", "BBB"]))
            writer.enqueue(batch(2, ["AAA", "BBB"]))
            writer.stop()
            manifest = json.loads(
                (Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "candidate_snapshot_manifest.json").read_text()
            )
            self.assertEqual(manifest["written_scan_batches"], 2)
            self.assertEqual(manifest["written_rows"], 4)
            self.assertEqual(manifest["session_completeness"], "COMPLETE")
            self.assertEqual(manifest["min_written_symbols_per_scan"], 2)

    def test_manifest_handles_top100_size_change_without_false_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1_000)
            writer.enqueue(batch(1, ["AAA", "BBB"]))
            writer.enqueue(batch(2, ["AAA"]))
            writer.stop()
            manifest = json.loads(
                (Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "candidate_snapshot_manifest.json").read_text()
            )
            self.assertEqual(manifest["session_completeness"], "COMPLETE")
            self.assertEqual(manifest["written_symbols_by_scan_uid"], {"p1:1": 2, "p1:2": 1})

    def test_expected_but_unfinished_scan_is_persisted_as_partial_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1_000)
            writer.note_expected_batch(
                "2026-08-05", process_start_id="p1", scan_id=1,
                scan_uid="p1:1", expected_symbols=100,
            )
            writer.stop()
            manifest = json.loads(
                (Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "candidate_snapshot_manifest.json").read_text()
            )
            self.assertEqual(manifest["session_completeness"], "PARTIAL_MISSING_SYMBOL_ROWS")
            self.assertEqual(manifest["missing_scan_ranges"], {"p1": ["1"]})

    def test_writer_exception_is_contained_and_manifest_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1)
            with patch(
                "src.live_trading.candidate_snapshot_telemetry._prepare_parquet",
                side_effect=OSError("disk unavailable"),
            ):
                writer.enqueue(batch(1, ["AAA"]))
                writer.stop()
            health = writer.health("2026-08-05")
            self.assertGreaterEqual(health["writer_error_count"], 1)
            self.assertEqual(health["session_completeness"], "PARTIAL_WRITE_ERROR")

    def test_chunk_publish_is_all_or_none_when_full_prepare_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1)
            original_prepare = __import__(
                "src.live_trading.candidate_snapshot_telemetry",
                fromlist=["_prepare_parquet"],
            )._prepare_parquet
            calls = 0

            def fail_second_prepare(path, rows):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("full chunk unavailable")
                return original_prepare(path, rows)

            with patch(
                "src.live_trading.candidate_snapshot_telemetry._prepare_parquet",
                side_effect=fail_second_prepare,
            ):
                writer.enqueue(batch(1, ["AAA"], full_rows=(full_row("AAA", 1),)))
                writer.stop()

            snapshot_root = Path(tmp) / "2026-08-05" / "top100_candidate_snapshots"
            self.assertFalse(list(snapshot_root.rglob("*.parquet")))
            self.assertFalse(list(snapshot_root.rglob("*.tmp-*")))
            self.assertEqual(writer.health("2026-08-05")["session_completeness"], "PARTIAL_WRITE_ERROR")

    def test_exporter_dedupes_repeated_light_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", chunk_rows=1)
            duplicate_batch = batch(1, ["AAA"])
            writer.enqueue(duplicate_batch)
            writer.enqueue(duplicate_batch)
            writer.stop()

            output = Path(tmp) / "light.csv"
            result = export_snapshot_csv(tmp, "2026-08-05", "light", output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(result["rows_written"], 1)
            self.assertEqual(len(rows), 1)
            manifest = json.loads(
                (Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "candidate_snapshot_manifest.json").read_text()
            )
            self.assertEqual(manifest["written_scan_batches"], 1)

    def test_one_hundred_symbols_across_many_scans(self) -> None:
        symbols = [f"S{index:03d}" for index in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateSnapshotWriter(tmp, process_start_id="p1", queue_size=64, chunk_rows=500)
            for scan_id in range(1, 21):
                self.assertTrue(writer.enqueue(batch(scan_id, symbols)))
            writer.stop()
            paths = list((Path(tmp) / "2026-08-05" / "top100_candidate_snapshots" / "light").glob("*.parquet"))
            rows = sum(pq.read_table(path).num_rows for path in paths)
            self.assertEqual(rows, 2_000)
            self.assertFalse(list(Path(tmp).rglob("*.tmp-*")))


if __name__ == "__main__":
    unittest.main()
