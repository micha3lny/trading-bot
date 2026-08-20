from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.causal_top100_resolver import (
    INVALID_TOP100_PARITY,
    TOP100_PARITY_OK,
    _light_generations_from_scanner,
    resolve_causal_top100,
    top100_transition_validation,
)
from src.live_trading.analysis.light_snapshot_scanner import BoundedLightScanner


SESSION = "2026-08-12"


def _write_top100(root: Path, source_date: str, symbols: list[str]) -> Path:
    path = root / f"daily_top100_{source_date}.csv"
    pd.DataFrame({"symbol": symbols, "rank": range(1, len(symbols) + 1), "score": range(100, 100 - len(symbols), -1)}).to_csv(path, index=False)
    return path


def _write_light(recorder: Path, rows: list[dict[str, object]]) -> None:
    path = recorder / SESSION / "top100_candidate_snapshots" / "light" / "part-test-000000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _scan_rows(
    scan_id: int,
    timestamp: str,
    source_date: str,
    symbols: list[str],
    *,
    process_start_id: str = "process-1",
) -> list[dict[str, object]]:
    return [
        {
            "session_date": SESSION,
            "process_start_id": process_start_id,
            "scan_id": scan_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "ranking_source_date": source_date,
            "in_runtime_top100": 1,
            "entry_symbol_allowed": 1,
        }
        for symbol in symbols
    ]


class CausalTop100ResolverTests(unittest.TestCase):
    def test_d_minus_one_runtime_wins_over_post_session_d_top100(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["DFNS", "ABCL"])
            _write_top100(universe, SESSION, ["BOXL", "ABCL"])
            _write_light(
                recorder,
                _scan_rows(1, f"{SESSION}T13:40:00Z", "2026-08-11", ["DFNS", "ABCL"])
                + _scan_rows(2, f"{SESSION}T21:50:00Z", SESSION, ["BOXL", "ABCL"]),
            )

            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)

            self.assertEqual(result.top100_runtime_parity, TOP100_PARITY_OK)
            self.assertEqual(result.analysis_top100_ranking_source_date, "2026-08-11")
            self.assertEqual(result.top100_generation_count, 2)
            self.assertTrue(result.membership_at("DFNS", f"{SESSION}T13:45:00Z"))
            self.assertTrue(result.membership_at("ABCL", f"{SESSION}T13:45:00Z"))
            self.assertFalse(result.membership_at("BOXL", f"{SESSION}T13:45:00Z"))
            self.assertTrue(result.membership_at("BOXL", f"{SESSION}T22:00:00Z"))
            with BoundedLightScanner(recorder, SESSION) as scanner:
                scanner.build_index()
                streamed = _light_generations_from_scanner(scanner)
            self.assertEqual(result.generations, streamed)

    def test_mixed_generations_validate_as_temporal_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["AAA", "COMMON"])
            _write_top100(universe, SESSION, ["BBB", "COMMON"])
            _write_light(
                recorder,
                _scan_rows(1, f"{SESSION}T13:40:00Z", "2026-08-11", ["AAA", "COMMON"])
                + _scan_rows(2, f"{SESSION}T21:50:00Z", SESSION, ["BBB", "COMMON"]),
            )
            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)
            validation = top100_transition_validation(result.generations, desired_new_symbols=["BBB", "COMMON"])
            self.assertEqual(validation["old_generation_count"], 2)
            self.assertEqual(validation["new_generation_count"], 2)
            self.assertEqual(validation["overlap_count"], 1)
            self.assertEqual(validation["removed_symbols"], ["AAA"])
            self.assertEqual(validation["added_symbols"], ["BBB"])
            self.assertEqual(validation["stale_old_symbols_after_reload"], [])
            self.assertEqual(validation["classification"], "temporal_union_only")

            stale = top100_transition_validation(result.generations, desired_new_symbols=["BBB"])
            self.assertEqual(stale["stale_old_symbols_after_reload"], ["COMMON"])
            self.assertEqual(stale["classification"], "runtime_top100_union_observed")

    def test_process_transition_reports_first_complete_scan_and_retained_old_symbols(self) -> None:
        process = "2026-08-12:2295282:1786571217747"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["AAA", "COMMON"])
            _write_top100(universe, SESSION, ["BBB", "COMMON"])
            rows = (
                _scan_rows(1, f"{SESSION}T21:49:00Z", "2026-08-11", ["AAA", "COMMON"], process_start_id=process)
                + _scan_rows(2, f"{SESSION}T21:50:00Z", SESSION, ["BBB"], process_start_id=process)
                + _scan_rows(3, f"{SESSION}T21:50:05Z", SESSION, ["BBB", "COMMON"], process_start_id=process)
            )
            rows.append({
                "session_date": SESSION,
                "process_start_id": process,
                "scan_id": 3,
                "timestamp": f"{SESSION}T21:50:05Z",
                "symbol": "AAA",
                "ranking_source_date": SESSION,
                "in_runtime_top100": 0,
                "entry_symbol_allowed": 0,
                "already_open": 1,
                "contract_present": 1,
                "ticker_present": 1,
            })
            _write_light(recorder, rows)

            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)
            validation = result.transition_validation(process)

            self.assertEqual(validation["old_generation_count"], 2)
            self.assertEqual(validation["new_generation_count"], 2)
            self.assertEqual(validation["first_observed_new_generation_count"], 1)
            self.assertEqual(validation["first_timestamp_of_new_generation"], f"{SESSION}T21:50:00+00:00")
            self.assertEqual(validation["first_complete_post_reload_scan"], "3")
            self.assertEqual(validation["first_complete_post_reload_timestamp"], f"{SESSION}T21:50:05+00:00")
            self.assertEqual(validation["stale_old_symbols_after_reload"], [])
            self.assertEqual(
                validation["old_symbols_retained_only_for_active_positions_or_subscriptions"],
                ["AAA"],
            )
            self.assertEqual(validation["classification"], "temporal_union_only")

    def test_entry_allowed_does_not_override_explicit_runtime_top100_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["CURRENT"])
            rows = _scan_rows(1, f"{SESSION}T13:40:00Z", "2026-08-11", ["CURRENT"])
            rows.append({
                "session_date": SESSION,
                "process_start_id": "process-1",
                "scan_id": 1,
                "timestamp": f"{SESSION}T13:40:00Z",
                "symbol": "OLD",
                "ranking_source_date": "2026-08-11",
                "in_runtime_top100": 0,
                "entry_symbol_allowed": 1,
            })
            _write_light(recorder, rows)

            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)

            self.assertTrue(result.membership_at("CURRENT", f"{SESSION}T13:45:00Z"))
            self.assertFalse(result.membership_at("OLD", f"{SESSION}T13:45:00Z"))
            self.assertIn("OLD", result.generations[0].entry_symbols)

    def test_removed_symbol_still_entry_allowed_is_reported_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["OLD", "COMMON"])
            _write_top100(universe, SESSION, ["NEW", "COMMON"])
            rows = (
                _scan_rows(1, f"{SESSION}T21:49:00Z", "2026-08-11", ["OLD", "COMMON"])
                + _scan_rows(2, f"{SESSION}T21:50:00Z", SESSION, ["NEW", "COMMON"])
            )
            rows.append({
                "session_date": SESSION,
                "process_start_id": "process-1",
                "scan_id": 2,
                "timestamp": f"{SESSION}T21:50:00Z",
                "symbol": "OLD",
                "ranking_source_date": SESSION,
                "in_runtime_top100": 0,
                "entry_symbol_allowed": 1,
            })
            _write_light(recorder, rows)

            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)
            validation = result.transition_validation("process-1")

            self.assertEqual(validation["stale_old_symbols_after_reload"], [])
            self.assertEqual(validation["stale_old_entry_symbols_after_reload"], ["OLD"])
            self.assertEqual(validation["classification"], "runtime_entry_universe_union_observed")

    def test_many_identical_scans_are_compressed_into_one_bounded_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["AAA", "BBB"])
            rows: list[dict[str, object]] = []
            for scan_id in range(1, 201):
                rows.extend(
                    _scan_rows(
                        scan_id,
                        f"{SESSION}T13:{30 + (scan_id // 60):02d}:{scan_id % 60:02d}Z",
                        "2026-08-11",
                        ["AAA", "BBB"],
                    )
                )
            _write_light(recorder, rows)

            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)

            self.assertEqual(len(result.generations), 1)
            self.assertEqual(result.generations[0].scan_count, 200)
            self.assertEqual(result.generations[0].symbols, frozenset({"AAA", "BBB"}))

    def test_same_day_and_latest_fallbacks_are_not_silently_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            same_day = _write_top100(universe, SESSION, ["BOXL"])
            latest = universe / "daily_top100_latest.csv"
            same_day.replace(latest)

            latest_result = resolve_causal_top100(
                session_date=SESSION, top100_dir=universe, recorder_dir=recorder, explicit_top100=latest
            )
            self.assertEqual(latest_result.top100_runtime_parity, INVALID_TOP100_PARITY)
            self.assertIn("latest_file_rejected", latest_result.top100_source_reason)

            same_day = _write_top100(universe, SESSION, ["BOXL"])
            same_day_result = resolve_causal_top100(
                session_date=SESSION, top100_dir=universe, recorder_dir=recorder, explicit_top100=same_day
            )
            self.assertEqual(same_day_result.top100_runtime_parity, INVALID_TOP100_PARITY)
            self.assertEqual(same_day_result.analysis_top100_ranking_source_date, SESSION)

    def test_previous_session_fallback_is_controlled_but_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            _write_top100(universe, "2026-08-11", ["AAA"])
            result = resolve_causal_top100(session_date=SESSION, top100_dir=universe, recorder_dir=recorder)
            self.assertEqual(result.top100_runtime_parity, INVALID_TOP100_PARITY)
            self.assertEqual(result.top100_source_reason, "controlled_previous_session_fallback_unverified")
            self.assertIsNone(result.membership_at("AAA", f"{SESSION}T13:45:00Z"))

    def test_exact_session_runtime_metadata_precedes_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe, recorder = root / "universe", root / "recorder"
            universe.mkdir()
            runtime_path = _write_top100(universe, "2026-08-11", ["RUNTIME"])
            explicit_path = _write_top100(universe, SESSION, ["EXPLICIT"])
            metadata = recorder / SESSION / "run_metadata.csv"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "session_date": SESSION,
                "metadata_json": '{"ranking_source_date":"2026-08-11"}',
            }]).to_csv(metadata, index=False)

            result = resolve_causal_top100(
                session_date=SESSION,
                top100_dir=universe,
                recorder_dir=recorder,
                explicit_top100=explicit_path,
            )

            self.assertEqual(result.top100_runtime_parity, TOP100_PARITY_OK)
            self.assertEqual(result.top100_source_path, runtime_path)
            self.assertEqual(result.top100_source_reason, "exact_session_runtime_metadata")


if __name__ == "__main__":
    unittest.main()
