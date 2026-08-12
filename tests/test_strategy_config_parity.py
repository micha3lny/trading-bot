from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.live_trading.analysis.strategy_config_parity import (
    load_session_thresholds,
    output_has_config_provenance,
    resolve_signal_thresholds,
    runtime_threshold_metadata,
)


class StrategyConfigParityTests(unittest.TestCase):
    def test_exact_session_run_metadata_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "2026-08-11" / "run_metadata.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["recorded_at", "session_date", "metadata_json"])
                writer.writeheader()
                writer.writerow({
                    "recorded_at": "2026-08-11T13:00:00Z",
                    "session_date": "2026-08-11",
                    "metadata_json": json.dumps({
                        "min_first_5m_high_pct": 0.5,
                        "min_first_15m_high_pct": 1.0,
                        "min_or_range_pct": 0.75,
                    }),
                })
            result = load_session_thresholds(root, "2026-08-11")
        self.assertEqual((result.min_first5, result.min_first15, result.min_or_range), (0.5, 1.0, 0.75))
        self.assertEqual(result.config_source, "run_metadata")

    def test_live_session_never_falls_back_to_low_threshold_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "2026-08-11" / "run_metadata.csv"
            path.parent.mkdir(parents=True)
            path.write_text(
                "recorded_at,session_date,metadata_json\n"
                '2026-08-11T12:00:00Z,2026-08-11,"{""min_first_5m_high_pct"": 0.5, '
                '""min_first_15m_high_pct"": 1.0, ""min_or_range_pct"": 0.5}"\n'
            )
            live = resolve_signal_thresholds(
                session_date="2026-08-11", recorder_dir=root,
                min_first5=None, min_first15=None, min_or_range=None,
            )
            low = resolve_signal_thresholds(
                session_date="2026-08-11", recorder_dir=Path("missing"),
                min_first5=None, min_first15=None, min_or_range=None,
                profile="low_threshold_causal",
            )
        self.assertEqual(live.output_fields(), {
            "effective_min_first5": 0.5,
            "effective_min_first15": 1.0,
            "effective_min_or_range": 0.5,
            "config_source": "run_metadata",
        })
        self.assertEqual(low.min_or_range, 5.0)
        self.assertNotEqual(live.min_or_range, low.min_or_range)

    def test_complete_cli_triple_overrides_metadata(self) -> None:
        result = resolve_signal_thresholds(
            session_date="2026-08-11", recorder_dir=Path("missing"),
            min_first5=2.0, min_first15=3.0, min_or_range=4.0,
        )
        self.assertEqual(result.output_fields(), {
            "effective_min_first5": 2.0,
            "effective_min_first15": 3.0,
            "effective_min_or_range": 4.0,
            "config_source": "cli_explicit",
        })

    def test_partial_cli_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "all three"):
            resolve_signal_thresholds(
                session_date="2026-08-11", recorder_dir=Path("missing"),
                min_first5=0.5, min_first15=None, min_or_range=None,
            )

    def test_missing_live_session_configuration_never_uses_static_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "STRATEGY_CONFIG_PARITY_UNRESOLVED"):
                resolve_signal_thresholds(
                    session_date="2026-08-11", recorder_dir=Path(tmp),
                    min_first5=None, min_first15=None, min_or_range=None,
                )

    def test_historical_profiles_remain_explicit(self) -> None:
        legacy = resolve_signal_thresholds(
            session_date="2026-08-11", recorder_dir=Path("missing"),
            min_first5=None, min_first15=None, min_or_range=None, profile="legacy_offline",
        )
        self.assertEqual((legacy.min_first5, legacy.min_first15, legacy.min_or_range), (0.5, 1.0, 5.0))
        self.assertEqual(legacy.config_source, "profile:legacy_offline_historical")

    def test_runtime_metadata_records_values_actually_used_by_trader(self) -> None:
        args = argparse.Namespace(
            min_first_5m_high_pct=0.5,
            min_first_15m_high_pct=1.0,
            min_or_range_pct=0.75,
        )
        self.assertEqual(runtime_threshold_metadata(args), {
            "min_first_5m_high_pct": 0.5,
            "min_first_15m_high_pct": 1.0,
            "min_or_range_pct": 0.75,
        })

    def test_output_provenance_requires_all_four_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "complete.csv"
            incomplete = Path(tmp) / "incomplete.csv"
            complete.write_text(
                "effective_min_first5,effective_min_first15,effective_min_or_range,config_source\n"
                "0.5,1.0,5.0,cli_explicit\n"
            )
            incomplete.write_text("effective_min_first5,effective_min_first15\n0.5,1.0\n")
            self.assertTrue(output_has_config_provenance(complete))
            self.assertFalse(output_has_config_provenance(incomplete))


if __name__ == "__main__":
    unittest.main()
