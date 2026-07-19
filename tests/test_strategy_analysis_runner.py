from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import scripts.run_daily_analysis as runner


def args_for(**overrides: object) -> argparse.Namespace:
    values = {
        "no_force": False,
        "only": None,
        "skip": None,
        "skip_coverage": False,
        "skip_missed": False,
        "skip_bad_entries": False,
        "skip_bad_entry_details": False,
        "skip_bad_entry_patterns": False,
        "skip_shs": False,
        "skip_nbas": False,
        "skip_offline_runtime_pre_signal": False,
        "sqlite_path": Path("data/runtime/trading_runtime.sqlite"),
        "history_dir": Path("data/history/universe_1m"),
        "output_dir": Path("data/analysis"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class StrategyAnalysisRunnerTests(unittest.TestCase):
    def test_registry_order_includes_required_strategy_analyzers(self) -> None:
        names = [spec.name for spec in runner.ANALYZER_REGISTRY]
        self.assertEqual(names[:4], ["coverage", "missed", "bad_entries", "early_loser"])
        self.assertIn("shs", names)
        self.assertIn("nbas", names)
        self.assertIn("offline_runtime_pre_signal", names)
        self.assertNotIn("bad_entry_details", names)

    def test_only_and_skip_use_registry(self) -> None:
        args = args_for(only=["bad_entries", "early_loser"], skip=["early_loser"])
        specs = runner.selected_specs(args)
        self.assertEqual([spec.name for spec in specs], ["bad_entries"])

    def test_force_and_common_paths_are_passed(self) -> None:
        args = args_for(output_dir=Path("out"), sqlite_path=Path("runtime.sqlite"), history_dir=Path("history"))
        commands = {spec.name: runner.command_for(spec, "2026-07-17", args) for spec in runner.ANALYZER_REGISTRY}
        self.assertIn("--force", commands["coverage"])
        self.assertNotIn("--force", commands["bad_entries"])
        self.assertIn("--sqlite-path", commands["bad_entries"])
        self.assertIn("--history-dir", commands["early_loser"])
        self.assertIn("--output-dir", commands["bad_entries"])

    def test_no_force_removes_force(self) -> None:
        args = args_for(no_force=True)
        for spec in runner.ANALYZER_REGISTRY:
            self.assertNotIn("--force", runner.command_for(spec, "2026-07-17", args))

    def test_expected_outputs_include_strategy_files(self) -> None:
        outputs = {p.name for p in runner.expected_outputs("2026-07-17", Path("out"))}
        self.assertIn("bad_entries_trades_2026-07-17.csv", outputs)
        self.assertIn("bad_entries_data_quality_2026-07-17.json", outputs)
        self.assertIn("early_loser_trade_paths_2026-07-17.csv", outputs)
        self.assertIn("offline_runtime_pre_signal_summary_2026-07-17.csv", outputs)

    def test_strategy_summary_mentions_required_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "bad_entries_data_quality_2026-07-17.json").write_text(json.dumps({"premarket_feature_coverage": "unavailable_for_session"}), encoding="utf-8")
            with (out / "bad_entries_filter_simulation_2026-07-17.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["filter_expression", "net_pnl"])
                writer.writeheader()
                writer.writerow({"filter_expression": "entry >= 09:40", "net_pnl": "1.23"})
            args = args_for(output_dir=out)
            path = runner.write_strategy_summary("2026-07-17", [], total_elapsed=1.0, final_failed=False, args=args)
            text = path.read_text(encoding="utf-8")
            for marker in ["FACT:", "HYPOTHESIS:", "NOT AVAILABLE:", "BASELINE ONLY:", "REQUIRES MULTI-DAY VALIDATION:", "POSSIBLE OVERFITTING:"]:
                self.assertIn(marker, text)
            self.assertIn("premarket_feature_coverage=unavailable_for_session", text)


if __name__ == "__main__":
    unittest.main()
