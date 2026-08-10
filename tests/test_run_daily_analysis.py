from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import scripts.run_daily_analysis as runner


def args_for(**overrides: object) -> argparse.Namespace:
    values = {
        "no_force": False,
        "skip_coverage": False,
        "skip_missed": False,
        "skip_bad_entries": False,
        "skip_bad_entry_details": False,
        "skip_bad_entry_patterns": False,
        "skip_shs": False,
        "skip_nbas": False,
        "skip_offline_runtime_pre_signal": False,
        "skip_top100_buy": False,
        "only": [],
        "skip": [],
        "sqlite_path": Path("data/runtime/trading_runtime.sqlite"),
        "history_dir": Path("data/history/universe_1m"),
        "output_dir": runner.DEFAULT_ANALYSIS_DIR,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RunDailyAnalysisTests(unittest.TestCase):
    def test_build_steps_order_includes_offline_runtime_pre_signal_after_nbas(self) -> None:
        steps = runner.build_steps("2026-07-09", args_for())

        names = [name for name, _command, _skipped in steps]
        self.assertEqual(
            names,
            [
                "coverage",
                "missed",
                "bad_entries",
                "early_loser",
                "stop_loss",
                "shs",
                "nbas",
                "offline_runtime_pre_signal",
                "top100_buy",
            ],
        )
        self.assertIn("scripts/early_loser_exit_analyzer.py", steps[3][1])
        self.assertIn("scripts/stop_loss_strategy_analyzer.py", steps[4][1])
        self.assertIn("scripts/investigate_no_buy_after_signal.py", steps[6][1])
        self.assertIn("scripts/investigate_offline_runtime_pre_signal.py", steps[7][1])
        self.assertIn("scripts/analyze_top100_buy.py", steps[8][1])

    def test_force_is_passed_to_supported_steps_by_default(self) -> None:
        steps = runner.build_steps("2026-07-09", args_for())
        commands = {name: command for name, command, _skipped in steps}

        for name in ("coverage", "missed", "shs", "nbas", "offline_runtime_pre_signal", "top100_buy"):
            self.assertIn("--force", commands[name])
        self.assertNotIn("--force", commands["bad_entries"])

    def test_no_force_removes_force_from_supported_steps(self) -> None:
        steps = runner.build_steps("2026-07-09", args_for(no_force=True))

        for _name, command, _skipped in steps:
            self.assertNotIn("--force", command)

    def test_skip_offline_runtime_pre_signal_marks_only_that_step_skipped(self) -> None:
        steps = runner.build_steps("2026-07-09", args_for(skip_offline_runtime_pre_signal=True))
        names = [name for name, _command, _skipped in steps]

        self.assertNotIn("offline_runtime_pre_signal", names)
        self.assertIn("nbas", names)

    def test_expected_outputs_include_offline_runtime_pre_signal_files(self) -> None:
        outputs = {path.name for path in runner.expected_outputs("2026-07-09")}

        self.assertIn("offline_runtime_pre_signal_cases_2026-07-09.csv", outputs)
        self.assertIn("offline_runtime_pre_signal_summary_2026-07-09.csv", outputs)
        self.assertIn("offline_runtime_pre_signal_summary_ALL.csv", outputs)
        self.assertIn("early_loser_rules_2026-07-09.csv", outputs)
        self.assertIn("stop_loss_fixed_grid_2026-07-09.csv", outputs)
        self.assertIn("top100_buy_symbol_day_2026-07-09.csv", outputs)
        self.assertIn("top100_buy_data_quality_2026-07-09.json", outputs)

    def test_skip_top100_buy(self) -> None:
        steps = runner.build_steps("2026-07-09", args_for(skip_top100_buy=True))
        self.assertNotIn("top100_buy", [name for name, _command, _skipped in steps])

    def test_daily_summary_includes_offline_runtime_pre_signal_summary(self) -> None:
        original_analysis_dir = runner.DEFAULT_ANALYSIS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "analysis"
            analysis_dir.mkdir(parents=True)
            runner.DEFAULT_ANALYSIS_DIR = analysis_dir
            target = analysis_dir / "offline_runtime_pre_signal_summary_2026-07-09.csv"
            with target.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["date", "final_classification", "total_cases"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "date": "2026-07-09",
                        "final_classification": "offline_should_have_signaled_runtime_signal_not_observed",
                        "total_cases": "22",
                    }
                )
            details = analysis_dir / "bad_entry_details_summary_2026-07-09.csv"
            with details.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["session_date", "metric", "bucket", "count"])
                writer.writeheader()
                writer.writerow({"session_date": "2026-07-09", "metric": "overall", "bucket": "all", "count": "3"})
            filters = analysis_dir / "bad_entry_candidate_filters.csv"
            with filters.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["filter_expression", "matching_trades"])
                writer.writeheader()
                writer.writerow({"filter_expression": "live_entry_score < 40", "matching_trades": "3"})
            (analysis_dir / "bad_entry_pattern_summary.md").write_text(
                "# Bad Entry Pattern Summary\n\n```text\nstatus=insufficient_sample\nfilters=1\n```\n",
                encoding="utf-8",
            )

            try:
                summary = runner.write_summary(
                    "2026-07-09",
                    [],
                    total_elapsed=1.23,
                    final_failed=False,
                )
            finally:
                runner.DEFAULT_ANALYSIS_DIR = original_analysis_dir

            text = summary.read_text(encoding="utf-8")
            self.assertIn("offline_runtime_pre_signal_summary_2026-07-09.csv", text)
            self.assertIn("final_classification", text)
            self.assertIn("offline_should_have_signaled_runtime_signal_not_observed=1", text)
            self.assertIn("offline_runtime_pre_signal_summary_2026-07-09.csv", text)


    def test_shs_command_receives_missed_output_from_selected_output_dir(self) -> None:
        output_dir = Path("custom/analysis")
        steps = runner.build_steps("2026-07-20", args_for(output_dir=output_dir))
        commands = {name: command for name, command, _skipped in steps}
        self.assertIn("--missed-runners-csv", commands["shs"])
        self.assertIn(str(output_dir / "missed_runners_2026-07-20.csv"), commands["shs"])
        self.assertIn("--should-have-signaled-csv", commands["nbas"])
        self.assertIn(str(output_dir / "should_have_signaled_cases_2026-07-20.csv"), commands["nbas"])
        self.assertIn("--cases-csv", commands["offline_runtime_pre_signal"])
        self.assertIn(str(output_dir / "no_buy_after_signal_cases_2026-07-20.csv"), commands["offline_runtime_pre_signal"])

    def test_shs_handoff_mismatch_marks_step_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            missed = output_dir / "missed_runners_2026-07-20.csv"
            with missed.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["symbol", "top100_no_signal_reason"])
                writer.writeheader()
                for symbol in ["AAA", "BBB", "CCC"]:
                    writer.writerow({"symbol": symbol, "top100_no_signal_reason": "should_have_signaled"})
            summary = output_dir / "should_have_signaled_summary_2026-07-20.csv"
            with summary.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["date", "total_should_have_signaled"])
                writer.writeheader()
                writer.writerow({"date": "2026-07-20", "total_should_have_signaled": "0"})
            result = runner.StepResult("shs", [], False, "ok", 0, 0.1)
            runner.validate_shs_handoff(result, output_dir=output_dir, session_date="2026-07-20")
            self.assertEqual(result.status, "failed")
            self.assertIn("missed_should_have_signaled_count=3", result.warnings)
            self.assertIn("shs_targets_count=0", result.warnings)

    def test_only_filters_registered_steps(self) -> None:
        steps = runner.build_steps("2026-07-09", args_for(only=["coverage", "nbas"]))
        self.assertEqual([name for name, _command, _skipped in steps], ["coverage", "nbas"])


if __name__ == "__main__":
    unittest.main()
