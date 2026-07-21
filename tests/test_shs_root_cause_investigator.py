from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.shs_root_cause_investigator import Evidence, build_case, classify_root_cause, run


def rec(ts: str, text: str, source: str = "recorder:order_lifecycle", event: str = "") -> dict:
    return {"time": pd.Timestamp(ts, tz="UTC"), "text": text, "source": source, "event": event or text.split()[0], "reason": "", "row": {}}


class SHSRootCauseInvestigatorTests(unittest.TestCase):
    def test_lost_global_ranking_from_nbas_fields(self) -> None:
        cause, _ = classify_root_cause(
            row={"candidates_ahead_count": 2, "better_candidates_ahead_symbols": "AAA:rank=1"},
            symbol_records=[rec("2026-07-20T13:45:02Z", "SIGNAL_READY symbol=XYZ")],
            window_records=[rec("2026-07-20T13:45:02Z", "SIGNAL_READY symbol=XYZ")],
            global_window=[],
            order_records=[],
            execution_record=None,
        )
        self.assertEqual(cause, "READY_BUT_LOST_GLOBAL_RANKING")

    def test_order_submitted_not_filled(self) -> None:
        order = rec("2026-07-20T13:45:04Z", "BUY_ORDER_SENT symbol=XYZ orderId=1 status=Submitted")
        cause, _ = classify_root_cause(
            row={},
            symbol_records=[order],
            window_records=[order],
            global_window=[],
            order_records=[order],
            execution_record=None,
        )
        self.assertEqual(cause, "ORDER_SUBMITTED_NOT_FILLED")

    def test_max_positions_from_global_window(self) -> None:
        cause, _ = classify_root_cause(
            row={},
            symbol_records=[rec("2026-07-20T13:45:02Z", "SIGNAL_READY symbol=XYZ")],
            window_records=[rec("2026-07-20T13:45:02Z", "SIGNAL_READY symbol=XYZ")],
            global_window=[rec("2026-07-20T13:45:03Z", "heartbeat entries_blocked=1 reason=max_positions managed_open=50", source="journal")],
            order_records=[],
            execution_record=None,
        )
        self.assertEqual(cause, "READY_BUT_MAX_POSITIONS_FULL")

    def test_runtime_evidence_missing(self) -> None:
        cause, _ = classify_root_cause(
            row={},
            symbol_records=[],
            window_records=[],
            global_window=[],
            order_records=[],
            execution_record=None,
        )
        self.assertEqual(cause, "RUNTIME_EVIDENCE_MISSING")

    def test_build_case_produces_one_final_root_cause(self) -> None:
        evidence = Evidence(sqlite_sources={}, recorder_sources={}, journal_lines=[], json_sources={})
        row = {"symbol": "XYZ", "possible_signal_time": "2026-07-20T13:45:00Z"}
        case, timeline = build_case(row, evidence, "2026-07-20")
        self.assertEqual(case["symbol"], "XYZ")
        self.assertEqual(case["final_root_cause"], "RUNTIME_EVIDENCE_MISSING")
        self.assertIn("ROOT CAUSE", timeline)

    def test_run_preserves_explicit_cases_csv_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases_csv = root / "should_have_signaled_cases_2026-07-20.csv"
            rows = [
                {
                    "symbol": f"SYM{i:02d}",
                    "possible_signal_time": "2026-07-20T13:45:00Z",
                    "final_classification": "runtime_signal_ready_but_no_buy",
                }
                for i in range(12)
            ]
            pd.DataFrame(rows).to_csv(cases_csv, index=False)
            output_dir = root / "analysis"
            args = argparse.Namespace(
                date="2026-07-20",
                cases_csv=str(cases_csv),
                sqlite_path=str(root / "missing.sqlite"),
                recorder_dir=str(root / "recorder"),
                analysis_dir=str(root / "analysis"),
                output_dir=str(output_dir),
                journal_log=None,
                max_cases=None,
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = run(args)

            self.assertEqual(rc, 0)
            text = stdout.getvalue()
            self.assertIn(f"selected_cases_csv={cases_csv}", text)
            self.assertIn("rows_loaded=12", text)
            self.assertIn("symbols_loaded=12", text)
            self.assertIn("SHS_ROOT_CAUSE_START date=2026-07-20 targets=12", text)
            out = pd.read_csv(output_dir / "shs_root_cause_cases_2026-07-20.csv")
            self.assertEqual(len(out), 12)
            self.assertTrue(out["final_root_cause"].fillna("").astype(str).str.len().gt(0).all())

    def test_run_raises_on_empty_explicit_cases_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases_csv = root / "empty_cases.csv"
            pd.DataFrame(columns=["symbol", "possible_signal_time"]).to_csv(cases_csv, index=False)
            args = argparse.Namespace(
                date="2026-07-20",
                cases_csv=str(cases_csv),
                sqlite_path=str(root / "missing.sqlite"),
                recorder_dir=str(root / "recorder"),
                analysis_dir=str(root / "analysis"),
                output_dir=str(root / "analysis"),
                journal_log=None,
                max_cases=None,
            )

            stdout = io.StringIO()
            with self.assertRaisesRegex(RuntimeError, "SHS_ROOT_CAUSE_NO_CASES"):
                with contextlib.redirect_stdout(stdout):
                    run(args)
            self.assertIn("rows_loaded=0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
