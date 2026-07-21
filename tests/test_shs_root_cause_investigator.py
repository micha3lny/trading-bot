from __future__ import annotations

import unittest

import pandas as pd

from src.live_trading.analysis.shs_root_cause_investigator import build_case, classify_root_cause, Evidence


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


if __name__ == "__main__":
    unittest.main()
