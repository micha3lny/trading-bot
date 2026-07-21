from __future__ import annotations

import unittest

import pandas as pd

from src.live_trading.analysis.should_have_signaled_investigator import build_symbol_index, sources_for_symbol
from src.live_trading.analysis.signal_replay_analyzer import row_belongs_to_session, symbol_rows


class SHSSessionScopingTests(unittest.TestCase):
    def test_top100_source_date_does_not_make_next_day_event_match_prior_session(self) -> None:
        row = {
            "symbol": "ADVB",
            "event": "ENTRY_SIGNAL",
            "event_time": "2026-07-21T13:45:01+00:00",
            "session_date": "2026-07-21",
            "raw_json": '{"symbol":"ADVB","top100_source_date":"2026-07-20","event":"ENTRY_SIGNAL"}',
        }
        self.assertFalse(row_belongs_to_session(row, "2026-07-20"))
        self.assertTrue(row_belongs_to_session(row, "2026-07-21"))

    def test_symbol_index_excludes_next_day_event_with_prior_top100_source_date(self) -> None:
        sources = {
            "runtime_events": pd.DataFrame([
                {
                    "symbol": "ADVB",
                    "event": "ENTRY_SIGNAL",
                    "event_time": "2026-07-21T13:45:01+00:00",
                    "session_date": "2026-07-21",
                    "raw_json": '{"symbol":"ADVB","top100_source_date":"2026-07-20","event":"ENTRY_SIGNAL"}',
                }
            ])
        }
        indexed = build_symbol_index(sources, {"ADVB"}, session_date="2026-07-20")
        self.assertTrue(indexed["runtime_events"]["ADVB"].empty)
        indexed_next_day = build_symbol_index(sources, {"ADVB"}, session_date="2026-07-21")
        self.assertEqual(len(indexed_next_day["runtime_events"]["ADVB"]), 1)

    def test_symbol_rows_excludes_next_day_entry_signal_for_prior_session(self) -> None:
        df = pd.DataFrame([
            {
                "symbol": "ADVB",
                "event": "ENTRY_SIGNAL",
                "event_time": "2026-07-21T13:45:01+00:00",
                "session_date": "2026-07-21",
                "raw_json": '{"symbol":"ADVB","top100_source_date":"2026-07-20","event":"ENTRY_SIGNAL"}',
            }
        ])
        center = pd.Timestamp("2026-07-20T13:45:00Z")
        self.assertTrue(symbol_rows(df, "ADVB", center, window_minutes=30, session_date="2026-07-20").empty)
        next_center = pd.Timestamp("2026-07-21T13:45:00Z")
        self.assertEqual(len(symbol_rows(df, "ADVB", next_center, window_minutes=30, session_date="2026-07-21")), 1)

    def test_sources_for_symbol_does_not_resurrect_prefiltered_rows(self) -> None:
        sources = {
            "runtime_events": pd.DataFrame([
                {
                    "symbol": "ADVB",
                    "event": "ENTRY_SIGNAL",
                    "event_time": "2026-07-21T13:45:01+00:00",
                    "session_date": "2026-07-21",
                    "raw_json": '{"symbol":"ADVB","top100_source_date":"2026-07-20","event":"ENTRY_SIGNAL"}',
                }
            ])
        }
        indexed = build_symbol_index(sources, {"ADVB"}, session_date="2026-07-20")
        scoped = sources_for_symbol(indexed, "ADVB", pd.Timestamp("2026-07-20T13:45:00Z"), window_minutes=30)
        self.assertTrue(scoped["runtime_events"].empty)


if __name__ == "__main__":
    unittest.main()
