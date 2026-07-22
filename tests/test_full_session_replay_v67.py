from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.live_trading.analysis.full_session_replay_v67 import ReplayConfig, replay_session
from src.live_trading.analysis.signal_opportunity_forensics import load_case_rows, run as run_signal_opportunity, build_parser as build_signal_parser


def candles(values: list[tuple[str, float, float, float, float]], spread: float | None = None) -> pd.DataFrame:
    rows = []
    for ts, op, high, low, close in values:
        row = {"timestamp": pd.Timestamp(ts), "open": op, "high": high, "low": low, "close": close, "volume": 1000}
        if spread is not None:
            row["spread_bps"] = spread
        rows.append(row)
    return pd.DataFrame(rows)


def ready_candles(close_at_1344: float = 11.0, spread: float | None = None) -> pd.DataFrame:
    return candles([
        ("2026-07-20T13:30:00Z", 10.0, 10.8, 9.8, 10.6),
        ("2026-07-20T13:34:00Z", 10.6, 10.9, 10.1, 10.8),
        ("2026-07-20T13:44:00Z", 10.8, 11.2, 10.1, close_at_1344),
        ("2026-07-20T13:45:00Z", close_at_1344, 11.4, 10.9, 11.1),
        ("2026-07-20T13:46:00Z", 11.1, 11.8, 11.0, 11.7),
        ("2026-07-20T13:47:00Z", 11.7, 11.8, 11.2, 11.3),
    ], spread=spread)


class FullSessionReplayTests(unittest.TestCase):
    def replay_with(self, data: dict[str, pd.DataFrame], **config_kwargs):
        top = pd.DataFrame({"symbol": list(data), "top100_rank": list(range(1, len(data) + 1))})
        config = ReplayConfig(entry_delay_after_open_minutes=0.0, max_entries_per_cycle=5, max_entries_per_minute=5, **config_kwargs)
        with patch("src.live_trading.analysis.full_session_replay_v67.load_top100", return_value=top), patch("src.live_trading.analysis.full_session_replay_v67.load_session_candles", side_effect=lambda _history, symbol, _date, _type: data[symbol]):
            return replay_session(session_date="2026-07-20", top100_path=Path("unused.csv"), history_dir=Path("unused"), config=config)

    def test_two_candidates_one_slot_higher_score_wins(self) -> None:
        result = self.replay_with({"HIGH": ready_candles(11.0), "LOW": ready_candles(10.8)}, max_open_positions=1)
        entries = [event for event in result.events if event["event_type"] == "ENTRY"]
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["symbol"], "HIGH")
        self.assertEqual(result.skipped.get("max_positions_full", 0) > 0, True)

    def test_slot_released_after_exit_later_candidate_enters(self) -> None:
        first = ready_candles(11.0)
        second = candles([
            ("2026-07-20T13:30:00Z", 10.0, 10.8, 9.8, 10.6),
            ("2026-07-20T13:34:00Z", 10.6, 10.9, 10.1, 10.8),
            ("2026-07-20T13:44:00Z", 10.8, 11.1, 4.8, 4.9),
            ("2026-07-20T13:45:00Z", 4.9, 5.1, 4.8, 4.9),
            ("2026-07-20T13:50:00Z", 4.9, 10.9, 4.9, 10.8),
            ("2026-07-20T13:51:00Z", 10.8, 11.0, 10.7, 10.9),
        ])
        result = self.replay_with({"FIRST": first, "SECOND": second}, max_open_positions=1)
        entries = [event["symbol"] for event in result.events if event["event_type"] == "ENTRY"]
        self.assertIn("FIRST", entries)
        self.assertIn("SECOND", entries)

    def test_spread_block(self) -> None:
        result = self.replay_with({"WIDE": ready_candles(11.0, spread=200.0)})
        entries = [event for event in result.events if event["event_type"] == "ENTRY"]
        self.assertEqual(entries, [])
        self.assertGreater(result.skipped.get("spread_too_wide", 0), 0)

    def test_entry_delay_block(self) -> None:
        result = self.replay_with({"DELAY": ready_candles(11.0)}, entry_delay_after_open_minutes=20.0)
        self.assertTrue(any(event["event_type"] == "ENTRY_BLOCKED" and event["reason"] == "entry_delay_after_open" for event in result.events))

    def test_no_lookahead_1345_boundary(self) -> None:
        result = self.replay_with({"BOUND": ready_candles(11.0)})
        entry = [event for event in result.events if event["event_type"] == "ENTRY"][0]
        self.assertEqual(entry["timestamp"], "2026-07-20T13:45:00+00:00")

    def test_deterministic_rerun(self) -> None:
        data = {"AAA": ready_candles(11.0), "BBB": ready_candles(10.9)}
        one = self.replay_with(data).events
        two = self.replay_with(data).events
        self.assertEqual(one, two)

    def test_cases_csv_three_rows_produces_three_signal_opportunity_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.csv"
            with cases.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["symbol", "possible_signal_time"])
                writer.writeheader()
                for symbol in ["NUAI", "IREN", "FBYD"]:
                    writer.writerow({"symbol": symbol, "possible_signal_time": "2026-07-20T13:45:00Z"})
            loaded = load_case_rows(cases, [], "2026-07-20")
            self.assertEqual(sorted(loaded), ["FBYD", "IREN", "NUAI"])

            def fake_analyze(**kwargs):
                symbol = kwargs["symbol"]
                return {"date": "2026-07-20", "symbol": symbol, "classification": "MISSING_CANDLES"}, []

            args = build_signal_parser().parse_args(["--date", "2026-07-20", "--cases-csv", str(cases), "--output-dir", str(tmp_path)])
            with patch("src.live_trading.analysis.signal_opportunity_forensics.analyze_symbol", side_effect=fake_analyze):
                run_signal_opportunity(args)
            with (tmp_path / "signal_opportunity_cases_2026-07-20.csv").open(newline="") as f:
                output_rows = list(csv.DictReader(f))
            self.assertEqual([row["symbol"] for row in output_rows], ["FBYD", "IREN", "NUAI"])



if __name__ == "__main__":
    unittest.main()
