from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.live_trading.analysis.full_session_replay_v67 import (
    PreparedReplayFeatures,
    PreparedSessionCache,
    ReplayConfig,
    _feature_at,
    _rows,
    build_effective_config,
    build_parser,
    profile_comparison_rows,
    profile_config,
    replay_performance_counters,
    replay_session,
    reset_replay_performance_counters,
)
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
        defaults = {"entry_delay_after_open_minutes": 0.0, "max_entries_per_cycle": 5, "max_entries_per_minute": 5}
        defaults.update(config_kwargs)
        config = ReplayConfig(**defaults)
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

    def test_live_partial_windows_can_enter_after_entry_delay(self) -> None:
        result = self.replay_with({"BOUND": ready_candles(11.0)}, **profile_config("live").__dict__)
        entry = [event for event in result.events if event["event_type"] == "ENTRY"][0]
        self.assertEqual(entry["timestamp"], "2026-07-20T13:35:00+00:00")

    def test_finalized_window_mode_waits_for_first15_boundary(self) -> None:
        config = profile_config("legacy_offline")
        config.breakout_mode = "live"
        result = self.replay_with({"BOUND": ready_candles(11.0)}, **config.__dict__)
        entry = [event for event in result.events if event["event_type"] == "ENTRY"][0]
        self.assertEqual(entry["timestamp"], "2026-07-20T13:45:00+00:00")

    def test_deterministic_rerun(self) -> None:
        data = {"AAA": ready_candles(11.0), "BBB": ready_candles(10.9)}
        one = self.replay_with(data).events
        two = self.replay_with(data).events
        self.assertEqual(one, two)

    def test_missing_empty_and_invalid_candles_are_skipped_without_changing_replay(self) -> None:
        complete = {"AAA": ready_candles(11.0)}
        baseline = self.replay_with(complete)
        mixed = self.replay_with({
            **complete,
            "NO_HISTORY": pd.DataFrame(),
            "INVALID": pd.DataFrame({"open": [10.0], "close": [10.1]}),
        })

        self.assertEqual(mixed, baseline)
        observed_symbols = {
            str(row.get("symbol") or "")
            for row in [*mixed.events, *mixed.trades]
            if row.get("symbol")
        }
        self.assertEqual(observed_symbols, {"AAA"})

    def test_prepared_features_match_reference_for_all_profiles(self) -> None:
        frame = ready_candles(11.0, spread=12.5)
        for profile in ("live", "low_threshold_causal", "legacy_offline"):
            config = profile_config(profile)
            rows = _rows(frame, config.bar_timestamp_semantics)
            prepared = PreparedReplayFeatures(rows, config)
            timestamps = pd.date_range("2026-07-20T13:29:00Z", "2026-07-20T13:50:00Z", freq="min")
            for timestamp in timestamps:
                expected = _feature_at(rows, timestamp, config)
                actual = prepared.at(timestamp)
                self.assertEqual(actual.keys(), expected.keys())
                for key, expected_value in expected.items():
                    actual_value = actual[key]
                    if isinstance(expected_value, float):
                        self.assertAlmostEqual(actual_value, expected_value, places=12, msg=f"{profile=} {timestamp=} {key=}")
                    else:
                        self.assertEqual(actual_value, expected_value, f"{profile=} {timestamp=} {key=}")

    def test_prepared_session_cache_key_includes_full_config_and_is_bounded(self) -> None:
        rows = _rows(ready_candles(11.0), "bar_start")
        live = profile_config("live")
        low = profile_config("low_threshold_causal")
        reset_replay_performance_counters()
        cache = PreparedSessionCache(max_entries=1, max_bytes=64 * 1024 * 1024)
        first = cache.get_or_build("AAA", "2026-07-20", rows, live)
        second = cache.get_or_build("AAA", "2026-07-20", rows, live)
        third = cache.get_or_build("AAA", "2026-07-20", rows, low)
        counters = replay_performance_counters()
        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(cache.frame_count, 1)
        self.assertGreater(cache.approximate_bytes, 0)
        self.assertEqual(counters["prepared_session_cache_hits"], 1)
        self.assertEqual(counters["prepared_session_cache_misses"], 2)

    def test_prepared_session_cache_handles_empty_rows(self) -> None:
        cache = PreparedSessionCache(max_entries=1, max_bytes=1024)
        prepared = cache.get_or_build("EMPTY", "2026-07-20", pd.DataFrame(), profile_config("live"))
        self.assertTrue(prepared.rows.empty)
        self.assertEqual(cache.frame_count, 1)
        self.assertEqual(cache.approximate_bytes, 0)

    def test_fast_replay_exposes_zero_legacy_full_frame_calls(self) -> None:
        reset_replay_performance_counters()
        result = self.replay_with({"AAA": ready_candles(11.0)})
        counters = replay_performance_counters()
        self.assertTrue(any(event["event_type"] == "ENTRY" for event in result.events))
        self.assertEqual(counters["legacy_full_frame_feature_calls"], 0)
        self.assertEqual(counters["replay_session_calls"], 1)

    def test_replay_session_does_not_call_full_frame_feature_function(self) -> None:
        with patch("src.live_trading.analysis.full_session_replay_v67._feature_at", side_effect=AssertionError("full-frame feature path used")):
            result = self.replay_with({"AAA": ready_candles(11.0)})
        self.assertTrue(any(event["event_type"] == "ENTRY" for event in result.events))

    def test_optimized_replay_matches_full_frame_reference_exactly(self) -> None:
        data = {"READY": ready_candles(11.0), "NEVER": ready_candles(5.1)}
        optimized = self.replay_with(data, max_open_positions=1)

        class ReferenceFeatures:
            def __init__(self, rows, config):
                self.rows = rows
                self.config = config

            def latest_row(self, timestamp):
                visible = self.rows[self.rows["available_at"] <= timestamp]
                return None if visible.empty else visible.iloc[-1]

            def at(self, timestamp):
                return _feature_at(self.rows, timestamp, self.config)

        with patch("src.live_trading.analysis.full_session_replay_v67.PreparedReplayFeatures", ReferenceFeatures):
            reference = self.replay_with(data, max_open_positions=1)
        self.assertEqual(optimized, reference)

    def test_golden_stop_and_trailing_exit_paths(self) -> None:
        common = [
            ("2026-07-20T13:30:00Z", 10.0, 10.8, 9.8, 10.7),
            ("2026-07-20T13:31:00Z", 10.7, 10.9, 10.5, 10.8),
            ("2026-07-20T13:32:00Z", 10.8, 11.0, 10.6, 10.9),
            ("2026-07-20T13:33:00Z", 10.9, 11.1, 10.7, 11.0),
            ("2026-07-20T13:34:00Z", 11.0, 11.2, 10.8, 11.1),
        ]
        stop = candles(common + [
            ("2026-07-20T13:35:00Z", 11.1, 11.2, 9.0, 9.5),
            ("2026-07-20T13:36:00Z", 9.5, 9.7, 9.2, 9.4),
        ])
        trailing = candles(common + [
            ("2026-07-20T13:35:00Z", 11.1, 11.5, 11.2, 11.4),
            ("2026-07-20T13:36:00Z", 11.4, 11.4, 11.0, 11.1),
        ])
        result = self.replay_with({"STOP": stop, "TRAIL": trailing}, entry_delay_after_open_minutes=5.0)
        trades = {trade["symbol"]: trade for trade in result.trades}

        self.assertEqual(trades["STOP"]["entry_time"], "2026-07-20T13:35:00+00:00")
        self.assertEqual(trades["STOP"]["exit_time"], "2026-07-20T13:36:00+00:00")
        self.assertEqual(trades["STOP"]["exit_reason"], "v46_wide_trail_stop_loss")
        self.assertEqual(trades["TRAIL"]["entry_time"], "2026-07-20T13:35:00+00:00")
        self.assertEqual(trades["TRAIL"]["exit_time"], "2026-07-20T13:37:00+00:00")
        self.assertEqual(trades["TRAIL"]["exit_reason"], "v46_wide_trail_trailing_stop")

    def test_live_profile_defaults_match_v67_thresholds(self) -> None:
        config = profile_config("live")
        self.assertEqual(config.first5_threshold, 4.0)
        self.assertEqual(config.first15_threshold, 6.5)
        self.assertEqual(config.breakout_mode, "live")
        self.assertEqual(config.bar_timestamp_semantics, "bar_start")

    def test_low_threshold_profile_changes_only_signal_thresholds(self) -> None:
        live = profile_config("live")
        low = profile_config("low_threshold_causal")
        self.assertEqual(low.first5_threshold, 0.5)
        self.assertEqual(low.first15_threshold, 1.0)
        self.assertEqual(low.breakout_mode, live.breakout_mode)
        self.assertEqual(low.bar_timestamp_semantics, live.bar_timestamp_semantics)
        self.assertEqual(low.window_availability_mode, live.window_availability_mode)

    def test_legacy_offline_profile_is_marked_non_causal(self) -> None:
        legacy = profile_config("legacy_offline")
        self.assertEqual(legacy.first5_threshold, 0.5)
        self.assertEqual(legacy.first15_threshold, 1.0)
        self.assertEqual(legacy.breakout_mode, "legacy_candle_high")
        self.assertEqual(legacy.bar_timestamp_semantics, "bar_end")
        self.assertEqual(legacy.window_availability_mode, "finalized_windows")

    def test_cli_overrides_profile_thresholds_and_risk_parameters(self) -> None:
        args = build_parser().parse_args([
            "--date", "2026-07-20",
            "--profile", "live",
            "--first5-threshold", "0.5",
            "--first15-threshold", "1.0",
            "--or-max-range-pct", "3.5",
            "--notional", "500",
            "--max-positions", "3",
            "--hard-stop-pct", "4",
            "--trailing-activation-pct", "2",
            "--trailing-stop-pct", "1.5",
            "--commission-model", "none",
        ])
        config = build_effective_config(args, "live")
        self.assertEqual(config.first5_threshold, 0.5)
        self.assertEqual(config.first15_threshold, 1.0)
        self.assertEqual(config.min_or_range_pct, 3.5)
        self.assertEqual(config.position_usd, 500)
        self.assertEqual(config.max_open_positions, 3)
        self.assertEqual(config.exit_stop_loss_pct, 4)
        self.assertEqual(config.exit_trailing_activation_pct, 2)
        self.assertEqual(config.exit_trailing_stop_pct, 1.5)
        self.assertEqual(config.commission_model, "none")

    def test_low_threshold_causal_can_enter_without_lookahead(self) -> None:
        low_threshold = candles([
            ("2026-07-20T13:30:00Z", 10.0, 10.1, 9.5, 10.0),
            ("2026-07-20T13:34:00Z", 10.0, 10.1, 9.6, 10.0),
            ("2026-07-20T13:44:00Z", 10.0, 10.2, 9.6, 10.1),
            ("2026-07-20T13:45:00Z", 10.1, 10.2, 10.0, 10.1),
            ("2026-07-20T13:46:00Z", 10.1, 10.3, 10.0, 10.2),
        ])
        live_result = self.replay_with({"LOWT": low_threshold}, **profile_config("live").__dict__)
        causal_result = self.replay_with({"LOWT": low_threshold}, **profile_config("low_threshold_causal").__dict__)
        self.assertFalse(any(event["event_type"] == "ENTRY" for event in live_result.events))
        entries = [event for event in causal_result.events if event["event_type"] == "ENTRY"]
        self.assertEqual(entries[0]["timestamp"], "2026-07-20T13:35:00+00:00")

    def test_legacy_candle_high_breakout_can_differ_from_current_price_breakout(self) -> None:
        data = {"LEG": ready_candles(close_at_1344=10.7)}
        current_price_breakout = profile_config("low_threshold_causal")
        current_price_breakout.breakout_mode = "current_price_or_high"
        legacy = profile_config("legacy_offline")
        current_price_result = self.replay_with(data, **current_price_breakout.__dict__)
        legacy_result = self.replay_with(data, **legacy.__dict__)
        current_entry = [event for event in current_price_result.events if event["event_type"] == "ENTRY"][0]
        legacy_entry = [event for event in legacy_result.events if event["event_type"] == "ENTRY"][0]
        self.assertEqual(legacy_entry["timestamp"], "2026-07-20T13:45:00+00:00")
        self.assertEqual(current_entry["timestamp"], "2026-07-20T13:47:00+00:00")

    def test_output_rows_include_effective_config(self) -> None:
        result = self.replay_with({"CFG": ready_candles(11.0)}, **profile_config("low_threshold_causal").__dict__)
        entry_event = [event for event in result.events if event["event_type"] == "ENTRY"][0]
        self.assertIn('"profile": "low_threshold_causal"', entry_event["effective_config_json"])
        if result.trades:
            self.assertIn('"first5_threshold": 0.5', result.trades[0]["effective_config_json"])

    def test_profile_comparison_contains_focus_symbol_statuses_and_config(self) -> None:
        result = self.replay_with({"NUAI": ready_candles(11.0)}, **profile_config("live").__dict__)
        rows = profile_comparison_rows("2026-07-20", {"live": result})
        self.assertEqual(rows[0]["signals"], 1)
        self.assertEqual(rows[0]["NUAI_status"], "entered")
        self.assertEqual(rows[0]["IREN_status"], "not_entered")
        self.assertIn('"profile": "live"', rows[0]["effective_config_json"])


    def test_parity_trace_uses_same_replay_config(self) -> None:
        from src.live_trading.analysis.full_session_replay_v67 import build_parity_trace

        data = {"FCEL": ready_candles(11.0)}
        result = self.replay_with(data, **profile_config("live").__dict__)
        top = pd.DataFrame({"symbol": ["FCEL"], "top100_rank": [1]})
        with patch("src.live_trading.analysis.full_session_replay_v67.load_top100", return_value=top), patch("src.live_trading.analysis.full_session_replay_v67.load_session_candles", side_effect=lambda _history, symbol, _date, _type: data[symbol]):
            rows = build_parity_trace(
                session_date="2026-07-20",
                symbols=["FCEL"],
                top100_path=Path("unused.csv"),
                history_dir=Path("unused"),
                config=profile_config("live"),
                replay=result,
                live=[],
            )
        entry_rows = [row for row in rows if row["replay_entered_by_this_time"]]
        self.assertEqual(entry_rows[0]["timestamp"], "2026-07-20T13:35:00+00:00")
        self.assertEqual(entry_rows[0]["window_availability_mode"], "live_partial")

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
