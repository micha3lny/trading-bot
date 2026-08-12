from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.candidate_lifetime_analyzer import (
    _canonical_event,
    _event_unblocked_by_only,
    aggregate_range,
    build_lifetimes,
    counterfactual_filter_rows,
    deduplicate_events,
    event_is_buy,
    future_outcome,
    load_sqlite_events,
    portfolio_counterfactuals,
)
from src.live_trading.analysis.full_session_replay_v67 import (
    PreparedReplayFeatures,
    ReplayConfig,
    _rows,
    effective_config_dict,
)


SESSION = "2026-08-11"


def event(symbol: str, timestamp: str, event_type: str, **values):
    return {
        "session_date": SESSION,
        "symbol": symbol,
        "timestamp": pd.Timestamp(timestamp),
        "event_type": event_type,
        "reason": values.pop("reason", ""),
        "outcome": values.pop("outcome", ""),
        "status": values.pop("status", ""),
        "scan_id": values.pop("scan_id", "scan-1"),
        "top100_rank": values.pop("top100_rank", 1),
        "top100_score": values.pop("top100_score", 100.0),
        "live_entry_score": values.pop("live_entry_score", 10.0),
        "price": values.pop("price", 10.0),
        "entries_blocked": values.pop("entries_blocked", 0),
        "entries_blocked_reason": values.pop("entries_blocked_reason", ""),
        "already_open": values.pop("already_open", 0),
        "source": values.pop("source", "sqlite:runtime_events"),
        "raw_json": values,
    }


def candles(start: str = "2026-08-11T13:30:00Z", periods: int = 90) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="min")
    closes = [10.0 + index * 0.01 for index in range(periods)]
    return pd.DataFrame({
        "timestamp": times,
        "open": closes,
        "high": [value + 0.05 for value in closes],
        "low": [value - 0.05 for value in closes],
        "close": closes,
        "volume": [1000] * periods,
    })


class CandidateLifetimeAnalyzerTests(unittest.TestCase):
    def test_observed_172_evaluated_30_ready_30_bought_pattern(self) -> None:
        events = []
        candle_map = {}
        for index in range(172):
            symbol = f"S{index:03d}"
            events.extend([
                event(symbol, "2026-08-11T13:35:00Z", "SIGNAL_EVALUATED", outcome="rejected", reason="first_5m_high_too_low"),
                event(symbol, "2026-08-11T13:35:00Z", "SIGNAL_REJECTED", outcome="rejected", reason="first_5m_high_too_low"),
            ])
            if index < 30:
                events.extend([
                    event(symbol, "2026-08-11T13:45:00Z", "SIGNAL_READY", outcome="ready", reason="ready"),
                    event(symbol, "2026-08-11T13:45:01Z", "BUY_ORDER_SENT", status="submitted"),
                ])
            candle_map[symbol] = candles()

        result, _ = build_lifetimes(SESSION, events, candle_map, missed_threshold_pct=3.0)

        self.assertEqual(len(result), 172)
        self.assertEqual(result["first_ready_at"].ne("").sum(), 30)
        self.assertEqual(result["first_buy_at"].ne("").sum(), 30)
        self.assertEqual((result["classification"] == "eventually_bought").sum(), 30)
        self.assertTrue(result["number_of_rejections"].eq(1).all())

    def test_future_outcome_excludes_pre_event_portion_of_unfinished_bar(self) -> None:
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-08-11T13:35:00Z", "2026-08-11T13:36:00Z"]),
            "open": [10.0, 10.0], "high": [50.0, 11.0], "low": [9.0, 9.8],
            "close": [10.0, 10.5], "volume": [1000, 1000],
        })
        outcome = future_outcome(frame, pd.Timestamp("2026-08-11T13:35:30Z"), 10.0)
        self.assertEqual(outcome["max_price_after_rejection"], 11.0)
        self.assertNotEqual(outcome["max_price_after_rejection"], 50.0)

    def test_only_one_filter_is_removed(self) -> None:
        only = event("ONLY", "2026-08-11T13:35:00Z", "SIGNAL_REJECTED", reason="first_5m_high_too_low")
        multiple = event("MULTI", "2026-08-11T13:35:00Z", "SIGNAL_REJECTED", reason="first_5m_high_too_low;spread_too_wide")
        blocked = event("BLOCKED", "2026-08-11T13:35:00Z", "SIGNAL_REJECTED", reason="first_5m_high_too_low", entries_blocked=1)
        self.assertTrue(_event_unblocked_by_only(only, "first_5m_high_too_low"))
        self.assertFalse(_event_unblocked_by_only(multiple, "first_5m_high_too_low"))
        self.assertFalse(_event_unblocked_by_only(blocked, "first_5m_high_too_low"))

        result = counterfactual_filter_rows(
            SESSION, [only, multiple, blocked], {symbol: candles() for symbol in ("ONLY", "MULTI", "BLOCKED")},
            missed_threshold_pct=3.0,
        ).set_index("filter")
        self.assertEqual(result.loc["first_5m_high_too_low", "rejected_candidates"], 3)
        self.assertEqual(result.loc["first_5m_high_too_low", "candidates_unblocked_if_removed"], 1)

    def test_dispatch_attempt_is_not_treated_as_completed_buy(self) -> None:
        self.assertFalse(event_is_buy(event("AAA", "2026-08-11T13:45:00Z", "ENTRY_ORDER_DISPATCH_ATTEMPT")))
        self.assertTrue(event_is_buy(event("AAA", "2026-08-11T13:45:01Z", "BUY_ORDER_SENT")))

    def test_strict_session_scope_and_cross_source_dedupe(self) -> None:
        row = {
            "event_time": "2026-08-11T13:35:00Z", "event_type": "SIGNAL_EVALUATED",
            "session_date": SESSION, "symbol": "AAA",
            "raw_json": json.dumps({"session_date": SESSION, "symbol": "AAA", "outcome": "rejected"}),
        }
        sqlite_event = _canonical_event(row, "sqlite:runtime_events", SESSION)
        recorder_event = _canonical_event(row, "recorder:trade_lifecycle.csv", SESSION)
        wrong_day = dict(row, event_time="2026-08-12T13:35:00Z", session_date="2026-08-12")
        self.assertIsNone(_canonical_event(wrong_day, "recorder:trade_lifecycle.csv", SESSION))
        result, duplicates = deduplicate_events([recorder_event, sqlite_event])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "sqlite:runtime_events")
        self.assertEqual(duplicates, 1)

    def test_sqlite_loader_uses_one_session_scoped_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.sqlite"
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE runtime_events (event_time TEXT, event_type TEXT, session_date TEXT, symbol TEXT, raw_json TEXT)")
                for day in (SESSION, "2026-08-12"):
                    conn.execute(
                        "INSERT INTO runtime_events VALUES (?, ?, ?, ?, ?)",
                        (f"{day}T13:35:00Z", "SIGNAL_EVALUATED", day, "AAA", json.dumps({"session_date": day, "symbol": "AAA"})),
                    )
            rows = load_sqlite_events(path, SESSION)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_date"], SESSION)

    def test_default_replay_output_is_unchanged_and_disabled_filter_is_explicit(self) -> None:
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-11T13:30:00Z", periods=16, freq="min"),
            "open": [10.0] * 16, "high": [10.1] * 16, "low": [9.9] * 16,
            "close": [10.0] * 16, "volume": [1000] * 16,
        })
        base = ReplayConfig(entry_delay_after_open_minutes=0)
        rows = _rows(frame, base.bar_timestamp_semantics)
        baseline = PreparedReplayFeatures(rows, base).at(pd.Timestamp("2026-08-11T13:45:00Z"))
        repeated = PreparedReplayFeatures(rows, ReplayConfig(entry_delay_after_open_minutes=0)).at(pd.Timestamp("2026-08-11T13:45:00Z"))
        self.assertEqual(baseline, repeated)
        self.assertNotIn("disabled_entry_filters", effective_config_dict(base))

        without_first5 = replace(base, disabled_entry_filters=("first_5m_high_too_low",))
        changed = PreparedReplayFeatures(rows, without_first5).at(pd.Timestamp("2026-08-11T13:45:00Z"))
        self.assertNotIn("first_5m_high_too_low", changed["reason"])
        self.assertIn("first_15m_high_too_low", changed["reason"])

    def test_portfolio_counterfactual_uses_shared_causal_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top100_path = root / "daily_top100_2026-08-11.csv"
            pd.DataFrame({"symbol": ["AAA"], "rank": [1], "score": [100]}).to_csv(top100_path, index=False)
            times = pd.date_range("2026-08-11T13:30:00Z", periods=20, freq="min")
            highs = [10.3] * 5 + [11.0] * 15
            frame = pd.DataFrame({
                "timestamp": times, "open": [10.0] * 20, "high": highs,
                "low": [9.8] * 20, "close": [10.0] * 5 + [10.9] * 15,
                "volume": [1000] * 20,
            })
            result = portfolio_counterfactuals(
                SESSION, top100_path, root, ["first_5m_high_too_low"], {"AAA": frame}
            ).set_index("removed_filter")
        self.assertEqual(result.loc["", "trade_count"], 0)
        self.assertGreater(result.loc["first_5m_high_too_low", "trade_count"], 0)
        self.assertEqual(result.loc["first_5m_high_too_low", "causal_valid"], 1)

    def test_minimum_sample_size_prevents_one_day_harmful_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counter = pd.DataFrame([{
                "session_date": SESSION, "filter": "first_5m_high_too_low", "rejected_candidates": 1,
                "candidates_unblocked_if_removed": 1, "later_peak_ge_3pct": 1,
                "missed_opportunity_rate_pct": 100.0, "avg_future_peak_pct": 10.0,
            }])
            portfolio = pd.DataFrame([
                {"session_date": SESSION, "removed_filter": "", "replay_supported": 1, "trade_count": 1, "winners": 1, "losers": 0, "net_pnl": 1, "incremental_net_pnl": 0},
                {"session_date": SESSION, "removed_filter": "first_5m_high_too_low", "replay_supported": 1, "trade_count": 2, "winners": 2, "losers": 0, "net_pnl": 10, "incremental_net_pnl": 9},
            ])
            counter_path = root / "counter.csv"
            portfolio_path = root / "portfolio.csv"
            counter.to_csv(counter_path, index=False)
            portfolio.to_csv(portfolio_path, index=False)
            paths = aggregate_range(
                [SESSION], [{"counterfactual": counter_path, "portfolio": portfolio_path}],
                output_dir=root, minimum_sample_size=30,
            )
            summary = pd.read_csv(paths["monthly_summary"])
        self.assertEqual(summary.iloc[0]["classification"], "NEUTRAL")
        self.assertIn("BASELINE ONLY", summary.iloc[0]["classification_basis"])


if __name__ == "__main__":
    unittest.main()
