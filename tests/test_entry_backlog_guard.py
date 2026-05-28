from __future__ import annotations

import unittest

from src.live_trading.v67_live_top100_expansion_paper_trader import (
    SymbolState,
    entry_minute_capacity,
    is_stale_ready_candidate,
    mark_entry_block_state,
    ready_candidate_age_seconds,
    record_entry_submission,
)


class EntryBacklogGuardTests(unittest.TestCase):
    def test_stale_ready_candidate_after_unblock_is_blocked(self) -> None:
        state = SymbolState(symbol="APPS", ready_since_ts=100.0, ready_since_utc="2026-05-28T18:15:00+00:00")
        runtime_state = {"last_unblock_timestamp": 200.0}

        self.assertTrue(
            is_stale_ready_candidate(
                state,
                runtime_state,
                max_age_seconds=60.0,
                now_ts=220.0,
            )
        )

    def test_ready_after_unblock_is_not_stale(self) -> None:
        state = SymbolState(symbol="CRSR", ready_since_ts=205.0, ready_since_utc="2026-05-28T18:17:05+00:00")
        runtime_state = {"last_unblock_timestamp": 200.0}

        self.assertFalse(
            is_stale_ready_candidate(
                state,
                runtime_state,
                max_age_seconds=60.0,
                now_ts=260.0,
            )
        )

    def test_recent_candidate_before_unblock_is_allowed_within_age_threshold(self) -> None:
        state = SymbolState(symbol="LPTH", ready_since_ts=195.0, ready_since_utc="2026-05-28T18:16:55+00:00")
        runtime_state = {"last_unblock_timestamp": 200.0}

        self.assertFalse(
            is_stale_ready_candidate(
                state,
                runtime_state,
                max_age_seconds=60.0,
                now_ts=220.0,
            )
        )
        self.assertEqual(ready_candidate_age_seconds(state, 220.0), 25.0)

    def test_unblock_timestamp_is_recorded_on_transition(self) -> None:
        runtime_state: dict[str, object] = {}
        mark_entry_block_state(runtime_state, True, 100.0)
        mark_entry_block_state(runtime_state, False, 130.0)

        self.assertEqual(runtime_state["last_unblock_timestamp"], 130.0)
        self.assertFalse(runtime_state["entry_previous_entries_blocked"])

    def test_entry_minute_capacity_limits_submissions(self) -> None:
        runtime_state: dict[str, object] = {}
        for offset in range(5):
            record_entry_submission(runtime_state, 100.0 + offset)

        self.assertEqual(entry_minute_capacity(runtime_state, 5, 130.0), 0)
        self.assertEqual(entry_minute_capacity(runtime_state, 5, 165.0), 5)


if __name__ == "__main__":
    unittest.main()
