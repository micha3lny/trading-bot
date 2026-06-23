from __future__ import annotations

import unittest

from src.live_trading.v67_live_top100_expansion_paper_trader import (
    SymbolState,
    entry_minute_capacity,
    is_stale_ready_candidate,
    low_live_entry_score_blocked,
    mark_entry_block_state,
    ready_candidate_diagnostics,
    ready_candidate_age_seconds,
    ready_candidate_rejection_reason,
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

    def test_recent_candidate_before_unblock_is_still_rejected(self) -> None:
        state = SymbolState(
            symbol="LPTH",
            ready_since_ts=195.0,
            ready_since_utc="2026-05-28T18:16:55+00:00",
            signal_source="live",
            last_live_update_ts=195.0,
        )
        runtime_state = {"last_unblock_timestamp": 200.0}

        self.assertTrue(
            is_stale_ready_candidate(
                state,
                runtime_state,
                max_age_seconds=60.0,
                now_ts=220.0,
            )
        )
        self.assertEqual(ready_candidate_age_seconds(state, 220.0), 25.0)
        self.assertEqual(
            ready_candidate_rejection_reason(state, runtime_state, max_age_seconds=60.0, now_ts=220.0),
            "signal_before_last_unblock",
        )

    def test_backfill_reconstructed_ready_candidate_is_context_only(self) -> None:
        state = SymbolState(
            symbol="AKTX",
            ready_since_ts=210.0,
            ready_since_utc="2026-05-28T18:17:10+00:00",
            signal_source="reconstructed",
            last_update_source="reconstructed",
        )
        runtime_state = {"last_unblock_timestamp": 200.0, "last_restart_unblock_timestamp": 200.0}

        self.assertEqual(
            ready_candidate_rejection_reason(state, runtime_state, max_age_seconds=60.0, now_ts=220.0),
            "signal_source_reconstructed",
        )

    def test_fresh_live_update_after_unblock_is_allowed(self) -> None:
        state = SymbolState(
            symbol="CRSR",
            ready_since_ts=205.0,
            ready_since_utc="2026-05-28T18:17:05+00:00",
            signal_source="live",
            last_live_update_ts=205.0,
            last_live_update_utc="2026-05-28T18:17:05+00:00",
        )
        runtime_state = {"last_unblock_timestamp": 200.0, "last_restart_unblock_timestamp": 200.0}

        self.assertEqual(ready_candidate_rejection_reason(state, runtime_state, max_age_seconds=60.0, now_ts=220.0), "")

    def test_paper_buy_diagnostics_include_live_signal_source_fields(self) -> None:
        state = SymbolState(
            symbol="CRSR",
            ready_since_ts=205.0,
            ready_since_utc="2026-05-28T18:17:05+00:00",
            signal_source="live",
            last_live_update_ts=205.0,
            last_live_update_utc="2026-05-28T18:17:05+00:00",
        )
        runtime_state = {
            "last_unblock_timestamp": 200.0,
            "last_unblock_utc": "2026-05-28T18:17:00+00:00",
            "last_restart_unblock_timestamp": 200.0,
            "last_restart_unblock_utc": "2026-05-28T18:17:00+00:00",
        }

        payload = ready_candidate_diagnostics(
            state,
            {"score": 81.57},
            runtime_state,
            now_ts=220.0,
            ranking_position=1,
        )

        self.assertEqual(payload["signal_source"], "live")
        self.assertEqual(payload["last_live_update_at"], "2026-05-28T18:17:05+00:00")
        self.assertEqual(payload["last_restart_unblock_time"], "2026-05-28T18:17:00+00:00")
        self.assertEqual(payload["ranking_position"], 1)

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

    def test_min_live_entry_score_blocks_only_when_enabled(self) -> None:
        self.assertFalse(low_live_entry_score_blocked(19.35, 0))
        self.assertFalse(low_live_entry_score_blocked(82.74, 80))
        self.assertTrue(low_live_entry_score_blocked(19.35, 20))
        self.assertTrue(low_live_entry_score_blocked(None, 1))


if __name__ == "__main__":
    unittest.main()
