from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import handle_ibkr_disconnect_and_recover


class FakeIB:
    def __init__(self, connected: bool = False) -> None:
        self.connected = connected
        self.disconnect_calls = 0

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False


class ReconnectSupervisorTests(unittest.TestCase):
    def test_disconnected_reconnect_attempt_resubscribes_and_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ib = FakeIB(connected=False)
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-21")
            runtime_state: dict = {}
            calls: list[str] = []

            def connect_fn(fake_ib, args):
                calls.append("connect")
                self.assertTrue(runtime_state["entries_blocked"])
                self.assertEqual(runtime_state["entries_blocked_reason"], "ibkr_reconnect")
                fake_ib.connected = True

            def resubscribe_fn(fake_ib, contracts):
                calls.append("resubscribe")
                return {"RKLB": object()}

            def reconcile_fn(fake_ib, rec, managed, contract_by_symbol, state, **kwargs):
                calls.append("reconcile")
                self.assertEqual(kwargs["log_prefix"], "POST_RECONNECT_RECONCILIATION")
                self.assertEqual(kwargs["reason_prefix"], "post_reconnect_reconciliation")
                state["entries_blocked"] = False
                state["entries_blocked_reason"] = ""
                return {"clean": True}

            result = handle_ibkr_disconnect_and_recover(
                ib,
                recorder,
                {},
                {},
                [("RKLB", object())],
                runtime_state,
                SimpleNamespace(reconnect_wait_seconds=0.0),
                reason="unit_test",
                seen_fills=set(),
                connect_fn=connect_fn,
                resubscribe_fn=resubscribe_fn,
                reconcile_fn=reconcile_fn,
                record_account_snapshot_fn=lambda *_: calls.append("account"),
                record_recent_fills_fn=lambda *_: calls.append("fills"),
            )

            self.assertTrue(result["ok"])
            self.assertEqual(calls, ["connect", "resubscribe", "account", "fills", "reconcile"])
            self.assertEqual(result["tickers"].keys(), {"RKLB"})
            self.assertFalse(runtime_state["reconnect_active"])
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertTrue(runtime_state["ibkr_connected"])
            self.assertTrue(runtime_state["post_reconnect_reconciliation_done"])

    def test_connected_ib_is_disconnected_before_reconnect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ib = FakeIB(connected=True)
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-21")
            runtime_state: dict = {}

            def connect_fn(fake_ib, args):
                fake_ib.connected = True

            def reconcile_fn(fake_ib, rec, managed, contract_by_symbol, state, **kwargs):
                state["entries_blocked"] = False
                state["entries_blocked_reason"] = ""
                return {"clean": True}

            result = handle_ibkr_disconnect_and_recover(
                ib,
                recorder,
                {},
                {},
                [],
                runtime_state,
                SimpleNamespace(reconnect_wait_seconds=0.0),
                reason="unit_test_connected",
                connect_fn=connect_fn,
                resubscribe_fn=lambda *_: {},
                reconcile_fn=reconcile_fn,
                record_account_snapshot_fn=lambda *_: None,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(ib.disconnect_calls, 1)

    def test_reconnect_failure_keeps_entries_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ib = FakeIB(connected=False)
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-21")
            runtime_state: dict = {}

            def connect_fn(fake_ib, args):
                raise RuntimeError("gateway still down")

            result = handle_ibkr_disconnect_and_recover(
                ib,
                recorder,
                {},
                {},
                [],
                runtime_state,
                SimpleNamespace(reconnect_wait_seconds=0.0),
                reason="unit_test_failure",
                connect_fn=connect_fn,
                resubscribe_fn=lambda *_: {},
                reconcile_fn=lambda *_args, **_kwargs: {"clean": True},
                record_account_snapshot_fn=lambda *_: None,
            )

            self.assertFalse(result["ok"])
            self.assertTrue(runtime_state["reconnect_active"])
            self.assertTrue(runtime_state["entries_blocked"])
            self.assertEqual(runtime_state["entries_blocked_reason"], "ibkr_reconnect")
            self.assertFalse(runtime_state["post_reconnect_reconciliation_done"])
            self.assertIn("gateway still down", runtime_state["reconnect_last_error"])

    def test_reconnect_success_unblocks_entries_after_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ib = FakeIB(connected=False)
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-21")
            runtime_state: dict = {"entries_blocked": False}

            def connect_fn(fake_ib, args):
                fake_ib.connected = True

            def reconcile_fn(fake_ib, rec, managed, contract_by_symbol, state, **kwargs):
                self.assertTrue(state["entries_blocked"])
                state["entries_blocked"] = False
                state["entries_blocked_reason"] = ""
                return {"clean": False, "orphans": ["ASST"]}

            result = handle_ibkr_disconnect_and_recover(
                ib,
                recorder,
                {},
                {},
                [],
                runtime_state,
                SimpleNamespace(reconnect_wait_seconds=0.0),
                reason="unit_test_success",
                connect_fn=connect_fn,
                resubscribe_fn=lambda *_: {},
                reconcile_fn=reconcile_fn,
                record_account_snapshot_fn=lambda *_: None,
            )

            self.assertTrue(result["ok"])
            self.assertFalse(runtime_state["reconnect_active"])
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertEqual(runtime_state["entries_blocked_reason"], "")
            self.assertEqual(result["reconciliation"]["orphans"], ["ASST"])


if __name__ == "__main__":
    unittest.main()
