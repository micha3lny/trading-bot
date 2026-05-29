from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_trading.control.control_api import process_history_collector_commands
from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    evaluate_risk_guard,
    handle_ibkr_order_rejection_event,
    open_position_price_diagnostics,
    persist_managed_positions,
    process_fill_lifecycle_diagnostics,
)
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


LIFECYCLE_FIELDS = [
    "recorded_at", "strategy", "event", "symbol", "action", "quantity", "price", "order_id",
    "execution_id", "reason", "entry_price", "peak_price", "pnl_pct",
    "decision_bid", "decision_ask", "decision_mid", "decision_last",
    "spread_pct", "fill_price", "fill_latency_ms",
    "estimated_commission", "realized_slippage_bps",
    "raw_json",
]


class FakeProcess:
    def __init__(self, rc=None):
        self._rc = rc
        self.pid = 1234
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True
        self._rc = -15

    def kill(self):
        self.killed = True
        self._rc = -9

    def wait(self, timeout=None):
        return self._rc


class BrokenProcess(FakeProcess):
    def poll(self):
        raise RuntimeError("stale handle")


class FakeContract:
    symbol = "RKLB"


class FakePortfolioItem:
    contract = FakeContract()
    position = 5


class FakeIB:
    def portfolio(self):
        return [FakePortfolioItem()]


class PreSqliteHardeningTests(unittest.TestCase):
    def test_collector_supervisor_failed_returncode_is_reported_and_cleared(self) -> None:
        state = {
            "history_collector_process": FakeProcess(rc=75),
            "history_collector_running_command": {"id": "cmd1", "end_date": "2026-05-22", "session_type": "RTH"},
            "history_collector_started_monotonic": time.monotonic() - 3,
        }

        process_history_collector_commands(runtime_state=state)

        self.assertIsNone(state["history_collector_process"])
        self.assertEqual(state["history_collector_last_returncode"], 75)
        self.assertIsNone(state.get("history_collector_last_run_key"))

    def test_collector_supervisor_timeout_terminates_process(self) -> None:
        proc = FakeProcess(rc=None)
        state = {
            "history_collector_process": proc,
            "history_collector_running_command": {"id": "cmd1"},
            "history_collector_started_monotonic": time.monotonic() - 121,
            "history_collector_max_runtime_minutes": 2 / 60,
        }

        process_history_collector_commands(runtime_state=state)

        self.assertTrue(proc.terminated)
        self.assertIsNone(state["history_collector_process"])
        self.assertEqual(state["history_collector_last_returncode"], -9)

    def test_collector_supervisor_clears_stale_handle(self) -> None:
        state = {
            "history_collector_process": BrokenProcess(rc=None),
            "history_collector_running_command": {"id": "cmd1"},
            "history_collector_started_monotonic": time.monotonic(),
        }

        process_history_collector_commands(runtime_state=state)

        self.assertIsNone(state["history_collector_process"])

    def test_max_daily_loss_blocks_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp)
            write_csv(
                recorder.path("portfolio_snapshots.csv"),
                [{"recorded_at": "2026-05-26T14:00:00+00:00", "realized_pnl": "-51", "unrealized_pnl": "0"}],
                ["recorded_at", "realized_pnl", "unrealized_pnl"],
            )
            args = SimpleNamespace(
                risk_guard_enabled=True,
                max_daily_loss_usd=50,
                max_trades_per_day=0,
                max_open_positions=0,
                max_gross_exposure_usd=0,
                max_single_position_usd=0,
                position_usd=1000,
            )

            status = evaluate_risk_guard(recorder, {}, {}, args, symbol="RKLB", candidate_notional=500)

            self.assertTrue(status["blocked"])
            self.assertEqual(status["reason"], "max_daily_loss")

    def test_max_trades_per_day_blocks_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp)
            write_csv(
                recorder.path("trade_lifecycle.csv"),
                [{"event": "BUY_ORDER_SENT", "symbol": "RKLB"}],
                ["event", "symbol"],
            )
            args = SimpleNamespace(
                risk_guard_enabled=True,
                max_daily_loss_usd=0,
                max_trades_per_day=1,
                max_open_positions=0,
                max_gross_exposure_usd=0,
                max_single_position_usd=0,
                position_usd=1000,
            )

            status = evaluate_risk_guard(recorder, {}, {}, args, symbol="RKLB", candidate_notional=500)

            self.assertTrue(status["blocked"])
            self.assertEqual(status["reason"], "max_trades_per_day")

    def test_max_open_positions_counts_exit_pending_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp)
            pos = ManagedPosition("RKLB", object(), 10, 10.0, "t", 11.0, active=True, exit_sent=True)
            args = SimpleNamespace(
                risk_guard_enabled=True,
                max_daily_loss_usd=0,
                max_trades_per_day=0,
                max_open_positions=1,
                max_gross_exposure_usd=0,
                max_single_position_usd=0,
                position_usd=1000,
            )

            status = evaluate_risk_guard(recorder, {"RKLB": pos}, {"RKLB": {"price": 10.5}}, args, symbol="AKTX", candidate_notional=500)

            self.assertTrue(status["blocked"])
            self.assertEqual(status["reason"], "max_open_positions")

    def test_delayed_fill_after_cancel_records_lifecycle_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp)
            write_csv(
                recorder.path("trade_lifecycle.csv"),
                [
                    {"event": "SELL_ORDER_SENT", "symbol": "RKLB", "action": "SELL", "quantity": "10", "order_id": "7"},
                    {"event": "EXIT_ORDER_CANCELLED", "symbol": "RKLB", "order_id": "7"},
                ],
                LIFECYCLE_FIELDS,
            )
            write_csv(
                recorder.path("fills.csv"),
                [{"execution_id": "E1", "symbol": "RKLB", "action": "SLD", "quantity": "5", "fill_price": "11", "order_id": "7"}],
                ["execution_id", "symbol", "action", "quantity", "fill_price", "order_id"],
            )
            pos = ManagedPosition("RKLB", object(), 10, 10.0, "t", 11.0, active=True, exit_sent=True)
            state: dict = {}

            emitted = process_fill_lifecycle_diagnostics(FakeIB(), recorder, {"RKLB": pos}, state)

            self.assertGreaterEqual(emitted, 1)
            with recorder.path("trade_lifecycle.csv").open() as fh:
                events = [row["event"] for row in csv.DictReader(fh)]
            self.assertIn("DELAYED_FILL_AFTER_CANCEL", events)
            self.assertIn("ORDER_CANCEL_BUT_POSITION_EXISTS", events)

    def test_partial_exit_keeps_managed_position_active_with_remaining_qty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp)
            write_csv(
                recorder.path("trade_lifecycle.csv"),
                [{"event": "SELL_ORDER_SENT", "symbol": "RKLB", "action": "SELL", "quantity": "10", "order_id": "8"}],
                LIFECYCLE_FIELDS,
            )
            write_csv(
                recorder.path("fills.csv"),
                [{"execution_id": "E2", "symbol": "RKLB", "action": "SLD", "quantity": "5", "fill_price": "11", "order_id": "8"}],
                ["execution_id", "symbol", "action", "quantity", "fill_price", "order_id"],
            )
            pos = ManagedPosition("RKLB", object(), 10, 10.0, "t", 11.0, active=True, exit_sent=True)

            state: dict = {}
            process_fill_lifecycle_diagnostics(FakeIB(), recorder, {"RKLB": pos}, state)

            self.assertTrue(pos.active)
            self.assertEqual(pos.quantity, 5)
            self.assertEqual(state["partial_fill_states"]["8"]["state"], "EXIT_PARTIAL")

    def test_persist_managed_positions_writes_market_price_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-05-28")
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            recorder.sqlite_store = store
            pos = ManagedPosition("CRBP", object(), 3, 10.0, "2026-05-28T13:31:00+00:00", 10.0, active=True)

            persist_managed_positions(
                recorder,
                {"CRBP": pos},
                latest_snapshots={"CRBP": {"price": 10.5, "observed_at": "2026-05-28T13:40:00+00:00"}},
            )

            row = store.get_latest_position("CRBP", "v67_top100_live_safe_expansion_v46_wide_trail")
            store.close()
            raw = json.loads(row["raw_json"])
            file_payload = json.loads(recorder.path("managed_positions.json").read_text())

            self.assertEqual(raw["market_price"], 10.5)
            self.assertEqual(raw["market_price_at"], "2026-05-28T13:40:00+00:00")
            self.assertEqual(raw["market_price_source"], "live_ticker")
            self.assertEqual(file_payload["positions"]["CRBP"]["market_price"], 10.5)

    def test_open_position_price_diagnostics_counts_missing_prices(self) -> None:
        pos = ManagedPosition("CRBP", object(), 3, 10.0, "2026-05-28T13:31:00+00:00", 10.0, active=True)

        ok, missing = open_position_price_diagnostics({"CRBP": pos}, latest_snapshots={})

        self.assertEqual(ok, 0)
        self.assertEqual(missing, 1)

    def test_error_201_rejects_entry_and_removes_managed_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-05-28")
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            recorder.sqlite_store = store
            managed_positions = {
                "CONL": ManagedPosition(
                    "CONL",
                    SimpleNamespace(symbol="CONL"),
                    2,
                    50.0,
                    "2026-05-28T13:35:00+00:00",
                    50.0,
                    active=True,
                    entry_fill_verified=False,
                )
            }
            runtime_state = {
                "entry_order_by_order_id": {
                    "123": {"symbol": "CONL", "quantity": 2, "price": 50.0}
                },
                "runtime_ineligible_path": str(Path(tmp) / "runtime" / "ineligible_symbols.json"),
            }

            handled = handle_ibkr_order_rejection_event(
                recorder,
                managed_positions,
                runtime_state,
                order_id=123,
                error_code=201,
                message="Order rejected - No Trading Permission, Customer Ineligible. product does not have a KID in English",
            )

            self.assertTrue(handled)
            self.assertNotIn("CONL", managed_positions)
            self.assertIn("CONL", runtime_state["ineligible_symbols"])
            lifecycle = recorder.path("trade_lifecycle.csv").read_text()
            self.assertIn("ENTRY_ORDER_REJECTED", lifecycle)
            rows = store.query("SELECT status, active, raw_json FROM positions WHERE symbol = 'CONL'")
            store.close()
            self.assertEqual(rows[0]["status"], "ENTRY_REJECTED")
            self.assertEqual(rows[0]["active"], 0)
            self.assertIn("no_trading_permission_kid", rows[0]["raw_json"])
            stored_positions = json.loads(recorder.path("managed_positions.json").read_text())["positions"]
            self.assertEqual(stored_positions, {})
            ineligible_payload = json.loads((Path(tmp) / "runtime" / "ineligible_symbols.json").read_text())
            self.assertEqual(ineligible_payload["symbols"]["CONL"]["reason"], "no_trading_permission_kid")
            self.assertEqual(ineligible_payload["symbols"]["CONL"]["ibkr_error_code"], 201)

    def test_rejected_symbol_is_blocked_from_reentry_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(tmp, session_date="2026-05-28")
            managed_positions: dict[str, ManagedPosition] = {}
            runtime_state = {
                "entry_order_by_order_id": {
                    "201": {"symbol": "ARMG", "quantity": 1, "price": 30.0}
                }
            }

            handle_ibkr_order_rejection_event(
                recorder,
                managed_positions,
                runtime_state,
                order_id=201,
                error_code=201,
                message="No Trading Permission, Customer Ineligible KID",
            )

            self.assertIn("ARMG", runtime_state["ineligible_symbols"])


if __name__ == "__main__":
    unittest.main()
