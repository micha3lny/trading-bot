from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.live_trading.analytics import v67_daily_report
from src.live_trading.order_lifecycle.models import LifecycleEventType
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore
from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    apply_pending_eod_entry_block,
    enforce_eod_flatten_if_due,
    hard_eod_flatten_portfolio,
    load_pending_eod_flatten,
    process_pending_eod_flatten_retry,
    process_portfolio_sync_pending_eod_retry,
    startup_reconcile_runtime_state,
)


class FakeContract:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.currency = "USD"


class FakePortfolioItem:
    def __init__(self, symbol: str, position: float, market_price: float = 10.0) -> None:
        self.contract = FakeContract(symbol)
        self.position = position
        self.marketPrice = market_price
        self.marketValue = position * market_price
        self.averageCost = market_price


class FakeIB:
    def __init__(self, portfolio=None, open_trades=None) -> None:
        self._portfolio = list(portfolio or [])
        self._open_trades = list(open_trades or [])

    def isConnected(self):
        return True

    def portfolio(self):
        return list(self._portfolio)

    def openTrades(self):
        return list(self._open_trades)

    def openOrders(self):
        return []

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        order.orderId = 10
        return SimpleNamespace(contract=contract, order=order, orderStatus=SimpleNamespace(status="Submitted"), log=[])

    def sleep(self, _seconds):
        return None


class DisconnectedFakeIB(FakeIB):
    def isConnected(self):
        return False


def recorder_in_tmp(tmp: str) -> LiveDataRecorder:
    return LiveDataRecorder(Path(tmp), session_date="2026-05-22")


class PostSessionDiagnosticsTests(unittest.TestCase):
    def test_fractional_orphan_logs_once_and_suppresses_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIB(portfolio=[FakePortfolioItem("ASST", 0.2)])
            runtime_state = {}

            startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)
            startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            fractional = [
                row for row in events
                if row["event_type"] == LifecycleEventType.POSITION_DRIFT_DETECTED.value
                and row.get("reason") == "startup_reconciliation_fractional_orphan_manual_required"
            ]
            self.assertEqual(len(fractional), 1)
            self.assertEqual(runtime_state["fractional_orphan_manual_required_suppressed_total"], 1)

    def test_eod_final_status_clean_when_portfolio_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state = {}

            hard_eod_flatten_portfolio(
                FakeIB(),
                recorder,
                {},
                SimpleNamespace(eod_retry_seconds=0.0, eod_max_retries=1),
                runtime_state,
                reason="unit_test",
            )

            summary = json.loads(recorder.path("eod_summary.json").read_text())
            self.assertTrue(summary["clean"])
            self.assertEqual(summary["open_positions"], 0)

    def test_eod_final_status_not_clean_when_fractional_orphan_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state = {}

            hard_eod_flatten_portfolio(
                FakeIB(portfolio=[FakePortfolioItem("ASST", 0.2)]),
                recorder,
                {},
                SimpleNamespace(eod_retry_seconds=0.0, eod_max_retries=1),
                runtime_state,
                reason="unit_test",
            )

            summary = json.loads(recorder.path("eod_summary.json").read_text())
            self.assertFalse(summary["clean"])
            self.assertEqual(summary["fractional_orphans"], ["ASST"])
            self.assertTrue(summary["pending_eod_flatten"])
            pending = json.loads(recorder.path("eod_pending.json").read_text())
            self.assertTrue(pending["pending_eod_flatten"])
            with recorder.path("trade_lifecycle.csv").open(encoding="utf-8") as fh:
                events = [row["event"] for row in csv.DictReader(fh)]
            self.assertIn("EOD_FLATTEN_GIVEUP", events)

    def test_eod_flatten_failure_persists_pending_and_startup_reload_retries_until_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state = {}

            submitted = hard_eod_flatten_portfolio(
                DisconnectedFakeIB(portfolio=[FakePortfolioItem("RKLB", 3)]),
                recorder,
                {},
                SimpleNamespace(eod_retry_seconds=0.0, eod_max_retries=1),
                runtime_state,
                reason="unit_test_disconnect",
            )

            self.assertEqual(submitted, 0)
            self.assertTrue(runtime_state["pending_eod_flatten"])
            self.assertTrue(json.loads(recorder.path("eod_pending.json").read_text())["pending_eod_flatten"])

            restored_state: dict = {}
            self.assertTrue(load_pending_eod_flatten(recorder, restored_state))
            retry_ib = FakeIB(portfolio=[FakePortfolioItem("RKLB", 3)])
            process_pending_eod_flatten_retry(
                retry_ib,
                recorder,
                {},
                SimpleNamespace(eod_retry_seconds=0.0, eod_max_retries=1),
                restored_state,
                reason="startup_pending_eod_flatten",
                force=True,
            )
            self.assertEqual(len(retry_ib.openTrades()), 0)
            self.assertTrue(restored_state["pending_eod_flatten"])

            retry_ib._portfolio = []
            process_pending_eod_flatten_retry(
                retry_ib,
                recorder,
                {},
                SimpleNamespace(eod_retry_seconds=0.0, eod_max_retries=1),
                restored_state,
                reason="startup_pending_eod_flatten",
                force=True,
            )
            self.assertFalse(restored_state["pending_eod_flatten"])
            self.assertFalse(json.loads(recorder.path("eod_pending.json").read_text())["pending_eod_flatten"])
            with recorder.path("trade_lifecycle.csv").open(encoding="utf-8") as fh:
                events = [row["event"] for row in csv.DictReader(fh)]
            self.assertIn("EOD_FLATTEN_RETRY", events)
            self.assertIn("EOD_FLATTEN_SUCCESS", events)

    def test_pending_eod_flatten_blocks_new_entries(self) -> None:
        runtime_state = {"pending_eod_flatten": True, "entries_blocked_reason": ""}

        self.assertTrue(apply_pending_eod_entry_block(runtime_state))

        self.assertEqual(runtime_state["entries_blocked_reason"], "pending_eod_flatten")

    def test_eod_active_failsafe_submits_and_persists_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state: dict = {}
            pos = ManagedPosition("RKLB", FakeContract("RKLB"), 3, 10.0, "t", 10.0, active=True)
            ib = FakeIB(portfolio=[FakePortfolioItem("RKLB", 3)])

            submitted = enforce_eod_flatten_if_due(
                ib,
                recorder,
                {"RKLB": pos},
                SimpleNamespace(enable_eod_flatten=True, eod_flatten_utc="19:45", eod_retry_seconds=60.0, eod_max_retries=1),
                runtime_state,
                eod_active=True,
                reason="unit_test_failsafe",
            )

            self.assertEqual(submitted, 1)
            self.assertTrue(runtime_state["pending_eod_flatten"])
            self.assertEqual(runtime_state["entries_blocked_reason"], "pending_eod_flatten")
            pending = json.loads(recorder.path("eod_pending.json").read_text())
            self.assertTrue(pending["pending_eod_flatten"])
            with recorder.path("trade_lifecycle.csv").open(encoding="utf-8") as fh:
                events = [row["event"] for row in csv.DictReader(fh)]
            self.assertIn("EOD_FLATTEN_SUBMIT", events)

    def test_eod_active_failsafe_respects_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state: dict = {}
            pos = ManagedPosition("RKLB", FakeContract("RKLB"), 3, 10.0, "t", 10.0, active=True)
            args = SimpleNamespace(enable_eod_flatten=True, eod_flatten_utc="19:45", eod_retry_seconds=60.0, eod_max_retries=1)
            ib = FakeIB(portfolio=[FakePortfolioItem("RKLB", 3)])
            stdout = io.StringIO()

            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.time.time", return_value=100.0):
                with contextlib.redirect_stdout(stdout):
                    enforce_eod_flatten_if_due(ib, recorder, {"RKLB": pos}, args, runtime_state, eod_active=True)
            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.time.time", return_value=102.0):
                with contextlib.redirect_stdout(stdout):
                    enforce_eod_flatten_if_due(ib, recorder, {"RKLB": pos}, args, runtime_state, eod_active=True)

            self.assertEqual(stdout.getvalue().count("EOD_FLATTEN_FAILSAFE_TRIGGER"), 1)

    def test_pending_eod_flatten_clears_only_after_ibkr_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state = {"pending_eod_flatten": True, "pending_eod_flatten_last_retry_ts": 0.0}
            args = SimpleNamespace(eod_retry_seconds=0.0, eod_max_retries=1)

            process_pending_eod_flatten_retry(
                FakeIB(portfolio=[FakePortfolioItem("RKLB", 3)]),
                recorder,
                {},
                args,
                runtime_state,
                reason="unit_test_pending",
                force=True,
            )
            self.assertTrue(runtime_state["pending_eod_flatten"])

            process_pending_eod_flatten_retry(
                FakeIB(portfolio=[]),
                recorder,
                {},
                args,
                runtime_state,
                reason="unit_test_pending",
                force=True,
            )
            self.assertFalse(runtime_state["pending_eod_flatten"])

    def test_portfolio_sync_triggers_immediate_pending_eod_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state = {"pending_eod_flatten": True, "pending_eod_flatten_last_retry_ts": 0.0}

            process_portfolio_sync_pending_eod_retry(
                FakeIB(portfolio=[FakePortfolioItem("RKLB", 3)]),
                recorder,
                {},
                SimpleNamespace(eod_retry_seconds=60.0, eod_max_retries=1),
                runtime_state,
            )

            with recorder.path("trade_lifecycle.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertTrue(any(row["event"] == "EOD_FLATTEN_RETRY" and row["reason"] == "portfolio_sync_pending_eod" for row in rows))

    def test_portfolio_sync_pending_eod_retry_respects_five_second_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            runtime_state = {"pending_eod_flatten": True, "pending_eod_flatten_last_retry_ts": 100.0}
            args = SimpleNamespace(eod_retry_seconds=60.0, eod_max_retries=1)
            ib = FakeIB(portfolio=[FakePortfolioItem("RKLB", 3)])

            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.time.time", return_value=103.0):
                sent = process_portfolio_sync_pending_eod_retry(ib, recorder, {}, args, runtime_state)
            self.assertEqual(sent, 0)
            self.assertFalse(recorder.path("trade_lifecycle.csv").exists())

            with patch("src.live_trading.v67_live_top100_expansion_paper_trader.time.time", return_value=106.0):
                process_portfolio_sync_pending_eod_retry(ib, recorder, {}, args, runtime_state)
            self.assertTrue(recorder.path("trade_lifecycle.csv").exists())

    def test_daily_report_includes_exit_simulation_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "2026-05-22"
            session.mkdir(parents=True)
            lifecycle = session / "trade_lifecycle.csv"
            with lifecycle.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "quantity", "price", "entry_price", "peak_price", "peak_gain_pct", "recorded_at", "reason"])
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10.5", "entry_price": "10", "peak_price": "10.4", "peak_gain_pct": "4", "recorded_at": "2026-05-22T13:40:00+00:00", "reason": "trail"})

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", tmp]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            text = out.getvalue()
            self.assertIn("=== EXIT SIMULATION ===", text)
            self.assertIn("fixed TP +3%", text)

    def test_daily_report_prefers_sqlite_commissions_and_watch_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-22"
            session.mkdir(parents=True)
            lifecycle = session / "trade_lifecycle.csv"
            with lifecycle.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "quantity", "price", "entry_price", "peak_price", "peak_gain_pct", "recorded_at", "reason"])
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10.5", "entry_price": "10", "peak_price": "10.4", "peak_gain_pct": "4", "recorded_at": "2026-05-22T13:40:00+00:00", "reason": "trail"})
            sqlite_path = root / "runtime.sqlite"
            store = SQLiteRuntimeStore(sqlite_path)
            store.upsert_execution({"execution_id": "B1", "session_date": "2026-05-22", "symbol": "RKLB", "side": "BOT", "quantity": 10, "price": 10, "commission": 0.35, "commission_source": "ibkr", "recorded_at": "2026-05-22T13:30:00+00:00"})
            store.upsert_execution({"execution_id": "S1", "session_date": "2026-05-22", "symbol": "RKLB", "side": "SLD", "quantity": 10, "price": 10.5, "commission": 0.35, "commission_source": "ibkr", "recorded_at": "2026-05-22T13:40:00+00:00"})
            store.upsert_trade({
                "trade_id": "RKLB-1",
                "session_date": "2026-05-22",
                "strategy_name": "v67",
                "symbol": "RKLB",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-22T13:30:00+00:00",
                "exit_fill_time": "2026-05-22T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 10.5,
                "quantity": 10,
                "gross_pnl": 5.0,
                "commission": 0.70,
                "net_pnl": 4.30,
                "mfe_pct": 4.0,
                "exit_reason": "trail",
            })
            store.close()

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path)]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            text = out.getvalue()
            self.assertIn("source:               sqlite", text)
            self.assertIn("ibkr_commissions_confirmed: $0.70", text)
            self.assertIn("net_after_commissions:$4.30", text)

            watch = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path), "--watch-summary"]), contextlib.redirect_stdout(watch):
                v67_daily_report.main()
            self.assertIn("SESSION SUMMARY 2026-05-22", watch.getvalue())
            self.assertIn("net actual pnl:       $4.30", watch.getvalue())
            self.assertIn("CLOSED POSITIONS", watch.getvalue())
            self.assertIn("GROSS", watch.getvalue())
            self.assertIn("IBKR_COMM", watch.getvalue())
            self.assertIn("NET_ACTUAL", watch.getvalue())
            self.assertNotIn("EST_FB", watch.getvalue())
            self.assertIn("fixed TP +3", watch.getvalue())

    def test_watch_summary_primary_net_uses_estimated_fallback_when_commission_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-22"
            session.mkdir(parents=True)
            lifecycle = session / "trade_lifecycle.csv"
            with lifecycle.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "quantity", "price", "entry_price", "peak_price", "peak_gain_pct", "recorded_at", "reason"])
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10.5", "entry_price": "10", "peak_price": "10.4", "peak_gain_pct": "4", "recorded_at": "2026-05-22T13:40:00+00:00", "reason": "trail"})
            with (session / "fills.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["execution_id", "symbol", "action", "quantity", "fill_price", "commission", "commission_source", "recorded_at"])
                writer.writeheader()
                writer.writerow({"execution_id": "B1", "symbol": "RKLB", "action": "BUY", "quantity": "10", "fill_price": "10", "commission": "", "commission_source": "missing", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"execution_id": "S1", "symbol": "RKLB", "action": "SELL", "quantity": "10", "fill_price": "10.5", "commission": "", "commission_source": "missing", "recorded_at": "2026-05-22T13:40:00+00:00"})

            watch = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--disable-sqlite", "--watch-summary"]), contextlib.redirect_stdout(watch):
                v67_daily_report.main()

            text = watch.getvalue()
            self.assertIn("gross closed pnl:     $5.00", text)
            self.assertIn("ibkr commissions:     $0.00", text)
            self.assertIn("net actual pnl:       $4.00", text)
            self.assertIn("commission coverage:  0/2", text)
            self.assertIn("EST_FB", text)
            self.assertIn("NET_ACTUAL*", text)
            self.assertIn("* includes estimated fallback for missing commissions", text)

    def test_active_managed_open_position_is_not_counted_as_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-22"
            session.mkdir(parents=True)
            lifecycle = session / "trade_lifecycle.csv"
            with lifecycle.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "quantity", "price", "entry_price", "peak_price", "recorded_at", "reason"])
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "RKLB", "quantity": "10", "price": "10", "peak_price": "10.8", "recorded_at": "2026-05-22T13:30:00+00:00"})
            (session / "managed_positions.json").write_text(json.dumps({
                "positions": [{"symbol": "RKLB", "quantity": 10, "entry_price": 10, "active": True}]
            }))
            with (session / "portfolio_snapshots.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["recorded_at", "positions_json"])
                writer.writeheader()
                writer.writerow({
                    "recorded_at": "2026-05-22T14:00:00+00:00",
                    "positions_json": json.dumps([{"symbol": "RKLB", "position": 10, "marketPrice": 10.5, "unrealizedPNL": 5.0}]),
                })

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root)]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            text = out.getvalue()
            self.assertLess(text.index("=== OPEN POSITIONS ==="), text.index("=== EXIT SIMULATION ==="))
            self.assertIn("RKLB", text)
            self.assertIn("=== CURRENT POSITION DIAGNOSTICS ===", text)
            self.assertIn("active_managed_positions: 1", text)
            self.assertIn("ibkr_positions:           1", text)
            self.assertIn("matched_positions:        1", text)
            self.assertIn("true_orphans:             0", text)
            self.assertIn("whole-share orphans:      0", text)

            watch = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--watch-summary"]), contextlib.redirect_stdout(watch):
                v67_daily_report.main()
            watch_text = watch.getvalue()
            self.assertIn("OPEN POSITIONS", watch_text)
            self.assertIn("SYM", watch_text)
            self.assertIn("FROM_PEAK", watch_text)
            self.assertIn("RKLB", watch_text)
            self.assertIn("CURRENT POSITION DIAGNOSTICS", watch_text)
            self.assertIn("true_orphans=0", watch_text)
            self.assertIn("whole_orphans=0", watch_text)

    def test_daily_report_uses_csv_when_sqlite_snapshot_is_partial_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-22"
            session.mkdir(parents=True)
            lifecycle_fields = ["event", "symbol", "quantity", "price", "entry_price", "peak_price", "peak_gain_pct", "recorded_at", "reason"]
            with (session / "trade_lifecycle.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=lifecycle_fields)
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "AAA", "quantity": "10", "price": "10", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "AAA", "quantity": "10", "price": "10.5", "entry_price": "10", "peak_price": "10.8", "peak_gain_pct": "8", "recorded_at": "2026-05-22T13:40:00+00:00", "reason": "trail"})
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "BBB", "quantity": "5", "price": "20", "recorded_at": "2026-05-22T13:31:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "BBB", "quantity": "5", "price": "19", "entry_price": "20", "peak_price": "20.4", "peak_gain_pct": "2", "recorded_at": "2026-05-22T13:45:00+00:00", "reason": "stop"})
            fill_fields = ["execution_id", "symbol", "action", "quantity", "fill_price", "commission", "commission_source", "recorded_at"]
            with (session / "fills.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fill_fields)
                writer.writeheader()
                writer.writerow({"execution_id": "A-B", "symbol": "AAA", "action": "BUY", "quantity": "10", "fill_price": "10", "commission": "0.25", "commission_source": "ibkr", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"execution_id": "A-S", "symbol": "AAA", "action": "SELL", "quantity": "10", "fill_price": "10.5", "commission": "0.25", "commission_source": "ibkr", "recorded_at": "2026-05-22T13:40:00+00:00"})
                writer.writerow({"execution_id": "B-B", "symbol": "BBB", "action": "BUY", "quantity": "5", "fill_price": "20", "commission": "", "commission_source": "missing", "recorded_at": "2026-05-22T13:31:00+00:00"})
                writer.writerow({"execution_id": "B-S", "symbol": "BBB", "action": "SELL", "quantity": "5", "fill_price": "19", "commission": "", "commission_source": "missing", "recorded_at": "2026-05-22T13:45:00+00:00"})
            sqlite_path = root / "runtime.sqlite"
            store = SQLiteRuntimeStore(sqlite_path)
            store.upsert_execution({"execution_id": "A-B", "session_date": "2026-05-22", "symbol": "AAA", "side": "BOT", "quantity": 10, "price": 10, "commission": 0.25, "commission_source": "ibkr", "recorded_at": "2026-05-22T13:30:00+00:00"})
            store.upsert_execution({"execution_id": "A-S", "session_date": "2026-05-22", "symbol": "AAA", "side": "SLD", "quantity": 10, "price": 10.5, "commission": 0.25, "commission_source": "ibkr", "recorded_at": "2026-05-22T13:40:00+00:00"})
            store.close()

            outputs = []
            for _ in range(2):
                out = io.StringIO()
                with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path)]), contextlib.redirect_stdout(out):
                    v67_daily_report.main()
                outputs.append(out.getvalue())

            self.assertIn("source:               csv", outputs[0])
            self.assertIn("closed trades:        2", outputs[0])
            self.assertIn("closed trades:        2", outputs[1])
            closed_rows_1 = [line for line in outputs[0].splitlines() if line.startswith(("AAA", "BBB"))]
            closed_rows_2 = [line for line in outputs[1].splitlines() if line.startswith(("AAA", "BBB"))]
            self.assertEqual(closed_rows_1, closed_rows_2)

    def test_watch_full_includes_open_and_closed_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-22"
            session.mkdir(parents=True)
            with (session / "trade_lifecycle.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "quantity", "price", "entry_price", "peak_price", "peak_gain_pct", "recorded_at", "reason"])
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "OPEN", "quantity": "3", "price": "10", "peak_price": "10.5", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "DONE", "quantity": "2", "price": "20", "recorded_at": "2026-05-22T13:35:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "DONE", "quantity": "2", "price": "21", "entry_price": "20", "peak_price": "21.5", "peak_gain_pct": "7.5", "recorded_at": "2026-05-22T13:50:00+00:00", "reason": "trail"})
            with (session / "portfolio_snapshots.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["recorded_at", "positions_json"])
                writer.writeheader()
                writer.writerow({"recorded_at": "2026-05-22T14:00:00+00:00", "positions_json": json.dumps([{"symbol": "OPEN", "position": 3, "marketPrice": 10.4, "unrealizedPNL": 1.2}])})

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--watch-full"]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            text = out.getvalue()
            self.assertIn("SESSION SUMMARY 2026-05-22", text)
            self.assertIn("OPEN POSITIONS", text)
            self.assertIn("CLOSED POSITIONS", text)
            self.assertIn("EXIT SIMULATION", text)
            self.assertIn("CURRENT POSITION DIAGNOSTICS", text)
            self.assertIn("OPEN", text)
            self.assertIn("DONE", text)

    def test_watch_full_summary_is_multiline_and_sorts_default_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-22"
            session.mkdir(parents=True)
            with (session / "trade_lifecycle.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "quantity", "price", "entry_price", "peak_price", "peak_gain_pct", "recorded_at", "reason"])
                writer.writeheader()
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "OPNA", "quantity": "1", "price": "10", "peak_price": "11", "recorded_at": "2026-05-22T13:30:00+00:00"})
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "OPNB", "quantity": "1", "price": "10", "peak_price": "10.2", "recorded_at": "2026-05-22T13:31:00+00:00"})
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "CLSW", "quantity": "1", "price": "20", "recorded_at": "2026-05-22T13:35:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "CLSW", "quantity": "1", "price": "23", "entry_price": "20", "peak_price": "24", "peak_gain_pct": "20", "recorded_at": "2026-05-22T13:50:00+00:00", "reason": "trail"})
                writer.writerow({"event": "BUY_ORDER_SENT", "symbol": "CLSL", "quantity": "1", "price": "30", "recorded_at": "2026-05-22T13:36:00+00:00"})
                writer.writerow({"event": "SELL_ORDER_SENT", "symbol": "CLSL", "quantity": "1", "price": "28", "entry_price": "30", "peak_price": "30.3", "peak_gain_pct": "1", "recorded_at": "2026-05-22T13:51:00+00:00", "reason": "stop"})
            with (session / "portfolio_snapshots.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["recorded_at", "positions_json"])
                writer.writeheader()
                writer.writerow({"recorded_at": "2026-05-22T14:00:00+00:00", "positions_json": json.dumps([
                    {"symbol": "OPNA", "position": 1, "marketPrice": 9.0, "unrealizedPNL": -1.0},
                    {"symbol": "OPNB", "position": 1, "marketPrice": 11.0, "unrealizedPNL": 1.0},
                ])})

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--watch-full"]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            lines = out.getvalue().splitlines()
            self.assertIn("closed trades:        2", lines)
            self.assertIn("open trades:          2", lines)
            open_rows = [line for line in lines if line.startswith(("OPNA", "OPNB"))]
            closed_rows = [line for line in lines if line.startswith(("CLSL", "CLSW"))]
            self.assertEqual([row[:4].strip() for row in open_rows], ["OPNA", "OPNB"])
            self.assertEqual([row[:4].strip() for row in closed_rows], ["CLSL", "CLSW"])

    def test_watch_sqlite_ignores_managed_json_when_latest_position_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-28"
            session.mkdir(parents=True)
            (session / "managed_positions.json").write_text(
                json.dumps({"positions": {"CONL": {"symbol": "CONL", "active": True, "quantity": 2, "entry_price": 50}}}),
                encoding="utf-8",
            )
            sqlite_path = root / "runtime.sqlite"
            store = SQLiteRuntimeStore(sqlite_path)
            store.upsert_position({
                "position_key": "v67:2026-05-28:CONL",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "CONL",
                "quantity": 2,
                "avg_price": 50,
                "active": 0,
                "status": "ENTRY_REJECTED",
                "updated_at": "2026-05-28T14:00:00+00:00",
                "raw_json": {"active": False, "reject_reason": "no_trading_permission_kid", "ibkr_error_code": 201},
            })
            store.record_runtime_event(
                event_time="2026-05-28T14:00:00+00:00",
                event_type="ENTRY_ORDER_REJECTED",
                severity="WARN",
                strategy_name="v67",
                session_date="2026-05-28",
                symbol="CONL",
                order_id="123",
                reason="no_trading_permission_kid",
                raw_json={"quantity": 2, "price": 50, "ibkr_error_code": 201},
            )
            store.close()

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-28", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path), "--watch-full"]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            text = out.getvalue()
            self.assertIn("open trades:          0", text)
            self.assertNotIn("OPEN POSITIONS", text)
            self.assertIn("REJECTED ENTRIES", text)
            self.assertIn("CONL", text)
            self.assertIn("stale_managed_ignored_count=1", text)

    def test_watch_sqlite_exit_order_without_ibkr_quantity_is_not_open_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-28"
            session.mkdir(parents=True)
            sqlite_path = root / "runtime.sqlite"
            store = SQLiteRuntimeStore(sqlite_path)
            store.upsert_position({
                "position_key": "v67:2026-05-28:EXITING",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "EXITING",
                "quantity": 3,
                "avg_price": 10,
                "ibkr_quantity": 0,
                "active": 1,
                "status": "EXIT_ORDER",
                "updated_at": "2026-05-28T20:05:00+00:00",
                "raw_json": {},
            })
            store.close()

            out = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-28", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path), "--watch-full"]), contextlib.redirect_stdout(out):
                v67_daily_report.main()

            text = out.getvalue()
            self.assertIn("open trades:          0", text)
            self.assertNotIn("OPEN POSITIONS", text)
            self.assertIn("exit_order_stale_count=1", text)

    def test_watch_sqlite_counts_are_stable_for_unchanged_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "2026-05-28"
            session.mkdir(parents=True)
            sqlite_path = root / "runtime.sqlite"
            store = SQLiteRuntimeStore(sqlite_path)
            store.upsert_trade({
                "trade_id": "T1",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "AKTX",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-28T13:35:00+00:00",
                "exit_fill_time": "2026-05-28T13:50:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 2,
                "gross_pnl": 2,
                "commission": 0.5,
                "net_pnl": 1.5,
            })
            store.close()

            outputs = []
            for _ in range(2):
                out = io.StringIO()
                with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-28", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path), "--watch-full"]), contextlib.redirect_stdout(out):
                    v67_daily_report.main()
                outputs.append([line for line in out.getvalue().splitlines() if "snapshot_loaded_at" not in line])

            self.assertEqual(outputs[0], outputs[1])
            self.assertIn("closed trades:        1", "\n".join(outputs[0]))
            self.assertIn("open trades:          0", "\n".join(outputs[0]))

    def test_tp_three_simulation_captures_trades_with_mfe_at_least_three(self) -> None:
        closed = [
            {"symbol": "AAA", "qty": 10, "buy": 10, "sell": 10.1, "gross": 1, "peak_gain_pct": 3.1},
            {"symbol": "BBB", "qty": 10, "buy": 10, "sell": 10.1, "gross": 1, "peak_gain_pct": 2.9},
        ]

        rows = v67_daily_report.simulate_exit_strategies(closed, commission_per_trade=1.0)
        tp3 = next(row for row in rows if row["name"] == "fixed TP +3%")

        self.assertEqual(tp3["captured"], 1)
        self.assertAlmostEqual(tp3["gross"], 4.0)


if __name__ == "__main__":
    unittest.main()
