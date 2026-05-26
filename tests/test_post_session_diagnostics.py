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
    hard_eod_flatten_portfolio,
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
            self.assertIn("net=4.30", watch.getvalue())
            self.assertIn("CLOSED POSITIONS", watch.getvalue())
            self.assertIn("GROSS", watch.getvalue())
            self.assertIn("IBKR_COMM", watch.getvalue())
            self.assertIn("NET_ACTUAL_OR_EST", watch.getvalue())
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
            self.assertIn("gross=5.00", text)
            self.assertIn("ibkr_comm_confirmed=0.00", text)
            self.assertIn("est_fallback=1.00", text)
            self.assertIn("net=4.00", text)
            self.assertIn("comm_coverage=0/2", text)

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
