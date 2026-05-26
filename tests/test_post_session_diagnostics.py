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
            self.assertIn("ibkr commissions:     $0.70", text)
            self.assertIn("net actual pnl:       $4.30", text)

            watch = io.StringIO()
            with patch("sys.argv", ["v67_daily_report", "--date", "2026-05-22", "--recorder-dir", str(root), "--sqlite-path", str(sqlite_path), "--watch-summary"]), contextlib.redirect_stdout(watch):
                v67_daily_report.main()
            self.assertIn("SESSION 2026-05-22", watch.getvalue())
            self.assertIn("net=4.30", watch.getvalue())

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
