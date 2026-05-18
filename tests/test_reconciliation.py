from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.live_trading.order_lifecycle.models import LifecycleEvent, LifecycleEventType, PositionRecord, PositionState
from src.live_trading.order_lifecycle.reconciliation import build_reconciliation_report, load_lifecycle_events
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore


class ReconciliationReportTests(unittest.TestCase):
    def test_clean_report(self) -> None:
        positions = {
            "RKLB": PositionRecord(
                symbol="RKLB",
                strategy="v67",
                session_date="2026-05-18",
                state=PositionState.OPEN,
                open_quantity=10,
            )
        }
        report = build_reconciliation_report(positions, {"RKLB": 10})
        self.assertTrue(report.clean)
        self.assertEqual(report.managed_symbols, ["RKLB"])
        self.assertEqual(report.ibkr_symbols, ["RKLB"])

    def test_orphan_ibkr_position(self) -> None:
        report = build_reconciliation_report({}, {"RKLB": 10})
        self.assertFalse(report.clean)
        self.assertEqual(report.orphan_in_ibkr, ["RKLB"])

    def test_local_position_missing_in_ibkr(self) -> None:
        positions = {
            "RKLB": PositionRecord(
                symbol="RKLB",
                strategy="v67",
                session_date="2026-05-18",
                state=PositionState.OPEN,
                open_quantity=10,
            )
        }
        report = build_reconciliation_report(positions, {})
        self.assertFalse(report.clean)
        self.assertEqual(report.missing_in_ibkr, ["RKLB"])

    def test_quantity_drift(self) -> None:
        positions = {
            "RKLB": PositionRecord(
                symbol="RKLB",
                strategy="v67",
                session_date="2026-05-18",
                state=PositionState.OPEN,
                open_quantity=10,
            )
        }
        report = build_reconciliation_report(positions, {"RKLB": 7})
        self.assertFalse(report.clean)
        self.assertEqual(report.quantity_drift["RKLB"]["managed_quantity"], 10)
        self.assertEqual(report.quantity_drift["RKLB"]["ibkr_quantity"], 7)

    def test_pending_order_makes_report_not_clean(self) -> None:
        report = build_reconciliation_report({}, {}, open_orders=[{"ib_order_id": "123"}])
        self.assertFalse(report.clean)
        self.assertEqual(report.pending_order_ids, ["123"])

    def test_lifecycle_symbols_loaded_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "order_lifecycle.jsonl"
            store = JsonlLifecycleStore(path)
            store.append_event(
                LifecycleEvent(
                    event_type=LifecycleEventType.ENTRY_SIGNAL,
                    symbol="RKLB",
                    strategy="v67",
                )
            )
            events = load_lifecycle_events(path)
            report = build_reconciliation_report({}, {}, lifecycle_events=events)
            self.assertEqual(report.lifecycle_symbols, ["RKLB"])


if __name__ == "__main__":
    unittest.main()
