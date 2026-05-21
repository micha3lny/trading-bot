from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.live_trading.order_lifecycle.models import OrderState, PositionState
from src.live_trading.order_lifecycle.reducer import reduce_lifecycle_events


def event(event_type: str, symbol: str = "RKLB", **kwargs):
    row = {
        "event_type": event_type,
        "symbol": symbol,
        "strategy": "v67",
        "recorded_at": kwargs.pop("recorded_at", "2026-05-18T13:30:00+00:00"),
    }
    row.update(kwargs)
    return row


class OrderLifecycleReducerTests(unittest.TestCase):
    def test_entry_submitted_then_filled_opens_position(self) -> None:
        snapshot = reduce_lifecycle_events(
            [
                event("ENTRY_ORDER_SUBMITTED", client_order_id="E1", quantity=10),
                event("ENTRY_ORDER_FILLED", client_order_id="E1", execution_id="X1", quantity=10, price=12.5),
            ]
        )

        position = snapshot.positions["RKLB"]
        order = snapshot.orders["E1"]
        self.assertEqual(position.state, PositionState.OPEN)
        self.assertEqual(position.open_quantity, 10)
        self.assertEqual(order.state, OrderState.FILLED)
        self.assertEqual(order.filled_quantity, 10)

    def test_entry_partial_is_open_with_partial_quantity(self) -> None:
        snapshot = reduce_lifecycle_events(
            [
                event("ENTRY_ORDER_SUBMITTED", client_order_id="E1", quantity=10),
                event("ENTRY_ORDER_PARTIAL", client_order_id="E1", execution_id="X1", quantity=4, price=12.5),
            ]
        )

        # Conservative runtime interpretation: a partial entry is already real IBKR exposure.
        position = snapshot.positions["RKLB"]
        self.assertEqual(position.state, PositionState.OPEN)
        self.assertEqual(position.open_quantity, 4)
        self.assertEqual(snapshot.orders["E1"].state, OrderState.PARTIAL)

    def test_exit_submitted_then_filled_closes_position(self) -> None:
        snapshot = reduce_lifecycle_events(
            [
                event("POSITION_ADOPTED", quantity=10, price=12.5),
                event("EXIT_ORDER_SUBMITTED", client_order_id="X1", quantity=10),
                event("EXIT_ORDER_FILLED", client_order_id="X1", execution_id="SELL1", quantity=10, price=13.0),
            ]
        )

        position = snapshot.positions["RKLB"]
        self.assertEqual(position.state, PositionState.CLOSED)
        self.assertEqual(position.open_quantity, 0)
        self.assertEqual(snapshot.orders["X1"].state, OrderState.FILLED)

    def test_duplicate_execution_id_does_not_double_count(self) -> None:
        snapshot = reduce_lifecycle_events(
            [
                event("ENTRY_ORDER_SUBMITTED", client_order_id="E1", quantity=10),
                event("ENTRY_ORDER_PARTIAL", client_order_id="E1", execution_id="DUP1", quantity=4, price=12.5),
                event("ENTRY_ORDER_PARTIAL", client_order_id="E1", execution_id="DUP1", quantity=4, price=12.5),
            ]
        )

        self.assertEqual(snapshot.positions["RKLB"].open_quantity, 4)
        self.assertEqual(snapshot.orders["E1"].filled_quantity, 4)
        self.assertTrue(any(a["kind"] == "duplicate_or_invalid_entry_execution" for a in snapshot.anomalies))

    def test_exit_rejected_moves_position_to_reconciling(self) -> None:
        snapshot = reduce_lifecycle_events(
            [
                event("POSITION_ADOPTED", quantity=10, price=12.5),
                event("EXIT_ORDER_SUBMITTED", client_order_id="X1", quantity=10),
                event("EXIT_ORDER_REJECTED", client_order_id="X1", quantity=10, reason="ibkr_cancelled"),
            ]
        )

        self.assertEqual(snapshot.positions["RKLB"].state, PositionState.RECONCILING)
        self.assertEqual(snapshot.orders["X1"].state, OrderState.REJECTED)

    def test_orphan_or_drift_event_moves_position_to_reconciling(self) -> None:
        snapshot = reduce_lifecycle_events(
            [
                event("POSITION_DRIFT_DETECTED", quantity=10, reason="orphan_ibkr_position"),
            ]
        )

        self.assertEqual(snapshot.positions["RKLB"].state, PositionState.RECONCILING)

    def test_adopted_position_is_open(self) -> None:
        snapshot = reduce_lifecycle_events([event("POSITION_ADOPTED", quantity=3, price=20.0)])

        position = snapshot.positions["RKLB"]
        self.assertEqual(position.state, PositionState.OPEN)
        self.assertEqual(position.open_quantity, 3)

    def test_impossible_transition_records_anomaly_without_crashing(self) -> None:
        snapshot = reduce_lifecycle_events([event("EXIT_ORDER_SUBMITTED", client_order_id="X1", quantity=10)])

        self.assertEqual(snapshot.positions["RKLB"].state, PositionState.RECONCILING)
        self.assertTrue(any(a["kind"] == "exit_submit_without_open_position" for a in snapshot.anomalies))

    def test_reconciliation_cli_json_includes_reducer_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "order_lifecycle.jsonl"
            rows = [
                event("ENTRY_ORDER_SUBMITTED", client_order_id="E1", quantity=1),
                event("ENTRY_ORDER_FILLED", client_order_id="E1", execution_id="F1", quantity=1, price=10.0),
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.live_trading.order_lifecycle.reconciliation",
                    "--lifecycle-jsonl",
                    str(path),
                    "--json",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)

            self.assertIn("reducer_snapshot", payload)
            self.assertEqual(payload["reducer_snapshot"]["positions"]["RKLB"]["state"], "OPEN")


if __name__ == "__main__":
    unittest.main()
