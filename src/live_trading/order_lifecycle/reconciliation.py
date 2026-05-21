from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.live_trading.order_lifecycle.models import PositionRecord, PositionState
from src.live_trading.order_lifecycle.reducer import reduce_lifecycle_events
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore


@dataclass(frozen=True)
class ReconciliationReport:
    managed_symbols: list[str] = field(default_factory=list)
    ibkr_symbols: list[str] = field(default_factory=list)
    lifecycle_symbols: list[str] = field(default_factory=list)
    pending_order_ids: list[str] = field(default_factory=list)
    missing_in_ibkr: list[str] = field(default_factory=list)
    orphan_in_ibkr: list[str] = field(default_factory=list)
    quantity_drift: dict[str, dict[str, float]] = field(default_factory=dict)
    raw_json: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.missing_in_ibkr and not self.orphan_in_ibkr and not self.quantity_drift and not self.pending_order_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "managed_symbols": self.managed_symbols,
            "ibkr_symbols": self.ibkr_symbols,
            "lifecycle_symbols": self.lifecycle_symbols,
            "pending_order_ids": self.pending_order_ids,
            "missing_in_ibkr": self.missing_in_ibkr,
            "orphan_in_ibkr": self.orphan_in_ibkr,
            "quantity_drift": self.quantity_drift,
            "raw_json": self.raw_json,
        }


def build_reconciliation_report(
    positions: dict[str, PositionRecord],
    ibkr_quantities: dict[str, float],
    *,
    lifecycle_events: list[dict[str, Any]] | None = None,
    open_orders: list[dict[str, Any]] | None = None,
) -> ReconciliationReport:
    managed_open = {
        symbol: position.open_quantity
        for symbol, position in positions.items()
        if position.state in {PositionState.OPEN, PositionState.EXIT_PENDING, PositionState.RECONCILING}
        and position.open_quantity > 0
    }
    ibkr_open = {symbol: qty for symbol, qty in ibkr_quantities.items() if abs(qty) > 0}

    missing = sorted([symbol for symbol in managed_open if symbol not in ibkr_open])
    orphan = sorted([symbol for symbol in ibkr_open if symbol not in managed_open])
    drift: dict[str, dict[str, float]] = {}
    for symbol, qty in managed_open.items():
        ib_qty = ibkr_open.get(symbol)
        if ib_qty is not None and abs(float(ib_qty) - float(qty)) > 1e-9:
            drift[symbol] = {"managed_quantity": float(qty), "ibkr_quantity": float(ib_qty)}

    events = lifecycle_events or []
    lifecycle_symbols = sorted(
        {
            str(row.get("symbol") or "").upper().strip()
            for row in events
            if str(row.get("symbol") or "").strip()
        }
    )
    pending_order_ids = sorted(
        {
            str(row.get("ib_order_id") or row.get("order_id") or row.get("client_order_id") or "").strip()
            for row in (open_orders or [])
            if str(row.get("ib_order_id") or row.get("order_id") or row.get("client_order_id") or "").strip()
        }
    )

    return ReconciliationReport(
        managed_symbols=sorted(managed_open.keys()),
        ibkr_symbols=sorted(ibkr_open.keys()),
        lifecycle_symbols=lifecycle_symbols,
        pending_order_ids=pending_order_ids,
        missing_in_ibkr=missing,
        orphan_in_ibkr=orphan,
        quantity_drift=drift,
        raw_json={"open_orders": open_orders or []},
    )


def load_lifecycle_events(path: str | Path) -> list[dict[str, Any]]:
    return JsonlLifecycleStore(path).load_events()


def log_reconciliation_report(report: ReconciliationReport, *, prefix: str = "") -> None:
    p = f"{prefix} " if prefix else ""
    print(
        f"{p}RECONCILIATION_START managed={len(report.managed_symbols)} ibkr={len(report.ibkr_symbols)} "
        f"lifecycle_symbols={len(report.lifecycle_symbols)} pending_orders={len(report.pending_order_ids)}",
        flush=True,
    )
    for symbol in report.orphan_in_ibkr:
        print(f"{p}RECONCILIATION_ORPHAN_IBKR_POSITION symbol={symbol}", flush=True)
    for symbol in report.missing_in_ibkr:
        print(f"{p}RECONCILIATION_LOCAL_POSITION_MISSING_IN_IBKR symbol={symbol}", flush=True)
    for symbol, drift in report.quantity_drift.items():
        print(
            f"{p}RECONCILIATION_DRIFT symbol={symbol} managed_quantity={drift['managed_quantity']} "
            f"ibkr_quantity={drift['ibkr_quantity']}",
            flush=True,
        )
    for order_id in report.pending_order_ids:
        print(f"{p}RECONCILIATION_PENDING_ORDER_FOUND order_id={order_id}", flush=True)
    if report.clean:
        print(f"{p}RECONCILIATION_CLEAN", flush=True)
    print(f"{p}RECONCILIATION_DONE clean={int(report.clean)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run order lifecycle reconciliation report")
    parser.add_argument("--dry-run", action="store_true", help="print a local lifecycle-only dry-run report")
    parser.add_argument("--lifecycle-jsonl", default="data/live/recorder/order_lifecycle.jsonl")
    parser.add_argument("--json", action="store_true", help="print JSON report")
    args = parser.parse_args()

    events = load_lifecycle_events(args.lifecycle_jsonl)
    snapshot = reduce_lifecycle_events(events)
    report = build_reconciliation_report(snapshot.positions, {}, lifecycle_events=events, open_orders=[])
    if args.json:
        print(
            json.dumps(
                {
                    "reconciliation": report.to_dict(),
                    "reducer_snapshot": snapshot.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    else:
        log_reconciliation_report(report)
        summary = snapshot.to_dict()["summary"]
        print(
            "LIFECYCLE_REDUCER_SUMMARY "
            f"positions={summary['positions']} orders={summary['orders']} anomalies={summary['anomalies']} "
            f"open={summary['open_positions']} exit_pending={summary['exit_pending_positions']} "
            f"reconciling={summary['reconciling_positions']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
