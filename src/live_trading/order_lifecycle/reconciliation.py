from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.live_trading.order_lifecycle.models import PositionRecord, PositionState


@dataclass(frozen=True)
class ReconciliationReport:
    managed_symbols: list[str] = field(default_factory=list)
    ibkr_symbols: list[str] = field(default_factory=list)
    missing_in_ibkr: list[str] = field(default_factory=list)
    orphan_in_ibkr: list[str] = field(default_factory=list)
    quantity_drift: dict[str, dict[str, float]] = field(default_factory=dict)
    raw_json: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.missing_in_ibkr and not self.orphan_in_ibkr and not self.quantity_drift


def build_reconciliation_report(
    positions: dict[str, PositionRecord],
    ibkr_quantities: dict[str, float],
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

    return ReconciliationReport(
        managed_symbols=sorted(managed_open.keys()),
        ibkr_symbols=sorted(ibkr_open.keys()),
        missing_in_ibkr=missing,
        orphan_in_ibkr=orphan,
        quantity_drift=drift,
    )

