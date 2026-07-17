from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


QTY_TOLERANCE = 1e-9
MAX_SORT_DATETIME = datetime.max.replace(tzinfo=timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_format(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "missing"
    if text.endswith("Z"):
        return "z_suffix"
    if "T" in text:
        return "t_separator"
    if " " in text:
        return "space_separator"
    return "other"


def _timestamp_is_naive(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    parsed = _parse_dt(text)
    if parsed is None:
        return False
    return not (text.endswith("Z") or "+" in text[10:] or "-" in text[10:])


def _iso(value: Any) -> str:
    parsed = _parse_dt(value)
    return parsed.isoformat() if parsed else str(value or "")


def _date_part(value: Any) -> str:
    parsed = _parse_dt(value)
    if parsed:
        return parsed.date().isoformat()
    return str(value or "")[:10]


def _side(value: Any) -> str:
    text = str(value or "").upper().strip()
    if text in {"BOT", "BUY", "BOUGHT", "B"}:
        return "BUY"
    if text in {"SLD", "SELL", "SOLD", "S"}:
        return "SELL"
    return text


def _primary_sort_dt(row: dict[str, Any]) -> datetime:
    return _parse_dt(row.get("executed_at")) or _parse_dt(row.get("recorded_at")) or MAX_SORT_DATETIME


def _recorded_sort_dt(row: dict[str, Any]) -> datetime:
    return _parse_dt(row.get("recorded_at")) or MAX_SORT_DATETIME


def _sort_key(row: dict[str, Any]) -> tuple[datetime, datetime, str]:
    return (
        _primary_sort_dt(row),
        _recorded_sort_dt(row),
        str(row.get("execution_id") or ""),
    )


def sort_execution_rows(executions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in executions), key=_sort_key)


def timestamp_diagnostics(executions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in executions]
    format_values: set[str] = set()
    parse_failures = 0
    naive_count = 0
    for row in rows:
        for field in ("executed_at", "recorded_at"):
            value = row.get(field)
            if value in (None, ""):
                continue
            format_values.add(_timestamp_format(value))
            if _parse_dt(value) is None:
                parse_failures += 1
            if _timestamp_is_naive(value):
                naive_count += 1
    raw_sorted_ids = [
        str(row.get("execution_id") or "")
        for row in sorted(
            rows,
            key=lambda item: (
                str(item.get("executed_at") or item.get("recorded_at") or ""),
                str(item.get("recorded_at") or ""),
                str(item.get("execution_id") or ""),
            ),
        )
    ]
    parsed_sorted_ids = [str(row.get("execution_id") or "") for row in sort_execution_rows(rows)]
    raw_diff_count = sum(1 for left, right in zip(raw_sorted_ids, parsed_sorted_ids) if left != right)
    return {
        "mixed_timestamp_formats": len(format_values) > 1,
        "timestamp_format_count": len(format_values),
        "timestamp_formats": sorted(format_values),
        "timestamp_parse_failures": parse_failures,
        "naive_timestamp_assumed_utc_count": naive_count,
        "raw_string_order_differs_from_parsed": raw_sorted_ids != parsed_sorted_ids,
        "raw_string_order_diff_count": raw_diff_count,
    }


def _raw(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("raw_json")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _commission(row: dict[str, Any], fraction: float) -> float:
    value = _safe_float(row.get("commission"))
    return (value or 0.0) * fraction


def _realized_pnl(row: dict[str, Any], fraction: float) -> float | None:
    value = _safe_float(row.get("realized_pnl"))
    if value is None:
        return None
    return value * fraction


@dataclass
class CanonicalLot:
    row: dict[str, Any]
    trade_id: str
    original_qty: float
    remaining_qty: float


@dataclass
class CanonicalComponent:
    component_id: str
    trade_id: str
    symbol: str
    session_date: str
    buy_execution_id: str
    sell_execution_id: str
    matched_qty: float
    buy_price: float
    sell_price: float
    entry_time: str
    exit_time: str
    buy_commission_alloc: float
    sell_commission_alloc: float
    realized_pnl_alloc: float | None
    gross_pnl: float
    net_pnl: float
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalTrade:
    trade_id: str
    symbol: str
    strategy_name: str
    session_date: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    commission: float
    net_pnl: float
    components: list[CanonicalComponent] = field(default_factory=list)
    buy_execution_ids: list[str] = field(default_factory=list)
    sell_execution_ids: list[str] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRebuild:
    symbol: str
    trades: list[CanonicalTrade]
    components: list[CanonicalComponent]
    open_lots: list[CanonicalLot]
    unmatched_sells: list[dict[str, Any]]
    buy_consumed: dict[str, float]
    sell_matched: dict[str, float]
    sell_unmatched: dict[str, float]
    timestamp_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def open_quantity(self) -> float:
        return sum(lot.remaining_qty for lot in self.open_lots)

    def conservation_summary(self) -> dict[str, Any]:
        return {
            "buy_execution_count": len(self.buy_consumed),
            "sell_execution_count": len(self.sell_matched),
            "unmatched_sell_count": sum(1 for qty in self.sell_unmatched.values() if qty > QTY_TOLERANCE),
            "unmatched_sell_quantity": sum(self.sell_unmatched.values()),
            "open_quantity": self.open_quantity,
            "component_count": len(self.components),
            "trade_count": len(self.trades),
        }


def _metadata_trade_id(symbol: str, cycle_index: int, row: dict[str, Any]) -> str:
    symbol = str(symbol or "").upper()
    entry_time = _iso(row.get("executed_at") or row.get("recorded_at"))
    entry_date = str(row.get("session_date") or "") or _date_part(entry_time)
    exec_id = str(row.get("execution_id") or f"cycle{cycle_index}")
    return f"canonical:{entry_date}:{symbol}:{cycle_index:04d}:{exec_id}"


def build_canonical_fifo(executions: list[dict[str, Any]], *, symbol: str | None = None) -> CanonicalRebuild:
    rows = sort_execution_rows(executions)
    ts_diagnostics = timestamp_diagnostics(executions)
    inferred_symbol = str(symbol or (rows[0].get("symbol") if rows else "") or "").upper().strip()
    open_lots: list[CanonicalLot] = []
    active_trade_id: str | None = None
    cycle_index = 0
    position_qty = 0.0
    components_by_trade: dict[str, list[CanonicalComponent]] = {}
    open_trade_first_row: dict[str, dict[str, Any]] = {}
    buy_consumed: dict[str, float] = {}
    sell_matched: dict[str, float] = {}
    sell_unmatched: dict[str, float] = {}
    unmatched_sells: list[dict[str, Any]] = []
    component_seq = 0

    for row in rows:
        row_symbol = str(row.get("symbol") or inferred_symbol).upper().strip()
        if symbol and row_symbol != inferred_symbol:
            continue
        side = _side(row.get("side"))
        qty = abs(_safe_float(row.get("quantity")) or 0.0)
        price = _safe_float(row.get("price")) or 0.0
        if qty <= QTY_TOLERANCE or price <= 0:
            continue
        exec_id = str(row.get("execution_id") or "")
        if side == "BUY":
            if position_qty <= QTY_TOLERANCE:
                cycle_index += 1
                active_trade_id = _metadata_trade_id(row_symbol, cycle_index, row)
                open_trade_first_row[active_trade_id] = row
            assert active_trade_id is not None
            open_lots.append(CanonicalLot(row=row, trade_id=active_trade_id, original_qty=qty, remaining_qty=qty))
            buy_consumed.setdefault(exec_id, 0.0)
            position_qty += qty
            continue
        if side != "SELL":
            continue

        remaining = qty
        sell_matched.setdefault(exec_id, 0.0)
        while remaining > QTY_TOLERANCE and open_lots:
            lot = open_lots[0]
            if lot.remaining_qty <= QTY_TOLERANCE:
                open_lots.pop(0)
                continue
            matched_qty = min(remaining, lot.remaining_qty)
            buy = lot.row
            buy_exec_id = str(buy.get("execution_id") or "")
            buy_fraction = matched_qty / (lot.original_qty or matched_qty)
            sell_fraction = matched_qty / qty
            buy_price = _safe_float(buy.get("price")) or 0.0
            sell_price = price
            buy_commission = _commission(buy, buy_fraction)
            sell_commission = _commission(row, sell_fraction)
            realized = _realized_pnl(row, sell_fraction)
            price_gross = (sell_price - buy_price) * matched_qty
            gross = realized if realized is not None else price_gross
            # The persisted trades table historically stores all execution
            # commissions in commission/net_pnl. Broker-compatible
            # realized-minus-sell-commission remains available in raw_json.
            net = gross - buy_commission - sell_commission
            component_seq += 1
            component_id = f"component:{lot.trade_id}:{buy_exec_id}:{exec_id}:{component_seq:06d}"
            entry_time = _iso(buy.get("executed_at") or buy.get("recorded_at"))
            exit_time = _iso(row.get("executed_at") or row.get("recorded_at"))
            component = CanonicalComponent(
                component_id=component_id,
                trade_id=lot.trade_id,
                symbol=row_symbol,
                session_date=str(row.get("session_date") or "") or _date_part(exit_time),
                buy_execution_id=buy_exec_id,
                sell_execution_id=exec_id,
                matched_qty=matched_qty,
                buy_price=buy_price,
                sell_price=sell_price,
                entry_time=entry_time,
                exit_time=exit_time,
                buy_commission_alloc=buy_commission,
                sell_commission_alloc=sell_commission,
                realized_pnl_alloc=realized,
                gross_pnl=gross,
                net_pnl=net,
                raw_json={
                    "reconstruction_source": "canonical_fifo_execution_reducer",
                    "price_gross_pnl": price_gross,
                    "broker_compatible_net_pnl": gross - sell_commission,
                    "net_pnl_formula": "allocated_realized_pnl_or_price_gross_minus_all_commission",
                    "all_commission_alloc": buy_commission + sell_commission,
                },
            )
            components_by_trade.setdefault(lot.trade_id, []).append(component)
            buy_consumed[buy_exec_id] = buy_consumed.get(buy_exec_id, 0.0) + matched_qty
            sell_matched[exec_id] = sell_matched.get(exec_id, 0.0) + matched_qty
            lot.remaining_qty -= matched_qty
            remaining -= matched_qty
            position_qty = max(0.0, position_qty - matched_qty)
            if lot.remaining_qty <= QTY_TOLERANCE:
                open_lots.pop(0)
            if position_qty <= QTY_TOLERANCE:
                active_trade_id = None
                position_qty = 0.0
        if remaining > QTY_TOLERANCE:
            sell_unmatched[exec_id] = remaining
            unmatched = dict(row)
            unmatched["unmatched_sell_quantity"] = remaining
            unmatched_sells.append(unmatched)
        else:
            sell_unmatched.setdefault(exec_id, 0.0)

    trades: list[CanonicalTrade] = []
    components: list[CanonicalComponent] = []
    for trade_id, trade_components in components_by_trade.items():
        total_qty = sum(component.matched_qty for component in trade_components)
        if total_qty <= QTY_TOLERANCE:
            continue
        entry_price = sum(component.buy_price * component.matched_qty for component in trade_components) / total_qty
        exit_price = sum(component.sell_price * component.matched_qty for component in trade_components) / total_qty
        gross = sum(component.gross_pnl for component in trade_components)
        all_commission = sum(component.buy_commission_alloc + component.sell_commission_alloc for component in trade_components)
        net = sum(component.net_pnl for component in trade_components)
        entry_times = [component.entry_time for component in trade_components if component.entry_time]
        exit_times = [component.exit_time for component in trade_components if component.exit_time]
        first_row = open_trade_first_row.get(trade_id) or {}
        trade = CanonicalTrade(
            trade_id=trade_id,
            symbol=str(first_row.get("symbol") or inferred_symbol).upper(),
            strategy_name=str(first_row.get("strategy_name") or "unknown"),
            session_date=str(first_row.get("session_date") or "") or _date_part(min(entry_times) if entry_times else ""),
            entry_time=min(entry_times) if entry_times else "",
            exit_time=max(exit_times) if exit_times else "",
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=total_qty,
            gross_pnl=gross,
            commission=all_commission,
            net_pnl=net,
            components=trade_components,
            buy_execution_ids=sorted({component.buy_execution_id for component in trade_components if component.buy_execution_id}),
            sell_execution_ids=sorted({component.sell_execution_id for component in trade_components if component.sell_execution_id}),
            raw_json={
                "reconstruction_source": "canonical_fifo_execution_reducer",
                "component_count": len(trade_components),
                "buy_execution_ids": sorted({component.buy_execution_id for component in trade_components if component.buy_execution_id}),
                "sell_execution_ids": sorted({component.sell_execution_id for component in trade_components if component.sell_execution_id}),
                "all_commission": all_commission,
                "broker_compatible_net_pnl": sum(
                    component.gross_pnl - component.sell_commission_alloc for component in trade_components
                ),
                "net_pnl_formula": "sum(component allocated_realized_pnl_or_price_gross_minus_all_commission)",
                "peak_rebuild_status": "needs_rebuild",
            },
        )
        trades.append(trade)
        components.extend(trade_components)

    trades.sort(key=lambda trade: (trade.entry_time, trade.exit_time, trade.trade_id))
    components.sort(key=lambda component: (component.entry_time, component.exit_time, component.component_id))
    return CanonicalRebuild(
        symbol=inferred_symbol,
        trades=trades,
        components=components,
        open_lots=open_lots,
        unmatched_sells=unmatched_sells,
        buy_consumed=buy_consumed,
        sell_matched=sell_matched,
        sell_unmatched=sell_unmatched,
        timestamp_diagnostics=ts_diagnostics,
    )
