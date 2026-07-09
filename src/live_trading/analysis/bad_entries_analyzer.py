from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_path_stats,
    entry_minutes_after_open,
    entry_time_bucket,
    first_existing_column,
    fnum,
    iso_ts,
    load_recorder_candles,
    load_session_candles,
    min_after_pct,
    nearest_row,
    parse_dt,
    parse_raw_json,
    pct,
    read_sql_table,
    safe_read_csv,
    simulate_tp_sl,
)
from src.live_trading.market_calendar import get_us_equity_session


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")

OUTPUT_COLUMNS = [
    "date",
    "trade_id",
    "analysis_source",
    "logical_trade_group",
    "symbol",
    "entry_time",
    "entry_time_source",
    "entry_time_normalization_warning",
    "matched_first_candle_time",
    "candle_source",
    "candle_coverage_warning",
    "candles_min_time_utc",
    "candles_max_time_utc",
    "entry_session_phase",
    "entry_price",
    "exit_time",
    "exit_price",
    "quantity",
    "net_pnl",
    "net_pnl_pct",
    "exit_reason",
    "price_after_1m_pct",
    "price_after_2m_pct",
    "price_after_5m_pct",
    "price_after_10m_pct",
    "mfe_pct",
    "mae_pct",
    "peak_pct",
    "low_after_entry_pct",
    "immediate_drop",
    "never_green",
    "min_after_1m_pct",
    "min_after_2m_pct",
    "min_after_3m_pct",
    "min_after_5m_pct",
    "min_after_10m_pct",
    "max_after_1m_pct",
    "max_after_2m_pct",
    "max_after_5m_pct",
    "max_after_10m_pct",
    "first_green_seconds",
    "entry_near_local_peak_pct",
    "pullback_before_entry_pct",
    "max_adverse_before_peak_pct",
    "time_to_peak_seconds",
    "time_to_low_seconds",
    "max_drawdown_from_peak_pct",
    "entry_minutes_after_open",
    "entry_time_bucket",
    "signal_age_seconds",
    "signal_age_warning",
    "spread_bps_at_entry",
    "top100_rank",
    "top100_score",
    "live_entry_score",
    "live_entry_rank",
    "tp2_sl1_5_exit_reason",
    "tp2_sl1_5_pnl_pct",
    "tp3_sl2_exit_reason",
    "tp3_sl2_pnl_pct",
    "tp4_sl2_exit_reason",
    "tp4_sl2_pnl_pct",
]


def load_closed_trades(sqlite_path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    trades = read_sql_table(
        sqlite_path,
        "trades",
        where=(
            "UPPER(COALESCE(status, '')) = 'CLOSED' "
            "AND ("
            "(COALESCE(session_date, '') != '' AND session_date BETWEEN ? AND ?) "
            "OR (COALESCE(session_date, '') = '' AND substr(entry_fill_time, 1, 10) BETWEEN ? AND ?)"
            ")"
        ),
        params=[start_date, end_date, start_date, end_date],
        order_by="COALESCE(exit_fill_time, closed_at), symbol, trade_id",
    )
    return trades


def load_executions(sqlite_path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    return read_sql_table(
        sqlite_path,
        "executions",
        where="session_date BETWEEN ? AND ? OR substr(executed_at, 1, 10) BETWEEN ? AND ? OR substr(recorded_at, 1, 10) BETWEEN ? AND ?",
        params=[start_date, end_date, start_date, end_date, start_date, end_date],
        order_by="COALESCE(executed_at, recorded_at), execution_id",
    )


def execution_closed_trades(executions: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if executions.empty:
        return pd.DataFrame()
    rows = executions.copy()
    rows["_side_norm"] = rows.get("side", pd.Series(dtype=str)).fillna("").astype(str).str.upper().map(
        lambda value: "BUY" if value in {"BOT", "BUY", "BOUGHT"} else ("SELL" if value in {"SLD", "SELL", "SOLD"} else value)
    )
    rows["_ts"] = [execution_timestamp(row) for row in rows.to_dict("records")]
    rows = rows.dropna(subset=["_ts"]).sort_values(["symbol", "_ts", "execution_id"], na_position="last")
    out: list[dict[str, Any]] = []
    for symbol, group in rows.groupby(rows["symbol"].fillna("").astype(str).str.upper()):
        if not symbol:
            continue
        open_lots: list[dict[str, Any]] = []
        for row in group.to_dict("records"):
            side = row.get("_side_norm")
            qty = abs(fnum(row.get("quantity"), 0.0) or 0.0)
            price = fnum(row.get("price"), 0.0) or 0.0
            if qty <= 0 or price <= 0:
                continue
            if side == "BUY":
                lot = dict(row)
                lot["remaining_qty"] = qty
                lot["original_qty"] = qty
                open_lots.append(lot)
                continue
            if side != "SELL":
                continue
            sell_ts = row.get("_ts")
            sell_date = sell_ts.strftime("%F") if sell_ts is not None else str(row.get("session_date") or "")
            if sell_date < start_date or sell_date > end_date:
                continue
            remaining = qty
            while remaining > 1e-9 and open_lots:
                lot = open_lots[0]
                lot_remaining = fnum(lot.get("remaining_qty"), 0.0) or 0.0
                if lot_remaining <= 1e-9:
                    open_lots.pop(0)
                    continue
                matched_qty = min(remaining, lot_remaining)
                buy_ts = lot.get("_ts")
                buy_price = fnum(lot.get("price"), 0.0) or 0.0
                gross = (price - buy_price) * matched_qty
                buy_fraction = matched_qty / (fnum(lot.get("original_qty"), matched_qty) or matched_qty or 1.0)
                sell_fraction = matched_qty / qty if qty else 1.0
                buy_commission = abs(fnum(lot.get("commission"), 0.0) or 0.0) * buy_fraction if str(lot.get("commission_source") or "").lower() == "ibkr" else 0.0
                sell_commission = abs(fnum(row.get("commission"), 0.0) or 0.0) * sell_fraction if str(row.get("commission_source") or "").lower() == "ibkr" else 0.0
                sell_realized = fnum(row.get("realized_pnl"))
                net_pnl = (sell_realized * sell_fraction - sell_commission) if sell_realized is not None else gross - buy_commission - sell_commission
                buy_raw = parse_raw_json(lot.get("raw_json"))
                sell_raw = parse_raw_json(row.get("raw_json"))
                raw = {**buy_raw}
                entry_order_id = lot.get("order_id") or lot.get("perm_id") or buy_raw.get("entry_order_id") or buy_raw.get("order_id")
                exit_order_id = row.get("order_id") or row.get("perm_id") or sell_raw.get("exit_order_id") or sell_raw.get("order_id")
                raw.update({
                    "reconstruction_source": "bad_entries_execution_fifo",
                    "buy_execution_id": lot.get("execution_id"),
                    "sell_execution_id": row.get("execution_id"),
                    "entry_order_id": entry_order_id,
                    "exit_order_id": exit_order_id,
                    "sell_raw_json": sell_raw,
                })
                trade_id = f"exec_fifo:{sell_date}:{symbol}:{lot.get('execution_id')}:{row.get('execution_id')}:{matched_qty:g}"
                out.append({
                    "trade_id": trade_id,
                    "analysis_source": "reconstructed_execution_fifo_fill",
                    "status": "CLOSED",
                    "session_date": str(lot.get("session_date") or (buy_ts.strftime("%F") if buy_ts is not None else sell_date)),
                    "symbol": symbol,
                    "entry_order_id": entry_order_id,
                    "exit_order_id": exit_order_id,
                    "entry_fill_time": iso_ts(buy_ts),
                    "exit_fill_time": iso_ts(sell_ts),
                    "closed_at": iso_ts(sell_ts),
                    "entry_price": buy_price,
                    "exit_price": price,
                    "quantity": matched_qty,
                    "gross_pnl": gross,
                    "commission": buy_commission + sell_commission,
                    "net_pnl": net_pnl,
                    "exit_reason": row.get("exit_reason") or sell_raw.get("exit_reason") or "",
                    "top100_rank": lot.get("top100_rank") or buy_raw.get("top100_rank"),
                    "top100_score": lot.get("top100_score") or buy_raw.get("top100_score"),
                    "live_entry_score": lot.get("live_entry_score") or buy_raw.get("live_entry_score"),
                    "live_entry_rank": lot.get("live_entry_rank") or buy_raw.get("live_entry_rank"),
                    "signal_time": lot.get("signal_time") or buy_raw.get("signal_time"),
                    "ready_since": lot.get("ready_since") or buy_raw.get("ready_since"),
                    "raw_json": raw,
                })
                lot["remaining_qty"] = lot_remaining - matched_qty
                remaining -= matched_qty
                if (fnum(lot.get("remaining_qty"), 0.0) or 0.0) <= 1e-9:
                    open_lots.pop(0)
    return pd.DataFrame(out)


def first_non_empty(values: pd.Series) -> Any:
    for value in values:
        if value not in (None, "") and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def first_non_empty_from_records(records: list[dict[str, Any]], names: list[str]) -> Any:
    for record in records:
        raw = parse_raw_json(record.get("raw_json"))
        for name in names:
            value = record.get(name)
            if value not in (None, "") and not (isinstance(value, float) and pd.isna(value)):
                return value
            raw_value = raw.get(name)
            if raw_value not in (None, "") and not (isinstance(raw_value, float) and pd.isna(raw_value)):
                return raw_value
    return None


def logical_group_from_row(row: dict[str, Any]) -> str:
    raw = parse_raw_json(row.get("raw_json"))
    symbol = str(row.get("symbol") or "").upper()
    session_date = str(row.get("session_date") or "")[:10]
    entry_order_id = (
        row.get("entry_order_id")
        or raw.get("entry_order_id")
        or raw.get("order_id")
        or raw.get("buy_order_id")
        or row.get("order_id")
    )
    if entry_order_id not in (None, ""):
        return f"{symbol}|{session_date}|entry_order:{entry_order_id}"
    entry_time = parse_dt(first_existing_column(row, ["entry_fill_time", "entry_time"]) or raw.get("entry_time"))
    entry_price = fnum(first_existing_column(row, ["entry_price", "buy_price"]) or raw.get("entry_price"))
    entry_minute = entry_time.floor("min").isoformat() if entry_time is not None else ""
    price_key = f"{entry_price:.4f}" if entry_price is not None else ""
    return f"{symbol}|{session_date}|entry:{entry_minute}|price:{price_key}"


def logical_match_key(row: dict[str, Any]) -> tuple[str, str, str]:
    raw = parse_raw_json(row.get("raw_json"))
    symbol = str(row.get("symbol") or "").upper()
    entry_order_id = str(row.get("entry_order_id") or raw.get("entry_order_id") or raw.get("order_id") or "")
    entry_time = parse_dt(first_existing_column(row, ["entry_fill_time", "entry_time"]) or raw.get("entry_time"))
    entry_minute = entry_time.floor("min").isoformat() if entry_time is not None else ""
    return symbol, entry_order_id, entry_minute


def row_metadata_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    raw = parse_raw_json(row.get("raw_json"))
    trade_id = str(row.get("trade_id") or "")
    source = str(row.get("analysis_source") or "")
    reconstructed = int(
        trade_id.startswith(("reconstructed:", "exec_fifo:"))
        or "reconstructed" in trade_id.lower()
        or "execution_fifo" in source.lower()
        or "reconstructed" in str(raw.get("reconstruction_source") or "").lower()
    )
    metadata_names = [
        "entry_order_id",
        "top100_rank",
        "top100_score",
        "live_entry_score",
        "live_entry_rank",
        "signal_time",
        "ready_since",
        "exit_reason",
    ]
    metadata_count = 0
    for name in metadata_names:
        value = row.get(name)
        if value in (None, ""):
            value = raw.get(name)
        if value not in (None, "") and not (isinstance(value, float) and pd.isna(value)):
            metadata_count += 1
    # Lower tuple is better: non-reconstructed first, richer metadata first,
    # then stable trade_id ordering for deterministic output.
    return reconstructed, -metadata_count, 0 if source == "sqlite_trades" else 1, trade_id


def is_reconstructed_analysis_row(row: dict[str, Any]) -> bool:
    raw = parse_raw_json(row.get("raw_json"))
    trade_id = str(row.get("trade_id") or "").lower()
    source = str(row.get("analysis_source") or "").lower()
    reconstruction_source = str(raw.get("reconstruction_source") or "").lower()
    return (
        trade_id.startswith(("reconstructed:", "exec_fifo:"))
        or "reconstructed" in trade_id
        or "execution_fifo" in source
        or "reconstructed" in reconstruction_source
        or "execution_fifo" in reconstruction_source
    )


def duplicate_representation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    raw = parse_raw_json(row.get("raw_json"))
    return (
        str(row.get("analysis_source") or ""),
        str(row.get("symbol") or "").upper(),
        str(row.get("entry_order_id") or raw.get("entry_order_id") or raw.get("order_id") or ""),
        iso_ts(first_existing_column(row, ["entry_fill_time", "entry_time"]) or raw.get("entry_time")),
        iso_ts(first_existing_column(row, ["exit_fill_time", "closed_at", "exit_time"]) or raw.get("exit_time")),
        round(fnum(row.get("quantity"), 0.0) or 0.0, 8),
        round(fnum(row.get("entry_price"), 0.0) or 0.0, 8),
        round(fnum(row.get("exit_price"), 0.0) or 0.0, 8),
        round(fnum(row.get("gross_pnl"), 0.0) or 0.0, 8),
        round(fnum(row.get("commission"), 0.0) or 0.0, 8),
        round(fnum(row.get("net_pnl"), 0.0) or 0.0, 8),
        str(raw.get("buy_execution_id") or ""),
        str(raw.get("sell_execution_id") or ""),
    )


def duplicate_trade_group_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[
            "logical_trade_group",
            "rows",
            "symbol",
            "entry_time",
            "exit_times",
            "quantity_sum",
            "net_pnl_sum",
            "sources",
            "trade_ids",
            "entry_order_ids",
            "buy_execution_ids",
            "sell_execution_ids",
        ])
    rows = trades.copy()
    if "logical_trade_group" not in rows.columns:
        rows["logical_trade_group"] = [logical_group_from_row(row) for row in rows.to_dict("records")]
    records_by_group: list[dict[str, Any]] = []
    for group_id, group in rows.groupby("logical_trade_group", dropna=False, sort=False):
        if len(group) <= 1:
            continue
        records = group.to_dict("records")
        raws = [parse_raw_json(row.get("raw_json")) for row in records]
        buy_ids = sorted({str(raw.get("buy_execution_id") or "") for raw in raws if raw.get("buy_execution_id")})
        sell_ids = sorted({str(raw.get("sell_execution_id") or "") for raw in raws if raw.get("sell_execution_id")})
        entry_order_ids = sorted({
            str(row.get("entry_order_id") or raw.get("entry_order_id") or raw.get("order_id") or "")
            for row, raw in zip(records, raws)
            if row.get("entry_order_id") or raw.get("entry_order_id") or raw.get("order_id")
        })
        entry_times = [
            iso_ts(parse_dt(first_existing_column(row, ["entry_fill_time", "entry_time"]) or parse_raw_json(row.get("raw_json")).get("entry_time")))
            for row in records
        ]
        exit_times = [
            iso_ts(parse_dt(first_existing_column(row, ["exit_fill_time", "closed_at", "exit_time"]) or parse_raw_json(row.get("raw_json")).get("exit_time")))
            for row in records
        ]
        records_by_group.append({
            "logical_trade_group": group_id,
            "rows": len(group),
            "symbol": str(first_non_empty(group.get("symbol", pd.Series(dtype=object))) or ""),
            "entry_time": first_non_empty(pd.Series([value for value in entry_times if value])),
            "exit_times": ",".join(sorted({value for value in exit_times if value})),
            "quantity_sum": float(pd.to_numeric(group.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
            "net_pnl_sum": float(pd.to_numeric(group.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
            "sources": ",".join(sorted({str(value) for value in group.get("analysis_source", pd.Series(dtype=object)) if value not in (None, "")})),
            "trade_ids": ",".join(sorted({str(value) for value in group.get("trade_id", pd.Series(dtype=object)) if value not in (None, "")})),
            "entry_order_ids": ",".join(entry_order_ids),
            "buy_execution_ids": ",".join(buy_ids),
            "sell_execution_ids": ",".join(sell_ids),
        })
    return pd.DataFrame(records_by_group)


def collapse_duplicate_trade_group(group_id: str, group: pd.DataFrame) -> dict[str, Any]:
    all_records = group.to_dict("records")
    reconstructed_flags = [is_reconstructed_analysis_row(row) for row in all_records]
    if any(reconstructed_flags) and not all(reconstructed_flags):
        records = [row for row, reconstructed in zip(all_records, reconstructed_flags) if not reconstructed]
        group = pd.DataFrame(records)
        dropped_records = [row for row, reconstructed in zip(all_records, reconstructed_flags) if reconstructed]
    else:
        records = all_records
        dropped_records = []
    unique_records: list[dict[str, Any]] = []
    dropped_exact_records: list[dict[str, Any]] = []
    seen_representation_keys: set[tuple[Any, ...]] = set()
    for row in records:
        key = duplicate_representation_key(row)
        if key in seen_representation_keys:
            dropped_exact_records.append(row)
            continue
        seen_representation_keys.add(key)
        unique_records.append(row)
    records = unique_records
    group = pd.DataFrame(records)
    records = sorted(records, key=row_metadata_score)
    base = dict(records[0])
    qty = pd.to_numeric(group.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    total_qty = float(qty.abs().sum())
    if total_qty <= 0:
        total_qty = float(qty.sum())
    entry_prices = pd.to_numeric(group.get("entry_price", pd.Series(dtype=float)), errors="coerce")
    exit_prices = pd.to_numeric(group.get("exit_price", pd.Series(dtype=float)), errors="coerce")
    for col in ["quantity", "gross_pnl", "commission", "net_pnl"]:
        if col in group.columns:
            base[col] = float(pd.to_numeric(group[col], errors="coerce").fillna(0.0).sum())
    if total_qty > 0 and entry_prices.notna().any():
        base["entry_price"] = float((entry_prices.fillna(0.0) * qty.abs()).sum() / total_qty)
    if total_qty > 0 and exit_prices.notna().any():
        base["exit_price"] = float((exit_prices.fillna(0.0) * qty.abs()).sum() / total_qty)
    entry_times = [
        parse_dt(first_existing_column(row, ["entry_fill_time", "entry_time"]) or parse_raw_json(row.get("raw_json")).get("entry_time"))
        for row in records
    ]
    exit_times = [
        parse_dt(first_existing_column(row, ["exit_fill_time", "closed_at", "exit_time"]) or parse_raw_json(row.get("raw_json")).get("exit_time"))
        for row in records
    ]
    entry_times = [value for value in entry_times if value is not None]
    exit_times = [value for value in exit_times if value is not None]
    if entry_times:
        base["entry_fill_time"] = iso_ts(min(entry_times))
    if exit_times:
        base["exit_fill_time"] = iso_ts(max(exit_times))
        base["closed_at"] = base["exit_fill_time"]
    for col, names in {
        "entry_order_id": ["entry_order_id", "order_id", "buy_order_id"],
        "exit_order_id": ["exit_order_id", "sell_order_id"],
        "top100_rank": ["top100_rank"],
        "top100_score": ["top100_score"],
        "live_entry_score": ["live_entry_score", "entry_score", "score"],
        "live_entry_rank": ["live_entry_rank", "ranking_position"],
        "signal_time": ["signal_time"],
        "ready_since": ["ready_since"],
        "exit_reason": ["exit_reason"],
    }.items():
        value = first_non_empty_from_records(records, names)
        if value not in (None, ""):
            base[col] = value
    raw = parse_raw_json(base.get("raw_json"))
    raw.update({
        "analysis_dedupe_source": "bad_entries_logical_trade_group",
        "dedupe_logical_trade_group": group_id,
        "dedupe_row_count": len(records),
        "dedupe_source_row_count": len(all_records),
        "dedupe_dropped_reconstructed_row_count": len(dropped_records),
        "dedupe_dropped_exact_row_count": len(dropped_exact_records),
        "dedupe_trade_ids": [str(row.get("trade_id") or "") for row in records if row.get("trade_id") not in (None, "")],
        "dedupe_dropped_trade_ids": [str(row.get("trade_id") or "") for row in dropped_records if row.get("trade_id") not in (None, "")],
        "dedupe_dropped_exact_trade_ids": [str(row.get("trade_id") or "") for row in dropped_exact_records if row.get("trade_id") not in (None, "")],
        "dedupe_sources": sorted({str(row.get("analysis_source") or "") for row in records if row.get("analysis_source") not in (None, "")}),
    })
    base["raw_json"] = raw
    base["trade_id"] = str(base.get("trade_id") or f"deduped:{group_id}")
    base["analysis_source"] = str(base.get("analysis_source") or "sqlite_trades")
    if len(records) > 1:
        base["analysis_source"] = "deduped_logical_trade"
    base["logical_trade_group"] = group_id
    return base


def dedupe_logical_trades_for_analysis(trades: pd.DataFrame, *, enabled: bool = True) -> pd.DataFrame:
    if trades.empty:
        trades.attrs["dedupe_diagnostics"] = {
            "source_rows_before_dedupe": 0,
            "source_rows_after_dedupe": 0,
            "dedupe_removed_rows": 0,
            "duplicate_group_count": 0,
            "duplicate_groups": [],
        }
        return trades
    rows = trades.copy()
    rows["logical_trade_group"] = [logical_group_from_row(row) for row in rows.to_dict("records")]
    duplicate_groups = duplicate_trade_group_diagnostics(rows)
    diagnostics = {
        "source_rows_before_dedupe": len(rows),
        "source_rows_after_dedupe": len(rows),
        "dedupe_removed_rows": 0,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups.to_dict("records") if not duplicate_groups.empty else [],
    }
    if not enabled or duplicate_groups.empty:
        rows.attrs["dedupe_diagnostics"] = diagnostics
        return rows
    collapsed: list[dict[str, Any]] = []
    for group_id, group in rows.groupby("logical_trade_group", sort=False, dropna=False):
        if len(group) == 1:
            collapsed.append(group.iloc[0].to_dict())
        else:
            collapsed.append(collapse_duplicate_trade_group(str(group_id), group))
    out = pd.DataFrame(collapsed)
    diagnostics["source_rows_after_dedupe"] = len(out)
    diagnostics["dedupe_removed_rows"] = len(rows) - len(out)
    out.attrs["dedupe_diagnostics"] = diagnostics
    return out


def aggregate_reconstructed_trades(exec_trades: pd.DataFrame) -> pd.DataFrame:
    if exec_trades.empty:
        return exec_trades
    rows = exec_trades.copy()
    rows["_logical_group"] = [logical_group_from_row(row) for row in rows.to_dict("records")]
    grouped_rows: list[dict[str, Any]] = []
    for group_id, group in rows.groupby("_logical_group", sort=False):
        records = group.to_dict("records")
        first = dict(records[0])
        qty = pd.to_numeric(group.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        total_qty = float(qty.sum())
        if total_qty <= 0:
            continue
        entry_prices = pd.to_numeric(group.get("entry_price", pd.Series(dtype=float)), errors="coerce")
        exit_prices = pd.to_numeric(group.get("exit_price", pd.Series(dtype=float)), errors="coerce")
        first["trade_id"] = f"exec_fifo_group:{group_id}"
        first["analysis_source"] = "reconstructed_execution_fifo"
        first["logical_trade_group"] = group_id
        first["quantity"] = total_qty
        first["entry_price"] = float((entry_prices.fillna(0.0) * qty).sum() / total_qty) if entry_prices.notna().any() else None
        first["exit_price"] = float((exit_prices.fillna(0.0) * qty).sum() / total_qty) if exit_prices.notna().any() else None
        for col in ["gross_pnl", "commission", "net_pnl"]:
            first[col] = float(pd.to_numeric(group.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        entry_times = [parse_dt(value) for value in group.get("entry_fill_time", pd.Series(dtype=str))]
        exit_times = [parse_dt(value) for value in group.get("exit_fill_time", pd.Series(dtype=str))]
        entry_times = [value for value in entry_times if value is not None]
        exit_times = [value for value in exit_times if value is not None]
        first["entry_fill_time"] = iso_ts(min(entry_times) if entry_times else None)
        first["exit_fill_time"] = iso_ts(max(exit_times) if exit_times else None)
        first["closed_at"] = first["exit_fill_time"]
        raw = parse_raw_json(first.get("raw_json"))
        raw.update({
            "reconstruction_source": "bad_entries_execution_fifo_grouped",
            "buy_execution_ids": [parse_raw_json(row.get("raw_json")).get("buy_execution_id") for row in records],
            "sell_execution_ids": [parse_raw_json(row.get("raw_json")).get("sell_execution_id") for row in records],
            "entry_order_id": first_non_empty(group.get("entry_order_id", pd.Series(dtype=object))),
            "exit_order_ids": sorted({str(value) for value in group.get("exit_order_id", pd.Series(dtype=object)) if value not in (None, "")}),
            "partial_fill_count": len(records),
        })
        first["raw_json"] = raw
        grouped_rows.append(first)
    return pd.DataFrame(grouped_rows)


def trade_execution_pair_key(row: dict[str, Any]) -> tuple[str, str]:
    raw = parse_raw_json(row.get("raw_json"))
    return str(raw.get("buy_execution_id") or ""), str(raw.get("sell_execution_id") or "")


def augment_trades_from_executions(
    trades: pd.DataFrame,
    executions: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    per_fill: bool = False,
) -> pd.DataFrame:
    exec_trades = execution_closed_trades(executions, start_date, end_date)
    if not per_fill:
        exec_trades = aggregate_reconstructed_trades(exec_trades)
    if exec_trades.empty:
        if not trades.empty:
            trades = trades.copy()
            trades["analysis_source"] = trades.get("analysis_source", "sqlite_trades")
            trades["logical_trade_group"] = [logical_group_from_row(row) for row in trades.to_dict("records")]
            trades.attrs["source_counts"] = {
                "sqlite_trades_count": len(trades),
                "reconstructed_trades_count": 0,
            }
        return trades
    if trades.empty:
        exec_trades.attrs["source_counts"] = {
            "sqlite_trades_count": 0,
            "reconstructed_trades_count": len(exec_trades),
        }
        return exec_trades
    trades = trades.copy()
    trades["analysis_source"] = trades.get("analysis_source", "sqlite_trades")
    trades["logical_trade_group"] = [logical_group_from_row(row) for row in trades.to_dict("records")]
    existing_pairs = {trade_execution_pair_key(row) for row in trades.to_dict("records")}
    existing_trade_ids = {str(value) for value in trades.get("trade_id", pd.Series(dtype=str)).fillna("").astype(str)}
    existing_match_keys = {logical_match_key(row) for row in trades.to_dict("records")}
    missing = []
    for row in exec_trades.to_dict("records"):
        pair = trade_execution_pair_key(row)
        if pair in existing_pairs and pair != ("", ""):
            continue
        if str(row.get("trade_id") or "") in existing_trade_ids:
            continue
        symbol, entry_order_id, entry_minute = logical_match_key(row)
        if entry_order_id and any(key[0] == symbol and key[1] == entry_order_id for key in existing_match_keys):
            continue
        if entry_minute and any(key[0] == symbol and key[2] == entry_minute for key in existing_match_keys):
            continue
        missing.append(row)
    if not missing:
        trades.attrs["source_counts"] = {
            "sqlite_trades_count": len(trades),
            "reconstructed_trades_count": 0,
        }
        return trades
    out = pd.concat([trades, pd.DataFrame(missing)], ignore_index=True, sort=False)
    out.attrs["source_counts"] = {
        "sqlite_trades_count": len(trades),
        "reconstructed_trades_count": len(missing),
    }
    return out


def load_spread_snapshots(recorder_dir: Path, session_date: str) -> pd.DataFrame:
    root = recorder_dir / session_date
    for name in ["spread_snapshots.csv", "market_snapshots.csv"]:
        df = safe_read_csv(root / name)
        if not df.empty:
            return df
    print(f"SPREAD_DATA_MISSING date={session_date} reason=no_spread_or_market_snapshots")
    return pd.DataFrame()


def signal_age(row: dict[str, Any], entry_time: pd.Timestamp | None) -> tuple[float | None, str]:
    if entry_time is None:
        return None, ""
    raw = parse_raw_json(row.get("raw_json"))
    signal_time = first_existing_column(row, ["ready_since"]) or raw.get("ready_since")
    if signal_time in (None, ""):
        signal_time = first_existing_column(row, ["signal_time"]) or raw.get("signal_time")
    signal_ts = parse_dt(signal_time)
    if signal_ts is None:
        return None, ""
    age = (entry_time - signal_ts).total_seconds()
    if age < 0:
        return None, "negative_age"
    return age, ""


def execution_timestamp(row: dict[str, Any]) -> pd.Timestamp | None:
    ts = parse_dt(first_existing_column(row, ["executed_at", "recorded_at"]))
    if ts is not None:
        return ts
    raw = parse_raw_json(row.get("raw_json"))
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    return parse_dt(
        execution.get("time")
        or execution.get("executionTime")
        or raw.get("executed_at")
        or raw.get("execution_time")
        or raw.get("time")
    )


def match_buy_execution(trade: dict[str, Any], executions: pd.DataFrame) -> dict[str, Any]:
    if executions.empty:
        return {}
    rows = executions.copy()
    sides = rows.get("side", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    rows = rows[sides.isin(["BOT", "BUY", "BOUGHT"])]
    if rows.empty:
        return {}
    trade_id = str(trade.get("trade_id") or "")
    symbol = str(trade.get("symbol") or "").upper()
    if trade_id and "trade_id" in rows.columns:
        exact = rows[rows["trade_id"].fillna("").astype(str) == trade_id]
        if not exact.empty:
            exact = exact.copy()
            exact["_ts"] = [execution_timestamp(r) for r in exact.to_dict("records")]
            return exact.sort_values("_ts", na_position="last").iloc[0].to_dict()
    if "symbol" in rows.columns:
        sym = rows[rows["symbol"].fillna("").astype(str).str.upper() == symbol].copy()
        if not sym.empty:
            sym["_ts"] = [execution_timestamp(r) for r in sym.to_dict("records")]
            return sym.sort_values("_ts", na_position="last").iloc[0].to_dict()
    return {}


def resolve_entry_time(
    trade: dict[str, Any],
    raw: dict[str, Any],
    buy_execution: dict[str, Any],
    candles_full: pd.DataFrame,
    session_date: str,
) -> tuple[pd.Timestamp | None, str, str, pd.Timestamp | None]:
    trade_time = parse_dt(first_existing_column(trade, ["entry_fill_time", "entry_time"]) or raw.get("entry_time"))
    exec_time = execution_timestamp(buy_execution) if buy_execution else None
    first_candle = candles_full.iloc[0]["timestamp"] if not candles_full.empty and "timestamp" in candles_full.columns else None
    open_ts = pd.Timestamp(get_us_equity_session(pd.Timestamp(session_date).date()).open_utc)
    warning = ""
    chosen = trade_time
    source = "trades.entry_fill_time"
    if exec_time is not None:
        chosen = exec_time
        source = "executions.executed_at"
    if chosen is not None and first_candle is not None:
        # Reconstructed trades can carry bogus premarket timestamps. If the
        # trade timestamp is before RTH but the matched execution/candles are
        # available in RTH, use the execution timestamp or mark as mismatch.
        if chosen < open_ts and first_candle >= open_ts:
            if exec_time is not None and exec_time >= open_ts:
                chosen = exec_time
                source = "executions.executed_at"
                warning = "trade_entry_before_rth_used_execution"
            else:
                warning = "time_mismatch"
    return chosen, source, warning, first_candle


def candle_minmax(candles: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if candles.empty or "timestamp" not in candles.columns:
        return None, None
    return candles["timestamp"].min(), candles["timestamp"].max()


def candles_cover_trade_window(candles: pd.DataFrame, entry_time: pd.Timestamp | None, exit_time: pd.Timestamp | None) -> bool:
    cmin, cmax = candle_minmax(candles)
    if cmin is None or cmax is None or entry_time is None:
        return False
    needed_end = entry_time + pd.Timedelta(minutes=10)
    if exit_time is not None:
        needed_end = min(needed_end, exit_time)
    return cmin <= entry_time <= cmax and cmax >= needed_end


def choose_trade_candles(
    *,
    history_dir: Path,
    recorder_dir: Path,
    symbol: str,
    session_date: str,
    entry_time: pd.Timestamp | None,
    exit_time: pd.Timestamp | None,
    session_type: str,
) -> tuple[pd.DataFrame, str, str, pd.Timestamp | None, pd.Timestamp | None]:
    recorder = load_recorder_candles(recorder_dir, session_date, symbol)
    if candles_cover_trade_window(recorder, entry_time, exit_time):
        cmin, cmax = candle_minmax(recorder)
        return recorder.sort_values("timestamp").reset_index(drop=True), "recorder", "ok", cmin, cmax
    parquet = load_session_candles(history_dir, symbol, session_date, session_type)
    if candles_cover_trade_window(parquet, entry_time, exit_time):
        cmin, cmax = candle_minmax(parquet)
        warning = "recorder_incomplete" if not recorder.empty else "ok"
        return parquet.sort_values("timestamp").reset_index(drop=True), "parquet", warning, cmin, cmax
    if not recorder.empty:
        cmin, cmax = candle_minmax(recorder)
        return pd.DataFrame(), "missing", "recorder_incomplete", cmin, cmax
    cmin, cmax = candle_minmax(parquet)
    return pd.DataFrame(), "missing", "parquet_missing", cmin, cmax


def entry_phase(entry_time: pd.Timestamp | None, session_date: str, warning: str) -> str:
    if warning == "time_mismatch":
        return "time_mismatch"
    if entry_time is None:
        return ""
    session = get_us_equity_session(pd.Timestamp(session_date).date())
    open_ts = pd.Timestamp(session.open_utc)
    close_ts = pd.Timestamp(session.close_utc)
    if entry_time < open_ts:
        return "premarket"
    if entry_time <= close_ts:
        return "rth"
    return "afterhours"


def window_after_entry(candles: pd.DataFrame, entry_time: pd.Timestamp | None, minutes: int) -> pd.DataFrame:
    if candles.empty or entry_time is None or "timestamp" not in candles.columns:
        return pd.DataFrame()
    return candles[(candles["timestamp"] >= entry_time) & (candles["timestamp"] <= entry_time + pd.Timedelta(minutes=minutes))]


def price_after_pct(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, minutes: int) -> float | None:
    rows = window_after_entry(candles, entry_time, minutes)
    if rows.empty or entry_price <= 0:
        return None
    return pct(fnum(rows.iloc[-1].get("close")), entry_price)


def max_after_pct(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, minutes: int) -> float | None:
    rows = window_after_entry(candles, entry_time, minutes)
    if rows.empty or entry_price <= 0:
        return None
    return pct(float(rows["high"].max()), entry_price)


def first_green_seconds(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None) -> float | None:
    if candles.empty or entry_time is None or entry_price <= 0 or "timestamp" not in candles.columns:
        return None
    rows = candles[candles["timestamp"] >= entry_time].sort_values("timestamp")
    green = rows[(pd.to_numeric(rows["close"], errors="coerce") > entry_price) | (pd.to_numeric(rows["high"], errors="coerce") > entry_price)]
    if green.empty:
        return None
    return (green.iloc[0]["timestamp"] - entry_time).total_seconds()


def local_peak_before_entry(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, minutes: int = 15) -> tuple[float | None, float | None]:
    if candles.empty or entry_time is None or entry_price <= 0 or "timestamp" not in candles.columns:
        return None, None
    prior = candles[(candles["timestamp"] < entry_time) & (candles["timestamp"] >= entry_time - pd.Timedelta(minutes=minutes))]
    if prior.empty:
        return None, None
    local_high = fnum(prior["high"].max())
    if local_high is None or local_high <= 0:
        return None, None
    return pct(entry_price, local_high), pct(entry_price, local_high)


def max_adverse_before_peak(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, peak_time: pd.Timestamp | None) -> float | None:
    if candles.empty or entry_time is None or peak_time is None or entry_price <= 0 or "timestamp" not in candles.columns:
        return None
    before_peak = candles[(candles["timestamp"] >= entry_time) & (candles["timestamp"] <= peak_time)]
    if before_peak.empty:
        return None
    return pct(float(before_peak["low"].min()), entry_price)


def row_value(row: dict[str, Any], raw: dict[str, Any], names: list[str]) -> Any:
    direct = first_existing_column(row, names)
    if direct not in (None, ""):
        return direct
    for name in names:
        value = raw.get(name)
        if value not in (None, ""):
            return value
    return None


def analyze_bad_entries(
    *,
    start_date: str,
    end_date: str,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    session_type: str = "RTH",
    per_fill: bool = False,
) -> pd.DataFrame:
    trades = load_closed_trades(sqlite_path, start_date, end_date)
    executions = load_executions(sqlite_path, start_date, end_date)
    trades = augment_trades_from_executions(trades, executions, start_date, end_date, per_fill=per_fill)
    source_counts = dict(trades.attrs.get("source_counts", {})) if hasattr(trades, "attrs") else {}
    trades = dedupe_logical_trades_for_analysis(trades, enabled=not per_fill)
    dedupe_diagnostics = dict(trades.attrs.get("dedupe_diagnostics", {})) if hasattr(trades, "attrs") else {}
    rows: list[dict[str, Any]] = []
    spread_cache: dict[str, pd.DataFrame] = {}
    diagnostics = {
        "candles_loaded_symbols": set(),
        "candles_min_time_utc": None,
        "candles_max_time_utc": None,
        "trades_min_entry_time_utc": None,
        "trades_max_entry_time_utc": None,
        "time_mismatch_count": 0,
        "metrics_computed_count": 0,
        "metrics_missing_count": 0,
    }
    signal_age_counts = {"ready_since": 0, "signal_time": 0, "missing": 0, "negative": 0}
    for trade in trades.to_dict("records"):
        raw = parse_raw_json(trade.get("raw_json"))
        symbol = str(trade.get("symbol") or "").upper()
        raw_entry_time = parse_dt(first_existing_column(trade, ["entry_fill_time", "entry_time"]) or raw.get("entry_time"))
        exit_time = parse_dt(first_existing_column(trade, ["exit_fill_time", "closed_at", "exit_time"]) or raw.get("exit_time"))
        session_date = str(first_existing_column(trade, ["session_date"]) or (exit_time.strftime("%F") if exit_time is not None else start_date))
        entry_price = fnum(first_existing_column(trade, ["entry_price", "buy_price"]) or raw.get("entry_price"))
        exit_price = fnum(first_existing_column(trade, ["exit_price", "sell_price"]) or raw.get("exit_price"))
        quantity = abs(fnum(trade.get("quantity"), 0.0) or 0.0)
        net_pnl = fnum(first_existing_column(trade, ["net_pnl", "net_actual"]))
        if net_pnl is None:
            gross = fnum(trade.get("gross_pnl"), 0.0) or 0.0
            commission = abs(fnum(trade.get("commission"), 0.0) or 0.0)
            net_pnl = gross - commission
        buy_exec = match_buy_execution(trade, executions)
        preliminary_candles = load_recorder_candles(recorder_dir, session_date, symbol)
        if preliminary_candles.empty:
            preliminary_candles = load_session_candles(history_dir, symbol, session_date, session_type)
        entry_time, entry_time_source, entry_warning, first_candle = resolve_entry_time(trade, raw, buy_exec, preliminary_candles, session_date)
        candles_full, candle_source, candle_warning, candles_min_time, candles_max_time = choose_trade_candles(
            history_dir=history_dir,
            recorder_dir=recorder_dir,
            symbol=symbol,
            session_date=session_date,
            entry_time=entry_time,
            exit_time=exit_time,
            session_type=session_type,
        )
        if not candles_full.empty:
            diagnostics["candles_loaded_symbols"].add(symbol)
            diagnostics["candles_min_time_utc"] = candles_min_time if diagnostics["candles_min_time_utc"] is None else min(diagnostics["candles_min_time_utc"], candles_min_time)
            diagnostics["candles_max_time_utc"] = candles_max_time if diagnostics["candles_max_time_utc"] is None else max(diagnostics["candles_max_time_utc"], candles_max_time)
        if raw_entry_time is not None:
            diagnostics["trades_min_entry_time_utc"] = raw_entry_time if diagnostics["trades_min_entry_time_utc"] is None else min(diagnostics["trades_min_entry_time_utc"], raw_entry_time)
            diagnostics["trades_max_entry_time_utc"] = raw_entry_time if diagnostics["trades_max_entry_time_utc"] is None else max(diagnostics["trades_max_entry_time_utc"], raw_entry_time)
        if entry_warning == "time_mismatch":
            diagnostics["time_mismatch_count"] += 1
        if entry_time is not None and not candles_full.empty and entry_warning != "time_mismatch":
            candles = candles_full[candles_full["timestamp"] >= entry_time]
            if exit_time is not None:
                candles = candles[candles["timestamp"] <= exit_time]
            candles = candles.reset_index(drop=True)
        else:
            candles = pd.DataFrame()
        metrics_ok = not candles.empty and entry_time is not None and entry_price is not None and entry_warning != "time_mismatch"
        diagnostics["metrics_computed_count" if metrics_ok else "metrics_missing_count"] += 1
        stats = calculate_path_stats(candles, entry_price or 0.0, entry_time) if metrics_ok else calculate_path_stats(pd.DataFrame(), 0.0, None)
        first_green = first_green_seconds(candles, entry_price or 0.0, entry_time) if metrics_ok else None
        local_peak_pct, pullback_pct = local_peak_before_entry(candles_full, entry_price or 0.0, entry_time) if metrics_ok else (None, None)
        first_window = window_after_entry(candles, entry_time, 1) if metrics_ok else pd.DataFrame()
        first_close = fnum(first_window.iloc[0].get("close")) if not first_window.empty else None
        fallback_exit_time = exit_time
        fallback_exit_price = exit_price
        sim_2 = simulate_tp_sl(candles, entry_price=entry_price or 0.0, tp_pct=2.0, sl_pct=-1.5, fallback_exit_time=fallback_exit_time, fallback_exit_price=fallback_exit_price)
        sim_3 = simulate_tp_sl(candles, entry_price=entry_price or 0.0, tp_pct=3.0, sl_pct=-2.0, fallback_exit_time=fallback_exit_time, fallback_exit_price=fallback_exit_price)
        sim_4 = simulate_tp_sl(candles, entry_price=entry_price or 0.0, tp_pct=4.0, sl_pct=-2.0, fallback_exit_time=fallback_exit_time, fallback_exit_price=fallback_exit_price)
        if session_date not in spread_cache:
            spread_cache[session_date] = load_spread_snapshots(recorder_dir, session_date)
        spread = nearest_row(spread_cache[session_date], entry_time, symbol)
        sig_age, sig_age_warning = signal_age(trade, entry_time)
        if sig_age_warning == "negative_age":
            signal_age_counts["negative"] += 1
        elif sig_age is not None:
            source_used = "ready_since" if first_existing_column(trade, ["ready_since"]) or raw.get("ready_since") else "signal_time"
            signal_age_counts[source_used] += 1
        else:
            signal_age_counts["missing"] += 1
        net_pnl_pct = None
        if entry_price and entry_price > 0 and quantity > 0 and net_pnl is not None:
            net_pnl_pct = net_pnl / (entry_price * quantity) * 100.0
        row = {
            "date": session_date,
            "trade_id": trade.get("trade_id"),
            "analysis_source": trade.get("analysis_source") or "sqlite_trades",
            "logical_trade_group": trade.get("logical_trade_group") or logical_group_from_row(trade),
            "symbol": symbol,
            "entry_time": iso_ts(entry_time),
            "entry_time_source": entry_time_source,
            "entry_time_normalization_warning": entry_warning,
            "matched_first_candle_time": iso_ts(first_candle),
            "candle_source": candle_source,
            "candle_coverage_warning": candle_warning,
            "candles_min_time_utc": iso_ts(candles_min_time),
            "candles_max_time_utc": iso_ts(candles_max_time),
            "entry_session_phase": entry_phase(entry_time, session_date, entry_warning),
            "entry_price": entry_price,
            "exit_time": iso_ts(exit_time),
            "exit_price": exit_price,
            "quantity": quantity,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl_pct,
            "exit_reason": first_existing_column(trade, ["exit_reason"]) or raw.get("exit_reason") or "unknown_exit_reason",
            "price_after_1m_pct": price_after_pct(candles, entry_price or 0.0, entry_time, 1) if metrics_ok else None,
            "price_after_2m_pct": price_after_pct(candles, entry_price or 0.0, entry_time, 2) if metrics_ok else None,
            "price_after_5m_pct": price_after_pct(candles, entry_price or 0.0, entry_time, 5) if metrics_ok else None,
            "price_after_10m_pct": price_after_pct(candles, entry_price or 0.0, entry_time, 10) if metrics_ok else None,
            "mfe_pct": stats.mfe_pct,
            "mae_pct": stats.mae_pct,
            "peak_pct": stats.mfe_pct,
            "low_after_entry_pct": stats.mae_pct,
            "immediate_drop": int(metrics_ok and ((first_close is not None and entry_price is not None and first_close < entry_price) or ((min_after_pct(candles, entry_price or 0.0, entry_time, 1) or 0.0) < 0.0))),
            "never_green": int(metrics_ok and first_green is None),
            "min_after_1m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 1) if metrics_ok else None,
            "min_after_2m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 2) if metrics_ok else None,
            "min_after_3m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 3) if metrics_ok else None,
            "min_after_5m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 5) if metrics_ok else None,
            "min_after_10m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 10) if metrics_ok else None,
            "max_after_1m_pct": max_after_pct(candles, entry_price or 0.0, entry_time, 1) if metrics_ok else None,
            "max_after_2m_pct": max_after_pct(candles, entry_price or 0.0, entry_time, 2) if metrics_ok else None,
            "max_after_5m_pct": max_after_pct(candles, entry_price or 0.0, entry_time, 5) if metrics_ok else None,
            "max_after_10m_pct": max_after_pct(candles, entry_price or 0.0, entry_time, 10) if metrics_ok else None,
            "first_green_seconds": first_green,
            "entry_near_local_peak_pct": local_peak_pct,
            "pullback_before_entry_pct": pullback_pct,
            "max_adverse_before_peak_pct": max_adverse_before_peak(candles, entry_price or 0.0, entry_time, stats.peak_time),
            "time_to_peak_seconds": stats.time_to_peak_seconds,
            "time_to_low_seconds": stats.time_to_low_seconds,
            "max_drawdown_from_peak_pct": stats.max_drawdown_from_peak_pct,
            "entry_minutes_after_open": entry_minutes_after_open(entry_time, session_date) if entry_phase(entry_time, session_date, entry_warning) == "rth" else None,
            "entry_time_bucket": entry_time_bucket(entry_time, session_date) if entry_phase(entry_time, session_date, entry_warning) == "rth" else entry_phase(entry_time, session_date, entry_warning),
            "signal_age_seconds": sig_age,
            "signal_age_warning": sig_age_warning,
            "spread_bps_at_entry": first_existing_column(spread, ["spread_bps", "bid_ask_spread_bps"]),
            "top100_rank": row_value(trade, raw, ["top100_rank"]),
            "top100_score": row_value(trade, raw, ["top100_score"]),
            "live_entry_score": row_value(trade, raw, ["live_entry_score", "entry_score", "score"]),
            "live_entry_rank": row_value(trade, raw, ["live_entry_rank", "ranking_position"]),
            "tp2_sl1_5_exit_reason": sim_2.exit_reason,
            "tp2_sl1_5_pnl_pct": sim_2.pnl_pct,
            "tp3_sl2_exit_reason": sim_3.exit_reason,
            "tp3_sl2_pnl_pct": sim_3.pnl_pct,
            "tp4_sl2_exit_reason": sim_4.exit_reason,
            "tp4_sl2_pnl_pct": sim_4.pnl_pct,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    out.attrs["diagnostics"] = diagnostics
    out.attrs["signal_age_counts"] = signal_age_counts
    out.attrs["source_counts"] = source_counts
    out.attrs["dedupe_diagnostics"] = dedupe_diagnostics
    return out


def score_bucket(value: Any) -> str:
    score = fnum(value)
    if score is None:
        return "missing"
    if score >= 80:
        return ">=80"
    if score >= 60:
        return "60-80"
    if score >= 40:
        return "40-60"
    if score >= 20:
        return "20-40"
    return "<20"


def rank_bucket(value: Any) -> str:
    rank = fnum(value)
    if rank is None:
        return "missing"
    if rank <= 10:
        return "1-10"
    if rank <= 25:
        return "11-25"
    if rank <= 50:
        return "26-50"
    if rank <= 75:
        return "51-75"
    if rank <= 100:
        return "76-100"
    return "missing"


def signal_age_bucket(value: Any, warning: Any = "") -> str:
    if str(warning or "") == "negative_age":
        return "invalid_negative"
    seconds = fnum(value)
    if seconds is None:
        return "missing"
    if seconds < 0:
        return "invalid_negative"
    if seconds <= 30:
        return "0-30s"
    if seconds <= 60:
        return "30-60s"
    if seconds <= 180:
        return "1-3m"
    if seconds <= 300:
        return "3-5m"
    return "5m+"


def seconds_bucket(value: Any) -> str:
    seconds = fnum(value)
    if seconds is None:
        return "missing"
    if seconds <= 30:
        return "0-30s"
    if seconds <= 60:
        return "30-60s"
    if seconds <= 180:
        return "1-3m"
    if seconds <= 300:
        return "3-5m"
    return "5m+"


def pct_bucket(value: Any) -> str:
    val = fnum(value)
    if val is None:
        return "missing"
    if val <= -5:
        return "<=-5%"
    if val <= -2:
        return "-5..-2%"
    if val < 0:
        return "-2..0%"
    if val < 2:
        return "0..2%"
    if val < 5:
        return "2..5%"
    return ">=5%"


def generic_bucket(value: Any) -> str:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    return str(value)


def print_bucket_summary(df: pd.DataFrame, column: str) -> None:
    if df.empty or column not in df.columns:
        return
    tmp = df.copy()
    if "score" in column:
        tmp["_bucket"] = tmp[column].map(score_bucket)
    elif column == "top100_rank":
        tmp["_bucket"] = tmp[column].map(rank_bucket)
    elif column == "signal_age_seconds":
        warnings = tmp["signal_age_warning"] if "signal_age_warning" in tmp.columns else pd.Series([""] * len(tmp))
        tmp["_bucket"] = [signal_age_bucket(value, warning) for value, warning in zip(tmp[column], warnings)]
    elif column == "first_green_seconds":
        tmp["_bucket"] = tmp[column].map(seconds_bucket)
    elif column in {"pullback_before_entry_pct", "entry_near_local_peak_pct"}:
        tmp["_bucket"] = tmp[column].map(pct_bucket)
    else:
        tmp["_bucket"] = tmp[column].map(generic_bucket)
    grouped = tmp.groupby("_bucket", dropna=False).agg(
        trade_count=("symbol", "count"),
        win_rate=("net_pnl", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean() * 100.0) if len(values) else 0.0),
        avg_net_pnl=("net_pnl", "mean"),
        expectancy=("net_pnl", "mean"),
    )
    print(f"{column}_bucket_summary:")
    print(grouped.reset_index().to_string(index=False))


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    net = pd.to_numeric(df.get("net_pnl", pd.Series(dtype=float)), errors="coerce")
    mfe = pd.to_numeric(df.get("mfe_pct", pd.Series(dtype=float)), errors="coerce")
    mae = pd.to_numeric(df.get("mae_pct", pd.Series(dtype=float)), errors="coerce")
    print(
        "BAD_ENTRIES "
        f"total_trades={total} closed_trades={total} net_pnl={net.sum():.2f} "
        f"avg_mfe={mfe.mean() if not mfe.empty else 0:.2f} median_mfe={mfe.median() if not mfe.empty else 0:.2f} "
        f"avg_mae={mae.mean() if not mae.empty else 0:.2f} median_mae={mae.median() if not mae.empty else 0:.2f}"
    )
    diag = df.attrs.get("diagnostics", {}) if hasattr(df, "attrs") else {}
    signal_counts = df.attrs.get("signal_age_counts", {}) if hasattr(df, "attrs") else {}
    if diag:
        print(
            "BAD_ENTRIES_TIME_DIAGNOSTICS "
            f"candles_loaded_symbols={len(diag.get('candles_loaded_symbols') or [])} "
            f"candles_min_time_utc={diag.get('candles_min_time_utc') or ''} "
            f"candles_max_time_utc={diag.get('candles_max_time_utc') or ''} "
            f"trades_min_entry_time_utc={diag.get('trades_min_entry_time_utc') or ''} "
            f"trades_max_entry_time_utc={diag.get('trades_max_entry_time_utc') or ''} "
            f"time_mismatch_count={diag.get('time_mismatch_count', 0)} "
            f"metrics_computed_count={diag.get('metrics_computed_count', 0)} "
            f"metrics_missing_count={diag.get('metrics_missing_count', 0)}"
        )
    source_counts = df.attrs.get("source_counts", {}) if hasattr(df, "attrs") else {}
    dedupe_diag = df.attrs.get("dedupe_diagnostics", {}) if hasattr(df, "attrs") else {}
    duplicate_groups = pd.DataFrame()
    duplicate_count = 0
    if not df.empty and "logical_trade_group" in df.columns:
        duplicate_groups = (
            df.groupby("logical_trade_group", dropna=False)
            .agg(
                rows=("symbol", "count"),
                symbol=("symbol", "first"),
                entry_time=("entry_time", "first"),
                quantity=("quantity", "sum"),
                sources=("analysis_source", lambda values: ",".join(sorted({str(value) for value in values if value not in (None, "")}))),
            )
            .reset_index()
        )
        duplicate_groups = duplicate_groups[duplicate_groups["rows"] > 1]
        duplicate_count = len(duplicate_groups)
    print(
        "BAD_ENTRIES_SOURCE_SUMMARY "
        f"sqlite_trades_count={source_counts.get('sqlite_trades_count', 0)} "
        f"reconstructed_trades_count={source_counts.get('reconstructed_trades_count', 0)} "
        f"source_rows_before_dedupe={dedupe_diag.get('source_rows_before_dedupe', total)} "
        f"source_rows_after_dedupe={dedupe_diag.get('source_rows_after_dedupe', total)} "
        f"dedupe_removed_rows={dedupe_diag.get('dedupe_removed_rows', 0)} "
        f"duplicate_groups_before_dedupe={dedupe_diag.get('duplicate_group_count', 0)} "
        f"duplicate_symbol_entry_time_count={duplicate_count}"
    )
    removed_groups = pd.DataFrame(dedupe_diag.get("duplicate_groups") or [])
    if not removed_groups.empty:
        print("duplicate_trade_groups_removed:")
        display_cols = [
            "logical_trade_group",
            "rows",
            "symbol",
            "entry_time",
            "exit_times",
            "quantity_sum",
            "net_pnl_sum",
            "sources",
            "trade_ids",
            "entry_order_ids",
            "buy_execution_ids",
            "sell_execution_ids",
        ]
        display_cols = [col for col in display_cols if col in removed_groups.columns]
        print(removed_groups.sort_values(["rows", "symbol"], ascending=[False, True])[display_cols].head(20).to_string(index=False))
    if duplicate_count:
        print("duplicate_trade_groups_after_dedupe:")
        print(duplicate_groups.sort_values(["rows", "symbol"], ascending=[False, True]).head(20).to_string(index=False))
    if signal_counts:
        print(
            "SIGNAL_AGE_SOURCE_COUNTS "
            f"ready_since={signal_counts.get('ready_since', 0)} "
            f"signal_time={signal_counts.get('signal_time', 0)} "
            f"missing={signal_counts.get('missing', 0)} "
            f"negative={signal_counts.get('negative', 0)}"
        )
    if df.empty:
        return
    first_green = pd.to_numeric(df.get("first_green_seconds", pd.Series(dtype=float)), errors="coerce")
    print(
        f"peak_ge_1={(mfe >= 1).sum()} peak_ge_2={(mfe >= 2).sum()} peak_ge_3={(mfe >= 3).sum()} "
        f"peak_eq_0={(mfe == 0).sum()} immediate_drop={int(df['immediate_drop'].sum())} never_green={int(df['never_green'].sum())} "
        f"avg_first_green_seconds={first_green.mean():.2f} median_first_green_seconds={first_green.median():.2f}"
    )
    for col in ["min_after_1m_pct", "min_after_2m_pct", "min_after_5m_pct", "min_after_10m_pct", "max_after_1m_pct", "max_after_2m_pct", "max_after_5m_pct", "max_after_10m_pct"]:
        vals = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
        print(f"{col} avg={vals.mean():.2f} median={vals.median():.2f}")
    for column in ["live_entry_score", "top100_rank", "entry_time_bucket", "spread_bps_at_entry", "signal_age_seconds", "first_green_seconds", "pullback_before_entry_pct", "entry_near_local_peak_pct"]:
        print_bucket_summary(df, column)
    print("worst_immediate_drops_by_min_after_5m_pct:")
    print(df.sort_values("min_after_5m_pct", ascending=True)[["symbol", "entry_time", "min_after_5m_pct", "first_green_seconds", "mfe_pct", "live_entry_score"]].head(20).to_string(index=False))
    print("worst_never_green_trades:")
    print(df[df["never_green"] == 1].sort_values("net_pnl_pct", ascending=True)[["symbol", "entry_time", "net_pnl_pct", "mae_pct", "live_entry_score"]].head(20).to_string(index=False))
    print("best_clean_entries:")
    print(df[df["never_green"] == 0].sort_values(["first_green_seconds", "mfe_pct"], ascending=[True, False])[["symbol", "entry_time", "first_green_seconds", "mfe_pct", "net_pnl_pct", "live_entry_score"]].head(20).to_string(index=False))
    print("worst20_by_net_pnl_pct:")
    print(df.sort_values("net_pnl_pct", ascending=True)[["symbol", "entry_time", "net_pnl_pct", "mfe_pct", "mae_pct", "live_entry_score"]].head(20).to_string(index=False))
    print("best20_by_peak_pct:")
    print(df.sort_values("peak_pct", ascending=False)[["symbol", "entry_time", "peak_pct", "net_pnl_pct", "live_entry_score"]].head(20).to_string(index=False))
    actual = net.sum()
    for prefix in ["tp2_sl1_5", "tp3_sl2", "tp4_sl2"]:
        pnl = pd.to_numeric(df[f"{prefix}_pnl_pct"], errors="coerce")
        print(f"{prefix}_simulation avg_pnl_pct={pnl.mean():.2f} wins={(pnl > 0).sum()} losses={(pnl <= 0).sum()} actual_net_pnl={actual:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze bot entries that dropped after buy and simulate TP/SL variants.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single session/exit date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for date range, YYYY-MM-DD. Required with --start-date.")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--per-fill", action="store_true", help="Keep reconstructed execution FIFO rows per partial fill instead of grouping by logical trade.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = args.date or args.start_date
    end = args.date or args.end_date
    if not end:
        parser.error("--end-date is required with --start-date")
    output = args.output or Path(f"data/analysis/bad_entries_{start if start == end else start + '_to_' + end}.csv")
    df = analyze_bad_entries(
        start_date=start,
        end_date=end,
        sqlite_path=args.sqlite_path,
        history_dir=args.history_dir,
        recorder_dir=args.recorder_dir,
        session_type=args.session_type,
        per_fill=args.per_fill,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print_summary(df)
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
