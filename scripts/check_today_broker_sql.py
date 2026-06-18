#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.dashboard.broker_reality import (
    closed_trades_from_commission_reports,
    fetch_ibkr_executions_diagnostic,
    fetch_ibkr_live_portfolio,
    load_sqlite_active_positions,
    load_sqlite_closed_trades,
    load_sqlite_executions,
    load_sqlite_trade_pnl,
)
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%F")


def fnum(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def symbol_qty_map(df: pd.DataFrame, qty_col: str = "quantity") -> dict[str, float]:
    if df.empty or "symbol" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for row in df.to_dict("records"):
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        out[symbol] = out.get(symbol, 0.0) + fnum(row.get(qty_col))
    return {symbol: qty for symbol, qty in out.items() if abs(qty) > 1e-9}


def exec_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "execution_id" not in df.columns:
        return set()
    return {str(value) for value in df["execution_id"].dropna().tolist() if str(value)}


def net_sum(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    if "net_pnl" in df.columns:
        return float(pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0).sum())
    return 0.0


def runtime_status(sqlite_path: str, selected_date: str) -> dict[str, Any]:
    store = SQLiteRuntimeStore(sqlite_path)
    try:
        states = store.get_runtime_state(["fill_ingest", "position_reconcile"])
        pending = store.runtime_pending_counts(selected_date)
    finally:
        store.close()
    flat: dict[str, Any] = dict(pending)
    for name in ("fill_ingest", "position_reconcile"):
        raw = states.get(name, {}).get("raw_json", {})
        flat[f"{name}_status"] = raw.get("status") or states.get(name, {}).get("value") or ""
        flat[f"last_{name}_started_at"] = raw.get("started_at") or ""
        flat[f"last_{name}_finished_at"] = raw.get("finished_at") or ""
    flat["in_progress"] = any(str(flat.get(f"{name}_status") or "").lower() == "running" for name in ("fill_ingest", "position_reconcile"))
    flat["pending_total"] = (
        int(flat.get("pending_commission_count") or 0)
        + int(flat.get("pending_realized_pnl_count") or 0)
        + int(flat.get("pending_trade_finalization_count") or 0)
    )
    return flat


def collect_sample(args: argparse.Namespace) -> dict[str, Any]:
    sqlite_path = resolve_sqlite_path(args.sqlite_path)
    broker_positions, broker_position_status = fetch_ibkr_live_portfolio(
        args.broker_host,
        args.broker_port,
        args.broker_client_id,
        timeout=args.timeout,
    )
    execution_result = fetch_ibkr_executions_diagnostic(
        args.broker_host,
        args.broker_port,
        args.broker_client_id,
        args.date,
        timeout=args.timeout,
    )
    broker_executions = execution_result.filtered_executions
    broker_closed = closed_trades_from_commission_reports(broker_executions, args.date)

    sqlite_positions = load_sqlite_active_positions(sqlite_path, args.date)
    sqlite_executions = load_sqlite_executions(sqlite_path, args.date)
    sqlite_closed = load_sqlite_closed_trades(sqlite_path, args.date)
    sqlite_trade_pnl = load_sqlite_trade_pnl(sqlite_path, args.date)
    sqlite_pnl_row = sqlite_trade_pnl.iloc[0].to_dict() if not sqlite_trade_pnl.empty else {}

    broker_pos = symbol_qty_map(broker_positions)
    sqlite_pos = symbol_qty_map(sqlite_positions)
    missing_open = sorted(set(broker_pos) - set(sqlite_pos))
    extra_open = sorted(set(sqlite_pos) - set(broker_pos))
    qty_diffs = {
        symbol: {"broker": broker_pos.get(symbol, 0.0), "sqlite": sqlite_pos.get(symbol, 0.0)}
        for symbol in sorted(set(broker_pos) & set(sqlite_pos))
        if abs(broker_pos.get(symbol, 0.0) - sqlite_pos.get(symbol, 0.0)) > args.qty_tolerance
    }

    broker_exec_ids = exec_ids(broker_executions)
    sqlite_exec_ids = exec_ids(sqlite_executions)
    missing_exec = sorted(broker_exec_ids - sqlite_exec_ids)
    extra_exec = sorted(sqlite_exec_ids - broker_exec_ids)

    status = runtime_status(sqlite_path, args.date)
    sample = {
        "sample_time": datetime.now(timezone.utc).isoformat(),
        "broker_position_status": broker_position_status,
        "broker_execution_status": execution_result.status,
        "broker_positions": len(broker_positions),
        "sqlite_positions": len(sqlite_positions),
        "broker_qty_sum": round(sum(abs(qty) for qty in broker_pos.values()), 6),
        "sqlite_qty_sum": round(sum(abs(qty) for qty in sqlite_pos.values()), 6),
        "open_ok": not missing_open and not extra_open and not qty_diffs,
        "missing_sqlite_open": missing_open,
        "extra_sqlite_open": extra_open,
        "position_qty_diffs": qty_diffs,
        "broker_executions": len(broker_executions),
        "sqlite_executions": len(sqlite_executions),
        "executions_ok": not missing_exec and not extra_exec,
        "missing_executions_in_sqlite": missing_exec,
        "extra_executions_in_sqlite": extra_exec,
        "broker_closed_symbols": len(broker_closed),
        "sqlite_closed_symbols": len(sqlite_closed),
        "broker_closed_net": round(net_sum(broker_closed), 6),
        "sqlite_closed_net": round(float(sqlite_pnl_row.get("sqlite_net") or 0.0), 6),
        "closed_net_diff": round(net_sum(broker_closed) - float(sqlite_pnl_row.get("sqlite_net") or 0.0), 6),
        "sqlite_closed_pnl_source": sqlite_pnl_row.get("reconciliation_sqlite_trade_source", ""),
        "runtime_status": status,
        "broker_execution_diagnostics": {
            "fills_count": execution_result.diagnostics.get("fills_count"),
            "req_executions_count": execution_result.diagnostics.get("req_executions_count"),
            "filtered_count": execution_result.diagnostics.get("execution_count_after_date_filtering"),
            "unique_execution_client_ids": execution_result.diagnostics.get("unique_execution_client_ids"),
        },
    }
    sample["transient_in_progress"] = bool(
        status.get("in_progress")
        or status.get("pending_total")
        or (abs(sample["broker_executions"] - sample["sqlite_executions"]) <= args.transient_execution_gap and not sample["executions_ok"])
    )
    sample["all_ok"] = bool(sample["open_ok"] and sample["executions_ok"] and abs(sample["closed_net_diff"]) <= args.pnl_tolerance)
    return sample


def print_sample(sample: dict[str, Any], *, verbose: bool = False) -> None:
    print("\nOPEN:")
    print(f"Broker positions={sample['broker_positions']} qty_sum={sample['broker_qty_sum']}")
    print(f"SQLite positions={sample['sqlite_positions']} qty_sum={sample['sqlite_qty_sum']}")
    print(f"OPEN_OK={sample['open_ok']}")
    if sample["missing_sqlite_open"]:
        print(f"Missing SQLite open: {', '.join(sample['missing_sqlite_open'])}")
    if sample["extra_sqlite_open"]:
        print(f"Extra SQLite open: {', '.join(sample['extra_sqlite_open'])}")
    if sample["position_qty_diffs"]:
        print(f"Qty diffs: {json.dumps(sample['position_qty_diffs'], sort_keys=True)}")

    print("\nEXECUTIONS:")
    print(f"Broker executions={sample['broker_executions']}")
    print(f"SQLite executions={sample['sqlite_executions']}")
    print(f"EXECUTIONS_OK={sample['executions_ok']}")
    print(f"Missing executions in SQLite: {sample['missing_executions_in_sqlite']}")
    print(f"Extra executions in SQLite: {sample['extra_executions_in_sqlite']}")

    print("\nCLOSED:")
    print(f"Broker closed symbols={sample['broker_closed_symbols']} net={sample['broker_closed_net']}")
    print(f"SQLite closed symbols={sample['sqlite_closed_symbols']} net={sample['sqlite_closed_net']}")
    if sample.get("sqlite_closed_pnl_source"):
        print(f"SQLite closed pnl source={sample['sqlite_closed_pnl_source']}")
    print(f"Closed net diff={sample['closed_net_diff']}")

    print("\nSQLITE RUNTIME STATUS:")
    print(json.dumps(sample["runtime_status"], indent=2, sort_keys=True))
    if sample["transient_in_progress"]:
        print("TRANSIENT_IN_PROGRESS=True")
    if verbose:
        print("\nBROKER EXECUTION DIAGNOSTICS:")
        print(json.dumps(sample["broker_execution_diagnostics"], indent=2, sort_keys=True, default=str))


def stable_signature(sample: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sample["broker_positions"],
        sample["sqlite_positions"],
        sample["broker_executions"],
        sample["sqlite_executions"],
        sample["broker_closed_symbols"],
        sample["sqlite_closed_symbols"],
        sample["broker_closed_net"],
        sample["sqlite_closed_net"],
        sample["closed_net_diff"],
        sample["open_ok"],
        sample["executions_ok"],
        sample["runtime_status"].get("pending_total"),
        sample["runtime_status"].get("in_progress"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare today's IBKR broker truth with SQLite runtime state.")
    parser.add_argument("--date", default=utc_today())
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--broker-host", default="127.0.0.1")
    parser.add_argument("--broker-port", type=int, default=4002)
    parser.add_argument("--broker-client-id", type=int, default=177)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--retry-seconds", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--transient-execution-gap", type=int, default=3)
    parser.add_argument("--qty-tolerance", type=float, default=1e-6)
    parser.add_argument("--pnl-tolerance", type=float, default=0.01)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    deadline = time.time() + (0.0 if args.no_retry else max(args.retry_seconds, 0.0))
    previous_signature: tuple[Any, ...] | None = None
    previous_sample: dict[str, Any] | None = None
    sample_number = 0
    while True:
        sample_number += 1
        sample = collect_sample(args)
        print(f"\n=== SAMPLE {sample_number} {sample['sample_time']} date={args.date} ===")
        print_sample(sample, verbose=args.verbose)
        if sample["all_ok"]:
            print("\nCHECK_RESULT=OK")
            return 0

        signature = stable_signature(sample)
        if previous_signature == signature and previous_sample is not None and not sample["transient_in_progress"]:
            print("\nCHECK_RESULT=TRUE_MISMATCH_STABLE")
            return 1
        previous_signature = signature
        previous_sample = sample

        if args.no_retry or time.time() >= deadline:
            result = "TRANSIENT_IN_PROGRESS" if sample["transient_in_progress"] else "MISMATCH"
            print(f"\nCHECK_RESULT={result}")
            return 2 if sample["transient_in_progress"] else 1
        print(f"\nRetrying in {args.sleep:.1f}s because SQLite may still be ingesting/reconciling...")
        time.sleep(max(args.sleep, 0.1))


if __name__ == "__main__":
    raise SystemExit(main())
