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


def normalize_exec_side(value: Any) -> str:
    side = str(value or "").upper()
    if side in {"BOT", "BUY", "BOUGHT"}:
        return "BUY"
    if side in {"SLD", "SELL", "SOLD"}:
        return "SELL"
    return side


def closed_execution_symbol_set(executions: pd.DataFrame) -> set[str]:
    if executions.empty or "symbol" not in executions.columns:
        return set()
    rows = executions.copy()
    rows["symbol_norm"] = rows["symbol"].astype(str).str.upper()
    rows["side_norm"] = rows.get("side", pd.Series(dtype=str)).map(normalize_exec_side)
    rows["quantity_num"] = pd.to_numeric(rows.get("quantity", pd.Series(dtype=float)), errors="coerce").abs().fillna(0.0)
    rows["signed_qty"] = rows.apply(
        lambda row: row["quantity_num"] if row["side_norm"] == "BUY" else (-row["quantity_num"] if row["side_norm"] == "SELL" else 0.0),
        axis=1,
    )
    grouped = rows.groupby("symbol_norm", dropna=False).agg(
        net_qty=("signed_qty", "sum"),
        sell_qty=("quantity_num", lambda values: float(values[rows.loc[values.index, "side_norm"] == "SELL"].sum())),
    )
    return {
        str(symbol)
        for symbol, row in grouped.iterrows()
        if str(symbol) and abs(float(row["net_qty"])) <= 1e-6 and float(row["sell_qty"]) > 0
    }


def execution_pnl_by_symbol(executions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return broker/SQLite execution PnL variants by closed symbol.

    IBKR commission reports expose realizedPNL on closing executions. The key
    ambiguity we want to diagnose is whether the broker comparison should use
    realizedPNL as-is, realizedPNL minus closing commissions only, or realizedPNL
    minus all entry+exit commissions for the closed symbol.
    """
    if executions.empty or "symbol" not in executions.columns:
        return {}
    rows = executions.copy()
    rows["symbol_norm"] = rows["symbol"].astype(str).str.upper()
    rows["side_norm"] = rows.get("side", pd.Series(dtype=str)).map(normalize_exec_side)
    rows["realized_pnl_num"] = pd.to_numeric(rows.get("realized_pnl", pd.Series(dtype=float)), errors="coerce")
    rows["commission_num"] = pd.to_numeric(rows.get("commission", pd.Series(dtype=float)), errors="coerce").abs().fillna(0.0)
    rows["quantity_num"] = pd.to_numeric(rows.get("quantity", pd.Series(dtype=float)), errors="coerce").abs().fillna(0.0)

    closed_symbols = closed_execution_symbol_set(rows)
    out: dict[str, dict[str, float]] = {}
    for symbol, group in rows[rows["symbol_norm"].isin(closed_symbols)].groupby("symbol_norm"):
        sell_group = group[group["side_norm"] == "SELL"]
        realized_sells = sell_group[sell_group["realized_pnl_num"].notna()]
        gross_realized = float(realized_sells["realized_pnl_num"].fillna(0.0).sum())
        sell_commission = float(realized_sells["commission_num"].fillna(0.0).sum())
        all_commission = float(group["commission_num"].fillna(0.0).sum())
        sell_qty = float(sell_group["quantity_num"].fillna(0.0).sum())
        out[str(symbol)] = {
            "closed_qty": sell_qty,
            "gross_realized": gross_realized,
            "sell_commission": sell_commission,
            "all_commission": all_commission,
            "net_if_realized_only": gross_realized,
            "net_if_realized_minus_sell_commission": gross_realized - sell_commission,
            "net_if_realized_minus_all_commission": gross_realized - all_commission,
        }
    return out


def pnl_formula_totals(by_symbol: dict[str, dict[str, float]]) -> dict[str, float]:
    fields = [
        "closed_qty",
        "gross_realized",
        "sell_commission",
        "all_commission",
        "net_if_realized_only",
        "net_if_realized_minus_sell_commission",
        "net_if_realized_minus_all_commission",
    ]
    return {field: round(sum(float(row.get(field) or 0.0) for row in by_symbol.values()), 6) for field in fields}


def pnl_formula_comparison(
    broker_by_symbol: dict[str, dict[str, float]],
    sqlite_by_symbol: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    broker_totals = pnl_formula_totals(broker_by_symbol)
    sqlite_totals = pnl_formula_totals(sqlite_by_symbol)
    formulas = {
        "realized_only": "net_if_realized_only",
        "realized_minus_sell_commission": "net_if_realized_minus_sell_commission",
        "realized_minus_all_commission": "net_if_realized_minus_all_commission",
    }
    comparison: dict[str, dict[str, float]] = {
        "gross_realized": {
            "broker": broker_totals["gross_realized"],
            "sqlite": sqlite_totals["gross_realized"],
            "diff": round(broker_totals["gross_realized"] - sqlite_totals["gross_realized"], 6),
        },
        "sell_commission": {
            "broker": broker_totals["sell_commission"],
            "sqlite": sqlite_totals["sell_commission"],
            "diff": round(broker_totals["sell_commission"] - sqlite_totals["sell_commission"], 6),
        },
        "all_commission": {
            "broker": broker_totals["all_commission"],
            "sqlite": sqlite_totals["all_commission"],
            "diff": round(broker_totals["all_commission"] - sqlite_totals["all_commission"], 6),
        },
    }
    for label, field in formulas.items():
        comparison[label] = {
            "broker": broker_totals[field],
            "sqlite": sqlite_totals[field],
            "diff": round(broker_totals[field] - sqlite_totals[field], 6),
        }
    return comparison


def pnl_symbol_diffs(
    broker_by_symbol: dict[str, dict[str, float]],
    sqlite_by_symbol: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in sorted(set(broker_by_symbol) | set(sqlite_by_symbol)):
        broker = broker_by_symbol.get(symbol, {})
        sqlite = sqlite_by_symbol.get(symbol, {})
        row = {
            "symbol": symbol,
            "broker_qty": round(float(broker.get("closed_qty") or 0.0), 6),
            "sqlite_qty": round(float(sqlite.get("closed_qty") or 0.0), 6),
            "broker_gross": round(float(broker.get("gross_realized") or 0.0), 6),
            "sqlite_gross": round(float(sqlite.get("gross_realized") or 0.0), 6),
            "broker_sell_commission": round(float(broker.get("sell_commission") or 0.0), 6),
            "sqlite_sell_commission": round(float(sqlite.get("sell_commission") or 0.0), 6),
            "broker_all_commission": round(float(broker.get("all_commission") or 0.0), 6),
            "sqlite_all_commission": round(float(sqlite.get("all_commission") or 0.0), 6),
            "diff_realized_only": round(float(broker.get("net_if_realized_only") or 0.0) - float(sqlite.get("net_if_realized_only") or 0.0), 6),
            "diff_minus_sell_commission": round(
                float(broker.get("net_if_realized_minus_sell_commission") or 0.0)
                - float(sqlite.get("net_if_realized_minus_sell_commission") or 0.0),
                6,
            ),
            "diff_minus_all_commission": round(
                float(broker.get("net_if_realized_minus_all_commission") or 0.0)
                - float(sqlite.get("net_if_realized_minus_all_commission") or 0.0),
                6,
            ),
        }
        row["max_abs_diff"] = max(
            abs(row["diff_realized_only"]),
            abs(row["diff_minus_sell_commission"]),
            abs(row["diff_minus_all_commission"]),
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row["max_abs_diff"], reverse=True)


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
    broker_pnl_by_symbol = execution_pnl_by_symbol(broker_executions)
    sqlite_pnl_by_symbol = execution_pnl_by_symbol(sqlite_executions)
    formula_comparison = pnl_formula_comparison(broker_pnl_by_symbol, sqlite_pnl_by_symbol)
    symbol_diffs = pnl_symbol_diffs(broker_pnl_by_symbol, sqlite_pnl_by_symbol)

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
    sqlite_closed_symbols = int(sqlite_pnl_row.get("closed_symbols") or sqlite_pnl_row.get("trades") or 0)
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
        "sqlite_closed_symbols": sqlite_closed_symbols,
        "broker_closed_net": round(net_sum(broker_closed), 6),
        "sqlite_closed_net": round(float(sqlite_pnl_row.get("sqlite_net") or 0.0), 6),
        "closed_net_diff": round(net_sum(broker_closed) - float(sqlite_pnl_row.get("sqlite_net") or 0.0), 6),
        "sqlite_closed_pnl_source": sqlite_pnl_row.get("reconciliation_sqlite_trade_source", ""),
        "broker_pnl_formulas": pnl_formula_totals(broker_pnl_by_symbol),
        "sqlite_pnl_formulas": pnl_formula_totals(sqlite_pnl_by_symbol),
        "closed_pnl_formula_comparison": formula_comparison,
        "closed_symbol_diffs": symbol_diffs,
        "closed_diff_limit": max(int(args.closed_diff_limit), 0),
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
    print("Broker closed net currently uses: realizedPNL - SELL-side commission from IBKR commission reports")

    print("\nCLOSED PNL FORMULAS:")
    comparison = sample.get("closed_pnl_formula_comparison") or {}
    for label in (
        "gross_realized",
        "sell_commission",
        "all_commission",
        "realized_only",
        "realized_minus_sell_commission",
        "realized_minus_all_commission",
    ):
        row = comparison.get(label, {})
        print(
            f"{label}: broker={row.get('broker', 0.0)} "
            f"sqlite={row.get('sqlite', 0.0)} diff={row.get('diff', 0.0)}"
        )

    diffs = list(sample.get("closed_symbol_diffs") or [])
    if diffs:
        limit = int(sample.get("closed_diff_limit", 25))
        print(f"\nCLOSED SYMBOL DIFFS top={min(len(diffs), limit)}:")
        for row in diffs[:limit]:
            print(
                f"{row['symbol']}: "
                f"broker_qty={row['broker_qty']} sqlite_qty={row['sqlite_qty']} "
                f"broker_gross={row['broker_gross']} sqlite_gross={row['sqlite_gross']} "
                f"broker_sell_comm={row['broker_sell_commission']} sqlite_sell_comm={row['sqlite_sell_commission']} "
                f"broker_all_comm={row['broker_all_commission']} sqlite_all_comm={row['sqlite_all_commission']} "
                f"diff_realized={row['diff_realized_only']} "
                f"diff_minus_sell_comm={row['diff_minus_sell_commission']} "
                f"diff_minus_all_comm={row['diff_minus_all_commission']}"
            )

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
    parser.add_argument("--closed-diff-limit", type=int, default=25)
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
