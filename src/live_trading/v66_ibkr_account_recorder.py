from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ib_insync import IB, ExecutionFilter

from src.live_trading.v62_live_data_recorder import FillEvent, LiveDataRecorder, PortfolioSnapshot
from src.live_trading.storage.sqlite_store import open_sqlite_store, safe_sqlite_call
from src.live_trading.unified_logger import install_unified_logger


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 66
DEFAULT_RECORDER_DIR = "data/live/recorder"

FILL_FIELDS = list(FillEvent.__dataclass_fields__.keys())


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def raw_object(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def rate_limited_recorder_log(
    recorder: LiveDataRecorder,
    bucket: str,
    message: str,
    *,
    key: str,
    max_unique: int = 20,
    window_seconds: float = 60.0,
) -> None:
    state = getattr(recorder, "_rate_limited_log_state", None)
    if not isinstance(state, dict):
        state = {}
        setattr(recorder, "_rate_limited_log_state", state)
    now = time.monotonic()
    item = state.setdefault(bucket, {"window_start": now, "keys": set(), "suppressed": 0})
    if now - float(item.get("window_start", now)) >= window_seconds:
        suppressed = int(item.get("suppressed", 0) or 0)
        if suppressed:
            print(f"{now_utc()} {bucket}_SUPPRESSED count={suppressed}", flush=True)
        item["window_start"] = now
        item["keys"] = set()
        item["suppressed"] = 0
    keys = item.setdefault("keys", set())
    if not isinstance(keys, set):
        keys = set()
        item["keys"] = keys
    if key in keys:
        return
    if len(keys) >= max_unique:
        item["suppressed"] = int(item.get("suppressed", 0) or 0) + 1
        return
    keys.add(key)
    print(message, flush=True)


def account_values_map(ib: IB) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for v in ib.accountValues():
        account = v.account or ""
        out.setdefault(account, {})[v.tag] = v.value
    return out


def portfolio_positions(ib: IB) -> tuple[int, float, list[dict[str, Any]], float, float]:
    positions = []
    gross_exposure = 0.0
    unrealized = 0.0
    realized = 0.0
    for item in ib.portfolio():
        market_value = safe_float(item.marketValue) or 0.0
        gross_exposure += abs(market_value)
        unrealized += safe_float(item.unrealizedPNL) or 0.0
        realized += safe_float(item.realizedPNL) or 0.0
        positions.append({
            "account": item.account,
            "symbol": getattr(item.contract, "symbol", ""),
            "conId": getattr(item.contract, "conId", None),
            "secType": getattr(item.contract, "secType", ""),
            "currency": getattr(item.contract, "currency", ""),
            "position": item.position,
            "marketPrice": item.marketPrice,
            "marketValue": item.marketValue,
            "averageCost": item.averageCost,
            "unrealizedPNL": item.unrealizedPNL,
            "realizedPNL": item.realizedPNL,
        })
    return len(positions), gross_exposure, positions, unrealized, realized


def record_account_snapshot(ib: IB, recorder: LiveDataRecorder) -> None:
    values_by_account = account_values_map(ib)
    open_positions, gross_exposure, positions, unrealized_pnl, realized_pnl = portfolio_positions(ib)

    if not values_by_account:
        recorder.record_portfolio(PortfolioSnapshot(
            account="",
            gross_exposure=gross_exposure,
            open_positions=open_positions,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            positions_json=json.dumps(positions, default=str),
        ))
        return

    for account, values in values_by_account.items():
        cash = safe_float(values.get("TotalCashValue") or values.get("CashBalance"))
        net_liq = safe_float(values.get("NetLiquidation"))
        buying_power = safe_float(values.get("BuyingPower") or values.get("AvailableFunds"))
        daily_pnl = safe_float(values.get("DailyPnL"))
        recorder.record_portfolio(PortfolioSnapshot(
            account=account,
            cash=cash,
            net_liquidation=net_liq,
            buying_power=buying_power,
            gross_exposure=gross_exposure,
            open_positions=open_positions,
            daily_pnl=daily_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            positions_json=json.dumps(positions, ensure_ascii=False, default=str),
        ))


def fill_key(fill) -> str:
    execution = fill.execution
    return str(getattr(execution, "execId", "")) or f"{getattr(execution, 'orderId', '')}-{getattr(execution, 'time', '')}"


def commission_source_for_report(commission_report: Any) -> str:
    commission = safe_float(getattr(commission_report, "commission", None))
    if commission is None or commission == 0:
        return "missing"
    return "ibkr"


def fill_row_from_ibkr_fill(fill: Any) -> dict[str, Any]:
    contract = fill.contract
    execution = fill.execution
    commission_report = getattr(fill, "commissionReport", None)
    recorded_at = now_utc()
    raw = {
        "contract": raw_object(contract),
        "execution": raw_object(execution),
        "commissionReport": raw_object(commission_report),
        "execution_insert_time": recorded_at,
    }
    commission = safe_float(getattr(commission_report, "commission", None))
    return {
        "execution_id": str(getattr(execution, "execId", "") or ""),
        "symbol": str(getattr(contract, "symbol", "") or "").upper(),
        "action": str(getattr(execution, "side", "") or ""),
        "quantity": safe_float(getattr(execution, "shares", None)),
        "fill_price": safe_float(getattr(execution, "price", None)),
        "order_id": str(getattr(execution, "orderId", "") or ""),
        "perm_id": str(getattr(execution, "permId", "") or ""),
        "exchange": str(getattr(execution, "exchange", "") or ""),
        "liquidity": str(getattr(execution, "lastLiquidity", "") or getattr(execution, "liquidity", "") or ""),
        "commission": commission if commission_source_for_report(commission_report) == "ibkr" else "",
        "commission_currency": str(getattr(commission_report, "currency", "") or ""),
        "realized_pnl": safe_float(getattr(commission_report, "realizedPNL", None)),
        "commission_source": commission_source_for_report(commission_report),
        "client_order_id": "",
        "slippage_bps": "",
        "raw_json": json.dumps(raw, default=str, ensure_ascii=False),
        "executed_at": str(getattr(execution, "time", "") or ""),
        "recorded_at": recorded_at,
    }


def commission_report_row(commission_report: Any) -> dict[str, Any]:
    commission = safe_float(getattr(commission_report, "commission", None))
    source = "ibkr" if commission is not None and commission != 0 else "missing"
    recorded_at = now_utc()
    return {
        "execution_id": str(getattr(commission_report, "execId", "") or ""),
        "commission": commission if source == "ibkr" else "",
        "commission_currency": str(getattr(commission_report, "currency", "") or ""),
        "realized_pnl": safe_float(getattr(commission_report, "realizedPNL", None)),
        "commission_source": source,
        "raw_json": json.dumps(
            {
                "commissionReport": raw_object(commission_report),
                "commission_report_time": recorded_at,
                "realized_pnl_ready_time": recorded_at if safe_float(getattr(commission_report, "realizedPNL", None)) is not None else "",
            },
            default=str,
            ensure_ascii=False,
        ),
        "recorded_at": recorded_at,
    }


def merge_raw_json(existing: Any, incoming: Any) -> str:
    def parse(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if not value:
            return {}
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    merged = parse(existing)
    incoming_parsed = parse(incoming)
    for key, value in incoming_parsed.items():
        if value not in (None, ""):
            merged[key] = value
    return json.dumps(merged, default=str, ensure_ascii=False)


def merge_fill_rows(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key in FILL_FIELDS:
        if key == "recorded_at" and existing.get("recorded_at"):
            continue
        if key == "raw_json":
            merged["raw_json"] = merge_raw_json(existing.get("raw_json"), incoming.get("raw_json"))
            continue
        value = incoming.get(key)
        if value not in (None, ""):
            merged[key] = value
    if existing.get("commission_source") == "ibkr" and incoming.get("commission_source") != "ibkr":
        merged["commission_source"] = "ibkr"
        merged["commission"] = existing.get("commission", "")
        merged["commission_currency"] = existing.get("commission_currency", "")
        merged["realized_pnl"] = existing.get("realized_pnl", "")
    return merged


def upsert_fill_row(recorder: LiveDataRecorder, row: dict[str, Any]) -> str:
    execution_id = str(row.get("execution_id") or "").strip()
    if not execution_id:
        raise ValueError("fill row missing execution_id")
    path = recorder.path("fills.csv")
    rows = read_csv_rows(path)
    for idx, existing in enumerate(rows):
        if str(existing.get("execution_id") or "").strip() == execution_id:
            merged = merge_fill_rows(existing, row)
            if merged == existing:
                return "duplicate"
            rows[idx] = merged
            write_csv_rows_atomic(path, rows, FILL_FIELDS)
            return "updated"
    clean = {k: row.get(k, "") for k in FILL_FIELDS}
    rows.append(clean)
    write_csv_rows_atomic(path, rows, FILL_FIELDS)
    return "inserted"


def record_commission_report(recorder: LiveDataRecorder, commission_report: Any) -> str:
    row = commission_report_row(commission_report)
    execution_id = row.get("execution_id")
    if not execution_id:
        return "missing_exec_id"
    path = recorder.path("fills.csv")
    rows = read_csv_rows(path)
    for idx, existing in enumerate(rows):
        if str(existing.get("execution_id") or "").strip() == str(execution_id):
            merged = merge_fill_rows(existing, row)
            rows[idx] = merged
            write_csv_rows_atomic(path, rows, FILL_FIELDS)
            safe_sqlite_call(getattr(recorder, "sqlite_store", None), "upsert_execution", merged)
            if merged.get("commission_source") == "ibkr":
                rate_limited_recorder_log(
                    recorder,
                    "COMMISSION_REPORT_MATCHED",
                    f"{now_utc()} COMMISSION_REPORT_MATCHED execution_id={execution_id}",
                    key=str(execution_id),
                )
                return "matched"
            print(f"{now_utc()} COMMISSION_REPORT_MISSING execution_id={execution_id}", flush=True)
            return "missing"
    placeholder = {k: "" for k in FILL_FIELDS}
    placeholder.update(row)
    rows.append(placeholder)
    write_csv_rows_atomic(path, rows, FILL_FIELDS)
    safe_sqlite_call(getattr(recorder, "sqlite_store", None), "upsert_execution", placeholder)
    print(f"{now_utc()} COMMISSION_REPORT_MISSING execution_id={execution_id} reason=execution_not_seen_yet", flush=True)
    return "placeholder"


def install_commission_report_handler(ib: IB, recorder: LiveDataRecorder) -> None:
    if getattr(ib, "_v67_commission_report_handler_installed", False):
        return
    if not hasattr(ib, "commissionReportEvent"):
        return

    def _on_commission_report(*event_args: Any) -> None:
        try:
            report = event_args[-1] if event_args else None
            record_commission_report(recorder, report)
        except Exception as exc:
            print(f"{now_utc()} COMMISSION_REPORT_HANDLER_FAILED error={exc!r}", flush=True)

    try:
        ib.commissionReportEvent += _on_commission_report
        setattr(ib, "_v67_commission_report_handler_installed", True)
    except Exception as exc:
        print(f"{now_utc()} COMMISSION_REPORT_HANDLER_INSTALL_FAILED error={exc!r}", flush=True)


def record_recent_fills(ib: IB, recorder: LiveDataRecorder, seen: set[str]) -> int:
    try:
        fills = ib.reqExecutions(ExecutionFilter())
    except Exception:
        fills = ib.fills()
    count = 0
    for fill in fills:
        key = fill_key(fill)
        row = fill_row_from_ibkr_fill(fill)
        status = upsert_fill_row(recorder, row)
        safe_sqlite_call(getattr(recorder, "sqlite_store", None), "upsert_execution", row)
        if status != "duplicate":
            if row.get("commission_source") != "ibkr":
                rate_limited_recorder_log(
                    recorder,
                    "FILLS_WITHOUT_COMMISSION",
                    f"{now_utc()} FILLS_WITHOUT_COMMISSION execution_id={row.get('execution_id')} symbol={row.get('symbol')}",
                    key=str(row.get("execution_id") or ""),
                )
            else:
                rate_limited_recorder_log(
                    recorder,
                    "COMMISSION_REPORT_MATCHED",
                    f"{now_utc()} COMMISSION_REPORT_MATCHED execution_id={row.get('execution_id')}",
                    key=str(row.get("execution_id") or ""),
                )
        if key in seen and status != "inserted":
            continue
        seen.add(key)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="v66 IBKR account/portfolio recorder")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--disable-sqlite", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=int, default=0, help="0 = run forever")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    install_unified_logger(args.log_dir)
    recorder = LiveDataRecorder(args.recorder_dir)
    sqlite_store = None if args.disable_sqlite else open_sqlite_store(args.sqlite_path)
    setattr(recorder, "sqlite_store", sqlite_store)
    recorder.record_run_metadata({
        "module": "v66_ibkr_account_recorder",
        "host": args.host,
        "port": args.port,
        "client_id": args.client_id,
        "interval_seconds": args.interval_seconds,
    })

    print("=== v66 IBKR account recorder ===")
    print(f"Recorder: {recorder.session_dir}")
    print(f"IBKR: {args.host}:{args.port} clientId={args.client_id}")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15, readonly=True)
    install_commission_report_handler(ib, recorder)
    seen_fills: set[str] = set()
    start = time.time()
    try:
        while True:
            ib.sleep(1)
            record_account_snapshot(ib, recorder)
            new_fills = record_recent_fills(ib, recorder, seen_fills)
            print(f"{now_utc()} portfolio recorded new_fills={new_fills}", flush=True)
            if args.duration_seconds and time.time() - start >= args.duration_seconds:
                break
            ib.sleep(max(0.0, args.interval_seconds - 1.0))
    finally:
        if sqlite_store is not None:
            sqlite_store.close()
        ib.disconnect()
        print("Disconnected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
