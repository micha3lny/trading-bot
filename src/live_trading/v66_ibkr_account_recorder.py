from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

from ib_insync import IB, ExecutionFilter

from src.live_trading.v62_live_data_recorder import FillEvent, LiveDataRecorder, PortfolioSnapshot


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 66
DEFAULT_RECORDER_DIR = "data/live/recorder"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


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


def record_recent_fills(ib: IB, recorder: LiveDataRecorder, seen: set[str]) -> int:
    try:
        fills = ib.reqExecutions(ExecutionFilter())
    except Exception:
        fills = ib.fills()
    count = 0
    for fill in fills:
        key = fill_key(fill)
        if key in seen:
            continue
        seen.add(key)
        contract = fill.contract
        execution = fill.execution
        commission_report = fill.commissionReport
        raw = {
            "contract": contract.__dict__ if hasattr(contract, "__dict__") else str(contract),
            "execution": execution.__dict__ if hasattr(execution, "__dict__") else str(execution),
            "commissionReport": commission_report.__dict__ if hasattr(commission_report, "__dict__") else str(commission_report),
        }
        recorder.record_fill(FillEvent(
            symbol=getattr(contract, "symbol", ""),
            action=getattr(execution, "side", ""),
            quantity=safe_float(getattr(execution, "shares", None)),
            fill_price=safe_float(getattr(execution, "price", None)),
            commission=safe_float(getattr(commission_report, "commission", None)),
            order_id=str(getattr(execution, "orderId", "")),
            execution_id=str(getattr(execution, "execId", "")),
            realized_pnl=safe_float(getattr(commission_report, "realizedPNL", None)),
            raw_json=json.dumps(raw, default=str, ensure_ascii=False),
        ))
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="v66 IBKR account/portfolio recorder")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    recorder = LiveDataRecorder(args.recorder_dir)
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
        ib.disconnect()
        print("Disconnected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
