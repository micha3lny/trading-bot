from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

try:
    from ib_insync import MarketOrder, Stock
except ImportError:  # pragma: no cover - exercised on minimal test environments
    class _MissingIbInsync:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("ib_insync is required for live Control API order placement")

    MarketOrder = Stock = _MissingIbInsync


JsonDict = dict[str, Any]
RecordLifecycleFn = Callable[..., None]
PersistManagedPositionsFn = Callable[[Any, dict[str, Any]], None]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **fields: Any) -> None:
    tail = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{_now_utc()} CONTROL_API_{event}" + (f" {tail}" if tail else ""), flush=True)


def _record_exit_order_intent(recorder: Any, *, order_id: Any, symbol: str, quantity: Any, reason: str, pos: Any = None, raw_json: JsonDict | None = None) -> None:
    store = getattr(recorder, "sqlite_store", None)
    if store is None:
        return
    try:
        store.record_exit_order_intent(
            order_id=order_id,
            symbol=symbol,
            exit_reason=reason,
            quantity=quantity,
            submitted_at=_now_utc(),
            position_key=f"{getattr(pos, 'strategy_name', '') or 'unknown'}:{getattr(recorder, 'session_date', '')}:{symbol}" if pos is not None else "",
            strategy_name=getattr(pos, "strategy_name", "") or "unknown",
            session_date=getattr(recorder, "session_date", ""),
            raw_json=raw_json or {},
        )
    except Exception as exc:
        _log("EXIT_REASON_INTENT_WRITE_FAILED", symbol=symbol, order_id=order_id, error=repr(exc))


@dataclass
class ControlApiContext:
    ib: Any
    recorder: Any
    managed_positions: dict[str, Any]
    runtime_state: dict[str, Any]
    record_lifecycle_fn: RecordLifecycleFn
    persist_managed_positions_fn: PersistManagedPositionsFn | None = None


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: JsonDict) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> JsonDict:
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except Exception:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        return {}


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_symbol(parsed: Any, body: JsonDict) -> str:
    qs = parse_qs(parsed.query or "")
    symbol = body.get("symbol") or (qs.get("symbol") or [""])[0]
    return str(symbol).upper().strip()


def _extract_dry_run(parsed: Any, body: JsonDict) -> bool:
    qs = parse_qs(parsed.query or "")
    value = body.get("dry_run")
    if value is None and "dry_run" in qs:
        value = qs.get("dry_run", [None])[0]
    return _bool_value(value, False)


def _extract_force(parsed: Any, body: JsonDict) -> bool:
    qs = parse_qs(parsed.query or "")
    value = body.get("force")
    if value is None and "force" in qs:
        value = qs.get("force", [None])[0]
    return _bool_value(value, False)


def _position_payload(symbol: str, pos: Any) -> JsonDict:
    return {
        "symbol": symbol,
        "quantity": getattr(pos, "quantity", None),
        "entry_price": getattr(pos, "entry_price", None),
        "peak_price": getattr(pos, "peak_price", None),
        "active": getattr(pos, "active", None),
        "exit_sent": getattr(pos, "exit_sent", None),
        "source": getattr(pos, "source", None),
    }


def _resolve_position(ctx: ControlApiContext, symbol: str) -> Any | None:
    pos = ctx.managed_positions.get(symbol)
    if pos is None:
        return None
    if not bool(getattr(pos, "active", False)):
        return None
    return pos


def _make_contract(pos: Any, symbol: str) -> Any:
    existing = getattr(pos, "contract", None)
    if existing is not None:
        exchange = getattr(existing, "exchange", None)
        if exchange:
            return existing
    return Stock(symbol, "SMART", "USD")


def _smart_stock_from_contract(symbol: str, contract: Any | None = None) -> Any:
    currency = str(getattr(contract, "currency", "") or "USD")
    return Stock(symbol, "SMART", currency)


def _portfolio_rows_by_symbol(ctx: ControlApiContext) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    try:
        portfolio = list(ctx.ib.portfolio())
    except Exception as exc:
        _log("PORTFOLIO_READ_FAILED", error=repr(exc))
        return rows
    for item in portfolio:
        contract = getattr(item, "contract", None)
        symbol = str(getattr(contract, "symbol", "") or "").upper().strip()
        if not symbol:
            continue
        try:
            quantity = float(getattr(item, "position", 0) or 0)
        except Exception:
            quantity = 0.0
        if abs(quantity) <= 0:
            continue
        rows[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "contract": contract,
            "average_cost": getattr(item, "averageCost", None),
            "market_price": getattr(item, "marketPrice", None),
            "source": "ibkr_portfolio",
        }
    return rows


def _is_fractional_quantity(quantity: float) -> bool:
    return not math.isclose(abs(quantity), round(abs(quantity)), rel_tol=0.0, abs_tol=1e-9)


def _public_command_payload(cmd: JsonDict) -> JsonDict:
    return {k: v for k, v in cmd.items() if k != "contract"}


def _utc_minutes(value: str) -> int:
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    return hh * 60 + mm


def _in_utc_window(start_hhmm: str, end_hhmm: str, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    now_min = now.hour * 60 + now.minute
    start = _utc_minutes(start_hhmm)
    end = _utc_minutes(end_hhmm)
    if end <= start:
        return now_min >= start or now_min < end
    return start <= now_min < end


def _collector_allowed(runtime_state: dict[str, Any], *, force: bool = False, now: datetime | None = None) -> tuple[bool, str]:
    if force:
        return True, "forced"
    market_open = str(runtime_state.get("market_open_utc", "15:00"))
    market_close = str(runtime_state.get("market_close_utc", "20:00"))
    if _in_utc_window(market_open, market_close, now):
        return False, "market_session_active"
    start = str(runtime_state.get("history_collector_start_utc", "20:15"))
    end = str(runtime_state.get("history_collector_end_utc", "15:00"))
    if not _in_utc_window(start, end, now):
        return False, "outside_collector_window"
    return True, "allowed"


def _history_collector_live_session_allowed(body_or_cmd: JsonDict) -> bool:
    return _bool_value(body_or_cmd.get("allow_live_session"), False)


def _collector_allowed_for_history_command(
    runtime_state: dict[str, Any],
    body_or_cmd: JsonDict,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> tuple[bool, str]:
    live_session_allowed = _history_collector_live_session_allowed(body_or_cmd)
    if force and not live_session_allowed:
        allowed_without_force, reason_without_force = _collector_allowed(runtime_state, force=False, now=now)
        if not allowed_without_force and reason_without_force == "market_session_active":
            return False, "market_session_active"
    return _collector_allowed(runtime_state, force=force, now=now)


def _ensure_queue(ctx: ControlApiContext) -> list[JsonDict]:
    queue = ctx.runtime_state.setdefault("control_api_commands", [])
    if not isinstance(queue, list):
        queue = []
        ctx.runtime_state["control_api_commands"] = queue
    return queue


def _ensure_history_queue(ctx: ControlApiContext) -> list[JsonDict]:
    queue = ctx.runtime_state.setdefault("history_collector_commands", [])
    if not isinstance(queue, list):
        queue = []
        ctx.runtime_state["history_collector_commands"] = queue
    return queue


def _flatten_request(ctx: ControlApiContext, symbol: str, dry_run: bool) -> JsonDict:
    pos = _resolve_position(ctx, symbol)
    portfolio_row = None
    if pos is None:
        portfolio_row = _portfolio_rows_by_symbol(ctx).get(symbol)
        if portfolio_row is None:
            _log("FLATTEN_REJECTED", symbol=symbol, status="not_found_or_inactive_or_ibkr_flat")
            return {"ok": False, "symbol": symbol, "status": "not_found_or_inactive_or_ibkr_flat"}
    if pos is not None and bool(getattr(pos, "exit_sent", False)):
        _log("FLATTEN_SKIPPED", symbol=symbol, status="already_exit_sent")
        return {"ok": True, "symbol": symbol, "status": "already_exit_sent", "position": _position_payload(symbol, pos)}
    if pos is not None and not bool(getattr(pos, "entry_fill_verified", False)):
        _log("FLATTEN_REJECTED", symbol=symbol, status="entry_fill_not_verified")
        ctx.record_lifecycle_fn(
            ctx.recorder,
            "EXIT_ORDER_BLOCKED_NO_ENTRY_FILL",
            symbol,
            action="SELL",
            quantity=getattr(pos, "quantity", 0),
            reason="control_api_entry_fill_not_verified",
            entry_fill_verified="false",
            raw_json={"source": "control_api_flatten_symbol"},
        )
        return {"ok": False, "symbol": symbol, "status": "entry_fill_not_verified", "position": _position_payload(symbol, pos)}

    qty_raw = portfolio_row["quantity"] if portfolio_row is not None else getattr(pos, "quantity", 0)
    try:
        qty_float = abs(float(qty_raw))
    except Exception:
        qty_float = 0.0
    if qty_float <= 0:
        _log("FLATTEN_REJECTED", symbol=symbol, status="bad_quantity", quantity=qty_raw)
        return {"ok": False, "symbol": symbol, "status": "bad_quantity", "quantity": qty_raw}
    fractional = _is_fractional_quantity(float(qty_raw))
    qty = int(round(qty_float)) if not fractional else qty_float

    action = "SELL" if float(qty_raw) > 0 else "BUY"
    source = "ibkr_portfolio" if portfolio_row is not None else "managed_position"
    payload: JsonDict = {
        "ok": True,
        "symbol": symbol,
        "status": "dry_run" if dry_run else "queued",
        "action": action,
        "quantity": qty,
        "source": source,
        "fractional": fractional,
        "position": _position_payload(symbol, pos) if pos is not None else None,
        "ibkr_position": {k: v for k, v in (portfolio_row or {}).items() if k != "contract"} or None,
    }

    if dry_run:
        _log("FLATTEN_DRY_RUN", symbol=symbol, action=action, quantity=qty, source=source)
        ctx.record_lifecycle_fn(ctx.recorder, "MANUAL_FLATTEN_DRY_RUN", symbol, action=action, quantity=qty, reason="control_api_flatten_symbol_dry_run", entry_price=getattr(pos, "entry_price", None) if pos is not None else None, peak_price=getattr(pos, "peak_price", None) if pos is not None else None, raw_json=payload)
        return payload

    command_id = uuid.uuid4().hex
    command = {
        "id": command_id,
        "type": "flatten_symbol",
        "symbol": symbol,
        "action": action,
        "quantity": qty,
        "ibkr_quantity": float(qty_raw),
        "source": source,
        "contract": portfolio_row.get("contract") if portfolio_row is not None else None,
    }
    queue = _ensure_queue(ctx)
    queue.append(command)
    _log("FLATTEN_QUEUED", symbol=symbol, action=action, quantity=qty, source=source, command_id=command_id, pending=len(queue))
    ctx.record_lifecycle_fn(ctx.recorder, "MANUAL_FLATTEN_QUEUED", symbol, action=action, quantity=qty, reason="control_api_flatten_symbol_queued", entry_price=getattr(pos, "entry_price", None) if pos is not None else None, peak_price=getattr(pos, "peak_price", None) if pos is not None else None, raw_json={**payload, "command_id": command_id})
    payload["command_id"] = command_id
    return payload


def _history_collector_status(runtime_state: dict[str, Any]) -> JsonDict:
    queue = runtime_state.setdefault("history_collector_commands", [])
    proc = runtime_state.get("history_collector_process")
    running = bool(proc is not None and proc.poll() is None)
    running_command = runtime_state.get("history_collector_running_command")
    return {
        "ok": True,
        "running": running,
        "pid": getattr(proc, "pid", None) if running else None,
        "running_command": running_command,
        "pending_commands": len(queue) if isinstance(queue, list) else 0,
        "last_run_key": runtime_state.get("history_collector_last_run_key"),
        "last_cancelled_at": runtime_state.get("history_collector_last_cancelled_at"),
        "last_returncode": runtime_state.get("history_collector_last_returncode"),
        "max_runtime_minutes": runtime_state.get("history_collector_max_runtime_minutes"),
    }


def _cancel_history_collector(ctx: ControlApiContext, *, force: bool = False) -> JsonDict:
    queue = _ensure_history_queue(ctx)
    cancelled_pending = len(queue)
    queue.clear()

    proc = ctx.runtime_state.get("history_collector_process")
    running_command = ctx.runtime_state.get("history_collector_running_command")
    killed = False
    if proc is not None and proc.poll() is None:
        if force:
            proc.kill()
            killed = True
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            if not force:
                proc.kill()
                killed = True
    ctx.runtime_state["history_collector_process"] = None
    ctx.runtime_state["history_collector_running_command"] = None
    ctx.runtime_state["history_collector_started_monotonic"] = None
    ctx.runtime_state["history_collector_last_cancelled_at"] = _now_utc()
    _log("HISTORY_COLLECTOR_CANCELLED", pending=cancelled_pending, force=force, killed=killed)
    return {
        "ok": True,
        "status": "cancelled",
        "cancelled_pending_commands": cancelled_pending,
        "running_command": running_command,
        "force": force,
        "killed": killed,
    }


def _queue_history_collector(ctx: ControlApiContext, body: JsonDict, *, force: bool = False) -> JsonDict:
    allowed, reason = _collector_allowed_for_history_command(ctx.runtime_state, body, force=force)
    if not allowed:
        _log("HISTORY_COLLECTOR_REJECTED", reason=reason)
        return {"ok": False, "status": "rejected", "reason": reason}

    if ctx.runtime_state.get("history_collector_process") is not None:
        _log("HISTORY_COLLECTOR_REJECTED", reason="already_running")
        return {"ok": False, "status": "rejected", "reason": "already_running"}

    command_id = uuid.uuid4().hex
    today = datetime.now(timezone.utc).date().isoformat()
    requested_date = body.get("date")
    start_date = str(body.get("start_date") or requested_date or ctx.runtime_state.get("history_collector_start_date") or "2026-01-01")
    end_date = str(body.get("end_date") or requested_date or today)
    cmd = {
        "id": command_id,
        "type": "history_collector",
        "start_date": start_date,
        "end_date": end_date,
        "session_type": str(body.get("session_type") or ctx.runtime_state.get("history_collector_session_type") or "RTH").upper(),
        "max_tasks": int(body.get("max_tasks") or ctx.runtime_state.get("history_collector_max_tasks") or 300),
        "max_attempts": int(body.get("max_attempts") or ctx.runtime_state.get("history_collector_max_attempts") or 5),
        "limit_symbols": int(body.get("limit_symbols") or ctx.runtime_state.get("history_collector_limit_symbols") or 0),
        "client_id": int(body.get("client_id") or ctx.runtime_state.get("history_collector_client_id") or 168),
        "force": bool(force),
        "allow_live_session": _history_collector_live_session_allowed(body),
        "plan_only": _bool_value(body.get("plan_only"), False),
        "include_weekends": _bool_value(body.get("include_weekends"), False),
        "retry_failed": _bool_value(body.get("retry_failed"), False),
    }
    queue = _ensure_history_queue(ctx)
    queue.append(cmd)
    _log("HISTORY_COLLECTOR_QUEUED", command_id=command_id, start=cmd["start_date"], end=cmd["end_date"], session=cmd["session_type"], max_tasks=cmd["max_tasks"], pending=len(queue))
    return {"ok": True, "status": "queued", "command_id": command_id, "command": cmd}


def _build_history_collector_args(cmd: JsonDict) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "src.live_trading.data.v68_universe_1m_parquet_collector",
        "--start-date", str(cmd.get("start_date") or "2026-01-01"),
        "--end-date", str(cmd.get("end_date") or datetime.now(timezone.utc).date().isoformat()),
        "--session-type", str(cmd.get("session_type") or "RTH"),
        "--client-id", str(int(cmd.get("client_id") or 168)),
        "--max-tasks", str(int(cmd.get("max_tasks") or 300)),
        "--max-attempts", str(int(cmd.get("max_attempts") or 5)),
        "--allow-outside-window",
    ]
    limit = int(cmd.get("limit_symbols") or 0)
    if limit > 0:
        args.extend(["--limit-symbols", str(limit)])
    if _bool_value(cmd.get("plan_only"), False):
        args.append("--plan-only")
    if _bool_value(cmd.get("include_weekends"), False):
        args.append("--include-weekends")
    if _bool_value(cmd.get("retry_failed"), False):
        args.append("--retry-failed")
    return args


def process_history_collector_commands(*, runtime_state: dict[str, Any], max_commands: int = 1) -> int:
    proc = runtime_state.get("history_collector_process")
    if proc is not None:
        cmd = runtime_state.get("history_collector_running_command") or {}
        started = runtime_state.get("history_collector_started_monotonic")
        duration = int(time.monotonic() - float(started)) if started is not None else None
        try:
            rc = proc.poll()
        except Exception as exc:
            _log("HISTORY_COLLECTOR_STALE_HANDLE_CLEARED", command_id=cmd.get("id"), pid=getattr(proc, "pid", None), error=repr(exc))
            runtime_state["history_collector_process"] = None
            runtime_state["history_collector_running_command"] = None
            runtime_state["history_collector_started_monotonic"] = None
            runtime_state["history_collector_last_returncode"] = None
            return 0
        if rc is None:
            max_runtime_minutes = float(runtime_state.get("history_collector_max_runtime_minutes") or 120)
            if max_runtime_minutes > 0 and duration is not None and duration >= int(max_runtime_minutes * 60):
                _log(
                    "HISTORY_COLLECTOR_TIMEOUT",
                    command_id=cmd.get("id"),
                    pid=getattr(proc, "pid", None),
                    max_runtime_minutes=max_runtime_minutes,
                    duration_seconds=duration,
                )
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                runtime_state["history_collector_process"] = None
                runtime_state["history_collector_running_command"] = None
                runtime_state["history_collector_started_monotonic"] = None
                runtime_state["history_collector_last_returncode"] = -9
                return 0
            return 0
        event = "HISTORY_COLLECTOR_DONE" if rc == 0 else "HISTORY_COLLECTOR_FAILED"
        if rc == 75:
            event = "HISTORY_COLLECTOR_LOCK_HELD"
        if duration is None:
            _log(event, command_id=cmd.get("id"), returncode=rc)
        else:
            _log(event, command_id=cmd.get("id"), returncode=rc, duration_seconds=duration)
        if cmd.get("source") == "overnight_scheduler":
            print(
                f"{_now_utc()} OVERNIGHT_COLLECTOR_DONE command_id={cmd.get('id')} "
                f"mode={cmd.get('collector_mode', 'unknown')} slot={cmd.get('schedule_slot_utc')} returncode={rc}",
                flush=True,
            )
        if str(cmd.get("collector_mode") or "") in {"daily", "startup_repair"}:
            catchup_event = "HISTORY_CATCHUP_DONE" if rc == 0 else "HISTORY_CATCHUP_FAILED"
            print(
                f"{_now_utc()} {catchup_event} command_id={cmd.get('id')} "
                f"ranking_date={cmd.get('end_date')} mode={cmd.get('collector_mode')} "
                f"returncode={rc} duration_seconds={duration if duration is not None else ''}",
                flush=True,
            )
        runtime_state["history_collector_process"] = None
        runtime_state["history_collector_running_command"] = None
        runtime_state["history_collector_started_monotonic"] = None
        runtime_state["history_collector_last_returncode"] = rc
        runtime_state["history_collector_last_run_at"] = _now_utc()
        if rc == 0:
            runtime_state["history_collector_last_run_key"] = f"{cmd.get('end_date')}_{cmd.get('session_type')}"
            runtime_state["history_collector_last_successful_run_at"] = runtime_state["history_collector_last_run_at"]
        elif str(cmd.get("collector_mode") or "") in {"daily", "startup_repair"}:
            queue = runtime_state.setdefault("history_collector_commands", [])
            if isinstance(queue, list):
                before = len(queue)
                end_date = str(cmd.get("end_date") or "")
                queue[:] = [
                    queued
                    for queued in queue
                    if not (
                        isinstance(queued, dict)
                        and str(queued.get("collector_mode") or "") == "backlog"
                        and str(queued.get("end_date") or "") == end_date
                    )
                ]
                dropped = before - len(queue)
                if dropped:
                    _log(
                        "HISTORY_COLLECTOR_BACKLOG_DROPPED_LATEST_INCOMPLETE",
                        latest_date=end_date,
                        dropped=dropped,
                        failed_command_id=cmd.get("id"),
                        returncode=rc,
                    )

    queue = runtime_state.setdefault("history_collector_commands", [])
    if not queue:
        return 0

    next_cmd = (queue[0] or {}) if queue else {}
    allowed, reason = _collector_allowed_for_history_command(
        runtime_state,
        next_cmd,
        force=bool(next_cmd.get("force")) if next_cmd else False,
    )
    if not allowed:
        _log("HISTORY_COLLECTOR_DEFERRED", reason=reason, pending=len(queue))
        return 0

    processed = 0
    while queue and processed < max_commands and runtime_state.get("history_collector_process") is None:
        cmd = queue.pop(0)
        processed += 1
        command_id = cmd.get("id")
        args = _build_history_collector_args(cmd)
        _log("HISTORY_COLLECTOR_START", command_id=command_id, cmd=" ".join(args))
        if cmd.get("source") == "overnight_scheduler":
            print(
                f"{_now_utc()} OVERNIGHT_COLLECTOR_START command_id={command_id} mode={cmd.get('collector_mode', 'unknown')} "
                f"slot={cmd.get('schedule_slot_utc')} start={cmd.get('start_date')} "
                f"end={cmd.get('end_date')} max_tasks={cmd.get('max_tasks')}",
                flush=True,
            )
        try:
            proc = subprocess.Popen(args)
            runtime_state["history_collector_process"] = proc
            runtime_state["history_collector_running_command"] = cmd
            runtime_state["history_collector_started_monotonic"] = time.monotonic()
            _log("HISTORY_COLLECTOR_PROCESS_STARTED", command_id=command_id, pid=proc.pid, remaining=len(queue))
        except Exception as exc:
            runtime_state["history_collector_started_monotonic"] = None
            _log("HISTORY_COLLECTOR_FAILED", command_id=command_id, error=repr(exc), remaining=len(queue))
    return processed


def process_control_api_commands(
    *,
    ib: Any,
    recorder: Any,
    managed_positions: dict[str, Any],
    runtime_state: dict[str, Any],
    record_lifecycle_fn: RecordLifecycleFn,
    persist_managed_positions_fn: PersistManagedPositionsFn | None = None,
    max_commands: int = 20,
) -> int:
    queue = runtime_state.setdefault("control_api_commands", [])
    if not queue:
        return 0

    _log("QUEUE_PROCESS_START", pending=len(queue), max_commands=max_commands)
    processed = 0
    while queue and processed < max_commands:
        cmd = queue.pop(0)
        processed += 1
        symbol = str(cmd.get("symbol") or "").upper().strip()
        command_id = cmd.get("id")
        if not symbol:
            _log("COMMAND_SKIPPED", reason="missing_symbol", command_id=command_id)
            continue
        pos = managed_positions.get(symbol)
        source = str(cmd.get("source") or "managed_position")
        ibkr_quantity = cmd.get("ibkr_quantity")
        is_portfolio_flatten = source == "ibkr_portfolio" and ibkr_quantity is not None
        if (pos is None or not bool(getattr(pos, "active", False))) and not is_portfolio_flatten:
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="not_found_or_inactive", command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="not_found_or_inactive", raw_json=_public_command_payload(cmd))
            continue
        if pos is not None and bool(getattr(pos, "exit_sent", False)):
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="already_exit_sent", command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="already_exit_sent", raw_json=_public_command_payload(cmd))
            continue
        if pos is not None and not is_portfolio_flatten and not bool(getattr(pos, "entry_fill_verified", False)):
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="entry_fill_not_verified", command_id=command_id)
            record_lifecycle_fn(
                recorder,
                "EXIT_ORDER_BLOCKED_NO_ENTRY_FILL",
                symbol,
                action=cmd.get("action"),
                quantity=getattr(pos, "quantity", cmd.get("quantity", 0)),
                reason="control_api_entry_fill_not_verified",
                entry_fill_verified="false",
                raw_json=_public_command_payload(cmd),
            )
            continue

        qty_raw = ibkr_quantity if is_portfolio_flatten else getattr(pos, "quantity", cmd.get("quantity", 0))
        try:
            qty_float = abs(float(qty_raw))
        except Exception:
            qty_float = 0.0
        if qty_float <= 0:
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="bad_quantity", quantity=qty_raw, command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="bad_quantity", raw_json=_public_command_payload(cmd))
            continue
        if _is_fractional_quantity(float(qty_raw)):
            _log("FLATTEN_FAILED", symbol=symbol, reason="fractional_quantity_api_unsupported", quantity=qty_raw, command_id=command_id)
            record_lifecycle_fn(
                recorder,
                "MANUAL_FLATTEN_FAILED",
                symbol,
                action=cmd.get("action"),
                quantity=qty_float,
                reason="fractional_quantity_api_unsupported",
                raw_json={**_public_command_payload(cmd), "manual_action_required": "close_fractional_position_in_ibkr_desktop"},
            )
            continue
        qty = int(round(qty_float))

        action = "SELL" if float(qty_raw) > 0 else "BUY"
        contract = _smart_stock_from_contract(symbol, cmd.get("contract")) if is_portfolio_flatten else _make_contract(pos, symbol)
        try:
            qualified = ib.qualifyContracts(contract)
            if qualified:
                contract = qualified[0]
        except Exception as exc:
            _log("FLATTEN_FAILED", symbol=symbol, action=action, quantity=qty, reason="qualify_failed", error=repr(exc), command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, reason=f"qualify_failed:{exc!r}", raw_json=_public_command_payload(cmd))
            continue

        order = MarketOrder(action, qty)
        order.tif = "DAY"
        order.outsideRth = False

        try:
            trade = ib.placeOrder(contract, order)
        except Exception as exc:
            _log("FLATTEN_FAILED", symbol=symbol, action=action, quantity=qty, reason="place_order_failed", error=repr(exc), command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, reason=f"place_order_failed:{exc!r}", raw_json=_public_command_payload(cmd))
            continue
        try:
            sleep_fn = getattr(ib, "sleep", None)
            if callable(sleep_fn):
                sleep_fn(0.5)
        except Exception:
            pass

        order_id = getattr(getattr(trade, "order", None), "orderId", None)
        status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
        if status.lower() == "cancelled":
            log = getattr(trade, "log", None)
            error_message = str(log[-1].message) if log else "cancelled"
            _log("FLATTEN_FAILED", symbol=symbol, action=action, quantity=qty, reason="ibkr_cancelled", error=error_message, order_id=order_id, command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, order_id=order_id, reason=f"ibkr_cancelled:{error_message}", raw_json={**_public_command_payload(cmd), "order_id": order_id, "status": status})
            continue
        if pos is not None:
            setattr(pos, "exit_sent", True)
            setattr(pos, "last_exit_order_ts", datetime.now(timezone.utc).timestamp())
        _log("FLATTEN_SENT", symbol=symbol, action=action, quantity=qty, order_id=order_id, command_id=command_id)
        if action == "SELL":
            _record_exit_order_intent(
                recorder,
                order_id=order_id,
                symbol=symbol,
                quantity=qty,
                reason="control_api_queue_main_loop",
                pos=pos,
                raw_json={**_public_command_payload(cmd), "order_id": order_id, "status": status, "source": "control_api"},
            )
        record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SENT", symbol, action=action, quantity=qty, order_id=order_id, reason="control_api_queue_main_loop", entry_price=getattr(pos, "entry_price", None) if pos is not None else None, peak_price=getattr(pos, "peak_price", None) if pos is not None else None, raw_json={**_public_command_payload(cmd), "order_id": order_id, "status": status})
        if persist_managed_positions_fn is not None:
            try:
                persist_managed_positions_fn(recorder, managed_positions)
            except Exception as exc:
                _log("PERSIST_WARNING", symbol=symbol, error=repr(exc))

    _log("QUEUE_PROCESS_DONE", processed=processed, remaining=len(queue))
    return processed


class _ControlHandler(BaseHTTPRequestHandler):
    ctx: ControlApiContext

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            active = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            pending = len(self.ctx.runtime_state.get("control_api_commands", []) or [])
            pending_history = len(self.ctx.runtime_state.get("history_collector_commands", []) or [])
            running_proc = self.ctx.runtime_state.get("history_collector_process")
            history_running = bool(running_proc is not None and running_proc.poll() is None)
            verification = self.ctx.runtime_state.get("portfolio_last_verification") or self.ctx.runtime_state.get("eod_last_verification") or {}
            ibkr_open_symbols = verification.get("ibkr_open_symbols", []) if isinstance(verification, dict) else []
            drift_symbols = verification.get("drift_symbols", []) if isinstance(verification, dict) else []
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "active_positions": len(active),
                    "active_symbols": active,
                    "ibkr_open_positions": len(ibkr_open_symbols),
                    "ibkr_open_symbols": ibkr_open_symbols,
                    "drift_symbols": drift_symbols,
                    "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False)),
                    "pending_commands": pending,
                    "pending_history_collector_commands": pending_history,
                    "history_collector_running": history_running,
                    "sqlite_writer_status": self.ctx.runtime_state.get("sqlite_writer_status") or {},
                },
            )
            return
        if parsed.path == "/history_collector/status":
            _json_response(self, 200, _history_collector_status(self.ctx.runtime_state))
            return
        if parsed.path == "/risk_status":
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False)),
                    "entries_blocked_reason": self.ctx.runtime_state.get("entries_blocked_reason", ""),
                    "risk_guard": self.ctx.runtime_state.get("risk_guard_last_status") or {},
                },
            )
            return
        if parsed.path == "/eod/status":
            _json_response(self, 200, {"ok": True, "eod_flatten_requested": bool(self.ctx.runtime_state.get("manual_eod_flatten_requested", False)), "eod_flatten_requested_at": self.ctx.runtime_state.get("manual_eod_flatten_requested_at"), "eod_flatten_force": bool(self.ctx.runtime_state.get("manual_eod_flatten_force", False)), "eod_last_verification": self.ctx.runtime_state.get("eod_last_verification")})
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = _read_json_body(self)

        if parsed.path == "/pause_entries":
            self.ctx.runtime_state["entries_blocked"] = True
            self.ctx.runtime_state["entries_blocked_reason"] = str(body.get("reason") or "manual_control_api")
            _log("PAUSE_ENTRIES")
            _json_response(self, 200, {"ok": True, "entries_blocked": True})
            return

        if parsed.path == "/resume_entries":
            self.ctx.runtime_state["entries_blocked"] = False
            if self.ctx.runtime_state.get("entries_blocked_reason") in {"manual_control_api", "block_entries"}:
                self.ctx.runtime_state["entries_blocked_reason"] = ""
            _log("RESUME_ENTRIES")
            _json_response(self, 200, {"ok": True, "entries_blocked": False})
            return

        if parsed.path == "/block_entries":
            reason = str(body.get("reason") or "block_entries")
            self.ctx.runtime_state["entries_blocked"] = True
            self.ctx.runtime_state["entries_blocked_reason"] = reason
            _log("BLOCK_ENTRIES", reason=reason)
            _json_response(self, 200, {"ok": True, "entries_blocked": True, "reason": reason})
            return

        if parsed.path == "/unblock_entries":
            self.ctx.runtime_state["entries_blocked"] = False
            self.ctx.runtime_state["entries_blocked_reason"] = ""
            _log("UNBLOCK_ENTRIES")
            _json_response(self, 200, {"ok": True, "entries_blocked": False})
            return

        if parsed.path == "/flatten_symbol":
            symbol = _extract_symbol(parsed, body)
            if not symbol:
                _log("FLATTEN_REJECTED", reason="missing_symbol")
                _json_response(self, 400, {"ok": False, "error": "missing_symbol"})
                return
            payload = _flatten_request(self.ctx, symbol, _extract_dry_run(parsed, body))
            _json_response(self, 200 if payload.get("ok") else 400, payload)
            return

        if parsed.path == "/flatten_all_positions":
            dry_run = _extract_dry_run(parsed, body)
            managed_symbols = {
                s
                for s, p in self.ctx.managed_positions.items()
                if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))
            }
            portfolio_symbols = set(_portfolio_rows_by_symbol(self.ctx))
            symbols = sorted(managed_symbols | portfolio_symbols)
            _log("FLATTEN_ALL_REQUEST", dry_run=dry_run, symbols=len(symbols))
            results = [_flatten_request(self.ctx, s, dry_run) for s in symbols]
            _json_response(self, 200, {"ok": True, "dry_run": dry_run, "count": len(results), "results": results})
            return

        if parsed.path == "/run_history_collector":
            payload = _queue_history_collector(self.ctx, body, force=_extract_force(parsed, body))
            _json_response(self, 200 if payload.get("ok") else 400, payload)
            return

        if parsed.path == "/history_collector/cancel":
            payload = _cancel_history_collector(self.ctx, force=_extract_force(parsed, body))
            _json_response(self, 200, payload)
            return

        if parsed.path in {"/eod_flatten", "/eod/flatten"}:
            dry_run = _extract_dry_run(parsed, body)
            force = _extract_force(parsed, body)
            active = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            if dry_run:
                _json_response(self, 200, {"ok": True, "status": "dry_run", "active_positions": active, "count": len(active), "force": force})
                return
            self.ctx.runtime_state["entries_blocked"] = True
            self.ctx.runtime_state["manual_eod_flatten_requested"] = True
            self.ctx.runtime_state["manual_eod_flatten_requested_at"] = _now_utc()
            self.ctx.runtime_state["manual_eod_flatten_force"] = force
            _log("EOD_FLATTEN_REQUESTED", force=force, active_positions=len(active))
            _json_response(self, 200, {"ok": True, "status": "queued", "force": force, "active_positions": active, "count": len(active)})
            return

        if parsed.path == "/emergency_flatten":
            dry_run = _extract_dry_run(parsed, body)
            active = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            if dry_run:
                _json_response(self, 200, {"ok": True, "status": "dry_run", "active_positions": active, "count": len(active), "force": True})
                return
            self.ctx.runtime_state["entries_blocked"] = True
            self.ctx.runtime_state["entries_blocked_reason"] = "emergency_flatten"
            self.ctx.runtime_state["manual_eod_flatten_requested"] = True
            self.ctx.runtime_state["manual_eod_flatten_requested_at"] = _now_utc()
            self.ctx.runtime_state["manual_eod_flatten_force"] = True
            self.ctx.runtime_state["manual_eod_flatten_reason"] = "emergency_flatten"
            _log("EMERGENCY_FLATTEN_REQUESTED", active_positions=len(active))
            _json_response(self, 200, {"ok": True, "status": "queued", "force": True, "active_positions": active, "count": len(active)})
            return

        _json_response(self, 404, {"ok": False, "error": "not_found"})


def start_control_api(
    *,
    ib: Any,
    recorder: Any,
    managed_positions: dict[str, Any],
    runtime_state: dict[str, Any],
    record_lifecycle_fn: RecordLifecycleFn,
    persist_managed_positions_fn: PersistManagedPositionsFn | None = None,
    host: str = "127.0.0.1",
    port: int = 8767,
) -> ThreadingHTTPServer:
    runtime_state.setdefault("control_api_commands", [])
    runtime_state.setdefault("history_collector_commands", [])
    runtime_state.setdefault("history_collector_process", None)
    runtime_state.setdefault("history_collector_running_command", None)
    ctx = ControlApiContext(
        ib=ib,
        recorder=recorder,
        managed_positions=managed_positions,
        runtime_state=runtime_state,
        record_lifecycle_fn=record_lifecycle_fn,
        persist_managed_positions_fn=persist_managed_positions_fn,
    )

    class Handler(_ControlHandler):
        pass

    Handler.ctx = ctx
    server = ThreadingHTTPServer((host, int(port)), Handler)
    thread = threading.Thread(target=server.serve_forever, name="control_api", daemon=True)
    thread.start()
    _log("STARTED", host=host, port=port)
    return server
