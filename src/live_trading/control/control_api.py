from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ib_insync import MarketOrder, Stock


JsonDict = dict[str, Any]
RecordLifecycleFn = Callable[..., None]
PersistManagedPositionsFn = Callable[[Any, dict[str, Any]], None]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **fields: Any) -> None:
    tail = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{_now_utc()} CONTROL_API_{event}" + (f" {tail}" if tail else ""), flush=True)


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


def _utc_minutes(value: str) -> int:
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    return hh * 60 + mm


def _in_utc_window(start_hhmm: str, end_hhmm: str) -> bool:
    now = datetime.now(timezone.utc)
    now_min = now.hour * 60 + now.minute
    start = _utc_minutes(start_hhmm)
    end = _utc_minutes(end_hhmm)
    if end <= start:
        return now_min >= start or now_min < end
    return start <= now_min < end


def _collector_allowed(runtime_state: dict[str, Any]) -> tuple[bool, str]:
    market_open = str(runtime_state.get("market_open_utc", "15:00"))
    market_close = str(runtime_state.get("market_close_utc", "20:00"))
    if _in_utc_window(market_open, market_close):
        return False, "market_session_active"
    start = str(runtime_state.get("history_collector_start_utc", "20:15"))
    end = str(runtime_state.get("history_collector_end_utc", "15:00"))
    if not _in_utc_window(start, end):
        return False, "outside_collector_window"
    return True, "allowed"


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
    if pos is None:
        _log("FLATTEN_REJECTED", symbol=symbol, status="not_found_or_inactive")
        return {"ok": False, "symbol": symbol, "status": "not_found_or_inactive"}
    if bool(getattr(pos, "exit_sent", False)):
        _log("FLATTEN_SKIPPED", symbol=symbol, status="already_exit_sent")
        return {"ok": True, "symbol": symbol, "status": "already_exit_sent", "position": _position_payload(symbol, pos)}

    qty_raw = getattr(pos, "quantity", 0)
    try:
        qty = int(abs(float(qty_raw)))
    except Exception:
        qty = 0
    if qty <= 0:
        _log("FLATTEN_REJECTED", symbol=symbol, status="bad_quantity", quantity=qty_raw)
        return {"ok": False, "symbol": symbol, "status": "bad_quantity", "quantity": qty_raw}

    action = "SELL" if float(qty_raw) > 0 else "BUY"
    payload: JsonDict = {
        "ok": True,
        "symbol": symbol,
        "status": "dry_run" if dry_run else "queued",
        "action": action,
        "quantity": qty,
        "position": _position_payload(symbol, pos),
    }

    if dry_run:
        _log("FLATTEN_DRY_RUN", symbol=symbol, action=action, quantity=qty)
        ctx.record_lifecycle_fn(ctx.recorder, "MANUAL_FLATTEN_DRY_RUN", symbol, action=action, quantity=qty, reason="control_api_flatten_symbol_dry_run", entry_price=getattr(pos, "entry_price", None), peak_price=getattr(pos, "peak_price", None), raw_json=payload)
        return payload

    command_id = uuid.uuid4().hex
    command = {"id": command_id, "type": "flatten_symbol", "symbol": symbol, "action": action, "quantity": qty}
    queue = _ensure_queue(ctx)
    queue.append(command)
    _log("FLATTEN_QUEUED", symbol=symbol, action=action, quantity=qty, command_id=command_id, pending=len(queue))
    ctx.record_lifecycle_fn(ctx.recorder, "MANUAL_FLATTEN_QUEUED", symbol, action=action, quantity=qty, reason="control_api_flatten_symbol_queued", entry_price=getattr(pos, "entry_price", None), peak_price=getattr(pos, "peak_price", None), raw_json={**payload, "command_id": command_id})
    payload["command_id"] = command_id
    return payload


def _queue_history_collector(ctx: ControlApiContext, body: JsonDict) -> JsonDict:
    allowed, reason = _collector_allowed(ctx.runtime_state)
    if not allowed:
        _log("HISTORY_COLLECTOR_REJECTED", reason=reason)
        return {"ok": False, "status": "rejected", "reason": reason}

    if ctx.runtime_state.get("history_collector_process") is not None:
        _log("HISTORY_COLLECTOR_REJECTED", reason="already_running")
        return {"ok": False, "status": "rejected", "reason": "already_running"}

    command_id = uuid.uuid4().hex
    today = datetime.now(timezone.utc).date().isoformat()
    cmd = {
        "id": command_id,
        "type": "history_collector",
        "start_date": str(body.get("start_date") or ctx.runtime_state.get("history_collector_start_date") or "2026-01-01"),
        "end_date": str(body.get("end_date") or today),
        "session_type": str(body.get("session_type") or ctx.runtime_state.get("history_collector_session_type") or "RTH").upper(),
        "max_tasks": int(body.get("max_tasks") or ctx.runtime_state.get("history_collector_max_tasks") or 300),
        "limit_symbols": int(body.get("limit_symbols") or ctx.runtime_state.get("history_collector_limit_symbols") or 0),
        "client_id": int(body.get("client_id") or ctx.runtime_state.get("history_collector_client_id") or 168),
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
        "--allow-outside-window",
    ]
    limit = int(cmd.get("limit_symbols") or 0)
    if limit > 0:
        args.extend(["--limit-symbols", str(limit)])
    return args


def process_history_collector_commands(*, runtime_state: dict[str, Any], max_commands: int = 1) -> int:
    proc = runtime_state.get("history_collector_process")
    if proc is not None:
        rc = proc.poll()
        if rc is None:
            return 0
        cmd = runtime_state.get("history_collector_running_command") or {}
        _log("HISTORY_COLLECTOR_DONE", command_id=cmd.get("id"), returncode=rc)
        runtime_state["history_collector_process"] = None
        runtime_state["history_collector_running_command"] = None
        if rc == 0:
            runtime_state["history_collector_last_run_key"] = f"{cmd.get('end_date')}_{cmd.get('session_type')}"

    queue = runtime_state.setdefault("history_collector_commands", [])
    if not queue:
        return 0

    allowed, reason = _collector_allowed(runtime_state)
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
        try:
            proc = subprocess.Popen(args)
            runtime_state["history_collector_process"] = proc
            runtime_state["history_collector_running_command"] = cmd
            _log("HISTORY_COLLECTOR_PROCESS_STARTED", command_id=command_id, pid=proc.pid, remaining=len(queue))
        except Exception as exc:
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
        if pos is None or not bool(getattr(pos, "active", False)):
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="not_found_or_inactive", command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="not_found_or_inactive", raw_json=cmd)
            continue
        if bool(getattr(pos, "exit_sent", False)):
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="already_exit_sent", command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="already_exit_sent", raw_json=cmd)
            continue

        qty_raw = getattr(pos, "quantity", cmd.get("quantity", 0))
        try:
            qty = int(abs(float(qty_raw)))
        except Exception:
            qty = 0
        if qty <= 0:
            _log("FLATTEN_SKIPPED", symbol=symbol, reason="bad_quantity", quantity=qty_raw, command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="bad_quantity", raw_json=cmd)
            continue

        action = "SELL" if float(qty_raw) > 0 else "BUY"
        contract = _make_contract(pos, symbol)
        try:
            qualified = ib.qualifyContracts(contract)
            if qualified:
                contract = qualified[0]
        except Exception as exc:
            _log("FLATTEN_FAILED", symbol=symbol, action=action, quantity=qty, reason="qualify_failed", error=repr(exc), command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, reason=f"qualify_failed:{exc!r}", raw_json=cmd)
            continue

        order = MarketOrder(action, qty)
        order.tif = "DAY"
        order.outsideRth = False

        try:
            trade = ib.placeOrder(contract, order)
        except Exception as exc:
            _log("FLATTEN_FAILED", symbol=symbol, action=action, quantity=qty, reason="place_order_failed", error=repr(exc), command_id=command_id)
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, reason=f"place_order_failed:{exc!r}", raw_json=cmd)
            continue

        order_id = getattr(getattr(trade, "order", None), "orderId", None)
        setattr(pos, "exit_sent", True)
        setattr(pos, "active", False)
        _log("FLATTEN_SENT", symbol=symbol, action=action, quantity=qty, order_id=order_id, command_id=command_id)
        record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SENT", symbol, action=action, quantity=qty, order_id=order_id, reason="control_api_queue_main_loop", entry_price=getattr(pos, "entry_price", None), peak_price=getattr(pos, "peak_price", None), raw_json={**cmd, "order_id": order_id})
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
            _json_response(self, 200, {"ok": True, "active_positions": len(active), "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False)), "pending_commands": pending, "pending_history_collector_commands": pending_history, "history_collector_running": history_running})
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = _read_json_body(self)

        if parsed.path == "/pause_entries":
            self.ctx.runtime_state["entries_blocked"] = True
            _log("PAUSE_ENTRIES")
            _json_response(self, 200, {"ok": True, "entries_blocked": True})
            return

        if parsed.path == "/resume_entries":
            self.ctx.runtime_state["entries_blocked"] = False
            _log("RESUME_ENTRIES")
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
            symbols = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            _log("FLATTEN_ALL_REQUEST", dry_run=dry_run, symbols=len(symbols))
            results = [_flatten_request(self.ctx, s, dry_run) for s in symbols]
            _json_response(self, 200, {"ok": True, "dry_run": dry_run, "count": len(results), "results": results})
            return

        if parsed.path == "/run_history_collector":
            payload = _queue_history_collector(self.ctx, body)
            _json_response(self, 200 if payload.get("ok") else 400, payload)
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
