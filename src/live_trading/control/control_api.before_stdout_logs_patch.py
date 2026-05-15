from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ib_insync import MarketOrder, Stock


JsonDict = dict[str, Any]
RecordLifecycleFn = Callable[..., None]
PersistManagedPositionsFn = Callable[[Any, dict[str, Any]], None]


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


def _ensure_queue(ctx: ControlApiContext) -> list[JsonDict]:
    queue = ctx.runtime_state.setdefault("control_api_commands", [])
    if not isinstance(queue, list):
        queue = []
        ctx.runtime_state["control_api_commands"] = queue
    return queue


def _flatten_request(ctx: ControlApiContext, symbol: str, dry_run: bool) -> JsonDict:
    pos = _resolve_position(ctx, symbol)
    if pos is None:
        return {"ok": False, "symbol": symbol, "status": "not_found_or_inactive"}

    if bool(getattr(pos, "exit_sent", False)):
        return {"ok": True, "symbol": symbol, "status": "already_exit_sent", "position": _position_payload(symbol, pos)}

    qty_raw = getattr(pos, "quantity", 0)
    try:
        qty = int(abs(float(qty_raw)))
    except Exception:
        qty = 0
    if qty <= 0:
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
        ctx.record_lifecycle_fn(
            ctx.recorder,
            "MANUAL_FLATTEN_DRY_RUN",
            symbol,
            action=action,
            quantity=qty,
            price=None,
            reason="control_api_flatten_symbol_dry_run",
            entry_price=getattr(pos, "entry_price", None),
            peak_price=getattr(pos, "peak_price", None),
            raw_json=payload,
        )
        return payload

    command_id = uuid.uuid4().hex
    command = {
        "id": command_id,
        "type": "flatten_symbol",
        "symbol": symbol,
        "action": action,
        "quantity": qty,
    }
    _ensure_queue(ctx).append(command)
    ctx.record_lifecycle_fn(
        ctx.recorder,
        "MANUAL_FLATTEN_QUEUED",
        symbol,
        action=action,
        quantity=qty,
        price=None,
        reason="control_api_flatten_symbol_queued",
        entry_price=getattr(pos, "entry_price", None),
        peak_price=getattr(pos, "peak_price", None),
        raw_json={**payload, "command_id": command_id},
    )
    payload["command_id"] = command_id
    return payload


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
    """Execute queued control API commands from the trader's main thread.

    HTTP request threads only enqueue commands. This function must be called from
    the main trader loop where ib_insync already has a working event loop.
    """
    queue = runtime_state.setdefault("control_api_commands", [])
    if not queue:
        return 0

    processed = 0
    while queue and processed < max_commands:
        cmd = queue.pop(0)
        processed += 1
        symbol = str(cmd.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        pos = managed_positions.get(symbol)
        if pos is None or not bool(getattr(pos, "active", False)):
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="not_found_or_inactive", raw_json=cmd)
            continue
        if bool(getattr(pos, "exit_sent", False)):
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="already_exit_sent", raw_json=cmd)
            continue

        qty_raw = getattr(pos, "quantity", cmd.get("quantity", 0))
        try:
            qty = int(abs(float(qty_raw)))
        except Exception:
            qty = 0
        if qty <= 0:
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_SKIPPED", symbol, reason="bad_quantity", raw_json=cmd)
            continue

        action = "SELL" if float(qty_raw) > 0 else "BUY"
        contract = _make_contract(pos, symbol)
        try:
            qualified = ib.qualifyContracts(contract)
            if qualified:
                contract = qualified[0]
        except Exception as exc:
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, reason=f"qualify_failed:{exc!r}", raw_json=cmd)
            continue

        order = MarketOrder(action, qty)
        order.tif = "DAY"
        order.outsideRth = False

        try:
            trade = ib.placeOrder(contract, order)
        except Exception as exc:
            record_lifecycle_fn(recorder, "MANUAL_FLATTEN_FAILED", symbol, action=action, quantity=qty, reason=f"place_order_failed:{exc!r}", raw_json=cmd)
            continue

        order_id = getattr(getattr(trade, "order", None), "orderId", None)
        setattr(pos, "exit_sent", True)
        setattr(pos, "active", False)
        record_lifecycle_fn(
            recorder,
            "MANUAL_FLATTEN_SENT",
            symbol,
            action=action,
            quantity=qty,
            order_id=order_id,
            price=None,
            reason="control_api_queue_main_loop",
            entry_price=getattr(pos, "entry_price", None),
            peak_price=getattr(pos, "peak_price", None),
            raw_json={**cmd, "order_id": order_id},
        )
        if persist_managed_positions_fn is not None:
            try:
                persist_managed_positions_fn(recorder, managed_positions)
            except Exception:
                pass

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
            _json_response(self, 200, {"ok": True, "active_positions": len(active), "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False)), "pending_commands": pending})
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = _read_json_body(self)

        if parsed.path == "/pause_entries":
            self.ctx.runtime_state["entries_blocked"] = True
            _json_response(self, 200, {"ok": True, "entries_blocked": True})
            return

        if parsed.path == "/resume_entries":
            self.ctx.runtime_state["entries_blocked"] = False
            _json_response(self, 200, {"ok": True, "entries_blocked": False})
            return

        if parsed.path == "/flatten_symbol":
            symbol = _extract_symbol(parsed, body)
            if not symbol:
                _json_response(self, 400, {"ok": False, "error": "missing_symbol"})
                return
            payload = _flatten_request(self.ctx, symbol, _extract_dry_run(parsed, body))
            _json_response(self, 200 if payload.get("ok") else 400, payload)
            return

        if parsed.path == "/flatten_all_positions":
            dry_run = _extract_dry_run(parsed, body)
            symbols = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            results = [_flatten_request(self.ctx, s, dry_run) for s in symbols]
            _json_response(self, 200, {"ok": True, "dry_run": dry_run, "count": len(results), "results": results})
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
    print(f"CONTROL_API_STARTED host={host} port={port}", flush=True)
    return server
