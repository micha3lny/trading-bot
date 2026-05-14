from __future__ import annotations

import json
import threading
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


def _flatten_position(ctx: ControlApiContext, symbol: str, dry_run: bool) -> JsonDict:
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
        "status": "dry_run" if dry_run else "order_sent",
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

    contract = _make_contract(pos, symbol)
    try:
        qualified = ctx.ib.qualifyContracts(contract)
        if qualified:
            contract = qualified[0]
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "status": "qualify_failed", "error": repr(exc)}

    order = MarketOrder(action, qty)
    order.tif = "DAY"
    order.outsideRth = False

    try:
        trade = ctx.ib.placeOrder(contract, order)
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "status": "place_order_failed", "error": repr(exc)}

    order_id = getattr(getattr(trade, "order", None), "orderId", None)
    setattr(pos, "exit_sent", True)
    setattr(pos, "active", False)

    payload["order_id"] = order_id
    ctx.record_lifecycle_fn(
        ctx.recorder,
        "MANUAL_FLATTEN_SENT",
        symbol,
        action=action,
        quantity=qty,
        order_id=order_id,
        price=None,
        reason="control_api_flatten_symbol",
        entry_price=getattr(pos, "entry_price", None),
        peak_price=getattr(pos, "peak_price", None),
        raw_json=payload,
    )
    if ctx.persist_managed_positions_fn is not None:
        try:
            ctx.persist_managed_positions_fn(ctx.recorder, ctx.managed_positions)
        except Exception:
            pass
    return payload


class _ControlHandler(BaseHTTPRequestHandler):
    ctx: ControlApiContext

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            active = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            _json_response(self, 200, {"ok": True, "active_positions": len(active), "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False))})
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
            payload = _flatten_position(self.ctx, symbol, _extract_dry_run(parsed, body))
            _json_response(self, 200 if payload.get("ok") else 400, payload)
            return

        if parsed.path == "/flatten_all_positions":
            dry_run = _extract_dry_run(parsed, body)
            symbols = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            results = [_flatten_position(self.ctx, s, dry_run) for s in symbols]
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
