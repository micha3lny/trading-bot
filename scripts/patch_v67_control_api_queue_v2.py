from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def insert_after_imports(txt: str) -> str:
    if 'from queue import SimpleQueue' not in txt:
        txt = txt.replace('from collections import Counter\n', 'from collections import Counter\nfrom queue import SimpleQueue\n', 1)
    return txt


QUEUE_API_CODE = r'''

def enqueue_control_command(ctx: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    q = ctx.get("command_queue")
    if q is None:
        return {"ok": False, "error": "command_queue_not_initialized"}
    command = dict(command)
    command.setdefault("requested_at", now_utc())
    q.put(command)
    return {"ok": True, "queued": command}


class V67ControlHandler(BaseHTTPRequestHandler):
    control_ctx: dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject_nonlocal(self) -> bool:
        host = str(self.client_address[0])
        if host not in {"127.0.0.1", "::1", "localhost"}:
            self._json(403, {"ok": False, "error": "local_only"})
            return True
        return False

    def do_GET(self) -> None:
        if self._reject_nonlocal():
            return
        parsed = urlparse(self.path)
        ctx = self.control_ctx
        if parsed.path == "/health":
            managed_positions = ctx.get("managed_positions", {})
            self._json(200, {
                "ok": True,
                "strategy": STRATEGY_NAME,
                "entries_paused": bool(ctx.get("entries_paused", False)),
                "managed_open": len([p for p in managed_positions.values() if getattr(p, "active", False) and not getattr(p, "exit_sent", False)]),
                "queue_ready": ctx.get("command_queue") is not None,
                "time": now_utc(),
            })
            return
        self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        if self._reject_nonlocal():
            return
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        ctx = self.control_ctx

        if parsed.path == "/pause_entries":
            payload = enqueue_control_command(ctx, {"type": "pause_entries", "reason": "manual_control_api"})
            self._json(200 if payload.get("ok") else 500, payload)
            return

        if parsed.path == "/resume_entries":
            payload = enqueue_control_command(ctx, {"type": "resume_entries", "reason": "manual_control_api"})
            self._json(200 if payload.get("ok") else 500, payload)
            return

        if parsed.path in {"/flatten_all", "/flatten_all_positions"}:
            payload = enqueue_control_command(ctx, {"type": "flatten", "symbol": None, "reason": "manual_control_api_flatten_all"})
            self._json(200 if payload.get("ok") else 500, payload)
            return

        if parsed.path == "/flatten_symbol":
            symbol = str((query.get("symbol") or [""])[0]).upper().strip()
            if not symbol:
                self._json(400, {"ok": False, "error": "missing_symbol"})
                return
            payload = enqueue_control_command(ctx, {"type": "flatten", "symbol": symbol, "reason": "manual_control_api_flatten_symbol"})
            self._json(200 if payload.get("ok") else 500, payload)
            return

        self._json(404, {"ok": False, "error": "not_found"})


def start_control_api(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ctx = {
        "ib": ib,
        "recorder": recorder,
        "contract_by_symbol": contract_by_symbol,
        "managed_positions": managed_positions,
        "latest_snapshots": latest_snapshots,
        "entries_paused": False,
        "command_queue": SimpleQueue(),
    }
    if not getattr(args, "control_api_enabled", True):
        print(f"{now_utc()} control_api_disabled", flush=True)
        return ctx
    host = str(getattr(args, "control_api_host", "127.0.0.1"))
    port = int(getattr(args, "control_api_port", 8767))
    V67ControlHandler.control_ctx = ctx
    server = ThreadingHTTPServer((host, port), V67ControlHandler)
    thread = threading.Thread(target=server.serve_forever, name="v67-control-api", daemon=True)
    thread.start()
    ctx["server"] = server
    print(f"{now_utc()} control_api_started host={host} port={port} mode=queue", flush=True)
    return ctx


def control_flatten_positions(ctx: dict[str, Any], symbol_filter: str | None, reason: str) -> int:
    ib: IB = ctx["ib"]
    recorder: LiveDataRecorder = ctx["recorder"]
    contract_by_symbol: dict[str, Any] = ctx.get("contract_by_symbol", {})
    managed_positions: dict[str, ManagedPosition] = ctx.get("managed_positions", {})
    latest_snapshots: dict[str, dict[str, Any]] = ctx.get("latest_snapshots", {})
    sent = 0

    record_lifecycle(
        recorder,
        "CONTROL_FLATTEN_REQUEST",
        symbol_filter or "__ALL__",
        action="CONTROL",
        reason=reason,
    )

    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper().strip()
        qty = safe_float(getattr(item, "position", None))
        if not symbol or qty is None or abs(qty) <= 0:
            continue
        if symbol_filter and symbol != symbol_filter:
            continue

        action = "SELL" if qty > 0 else "BUY"
        quantity = int(abs(qty))
        if quantity <= 0:
            print(f"{now_utc()} CONTROL_FLATTEN_SKIP_FRACTIONAL symbol={symbol} qty={qty}", flush=True)
            continue

        source_contract = contract_by_symbol.get(symbol) or getattr(item, "contract", None)
        price = safe_float((latest_snapshots.get(symbol) or {}).get("price")) or safe_float(getattr(item, "marketPrice", None))
        avg_cost = safe_float(getattr(item, "averageCost", None)) or price
        managed = managed_positions.get(symbol)
        peak = getattr(managed, "peak_price", None) if managed is not None else None
        peak = peak or price or avg_cost

        if 'build_smart_stock_contract_for_close' in globals():
            close_contract = build_smart_stock_contract_for_close(ib, symbol, source_contract)
        else:
            close_contract = Stock(symbol, "SMART", getattr(source_contract, "currency", None) or "USD", primaryExchange=getattr(source_contract, "primaryExchange", None) or "")
            qualified = ib.qualifyContracts(close_contract)
            if qualified:
                close_contract = qualified[0]

        order = MarketOrder(action, quantity)
        order.tif = "DAY"
        order.outsideRth = False
        trade = ib.placeOrder(close_contract, order)
        pnl_pct = ((price / avg_cost - 1.0) * 100.0) if price and avg_cost and avg_cost > 0 else None

        event = "SELL_ORDER_SENT" if action == "SELL" else "BUY_TO_COVER_SENT"
        record_lifecycle(
            recorder,
            event,
            symbol,
            action=action,
            quantity=quantity,
            price=price,
            order_id=trade.order.orderId,
            reason=reason,
            entry_price=avg_cost,
            peak_price=peak,
            pnl_pct=pnl_pct,
            decision_last=price,
        )
        if managed is not None:
            managed.exit_sent = True
            managed.active = False
        sent += 1
        print(
            f"CONTROL FLATTEN SENT symbol={symbol} action={action} qty={quantity} "
            f"price={price if price else 0:.2f} reason={reason} orderId={trade.order.orderId}",
            flush=True,
        )
    persist_managed_positions(recorder, managed_positions)
    return sent


def process_control_commands(ctx: dict[str, Any]) -> int:
    q = ctx.get("command_queue")
    if q is None:
        return 0
    processed = 0
    while not q.empty():
        cmd = q.get()
        processed += 1
        typ = str(cmd.get("type", "")).strip()
        reason = str(cmd.get("reason") or "manual_control_api")
        try:
            if typ == "pause_entries":
                ctx["entries_paused"] = True
                record_lifecycle(ctx["recorder"], "CONTROL_PAUSE_ENTRIES", "__ALL__", action="CONTROL", reason=reason)
                print(f"{now_utc()} CONTROL_COMMAND_PROCESSED type=pause_entries", flush=True)
            elif typ == "resume_entries":
                ctx["entries_paused"] = False
                record_lifecycle(ctx["recorder"], "CONTROL_RESUME_ENTRIES", "__ALL__", action="CONTROL", reason=reason)
                print(f"{now_utc()} CONTROL_COMMAND_PROCESSED type=resume_entries", flush=True)
            elif typ == "flatten":
                symbol = cmd.get("symbol")
                symbol = str(symbol).upper().strip() if symbol else None
                sent = control_flatten_positions(ctx, symbol_filter=symbol, reason=reason)
                print(f"{now_utc()} CONTROL_COMMAND_PROCESSED type=flatten symbol={symbol or '__ALL__'} sent={sent}", flush=True)
            else:
                print(f"{now_utc()} CONTROL_COMMAND_UNKNOWN cmd={cmd}", flush=True)
        except Exception as exc:
            print(f"{now_utc()} CONTROL_COMMAND_ERROR cmd={cmd} error={exc!r}", flush=True)
            record_lifecycle(ctx["recorder"], "CONTROL_COMMAND_ERROR", str(cmd.get("symbol") or "__ALL__"), action="CONTROL", reason=repr(exc), raw_json=cmd)
    return processed

'''


def remove_old_control_api(txt: str) -> str:
    starts = ['\nclass V67ControlHandler(BaseHTTPRequestHandler):', '\nclass ControlHandler(BaseHTTPRequestHandler):']
    for start_marker in starts:
        start = txt.find(start_marker)
        if start == -1:
            continue
        end_candidates = [txt.find('\ndef is_eod_flatten_time(', start), txt.find('\ndef send_exit_order(', start), txt.find('\ndef manage_exits(', start)]
        end_candidates = [x for x in end_candidates if x != -1]
        if not end_candidates:
            raise SystemExit('old control api end marker not found')
        end = min(end_candidates)
        return txt[:start] + QUEUE_API_CODE + txt[end:]
    for marker in ['\ndef is_eod_flatten_time(', '\ndef send_exit_order(', '\ndef manage_exits(']:
        if marker in txt:
            return txt.replace(marker, QUEUE_API_CODE + marker, 1)
    raise SystemExit('control API insertion marker not found')


def main() -> None:
    txt = P.read_text()
    txt = insert_after_imports(txt)

    if 'ThreadingHTTPServer' not in txt:
        txt = txt.replace('import time\n', 'import time\nimport threading\nfrom http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\nfrom urllib.parse import parse_qs, urlparse\n', 1)
    if 'import threading' not in txt:
        txt = txt.replace('import time\n', 'import time\nimport threading\n', 1)
    if 'from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer' not in txt:
        txt = txt.replace('import threading\n', 'import threading\nfrom http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\nfrom urllib.parse import parse_qs, urlparse\n', 1)
    if 'from urllib.parse import parse_qs, urlparse' not in txt:
        txt = txt.replace('from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n', 'from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\nfrom urllib.parse import parse_qs, urlparse\n', 1)

    if '--control-api-port' not in txt:
        markers = ['    parser.add_argument("--client-id"', '    parser.add_argument("--port"', '    parser.add_argument("--host"']
        for marker in markers:
            if marker in txt:
                line_end = txt.find('\n', txt.find(marker)) + 1
                txt = txt[:line_end] + (
                    '    parser.add_argument("--control-api-enabled", action=argparse.BooleanOptionalAction, default=True)\n'
                    '    parser.add_argument("--control-api-host", default="127.0.0.1")\n'
                    '    parser.add_argument("--control-api-port", type=int, default=8767)\n'
                ) + txt[line_end:]
                break
        else:
            raise SystemExit('control parser marker not found')

    txt = remove_old_control_api(txt)

    if 'control_ctx = start_control_api(' not in txt:
        markers = ['        traded_symbols_today = load_traded_symbols_today(recorder)', '    traded_symbols_today = load_traded_symbols_today(recorder)', '        recorder.record_run_metadata({']
        insertion = '''        control_ctx = start_control_api(\n            ib=ib,\n            recorder=recorder,\n            contract_by_symbol=contract_by_symbol,\n            managed_positions=managed_positions,\n            latest_snapshots=latest_snapshots,\n            args=args,\n        )\n\n'''
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, insertion + marker, 1)
                break
        else:
            raise SystemExit('start_control_api insertion marker not found')

    if 'process_control_commands(control_ctx)' not in txt:
        markers = ['        exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args)', '            exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args)', '        entries_blocked = 0', '        ready_new = 0']
        for marker in markers:
            if marker in txt:
                indent = marker[: len(marker) - len(marker.lstrip())]
                txt = txt.replace(marker, f'{indent}process_control_commands(control_ctx)\n' + marker, 1)
                break
        else:
            raise SystemExit('loop command processing marker not found')

    if 'CONTROL_PAUSED_SKIP_BUY' not in txt:
        buy_markers = ['if features.get("ready"):', 'if features["ready"]:', 'if features.get("ready", False):']
        for cand in buy_markers:
            if cand in txt:
                txt = txt.replace(cand, cand + '''\n                    if control_ctx.get("entries_paused"):\n                        record_lifecycle(\n                            recorder,\n                            "CONTROL_PAUSED_SKIP_BUY",\n                            symbol,\n                            action="SKIP_BUY",\n                            price=features.get("entry_price"),\n                            reason="entries_paused_by_control_api",\n                            raw_json=features,\n                        )\n                        entries_blocked += 1\n                        continue''', 1)
                break

    P.write_text(txt)
    print('patched v67 queue-based control API: HTTP thread queues commands, main loop executes IBKR orders')


if __name__ == '__main__':
    main()
