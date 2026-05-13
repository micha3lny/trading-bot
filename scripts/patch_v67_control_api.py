from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


CONTROL_API_CODE = r'''
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
        try:
            if parsed.path == "/pause_entries":
                ctx["entries_paused"] = True
                record_lifecycle(ctx["recorder"], "CONTROL_PAUSE_ENTRIES", "__ALL__", action="CONTROL", reason="manual_control_api")
                self._json(200, {"ok": True, "entries_paused": True})
                return
            if parsed.path == "/resume_entries":
                ctx["entries_paused"] = False
                record_lifecycle(ctx["recorder"], "CONTROL_RESUME_ENTRIES", "__ALL__", action="CONTROL", reason="manual_control_api")
                self._json(200, {"ok": True, "entries_paused": False})
                return
            if parsed.path == "/flatten_all_positions":
                sent = control_flatten_positions(ctx, symbol_filter=None, reason="manual_control_api_flatten_all")
                self._json(200, {"ok": True, "sent": sent})
                return
            if parsed.path == "/flatten_symbol":
                symbol = str((query.get("symbol") or [""])[0]).upper().strip()
                if not symbol:
                    self._json(400, {"ok": False, "error": "missing_symbol"})
                    return
                sent = control_flatten_positions(ctx, symbol_filter=symbol, reason="manual_control_api_flatten_symbol")
                self._json(200, {"ok": True, "symbol": symbol, "sent": sent})
                return
            self._json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            print(f"{now_utc()} CONTROL_API_ERROR path={parsed.path} error={exc!r}", flush=True)
            self._json(500, {"ok": False, "error": repr(exc)})


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

        original_contract = getattr(item, "contract", None)
        contract = contract_by_symbol.get(symbol)
        if contract is None:
            currency = getattr(original_contract, "currency", None) or "USD"
            primary = getattr(original_contract, "primaryExchange", None) or ""
            contract = Stock(symbol, "SMART", currency, primaryExchange=primary)
            try:
                ib.qualifyContracts(contract)
            except Exception as exc:
                print(f"{now_utc()} CONTROL_FLATTEN_QUALIFY_ERROR symbol={symbol} error={exc!r}", flush=True)
                contract = original_contract
        if contract is None:
            print(f"{now_utc()} CONTROL_FLATTEN_SKIP_NO_CONTRACT symbol={symbol} qty={qty}", flush=True)
            continue

        price = safe_float((latest_snapshots.get(symbol) or {}).get("price")) or safe_float(getattr(item, "marketPrice", None))
        avg_cost = safe_float(getattr(item, "averageCost", None)) or price
        managed = managed_positions.get(symbol)
        peak = getattr(managed, "peak_price", None) if managed is not None else None
        peak = peak or price or avg_cost

        order = MarketOrder(action, quantity)
        order.tif = "DAY"
        order.outsideRth = False
        trade = ib.placeOrder(contract, order)
        pnl_pct = ((price / avg_cost - 1.0) * 100.0) if price and avg_cost and avg_cost > 0 else None

        record_lifecycle(
            recorder,
            "SELL_ORDER_SENT" if action == "SELL" else "BUY_TO_COVER_SENT",
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
    return sent


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
    print(f"{now_utc()} control_api_started host={host} port={port}", flush=True)
    return ctx

'''


def main() -> None:
    txt = P.read_text()

    # Imports
    if 'from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer' not in txt:
        txt = txt.replace('import time\n', 'import time\nimport threading\nfrom http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\nfrom urllib.parse import parse_qs, urlparse\n', 1)

    # Parser args
    if '--control-api-port' not in txt:
        marker = '    parser.add_argument("--client-id"'
        idx = txt.find(marker)
        if idx == -1:
            marker = '    parser.add_argument("--port"'
            idx = txt.find(marker)
        if idx == -1:
            raise SystemExit('control parser marker not found')
        insert_at = txt.find('\n', idx) + 1
        txt = txt[:insert_at] + (
            '    parser.add_argument("--control-api-enabled", action=argparse.BooleanOptionalAction, default=True)\n'
            '    parser.add_argument("--control-api-host", default="127.0.0.1")\n'
            '    parser.add_argument("--control-api-port", type=int, default=8767)\n'
        ) + txt[insert_at:]

    # Helper functions
    if 'def start_control_api(' not in txt:
        txt = replace_once(txt, '\ndef is_eod_flatten_time(', CONTROL_API_CODE + '\ndef is_eod_flatten_time(', 'insert control API code')

    # Start API after managed_positions/latest_snapshots exist. Try several markers.
    if 'control_ctx = start_control_api(' not in txt:
        markers = [
            '    traded_symbols_today = load_traded_symbols_today(recorder)',
            '    existing_fills = load_existing_fill_keys(recorder)',
            '    print(f"{now_utc()} traded_symbols_today_loaded=',
        ]
        for marker in markers:
            if marker in txt:
                insertion = '''    control_ctx = start_control_api(
        ib=ib,
        recorder=recorder,
        contract_by_symbol=contract_by_symbol,
        managed_positions=managed_positions,
        latest_snapshots=latest_snapshots,
        args=args,
    )
'''
                txt = txt.replace(marker, insertion + marker, 1)
                break
        else:
            raise SystemExit('start_control_api insertion marker not found')

    # Gate entries if paused.
    if 'CONTROL_PAUSED_SKIP_BUY' not in txt:
        candidates = ['if features.get("ready"):', 'if features["ready"]:', 'if features.get("ready", False):']
        for cand in candidates:
            if cand in txt:
                replacement = cand + '''
                    if control_ctx.get("entries_paused"):
                        record_lifecycle(
                            recorder,
                            "CONTROL_PAUSED_SKIP_BUY",
                            symbol,
                            action="SKIP_BUY",
                            price=features.get("entry_price"),
                            reason="entries_paused_by_control_api",
                            raw_json=features,
                        )
                        entries_blocked += 1
                        continue'''
                txt = txt.replace(cand, replacement, 1)
                break
        else:
            print('warning: buy ready marker not found; pause_entries endpoint will not block buys')

    # Heartbeat flag if possible.
    if 'control_paused={1 if control_ctx.get("entries_paused") else 0}' not in txt:
        txt = txt.replace(
            'entries_blocked={entries_blocked}',
            'entries_blocked={entries_blocked} control_paused={1 if control_ctx.get("entries_paused") else 0}',
        )

    P.write_text(txt)
    print('patched v67 local control API')


if __name__ == '__main__':
    main()
