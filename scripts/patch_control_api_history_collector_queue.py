from __future__ import annotations

from pathlib import Path

P = Path("src/live_trading/control/control_api.py")


def main() -> None:
    txt = P.read_text()
    original = txt
    backup = P.with_suffix(".before_history_collector_queue_patch.py")
    backup.write_text(original)

    if "import subprocess" not in txt:
        txt = txt.replace("import json\n", "import json\nimport subprocess\nimport sys\n", 1)
    elif "import sys" not in txt:
        txt = txt.replace("import subprocess\n", "import subprocess\nimport sys\n", 1)

    if "def _utc_minutes(" not in txt:
        marker = "def _make_contract(pos: Any, symbol: str) -> Any:\n"
        block = '''def _utc_minutes(value: str) -> int:
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
    market_open = str(runtime_state.get("market_open_utc", "13:30"))
    market_close = str(runtime_state.get("market_close_utc", "20:00"))
    if _in_utc_window(market_open, market_close):
        return False, "market_session_active"
    start = str(runtime_state.get("history_collector_start_utc", "20:15"))
    end = str(runtime_state.get("history_collector_end_utc", "23:00"))
    if not _in_utc_window(start, end):
        return False, "outside_collector_window"
    return True, "allowed"


'''
        txt = txt.replace(marker, block + marker, 1)

    if "def _ensure_history_queue(" not in txt:
        marker = "def _ensure_queue(ctx: ControlApiContext) -> list[JsonDict]:\n"
        block = '''def _ensure_history_queue(ctx: ControlApiContext) -> list[JsonDict]:
    queue = ctx.runtime_state.setdefault("history_collector_commands", [])
    if not isinstance(queue, list):
        queue = []
        ctx.runtime_state["history_collector_commands"] = queue
    return queue


'''
        txt = txt.replace(marker, block + marker, 1)

    if "def _queue_history_collector(" not in txt:
        marker = "def process_control_api_commands(\n"
        block = '''def _queue_history_collector(ctx: ControlApiContext, body: JsonDict) -> JsonDict:
    allowed, reason = _collector_allowed(ctx.runtime_state)
    if not allowed:
        _log("HISTORY_COLLECTOR_REJECTED", reason=reason)
        return {"ok": False, "status": "rejected", "reason": reason}

    command_id = uuid.uuid4().hex
    today = datetime.now(timezone.utc).date().isoformat()
    cmd = {
        "id": command_id,
        "type": "history_collector",
        "start_date": str(body.get("start_date") or ctx.runtime_state.get("history_collector_start_date") or "2026-01-01"),
        "end_date": str(body.get("end_date") or today),
        "session_type": str(body.get("session_type") or ctx.runtime_state.get("history_collector_session_type") or "RTH").upper(),
        "max_tasks": int(body.get("max_tasks") or ctx.runtime_state.get("history_collector_max_tasks") or 500),
        "limit_symbols": int(body.get("limit_symbols") or ctx.runtime_state.get("history_collector_limit_symbols") or 0),
        "client_id": int(body.get("client_id") or ctx.runtime_state.get("history_collector_client_id") or 168),
    }
    queue = _ensure_history_queue(ctx)
    queue.append(cmd)
    _log("HISTORY_COLLECTOR_QUEUED", command_id=command_id, start=cmd["start_date"], end=cmd["end_date"], session=cmd["session_type"], pending=len(queue))
    return {"ok": True, "status": "queued", "command_id": command_id, "command": cmd}


def process_history_collector_commands(*, runtime_state: dict[str, Any], max_commands: int = 1) -> int:
    queue = runtime_state.setdefault("history_collector_commands", [])
    if not queue:
        return 0

    allowed, reason = _collector_allowed(runtime_state)
    if not allowed:
        _log("HISTORY_COLLECTOR_DEFERRED", reason=reason, pending=len(queue))
        return 0

    processed = 0
    while queue and processed < max_commands:
        cmd = queue.pop(0)
        processed += 1
        command_id = cmd.get("id")
        run_key = f"{cmd.get('end_date')}_{cmd.get('session_type')}_{command_id}"
        args = [
            sys.executable,
            "-m",
            "src.live_trading.data.v68_universe_1m_parquet_collector",
            "--start-date", str(cmd.get("start_date") or "2026-01-01"),
            "--end-date", str(cmd.get("end_date") or datetime.now(timezone.utc).date().isoformat()),
            "--session-type", str(cmd.get("session_type") or "RTH"),
            "--client-id", str(int(cmd.get("client_id") or 168)),
            "--max-tasks", str(int(cmd.get("max_tasks") or 500)),
            "--allow-outside-window",
        ]
        limit = int(cmd.get("limit_symbols") or 0)
        if limit > 0:
            args.extend(["--limit-symbols", str(limit)])
        _log("HISTORY_COLLECTOR_START", command_id=command_id, cmd=" ".join(args))
        try:
            result = subprocess.run(args, check=False)
            runtime_state["history_collector_last_run_key"] = run_key if result.returncode == 0 else runtime_state.get("history_collector_last_run_key")
            _log("HISTORY_COLLECTOR_DONE", command_id=command_id, returncode=result.returncode, remaining=len(queue))
        except Exception as exc:
            _log("HISTORY_COLLECTOR_FAILED", command_id=command_id, error=repr(exc), remaining=len(queue))
    return processed


'''
        txt = txt.replace(marker, block + marker, 1)

    # Health should show history collector queue.
    txt = txt.replace(
        'pending = len(self.ctx.runtime_state.get("control_api_commands", []) or [])\n            _json_response(self, 200, {"ok": True, "active_positions": len(active), "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False)), "pending_commands": pending})',
        'pending = len(self.ctx.runtime_state.get("control_api_commands", []) or [])\n            pending_history = len(self.ctx.runtime_state.get("history_collector_commands", []) or [])\n            _json_response(self, 200, {"ok": True, "active_positions": len(active), "entries_blocked": bool(self.ctx.runtime_state.get("entries_blocked", False)), "pending_commands": pending, "pending_history_collector_commands": pending_history})',
        1,
    )

    if 'parsed.path == "/run_history_collector"' not in txt:
        marker = '''        if parsed.path == "/flatten_all_positions":
            dry_run = _extract_dry_run(parsed, body)
            symbols = [s for s, p in self.ctx.managed_positions.items() if bool(getattr(p, "active", False)) and not bool(getattr(p, "exit_sent", False))]
            results = [_flatten_request(self.ctx, s, dry_run) for s in symbols]
            _json_response(self, 200, {"ok": True, "dry_run": dry_run, "count": len(results), "results": results})
            return
'''
        insert = marker + '''
        if parsed.path == "/run_history_collector":
            payload = _queue_history_collector(self.ctx, body)
            _json_response(self, 200 if payload.get("ok") else 400, payload)
            return
'''
        if marker not in txt:
            raise SystemExit("flatten_all_positions block marker not found")
        txt = txt.replace(marker, insert, 1)

    if 'runtime_state.setdefault("history_collector_commands", [])' not in txt:
        txt = txt.replace('runtime_state.setdefault("control_api_commands", [])', 'runtime_state.setdefault("control_api_commands", [])\n    runtime_state.setdefault("history_collector_commands", [])', 1)

    if txt == original:
        print("no changes needed")
        return
    P.write_text(txt)
    print(f"patched control API history collector queue; backup={backup}")


if __name__ == "__main__":
    main()
