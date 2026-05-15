from __future__ import annotations

from pathlib import Path

P = Path("src/live_trading/control/control_api.py")


def main() -> None:
    txt = P.read_text()
    original = txt
    backup = P.with_suffix(".before_stdout_logs_patch.py")
    backup.write_text(original)

    if "from datetime import datetime, timezone" not in txt:
        txt = txt.replace("from dataclasses import dataclass\n", "from dataclasses import dataclass\nfrom datetime import datetime, timezone\n", 1)

    if "def _now_utc()" not in txt:
        marker = "PersistManagedPositionsFn = Callable[[Any, dict[str, Any]], None]\n"
        block = '''PersistManagedPositionsFn = Callable[[Any, dict[str, Any]], None]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **fields: Any) -> None:
    tail = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{_now_utc()} CONTROL_API_{event}" + (f" {tail}" if tail else ""), flush=True)
'''
        txt = txt.replace(marker, block, 1)

    replacements = [
        (
            'if pos is None:\n        return {"ok": False, "symbol": symbol, "status": "not_found_or_inactive"}',
            'if pos is None:\n        _log("FLATTEN_REJECTED", symbol=symbol, status="not_found_or_inactive")\n        return {"ok": False, "symbol": symbol, "status": "not_found_or_inactive"}',
        ),
        (
            'if bool(getattr(pos, "exit_sent", False)):\n        return {"ok": True, "symbol": symbol, "status": "already_exit_sent", "position": _position_payload(symbol, pos)}',
            'if bool(getattr(pos, "exit_sent", False)):\n        _log("FLATTEN_SKIPPED", symbol=symbol, status="already_exit_sent")\n        return {"ok": True, "symbol": symbol, "status": "already_exit_sent", "position": _position_payload(symbol, pos)}',
        ),
        (
            'if qty <= 0:\n        return {"ok": False, "symbol": symbol, "status": "bad_quantity", "quantity": qty_raw}',
            'if qty <= 0:\n        _log("FLATTEN_REJECTED", symbol=symbol, status="bad_quantity", quantity=qty_raw)\n        return {"ok": False, "symbol": symbol, "status": "bad_quantity", "quantity": qty_raw}',
        ),
        (
            'if dry_run:\n        ctx.record_lifecycle_fn(',
            'if dry_run:\n        _log("FLATTEN_DRY_RUN", symbol=symbol, action=action, quantity=qty)\n        ctx.record_lifecycle_fn(',
        ),
        (
            '_ensure_queue(ctx).append(command)\n    ctx.record_lifecycle_fn(',
            'queue = _ensure_queue(ctx)\n    queue.append(command)\n    _log("FLATTEN_QUEUED", symbol=symbol, action=action, quantity=qty, command_id=command_id, pending=len(queue))\n    ctx.record_lifecycle_fn(',
        ),
        (
            'if not queue:\n        return 0\n\n    processed = 0',
            'if not queue:\n        return 0\n\n    _log("QUEUE_PROCESS_START", pending=len(queue), max_commands=max_commands)\n    processed = 0',
        ),
        (
            'return processed\n\n\nclass _ControlHandler',
            '_log("QUEUE_PROCESS_DONE", processed=processed, remaining=len(queue))\n    return processed\n\n\nclass _ControlHandler',
        ),
        (
            'order_id = getattr(getattr(trade, "order", None), "orderId", None)\n        setattr(pos, "exit_sent", True)',
            'order_id = getattr(getattr(trade, "order", None), "orderId", None)\n        _log("FLATTEN_SENT", symbol=symbol, action=action, quantity=qty, order_id=order_id, command_id=cmd.get("id"))\n        setattr(pos, "exit_sent", True)',
        ),
        (
            'self.ctx.runtime_state["entries_blocked"] = True\n            _json_response',
            'self.ctx.runtime_state["entries_blocked"] = True\n            _log("PAUSE_ENTRIES")\n            _json_response',
        ),
        (
            'self.ctx.runtime_state["entries_blocked"] = False\n            _json_response',
            'self.ctx.runtime_state["entries_blocked"] = False\n            _log("RESUME_ENTRIES")\n            _json_response',
        ),
        (
            'print(f"CONTROL_API_STARTED host={host} port={port}", flush=True)',
            '_log("STARTED", host=host, port=port)',
        ),
    ]

    for old, new in replacements:
        if old in txt and new not in txt:
            txt = txt.replace(old, new, 1)

    if txt == original:
        print("no changes needed")
        return
    P.write_text(txt)
    print(f"patched control API stdout logs; backup={backup}")


if __name__ == "__main__":
    main()
