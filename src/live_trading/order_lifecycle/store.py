from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from src.live_trading.order_lifecycle.models import ExecutionRecord, LifecycleEvent


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class JsonlLifecycleStore:
    """Append-only lifecycle event store.

    This is intentionally small and boring for stage 1. It is not the final
    reconciliation database; it gives us durable formal events without changing
    current trading behavior.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: LifecycleEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=_json_default, sort_keys=True))
            f.write("\n")

    def append_raw(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {"event_type": event_type, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default, sort_keys=True))
            f.write("\n")

    def load_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def execution_ids(self) -> set[str]:
        ids: set[str] = set()
        for row in self.load_events():
            execution_id = str(row.get("execution_id") or "").strip()
            if execution_id:
                ids.add(execution_id)
        return ids

    def append_execution_once(self, execution: ExecutionRecord, event: LifecycleEvent) -> bool:
        if execution.execution_id in self.execution_ids():
            return False
        self.append_event(event)
        return True

    def extend(self, events: Iterable[LifecycleEvent]) -> int:
        count = 0
        for event in events:
            self.append_event(event)
            count += 1
        return count

