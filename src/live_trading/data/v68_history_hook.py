from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


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


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def maybe_run_history_collector(args: Any, state: dict[str, Any]) -> int:
    """Run the v68 Parquet history collector once per day inside a guarded time window.

    This hook is intentionally tiny and reusable across bot versions.
    It does not download candles itself. It only decides whether to launch the
    already-tested collector module as a subprocess.

    Safety rules:
    - disabled unless --history-collector-enabled is true
    - never runs during market window
    - only runs inside collector window
    - runs at most once per UTC date/session_type
    """
    if not bool(getattr(args, "history_collector_enabled", False)):
        return 0

    now = datetime.now(timezone.utc)

    market_open = str(getattr(args, "market_open_utc", "13:30"))
    market_close = str(getattr(args, "history_collector_market_close_utc", "20:00"))
    if _in_utc_window(market_open, market_close, now):
        return 0

    collector_start = str(getattr(args, "history_collector_start_utc", "20:15"))
    collector_end = str(getattr(args, "history_collector_end_utc", "23:00"))
    if not _in_utc_window(collector_start, collector_end, now):
        return 0

    session_type = str(getattr(args, "history_collector_session_type", "RTH")).upper()
    today = now.date().isoformat()
    key = f"{today}_{session_type}"

    if state.get("last_run_key") == key:
        return 0
    if state.get("running"):
        return 0

    cmd = [
        sys.executable,
        "-m",
        "src.live_trading.data.v68_universe_1m_parquet_collector",
        "--start-date",
        str(getattr(args, "history_collector_start_date", today) or today),
        "--end-date",
        today,
        "--session-type",
        session_type,
        "--client-id",
        str(int(getattr(args, "history_collector_client_id", 168))),
        "--max-tasks",
        str(int(getattr(args, "history_collector_max_tasks", 500))),
        "--allow-outside-window",
    ]

    limit_symbols = int(getattr(args, "history_collector_limit_symbols", 0) or 0)
    if limit_symbols > 0:
        cmd.extend(["--limit-symbols", str(limit_symbols)])

    state["running"] = True
    print(f"{_now_utc_iso()} HISTORY_COLLECTOR_HOOK_LAUNCH key={key} cmd={' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, check=False)
        print(f"{_now_utc_iso()} HISTORY_COLLECTOR_HOOK_EXIT key={key} returncode={result.returncode}", flush=True)
        if result.returncode == 0:
            state["last_run_key"] = key
        return int(result.returncode)
    except Exception as exc:
        print(f"{_now_utc_iso()} HISTORY_COLLECTOR_HOOK_ERROR key={key} error={exc!r}", flush=True)
        return 1
    finally:
        state["running"] = False
