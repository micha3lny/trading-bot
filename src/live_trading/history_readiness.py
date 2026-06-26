from __future__ import annotations

from typing import Any


def pct(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((done / total) * 100.0, 2)


def canonical_history_readiness(
    *,
    session_date: str,
    session_type: str = "RTH",
    expected_symbols: int,
    complete_symbols: int,
    partial_symbols: int,
    no_data_symbols: int,
    missing_symbols: int,
    failed_symbols: int,
    parquet_files: int | None = None,
    min_effective_completion_pct: float = 100.0,
) -> dict[str, Any]:
    expected = max(0, int(expected_symbols or 0))
    complete = max(0, int(complete_symbols or 0))
    partial = max(0, int(partial_symbols or 0))
    no_data = max(0, int(no_data_symbols or 0))
    failed = max(0, int(failed_symbols or 0))
    missing = max(0, int(missing_symbols or 0))
    parquet_count = complete if parquet_files is None else max(0, int(parquet_files or 0))

    effective_symbols = complete + no_data
    effective_completion_pct = pct(effective_symbols, expected)
    parquet_completion_pct = pct(parquet_count, expected)
    ready = expected > 0 and missing == 0 and partial == 0 and failed == 0
    acceptable_partial = (
        expected > 0
        and missing == 0
        and failed == 0
        and effective_completion_pct >= float(min_effective_completion_pct)
    )
    if ready:
        readiness_status = "OK"
    elif effective_symbols or partial or failed:
        readiness_status = "PARTIAL"
    else:
        readiness_status = "NOT_READY"

    return {
        "date": session_date,
        "session_type": session_type.upper(),
        "history_session_date": session_date,
        "latest_completed_session": session_date,
        "expected_symbols": expected,
        "parquet_files": parquet_count,
        "status_done": effective_symbols,
        "complete_symbols": complete,
        "partial_symbols": partial,
        "no_data_symbols": no_data,
        "failed": failed,
        "missing": missing,
        "failed_symbols": failed,
        "missing_symbols": missing,
        "completed": effective_symbols,
        "terminal_symbols": effective_symbols,
        "effective_symbols": effective_symbols,
        "completion_pct": effective_completion_pct,
        "effective_completion_pct": effective_completion_pct,
        "parquet_completion_pct": parquet_completion_pct,
        "readiness_status": readiness_status,
        "status": "OK" if ready else ("PARTIAL" if effective_symbols or partial or failed else "MISSING"),
        "ready": ready,
        "acceptable_partial": acceptable_partial,
    }
