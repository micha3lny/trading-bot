from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import resource
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.live_trading.analysis.common import (
    fnum,
    iso_ts,
    load_session_candles,
    normalize_symbol,
    parse_dt,
    pct,
    sqlite_connect_readonly,
    table_columns,
)
from src.live_trading.analysis.full_session_replay_v67 import (
    PreparedSessionCache,
    ReplayConfig,
    ReplayResult,
    _rows,
    effective_config_dict,
    profile_config,
    replay_session,
)
from src.live_trading.analysis.top100_analysis_common import (
    load_top100_source,
    session_dates,
    write_dataframe,
)
from src.live_trading.candidate_snapshot_telemetry import snapshot_chunk_paths
from src.live_trading.analysis.strategy_config_parity import (
    EffectiveSignalThresholds,
    add_threshold_cli,
    resolve_threshold_args,
    output_has_config_provenance,
)


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")
DEFAULT_TOP100_DIR = Path("data/universe")
DEFAULT_OUTPUT_DIR = Path("data/analysis")

READY_EVENTS = {"SIGNAL_READY", "ENTRY_SIGNAL"}
BUY_EVENTS = {
    "BUY_ORDER_SENT", "PAPER_BUY_SENT", "ENTRY_ORDER_DISPATCH_ATTEMPT",
    "ENTRY_ORDER_SUBMITTED", "ENTRY_ORDER_IBKR_SUBMITTED", "BUY_DECISION",
}
EVALUATION_EVENTS = {"SIGNAL_EVALUATED", "SIGNAL_REJECTED", *READY_EVENTS, *BUY_EVENTS}
SUPPORTED_REPLAY_FILTERS = {
    "first_5m_high_too_low",
    "first_15m_high_too_low",
    "or_range_too_low",
    "spread_too_wide",
    "price_too_low",
    "legacy_candle_high_breakout_not_met",
    "current_price_breakout_not_met",
}
NON_FILTER_READY_REASONS = {"", "ready", "live_safe_expansion_ready", "would_emit_signal_ready"}

LIGHT_REQUIRED_COLUMNS = {
    "session_date", "trading_session_date", "trade_session_date", "timestamp", "event_time", "recorded_at",
    "process_start_id", "scan_id", "ranking_generation", "symbol", "ready", "would_emit_signal_ready",
    "rejection_reason", "selection_rejected_reason", "signal_ready_reason", "top100_rank", "rank",
    "top100_score", "live_entry_score", "score", "current_price", "price", "last", "decision_price",
    "entries_blocked", "entries_blocked_reason", "blocking_reason", "already_open",
}
EVENT_SOURCE_PRIORITY = {
    "sqlite:runtime_events": 3,
    "recorder:trade_lifecycle.csv": 2,
    "recorder:order_lifecycle.jsonl": 2,
    "p1:light_snapshot": 1,
}

LIFETIME_COLUMNS = [
    "session_date", "symbol", "top100_rank", "top100_score",
    "first_evaluated_at", "first_rejected_at", "first_ready_at", "first_buy_at", "last_evaluated_at",
    "all_rejection_reasons", "initial_rejection_reason_set", "number_of_evaluations", "number_of_rejections",
    "price_at_first_evaluation", "price_at_first_rejection", "price_at_first_ready", "price_at_buy",
    "max_price_after_rejection", "max_pct_after_rejection", "time_to_max_after_rejection",
    "time_to_max_after_rejection_minutes",
    "max_price_after_30m", "max_pct_after_30m", "max_price_after_60m", "max_pct_after_60m",
    "max_price_to_close", "max_pct_to_close", "close_price", "close_pct_from_rejection",
    "mae_after_rejection_pct", "time_to_1pct_minutes", "time_to_2pct_minutes", "time_to_3pct_minutes",
    "time_to_5pct_minutes", "classification", "future_data_quality", "evidence_sources",
    "effective_min_first5", "effective_min_first15", "effective_min_or_range", "config_source",
]

COUNTERFACTUAL_COLUMNS = [
    "session_date", "filter", "rejected_candidates", "candidates_unblocked_if_removed",
    "later_peak_ge_1pct", "later_peak_ge_2pct", "later_peak_ge_3pct", "later_peak_ge_5pct",
    "later_peak_ge_10pct", "avg_future_peak_pct", "median_future_peak_pct", "p90_future_peak_pct",
    "avg_close_pct", "estimated_additional_entries", "future_data_candidates", "missing_future_data_candidates",
    "missed_opportunity_rate_pct", "counterfactual_basis",
    "effective_min_first5", "effective_min_first15", "effective_min_or_range", "config_source",
]

PORTFOLIO_COLUMNS = [
    "session_date", "variant", "removed_filter", "replay_supported", "causal_valid", "trade_count",
    "winners", "losers", "gross_pnl", "net_pnl", "win_rate", "avg_trade", "median_trade", "profit_factor", "max_drawdown",
    "incremental_trade_count", "incremental_net_pnl", "effective_config_json", "replay_skip_reason",
]


@dataclass
class PhaseCounter:
    phases: list[dict[str, Any]]


def _rss_mb() -> float | None:
    status = Path(f"/proc/{os.getpid()}/status")
    if status.exists():
        try:
            for line in status.read_text(errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0
    except Exception:
        return None


def _peak_rss_mb() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0
    except Exception:
        return None


@contextmanager
def evidence_timing(name: str, records: list[dict[str, Any]], **fields: Any):
    started = time.perf_counter()
    before = _rss_mb()
    try:
        yield fields
    finally:
        after = _rss_mb()
        record = {
            "subphase": name,
            "elapsed_seconds": time.perf_counter() - started,
            "rss_before_mb": before,
            "rss_after_mb": after,
            "rss_delta_mb": (after - before) if before is not None and after is not None else None,
            "peak_rss_mb": _peak_rss_mb(),
            **fields,
        }
        records.append(record)
        print(
            "CANDIDATE_LIFETIME_EVIDENCE_TIMING "
            + " ".join(f"{key}={value if value is not None else ''}" for key, value in record.items()),
            flush=True,
        )


@contextmanager
def phase(name: str, timings: PhaseCounter, **fields: Any):
    started = time.perf_counter()
    print("CANDIDATE_LIFETIME_PHASE_START " + " ".join([f"phase={name}", *[f"{k}={v}" for k, v in fields.items()]]), flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        record = {"phase": name, "elapsed_seconds": elapsed, **fields}
        timings.phases.append(record)
        print("CANDIDATE_LIFETIME_PHASE_DONE " + " ".join([f"phase={name}", f"elapsed_seconds={elapsed:.3f}", *[f"{k}={v}" for k, v in fields.items()]]), flush=True)


def _raw(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _value(row: dict[str, Any], raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return None


def _event_session_date(row: dict[str, Any], raw: dict[str, Any]) -> str:
    for key in ("session_date", "trading_session_date", "trade_session_date"):
        value = raw.get(key) or row.get(key)
        if value not in (None, ""):
            return str(value)[:10]
    timestamp = parse_dt(_value(row, raw, "timestamp", "event_time", "recorded_at", "time"))
    return timestamp.date().isoformat() if timestamp is not None else ""


def _canonical_event(row: dict[str, Any], source: str, requested_date: str) -> dict[str, Any] | None:
    raw = _raw(row.get("raw_json") or row.get("payload") or row.get("details"))
    event = str(_value(row, raw, "event_type", "event", "legacy_event", "type") or "").strip().upper()
    symbol = normalize_symbol(_value(row, raw, "symbol"))
    timestamp = parse_dt(_value(row, raw, "timestamp", "event_time", "recorded_at", "time"))
    if event not in EVALUATION_EVENTS or not symbol or timestamp is None:
        return None
    if _event_session_date(row, raw) != requested_date or timestamp.date().isoformat() != requested_date:
        return None
    outcome = str(_value(row, raw, "outcome", "status") or "").lower()
    reason = str(_value(row, raw, "reason", "rejection_reason", "signal_ready_reason") or "").strip()
    if event == "SIGNAL_EVALUATED" and outcome == "ready" and reason.lower() in NON_FILTER_READY_REASONS:
        reason = ""
    return {
        "session_date": requested_date,
        "symbol": symbol,
        "timestamp": timestamp,
        "event_type": event,
        "reason": reason,
        "outcome": outcome,
        "status": str(_value(row, raw, "status") or outcome),
        "scan_id": str(_value(row, raw, "scan_id", "ranking_generation") or ""),
        "top100_rank": fnum(_value(row, raw, "top100_rank", "rank")),
        "top100_score": fnum(_value(row, raw, "top100_score")),
        "live_entry_score": fnum(_value(row, raw, "live_entry_score", "score")),
        "price": fnum(_value(row, raw, "current_price", "price", "last", "decision_price")),
        "entries_blocked": int(bool(fnum(_value(row, raw, "entries_blocked"), 0))),
        "entries_blocked_reason": str(_value(row, raw, "entries_blocked_reason", "blocking_reason") or ""),
        "already_open": int(bool(fnum(_value(row, raw, "already_open"), 0))),
        "source": source,
        "raw_json": raw,
    }


def _event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["session_date"], event["symbol"], event["event_type"], event["timestamp"].isoformat(),
        event.get("scan_id") or "", event.get("reason") or "", event.get("outcome") or "",
    )


def load_sqlite_events(sqlite_path: Path, session_date: str) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    conn = sqlite_connect_readonly(sqlite_path)
    try:
        columns = table_columns(conn, "runtime_events")
        if not columns or "event_type" not in columns:
            return []
        selected = [column for column in (
            "event_time", "event_type", "session_date", "symbol", "strategy_name", "raw_json"
        ) if column in columns]
        end_date = (date.fromisoformat(session_date) + timedelta(days=1)).isoformat()
        if "session_date" in columns and "event_time" in columns:
            date_clause = "(session_date = ? OR (session_date IS NULL AND event_time >= ? AND event_time < ?))"
            params: list[Any] = [session_date, session_date, end_date]
        elif "session_date" in columns:
            date_clause = "session_date = ?"
            params = [session_date]
        elif "event_time" in columns:
            date_clause = "event_time >= ? AND event_time < ?"
            params = [session_date, end_date]
        else:
            return []
        sql = (
            f"SELECT {', '.join(selected)} FROM runtime_events WHERE {date_clause} "
            f"AND event_type IN ({','.join('?' for _ in EVALUATION_EVENTS)})"
            + (" ORDER BY event_time" if "event_time" in columns else "")
        )
        params.extend(sorted(EVALUATION_EVENTS))
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
    return [event for row in rows if (event := _canonical_event(row, "sqlite:runtime_events", session_date)) is not None]


def _stream_csv(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    def iterator():
        try:
            with path.open("r", newline="", errors="replace") as handle:
                yield from csv.DictReader(handle)
        except Exception:
            return
    return iterator()


def load_recorder_events(recorder_dir: Path, session_date: str) -> tuple[list[dict[str, Any]], int]:
    root = recorder_dir / session_date
    events: list[dict[str, Any]] = []
    rows_used = 0
    for row in _stream_csv(root / "trade_lifecycle.csv"):
        rows_used += 1
        event = _canonical_event(row, "recorder:trade_lifecycle.csv", session_date)
        if event is not None:
            events.append(event)
    path = root / "order_lifecycle.jsonl"
    if path.exists():
        try:
            with path.open("r", errors="replace") as handle:
                for line in handle:
                    rows_used += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    event = _canonical_event(row, "recorder:order_lifecycle.jsonl", session_date)
                    if event is not None:
                        events.append(event)
        except Exception:
            pass
    return events, rows_used


def _light_snapshot_key(row: dict[str, Any], key_columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(normalize_symbol(row.get(column)) if column == "symbol" else row.get(column) for column in key_columns)


def _light_event(row: dict[str, Any], session_date: str) -> dict[str, Any] | None:
    raw = dict(row)
    raw["event_type"] = "SIGNAL_EVALUATED"
    ready_value = fnum(row.get("ready"))
    if ready_value is None:
        ready_value = fnum(row.get("would_emit_signal_ready"), 0)
    ready = bool(ready_value)
    raw["outcome"] = "ready" if ready else "rejected"
    raw["reason"] = "" if ready else str(row.get("rejection_reason") or row.get("selection_rejected_reason") or "")
    return _canonical_event(raw, "p1:light_snapshot", session_date)


def load_light_events(
    recorder_dir: Path,
    session_date: str,
    *,
    higher_priority_keys: set[tuple[Any, ...]] | None = None,
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    paths = snapshot_chunk_paths(recorder_dir, session_date, "light")
    if not paths:
        return [], 0, {"chunks": 0, "rows_session_scoped": 0, "higher_priority_duplicates_skipped": 0}

    schema_names: set[str] = set()
    for path in paths:
        try:
            schema_names.update(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
    key_columns = tuple(
        column for column in ("session_date", "process_start_id", "scan_id", "symbol")
        if column in schema_names
    )
    selected_by_snapshot: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows_scanned = 0
    rows_session_scoped = 0
    for path in paths:
        try:
            parquet = pq.ParquetFile(path)
            columns = sorted(LIGHT_REQUIRED_COLUMNS.intersection(parquet.schema_arrow.names))
            if not columns:
                continue
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                py_rows = batch.to_pylist()
                timestamp_column = next(
                    (name for name in ("timestamp", "event_time", "recorded_at") if name in batch.schema.names),
                    None,
                )
                parsed_timestamps = (
                    pd.to_datetime(batch.column(batch.schema.get_field_index(timestamp_column)).to_pandas(), errors="coerce", utc=True).tolist()
                    if timestamp_column else [None] * len(py_rows)
                )
                for row, parsed_timestamp in zip(py_rows, parsed_timestamps):
                    rows_scanned += 1
                    if timestamp_column and parsed_timestamp is not None and not pd.isna(parsed_timestamp):
                        row[timestamp_column] = parsed_timestamp
                    raw_session = str(
                        row.get("session_date") or row.get("trading_session_date")
                        or row.get("trade_session_date") or ""
                    )[:10]
                    if raw_session and raw_session != session_date:
                        continue
                    event = _light_event(row, session_date)
                    if event is None:
                        continue
                    rows_session_scoped += 1
                    key = _light_snapshot_key(row, key_columns) if key_columns else (rows_scanned,)
                    # Matches read_snapshot_chunks(...).drop_duplicates(..., keep="last").
                    selected_by_snapshot[key] = event
        except Exception:
            continue

    higher = higher_priority_keys or set()
    events: list[dict[str, Any]] = []
    skipped = 0
    for event in selected_by_snapshot.values():
        if _event_key(event) in higher:
            skipped += 1
            continue
        events.append(event)
    events.sort(key=lambda item: (item["timestamp"], item["symbol"], item["event_type"]))
    stats = {
        "chunks": len(paths),
        "rows_session_scoped": rows_session_scoped,
        "higher_priority_duplicates_skipped": skipped,
    }
    return events, rows_scanned, stats


def enrich_event_ranking(events: list[dict[str, Any]], top100: pd.DataFrame) -> None:
    if top100.empty or "symbol" not in top100.columns:
        return
    indexed = top100.drop_duplicates("symbol").set_index("symbol")
    for event in events:
        symbol = event["symbol"]
        if symbol not in indexed.index:
            continue
        source = indexed.loc[symbol]
        if event.get("top100_rank") is None:
            event["top100_rank"] = fnum(source.get("top100_rank", source.get("rank")))
        if event.get("top100_score") is None:
            event["top100_score"] = fnum(source.get("top100_score", source.get("score")))


def deduplicate_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = 0
    for event in sorted(events, key=lambda item: (item["timestamp"], -EVENT_SOURCE_PRIORITY.get(item["source"], 0))):
        key = _event_key(event)
        if key in selected:
            duplicates += 1
            if EVENT_SOURCE_PRIORITY.get(event["source"], 0) > EVENT_SOURCE_PRIORITY.get(selected[key]["source"], 0):
                selected[key] = event
        else:
            selected[key] = event
    return sorted(selected.values(), key=lambda item: (item["timestamp"], item["symbol"], item["event_type"])), duplicates


def deduplicate_event_sources(*sources: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = 0
    for source in sources:
        for event in source:
            key = _event_key(event)
            previous = selected.get(key)
            if previous is None:
                selected[key] = event
                continue
            duplicates += 1
            if EVENT_SOURCE_PRIORITY.get(event["source"], 0) > EVENT_SOURCE_PRIORITY.get(previous["source"], 0):
                selected[key] = event
    return sorted(selected.values(), key=lambda item: (item["timestamp"], item["symbol"], item["event_type"])), duplicates


def rejection_reasons(value: Any) -> tuple[str, ...]:
    return tuple(sorted({part.strip().lower() for part in re.split(r"[;,|]", str(value or "")) if part.strip() and part.strip().lower() not in NON_FILTER_READY_REASONS}))


def event_is_ready(event: dict[str, Any]) -> bool:
    return event["event_type"] in READY_EVENTS or (
        event["event_type"] == "SIGNAL_EVALUATED" and event.get("outcome") == "ready"
    )


def event_is_buy(event: dict[str, Any]) -> bool:
    if event["event_type"] not in BUY_EVENTS:
        return False
    if event["event_type"] == "BUY_DECISION":
        return str(event.get("status") or event.get("outcome") or "").lower() in {"submitted", "sent", "accepted", "filled", "buy", "entered"}
    return event["event_type"] != "ENTRY_ORDER_DISPATCH_ATTEMPT"


def _unique_rejections(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        if not (
            event["event_type"] == "SIGNAL_REJECTED"
            or (event["event_type"] == "SIGNAL_EVALUATED" and not event_is_ready(event))
        ):
            continue
        decision_id = event.get("scan_id") or event["timestamp"].isoformat()
        key = (str(decision_id), event["timestamp"].isoformat(), ";".join(rejection_reasons(event.get("reason"))))
        previous = selected.get(key)
        if previous is None or event["event_type"] == "SIGNAL_REJECTED":
            selected[key] = event
    return sorted(selected.values(), key=lambda item: (item["timestamp"], item["event_type"]))


def _future_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.ceil("min") if timestamp.second or timestamp.microsecond else timestamp


def future_outcome(candles: pd.DataFrame, rejected_at: pd.Timestamp, base_price: float | None) -> dict[str, Any]:
    empty = {
        "max_price_after_rejection": None, "max_pct_after_rejection": None,
        "time_to_max_after_rejection": None, "time_to_max_after_rejection_minutes": None,
        "max_price_after_30m": None, "max_pct_after_30m": None,
        "max_price_after_60m": None, "max_pct_after_60m": None, "max_price_to_close": None,
        "max_pct_to_close": None, "close_price": None, "close_pct_from_rejection": None,
        "mae_after_rejection_pct": None, "time_to_1pct_minutes": None, "time_to_2pct_minutes": None,
        "time_to_3pct_minutes": None, "time_to_5pct_minutes": None, "future_data_quality": "missing",
        "history_rows_used": 0,
    }
    if candles.empty or "timestamp" not in candles:
        return empty
    rows = candles.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp")
    future = rows[rows["timestamp"] >= _future_start(rejected_at)].copy()
    if future.empty:
        return empty
    high = pd.to_numeric(future.get("high"), errors="coerce")
    low = pd.to_numeric(future.get("low"), errors="coerce")
    close = pd.to_numeric(future.get("close"), errors="coerce")
    price = base_price or fnum(future.iloc[0].get("open")) or fnum(close.iloc[0])
    if price is None or price <= 0 or high.notna().sum() == 0:
        return {**empty, "history_rows_used": len(future), "future_data_quality": "invalid_price"}
    max_index = high.idxmax()
    maximum = float(high.loc[max_index])
    minimum = float(low.min()) if low.notna().any() else None
    final_close = float(close.dropna().iloc[-1]) if close.notna().any() else None
    time_to_max = (future.loc[max_index, "timestamp"] - rejected_at).total_seconds() / 60.0
    result = dict(empty)
    result.update({
        "max_price_after_rejection": maximum,
        "max_pct_after_rejection": pct(maximum, price),
        "time_to_max_after_rejection": time_to_max,
        "time_to_max_after_rejection_minutes": time_to_max,
        "max_price_to_close": maximum,
        "max_pct_to_close": pct(maximum, price),
        "close_price": final_close,
        "close_pct_from_rejection": pct(final_close, price),
        "mae_after_rejection_pct": pct(minimum, price),
        "future_data_quality": "complete" if len(future) >= 30 else "partial",
        "history_rows_used": len(future),
    })
    for minutes in (30, 60):
        subset = future[future["timestamp"] < rejected_at + pd.Timedelta(minutes=minutes)]
        peak = fnum(pd.to_numeric(subset.get("high"), errors="coerce").max()) if not subset.empty else None
        result[f"max_price_after_{minutes}m"] = peak
        result[f"max_pct_after_{minutes}m"] = pct(peak, price)
    for threshold in (1, 2, 3, 5):
        crossed = future[high >= price * (1.0 + threshold / 100.0)]
        result[f"time_to_{threshold}pct_minutes"] = (
            (crossed.iloc[0]["timestamp"] - rejected_at).total_seconds() / 60.0 if not crossed.empty else None
        )
    return result


def build_lifetimes(
    session_date: str,
    events: list[dict[str, Any]],
    candles_by_symbol: dict[str, pd.DataFrame],
    *,
    missed_threshold_pct: float,
    effective: EffectiveSignalThresholds | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str, float | None], dict[str, Any]]]:
    effective = effective or EffectiveSignalThresholds(4.0, 6.5, 5.0, "programmatic_historical_strict_default")
    rows: list[dict[str, Any]] = []
    outcome_cache: dict[tuple[str, str, float | None], dict[str, Any]] = {}
    for symbol, symbol_events in sorted(_group_events(events).items()):
        evaluations = [event for event in symbol_events if event["event_type"] == "SIGNAL_EVALUATED"]
        rejections = _unique_rejections(symbol_events)
        ready = [event for event in symbol_events if event_is_ready(event)]
        buys = [event for event in symbol_events if event_is_buy(event)]
        first_evaluation = evaluations[0] if evaluations else (symbol_events[0] if symbol_events else None)
        first_rejection = rejections[0] if rejections else None
        first_ready = ready[0] if ready else None
        first_buy = buys[0] if buys else None
        all_reasons = sorted({reason for event in rejections for reason in rejection_reasons(event.get("reason"))})
        initial_reasons = rejection_reasons(first_rejection.get("reason")) if first_rejection else ()
        outcome = future_outcome(
            candles_by_symbol.get(symbol, pd.DataFrame()),
            first_rejection["timestamp"],
            first_rejection.get("price"),
        ) if first_rejection else {"future_data_quality": "not_rejected", "history_rows_used": 0}
        if first_rejection:
            outcome_cache[(symbol, first_rejection["timestamp"].isoformat(), first_rejection.get("price"))] = outcome
        if first_buy:
            classification = "eventually_bought"
        elif not first_rejection or outcome.get("future_data_quality") in {"missing", "invalid_price"}:
            classification = "insufficient_data"
        elif fnum(outcome.get("max_pct_after_rejection"), -math.inf) >= missed_threshold_pct:
            classification = "missed_opportunity"
        else:
            classification = "true_negative"
        rank = next((event.get("top100_rank") for event in symbol_events if event.get("top100_rank") is not None), None)
        score = next((event.get("top100_score") for event in symbol_events if event.get("top100_score") is not None), None)
        row = {
            "session_date": session_date,
            "symbol": symbol,
            "top100_rank": rank,
            "top100_score": score,
            "first_evaluated_at": iso_ts(first_evaluation["timestamp"]) if first_evaluation else "",
            "first_rejected_at": iso_ts(first_rejection["timestamp"]) if first_rejection else "",
            "first_ready_at": iso_ts(first_ready["timestamp"]) if first_ready else "",
            "first_buy_at": iso_ts(first_buy["timestamp"]) if first_buy else "",
            "last_evaluated_at": iso_ts(evaluations[-1]["timestamp"]) if evaluations else "",
            "all_rejection_reasons": ";".join(all_reasons),
            "initial_rejection_reason_set": ";".join(initial_reasons),
            "number_of_evaluations": len(evaluations),
            "number_of_rejections": len(rejections),
            "price_at_first_evaluation": first_evaluation.get("price") if first_evaluation else None,
            "price_at_first_rejection": first_rejection.get("price") if first_rejection else None,
            "price_at_first_ready": first_ready.get("price") if first_ready else None,
            "price_at_buy": first_buy.get("price") if first_buy else None,
            "classification": classification,
            "evidence_sources": ";".join(sorted({event["source"] for event in symbol_events})),
            **effective.output_fields(),
            **{key: value for key, value in outcome.items() if key != "history_rows_used"},
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=LIFETIME_COLUMNS), outcome_cache


def _group_events(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["symbol"]].append(event)
    for values in grouped.values():
        values.sort(key=lambda item: (item["timestamp"], item["event_type"]))
    return grouped


def _event_unblocked_by_only(event: dict[str, Any], filter_name: str) -> bool:
    reasons = set(rejection_reasons(event.get("reason")))
    if reasons != {filter_name}:
        return False
    if event.get("entries_blocked") or event.get("entries_blocked_reason") or event.get("already_open"):
        return False
    return True


def counterfactual_filter_rows(
    session_date: str,
    events: list[dict[str, Any]],
    candles_by_symbol: dict[str, pd.DataFrame],
    *,
    missed_threshold_pct: float,
    effective: EffectiveSignalThresholds | None = None,
    outcome_cache: dict[tuple[str, str, float | None], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    effective = effective or EffectiveSignalThresholds(4.0, 6.5, 5.0, "programmatic_historical_strict_default")
    outcome_cache = outcome_cache if outcome_cache is not None else {}
    by_symbol = _group_events(events)
    discovered = sorted({reason for event in events for reason in rejection_reasons(event.get("reason"))})
    output: list[dict[str, Any]] = []
    for filter_name in discovered:
        rejected_symbols: set[str] = set()
        unblocked: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for symbol, symbol_events in by_symbol.items():
            matching = [event for event in symbol_events if filter_name in rejection_reasons(event.get("reason"))]
            if not matching:
                continue
            rejected_symbols.add(symbol)
            causal = next((event for event in matching if _event_unblocked_by_only(event, filter_name)), None)
            if causal is not None:
                cache_key = (symbol, causal["timestamp"].isoformat(), causal.get("price"))
                outcome = outcome_cache.get(cache_key)
                if outcome is None:
                    outcome = future_outcome(
                        candles_by_symbol.get(symbol, pd.DataFrame()), causal["timestamp"], causal.get("price")
                    )
                    outcome_cache[cache_key] = outcome
                unblocked[symbol] = (causal, outcome)
        peaks = pd.Series([item[1].get("max_pct_after_rejection") for item in unblocked.values()], dtype=float).dropna()
        closes = pd.Series([item[1].get("close_pct_from_rejection") for item in unblocked.values()], dtype=float).dropna()
        valid = [item for item in unblocked.values() if item[1].get("future_data_quality") not in {"missing", "invalid_price"}]
        row = {
            "session_date": session_date,
            "filter": filter_name,
            "rejected_candidates": len(rejected_symbols),
            "candidates_unblocked_if_removed": len(unblocked),
            **{f"later_peak_ge_{threshold}pct": int((peaks >= threshold).sum()) for threshold in (1, 2, 3, 5, 10)},
            "avg_future_peak_pct": peaks.mean() if not peaks.empty else None,
            "median_future_peak_pct": peaks.median() if not peaks.empty else None,
            "p90_future_peak_pct": peaks.quantile(0.9) if not peaks.empty else None,
            "avg_close_pct": closes.mean() if not closes.empty else None,
            "estimated_additional_entries": len(unblocked),
            "future_data_candidates": len(valid),
            "missing_future_data_candidates": len(unblocked) - len(valid),
            "missed_opportunity_rate_pct": float((peaks >= missed_threshold_pct).mean() * 100.0) if not peaks.empty else None,
            "counterfactual_basis": "single_observed_rejection_reason_removed;all_other_observed_constraints_preserved",
            **effective.output_fields(),
        }
        output.append(row)
    return pd.DataFrame(output, columns=COUNTERFACTUAL_COLUMNS)


def _portfolio_metrics(session_date: str, variant: str, removed_filter: str, result: ReplayResult, config: ReplayConfig) -> dict[str, Any]:
    trades = pd.DataFrame(result.trades)
    pnl = pd.to_numeric(trades.get("net_pnl"), errors="coerce") if not trades.empty else pd.Series(dtype=float)
    gross = pd.to_numeric(trades.get("gross_pnl"), errors="coerce") if not trades.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    equity = pd.Series([value for _timestamp, value in result.equity_curve], dtype=float)
    max_drawdown = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
    return {
        "session_date": session_date,
        "variant": variant,
        "removed_filter": removed_filter,
        "replay_supported": 1,
        "causal_valid": int(bool(effective_config_dict(config)["causal_valid"])),
        "trade_count": len(trades),
        "winners": int((pnl > 0).sum()),
        "losers": int((pnl <= 0).sum()),
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
        "avg_trade": float(pnl.mean()) if len(pnl) else 0.0,
        "median_trade": float(pnl.median()) if len(pnl) else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
        "max_drawdown": max_drawdown,
        "effective_config_json": json.dumps(effective_config_dict(config), sort_keys=True),
        "replay_skip_reason": "",
    }


def portfolio_counterfactuals(
    session_date: str,
    top100_path: Path,
    history_dir: Path,
    filters: Iterable[str],
    candles_by_symbol: dict[str, pd.DataFrame],
    effective: EffectiveSignalThresholds | None = None,
    replay_symbols: set[str] | None = None,
) -> pd.DataFrame:
    effective = effective or EffectiveSignalThresholds(4.0, 6.5, 5.0, "programmatic_historical_strict_default")
    baseline = replace(
        profile_config("live"),
        first5_threshold=effective.min_first5,
        first15_threshold=effective.min_first15,
        min_or_range_pct=effective.min_or_range,
        config_source=effective.config_source,
    )
    prepared_rows = {
        symbol: _rows(frame, baseline.bar_timestamp_semantics)
        for symbol, frame in candles_by_symbol.items()
        if replay_symbols is None or symbol in replay_symbols
    }
    cache = PreparedSessionCache(max_entries=max(1, len(prepared_rows)), max_bytes=256 * 1024 * 1024)
    prepared = {
        symbol: cache.get_or_build(symbol, session_date, rows, baseline)
        for symbol, rows in prepared_rows.items() if not rows.empty
    }
    baseline_result = replay_session(
        session_date=session_date, top100_path=top100_path, history_dir=history_dir, config=baseline,
        prepared_rows_by_symbol=prepared_rows, prepared_sessions_by_symbol=prepared,
    )
    rows = [_portfolio_metrics(session_date, "production", "", baseline_result, baseline)]
    baseline_row = rows[0]
    for filter_name in sorted(set(filters)):
        if filter_name not in SUPPORTED_REPLAY_FILTERS:
            rows.append({
                "session_date": session_date, "variant": f"production_minus_{filter_name}", "removed_filter": filter_name,
                "replay_supported": 0, "causal_valid": 1, "trade_count": None, "winners": None, "losers": None,
                "gross_pnl": None, "net_pnl": None,
                "win_rate": None, "avg_trade": None, "median_trade": None, "profit_factor": None,
                "max_drawdown": None, "effective_config_json": "", "replay_skip_reason": "filter_not_represented_by_shared_replay_config",
                "incremental_trade_count": None, "incremental_net_pnl": None,
            })
            continue
        config = replace(baseline, profile=f"live_without_{filter_name}", disabled_entry_filters=(filter_name,))
        variant_prepared = {
            symbol: cache.get_or_build(symbol, session_date, prepared_rows[symbol], config)
            for symbol in prepared if symbol in prepared_rows
        }
        result = replay_session(
            session_date=session_date, top100_path=top100_path, history_dir=history_dir, config=config,
            prepared_rows_by_symbol=prepared_rows, prepared_sessions_by_symbol=variant_prepared,
        )
        row = _portfolio_metrics(session_date, f"production_minus_{filter_name}", filter_name, result, config)
        row["incremental_trade_count"] = row["trade_count"] - baseline_row["trade_count"]
        row["incremental_net_pnl"] = row["net_pnl"] - baseline_row["net_pnl"]
        rows.append(row)
    rows[0]["incremental_trade_count"] = 0
    rows[0]["incremental_net_pnl"] = 0.0
    return pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)


def load_session_evidence(sqlite_path: Path, recorder_dir: Path, session_date: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_timings: list[dict[str, Any]] = []
    with evidence_timing("sqlite_runtime_events", evidence_timings) as metrics:
        sqlite_events = load_sqlite_events(sqlite_path, session_date)
        metrics["rows"] = len(sqlite_events)
    with evidence_timing("recorder_lifecycle", evidence_timings) as metrics:
        recorder_events, recorder_rows = load_recorder_events(recorder_dir, session_date)
        metrics["rows_scanned"] = recorder_rows
        metrics["rows"] = len(recorder_events)

    higher_priority, higher_duplicate_count = deduplicate_event_sources(sqlite_events, recorder_events)
    higher_keys = {_event_key(event) for event in higher_priority}
    with evidence_timing("p1_light_snapshots", evidence_timings) as metrics:
        light_events, light_rows, light_stats = load_light_events(
            recorder_dir, session_date, higher_priority_keys=higher_keys,
        )
        metrics["rows_scanned"] = light_rows
        metrics["rows"] = len(light_events)
        metrics.update(light_stats)
    with evidence_timing("merge_dedup", evidence_timings) as metrics:
        combined, merge_duplicate_count = deduplicate_event_sources(higher_priority, light_events)
        metrics["rows"] = len(combined)
    duplicate_count = (
        higher_duplicate_count + merge_duplicate_count
        + int(light_stats.get("higher_priority_duplicates_skipped", 0))
    )
    quality = {
        "sqlite_event_rows_used": len(sqlite_events),
        "recorder_rows_scanned": recorder_rows,
        "recorder_rows_used": len(recorder_events),
        "p1_light_rows_scanned": light_rows,
        "p1_light_rows_used": len(light_events),
        "candidate_event_rows_after_dedupe": len(combined),
        "duplicate_event_count": duplicate_count,
        "evidence_timings": evidence_timings,
        "evidence_peak_rss_mb": _peak_rss_mb(),
        "p1_light_chunks": light_stats.get("chunks", 0),
        "p1_light_rows_session_scoped": light_stats.get("rows_session_scoped", 0),
        "p1_light_higher_priority_duplicates_skipped": light_stats.get("higher_priority_duplicates_skipped", 0),
    }
    return combined, quality


def analyze_session(
    session_date: str,
    *,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    top100_dir: Path,
    output_dir: Path,
    missed_threshold_pct: float = 3.0,
    effective: EffectiveSignalThresholds | None = None,
) -> dict[str, Path]:
    effective = effective or EffectiveSignalThresholds(
        4.0, 6.5, 5.0, "programmatic_historical_strict_default"
    )
    timings = PhaseCounter([])
    with phase("load_evidence", timings, date=session_date):
        events, quality = load_session_evidence(sqlite_path, recorder_dir, session_date)
    if not events:
        raise RuntimeError(f"no candidate evaluation evidence for {session_date}")
    evidence_timings = quality.setdefault("evidence_timings", [])
    with evidence_timing("top100", evidence_timings) as metrics:
        top100, top100_path, ranking_source_date = load_top100_source(top100_dir, session_date)
        metrics["rows"] = len(top100)
    if top100_path is None:
        raise RuntimeError(f"dated Top100 unavailable for {session_date}")
    enrich_event_ranking(events, top100)
    evidence_symbols = sorted({event["symbol"] for event in events})
    top100_symbols = {
        normalize_symbol(value) for value in top100.get("symbol", pd.Series(dtype=str)).tolist()
        if normalize_symbol(value)
    }
    history_symbols = sorted(set(evidence_symbols) | top100_symbols)
    with phase("load_history", timings, date=session_date, symbols=len(history_symbols)):
        with evidence_timing("history_candles", evidence_timings) as metrics:
            candles_by_symbol = {
                symbol: load_session_candles(history_dir, symbol, session_date) for symbol in history_symbols
            }
            metrics["symbols"] = len(history_symbols)
            metrics["rows"] = sum(len(frame) for frame in candles_by_symbol.values())
    with phase("candidate_lifetimes", timings, date=session_date, symbols=len(evidence_symbols)):
        lifetimes, outcome_cache = build_lifetimes(
            session_date, events, candles_by_symbol, missed_threshold_pct=missed_threshold_pct,
            effective=effective,
        )
    with phase("counterfactual_filters", timings, date=session_date):
        counterfactual = counterfactual_filter_rows(
            session_date, events, candles_by_symbol, missed_threshold_pct=missed_threshold_pct,
            effective=effective, outcome_cache=outcome_cache,
        )
    with phase("portfolio_replays", timings, date=session_date, filters=len(counterfactual)):
        portfolio = portfolio_counterfactuals(
            session_date, top100_path, history_dir, counterfactual.get("filter", pd.Series(dtype=str)),
            candles_by_symbol, replay_symbols=top100_symbols,
            effective=effective,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "lifetime_csv": output_dir / f"candidate_lifetime_{session_date}.csv",
        "lifetime_json": output_dir / f"candidate_lifetime_{session_date}.json",
        "counterfactual": output_dir / f"candidate_filter_counterfactual_{session_date}.csv",
        "portfolio": output_dir / f"candidate_filter_counterfactual_portfolio_{session_date}.csv",
        "quality": output_dir / f"candidate_filter_data_quality_{session_date}.json",
    }
    write_dataframe(lifetimes, paths["lifetime_csv"])
    write_dataframe(counterfactual, paths["counterfactual"])
    write_dataframe(portfolio, paths["portfolio"])
    json_records = lifetimes.astype(object).where(pd.notna(lifetimes), None).to_dict("records")
    paths["lifetime_json"].write_text(
        json.dumps(json_records, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    history_rows = sum(len(frame) for frame in candles_by_symbol.values())
    candidate_count = int(lifetimes["symbol"].nunique()) if not lifetimes.empty else 0
    rejected_count = int(lifetimes["first_rejected_at"].fillna("").ne("").sum()) if not lifetimes.empty else 0
    with_future = int(lifetimes["future_data_quality"].isin(["complete", "partial"]).sum()) if not lifetimes.empty else 0
    quality.update({
        "session_date": session_date,
        "ranking_source_date": ranking_source_date,
        "candidate_count": candidate_count,
        "rejected_candidate_count": rejected_count,
        "candidate_with_future_data_count": with_future,
        "candidate_missing_future_data_count": max(0, rejected_count - with_future),
        "history_rows_used": history_rows,
        "future_data_coverage_pct": (with_future / rejected_count * 100.0) if rejected_count else None,
        **effective.output_fields(),
        "phase_timings": timings.phases,
    })
    paths["quality"].write_text(json.dumps(quality, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return paths


def _period_rows(frame: pd.DataFrame, period_type: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame.copy()
    dates = pd.to_datetime(work["session_date"], errors="coerce")
    if period_type == "week":
        periods = dates.dt.to_period("W-SUN")
    elif period_type == "month":
        periods = dates.dt.to_period("M")
    else:
        periods = pd.Series(["all"] * len(work), index=work.index)
    output: list[dict[str, Any]] = []
    work["_period"] = periods.astype(str)
    for (period, filter_name), group in work.groupby(["_period", "filter"], dropna=False):
        weights = pd.to_numeric(group["candidates_unblocked_if_removed"], errors="coerce").fillna(0)
        rejected = int(pd.to_numeric(group["rejected_candidates"], errors="coerce").sum())
        unblocked = int(weights.sum())
        missed = int(pd.to_numeric(group["later_peak_ge_3pct"], errors="coerce").sum())
        output.append({
            "period_type": period_type,
            "period": period,
            "start_date": group["session_date"].min(),
            "end_date": group["session_date"].max(),
            "filter": filter_name,
            "sessions": group["session_date"].nunique(),
            "candidates_rejected": rejected,
            "candidates_unblocked_if_removed": unblocked,
            "opportunities_correctly_filtered": max(0, unblocked - missed),
            "missed_opportunities": missed,
            "missed_opportunity_rate_pct": (missed / unblocked * 100.0) if unblocked else None,
            "avg_future_peak_pct": np.average(pd.to_numeric(group["avg_future_peak_pct"], errors="coerce").fillna(0), weights=weights) if weights.sum() else None,
        })
    return output


def aggregate_range(
    dates: list[str],
    daily_paths: list[dict[str, Path]],
    *,
    output_dir: Path,
    minimum_sample_size: int,
) -> dict[str, Path]:
    counter = pd.concat([pd.read_csv(paths["counterfactual"]) for paths in daily_paths], ignore_index=True)
    portfolio = pd.concat([pd.read_csv(paths["portfolio"]) for paths in daily_paths], ignore_index=True)
    periods = pd.DataFrame([
        *_period_rows(counter, "week"),
        *_period_rows(counter, "month"),
        *_period_rows(counter, "all"),
    ])
    supported = portfolio[pd.to_numeric(portfolio["replay_supported"], errors="coerce").fillna(0).gt(0)].copy()
    variant = supported[supported["removed_filter"].fillna("").ne("")].groupby("removed_filter").agg(
        estimated_incremental_pnl=("incremental_net_pnl", "sum"),
        counterfactual_trade_count=("trade_count", "sum"),
        counterfactual_winners=("winners", "sum"),
        counterfactual_losers=("losers", "sum"),
        counterfactual_gross_profit=("net_pnl", lambda values: pd.to_numeric(values, errors="coerce").clip(lower=0).sum()),
        counterfactual_gross_loss=("net_pnl", lambda values: abs(pd.to_numeric(values, errors="coerce").clip(upper=0).sum())),
    ).reset_index().rename(columns={"removed_filter": "filter"})
    all_rows = periods[periods["period_type"].eq("all")].merge(variant, on="filter", how="left")
    details: list[dict[str, Any]] = []
    for _, row in all_rows.iterrows():
        sample = int(row.get("candidates_unblocked_if_removed") or 0)
        missed_rate = fnum(row.get("missed_opportunity_rate_pct"), 0) or 0
        incremental = fnum(row.get("estimated_incremental_pnl"), 0) or 0
        profit = fnum(row.get("counterfactual_gross_profit"), 0) or 0
        loss = fnum(row.get("counterfactual_gross_loss"), 0) or 0
        pf = profit / loss if loss > 0 else (math.inf if profit > 0 else 0.0)
        if sample < minimum_sample_size:
            classification = "NEUTRAL"
            basis = f"BASELINE ONLY; sample {sample} < minimum {minimum_sample_size}"
        elif incremental > 0 and missed_rate >= 25 and pf > 1:
            classification = "HARMFUL"
            basis = "removal improved replay PnL and rejected many later opportunities"
        elif incremental > 0 or missed_rate >= 20:
            classification = "SUSPECT"
            basis = "positive removal signal requires multi-day validation"
        elif incremental < 0 and missed_rate < 20:
            classification = "HELPFUL"
            basis = "removal reduced replay PnL with low missed-opportunity rate"
        else:
            classification = "NEUTRAL"
            basis = "no material direction"
        trade_count = fnum(row.get("counterfactual_trade_count"), 0) or 0
        winners = fnum(row.get("counterfactual_winners"), 0) or 0
        details.append({
            **row.to_dict(),
            "counterfactual_win_rate": winners / trade_count * 100.0 if trade_count else None,
            "counterfactual_profit_factor": pf,
            "classification": classification,
            "classification_basis": basis,
        })
    summary = pd.DataFrame(details)
    suffix = f"{dates[0]}_{dates[-1]}"
    paths = {
        "monthly_summary": output_dir / f"candidate_filter_monthly_summary_{suffix}.csv",
        "period_breakdown": output_dir / f"candidate_filter_period_breakdown_{suffix}.csv",
        "report": output_dir / f"candidate_filter_report_{suffix}.md",
    }
    write_dataframe(summary, paths["monthly_summary"])
    write_dataframe(periods, paths["period_breakdown"])
    lines = [
        f"# Candidate Filter Report {dates[0]} to {dates[-1]}", "",
        f"FACT: sessions analyzed={len(dates)}.",
        f"FACT: minimum sample size={minimum_sample_size}.",
        "HYPOTHESIS: counterfactual portfolio results remove exactly one observed/replay gate.",
        "REQUIRES MULTI-DAY VALIDATION.", "POSSIBLE OVERFITTING.", "",
        "| Filter | Rejected | Unblocked | Missed | Incremental PnL | Classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row.get('filter')} | {int(row.get('candidates_rejected') or 0)} | "
            f"{int(row.get('candidates_unblocked_if_removed') or 0)} | {int(row.get('missed_opportunities') or 0)} | "
            f"{fnum(row.get('estimated_incremental_pnl'), 0):.2f} | {row.get('classification')} |"
        )
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze candidate lifetimes and one-filter-at-a-time counterfactuals.")
    parser.add_argument("--date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE_PATH))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--recorder-dir", default=str(DEFAULT_RECORDER_DIR))
    parser.add_argument("--top100-dir", default=str(DEFAULT_TOP100_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--missed-opportunity-threshold-pct", type=float, default=3.0)
    parser.add_argument("--minimum-sample-size", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    add_threshold_cli(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dates = session_dates(args.date, args.start_date, args.end_date)
    output_dir = Path(args.output_dir)
    completed: list[dict[str, Path]] = []
    for session_date in dates:
        expected = output_dir / f"candidate_lifetime_{session_date}.csv"
        if expected.exists() and not args.force and output_has_config_provenance(expected):
            print(f"CANDIDATE_LIFETIME_SKIP date={session_date} reason=output_exists", flush=True)
            existing = {
                "lifetime_csv": expected,
                "lifetime_json": output_dir / f"candidate_lifetime_{session_date}.json",
                "counterfactual": output_dir / f"candidate_filter_counterfactual_{session_date}.csv",
                "portfolio": output_dir / f"candidate_filter_counterfactual_portfolio_{session_date}.csv",
                "quality": output_dir / f"candidate_filter_data_quality_{session_date}.json",
            }
            if all(path.exists() for path in existing.values()):
                completed.append(existing)
            continue
        print(f"CANDIDATE_LIFETIME_START date={session_date}", flush=True)
        effective = resolve_threshold_args(args, session_date)
        paths = analyze_session(
            session_date,
            sqlite_path=Path(args.sqlite_path), history_dir=Path(args.history_dir),
            recorder_dir=Path(args.recorder_dir), top100_dir=Path(args.top100_dir),
            output_dir=output_dir, missed_threshold_pct=args.missed_opportunity_threshold_pct,
            effective=effective,
        )
        completed.append(paths)
        print(f"CANDIDATE_LIFETIME_DONE date={session_date} output={paths['lifetime_csv']}", flush=True)
    if completed:
        aggregate_range(dates, completed, output_dir=output_dir, minimum_sample_size=args.minimum_sample_size)
    return 0
