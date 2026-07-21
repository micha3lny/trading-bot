from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.live_trading.analysis.common import normalize_symbol, parse_dt, safe_read_csv
from src.live_trading.analysis.signal_replay_analyzer import read_sqlite_sources, row_symbol, row_text, row_time
from src.live_trading.analysis.symbol_subscription_inspector import line_time, parse_key_values, read_text_lines

DEFAULT_ANALYSIS_DIR = Path("data/analysis")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")
DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")

FINAL_CLASSES = [
    "READY_BUT_LOST_GLOBAL_RANKING",
    "READY_BUT_MAX_POSITIONS_FULL",
    "READY_BUT_PENDING_ORDER_LIMIT",
    "READY_BUT_ENTRY_BUDGET",
    "READY_BUT_REENTRY_BLOCK",
    "READY_BUT_DUPLICATE_SYMBOL",
    "READY_BUT_COOLDOWN",
    "READY_BUT_SPREAD_CHANGED",
    "READY_BUT_PRICE_MOVED",
    "READY_BUT_MARKET_DATA_STALE",
    "READY_BUT_ORDER_NOT_CREATED",
    "ORDER_CREATED_NOT_SUBMITTED",
    "ORDER_SUBMITTED_REJECTED",
    "ORDER_SUBMITTED_CANCELLED",
    "ORDER_SUBMITTED_NOT_FILLED",
    "BUY_EXISTS_MATCHING_FAILED",
    "RUNTIME_EVIDENCE_MISSING",
    "RECORDER_EVIDENCE_MISSING",
    "UNKNOWN_FINAL",
]

CASE_COLUMNS = [
    "symbol",
    "possible_signal_time",
    "runtime_ready_time",
    "signal_ready_reason",
    "candidate_rank",
    "global_rank",
    "top100_rank",
    "live_entry_score",
    "live_entry_rank",
    "spread_bps",
    "open_positions",
    "pending_orders",
    "entries_blocked",
    "blocking_reason",
    "buy_intent_created",
    "order_created",
    "order_submitted",
    "order_id",
    "perm_id",
    "order_status",
    "execution_found",
    "execution_time",
    "final_root_cause",
    "evidence_quality",
    "evidence_sources",
]

RECORDER_FILES = {
    "trade_lifecycle": ["trade_lifecycle.csv"],
    "order_lifecycle": ["order_lifecycle.jsonl"],
    "fills": ["fills.csv"],
    "run_metadata": ["run_metadata.csv"],
    "strategy_equity": ["strategy_equity.csv"],
    "portfolio_snapshots": ["portfolio_snapshots.csv"],
    "contract_metadata": ["contract_metadata.csv"],
    "candles_1m": ["candles_1m.csv"],
}

JSON_FILES = ["managed_positions.json", "eod_summary.json", "eod_pending.json"]


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def num(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def text_has_symbol(text: str, symbol: str) -> bool:
    sym = normalize_symbol(symbol)
    if not sym:
        return False
    return re.search(rf"(?<![A-Z0-9]){re.escape(sym)}(?![A-Z0-9])", str(text or "").upper()) is not None


def iso(value: Any) -> str:
    ts = parse_dt(value)
    return "" if ts is None else ts.isoformat()


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return ""


def source_path(recorder_dir: Path, session_date: str, name: str) -> Path:
    return recorder_dir / session_date / name


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            parsed = {"raw_line": line}
        if isinstance(parsed, dict):
            rows.append(parsed)
    return pd.DataFrame(rows)


def load_recorder_table(recorder_dir: Path, session_date: str, filename: str) -> pd.DataFrame:
    path = source_path(recorder_dir, session_date, filename)
    if filename.endswith(".jsonl"):
        return load_jsonl(path)
    return safe_read_csv(path)


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(errors="replace"))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception as exc:
        return {"_load_error": repr(exc)}


@dataclass(frozen=True)
class Evidence:
    sqlite_sources: dict[str, pd.DataFrame]
    recorder_sources: dict[str, pd.DataFrame]
    journal_lines: list[str]
    json_sources: dict[str, dict[str, Any]]


def default_journal_path(analysis_dir: Path, session_date: str) -> Path | None:
    candidates = sorted(analysis_dir.glob(f"journal_v67_{session_date}*.log"))
    return candidates[0] if candidates else None


def load_evidence(sqlite_path: Path, recorder_dir: Path, analysis_dir: Path, session_date: str, journal_log: Path | None) -> Evidence:
    sqlite_sources = read_sqlite_sources(sqlite_path, session_date)
    recorder_sources: dict[str, pd.DataFrame] = {}
    for source, names in RECORDER_FILES.items():
        frames = [load_recorder_table(recorder_dir, session_date, name) for name in names]
        recorder_sources[source] = next((frame for frame in frames if not frame.empty), pd.DataFrame())
    json_sources = {name: load_json_file(source_path(recorder_dir, session_date, name)) for name in JSON_FILES}
    journal_path = journal_log or default_journal_path(analysis_dir, session_date)
    journal_lines = read_text_lines(journal_path) if journal_path else []
    return Evidence(sqlite_sources=sqlite_sources, recorder_sources=recorder_sources, journal_lines=journal_lines, json_sources=json_sources)


def row_event(row: dict[str, Any], fallback: str = "") -> str:
    raw = row.get("raw_json")
    raw_dict: dict[str, Any] = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            raw_dict = parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    elif isinstance(raw, dict):
        raw_dict = raw
    value = first_nonempty(
        row.get("event_type"), row.get("event"), row.get("legacy_event"), row.get("status"),
        row.get("order_status"), row.get("action"), raw_dict.get("event_type"), raw_dict.get("event"), raw_dict.get("status"), fallback,
    )
    return str(value or "").strip().upper().replace(" ", "_")


def row_reason(row: dict[str, Any]) -> str:
    text = row_text(row)
    kv = parse_key_values(text)
    return str(first_nonempty(row.get("reason"), row.get("blocked_reason"), row.get("reject_reason"), kv.get("reason"), kv.get("blocked_reason"), ""))


def event_records_for_symbol(evidence: Evidence, symbol: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source, df in evidence.sqlite_sources.items():
        if df.empty:
            continue
        for row in df.to_dict("records"):
            text = row_text(row)
            if row_symbol(row) != normalize_symbol(symbol) and not text_has_symbol(text, symbol):
                continue
            records.append({
                "time": row_time(row),
                "source": f"sqlite:{source}",
                "event": row_event(row, source),
                "reason": row_reason(row),
                "text": text[:1200],
                "row": row,
            })
    for source, df in evidence.recorder_sources.items():
        if df.empty:
            continue
        for row in df.to_dict("records"):
            text = row_text(row)
            if row_symbol(row) != normalize_symbol(symbol) and not text_has_symbol(text, symbol):
                continue
            records.append({
                "time": row_time(row),
                "source": f"recorder:{source}",
                "event": row_event(row, source),
                "reason": row_reason(row),
                "text": text[:1200],
                "row": row,
            })
    for line in evidence.journal_lines:
        if not text_has_symbol(line, symbol):
            continue
        records.append({
            "time": line_time(line) or parse_dt(line[:32]),
            "source": "journal",
            "event": parse_key_values(line).get("event") or "journal",
            "reason": parse_key_values(line).get("reason") or "",
            "text": line[:1200],
            "row": parse_key_values(line),
        })
    return sorted(records, key=lambda item: item.get("time") or pd.Timestamp.max.tz_localize("UTC"))


def global_records(evidence: Evidence) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in ["run_metadata", "strategy_equity", "portfolio_snapshots"]:
        df = evidence.recorder_sources.get(source, pd.DataFrame())
        if df.empty:
            continue
        for row in df.to_dict("records"):
            records.append({"time": row_time(row), "source": f"recorder:{source}", "event": row_event(row, source), "text": row_text(row)[:1200], "row": row})
    for line in evidence.journal_lines:
        lower = line.lower()
        if any(tok in lower for tok in ["heartbeat", "entries_blocked", "managed_open", "pending_orders", "max_positions", "entry_budget"]):
            records.append({"time": line_time(line) or parse_dt(line[:32]), "source": "journal", "event": "global", "text": line[:1200], "row": parse_key_values(line)})
    return sorted(records, key=lambda item: item.get("time") or pd.Timestamp.max.tz_localize("UTC"))


def records_between(records: list[dict[str, Any]], start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        ts = record.get("time")
        if ts is None:
            continue
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        out.append(record)
    return out


def nearest_before_or_after(records: list[dict[str, Any]], center: pd.Timestamp | None, *, minutes: int = 5) -> dict[str, Any] | None:
    if center is None:
        return records[-1] if records else None
    window = [r for r in records if r.get("time") is not None and abs((r["time"] - center).total_seconds()) <= minutes * 60]
    if not window:
        return None
    return min(window, key=lambda r: abs((r["time"] - center).total_seconds()))


def text_blob(records: Iterable[dict[str, Any]]) -> str:
    return "\n".join(str(r.get("event", "")) + " " + str(r.get("reason", "")) + " " + str(r.get("text", "")) for r in records).upper()


def extract_kv_value(records: list[dict[str, Any]], aliases: list[str]) -> Any:
    for record in records:
        row = record.get("row") or {}
        text = str(record.get("text") or "")
        kv = parse_key_values(text)
        for alias in aliases:
            value = row.get(alias) if isinstance(row, dict) else None
            if value not in (None, ""):
                return value
            if kv.get(alias) not in (None, ""):
                return kv.get(alias)
    return ""


def find_first(records: list[dict[str, Any]], tokens: list[str]) -> dict[str, Any] | None:
    upper_tokens = [token.upper() for token in tokens]
    for record in records:
        text = (str(record.get("event") or "") + " " + str(record.get("reason") or "") + " " + str(record.get("text") or "")).upper()
        if any(token in text for token in upper_tokens):
            return record
    return None


def find_order_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = ["ENTRY_ORDER_DISPATCH", "BUY_ORDER", "PAPER BUY", "PLACEORDER", "ORDER_SUBMITTED", "SUBMITTED", "PRESUBMITTED", "ORDERSTATUS", "ORDER_STATUS", "CANCEL", "REJECT", "INACTIVE", "FILLED"]
    return [r for r in records if any(tok in (str(r.get("event"))+" "+str(r.get("text"))).upper() for tok in tokens)]


def load_cases(session_date: str, analysis_dir: Path, cases_csv: Path | None) -> pd.DataFrame:
    candidates = []
    if cases_csv:
        candidates.append(cases_csv)
    candidates.extend([
        analysis_dir / f"no_buy_after_signal_cases_{session_date}.csv",
        analysis_dir / f"should_have_signaled_cases_{session_date}.csv",
    ])
    for path in candidates:
        df = safe_read_csv(path)
        if df.empty:
            continue
        if "final_no_buy_reason" in df.columns:
            out = df[df["final_no_buy_reason"].fillna("").astype(str).isin([
                "runtime_signal_ready_but_no_buy",
                "unexplained_after_signal_before_dispatch",
                "unknown_no_buy_after_signal",
                "entries_blocked",
                "lower_rank_candidate_not_selected",
            ]) | df.get("order_dispatch_attempted", pd.Series(dtype=object)).fillna("0").astype(str).eq("0")].copy()
            if not out.empty:
                return out
        if "final_classification" in df.columns:
            out = df[df["final_classification"].fillna("").astype(str) == "runtime_signal_ready_but_no_buy"].copy()
            if not out.empty:
                return out
    return pd.DataFrame()


def classify_root_cause(*, row: dict[str, Any], symbol_records: list[dict[str, Any]], window_records: list[dict[str, Any]], global_window: list[dict[str, Any]], order_records: list[dict[str, Any]], execution_record: dict[str, Any] | None) -> tuple[str, str]:
    all_text = text_blob(window_records + order_records)
    global_text = text_blob(global_window)
    if execution_record is not None:
        if not order_records:
            return "BUY_EXISTS_MATCHING_FAILED", "execution/fill exists but matching BUY/order evidence is missing"
        return "BUY_EXISTS_MATCHING_FAILED", "BUY execution exists despite SHS/NBAS no-buy classification"
    if not symbol_records:
        return "RUNTIME_EVIDENCE_MISSING", "no symbol-specific runtime/recorder/sqlite evidence"
    if not any(str(r.get("source", "")).startswith("recorder:") for r in symbol_records):
        recorder_sources = {str(r.get("source")) for r in symbol_records if str(r.get("source", "")).startswith("recorder:")}
        if not recorder_sources:
            return "RECORDER_EVIDENCE_MISSING", "SQLite/journal evidence exists but recorder evidence is missing"
    if any(tok in all_text for tok in ["REJECT", "NO_TRADING_PERMISSION", "ERROR 201", "INACTIVE"]):
        return "ORDER_SUBMITTED_REJECTED", "order rejection/inactive evidence found"
    if any(tok in all_text for tok in ["CANCELLED", "CANCELED", "ORDER_CANCELLED"]):
        return "ORDER_SUBMITTED_CANCELLED", "order cancellation evidence found"
    if order_records and not execution_record:
        if any(tok in all_text for tok in ["SUBMITTED", "PRESUBMITTED", "ENTRY_ORDER_IBKR_SUBMITTED", "BUY_ORDER_SENT", "PAPER BUY"]):
            return "ORDER_SUBMITTED_NOT_FILLED", "order submitted/sent but no execution/fill found"
        return "ORDER_CREATED_NOT_SUBMITTED", "order lifecycle evidence exists but submitted marker/fill missing"
    if any(tok in all_text for tok in ["ENTRY_ORDER_DISPATCH_ATTEMPT", "BUY_INTENT", "ENTRY_ORDER_DISPATCH_RETURNED"]):
        return "ORDER_CREATED_NOT_SUBMITTED", "dispatch/intent evidence exists but no submitted/fill evidence"
    if any(tok in all_text for tok in ["ALREADY_OPEN", "DUPLICATE", "already_open_position".upper()]):
        return "READY_BUT_DUPLICATE_SYMBOL", "already-open/duplicate symbol evidence found"
    if any(tok in all_text for tok in ["COOLDOWN", "REENTRY"]):
        return "READY_BUT_COOLDOWN" if "COOLDOWN" in all_text else "READY_BUT_REENTRY_BLOCK", "cooldown/reentry block evidence found"
    if any(tok in all_text for tok in ["SPREAD"]):
        return "READY_BUT_SPREAD_CHANGED", "spread-related post-ready evidence found"
    if any(tok in all_text for tok in ["NO_USABLE_TICKER_PRICE", "PRICE_MISSING", "PRICE_MOVED", "MARKET_DATA_STALE", "STALE_TICKER"]):
        if "STALE" in all_text:
            return "READY_BUT_MARKET_DATA_STALE", "market-data stale evidence found"
        return "READY_BUT_PRICE_MOVED", "price/ticker evidence changed before order"
    if any(tok in all_text for tok in ["PENDING_ORDER", "PENDING ORDER", "ORDER_LIMIT"]):
        return "READY_BUT_PENDING_ORDER_LIMIT", "pending order limit evidence found"
    if any(tok in all_text + global_text for tok in ["MAX_POSITIONS", "MAX_POSITION", "MANAGED_OPEN", "POSITION_LIMIT"]):
        return "READY_BUT_MAX_POSITIONS_FULL", "max position/open-position evidence found"
    if any(tok in all_text + global_text for tok in ["ENTRY_BUDGET", "MAX_ENTRIES", "PER_SCAN", "ENTRY_RATE_LIMIT"]):
        return "READY_BUT_ENTRY_BUDGET", "entry budget/per-scan limit evidence found"
    ahead = str(first_nonempty(row.get("better_candidates_ahead_symbols"), ""))
    ahead_count = num(row.get("candidates_ahead_count")) or 0
    if ahead or ahead_count > 0:
        return "READY_BUT_LOST_GLOBAL_RANKING", "better candidates ahead in NBAS evidence"
    if truthy(row.get("entries_blocked_at_scan")) or str(row.get("entries_blocked_reason_at_scan") or ""):
        reason = str(row.get("entries_blocked_reason_at_scan") or "")
        if "max" in reason.lower() or "position" in reason.lower():
            return "READY_BUT_MAX_POSITIONS_FULL", reason
        if "pending" in reason.lower():
            return "READY_BUT_PENDING_ORDER_LIMIT", reason
        if "budget" in reason.lower() or "entry" in reason.lower():
            return "READY_BUT_ENTRY_BUDGET", reason
        return "READY_BUT_ORDER_NOT_CREATED", f"entries blocked at scan: {reason or 'unknown'}"
    if any("SIGNAL_READY" in (str(r.get("event"))+" "+str(r.get("text"))).upper() for r in symbol_records):
        return "READY_BUT_ORDER_NOT_CREATED", "SIGNAL_READY observed but no order/terminal evidence found"
    return "UNKNOWN_FINAL", "insufficient evidence after multi-source timeline"


def build_case(row: dict[str, Any], evidence: Evidence, session_date: str) -> tuple[dict[str, Any], str]:
    symbol = normalize_symbol(row.get("symbol"))
    possible_time = parse_dt(first_nonempty(row.get("possible_signal_time"), row.get("signal_ready_time"), row.get("opening_range_break_time")))
    ready_time = parse_dt(first_nonempty(row.get("signal_ready_event_timestamp"), row.get("signal_ready_time"), row.get("runtime_ready_time")))
    center = ready_time or possible_time
    start = center - pd.Timedelta(minutes=10) if center is not None else None
    end = center + pd.Timedelta(minutes=45) if center is not None else None
    symbol_records = event_records_for_symbol(evidence, symbol)
    window_records = records_between(symbol_records, start, end)
    global_window = records_between(global_records(evidence), start, end)
    order_records = find_order_records(window_records)
    execution_record = find_first(window_records, ["EXECUTION", "FILL", "FILLED"])
    signal_record = find_first(window_records, ["SIGNAL_READY"])
    if signal_record and ready_time is None:
        ready_time = signal_record.get("time")
    signal_reason = first_nonempty(row.get("signal_ready_reason"), row.get("post_signal_terminal_reason"), signal_record.get("reason") if signal_record else "")
    heartbeat = nearest_before_or_after(global_window, center, minutes=5) or {}
    heartbeat_row = heartbeat.get("row", {}) if isinstance(heartbeat.get("row"), dict) else {}
    heartbeat_text = str(heartbeat.get("text") or "")
    heartbeat_kv = {**parse_key_values(heartbeat_text), **heartbeat_row}
    order_first = order_records[0] if order_records else None
    order_text = text_blob(order_records)
    order_kv = parse_key_values(order_first.get("text", "") if order_first else "")
    order_id = first_nonempty(row.get("buy_order_id"), row.get("order_id"), order_kv.get("orderId"), order_kv.get("order_id"), extract_kv_value(order_records, ["order_id", "orderId"]))
    perm_id = first_nonempty(row.get("perm_id"), order_kv.get("permId"), order_kv.get("perm_id"), extract_kv_value(order_records, ["perm_id", "permId"]))
    order_status = first_nonempty(extract_kv_value(order_records, ["order_status", "status"]), "SUBMITTED" if "SUBMITTED" in order_text else "")
    cause, cause_reason = classify_root_cause(row=row, symbol_records=symbol_records, window_records=window_records, global_window=global_window, order_records=order_records, execution_record=execution_record)
    sources = sorted({str(r.get("source")) for r in window_records + global_window if r.get("source")})
    evidence_quality = "high" if cause != "UNKNOWN_FINAL" and len(sources) >= 2 else ("medium" if cause != "UNKNOWN_FINAL" else "low")
    timeline = build_timeline_markdown(symbol, possible_time, ready_time, window_records, global_window, cause, cause_reason)
    case = {
        "symbol": symbol,
        "possible_signal_time": iso(possible_time),
        "runtime_ready_time": iso(ready_time),
        "signal_ready_reason": signal_reason,
        "candidate_rank": first_nonempty(row.get("candidate_rank_at_scan"), row.get("candidate_rank"), row.get("top100_rank")),
        "global_rank": first_nonempty(row.get("candidate_rank_at_scan"), row.get("global_rank"), ""),
        "top100_rank": row.get("top100_rank"),
        "live_entry_score": first_nonempty(row.get("candidate_score_at_scan"), row.get("live_entry_score")),
        "live_entry_rank": first_nonempty(row.get("live_entry_rank"), row.get("candidate_rank_at_scan")),
        "spread_bps": first_nonempty(row.get("spread_bps_at_scan"), row.get("spread_bps"), extract_kv_value(window_records, ["spread_bps", "spread_bps_at_entry"])),
        "open_positions": first_nonempty(row.get("managed_open_at_scan"), row.get("max_positions_at_scan"), heartbeat_kv.get("managed_open"), heartbeat_kv.get("managed_open_positions")),
        "pending_orders": first_nonempty(heartbeat_kv.get("pending_orders"), heartbeat_kv.get("pending_order_count"), row.get("pending_orders")),
        "entries_blocked": int(truthy(row.get("entries_blocked_at_scan")) or truthy(heartbeat_kv.get("entries_blocked"))),
        "blocking_reason": first_nonempty(row.get("entries_blocked_reason_at_scan"), heartbeat_kv.get("entries_blocked_reason"), heartbeat_kv.get("risk_guard_reason"), cause_reason),
        "buy_intent_created": int(find_first(window_records, ["BUY_INTENT", "ENTRY_ORDER_DISPATCH_ATTEMPT"]) is not None),
        "order_created": int(bool(order_records)),
        "order_submitted": int(any(tok in order_text for tok in ["SUBMITTED", "PRESUBMITTED", "ENTRY_ORDER_IBKR_SUBMITTED", "BUY_ORDER_SENT", "PAPER BUY"])),
        "order_id": order_id,
        "perm_id": perm_id,
        "order_status": order_status,
        "execution_found": int(execution_record is not None),
        "execution_time": iso(execution_record.get("time") if execution_record else ""),
        "final_root_cause": cause,
        "evidence_quality": evidence_quality,
        "evidence_sources": ";".join(sources),
    }
    return case, timeline


def build_timeline_markdown(symbol: str, possible_time: pd.Timestamp | None, ready_time: pd.Timestamp | None, records: list[dict[str, Any]], global_records_: list[dict[str, Any]], cause: str, reason: str) -> str:
    lines = [f"### {symbol}", ""]
    if possible_time is not None:
        lines += [possible_time.strftime("%H:%M:%S"), "", "history says BUY", ""]
    if ready_time is not None:
        lines += [ready_time.strftime("%H:%M:%S"), "", "runtime SIGNAL_READY / ready evidence", ""]
    merged = sorted(records + global_records_, key=lambda r: r.get("time") or pd.Timestamp.max.tz_localize("UTC"))
    seen = 0
    for record in merged:
        if seen >= 60:
            break
        ts = record.get("time")
        if ts is None:
            continue
        text = str(record.get("event") or record.get("source") or "event")
        reason_text = str(record.get("reason") or "")
        detail = str(record.get("text") or "")[:280].replace("\n", " ")
        lines += [ts.strftime("%H:%M:%S"), "", f"{record.get('source')}: {text} {reason_text} {detail}".strip(), ""]
        seen += 1
    lines += ["ROOT CAUSE:", "", cause, "", f"Evidence: {reason}", ""]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CASE_COLUMNS})


def write_summary(path: Path, session_date: str, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("final_root_cause") for row in rows)
    lines = [f"# SHS Root Cause Summary {session_date}", "", f"total_cases={len(rows)}", ""]
    for klass in FINAL_CLASSES:
        if counts.get(klass):
            lines.append(f"- {klass}: {counts[klass]}")
    lines += ["", "## Symbols", ""]
    for row in rows:
        lines.append(f"- {row['symbol']}: {row['final_root_cause']} ({row.get('blocking_reason') or row.get('evidence_quality')})")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    started = time.time()
    session_date = args.date
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir or analysis_dir)
    cases = load_cases(session_date, analysis_dir, Path(args.cases_csv) if args.cases_csv else None)
    if args.max_cases is not None and not cases.empty:
        cases = cases.head(args.max_cases).copy()
    target_symbols = {normalize_symbol(v) for v in cases.get("symbol", pd.Series(dtype=object)).tolist() if normalize_symbol(v)}
    print(f"SHS_ROOT_CAUSE_START date={session_date} targets={len(cases)}", flush=True)
    evidence = load_evidence(Path(args.sqlite_path), Path(args.recorder_dir), analysis_dir, session_date, Path(args.journal_log) if args.journal_log else None)
    print(
        f"SHS_ROOT_CAUSE_LOAD_EVIDENCE_DONE date={session_date} sqlite_sources={len(evidence.sqlite_sources)} "
        f"recorder_sources={len(evidence.recorder_sources)} journal_lines={len(evidence.journal_lines)} json_sources={len(evidence.json_sources)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    timelines: list[str] = []
    for idx, row in enumerate(cases.to_dict("records"), start=1):
        case, timeline = build_case(row, evidence, session_date)
        rows.append(case)
        timelines.append(timeline)
        if idx % 5 == 0 or idx == len(cases):
            print(f"SHS_ROOT_CAUSE_PROGRESS date={session_date} processed={idx}/{len(cases)}", flush=True)
    missing = [row for row in rows if not row.get("final_root_cause")]
    if missing:
        raise RuntimeError(f"missing final_root_cause for {len(missing)} rows")
    cases_path = output_dir / f"shs_root_cause_cases_{session_date}.csv"
    summary_path = output_dir / f"shs_root_cause_summary_{session_date}.md"
    timeline_path = output_dir / f"shs_timelines_{session_date}.md"
    write_csv(cases_path, rows)
    write_summary(summary_path, session_date, rows)
    timeline_path.write_text("\n---\n".join(timelines) + ("\n" if timelines else ""))
    print(
        f"SHS_ROOT_CAUSE_DONE date={session_date} elapsed_seconds={time.time()-started:.1f} "
        f"cases={cases_path} summary={summary_path} timelines={timeline_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find final root causes for Should-Have-Signaled runtime-ready no-buy cases.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--cases-csv")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE_PATH))
    parser.add_argument("--recorder-dir", default=str(DEFAULT_RECORDER_DIR))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--journal-log")
    parser.add_argument("--max-cases", type=int)
    return parser


def main() -> int:
    return run(build_parser().parse_args())
