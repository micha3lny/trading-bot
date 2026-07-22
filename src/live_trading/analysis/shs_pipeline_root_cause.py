from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import iso_ts, live_signal_replay, load_session_candles, load_top100, normalize_symbol, parse_dt, safe_read_csv

DEFAULT_ANALYSIS_DIR = Path("data/analysis")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")
DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_LOG_DIR = Path("data/logs")

PIPELINE_STAGES = [
    "daily_top100_inclusion",
    "runtime_top100_watchlist_loading",
    "symbol_registration",
    "contract_qualification_request",
    "contract_resolution",
    "market_data_subscription_request",
    "subscription_acknowledgement",
    "first_ticker_price_update",
    "candidate_state_creation",
    "first5_first15_or_state",
    "signal_evaluation",
    "ranking_replacement",
    "buy_decision",
]

FINAL_CLASSIFICATIONS = {
    "TOP100_NOT_LOADED_RUNTIME",
    "WATCHLIST_REGISTRATION_MISSING",
    "CONTRACT_REQUEST_MISSING",
    "CONTRACT_QUALIFICATION_FAILED",
    "MARKET_DATA_SUBSCRIPTION_MISSING",
    "MARKET_DATA_SUBSCRIPTION_FAILED",
    "NO_TICKER_RECEIVED",
    "SYMBOL_STATE_NOT_CREATED",
    "SYMBOL_STATE_DROPPED_AFTER_RECONNECT",
    "RECORDER_EVIDENCE_MISSING",
    "ACTUAL_RUNTIME_PROCESSING_BUG",
    "DATA_RETENTION_PREVENTS_ROOT_CAUSE",
    "CONTRACT_REQUEST_NOT_SENT",
    "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED",
    "CONTRACT_RESOLVED_FROM_CACHE",
    "CONTRACT_REQUEST_FAILED",
    "SUBSCRIPTION_REQUEST_NOT_SENT",
    "SUBSCRIPTION_REQUEST_SENT_BUT_NOT_RECORDED",
    "INSUFFICIENT_TELEMETRY",
}

CSV_COLUMNS = [
    "date",
    "symbol",
    "top100_rank",
    "top100_score",
    "offline_possible_signal_time",
    "live_equivalent_signal_time",
    "present_in_daily_top100",
    "present_in_runtime_top100_state",
    "contract_request_seen",
    "contract_resolved_seen",
    "market_data_subscription_seen",
    "ticker_seen",
    "state_seen",
    "signal_evaluation_seen",
    "buy_decision_seen",
    "first_confirmed_divergence_stage",
    "final_root_cause",
    "confidence",
    "missed_trade_was_real",
    "live_would_have_been_eligible_at_offline_signal_time",
    "exact_reason_missed",
    "runtime_evidence_basis",
    "runtime_evidence_source",
    "runtime_evidence_payload_excerpt",
    "contract_telemetry_assessment",
    "subscription_telemetry_assessment",
    "could_contract_have_been_cached",
    "required_fix",
    "regression_test_needed",
    "another_symbol_occupied_subscription_slot",
    "reconnect_or_rollover_evidence",
    "could_recur",
    "evidence_sources",
]


@dataclass
class EvidenceItem:
    timestamp: pd.Timestamp | None
    source: str
    stage: str
    event_type: str
    symbol: str
    payload: str


def _read_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _row_symbol(row: dict[str, Any]) -> str:
    raw = _read_json(row.get("raw_json") or row.get("payload"))
    for key in ["symbol", "contract_symbol", "ticker"]:
        value = row.get(key) or raw.get(key)
        if value:
            return normalize_symbol(value)
    text = " ".join(str(v) for v in row.values() if v not in (None, ""))
    for token in text.replace(",", " ").split():
        if token.startswith("symbol="):
            return normalize_symbol(token.split("=", 1)[1])
    return ""


def _row_time(row: dict[str, Any]) -> pd.Timestamp | None:
    raw = _read_json(row.get("raw_json") or row.get("payload"))
    for key in ["timestamp", "event_time", "recorded_at", "time", "created_at", "updated_at", "executed_at", "entry_time", "signal_time"]:
        value = row.get(key) or raw.get(key)
        parsed = parse_dt(value)
        if parsed is not None:
            return parsed
    return None


def _event_type(row: dict[str, Any], source: str) -> str:
    raw = _read_json(row.get("raw_json") or row.get("payload"))
    for key in ["event_type", "event", "lifecycle_event", "status", "action", "type"]:
        value = row.get(key) or raw.get(key)
        if value:
            return str(value).upper()
    return source.upper()


def _payload(row: dict[str, Any]) -> str:
    try:
        return json.dumps(row, sort_keys=True, default=str)[:1000]
    except Exception:
        return str(row)[:1000]


def _session_date_from_timestamp(value: Any) -> str:
    parsed = parse_dt(value)
    if parsed is None:
        return ""
    return str(parsed.date())


def _explicit_session_date(row: dict[str, Any]) -> str:
    raw = _read_json(row.get("raw_json") or row.get("payload"))
    for key in ["session_date", "trading_session_date", "trade_session_date"]:
        value = row.get(key) or raw.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if len(text) >= 10:
            return text[:10]
    return ""


def row_matches_session(row: dict[str, Any], session_date: str) -> bool:
    """Strictly scope evidence to the requested runtime session.

    Runtime rows from D+1 can legitimately carry top100_source_date=D. That field
    is not session evidence and must never pull next-day runtime events into D.
    """
    explicit = _explicit_session_date(row)
    if explicit:
        return explicit == session_date
    raw = _read_json(row.get("raw_json") or row.get("payload"))
    for key in ["timestamp", "event_time", "recorded_at", "time", "created_at", "updated_at", "executed_at", "entry_time", "signal_time"]:
        value = row.get(key) or raw.get(key)
        stamp_date = _session_date_from_timestamp(value)
        if stamp_date:
            return stamp_date == session_date
    return True


def _stage_for_event(event_type: str, source: str, payload: str) -> str:
    text = f"{event_type} {source} {payload}".upper()
    if "TOP100" in text or "WATCHLIST" in text:
        return "runtime_top100_watchlist_loading"
    if "CONTRACT" in text and any(key in text for key in ["QUAL", "RESOL", "META", "CONID", "DETAIL"]):
        if any(key in text for key in ["FAILED", "ERROR", "NOT_QUALIFIED"]):
            return "contract_qualification_request"
        return "contract_resolution"
    if any(key in text for key in ["SUBSCRIB", "REQMKTDATA", "MARKET_DATA"]):
        return "market_data_subscription_request"
    if any(key in text for key in ["NO_USABLE_TICKER_PRICE", "TICKER", "CANDLE", "PRICE", "QUOTE"]):
        return "first_ticker_price_update"
    if any(key in text for key in ["PRE_SIGNAL_RUNTIME_SNAPSHOT", "SYMBOL_PIPELINE_HEALTH", "STATE_PRESENT", "FIRST5", "FIRST15", "OR_RANGE"]):
        return "candidate_state_creation"
    if any(key in text for key in ["SIGNAL_READY", "ENTRY_SIGNAL", "WOULD_EMIT_SIGNAL_READY"]):
        return "signal_evaluation"
    if any(key in text for key in ["RANK", "CANDIDATE"]):
        return "ranking_replacement"
    if any(key in text for key in ["BUY", "ORDER", "DISPATCH", "PLACEORDER", "RISK_GUARD", "ENTRY_BLOCKED"]):
        return "buy_decision"
    return "symbol_registration"


def _stage_flags_for_item(item: EvidenceItem) -> set[str]:
    text = f"{item.event_type} {item.source} {item.payload}".upper()
    flags = {item.stage} if item.stage in PIPELINE_STAGES else set()
    if any(key in text for key in ["TOP100_RELOAD", "TOP100_REFRESH", "TOP100_SUBSCRIPTION_RECONCILE", "WATCHLIST", "ENTRY_SYMBOLS", "SUBSCRIBED_TOP100"]):
        flags.add("runtime_top100_watchlist_loading")
    if any(key in text for key in ["CONTRACT", "CONID", "QUALIFY", "QUALIFIED"]):
        flags.add("contract_qualification_request")
    if any(key in text for key in ["CONID", "CONTRACT_METADATA", "CONTRACT_RESOLVED", "QUALIFIED"]):
        flags.add("contract_resolution")
    if any(key in text for key in ["REQMKTDATA", "SUBSCRIBE", "SUBSCRIPTION", "MARKET_DATA_SUBSCRIPTION"]):
        flags.add("market_data_subscription_request")
    if any(key in text for key in ["SUBSCRIBED", "TICKER", "PRICE", "CANDLE", "QUOTE"]):
        flags.add("subscription_acknowledgement")
    if any(key in text for key in ["TICKER", "PRICE", "CANDLE", "QUOTE", "USABLE_PRICE"]):
        flags.add("first_ticker_price_update")
    if any(key in text for key in ["PRE_SIGNAL_RUNTIME_SNAPSHOT", "SYMBOLSTATE", "STATE_PRESENT", "FIRST_PRICE_INITIALIZED", "READY_SINCE"]):
        flags.add("candidate_state_creation")
    if any(key in text for key in ["FIRST5", "FIRST_5M", "FIRST15", "FIRST_15M", "OR_RANGE", "OPENING_RANGE"]):
        flags.add("first5_first15_or_state")
    if any(key in text for key in ["SIGNAL_READY", "ENTRY_SIGNAL", "WOULD_EMIT_SIGNAL_READY"]):
        flags.add("signal_evaluation")
    if any(key in text for key in ["RANKING_POSITION", "GLOBAL_RANK", "CANDIDATE_RANK", "LIVE_ENTRY_RANK", "READY_CANDIDATE"]):
        flags.add("ranking_replacement")
    if any(key in text for key in ["ENTRY_ORDER_DISPATCH", "BUY_ORDER_SENT", "PAPER BUY", "PLACEORDER", "BUY_BLOCKED", "ENTRY_BLOCKED", "RISK_GUARD"]):
        flags.add("buy_decision")
    if flags - {"symbol_registration"}:
        flags.add("symbol_registration")
    return flags


def _iter_recorder_rows(recorder_dir: Path, session_date: str, symbol: str) -> list[EvidenceItem]:
    root = recorder_dir / session_date
    out: list[EvidenceItem] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() == ".csv":
            try:
                with path.open(newline="", errors="replace") as f:
                    for row in csv.DictReader(f):
                        if not row_matches_session(row, session_date):
                            continue
                        if _row_symbol(row) != symbol:
                            continue
                        et = _event_type(row, path.name)
                        payload = _payload(row)
                        out.append(EvidenceItem(_row_time(row), f"recorder:{path.name}", _stage_for_event(et, path.name, payload), et, symbol, payload))
            except Exception as exc:
                out.append(EvidenceItem(None, f"recorder:{path.name}", "RECORDER_READ_ERROR", "RECORDER_READ_ERROR", symbol, repr(exc)))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except Exception as exc:
                out.append(EvidenceItem(None, f"recorder:{path.name}", "RECORDER_READ_ERROR", "RECORDER_READ_ERROR", symbol, repr(exc)))
                continue
            for line in lines:
                if symbol not in line.upper():
                    continue
                row = _read_json(line)
                if row and not row_matches_session(row, session_date):
                    continue
                if row and _row_symbol(row) not in {"", symbol}:
                    continue
                et = _event_type(row, path.name) if row else path.name.upper()
                payload = _payload(row) if row else line[:1000]
                out.append(EvidenceItem(_row_time(row) if row else None, f"recorder:{path.name}", _stage_for_event(et, path.name, payload), et, symbol, payload))
    return out


def _iter_journal_rows(log_dir: Path, session_date: str, symbol: str) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    paths = [log_dir / f"trading-bot-{session_date}.log", Path(f"data/analysis/journal_v67_{session_date}_1320_1530_utc.log")]
    for path in paths:
        if not path.exists():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            upper = line.upper()
            if symbol not in upper:
                continue
            ts = parse_dt(line[:35].strip(" []"))
            if ts is not None and str(ts.date()) != session_date:
                continue
            et = next((token for token in ["SIGNAL_READY", "ENTRY_SIGNAL", "PRE_SIGNAL_RUNTIME_SNAPSHOT", "TOP100_RELOAD", "SUBSCRIBED", "CONTRACT", "NO_USABLE_TICKER_PRICE", "BUY", "ORDER", "RISK_GUARD"] if token in upper), "JOURNAL")
            out.append(EvidenceItem(ts, f"journal:{path.name}", _stage_for_event(et, path.name, line), et, symbol, line[:1000]))
    return out


def _sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    try:
        return [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    except Exception:
        return []


def _iter_sqlite_rows(sqlite_path: Path, session_date: str, symbol: str) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    if not sqlite_path.exists():
        return out
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        for table in _sqlite_tables(conn):
            cols = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if not cols:
                continue
            where = []
            params: list[Any] = []
            if "symbol" in cols:
                where.append("UPPER(symbol)=?")
                params.append(symbol)
            raw_cols = [c for c in cols if c in {"raw_json", "payload", "details", "message"}]
            for col in raw_cols:
                where.append(f"{col} LIKE ?")
                params.append(f"%{symbol}%")
            if not where:
                continue
            session_filters = []
            for col in ["session_date", "trading_session_date", "trade_session_date"]:
                if col in cols:
                    session_filters.append(f"({col} IS NULL OR {col}='' OR {col}=?)")
                    params.append(session_date)
            sql = f"SELECT * FROM {table} WHERE ({' OR '.join(where)})"
            if session_filters:
                sql += " AND " + " AND ".join(session_filters)
            try:
                rows = conn.execute(sql, params).fetchall()
            except Exception:
                continue
            for row in rows:
                d = dict(row)
                if not row_matches_session(d, session_date):
                    continue
                if _row_symbol(d) not in {"", symbol}:
                    continue
                et = _event_type(d, table)
                payload = _payload(d)
                out.append(EvidenceItem(_row_time(d), f"sqlite:{table}", _stage_for_event(et, table, payload), et, symbol, payload))
    finally:
        conn.close()
    return out


def load_case(cases_csv: Path, symbol: str) -> dict[str, Any]:
    df = safe_read_csv(cases_csv)
    if df.empty or "symbol" not in df.columns:
        return {}
    subset = df[df["symbol"].map(normalize_symbol) == symbol]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def top100_row(top100_path: Path, symbol: str) -> dict[str, Any]:
    df = load_top100(top100_path)
    if df.empty or "symbol" not in df.columns:
        return {}
    subset = df[df["symbol"].map(normalize_symbol) == symbol]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def stage_presence(evidence: list[EvidenceItem]) -> dict[str, bool]:
    stages = {stage: False for stage in PIPELINE_STAGES}
    for item in evidence:
        flags = _stage_flags_for_item(item)
        for stage in flags:
            stages[stage] = True
        if "candidate_state_creation" in flags:
            stages["symbol_registration"] = True
        if "first_ticker_price_update" in flags:
            stages["market_data_subscription_request"] = True
            stages["subscription_acknowledgement"] = True
    return stages


def _text(item: EvidenceItem) -> str:
    return f"{item.event_type} {item.source} {item.payload}".upper()


def runtime_reach_evidence(evidence: list[EvidenceItem]) -> EvidenceItem | None:
    for stage in PIPELINE_STAGES[1:]:
        for item in evidence:
            if stage in _stage_flags_for_item(item):
                return item
    return evidence[0] if evidence else None


def _positive_contract_absence(evidence: list[EvidenceItem]) -> bool:
    tokens = ["CONTRACT_PRESENT=0", "CONTRACT_PRESENT\": 0", "CONTRACT_PRESENT\": FALSE", "NO_CONTRACT", "CONTRACT_MISSING", "MISSING_CONTRACT", "CONTRACT_REQUEST_NOT_SENT"]
    return any(any(token in _text(item) for token in tokens) for item in evidence)


def _positive_subscription_absence(evidence: list[EvidenceItem]) -> bool:
    tokens = ["TICKER_PRESENT=0", "TICKER_PRESENT\": 0", "SUBSCRIPTION_PRESENT=0", "SUBSCRIPTION_PRESENT\": 0", "MISSING_SUBSCRIPTION", "NO_SUBSCRIPTION", "SUBSCRIPTION_REQUEST_NOT_SENT"]
    return any(any(token in _text(item) for token in tokens) for item in evidence)


def contract_telemetry_assessment(evidence: list[EvidenceItem]) -> str:
    stages = stage_presence(evidence)
    if any(any(token in _text(item) for token in ["CONTRACT_FAILED", "NOT_QUALIFIED", "NO SECURITY DEFINITION"]) for item in evidence):
        return "CONTRACT_REQUEST_FAILED"
    if stages["contract_resolution"]:
        if any(any(token in _text(item) for token in ["CACHE", "CONTRACT_CACHE", "CONTRACT_PRESENT=1", "CONTRACT_PRESENT\": 1", "CONID"]) for item in evidence):
            return "CONTRACT_RESOLVED_FROM_CACHE" if not stages["contract_qualification_request"] else "CONTRACT_REQUEST_RECORDED"
        return "CONTRACT_REQUEST_RECORDED"
    if stages["contract_qualification_request"]:
        return "CONTRACT_REQUEST_RECORDED"
    if _positive_contract_absence(evidence):
        return "CONTRACT_REQUEST_NOT_SENT"
    if stages["market_data_subscription_request"] or stages["first_ticker_price_update"] or stages["candidate_state_creation"]:
        return "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED"
    return "INSUFFICIENT_TELEMETRY"


def subscription_telemetry_assessment(evidence: list[EvidenceItem]) -> str:
    stages = stage_presence(evidence)
    if any(any(token in _text(item) for token in ["SUBSCRIBE_ERROR", "DELAYED", "PERMISSION", "NO MARKET DATA"]) for item in evidence):
        return "MARKET_DATA_SUBSCRIPTION_FAILED"
    if stages["market_data_subscription_request"]:
        return "SUBSCRIPTION_REQUEST_RECORDED"
    if _positive_subscription_absence(evidence):
        return "SUBSCRIPTION_REQUEST_NOT_SENT"
    if stages["first_ticker_price_update"] or stages["candidate_state_creation"]:
        return "SUBSCRIPTION_REQUEST_SENT_BUT_NOT_RECORDED"
    return "INSUFFICIENT_TELEMETRY"


def classify_root_cause(top_row: dict[str, Any], evidence: list[EvidenceItem], replay_ready: bool) -> tuple[str, str, str]:
    stages = stage_presence(evidence)
    contract_assessment = contract_telemetry_assessment(evidence)
    subscription_assessment = subscription_telemetry_assessment(evidence)
    if not top_row:
        return "TOP100_NOT_LOADED_RUNTIME", "daily Top100 row missing", "high"
    if not evidence:
        return "DATA_RETENTION_PREVENTS_ROOT_CAUSE", "no symbol-specific runtime/recorder/SQLite/journal evidence retained for this session", "medium"
    if not stages["runtime_top100_watchlist_loading"] and not stages["symbol_registration"]:
        return "WATCHLIST_REGISTRATION_MISSING", "Top100 file contains symbol but runtime watchlist/state evidence is missing", "medium"
    contract_failed = any("FAILED" in item.event_type or "NOT_QUALIFIED" in item.payload.upper() for item in evidence if "CONTRACT" in item.payload.upper() or "CONTRACT" in item.event_type)
    if contract_failed:
        return "CONTRACT_REQUEST_FAILED", "contract qualification failure evidence found", "high"
    if contract_assessment == "CONTRACT_REQUEST_NOT_SENT":
        return "CONTRACT_REQUEST_NOT_SENT", "positive telemetry says no contract was present/requested for this symbol", "high"
    if contract_assessment == "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED":
        return "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED", "downstream runtime evidence exists without a recorded contract request, consistent with cache or missing contract telemetry", "low"
    if contract_assessment == "INSUFFICIENT_TELEMETRY":
        return "INSUFFICIENT_TELEMETRY", "symbol reached runtime evidence, but contract request/cache telemetry is not comprehensive enough to prove whether a request was sent", "low"
    if subscription_assessment == "SUBSCRIPTION_REQUEST_NOT_SENT":
        return "SUBSCRIPTION_REQUEST_NOT_SENT", "positive telemetry says ticker/subscription was absent for this symbol", "high"
    sub_failed = any(any(token in item.payload.upper() for token in ["SUBSCRIBE_ERROR", "DELAYED", "PERMISSION", "NO MARKET DATA"]) for item in evidence)
    if sub_failed:
        return "MARKET_DATA_SUBSCRIPTION_FAILED", "subscription or market-data error evidence found", "high"
    if not stages["market_data_subscription_request"]:
        if subscription_assessment == "SUBSCRIPTION_REQUEST_SENT_BUT_NOT_RECORDED":
            return "SUBSCRIPTION_REQUEST_SENT_BUT_NOT_RECORDED", "ticker/state evidence exists without a recorded reqMktData call", "low"
        return "INSUFFICIENT_TELEMETRY", "contract evidence exists, but market-data request telemetry is not comprehensive enough to prove whether reqMktData was sent", "low"
    if not stages["first_ticker_price_update"]:
        return "NO_TICKER_RECEIVED", "subscription evidence exists but no ticker/price/candle evidence found", "medium"
    if not stages["candidate_state_creation"]:
        return "SYMBOL_STATE_NOT_CREATED", "price evidence exists but no SymbolState/pre-signal evidence found", "medium"
    if replay_ready and not stages["signal_evaluation"]:
        return "ACTUAL_RUNTIME_PROCESSING_BUG", "offline live-equivalent replay is ready but runtime signal evaluation is absent", "medium"
    return "RECORDER_EVIDENCE_MISSING", "evidence is partial and does not prove a later pipeline stage", "low"


def timeline_rows(session_date: str, symbol: str, top_row: dict[str, Any], evidence: list[EvidenceItem], root: str, reason: str) -> list[dict[str, Any]]:
    present = stage_presence(evidence)
    rows = []
    rows.append({"timestamp": "", "pipeline_stage": "daily_top100_inclusion", "expected_event": "symbol present in dated Top100", "actual_evidence_found": int(bool(top_row)), "missing_evidence": "" if top_row else "daily Top100 row", "first_confirmed_divergence": "", "root_cause": root, "confidence": ""})
    for stage in PIPELINE_STAGES[1:]:
        matching = [item for item in evidence if stage in _stage_flags_for_item(item)]
        rows.append({
            "timestamp": iso_ts(matching[0].timestamp) if matching else "",
            "pipeline_stage": stage,
            "expected_event": stage,
            "actual_evidence_found": len(matching),
            "missing_evidence": "" if matching else stage,
            "first_confirmed_divergence": root if not matching and not any(row.get("first_confirmed_divergence") for row in rows) else "",
            "root_cause": root,
            "confidence": reason if not matching else "",
        })
    return rows


def evidence_sources(evidence: list[EvidenceItem]) -> str:
    return ",".join(sorted({item.source for item in evidence}))


def _has_any(evidence: list[EvidenceItem], tokens: list[str]) -> bool:
    upper_tokens = [token.upper() for token in tokens]
    return any(any(token in f"{item.event_type} {item.payload}".upper() for token in upper_tokens) for item in evidence)


def run_symbol(args: argparse.Namespace, symbol: str) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)
    top_path = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    cases_path = args.cases_csv or Path(f"data/analysis/should_have_signaled_cases_{args.date}.csv")
    top = top100_row(top_path, symbol)
    case = load_case(cases_path, symbol)
    evidence = []
    evidence.extend(_iter_sqlite_rows(args.sqlite_path, args.date, symbol))
    evidence.extend(_iter_recorder_rows(args.recorder_dir, args.date, symbol))
    evidence.extend(_iter_journal_rows(args.log_dir, args.date, symbol))
    evidence.sort(key=lambda item: item.timestamp or pd.Timestamp.max.tz_localize("UTC"))
    candles = load_session_candles(args.history_dir, symbol, args.date, "RTH")
    replay = live_signal_replay(candles)
    replay_ready = replay.possible_signal_time is not None
    root, reason, confidence = classify_root_cause(top, evidence, replay_ready)
    present = stage_presence(evidence)
    missed_real = "yes" if replay_ready else "no"
    eligible = "yes" if replay_ready else "no"
    required_fix = {
        "DATA_RETENTION_PREVENTS_ROOT_CAUSE": "retain symbol-specific TOP100/contract/subscription/ticker/pre-signal evidence for every Top100 symbol",
        "WATCHLIST_REGISTRATION_MISSING": "add/assert runtime watchlist registration log and fail if Top100 symbol is absent from runtime entry_symbols",
        "CONTRACT_REQUEST_MISSING": "log and validate contract qualification request for every Top100 symbol",
        "MARKET_DATA_SUBSCRIPTION_MISSING": "log reqMktData per symbol and reconcile desired vs active subscriptions",
        "NO_TICKER_RECEIVED": "add subscription/ticker watchdog and retry or mark symbol blocked with reason",
        "SYMBOL_STATE_NOT_CREATED": "assert SymbolState creation after first usable ticker",
        "ACTUAL_RUNTIME_PROCESSING_BUG": "instrument/evaluate pre-signal state for symbol at replay signal time",
        "CONTRACT_REQUEST_NOT_SENT": "add an invariant test that every runtime Top100 symbol has contract_present=1 or a recorded contract failure before market-data subscription",
        "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED": "record contract cache hits and qualifyContracts calls/results for every Top100 symbol",
        "CONTRACT_RESOLVED_FROM_CACHE": "record cache-hit source and conId in symbol-specific contract telemetry",
        "CONTRACT_REQUEST_FAILED": "record contract failure reason and keep symbol-specific failure evidence in recorder/SQLite",
        "SUBSCRIPTION_REQUEST_NOT_SENT": "add an invariant test that every resolved contract produces reqMktData or a symbol-specific skip reason",
        "SUBSCRIPTION_REQUEST_SENT_BUT_NOT_RECORDED": "record reqMktData calls and active ticker map reconciliation per symbol",
        "INSUFFICIENT_TELEMETRY": "add comprehensive symbol-specific Top100 watchlist, contract cache/request/result, reqMktData, ticker-map, and SymbolState telemetry",
    }.get(root, "investigate with retained symbol-specific runtime evidence")
    basis = runtime_reach_evidence(evidence)
    contract_assessment = contract_telemetry_assessment(evidence)
    subscription_assessment = subscription_telemetry_assessment(evidence)
    summary = {
        "date": args.date,
        "symbol": symbol,
        "top100_rank": top.get("top100_rank", case.get("top100_rank", "")),
        "top100_score": top.get("top100_score", case.get("top100_score", "")),
        "offline_possible_signal_time": case.get("possible_signal_time", ""),
        "live_equivalent_signal_time": iso_ts(replay.possible_signal_time),
        "present_in_daily_top100": int(bool(top)),
        "present_in_runtime_top100_state": int(present["runtime_top100_watchlist_loading"]),
        "contract_request_seen": int(present["contract_qualification_request"]),
        "contract_resolved_seen": int(present["contract_resolution"]),
        "market_data_subscription_seen": int(present["market_data_subscription_request"]),
        "ticker_seen": int(present["first_ticker_price_update"]),
        "state_seen": int(present["candidate_state_creation"]),
        "signal_evaluation_seen": int(present["signal_evaluation"]),
        "buy_decision_seen": int(present["buy_decision"]),
        "first_confirmed_divergence_stage": next((stage for stage, ok in present.items() if not ok and stage != "daily_top100_inclusion"), ""),
        "final_root_cause": root,
        "confidence": confidence,
        "missed_trade_was_real": missed_real,
        "live_would_have_been_eligible_at_offline_signal_time": eligible,
        "exact_reason_missed": reason,
        "runtime_evidence_basis": basis.event_type if basis else "",
        "runtime_evidence_source": basis.source if basis else "",
        "runtime_evidence_payload_excerpt": basis.payload[:300] if basis else "",
        "contract_telemetry_assessment": contract_assessment,
        "subscription_telemetry_assessment": subscription_assessment,
        "could_contract_have_been_cached": int(contract_assessment in {"CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED", "CONTRACT_RESOLVED_FROM_CACHE"}),
        "required_fix": required_fix,
        "regression_test_needed": int(root in {"WATCHLIST_REGISTRATION_MISSING", "CONTRACT_REQUEST_NOT_SENT", "CONTRACT_REQUEST_SENT_BUT_NOT_RECORDED", "CONTRACT_REQUEST_FAILED", "SUBSCRIPTION_REQUEST_NOT_SENT", "SUBSCRIPTION_REQUEST_SENT_BUT_NOT_RECORDED", "MARKET_DATA_SUBSCRIPTION_FAILED", "NO_TICKER_RECEIVED", "SYMBOL_STATE_NOT_CREATED", "ACTUAL_RUNTIME_PROCESSING_BUG", "INSUFFICIENT_TELEMETRY"}),
        "another_symbol_occupied_subscription_slot": "unknown" if not evidence else "not_observed",
        "reconnect_or_rollover_evidence": int(_has_any(evidence, ["RECONNECT", "RECORDER_SESSION_ROTATED", "SESSION_BOUNDARY", "ROLLOVER"])),
        "could_recur": "yes" if root in {"WATCHLIST_REGISTRATION_MISSING", "CONTRACT_REQUEST_MISSING", "MARKET_DATA_SUBSCRIPTION_MISSING", "NO_TICKER_RECEIVED", "SYMBOL_STATE_NOT_CREATED", "DATA_RETENTION_PREVENTS_ROOT_CAUSE", "ACTUAL_RUNTIME_PROCESSING_BUG"} else "uncertain",
        "evidence_sources": evidence_sources(evidence),
    }
    write_symbol_markdown(args.output_dir / f"shs_root_cause_{symbol}_{args.date}.md", summary, top, case, evidence, timeline_rows(args.date, symbol, top, evidence, root, reason))
    write_symbol_timeline(args.output_dir / f"shs_root_cause_{symbol}_{args.date}_timeline.csv", timeline_rows(args.date, symbol, top, evidence, root, reason))
    return summary


def write_symbol_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "pipeline_stage", "expected_event", "actual_evidence_found", "missing_evidence", "first_confirmed_divergence", "root_cause", "confidence"])
        writer.writeheader()
        writer.writerows(rows)


def write_symbol_markdown(path: Path, summary: dict[str, Any], top: dict[str, Any], case: dict[str, Any], evidence: list[EvidenceItem], timeline: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# SHS Root Cause {summary['symbol']} {summary['date']}",
        "",
        f"Final classification: `{summary['final_root_cause']}`",
        f"Confidence: `{summary['confidence']}`",
        f"Root cause: {summary['exact_reason_missed']}",
        "",
        "## Facts",
        f"- Daily Top100 present: {summary['present_in_daily_top100']} rank={summary.get('top100_rank')} score={summary.get('top100_score')}",
        f"- Present in live Top100 runtime state: {summary['present_in_runtime_top100_state']}",
        f"- IBKR contract request seen: {summary['contract_request_seen']}",
        f"- Contract resolved seen: {summary['contract_resolved_seen']}",
        f"- Market data subscription seen: {summary['market_data_subscription_seen']}",
        f"- Ticker/price seen: {summary['ticker_seen']}",
        f"- Symbol state seen: {summary['state_seen']}",
        f"- Signal evaluation seen: {summary['signal_evaluation_seen']}",
        f"- Buy decision seen: {summary['buy_decision_seen']}",
        f"- Runtime evidence basis: {summary['runtime_evidence_basis']} from {summary['runtime_evidence_source']}",
        f"- Runtime evidence payload excerpt: {summary['runtime_evidence_payload_excerpt']}",
        f"- Contract telemetry assessment: {summary['contract_telemetry_assessment']}",
        f"- Subscription telemetry assessment: {summary['subscription_telemetry_assessment']}",
        f"- Could a cached contract have been used without a new qualification request: {summary['could_contract_have_been_cached']}",
        f"- Missed trade was real: {summary['missed_trade_was_real']}",
        f"- Live would have been eligible at offline signal time: {summary['live_would_have_been_eligible_at_offline_signal_time']}",
        f"- Another symbol occupied intended subscription slot: {summary['another_symbol_occupied_subscription_slot']}",
        f"- Reconnect/rollover evidence for this symbol: {summary['reconnect_or_rollover_evidence']}",
        f"- Could recur on following sessions: {summary['could_recur']}",
        "",
        "## Required Fix",
        summary["required_fix"],
        "",
        "## Timeline",
    ]
    for row in timeline:
        lines.append(f"- {row['pipeline_stage']}: evidence={row['actual_evidence_found']} missing={row['missing_evidence']} divergence={row['first_confirmed_divergence']}")
    lines.extend(["", "## Raw Evidence Samples"])
    for item in evidence[:50]:
        lines.append(f"- {iso_ts(item.timestamp)} `{item.source}` `{item.event_type}` stage={item.stage}: {item.payload[:500]}")
    if not evidence:
        lines.append("- No symbol-specific runtime/recorder/SQLite/journal evidence found for this session.")
    path.write_text("\n".join(lines) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only SHS pipeline root-cause investigator for runtime-never-processed symbols.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", default="NUAI,IREN")
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--cases-csv", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = [normalize_symbol(value) for value in str(args.symbols).split(",") if normalize_symbol(value)]
    rows = [run_symbol(args, symbol) for symbol in symbols]
    out = args.output_dir / f"shs_root_cause_{args.date}.csv"
    write_summary_csv(out, rows)
    print(f"SHS_PIPELINE_ROOT_CAUSE_DONE date={args.date} symbols={len(rows)} output={out}", flush=True)
    for row in rows:
        print(f"{row['symbol']} final_root_cause={row['final_root_cause']} confidence={row['confidence']} reason={row['exact_reason_missed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
