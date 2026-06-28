from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    first_existing_column,
    fnum,
    iso_ts,
    load_top100,
    normalize_symbol,
    parse_dt,
    parse_raw_json,
    read_sql_table,
    safe_read_csv,
)


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")

OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "top100_rank",
    "top100_score",
    "open_to_high_pct",
    "possible_signal_time",
    "candle_signal_ready",
    "runtime_signal_seen",
    "runtime_buy_attempt_seen",
    "runtime_buy_blocked_seen",
    "runtime_order_rejected_seen",
    "runtime_trade_seen",
    "runtime_fill_seen",
    "runtime_events_count",
    "risk_events_count",
    "orders_count",
    "trades_count",
    "executions_count",
    "nearest_runtime_event_time",
    "nearest_runtime_event_type",
    "nearest_runtime_event_reason",
    "block_reason",
    "final_replay_reason",
    "timeline_json",
]

EVENT_KEYWORDS = [
    "SIGNAL_READY",
    "BUY_BLOCKED",
    "RISK_GUARD_BLOCK_ENTRY",
    "PAPER BUY",
    "PAPER_BUY",
    "BUY_ORDER_SENT",
    "ORDER_SUBMITTED",
    "ORDER_REJECTED",
    "ORDER_CANCELLED",
    "RESTART_BLOCK",
    "TOP100_BLOCK",
    "MAX_POSITIONS",
    "DAILY_LOSS",
    "EXPOSURE",
    "COOLDOWN",
    "STALE_CANDIDATE",
    "DUPLICATE_POSITION",
    "ORDER_FAILED",
    "MISSING_CONTRACT",
    "NO_MARKET_DATA",
    "INELIGIBLE",
    "DENIED_SYMBOL",
    "HALTED",
    "TRADING_DISABLED",
    "OUTSIDE_SESSION",
]


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.DataFrame()


def load_recorder_source(recorder_dir: Path, session_date: str, name: str) -> pd.DataFrame:
    root = recorder_dir / session_date
    for path in [root / name, root / f"{name}.csv", root / f"{name}.jsonl"]:
        if path.suffix == ".jsonl":
            df = load_jsonl(path)
        else:
            df = safe_read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def normalize_event_type(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def row_text(row: dict[str, Any]) -> str:
    raw = parse_raw_json(row.get("raw_json"))
    bits = [str(value) for value in row.values() if value not in (None, "")]
    bits.extend(str(value) for value in raw.values() if value not in (None, ""))
    return " ".join(bits)


def row_symbol(row: dict[str, Any]) -> str:
    raw = parse_raw_json(row.get("raw_json"))
    return normalize_symbol(
        first_existing_column(row, ["symbol", "contract_symbol", "ticker"])
        or raw.get("symbol")
        or raw.get("ticker")
        or raw.get("contract_symbol")
    )


def row_time(row: dict[str, Any]) -> pd.Timestamp | None:
    raw = parse_raw_json(row.get("raw_json"))
    return parse_dt(
        first_existing_column(row, ["event_time", "timestamp", "time", "created_at", "updated_at", "recorded_at", "executed_at", "submitted_at"])
        or raw.get("event_time")
        or raw.get("timestamp")
        or raw.get("time")
        or raw.get("submitted_at")
        or raw.get("executed_at")
    )


def row_event_type(row: dict[str, Any], default: str = "") -> str:
    raw = parse_raw_json(row.get("raw_json"))
    return normalize_event_type(
        first_existing_column(row, ["event_type", "event", "status", "action", "order_status", "state"])
        or raw.get("event_type")
        or raw.get("event")
        or raw.get("status")
        or default
    )


def row_reason(row: dict[str, Any]) -> str:
    raw = parse_raw_json(row.get("raw_json"))
    value = (
        first_existing_column(row, ["reason", "blocked_reason", "reject_reason", "error", "message", "status_reason"])
        or raw.get("reason")
        or raw.get("blocked_reason")
        or raw.get("reject_reason")
        or raw.get("error")
        or raw.get("message")
    )
    return str(value or "")


def within_window(ts: pd.Timestamp | None, center: pd.Timestamp | None, minutes: int = 15) -> bool:
    if ts is None:
        return True if center is None else False
    if center is None:
        return True
    return center - pd.Timedelta(minutes=minutes) <= ts <= center + pd.Timedelta(minutes=minutes)


def compact_event(source: str, row: dict[str, Any], *, default_event: str = "", symbol: str = "") -> dict[str, Any]:
    ts = row_time(row)
    event = row_event_type(row, default_event)
    reason = row_reason(row)
    return {
        "time": iso_ts(ts),
        "source": source,
        "event": event or default_event,
        "reason": reason,
        "details": row_text(row)[:300],
        "symbol": row_symbol(row) or symbol,
    }


def merge_timeline_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(event: dict[str, Any]) -> tuple[pd.Timestamp, str, str]:
        ts = parse_dt(event.get("time")) or pd.Timestamp.max.tz_localize("UTC")
        return ts, str(event.get("source") or ""), str(event.get("event") or "")

    return sorted(events, key=key)


def filter_should_have_signaled_targets(missed: pd.DataFrame) -> pd.DataFrame:
    if missed.empty:
        return pd.DataFrame()
    required = {"source_bucket", "was_bought", "top100_no_signal_reason"}
    if not required.issubset(set(missed.columns)):
        return pd.DataFrame()
    out = missed[
        (missed["source_bucket"].fillna("").astype(str) == "top100")
        & (pd.to_numeric(missed["was_bought"], errors="coerce").fillna(0) == 0)
        & (missed["top100_no_signal_reason"].fillna("").astype(str) == "should_have_signaled")
    ].copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    return out


def read_sqlite_sources(sqlite_path: Path, session_date: str) -> dict[str, pd.DataFrame]:
    tables: dict[str, tuple[str, list[Any]]] = {
        "runtime_events": ("session_date = ? OR substr(event_time, 1, 10) = ?", [session_date, session_date]),
        "risk_events": ("session_date = ? OR substr(event_time, 1, 10) = ?", [session_date, session_date]),
        "orders": ("session_date = ? OR substr(updated_at, 1, 10) = ? OR substr(created_at, 1, 10) = ?", [session_date, session_date, session_date]),
        "trades": ("session_date = ? OR substr(entry_fill_time, 1, 10) = ? OR substr(exit_fill_time, 1, 10) = ? OR substr(closed_at, 1, 10) = ?", [session_date, session_date, session_date, session_date]),
        "executions": ("session_date = ? OR substr(executed_at, 1, 10) = ? OR substr(recorded_at, 1, 10) = ?", [session_date, session_date, session_date]),
    }
    out: dict[str, pd.DataFrame] = {}
    for table, (where, params) in tables.items():
        try:
            out[table] = read_sql_table(sqlite_path, table, where=where, params=params)
        except Exception:
            out[table] = pd.DataFrame()
    return out


def symbol_rows(df: pd.DataFrame, symbol: str, center: pd.Timestamp | None = None, *, window_minutes: int = 15) -> pd.DataFrame:
    if df.empty:
        return df
    records = []
    needle = normalize_symbol(symbol)
    for row in df.to_dict("records"):
        text = row_text(row).upper()
        if row_symbol(row) != needle and needle not in text:
            continue
        if not within_window(row_time(row), center, window_minutes):
            continue
        records.append(row)
    return pd.DataFrame(records)


def recorder_events(recorder_dir: Path, session_date: str, symbol: str, center: pd.Timestamp | None) -> dict[str, pd.DataFrame]:
    names = {
        "trade_lifecycle": "trade_lifecycle",
        "order_lifecycle": "order_lifecycle",
        "fills": "fills",
        "run_metadata": "run_metadata",
        "strategy_equity": "strategy_equity",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, name in names.items():
        out[key] = symbol_rows(load_recorder_source(recorder_dir, session_date, name), symbol, center)
    return out


def has_event(events: list[dict[str, Any]], keywords: list[str]) -> bool:
    joined = "\n".join(f"{event.get('event','')} {event.get('reason','')} {event.get('details','')}" for event in events).upper()
    return any(keyword.upper() in joined for keyword in keywords)


def first_matching_reason(events: list[dict[str, Any]], keywords: list[str]) -> str:
    for event in events:
        text = f"{event.get('event','')} {event.get('reason','')} {event.get('details','')}".upper()
        if any(keyword.upper() in text for keyword in keywords):
            return str(event.get("reason") or event.get("event") or "")
    return ""


def classify_replay_reason(events: list[dict[str, Any]], counts: dict[str, int]) -> str:
    if counts.get("trades_count", 0) or counts.get("executions_count", 0) or has_event(events, ["FILL", "POSITION_OPENED", "POSITION_CLOSED"]):
        return "bought_but_missing_in_missed_runner"
    if has_event(events, ["ORDER_REJECTED", "ORDER_CANCELLED", "ORDER_FAILED", "ERROR 201", "MISSING_CONTRACT"]):
        return "contract_or_order_error"
    if has_event(events, ["INELIGIBLE", "DENIED_SYMBOL", "NO_TRADING_PERMISSION", "KID"]):
        return "symbol_ineligible"
    if has_event(events, ["RISK_GUARD_BLOCK_ENTRY", "RISK_GUARD"]):
        return "risk_guard_blocked"
    if has_event(events, ["MAX_POSITIONS", "MAX_POSITION"]):
        return "max_positions_blocked"
    if has_event(events, ["RESTART_BLOCK", "RESTART_BLOCKED"]):
        return "restart_blocked"
    if has_event(events, ["TOP100_BLOCK"]):
        return "top100_blocked"
    if has_event(events, ["COOLDOWN"]):
        return "cooldown_blocked"
    if has_event(events, ["STALE_CANDIDATE", "CANDIDATE_AGE"]):
        return "stale_candidate"
    if has_event(events, ["DUPLICATE_POSITION"]):
        return "duplicate_position_blocked"
    if has_event(events, ["NO_MARKET_DATA", "MISSING_MARKET_DATA"]):
        return "no_market_data"
    if has_event(events, ["BUY_BLOCKED"]):
        return "max_positions_blocked" if has_event(events, ["MAX_POSITION"]) else "runtime_saw_symbol_no_signal"
    if has_event(events, ["PAPER_BUY", "PAPER BUY", "BUY_ORDER_SENT", "ORDER_SUBMITTED"]):
        return "buy_attempt_no_fill"
    if has_event(events, ["SIGNAL_READY"]):
        return "signal_ready_but_no_buy_attempt"
    runtime_seen = counts.get("runtime_events_count", 0) + counts.get("risk_events_count", 0) + counts.get("orders_count", 0)
    if runtime_seen > 0:
        return "runtime_saw_symbol_no_signal"
    return "no_runtime_evidence"


def nearest_runtime_event(events: list[dict[str, Any]], center: pd.Timestamp | None) -> dict[str, Any]:
    runtime = [event for event in events if event.get("source") in {"runtime_events", "risk_events", "orders", "trade_lifecycle", "order_lifecycle"}]
    if not runtime:
        return {}
    if center is None:
        return runtime[0]
    return min(runtime, key=lambda event: abs(((parse_dt(event.get("time")) or center) - center).total_seconds()))


def build_symbol_timeline(
    *,
    row: dict[str, Any],
    sqlite_sources: dict[str, pd.DataFrame],
    recorder_sources: dict[str, pd.DataFrame],
    center: pd.Timestamp | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    symbol = normalize_symbol(row.get("symbol"))
    events: list[dict[str, Any]] = []
    if center is not None:
        events.append({
            "time": iso_ts(center),
            "source": "candle",
            "event": "possible_signal",
            "reason": "",
            "details": f"candle_signal_ready open_to_high_pct={row.get('open_to_high_pct')}",
            "symbol": symbol,
        })
    counts: dict[str, int] = {}
    for source, df in sqlite_sources.items():
        rows = symbol_rows(df, symbol, center)
        counts[f"{source}_count"] = len(rows)
        default_event = source.upper()
        for item in rows.to_dict("records"):
            events.append(compact_event(source, item, default_event=default_event, symbol=symbol))
    for source, df in recorder_sources.items():
        counts[f"{source}_count"] = len(df)
        for item in df.to_dict("records"):
            events.append(compact_event(source, item, default_event=source.upper(), symbol=symbol))
    return merge_timeline_events(events), counts


def load_targets(args: argparse.Namespace) -> pd.DataFrame:
    missed_path = args.missed_runners_csv or Path(f"data/analysis/missed_runners_{args.date}.csv")
    missed = safe_read_csv(missed_path)
    targets = filter_should_have_signaled_targets(missed)
    if targets.empty:
        return targets
    top100_path = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    top100 = load_top100(top100_path)
    if not top100.empty:
        targets = targets.merge(top100[["symbol", "top100_rank", "top100_score"]], on="symbol", how="left", suffixes=("", "_top100_file"))
        for col in ["top100_rank", "top100_score"]:
            alt = f"{col}_top100_file"
            if alt in targets.columns:
                targets[col] = targets[col].combine_first(targets[alt])
                targets = targets.drop(columns=[alt])
    return targets


def analyze_signal_replay(
    *,
    date: str,
    missed_runners_csv: Path | None,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    top100_path: Path | None,
    threshold_pct: float = 8.0,
) -> pd.DataFrame:
    args = argparse.Namespace(
        date=date,
        missed_runners_csv=missed_runners_csv,
        top100=top100_path,
        threshold_pct=threshold_pct,
    )
    targets = load_targets(args)
    if targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    sqlite_sources = read_sqlite_sources(sqlite_path, date)
    rows: list[dict[str, Any]] = []
    for target in targets.to_dict("records"):
        symbol = normalize_symbol(target.get("symbol"))
        center = parse_dt(target.get("possible_signal_time") or target.get("opening_range_break_time") or target.get("first_time_above_8pct"))
        rec_sources = recorder_events(recorder_dir, date, symbol, center)
        timeline, raw_counts = build_symbol_timeline(row=target, sqlite_sources=sqlite_sources, recorder_sources=rec_sources, center=center)
        counts = {
            "runtime_events_count": raw_counts.get("runtime_events_count", 0),
            "risk_events_count": raw_counts.get("risk_events_count", 0),
            "orders_count": raw_counts.get("orders_count", 0) + raw_counts.get("order_lifecycle_count", 0),
            "trades_count": raw_counts.get("trades_count", 0) + raw_counts.get("trade_lifecycle_count", 0),
            "executions_count": raw_counts.get("executions_count", 0) + raw_counts.get("fills_count", 0),
        }
        reason = classify_replay_reason(timeline, counts)
        nearest = nearest_runtime_event(timeline, center)
        block_reason = first_matching_reason(timeline, ["BUY_BLOCKED", "RISK_GUARD", "MAX_POSITION", "RESTART_BLOCK", "TOP100_BLOCK", "COOLDOWN", "STALE", "NO_MARKET_DATA", "INELIGIBLE"])
        rows.append({
            "date": date,
            "symbol": symbol,
            "top100_rank": target.get("top100_rank"),
            "top100_score": target.get("top100_score"),
            "open_to_high_pct": target.get("open_to_high_pct"),
            "possible_signal_time": iso_ts(center),
            "candle_signal_ready": 1,
            "runtime_signal_seen": int(has_event(timeline, ["SIGNAL_READY"])),
            "runtime_buy_attempt_seen": int(has_event(timeline, ["PAPER_BUY", "PAPER BUY", "BUY_ORDER_SENT", "ORDER_SUBMITTED"])),
            "runtime_buy_blocked_seen": int(has_event(timeline, ["BUY_BLOCKED", "RISK_GUARD_BLOCK_ENTRY"])),
            "runtime_order_rejected_seen": int(has_event(timeline, ["ORDER_REJECTED", "ORDER_CANCELLED", "ORDER_FAILED"])),
            "runtime_trade_seen": int(counts["trades_count"] > 0),
            "runtime_fill_seen": int(counts["executions_count"] > 0),
            **counts,
            "nearest_runtime_event_time": nearest.get("time", ""),
            "nearest_runtime_event_type": nearest.get("event", ""),
            "nearest_runtime_event_reason": nearest.get("reason", ""),
            "block_reason": block_reason,
            "final_replay_reason": reason,
            "timeline_json": json.dumps(timeline, ensure_ascii=True),
        })
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"SIGNAL_REPLAY target_should_have_signaled_count={total}")
    if df.empty:
        return
    counts = Counter(df["final_replay_reason"].fillna("unknown").astype(str))
    print("final_replay_reason_counts=" + ", ".join(f"{key}:{value}" for key, value in counts.most_common()))
    print(f"runtime_never_saw_symbol={int((df['final_replay_reason'] == 'runtime_never_saw_symbol').sum())}")
    print(f"signal_ready_but_no_buy_attempt={int((df['final_replay_reason'] == 'signal_ready_but_no_buy_attempt').sum())}")
    blocked = df["final_replay_reason"].fillna("").astype(str).str.contains("blocked|stale_candidate|no_market_data|symbol_ineligible", regex=True)
    print(f"blocked_count={int(blocked.sum())}")
    print(f"buy_attempt_no_fill={int((df['final_replay_reason'] == 'buy_attempt_no_fill').sum())}")
    print(f"unknown={int((df['final_replay_reason'] == 'unknown').sum())}")
    print("top20_by_open_to_high_pct:")
    print(df.sort_values("open_to_high_pct", ascending=False)[["symbol", "open_to_high_pct", "top100_rank", "possible_signal_time", "final_replay_reason", "block_reason"]].head(20).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay should-have-signaled Top100 missed runners against runtime recorder and SQLite evidence.")
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD.")
    parser.add_argument("--threshold-pct", type=float, default=8.0)
    parser.add_argument("--missed-runners-csv", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or Path(f"data/analysis/signal_replay_{args.date}.csv")
    df = analyze_signal_replay(
        date=args.date,
        missed_runners_csv=args.missed_runners_csv,
        sqlite_path=args.sqlite_path,
        history_dir=args.history_dir,
        recorder_dir=args.recorder_dir,
        top100_path=args.top100,
        threshold_pct=args.threshold_pct,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print_summary(df)
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
