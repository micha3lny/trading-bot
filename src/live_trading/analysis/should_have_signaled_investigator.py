from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.live_trading.analysis.common import (
    first_existing_column,
    iso_ts,
    load_top100,
    normalize_symbol,
    parse_dt,
    read_sql_table,
    safe_read_csv,
)
from src.live_trading.analysis.signal_replay_analyzer import (
    build_symbol_timeline,
    filter_should_have_signaled_targets,
    has_event,
    load_recorder_source,
    read_sqlite_sources,
    row_symbol,
    row_text,
    row_time,
)
from src.live_trading.analysis.symbol_subscription_inspector import (
    line_time,
    parse_key_values,
    read_text_lines,
)


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")
DEFAULT_ANALYSIS_DIR = Path("data/analysis")

CASE_COLUMNS = [
    "date",
    "symbol",
    "top100_rank",
    "top100_score",
    "open_to_high_pct",
    "possible_signal_time",
    "first_time_above_5pct",
    "first_time_above_8pct",
    "opening_range_break_time",
    "was_bought",
    "buy_time",
    "buy_order_id",
    "runtime_evidence_found",
    "ready_candidate_seen",
    "signal_ready_seen",
    "buy_attempt_seen",
    "buy_fill_seen",
    "risk_guard_blocked",
    "max_positions_blocked",
    "restart_blocked",
    "top100_blocked",
    "spread_blocked",
    "price_missing_blocked",
    "subscription_missing_blocked",
    "stale_candidate",
    "candidate_replaced_by_higher_rank",
    "position_limit_reached",
    "already_open",
    "unknown_reason",
    "final_classification",
]

SUMMARY_CLASSIFICATIONS = [
    "bought_late",
    "runtime_signal_ready_but_no_buy",
    "candidate_replaced_by_higher_rank",
    "max_positions_blocked",
    "risk_guard_blocked",
    "restart_blocked",
    "top100_blocked",
    "spread_or_price_blocked",
    "subscription_missing",
    "runtime_never_processed_symbol",
    "unknown",
]

RECORDER_SOURCE_NAMES = {
    "trade_lifecycle": "trade_lifecycle",
    "order_lifecycle": "order_lifecycle",
    "fills": "fills",
    "run_metadata": "run_metadata",
    "strategy_equity": "strategy_equity",
}


@dataclass
class EvidenceBundle:
    sqlite_sources: dict[str, pd.DataFrame]
    recorder_sources: dict[str, pd.DataFrame]
    sqlite_by_symbol: dict[str, dict[str, pd.DataFrame]]
    recorder_by_symbol: dict[str, dict[str, pd.DataFrame]]
    journal_lines: list[str]
    heartbeat_states: list[tuple[pd.Timestamp, dict[str, str]]]


def iter_dates(start_date: str, end_date: str) -> Iterable[str]:
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)


def default_journal_path(session_date: str) -> Path:
    candidates = sorted(DEFAULT_ANALYSIS_DIR.glob(f"journal_v67_{session_date}*.log"))
    return candidates[0] if candidates else DEFAULT_ANALYSIS_DIR / f"journal_v67_{session_date}.log"


def load_targets_for_date(session_date: str, missed_path: Path | None, top100_path: Path | None) -> pd.DataFrame:
    path = missed_path or DEFAULT_ANALYSIS_DIR / f"missed_runners_{session_date}.csv"
    missed = safe_read_csv(path)
    targets = filter_should_have_signaled_targets(missed)
    if targets.empty:
        return pd.DataFrame()
    top100 = load_top100(top100_path or Path(f"data/universe/daily_top100_{session_date}.csv"))
    if not top100.empty:
        targets = targets.merge(
            top100[["symbol", "top100_rank", "top100_score"]],
            on="symbol",
            how="left",
            suffixes=("", "_top100_file"),
        )
        for col in ["top100_rank", "top100_score"]:
            alt = f"{col}_top100_file"
            if alt in targets.columns:
                targets[col] = targets[col].combine_first(targets[alt])
                targets = targets.drop(columns=[alt])
    return targets


def event_text(timeline: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{event.get('event', '')} {event.get('reason', '')} {event.get('details', '')}"
        for event in timeline
    ).upper()


def journal_text_for_symbol(lines: list[str], symbol: str, center: pd.Timestamp | None, minutes: int = 15) -> str:
    if not lines:
        return ""
    needle = normalize_symbol(symbol)
    out: list[str] = []
    start = center - pd.Timedelta(minutes=minutes) if center is not None else None
    end = center + pd.Timedelta(minutes=minutes) if center is not None else None
    for line in lines:
        upper = line.upper()
        if needle not in upper:
            continue
        ts = parse_dt(line[:32])
        if start is not None and ts is not None and not (start <= ts <= end):
            continue
        out.append(line)
    return "\n".join(out).upper()


def nearest_heartbeat_state_from_index(index: list[tuple[pd.Timestamp, dict[str, str]]], center: pd.Timestamp | None) -> dict[str, str]:
    if not index:
        return {}
    if center is None:
        return index[-1][1]
    return min(index, key=lambda item: abs((item[0] - center).total_seconds()))[1]


def load_recorder_sources(recorder_dir: Path, session_date: str) -> dict[str, pd.DataFrame]:
    return {
        key: load_recorder_source(recorder_dir, session_date, name)
        for key, name in RECORDER_SOURCE_NAMES.items()
    }


def rows_in_window(df: pd.DataFrame, center: pd.Timestamp | None, minutes: int = 30) -> pd.DataFrame:
    if df.empty or center is None:
        return df
    records = []
    start = center - pd.Timedelta(minutes=minutes)
    end = center + pd.Timedelta(minutes=minutes)
    for row in df.to_dict("records"):
        ts = row_time(row)
        if ts is None or start <= ts <= end:
            records.append(row)
    return pd.DataFrame(records)


def build_symbol_index(sources: dict[str, pd.DataFrame], target_symbols: set[str]) -> dict[str, dict[str, pd.DataFrame]]:
    indexed: dict[str, dict[str, list[dict[str, Any]]]] = {
        source: {symbol: [] for symbol in target_symbols}
        for source in sources
    }
    pattern = None
    if target_symbols:
        body = "|".join(re.escape(symbol) for symbol in sorted(target_symbols, key=len, reverse=True))
        pattern = re.compile(rf"(?<![A-Z0-9])({body})(?![A-Z0-9])")
    for source, df in sources.items():
        if df.empty:
            continue
        for row in df.to_dict("records"):
            symbol = row_symbol(row)
            text = row_text(row).upper()
            matched: set[str] = set()
            if symbol in target_symbols:
                matched.add(symbol)
            if pattern is not None:
                matched.update(match.group(1) for match in pattern.finditer(text))
            for target in matched:
                indexed[source][target].append(row)
    return {
        source: {
            symbol: pd.DataFrame(rows)
            for symbol, rows in symbol_map.items()
        }
        for source, symbol_map in indexed.items()
    }


def build_heartbeat_index(lines: list[str]) -> list[tuple[pd.Timestamp, dict[str, str]]]:
    out: list[tuple[pd.Timestamp, dict[str, str]]] = []
    for line in lines:
        if "heartbeat" not in line.lower():
            continue
        ts = line_time(line)
        if ts is not None:
            out.append((ts, parse_key_values(line)))
    return sorted(out, key=lambda item: item[0])


def load_evidence_bundle(
    *,
    sqlite_path: Path,
    recorder_dir: Path,
    session_date: str,
    journal_log: Path | None,
    target_symbols: set[str],
) -> EvidenceBundle:
    sqlite_sources = read_sqlite_sources(sqlite_path, session_date)
    recorder_sources = load_recorder_sources(recorder_dir, session_date)
    journal = read_text_lines(journal_log or default_journal_path(session_date))
    return EvidenceBundle(
        sqlite_sources=sqlite_sources,
        recorder_sources=recorder_sources,
        sqlite_by_symbol=build_symbol_index(sqlite_sources, target_symbols),
        recorder_by_symbol=build_symbol_index(recorder_sources, target_symbols),
        journal_lines=journal,
        heartbeat_states=build_heartbeat_index(journal),
    )


def sources_for_symbol(
    indexed_sources: dict[str, dict[str, pd.DataFrame]],
    symbol: str,
    center: pd.Timestamp | None,
    window_minutes: int = 30,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    symbol = normalize_symbol(symbol)
    for source, symbol_map in indexed_sources.items():
        out[source] = rows_in_window(symbol_map.get(symbol, pd.DataFrame()), center, window_minutes)
    return out


def bool_text(text: str, *needles: str) -> int:
    return int(any(needle.upper() in text for needle in needles))


def buy_details(symbol: str, sqlite_sources: dict[str, pd.DataFrame], recorder_sources: dict[str, pd.DataFrame]) -> tuple[str, str]:
    symbol = normalize_symbol(symbol)
    candidates: list[dict[str, Any]] = []
    for name in ("executions", "fills", "orders", "order_lifecycle"):
        df = sqlite_sources.get(name, pd.DataFrame())
        if df.empty:
            df = recorder_sources.get(name, pd.DataFrame())
        if df.empty:
            continue
        for row in df.to_dict("records"):
            text = json.dumps(row, default=str).upper()
            if symbol not in text:
                continue
            side = str(first_existing_column(row, ["side", "action", "order_action"]) or "").upper()
            if side and side not in {"BOT", "BUY", "BOUGHT"}:
                continue
            candidates.append(row)
    if not candidates:
        return "", ""
    candidates.sort(key=lambda row: parse_dt(first_existing_column(row, ["executed_at", "recorded_at", "created_at", "updated_at", "event_time", "timestamp"])) or pd.Timestamp.max.tz_localize("UTC"))
    row = candidates[0]
    buy_time = iso_ts(first_existing_column(row, ["executed_at", "recorded_at", "created_at", "updated_at", "event_time", "timestamp"]))
    order_id = first_existing_column(row, ["order_id", "entry_order_id", "perm_id"])
    return buy_time, str(order_id or "")


def duplicate_trade_diagnostics(sqlite_path: Path, session_date: str) -> dict[str, int]:
    trades = read_sql_table(
        sqlite_path,
        "trades",
        where="session_date = ? OR substr(entry_fill_time, 1, 10) = ? OR substr(exit_fill_time, 1, 10) = ? OR substr(closed_at, 1, 10) = ?",
        params=[session_date, session_date, session_date, session_date],
    )
    if trades.empty:
        return {
            "duplicate_trade_groups_count": 0,
            "duplicate_symbol_entry_time_count": 0,
            "duplicate_order_id_count": 0,
        }
    duplicate_trade_groups = 0
    if "trade_id" in trades.columns:
        duplicate_trade_groups = int((trades.groupby("trade_id", dropna=False).size() > 1).sum())
    symbol_entry_dupes = 0
    if {"symbol", "entry_fill_time"}.issubset(trades.columns):
        symbol_entry_dupes = int((trades.groupby(["symbol", "entry_fill_time"], dropna=False).size() > 1).sum())
    order_dupes = 0
    for col in ["entry_order_id", "order_id", "perm_id"]:
        if col in trades.columns:
            series = trades[col].fillna("").astype(str)
            non_blank = trades[series.str.strip().ne("")]
            if not non_blank.empty:
                order_dupes += int((non_blank.groupby(col, dropna=False).size() > 1).sum())
    return {
        "duplicate_trade_groups_count": duplicate_trade_groups,
        "duplicate_symbol_entry_time_count": symbol_entry_dupes,
        "duplicate_order_id_count": order_dupes,
    }


def classify_case(flags: dict[str, int]) -> str:
    if flags["was_bought"] and flags.get("buy_time"):
        return "bought_late"
    if flags["buy_fill_seen"]:
        return "bought_late"
    if flags["restart_blocked"]:
        return "restart_blocked"
    if flags["top100_blocked"]:
        return "top100_blocked"
    if flags["risk_guard_blocked"]:
        return "risk_guard_blocked"
    if flags["max_positions_blocked"] or flags["position_limit_reached"]:
        return "max_positions_blocked"
    if flags["spread_blocked"] or flags["price_missing_blocked"]:
        return "spread_or_price_blocked"
    if flags["subscription_missing_blocked"]:
        return "subscription_missing"
    if flags["candidate_replaced_by_higher_rank"]:
        return "candidate_replaced_by_higher_rank"
    if flags["signal_ready_seen"] and not flags["buy_attempt_seen"]:
        return "runtime_signal_ready_but_no_buy"
    if not flags["runtime_evidence_found"]:
        return "runtime_never_processed_symbol"
    return "unknown"


def investigate_case(
    *,
    target: dict[str, Any],
    session_date: str,
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    symbol = normalize_symbol(target.get("symbol"))
    center = parse_dt(
        target.get("possible_signal_time")
        or target.get("opening_range_break_time")
        or target.get("first_time_above_8pct")
    )
    sqlite_sources = sources_for_symbol(evidence.sqlite_by_symbol, symbol, center)
    recorder_sources = sources_for_symbol(evidence.recorder_by_symbol, symbol, center)
    timeline, raw_counts = build_symbol_timeline(
        row=target,
        sqlite_sources=sqlite_sources,
        recorder_sources=recorder_sources,
        center=center,
    )
    text = event_text(timeline)
    symbol_journal = journal_text_for_symbol(evidence.journal_lines, symbol, center)
    heartbeat = nearest_heartbeat_state_from_index(evidence.heartbeat_states, center)
    heartbeat_text = json.dumps(heartbeat, default=str).upper()
    combined = "\n".join([text, symbol_journal])
    global_combined = "\n".join([combined, heartbeat_text])

    buy_time, buy_order_id = buy_details(symbol, sqlite_sources, recorder_sources)
    runtime_count = sum(
        int(raw_counts.get(key, 0) or 0)
        for key in [
            "runtime_events_count",
            "risk_events_count",
            "orders_count",
            "trades_count",
            "executions_count",
            "trade_lifecycle_count",
            "order_lifecycle_count",
            "fills_count",
        ]
    )
    flags: dict[str, Any] = {
        "was_bought": int(pd.to_numeric(pd.Series([target.get("was_bought")]), errors="coerce").fillna(0).iloc[0] == 1),
        "buy_time": buy_time,
        "runtime_evidence_found": int(runtime_count > 0 or bool(symbol_journal)),
        "ready_candidate_seen": bool_text(combined, "READY_CANDIDATES", "LIVE_READY_CANDIDATES"),
        "signal_ready_seen": bool_text(combined, "SIGNAL_READY"),
        "buy_attempt_seen": bool_text(combined, "PAPER BUY", "PAPER_BUY", "BUY_ORDER_SENT", "ORDER_SUBMITTED"),
        "buy_fill_seen": bool_text(combined, "FILL", "EXECUTION", "BOT", "BOUGHT"),
        "risk_guard_blocked": bool_text(combined, "RISK_GUARD_BLOCK_ENTRY", "RISK_GUARD"),
        "max_positions_blocked": bool_text(combined, "MAX_POSITION", "MAX_POSITIONS"),
        "restart_blocked": bool_text(global_combined, "RESTART_BLOCK", "SIGNAL_BEFORE_LAST_UNBLOCK"),
        "top100_blocked": bool_text(global_combined, "TOP100_BLOCK"),
        "spread_blocked": bool_text(combined, "SPREAD"),
        "price_missing_blocked": bool_text(combined, "NO_USABLE_TICKER_PRICE", "NO_MARKET_DATA", "MISSING_MARKET_DATA", "PRICE_MISSING"),
        "subscription_missing_blocked": bool_text(combined, "SUBSCRIPTION", "NO_CONTRACT", "CONTRACT_FAILED", "NOT_SUBSCRIBED"),
        "stale_candidate": bool_text(combined, "STALE_CANDIDATE", "CANDIDATE_AGE", "STALE_OR_BACKFILL"),
        "candidate_replaced_by_higher_rank": bool_text(combined, "REPLACED_BY_HIGHER", "LOWER_RANK", "COMPETING_CANDIDATE"),
        "position_limit_reached": bool_text(combined, "POSITION_LIMIT", "MAX_SINGLE_POSITION", "MAX_POSITIONS"),
        "already_open": bool_text(combined, "ALREADY_OPEN", "DUPLICATE_POSITION"),
    }
    final = classify_case(flags)
    flags["unknown_reason"] = int(final == "unknown")
    return {
        "date": session_date,
        "symbol": symbol,
        "top100_rank": target.get("top100_rank"),
        "top100_score": target.get("top100_score"),
        "open_to_high_pct": target.get("open_to_high_pct"),
        "possible_signal_time": target.get("possible_signal_time"),
        "first_time_above_5pct": target.get("first_time_above_5pct"),
        "first_time_above_8pct": target.get("first_time_above_8pct"),
        "opening_range_break_time": target.get("opening_range_break_time"),
        "was_bought": flags["was_bought"],
        "buy_time": buy_time,
        "buy_order_id": buy_order_id,
        **{key: int(value) for key, value in flags.items() if key not in {"buy_time"}},
        "final_classification": final,
    }


def summary_for_cases(cases: pd.DataFrame, session_date: str, duplicate_diag: dict[str, int] | None = None) -> pd.DataFrame:
    counts = Counter(cases.get("final_classification", pd.Series(dtype=str)).fillna("unknown").astype(str)) if not cases.empty else Counter()
    row: dict[str, Any] = {"date": session_date, "total_should_have_signaled": int(len(cases))}
    for name in SUMMARY_CLASSIFICATIONS:
        row[name] = int(counts.get(name, 0))
    row.update(duplicate_diag or {
        "duplicate_trade_groups_count": 0,
        "duplicate_symbol_entry_time_count": 0,
        "duplicate_order_id_count": 0,
    })
    return pd.DataFrame([row])


def investigate_date(
    *,
    session_date: str,
    missed_path: Path | None,
    sqlite_path: Path,
    recorder_dir: Path,
    top100_path: Path | None,
    journal_log: Path | None,
    output_dir: Path,
    force: bool = False,
    max_cases: int | None = None,
) -> tuple[Path, Path]:
    cases_path = output_dir / f"should_have_signaled_cases_{session_date}.csv"
    summary_path = output_dir / f"should_have_signaled_summary_{session_date}.csv"
    if cases_path.exists() and summary_path.exists() and not force:
        print(f"SHS_SKIPPED_EXISTING date={session_date} output={cases_path}", flush=True)
        return cases_path, summary_path

    started = time.monotonic()
    targets = load_targets_for_date(session_date, missed_path, top100_path)
    if max_cases is not None and not targets.empty:
        targets = targets.head(max_cases).copy()
    print(f"SHS_START date={session_date} targets={len(targets)}", flush=True)
    target_symbols = set(targets["symbol"].map(normalize_symbol).dropna().tolist()) if not targets.empty else set()
    evidence = load_evidence_bundle(
        sqlite_path=sqlite_path,
        recorder_dir=recorder_dir,
        session_date=session_date,
        journal_log=journal_log,
        target_symbols=target_symbols,
    )
    duplicate_diag = duplicate_trade_diagnostics(sqlite_path, session_date)
    load_elapsed = time.monotonic() - started
    sqlite_rows = sum(len(df) for df in evidence.sqlite_sources.values())
    recorder_rows = sum(len(df) for df in evidence.recorder_sources.values())
    print(
        f"SHS_LOAD_EVIDENCE_DONE date={session_date} elapsed={load_elapsed:.1f} "
        f"sqlite_rows={sqlite_rows} recorder_rows={recorder_rows} journal_lines={len(evidence.journal_lines)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets.to_dict("records"), start=1):
        rows.append(
            investigate_case(
                target=target,
                session_date=session_date,
                evidence=evidence,
            )
        )
        if idx % 10 == 0 or idx == len(targets):
            elapsed = time.monotonic() - started
            print(f"SHS_PROGRESS date={session_date} processed={idx}/{len(targets)} elapsed={elapsed:.1f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(rows, columns=CASE_COLUMNS) if rows else pd.DataFrame(columns=CASE_COLUMNS)
    summary = summary_for_cases(cases, session_date, duplicate_diag)
    cases.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)
    elapsed = time.monotonic() - started
    print(f"SHS_DONE date={session_date} elapsed_seconds={elapsed:.1f} output={cases_path}", flush=True)
    return cases_path, summary_path


def update_all_summary(output_dir: Path, summaries: list[Path]) -> Path:
    frames = [pd.read_csv(path) for path in summaries if path.exists()]
    if not frames:
        out = pd.DataFrame(columns=["date", "total_should_have_signaled", *SUMMARY_CLASSIFICATIONS])
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    path = output_dir / "should_have_signaled_summary_ALL.csv"
    out.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investigate Top100 missed runners marked should_have_signaled.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Session date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for an inclusive date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for --start-date range, YYYY-MM-DD.")
    parser.add_argument("--missed-runners-csv", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--journal-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dates = [args.date] if args.date else list(iter_dates(args.start_date, args.end_date or args.start_date))
    summaries: list[Path] = []
    for session_date in dates:
        cases_path, summary_path = investigate_date(
            session_date=session_date,
            missed_path=args.missed_runners_csv if args.date else None,
            sqlite_path=args.sqlite_path,
            recorder_dir=args.recorder_dir,
            top100_path=args.top100 if args.date else None,
            journal_log=args.journal_log if args.date else None,
            output_dir=args.output_dir,
            force=args.force,
            max_cases=args.max_cases,
        )
        del cases_path
        summaries.append(summary_path)
    all_path = update_all_summary(args.output_dir, summaries)
    print(f"SHS_SUMMARY_ALL output={all_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
