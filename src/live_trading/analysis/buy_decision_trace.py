from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_runner_stats,
    fnum,
    iso_ts,
    load_session_candles,
    load_top100,
    normalize_symbol,
    parse_dt,
    safe_read_csv,
)
from src.live_trading.analysis.missed_runners_analyzer import no_signal_diagnostics
from src.live_trading.analysis.signal_case_trace import DEFAULT_TRADER_SOURCE, static_line_refs
from src.live_trading.analysis.signal_replay_analyzer import (
    read_sqlite_sources,
    recorder_events,
    row_event_type,
    row_reason,
    row_text,
    row_time,
    symbol_rows,
)
from src.live_trading.analysis.symbol_subscription_inspector import (
    extract_last_restart_unblock_time,
    line_time,
    parse_key_values,
    read_text_lines,
    symbol_journal_lines,
)
from src.live_trading.ranking.daily_top100_builder import parquet_path


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")

SUMMARY_COLUMNS = [
    "date",
    "symbol",
    "top100_rank",
    "top100_score",
    "possible_signal_time",
    "offline_signal_ready",
    "first5_pass",
    "first15_pass",
    "or_pass",
    "breakout_pass",
    "market_data_available",
    "runtime_symbol_lines",
    "runtime_signal_seen",
    "runtime_buy_attempt_seen",
    "ready_candidate_seen",
    "entries_blocked_at_signal_time",
    "entries_blocked_reasons",
    "global_risk_guard_block",
    "global_risk_guard_reason",
    "symbol_specific_risk_guard_seen",
    "managed_open",
    "competing_buy_count",
    "competing_buy_symbols",
    "last_restart_unblock_time",
    "signal_before_last_unblock",
    "summary_verdict",
]


BLOCK_FIELDS = [
    "manual_block",
    "restart_block",
    "reconnect_block",
    "top100_block",
    "disk_block",
    "pending_eod_flatten",
    "risk_guard_block",
    "eod_recovery_active",
    "subscription_cap_block",
]


def pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def default_journal_path(session_date: str) -> Path:
    compact = session_date.replace("-", "")
    candidates = [
        Path(f"data/analysis/journal_v67_{session_date}_1320_1530_utc.log"),
        Path(f"data/analysis/journal_v67_{compact}_1320_1530_utc.log"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def line_window(lines: list[str], center: pd.Timestamp | None, minutes: int = 5) -> list[str]:
    if center is None:
        return lines
    out: list[str] = []
    start = center - pd.Timedelta(minutes=minutes)
    end = center + pd.Timedelta(minutes=minutes)
    for line in lines:
        ts = line_time(line)
        if ts is not None and start <= ts <= end:
            out.append(line)
    return out


def nearest_heartbeat(lines: list[str], center: pd.Timestamp | None) -> str:
    heartbeats = [line for line in lines if "heartbeat" in line]
    if not heartbeats:
        return ""
    if center is None:
        return heartbeats[-1]
    timed = [(line, line_time(line)) for line in heartbeats]
    timed = [(line, ts) for line, ts in timed if ts is not None]
    if not timed:
        return heartbeats[-1]
    return min(timed, key=lambda item: abs((item[1] - center).total_seconds()))[0]


def heartbeat_block_state(line: str) -> dict[str, Any]:
    kv = parse_key_values(line)
    active = [field for field in BLOCK_FIELDS if truthy(kv.get(field, ""))]
    entries_blocked = truthy(kv.get("entries_blocked", "")) or bool(active)
    if truthy(kv.get("risk_guard_block", "")):
        reason = kv.get("risk_guard_reason") or kv.get("entries_blocked_reason") or "risk_guard"
        if reason not in active:
            active.append(f"risk_guard_reason:{reason}")
    elif kv.get("entries_blocked_reason"):
        active.append(f"entries_blocked_reason:{kv.get('entries_blocked_reason')}")
    return {
        "entries_blocked": int(entries_blocked),
        "active_reasons": active,
        "risk_guard_block": int(truthy(kv.get("risk_guard_block", ""))),
        "risk_guard_reason": kv.get("risk_guard_reason", ""),
        "managed_open": kv.get("managed_open", ""),
        "ready_candidates_raw": kv.get("ready_candidates", ""),
        "live_ready_candidates": kv.get("live_ready_candidates", ""),
        "stale_ready_candidates": kv.get("stale_ready_candidates", ""),
        "raw": line,
    }


def heartbeat_symbol_seen(lines: list[str], symbol: str, center: pd.Timestamp | None) -> bool:
    needle = normalize_symbol(symbol)
    pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(needle)}(?![A-Z0-9])")
    for line in line_window(lines, center, minutes=30):
        upper = line.upper()
        if "HEARTBEAT" not in upper:
            continue
        if not any(token in upper for token in ["READY_CANDIDATES", "TOP5", "REJECTS"]):
            continue
        if pattern.search(upper):
            return True
    return False


def parse_buy_lines(lines: list[str], center: pd.Timestamp | None, minutes: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in line_window(lines, center, minutes=minutes):
        upper = line.upper()
        if "PAPER BUY SENT" not in upper and "BUY_ORDER_SENT" not in upper:
            continue
        kv = parse_key_values(line)
        symbol = normalize_symbol(kv.get("symbol", ""))
        if not symbol:
            match = re.search(r"symbol=([A-Za-z0-9.\-]+)", line)
            symbol = normalize_symbol(match.group(1) if match else "")
        out.append({
            "time": iso_ts(line_time(line)),
            "symbol": symbol,
            "score": kv.get("score") or kv.get("live_entry_score", ""),
            "ranking_position": kv.get("ranking_position") or kv.get("live_entry_rank", ""),
            "signal_time": kv.get("signal_time", ""),
            "candidate_age_seconds": kv.get("candidate_age_seconds", ""),
            "raw": line,
        })
    return out


def load_top100_row(path: Path, symbol: str) -> dict[str, Any]:
    top100 = load_top100(path)
    if top100.empty or "symbol" not in top100.columns:
        return {}
    rows = top100[top100["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def possible_signal_from_artifacts(session_date: str, symbol: str) -> str:
    for path in [
        Path(f"data/analysis/missed_runners_{session_date}.csv"),
        Path(f"data/analysis/signal_replay_{session_date}.csv"),
        Path(f"data/analysis/subscription_trace_summary_{session_date}.csv"),
    ]:
        df = safe_read_csv(path)
        if df.empty or "symbol" not in df.columns:
            continue
        rows = df[df["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
        if rows.empty:
            continue
        for col in ["possible_signal_time", "opening_range_break_time", "first_time_above_8pct"]:
            if col in rows.columns:
                value = rows.iloc[0].get(col)
                if value not in (None, "") and not pd.isna(value):
                    return str(value)
    return ""


def offline_signal_details(
    *,
    session_date: str,
    symbol: str,
    history_dir: Path,
    top100_path: Path,
    min_first_5m_high_pct: float,
    min_first_15m_high_pct: float,
    min_or_range_pct: float,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)
    top = load_top100_row(top100_path, symbol)
    candles = load_session_candles(history_dir, symbol, session_date, "RTH")
    stats = calculate_runner_stats(candles)
    diag = no_signal_diagnostics(
        candles,
        min_first_5m_high_pct=min_first_5m_high_pct,
        min_first_15m_high_pct=min_first_15m_high_pct,
        min_or_range_pct=min_or_range_pct,
    )
    first5 = bool(diag.get("had_required_first5"))
    first15 = bool(diag.get("had_required_first15"))
    or_pass = bool(diag.get("had_required_or_range"))
    breakout = bool(diag.get("did_break_or_high"))
    possible = (
        possible_signal_from_artifacts(session_date, symbol)
        or str(diag.get("possible_signal_time") or diag.get("opening_range_break_time") or "")
    )
    return {
        "symbol": symbol,
        "top100_row": top,
        "in_top100": bool(top),
        "top100_rank": top.get("top100_rank", ""),
        "top100_score": top.get("top100_score", ""),
        "candles": candles,
        "parquet_path": parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), "RTH"),
        "stats": stats,
        "diag": diag,
        "possible_signal_time": possible,
        "possible_signal_ts": parse_dt(possible),
        "first5_pass": first5,
        "first15_pass": first15,
        "or_pass": or_pass,
        "breakout_pass": breakout,
        "offline_signal_ready": bool(top) and first5 and first15 and or_pass and breakout,
    }


def runtime_symbol_evidence(sqlite_path: Path, recorder_dir: Path, session_date: str, symbol: str, center: pd.Timestamp | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources = read_sqlite_sources(sqlite_path, session_date)
    rec_sources = recorder_events(recorder_dir, session_date, symbol, center)
    events: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for name, df in {**sources, **rec_sources}.items():
        rows = symbol_rows(df, symbol, center, window_minutes=30)
        counts[name] = len(rows)
        for row in rows.to_dict("records"):
            events.append({
                "time": iso_ts(row_time(row)),
                "source": name,
                "event": row_event_type(row, name),
                "reason": row_reason(row),
                "details": row_text(row)[:300],
            })
    events.sort(key=lambda item: parse_dt(item.get("time")) or pd.Timestamp.max.tz_localize("UTC"))
    return events, counts


def classify_verdict(
    *,
    offline_ready: bool,
    center: pd.Timestamp | None,
    last_unblock: pd.Timestamp | None,
    block_state: dict[str, Any],
    runtime_events: list[dict[str, Any]],
    ready_candidate_seen: bool,
    competing_buys: list[dict[str, Any]],
) -> str:
    if not offline_ready:
        return "offline_signal_not_ready"
    if center is not None and last_unblock is not None and center < last_unblock:
        return "missed_due_to_restart_block"
    reasons = set(block_state.get("active_reasons") or [])
    if "restart_block" in reasons or "reconnect_block" in reasons:
        return "missed_due_to_restart_block"
    if "top100_block" in reasons:
        return "missed_due_to_top100_block"
    text = "\n".join(f"{event.get('event')} {event.get('reason')} {event.get('details')}" for event in runtime_events).upper()
    if "RISK_GUARD_BLOCK_ENTRY" in text or "RISK_GUARD" in text:
        return "missed_due_to_risk_guard"
    if "STALE" in text or "CANDIDATE_AGE" in text:
        return "missed_due_to_candidate_age_stale"
    if not runtime_events and not ready_candidate_seen:
        return "runtime_never_processed_symbol"
    if competing_buys and not ready_candidate_seen:
        return "missed_due_to_not_in_ready_candidates"
    if competing_buys:
        return "missed_due_to_lower_rank_than_other_candidates"
    if not ready_candidate_seen:
        return "missed_due_to_not_in_ready_candidates"
    return "missed_due_to_unknown_logic_gap"


def trace_buy_decision(
    *,
    session_date: str,
    symbol: str,
    journal_log: Path | None,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    top100_path: Path,
    min_first_5m_high_pct: float = 0.5,
    min_first_15m_high_pct: float = 1.0,
    min_or_range_pct: float = 0.5,
) -> tuple[str, dict[str, Any]]:
    symbol = normalize_symbol(symbol)
    offline = offline_signal_details(
        session_date=session_date,
        symbol=symbol,
        history_dir=history_dir,
        top100_path=top100_path,
        min_first_5m_high_pct=min_first_5m_high_pct,
        min_first_15m_high_pct=min_first_15m_high_pct,
        min_or_range_pct=min_or_range_pct,
    )
    center = offline["possible_signal_ts"]
    journal = read_text_lines(journal_log)
    symbol_lines = symbol_journal_lines(journal, symbol)
    nearest_hb = nearest_heartbeat(journal, center)
    block_state = heartbeat_block_state(nearest_hb)
    last_unblock = extract_last_restart_unblock_time(journal)
    ready_candidate_seen = heartbeat_symbol_seen(journal, symbol, center)
    competing_buys = parse_buy_lines(journal, center, minutes=5)
    runtime_events, runtime_counts = runtime_symbol_evidence(sqlite_path, recorder_dir, session_date, symbol, center)
    verdict = classify_verdict(
        offline_ready=bool(offline["offline_signal_ready"]),
        center=center,
        last_unblock=last_unblock,
        block_state=block_state,
        runtime_events=runtime_events,
        ready_candidate_seen=ready_candidate_seen,
        competing_buys=competing_buys,
    )
    stats = offline["stats"]
    candles = offline["candles"]
    market_data_available = not candles.empty
    runtime_text = "\n".join(f"{event.get('event')} {event.get('details')}" for event in runtime_events).upper()
    symbol_specific_risk_guard_seen = int("RISK_GUARD_BLOCK_ENTRY" in runtime_text or "RISK_GUARD" in runtime_text)
    runtime_signal_seen = int("SIGNAL_READY" in runtime_text)
    runtime_buy_attempt_seen = int(any(token in runtime_text for token in ["PAPER BUY", "PAPER_BUY", "BUY_ORDER_SENT", "ORDER_SUBMITTED"]))
    summary = {
        "date": session_date,
        "symbol": symbol,
        "top100_rank": offline["top100_rank"],
        "top100_score": offline["top100_score"],
        "possible_signal_time": iso_ts(center),
        "offline_signal_ready": int(offline["offline_signal_ready"]),
        "first5_pass": int(offline["first5_pass"]),
        "first15_pass": int(offline["first15_pass"]),
        "or_pass": int(offline["or_pass"]),
        "breakout_pass": int(offline["breakout_pass"]),
        "market_data_available": int(market_data_available),
        "runtime_symbol_lines": len(symbol_lines),
        "runtime_signal_seen": runtime_signal_seen,
        "runtime_buy_attempt_seen": runtime_buy_attempt_seen,
        "ready_candidate_seen": int(ready_candidate_seen),
        "entries_blocked_at_signal_time": block_state["entries_blocked"],
        "entries_blocked_reasons": ",".join(block_state["active_reasons"]),
        "global_risk_guard_block": block_state["risk_guard_block"],
        "global_risk_guard_reason": block_state["risk_guard_reason"],
        "symbol_specific_risk_guard_seen": symbol_specific_risk_guard_seen,
        "managed_open": block_state["managed_open"],
        "competing_buy_count": len(competing_buys),
        "competing_buy_symbols": ",".join([buy["symbol"] for buy in competing_buys if buy.get("symbol")]),
        "last_restart_unblock_time": iso_ts(last_unblock),
        "signal_before_last_unblock": int(center is not None and last_unblock is not None and center < last_unblock),
        "summary_verdict": verdict,
    }
    lines: list[str] = [
        f"BUY DECISION TRACE date={session_date} symbol={symbol}",
        "=" * 88,
        "",
        "Summary verdict",
        f"- summary_verdict={verdict}",
        f"- possible_signal_time={summary['possible_signal_time']}",
        f"- last_restart_unblock_time={summary['last_restart_unblock_time']}",
        f"- signal_before_last_unblock={summary['signal_before_last_unblock']}",
        f"- entries_blocked_at_signal_time={summary['entries_blocked_at_signal_time']}",
        f"- entries_blocked_reasons={summary['entries_blocked_reasons']}",
        f"- global_risk_guard_block={summary['global_risk_guard_block']} global_risk_guard_reason={summary['global_risk_guard_reason']}",
        f"- symbol_specific_risk_guard_seen={summary['symbol_specific_risk_guard_seen']}",
        f"- managed_open={summary['managed_open']}",
        f"- ready_candidate_seen={summary['ready_candidate_seen']}",
        f"- competing_buy_count={summary['competing_buy_count']} competing_buy_symbols={summary['competing_buy_symbols']}",
        "",
        "1. Top100 / Universe",
        f"- in_top100={int(offline['in_top100'])}",
        f"- top100_rank={offline['top100_rank']} top100_score={offline['top100_score']}",
        f"- top100_row={offline['top100_row']}",
        "",
        "2. Market Data / Offline Signal",
        f"- parquet_path={offline['parquet_path']}",
        f"- candle_rows={len(candles)}",
        f"- candles_min_time_utc={iso_ts(candles['timestamp'].min()) if not candles.empty else ''}",
        f"- candles_max_time_utc={iso_ts(candles['timestamp'].max()) if not candles.empty else ''}",
        f"- first_5m_high_pct={fnum(stats.first_5m_high_pct) if stats else ''} {pass_fail(bool(offline['first5_pass']))}",
        f"- first_15m_high_pct={fnum(stats.first_15m_high_pct) if stats else ''} {pass_fail(bool(offline['first15_pass']))}",
        f"- or_range_pct={fnum(stats.or_range_pct) if stats else ''} {pass_fail(bool(offline['or_pass']))}",
        f"- open_to_high_pct={fnum(stats.open_to_high_pct) if stats else ''}",
        f"- opening_range_break_time={offline['diag'].get('opening_range_break_time')}",
        f"- possible_signal_time={summary['possible_signal_time']}",
        f"- offline_signal_ready={'YES' if offline['offline_signal_ready'] else 'NO'}",
        "",
        "3. Global Heartbeat State At Signal Time",
        f"- journal_log={journal_log}",
        f"- journal_symbol_lines_count={len(symbol_lines)}",
        f"- nearest_heartbeat={nearest_hb}",
        f"- parsed_heartbeat_block_state={block_state}",
        "- NOTE: global heartbeat blocks are diagnostics only; they are not treated as symbol-specific BUY rejection evidence.",
        "",
        "4. Symbol-Specific Evidence",
        f"- symbol_specific_risk_guard_seen={summary['symbol_specific_risk_guard_seen']}",
        f"- runtime_symbol_event_count={len(runtime_events)}",
    ]
    lines.extend(["", "Symbol-specific journal lines"])
    lines.extend(f"- {line}" for line in symbol_lines[:120])
    if len(symbol_lines) > 120:
        lines.append(f"... {len(symbol_lines) - 120} more symbol lines omitted")
    lines.extend(["", "Journal lines around possible signal (+/-5m)"])
    for line in line_window(journal, center, minutes=5)[:180]:
        lines.append(f"- {line}")
    lines.extend(["", "5. Competing BUY candidates/orders around signal (+/-5m)"])
    if competing_buys:
        for buy in competing_buys:
            lines.append(
                f"- time={buy['time']} symbol={buy['symbol']} score={buy['score']} "
                f"ranking_position={buy['ranking_position']} signal_time={buy['signal_time']} "
                f"candidate_age_seconds={buy['candidate_age_seconds']} raw={buy['raw']}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "6. Runtime Artifact Evidence For Symbol"])
    for name, count in runtime_counts.items():
        lines.append(f"- {name}_rows={count}")
    for event in runtime_events[:160]:
        lines.append(f"- {event['time']} [{event['source']}] {event['event']} reason={event['reason']} details={event['details']}")
    if len(runtime_events) > 160:
        lines.append(f"... {len(runtime_events) - 160} more runtime events omitted")
    lines.extend(["", "7. Code Branch References"])
    for ref in static_line_refs(DEFAULT_TRADER_SOURCE)[:260]:
        lines.append(f"- {ref}")
    return "\n".join(lines) + "\n", summary


def load_no_signal_targets_after_unblock(session_date: str, last_unblock: pd.Timestamp | None) -> list[str]:
    missed = safe_read_csv(Path(f"data/analysis/missed_runners_{session_date}.csv"))
    if missed.empty or "symbol" not in missed.columns:
        replay = safe_read_csv(Path(f"data/analysis/signal_replay_{session_date}.csv"))
        if replay.empty or "symbol" not in replay.columns:
            return []
        rows = replay[replay["final_replay_reason"].astype(str).isin(["signal_ready_but_no_buy_attempt", "no_runtime_evidence"])] if "final_replay_reason" in replay.columns else replay
    else:
        mask = pd.Series(True, index=missed.index)
        if "source_bucket" in missed.columns:
            mask &= missed["source_bucket"].astype(str) == "top100"
        if "was_bought" in missed.columns:
            mask &= pd.to_numeric(missed["was_bought"], errors="coerce").fillna(0) == 0
        reason_col = "missed_reason_group" if "missed_reason_group" in missed.columns else "top100_no_signal_reason"
        if reason_col in missed.columns:
            mask &= missed[reason_col].astype(str).isin(["no_signal", "should_have_signaled"])
        rows = missed[mask].copy()
    if rows.empty:
        return []
    if last_unblock is not None and "possible_signal_time" in rows.columns:
        rows["_possible_signal_ts"] = rows["possible_signal_time"].map(parse_dt)
        rows = rows[(rows["_possible_signal_ts"].isna()) | (rows["_possible_signal_ts"] >= last_unblock)]
    return sorted({normalize_symbol(value) for value in rows["symbol"].tolist() if normalize_symbol(value)})


def write_trace_outputs(
    *,
    session_date: str,
    symbols: list[str],
    journal_log: Path | None,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    top100_path: Path,
    output: Path | None,
    min_first_5m_high_pct: float,
    min_first_15m_high_pct: float,
    min_or_range_pct: float,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for symbol in symbols:
        default_output = Path(f"data/analysis/buy_decision_trace_{normalize_symbol(symbol)}_{session_date}.txt")
        out_path = output if output and len(symbols) == 1 else default_output
        text, summary = trace_buy_decision(
            session_date=session_date,
            symbol=symbol,
            journal_log=journal_log,
            sqlite_path=sqlite_path,
            history_dir=history_dir,
            recorder_dir=recorder_dir,
            top100_path=top100_path,
            min_first_5m_high_pct=min_first_5m_high_pct,
            min_first_15m_high_pct=min_first_15m_high_pct,
            min_or_range_pct=min_or_range_pct,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        summaries.append(summary)
        if len(symbols) == 1:
            print(text)
            print(f"output={out_path}")
        else:
            print(f"wrote {out_path}")
    df = pd.DataFrame(summaries)
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return df[SUMMARY_COLUMNS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace why a should-have-signaled Top100 symbol did or did not get a BUY decision.")
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD.")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Defaults to --symbol.")
    parser.add_argument("--journal-log", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--no-summary-all", action="store_true", help="Only trace requested symbol(s), without building all post-unblock no-signal summary.")
    parser.add_argument("--min-first-5m-high-pct", type=float, default=0.5)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=1.0)
    parser.add_argument("--min-or-range-pct", type=float, default=0.5)
    return parser


def parse_symbols(args: argparse.Namespace) -> list[str]:
    raw = args.symbols or args.symbol
    if not raw:
        return []
    return [normalize_symbol(value) for value in str(raw).split(",") if normalize_symbol(value)]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = parse_symbols(args)
    if not symbols:
        parser.error("--symbol or --symbols is required")
    top100 = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    journal = args.journal_log or default_journal_path(args.date)
    requested_df = write_trace_outputs(
        session_date=args.date,
        symbols=symbols,
        journal_log=journal,
        sqlite_path=args.sqlite_path,
        history_dir=args.history_dir,
        recorder_dir=args.recorder_dir,
        top100_path=top100,
        output=args.output,
        min_first_5m_high_pct=args.min_first_5m_high_pct,
        min_first_15m_high_pct=args.min_first_15m_high_pct,
        min_or_range_pct=args.min_or_range_pct,
    )
    summary_path = args.summary_output or Path(f"data/analysis/buy_decision_trace_summary_{args.date}.csv")
    summary_symbols = symbols
    if not args.no_summary_all:
        last_unblock = extract_last_restart_unblock_time(read_text_lines(journal))
        summary_symbols = sorted(set(summary_symbols) | set(load_no_signal_targets_after_unblock(args.date, last_unblock)))
    if summary_symbols != symbols:
        summary_df = write_trace_outputs(
            session_date=args.date,
            symbols=summary_symbols,
            journal_log=journal,
            sqlite_path=args.sqlite_path,
            history_dir=args.history_dir,
            recorder_dir=args.recorder_dir,
            top100_path=top100,
            output=None,
            min_first_5m_high_pct=args.min_first_5m_high_pct,
            min_first_15m_high_pct=args.min_first_15m_high_pct,
            min_or_range_pct=args.min_or_range_pct,
        )
    else:
        summary_df = requested_df
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    if not summary_df.empty:
        counts = Counter(summary_df["summary_verdict"].fillna("unknown").astype(str))
        print("summary_verdict_counts=" + ", ".join(f"{key}:{value}" for key, value in counts.most_common()))
    print(f"summary_output={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
