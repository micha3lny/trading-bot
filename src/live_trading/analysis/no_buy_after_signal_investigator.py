from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    iso_ts,
    normalize_symbol,
    parse_dt,
    safe_read_csv,
)
from src.live_trading.analysis.signal_replay_analyzer import (
    build_symbol_timeline,
)
from src.live_trading.analysis.should_have_signaled_investigator import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_RECORDER_DIR,
    DEFAULT_SQLITE_PATH,
    EvidenceBundle,
    default_journal_path,
    event_text,
    iter_dates,
    journal_text_for_symbol,
    load_evidence_bundle,
    sources_for_symbol,
)
from src.live_trading.analysis.symbol_subscription_inspector import (
    line_time,
    parse_key_values,
)


CASE_COLUMNS = [
    "date",
    "symbol",
    "top100_rank",
    "top100_score",
    "possible_signal_time",
    "signal_ready_time",
    "next_entry_scan_time_after_signal",
    "candidate_present_in_ready_list_after_signal",
    "candidate_rank_at_scan",
    "candidate_score_at_scan",
    "candidates_ahead_count",
    "better_candidates_ahead_symbols",
    "max_entries_per_scan_reached",
    "max_positions_at_scan",
    "managed_open_at_scan",
    "entries_blocked_at_scan",
    "entries_blocked_reason_at_scan",
    "risk_guard_state_at_scan",
    "subscription_cap_state_at_scan",
    "spread_bps_at_scan",
    "price_at_scan",
    "passed_final_entry_filters",
    "failed_final_entry_filter_reason",
    "order_dispatch_attempted",
    "order_dispatch_skip_reason",
    "post_signal_terminal_event",
    "post_signal_terminal_reason",
    "stale_or_backfill_reason",
    "already_open_after_signal",
    "post_signal_continue_detected",
    "ready_list_stage",
    "ranking_stage",
    "selection_stage",
    "final_filter_stage",
    "dispatch_queue_stage",
    "ibkr_order_submission_stage",
    "order_ack_stage",
    "candidate_disappeared_stage",
    "candidate_lifecycle_trace",
    "runtime_code_path",
    "final_no_buy_reason",
]

NO_BUY_CLASSIFICATIONS = [
    "lower_rank_candidate_not_selected",
    "per_scan_entry_limit_reached",
    "max_positions_reached",
    "entries_blocked",
    "final_filter_failed_spread",
    "final_filter_failed_price",
    "stale_candidate",
    "already_open_or_pending",
    "post_signal_stale_or_backfill_skip",
    "post_signal_already_open_skip",
    "unexplained_after_signal_before_dispatch",
    "unknown_no_buy_after_signal",
]

BLOCK_FIELDS = [
    "manual_block",
    "restart_block",
    "reconnect_block",
    "top100_block",
    "disk_block",
    "pending_eod_flatten",
]


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def numeric(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_targets_for_date(session_date: str, cases_csv: Path | None, max_cases: int | None = None) -> pd.DataFrame:
    path = cases_csv or DEFAULT_ANALYSIS_DIR / f"should_have_signaled_cases_{session_date}.csv"
    df = safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()
    if "final_classification" not in df.columns:
        return pd.DataFrame()
    out = df[df["final_classification"].fillna("").astype(str) == "runtime_signal_ready_but_no_buy"].copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    if max_cases is not None:
        out = out.head(max_cases).copy()
    return out


def heartbeat_after_signal(index: list[tuple[pd.Timestamp, dict[str, str]]], signal_time: pd.Timestamp | None) -> tuple[pd.Timestamp | None, dict[str, str]]:
    if not index:
        return None, {}
    if signal_time is None:
        return index[-1]
    for ts, state in index:
        if ts >= signal_time:
            return ts, state
    return index[-1]


def symbol_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Z0-9]){re.escape(normalize_symbol(symbol))}(?![A-Z0-9])")


def symbol_lines(lines: list[str], symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[str]:
    pattern = symbol_pattern(symbol)
    out: list[str] = []
    for line in lines:
        if not pattern.search(line.upper()):
            continue
        ts = line_time(line)
        if start is not None and ts is not None and ts < start:
            continue
        if end is not None and ts is not None and ts > end:
            continue
        out.append(line)
    return out


def find_signal_ready_time(timeline: list[dict[str, Any]], journal_lines: list[str], symbol: str, fallback: pd.Timestamp | None) -> pd.Timestamp | None:
    candidates: list[pd.Timestamp] = []
    for event in timeline:
        text = f"{event.get('event', '')} {event.get('reason', '')} {event.get('details', '')}".upper()
        if "SIGNAL_READY" not in text:
            continue
        ts = parse_dt(event.get("time"))
        if ts is not None:
            candidates.append(ts)
    pattern = symbol_pattern(symbol)
    for line in journal_lines:
        upper = line.upper()
        if "SIGNAL_READY" not in upper or not pattern.search(upper):
            continue
        ts = line_time(line)
        if ts is not None:
            candidates.append(ts)
    if candidates:
        return min(candidates)
    return fallback


def extract_symbol_values(text: str, symbol: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    pattern = symbol_pattern(symbol)
    for line in text.splitlines():
        upper = line.upper()
        if not pattern.search(upper):
            continue
        kv = parse_key_values(line)
        for key, aliases in {
            "candidate_rank_at_scan": ["ranking_position", "live_entry_rank", "rank"],
            "candidate_score_at_scan": ["score", "live_entry_score"],
            "spread_bps_at_scan": ["spread_bps"],
            "price_at_scan": ["price", "entry_price", "current_price", "last_price"],
        }.items():
            if values.get(key) not in (None, ""):
                continue
            for alias in aliases:
                if alias in kv and kv[alias] not in (None, ""):
                    values[key] = kv[alias]
                    break
    return values


def parse_buy_lines(lines: list[str], start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        upper = line.upper()
        if "PAPER BUY SENT" not in upper and "BUY_ORDER_SENT" not in upper:
            continue
        ts = line_time(line)
        if start is not None and ts is not None and ts < start:
            continue
        if end is not None and ts is not None and ts > end:
            continue
        kv = parse_key_values(line)
        out.append(
            {
                "time": ts,
                "symbol": normalize_symbol(kv.get("symbol", "")),
                "score": numeric(kv.get("score") or kv.get("live_entry_score")),
                "rank": numeric(kv.get("ranking_position") or kv.get("live_entry_rank")),
                "raw": line,
            }
        )
    return out


def better_candidates_ahead(lines: list[str], symbol: str, signal_time: pd.Timestamp | None, candidate_rank: Any) -> list[str]:
    start = signal_time
    end = signal_time + pd.Timedelta(minutes=5) if signal_time is not None else None
    buys = parse_buy_lines(lines, start, end)
    target_rank = numeric(candidate_rank)
    out: list[str] = []
    for buy in buys:
        buy_symbol = normalize_symbol(buy.get("symbol"))
        if not buy_symbol or buy_symbol == normalize_symbol(symbol):
            continue
        buy_rank = numeric(buy.get("rank"))
        if target_rank is not None and buy_rank is not None and buy_rank > target_rank:
            continue
        label = buy_symbol
        if buy_rank is not None:
            label += f":rank={int(buy_rank)}"
        if buy.get("score") is not None:
            label += f":score={buy['score']:.2f}"
        out.append(label)
    return out


def order_dispatch_attempted(text: str, symbol: str) -> int:
    pattern = symbol_pattern(symbol)
    for line in text.splitlines():
        upper = line.upper()
        if not pattern.search(upper):
            continue
        if any(token in upper for token in ["PAPER BUY SENT", "BUY_ORDER_SENT", "ORDER_SUBMITTED", "ENTRY_ORDER_PARTIAL"]):
            return 1
    return 0


def order_ack_seen(text: str, symbol: str) -> int:
    pattern = symbol_pattern(symbol)
    for line in text.splitlines():
        upper = line.upper()
        if not pattern.search(upper):
            continue
        if any(token in upper for token in ["ENTRY_ORDER_PARTIAL", "ENTRY_ORDER_FILLED", "ORDER_STATUS", "IBKR_STATUS", "SUBMITTED"]):
            return 1
    return 0


def skip_reason_from_text(text: str, symbol: str) -> str:
    pattern = symbol_pattern(symbol)
    preferred = [
        "STALE_OR_BACKFILL_READY_SKIPPED",
        "ENTRY_BLOCKED_LOW_SCORE",
        "RISK_GUARD_BLOCK_ENTRY",
        "BUY_BLOCKED",
        "ENTRY_RATE_LIMIT_BLOCK",
        "ENTRY_SYMBOL_INELIGIBLE_SKIPPED",
    ]
    for line in text.splitlines():
        upper = line.upper()
        if not pattern.search(upper):
            continue
        if not any(token in upper for token in preferred):
            continue
        kv = parse_key_values(line)
        return kv.get("reason") or kv.get("skip_reason") or line[:240]
    return ""


def post_signal_terminal_evidence(text: str, symbol: str) -> dict[str, Any]:
    stale_reason = ""
    already_open_reason = ""
    terminal_lines = []
    for line in text.splitlines():
        upper = line.upper()
        if symbol_pattern(symbol).search(upper) or "STALE_OR_BACKFILL_READY_SKIPPED" in upper or "ALREADY_OPEN_POSITION" in upper:
            terminal_lines.append(line)
    for line in terminal_lines:
        upper = line.upper()
        kv = parse_key_values(line)
        if "STALE_OR_BACKFILL_READY_SKIPPED" in upper:
            stale_reason = kv.get("reason") or kv.get("stale_or_backfill_reason") or line[:240]
        if "ALREADY_OPEN_POSITION" in upper or "ALREADY_OPEN" in upper:
            already_open_reason = kv.get("reason") or "already_open_position"
    if stale_reason:
        return {
            "post_signal_terminal_event": "STALE_OR_BACKFILL_READY_SKIPPED",
            "post_signal_terminal_reason": stale_reason,
            "stale_or_backfill_reason": stale_reason,
            "already_open_after_signal": 0,
            "post_signal_continue_detected": 1,
        }
    if already_open_reason:
        return {
            "post_signal_terminal_event": "already_open_position",
            "post_signal_terminal_reason": already_open_reason,
            "stale_or_backfill_reason": "",
            "already_open_after_signal": 1,
            "post_signal_continue_detected": 1,
        }
    return {
        "post_signal_terminal_event": "",
        "post_signal_terminal_reason": "",
        "stale_or_backfill_reason": "",
        "already_open_after_signal": 0,
        "post_signal_continue_detected": 0,
    }


def heartbeat_entries_blocked_reason(state: dict[str, str]) -> str:
    reasons: list[str] = []
    for field in BLOCK_FIELDS:
        if truthy(state.get(field)):
            reasons.append(field)
    if truthy(state.get("risk_guard_block")):
        reasons.append(f"risk_guard:{state.get('risk_guard_reason') or 'unknown'}")
    if truthy(state.get("subscription_cap_block")):
        reasons.append("subscription_cap")
    if state.get("entries_blocked_reason"):
        reasons.append(f"entries_blocked_reason:{state.get('entries_blocked_reason')}")
    return ",".join(dict.fromkeys(reasons))


def classify_no_buy_reason(row: dict[str, Any]) -> str:
    failed = str(row.get("failed_final_entry_filter_reason") or "").lower()
    skip = str(row.get("order_dispatch_skip_reason") or "").lower()
    terminal_event = str(row.get("post_signal_terminal_event") or "").upper()
    terminal_reason = str(row.get("post_signal_terminal_reason") or "").lower()
    if (
        not truthy(row.get("order_dispatch_attempted"))
        and terminal_event == "STALE_OR_BACKFILL_READY_SKIPPED"
    ):
        return "post_signal_stale_or_backfill_skip"
    if (
        not truthy(row.get("order_dispatch_attempted"))
        and (terminal_event == "ALREADY_OPEN_POSITION" or truthy(row.get("already_open_after_signal")))
    ):
        return "post_signal_already_open_skip"
    if "already" in failed or "already" in skip or "pending" in failed or "pending" in skip:
        return "already_open_or_pending"
    if "already_open_position" in terminal_reason:
        return "post_signal_already_open_skip"
    if "stale" in failed or "stale" in skip or "signal_before_last_unblock" in skip or "candidate_age" in skip:
        return "stale_candidate"
    if "spread" in failed or "spread" in skip:
        return "final_filter_failed_spread"
    if "price" in failed or "price" in skip or "market_data" in skip or "ticker" in skip:
        return "final_filter_failed_price"
    if "max_positions" in failed or "max_position" in skip or "position_limit" in failed:
        return "max_positions_reached"
    if "max_positions" in str(row.get("entries_blocked_reason_at_scan") or "").lower():
        return "max_positions_reached"
    if truthy(row.get("entries_blocked_at_scan")):
        return "entries_blocked"
    if truthy(row.get("max_entries_per_scan_reached")):
        return "per_scan_entry_limit_reached"
    if int(numeric(row.get("candidates_ahead_count")) or 0) > 0:
        return "lower_rank_candidate_not_selected"
    if not truthy(row.get("order_dispatch_attempted")) and row.get("signal_ready_time"):
        return "unexplained_after_signal_before_dispatch"
    return "unknown_no_buy_after_signal"


def final_filter_reason(text: str, symbol: str, heartbeat_reason: str) -> str:
    skip = skip_reason_from_text(text, symbol)
    if skip:
        return skip
    upper = text.upper()
    if re.search(rf"(?<![A-Z0-9]){re.escape(normalize_symbol(symbol))}(?![A-Z0-9]).*SPREAD", upper):
        return "spread"
    if re.search(rf"(?<![A-Z0-9]){re.escape(normalize_symbol(symbol))}(?![A-Z0-9]).*(NO_USABLE_TICKER_PRICE|PRICE_MISSING|NO_MARKET_DATA)", upper):
        return "price_missing"
    return heartbeat_reason


def lifecycle_stage_trace(
    *,
    symbol: str,
    signal_ready_seen: bool,
    dispatch: int,
    ack: int,
    skip_reason: str,
    failed_reason: str,
    entries_blocked: int,
    max_entries_reached: int,
    better: list[str],
    post_signal_terminal_event: str = "",
    post_signal_terminal_reason: str = "",
) -> dict[str, str]:
    # In v67, SIGNAL_READY is emitted inside ordered_entry_candidates after the
    # symbol has entered entry_candidates, survived candidate_rejection_reasons,
    # passed per-cycle/per-minute checks, and has been selected for entry
    # evaluation. Missing dispatch after SIGNAL_READY therefore narrows the
    # disappearance point to the final-filter/placeOrder/ack segment.
    if signal_ready_seen:
        ready_stage = "seen_in_entry_candidates_inferred_from_SIGNAL_READY"
        ranking_stage = "ranked_in_ordered_entry_candidates_inferred_from_SIGNAL_READY"
        selection_stage = "selected_for_entry_evaluation_SIGNAL_READY"
    else:
        ready_stage = "not_proven_in_ready_list"
        ranking_stage = "not_proven_ranked"
        selection_stage = "not_selected_or_signal_missing"

    final_stage = "passed_or_no_explicit_final_filter"
    disappeared = ""
    if post_signal_terminal_event:
        final_stage = f"direct_observed_post_SIGNAL_READY_continue:{post_signal_terminal_event}:{post_signal_terminal_reason}"
        disappeared = "post_signal_terminal_continue"
    elif skip_reason:
        final_stage = f"blocked_after_SIGNAL_READY:{skip_reason}"
        disappeared = "final_entry_filter"
    elif failed_reason:
        final_stage = f"blocked_or_global_state_after_SIGNAL_READY:{failed_reason}"
        disappeared = "final_entry_filter_or_global_block"
    elif entries_blocked and not signal_ready_seen:
        final_stage = "entries_blocked_before_signal_ready"
        disappeared = "ready_list_before_selection"
    elif max_entries_reached and not signal_ready_seen:
        final_stage = "rate_limited_before_signal_ready"
        disappeared = "per_scan_entry_limit"

    if dispatch:
        dispatch_stage = "dispatch_attempt_seen"
        ibkr_stage = "ibkr_placeOrder_or_BUY_ORDER_SENT_seen"
    elif signal_ready_seen and not disappeared:
        dispatch_stage = "missing_after_SIGNAL_READY_before_ENTRY_ORDER_DISPATCH_ATTEMPT"
        ibkr_stage = "not_seen"
        disappeared = "unexplained_after_SIGNAL_READY_before_dispatch_attempt"
    elif better and not signal_ready_seen:
        dispatch_stage = "not_dispatched_lower_rank_candidates_ahead"
        ibkr_stage = "not_seen"
        disappeared = "ranking_selection"
    else:
        dispatch_stage = "not_seen"
        ibkr_stage = "not_seen"

    ack_stage = "ack_seen" if ack else ("not_seen_after_dispatch" if dispatch else "not_applicable_no_dispatch")
    trace = [
        f"ready_list_inferred={ready_stage}",
        f"ranking_inferred={ranking_stage}",
        f"selection_observed_or_inferred={selection_stage}",
        f"final_filter={final_stage}",
        f"dispatch_queue={dispatch_stage}",
        f"ibkr_order_submission={ibkr_stage}",
        f"ack={ack_stage}",
    ]
    return {
        "ready_list_stage": ready_stage,
        "ranking_stage": ranking_stage,
        "selection_stage": selection_stage,
        "final_filter_stage": final_stage,
        "dispatch_queue_stage": dispatch_stage,
        "ibkr_order_submission_stage": ibkr_stage,
        "order_ack_stage": ack_stage,
        "candidate_disappeared_stage": disappeared or "unknown",
        "candidate_lifecycle_trace": " | ".join(trace),
        "runtime_code_path": (
            "entry_candidates -> candidate_rejection_reasons -> ordered_entry_candidates "
            "-> SIGNAL_READY -> ENTRY_BLOCKED_LOW_SCORE/RISK_GUARD_BLOCK_ENTRY "
            "-> ib.placeOrder -> upsert_order -> BUY_ORDER_SENT"
        ),
    }


def investigate_case(target: dict[str, Any], session_date: str, evidence: EvidenceBundle) -> dict[str, Any]:
    symbol = normalize_symbol(target.get("symbol"))
    center = parse_dt(
        target.get("signal_ready_time")
        or target.get("possible_signal_time")
        or target.get("opening_range_break_time")
        or target.get("first_time_above_8pct")
    )
    sqlite_sources = sources_for_symbol(evidence.sqlite_by_symbol, symbol, center, window_minutes=90)
    recorder_sources = sources_for_symbol(evidence.recorder_by_symbol, symbol, center, window_minutes=90)
    timeline, _raw_counts = build_symbol_timeline(
        row=target,
        sqlite_sources=sqlite_sources,
        recorder_sources=recorder_sources,
        center=center,
    )
    signal_ready_ts = find_signal_ready_time(timeline, evidence.journal_lines, symbol, center)
    scan_ts, heartbeat = heartbeat_after_signal(evidence.heartbeat_states, signal_ready_ts)
    window_end = scan_ts + pd.Timedelta(minutes=5) if scan_ts is not None else None
    journal_symbol = journal_text_for_symbol(evidence.journal_lines, symbol, signal_ready_ts or center, minutes=90)
    timeline_text = event_text(timeline)
    symbol_window_lines = symbol_lines(evidence.journal_lines, symbol, signal_ready_ts, window_end)
    combined = "\n".join([timeline_text, journal_symbol, "\n".join(symbol_window_lines)])
    terminal = post_signal_terminal_evidence("\n".join([timeline_text, "\n".join(symbol_window_lines)]), symbol)
    values = extract_symbol_values(combined, symbol)
    candidate_rank = values.get("candidate_rank_at_scan") or target.get("top100_rank")
    better = better_candidates_ahead(evidence.journal_lines, symbol, signal_ready_ts, candidate_rank)
    heartbeat_reason = heartbeat_entries_blocked_reason(heartbeat)
    entries_blocked = int(truthy(heartbeat.get("entries_blocked")) or bool(heartbeat_reason))
    skip_reason = skip_reason_from_text(combined, symbol)
    failed_reason = final_filter_reason(combined, symbol, heartbeat_reason)
    dispatch = order_dispatch_attempted(combined, symbol)
    ack = order_ack_seen(combined, symbol)
    max_entries_reached = int("ENTRY_RATE_LIMIT_BLOCK" in combined.upper() or "MAX_ENTRIES_PER" in combined.upper())
    managed_open = heartbeat.get("managed_open", "")
    max_positions = heartbeat.get("max_positions") or heartbeat.get("max_open_positions") or ""
    signal_ready_seen = "SIGNAL_READY" in combined.upper()
    lifecycle = lifecycle_stage_trace(
        symbol=symbol,
        signal_ready_seen=signal_ready_seen,
        dispatch=dispatch,
        ack=ack,
        skip_reason=skip_reason,
        failed_reason=failed_reason,
        entries_blocked=entries_blocked,
        max_entries_reached=max_entries_reached,
        better=better,
        post_signal_terminal_event=terminal["post_signal_terminal_event"],
        post_signal_terminal_reason=terminal["post_signal_terminal_reason"],
    )
    row = {
        "date": session_date,
        "symbol": symbol,
        "top100_rank": target.get("top100_rank"),
        "top100_score": target.get("top100_score"),
        "possible_signal_time": target.get("possible_signal_time"),
        "signal_ready_time": iso_ts(signal_ready_ts),
        "next_entry_scan_time_after_signal": iso_ts(scan_ts),
        "candidate_present_in_ready_list_after_signal": int(
            symbol in str(heartbeat.get("ready_candidates", "")).upper()
            or symbol in str(heartbeat.get("live_ready_candidates", "")).upper()
            or "SIGNAL_READY" in combined.upper()
        ),
        "candidate_rank_at_scan": candidate_rank,
        "candidate_score_at_scan": values.get("candidate_score_at_scan"),
        "candidates_ahead_count": len(better),
        "better_candidates_ahead_symbols": ",".join(better[:20]),
        "max_entries_per_scan_reached": max_entries_reached,
        "max_positions_at_scan": max_positions,
        "managed_open_at_scan": managed_open,
        "entries_blocked_at_scan": entries_blocked,
        "entries_blocked_reason_at_scan": heartbeat_reason,
        "risk_guard_state_at_scan": f"risk_guard_block={heartbeat.get('risk_guard_block', '')};risk_guard_reason={heartbeat.get('risk_guard_reason', '')}",
        "subscription_cap_state_at_scan": f"subscription_cap_block={heartbeat.get('subscription_cap_block', '')};subscriptions_active={heartbeat.get('subscriptions_active', '')};subscriptions_cap={heartbeat.get('subscriptions_cap', '')}",
        "spread_bps_at_scan": values.get("spread_bps_at_scan"),
        "price_at_scan": values.get("price_at_scan"),
        "passed_final_entry_filters": int(not failed_reason and not entries_blocked and not max_entries_reached),
        "failed_final_entry_filter_reason": failed_reason,
        "order_dispatch_attempted": dispatch,
        "order_dispatch_skip_reason": skip_reason,
        **terminal,
        **lifecycle,
    }
    row["final_no_buy_reason"] = classify_no_buy_reason(row)
    return row


def summary_for_cases(cases: pd.DataFrame, session_date: str) -> pd.DataFrame:
    counts = Counter(cases.get("final_no_buy_reason", pd.Series(dtype=str)).fillna("unknown_no_buy_after_signal").astype(str)) if not cases.empty else Counter()
    row: dict[str, Any] = {
        "date": session_date,
        "total_runtime_signal_ready_but_no_buy": int(len(cases)),
    }
    for name in NO_BUY_CLASSIFICATIONS:
        row[name] = int(counts.get(name, 0))
    return pd.DataFrame([row])


def investigate_date(
    *,
    session_date: str,
    cases_csv: Path | None,
    sqlite_path: Path,
    recorder_dir: Path,
    journal_log: Path | None,
    output_dir: Path,
    force: bool = False,
    max_cases: int | None = None,
) -> tuple[Path, Path]:
    cases_path = output_dir / f"no_buy_after_signal_cases_{session_date}.csv"
    summary_path = output_dir / f"no_buy_after_signal_summary_{session_date}.csv"
    if cases_path.exists() and summary_path.exists() and not force:
        print(f"NBAS_SKIPPED_EXISTING date={session_date} output={cases_path}", flush=True)
        return cases_path, summary_path

    started = time.monotonic()
    targets = load_targets_for_date(session_date, cases_csv, max_cases=max_cases)
    print(f"NBAS_START date={session_date} targets={len(targets)}", flush=True)
    target_symbols = set(targets["symbol"].map(normalize_symbol).dropna().tolist()) if not targets.empty and "symbol" in targets.columns else set()
    evidence = load_evidence_bundle(
        sqlite_path=sqlite_path,
        recorder_dir=recorder_dir,
        session_date=session_date,
        journal_log=journal_log or default_journal_path(session_date),
        target_symbols=target_symbols,
    )
    load_elapsed = time.monotonic() - started
    sqlite_rows = sum(len(df) for df in evidence.sqlite_sources.values())
    recorder_rows = sum(len(df) for df in evidence.recorder_sources.values())
    print(
        f"NBAS_LOAD_EVIDENCE_DONE date={session_date} elapsed={load_elapsed:.1f} "
        f"sqlite_rows={sqlite_rows} recorder_rows={recorder_rows} journal_lines={len(evidence.journal_lines)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets.to_dict("records"), start=1):
        rows.append(investigate_case(target, session_date, evidence))
        if idx % 10 == 0 or idx == len(targets):
            elapsed = time.monotonic() - started
            print(f"NBAS_PROGRESS date={session_date} processed={idx}/{len(targets)} elapsed={elapsed:.1f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(rows, columns=CASE_COLUMNS) if rows else pd.DataFrame(columns=CASE_COLUMNS)
    summary = summary_for_cases(cases, session_date)
    cases.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)
    elapsed = time.monotonic() - started
    print(f"NBAS_DONE date={session_date} elapsed_seconds={elapsed:.1f} output={cases_path}", flush=True)
    return cases_path, summary_path


def update_all_summary(output_dir: Path, summaries: list[Path]) -> Path:
    frames = [pd.read_csv(path) for path in summaries if path.exists()]
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    else:
        out = pd.DataFrame(columns=["date", "total_runtime_signal_ready_but_no_buy", *NO_BUY_CLASSIFICATIONS])
    path = output_dir / "no_buy_after_signal_summary_ALL.csv"
    out.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investigate runtime SIGNAL_READY cases that did not become BUY orders.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Session date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for an inclusive date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for --start-date range, YYYY-MM-DD.")
    parser.add_argument("--should-have-signaled-csv", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--journal-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dates = [args.date] if args.date else list(iter_dates(args.start_date, args.end_date or args.start_date))
    summaries: list[Path] = []
    for session_date in dates:
        _cases_path, summary_path = investigate_date(
            session_date=session_date,
            cases_csv=args.should_have_signaled_csv if args.date else None,
            sqlite_path=args.sqlite_path,
            recorder_dir=args.recorder_dir,
            journal_log=args.journal_log if args.date else None,
            output_dir=args.output_dir,
            force=args.force,
            max_cases=args.max_cases,
        )
        summaries.append(summary_path)
    all_path = update_all_summary(args.output_dir, summaries)
    print(f"NBAS_SUMMARY_ALL output={all_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
