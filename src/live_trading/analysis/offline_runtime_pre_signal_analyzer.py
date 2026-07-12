from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_runner_stats,
    first_existing_column,
    fnum,
    iso_ts,
    load_session_candles,
    load_top100,
    normalize_symbol,
    parse_dt,
    safe_read_csv,
)
from src.live_trading.analysis.missed_runners_analyzer import no_signal_diagnostics
from src.live_trading.analysis.should_have_signaled_investigator import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_RECORDER_DIR,
    DEFAULT_SQLITE_PATH,
    EvidenceBundle,
    default_journal_path,
    iter_dates,
    load_evidence_bundle,
    load_targets_for_date as load_shs_targets_for_date,
    sources_for_symbol,
)
from src.live_trading.analysis.signal_replay_analyzer import build_symbol_timeline
from src.live_trading.analysis.symbol_subscription_inspector import (
    extract_last_restart_unblock_time,
    line_time,
    parse_key_values,
    read_text_lines,
)
from src.live_trading.ranking.daily_top100_builder import parquet_path


DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")

OFFLINE_CLASSIFICATION = "offline_should_have_signaled_runtime_signal_not_observed"

DIVERGENCE_STAGES = [
    "top100_mismatch",
    "subscription",
    "ticker/price",
    "runtime_state_missing",
    "session/open-price initialization",
    "first5",
    "first15",
    "OR range",
    "spread",
    "score",
    "stale state",
    "signal_sent carryover",
    "runtime feature snapshot missing",
    "runtime rejection reason not logged",
    "runtime evidence missing",
    "runtime evidence unavailable",
]

CASE_COLUMNS = [
    "date",
    "symbol",
    "top100_rank",
    "top100_score",
    "possible_signal_time",
    "offline_candle_source",
    "offline_candle_timestamp",
    "offline_open_price",
    "offline_current_price",
    "offline_high_price",
    "offline_first_5m_high",
    "offline_first_5m_high_pct",
    "offline_first_15m_high",
    "offline_first_15m_high_pct",
    "offline_or_high",
    "offline_or_low",
    "offline_or_range_pct",
    "offline_score",
    "offline_first5_pass",
    "offline_first15_pass",
    "offline_or_pass",
    "offline_breakout_pass",
    "runtime_symbol_present_in_top100",
    "runtime_contract_present",
    "runtime_ticker_present",
    "runtime_usable_price_present",
    "runtime_latest_live_update_timestamp",
    "runtime_state_present",
    "runtime_first_price_initialized",
    "runtime_first5_initialized",
    "runtime_first15_initialized",
    "runtime_open_price",
    "runtime_open_price_source",
    "runtime_first_5m_high",
    "runtime_first_15m_high",
    "runtime_or_high",
    "runtime_or_low",
    "runtime_or_range_pct",
    "runtime_live_entry_score",
    "runtime_ready",
    "runtime_signal_sent",
    "runtime_ready_since",
    "runtime_candidate_age_seconds",
    "runtime_spread_bps",
    "runtime_price",
    "runtime_price_eligibility",
    "runtime_symbol_eligibility",
    "runtime_stale_or_backfill_status",
    "runtime_entries_blocked_status",
    "runtime_candidate_rejection_reason",
    "runtime_subscription_state",
    "runtime_last_restart_unblock_time",
    "runtime_session_boundary_state",
    "runtime_evidence_source",
    "runtime_evidence_observed",
    "first_divergence_stage",
    "first_divergence_reason",
    "offline_value",
    "runtime_value",
    "confidence",
    "likely_live_impact",
    "restart_could_explain",
    "restart_mechanism",
]

SUMMARY_COLUMNS = [
    "date",
    "total_cases",
    *DIVERGENCE_STAGES,
]


@dataclass(frozen=True)
class RuntimeSnapshot:
    values: dict[str, Any]
    source: str
    observed: bool
    raw_line: str = ""


def _ts(value: Any) -> pd.Timestamp | None:
    return parse_dt(value)


def _truth(value: Any) -> int | str:
    if value in (None, ""):
        return ""
    return int(str(value).strip().lower() in {"1", "true", "yes", "y", "on"})


def _float_text(value: Any) -> str:
    num = fnum(value)
    return "" if num is None else str(num)


def _symbol_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Z0-9]){re.escape(normalize_symbol(symbol))}(?![A-Z0-9])")


def _line_near(line: str, center: pd.Timestamp | None, before_minutes: int = 120, after_minutes: int = 30) -> bool:
    if center is None:
        return True
    ts = line_time(line) or _ts(line[:32])
    if ts is None:
        return True
    return center - pd.Timedelta(minutes=before_minutes) <= ts <= center + pd.Timedelta(minutes=after_minutes)


def symbol_journal_lines(lines: list[str], symbol: str, center: pd.Timestamp | None) -> list[str]:
    pattern = _symbol_pattern(symbol)
    out: list[str] = []
    for line in lines:
        if pattern.search(line.upper()) and _line_near(line, center):
            out.append(line)
    return out


def latest_key_value_line(lines: list[str], symbol: str, event_name: str, center: pd.Timestamp | None) -> tuple[dict[str, str], str]:
    matching = [
        line for line in symbol_journal_lines(lines, symbol, center)
        if event_name.upper() in line.upper()
    ]
    if not matching:
        return {}, ""
    def distance(line: str) -> float:
        ts = line_time(line) or _ts(line[:32])
        if center is None or ts is None:
            return 0.0
        return abs((ts - center).total_seconds())
    line = min(matching, key=distance)
    return parse_key_values(line), line


def event_text(events: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{event.get('time', '')} {event.get('source', '')} {event.get('event', '')} "
        f"{event.get('reason', '')} {event.get('details', '')}"
        for event in events
    )


def load_targets_for_pre_signal(session_date: str, cases_csv: Path | None, max_cases: int | None) -> pd.DataFrame:
    if cases_csv is not None:
        cases = safe_read_csv(cases_csv)
    else:
        cases = safe_read_csv(DEFAULT_ANALYSIS_DIR / f"no_buy_after_signal_cases_{session_date}.csv")
    if not cases.empty and "final_no_buy_reason" in cases.columns:
        out = cases[cases["final_no_buy_reason"].fillna("").astype(str) == OFFLINE_CLASSIFICATION].copy()
    else:
        out = pd.DataFrame()
    if out.empty:
        shs = safe_read_csv(DEFAULT_ANALYSIS_DIR / f"should_have_signaled_cases_{session_date}.csv")
        if not shs.empty and "final_classification" in shs.columns:
            out = shs[shs["final_classification"].fillna("").astype(str).isin({
                "runtime_signal_ready_but_no_buy",
                "runtime_never_processed_symbol",
                "unknown",
            })].copy()
    if out.empty:
        out = load_shs_targets_for_date(session_date, None, None)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    if max_cases is not None:
        out = out.head(max_cases).copy()
    return out


def offline_features_for_target(
    *,
    target: dict[str, Any],
    session_date: str,
    history_dir: Path,
    min_first_5m_high_pct: float,
    min_first_15m_high_pct: float,
    min_or_range_pct: float,
) -> dict[str, Any]:
    symbol = normalize_symbol(target.get("symbol"))
    candles = load_session_candles(history_dir, symbol, session_date)
    source_path = parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), "RTH")
    stats = calculate_runner_stats(candles)
    diag = no_signal_diagnostics(
        candles,
        min_first_5m_high_pct=min_first_5m_high_pct,
        min_first_15m_high_pct=min_first_15m_high_pct,
        min_or_range_pct=min_or_range_pct,
    )
    possible = _ts(target.get("possible_signal_time") or diag.get("possible_signal_time"))
    current_price = ""
    candle_ts = ""
    if possible is not None and not candles.empty:
        rows = candles.sort_values("timestamp")
        upto = rows[rows["timestamp"] <= possible]
        row = upto.iloc[-1] if not upto.empty else rows.iloc[0]
        current_price = first_existing_column(row.to_dict(), ["close", "high", "open"])
        candle_ts = iso_ts(row.get("timestamp"))
    first15 = candles.sort_values("timestamp").head(15) if not candles.empty else pd.DataFrame()
    first5 = candles.sort_values("timestamp").head(5) if not candles.empty else pd.DataFrame()
    or_high = fnum(first15["high"].max()) if not first15.empty else None
    or_low = fnum(first15["low"].min()) if not first15.empty else None
    first5_high = fnum(first5["high"].max()) if not first5.empty else None
    first15_high = fnum(first15["high"].max()) if not first15.empty else None
    score = 0.0
    for value, weight in [
        (stats.first_5m_high_pct if stats else None, 2.0),
        (stats.first_15m_high_pct if stats else None, 2.0),
        (stats.or_range_pct if stats else None, 1.0),
    ]:
        if value is not None:
            score += float(value) * weight
    return {
        "possible_signal_time": iso_ts(possible),
        "offline_candle_source": str(source_path),
        "offline_candle_timestamp": candle_ts,
        "offline_open_price": stats.open_price if stats else "",
        "offline_current_price": current_price,
        "offline_high_price": stats.high_price if stats else "",
        "offline_first_5m_high": first5_high,
        "offline_first_5m_high_pct": stats.first_5m_high_pct if stats else "",
        "offline_first_15m_high": first15_high,
        "offline_first_15m_high_pct": stats.first_15m_high_pct if stats else "",
        "offline_or_high": or_high,
        "offline_or_low": or_low,
        "offline_or_range_pct": stats.or_range_pct if stats else "",
        "offline_score": round(score, 4),
        "offline_first5_pass": int(bool(stats and (stats.first_5m_high_pct or -999.0) >= min_first_5m_high_pct)),
        "offline_first15_pass": int(bool(stats and (stats.first_15m_high_pct or -999.0) >= min_first_15m_high_pct)),
        "offline_or_pass": int(bool(stats and (stats.or_range_pct or -999.0) >= min_or_range_pct)),
        "offline_breakout_pass": int(bool(diag.get("possible_signal_time"))),
    }


def runtime_snapshot_from_evidence(
    *,
    symbol: str,
    center: pd.Timestamp | None,
    evidence: EvidenceBundle,
    top100: pd.DataFrame,
) -> RuntimeSnapshot:
    symbol = normalize_symbol(symbol)
    sqlite_sources = sources_for_symbol(evidence.sqlite_by_symbol, symbol, center, window_minutes=120)
    recorder_sources = sources_for_symbol(evidence.recorder_by_symbol, symbol, center, window_minutes=120)
    timeline, raw_counts = build_symbol_timeline(
        row={"symbol": symbol, "possible_signal_time": iso_ts(center)},
        sqlite_sources=sqlite_sources,
        recorder_sources=recorder_sources,
        center=center,
    )
    text = event_text(timeline).upper()
    lines = symbol_journal_lines(evidence.journal_lines, symbol, center)
    journal_text = "\n".join(lines).upper()
    combined = "\n".join([text, journal_text])

    health, health_line = latest_key_value_line(evidence.journal_lines, symbol, "SYMBOL_PIPELINE_HEALTH", center)
    live, live_line = latest_key_value_line(evidence.journal_lines, symbol, "LIVE_FEATURE_DEBUG", center)
    no_price, no_price_line = latest_key_value_line(evidence.journal_lines, symbol, "NO_USABLE_TICKER_PRICE", center)
    subscription, subscription_line = latest_key_value_line(evidence.journal_lines, symbol, "TOP100_RELOAD_SUBSCRIBED", center)
    contract_failed, contract_failed_line = latest_key_value_line(evidence.journal_lines, symbol, "TOP100_RELOAD_CONTRACT_FAILED", center)
    ineligible, ineligible_line = latest_key_value_line(evidence.journal_lines, symbol, "ENTRY_SYMBOL_INELIGIBLE_SKIPPED", center)
    stale, stale_line = latest_key_value_line(evidence.journal_lines, symbol, "STALE_OR_BACKFILL_READY_SKIPPED", center)
    top100_symbols = set(top100["symbol"].map(normalize_symbol).tolist()) if not top100.empty and "symbol" in top100.columns else set()
    runtime_rows = sum(int(raw_counts.get(key, 0) or 0) for key in raw_counts)
    observed = bool(lines or runtime_rows)

    source_bits = []
    for name, line in [
        ("SYMBOL_PIPELINE_HEALTH", health_line),
        ("LIVE_FEATURE_DEBUG", live_line),
        ("NO_USABLE_TICKER_PRICE", no_price_line),
        ("TOP100_RELOAD_SUBSCRIBED", subscription_line),
        ("TOP100_RELOAD_CONTRACT_FAILED", contract_failed_line),
        ("ENTRY_SYMBOL_INELIGIBLE_SKIPPED", ineligible_line),
        ("STALE_OR_BACKFILL_READY_SKIPPED", stale_line),
    ]:
        if line:
            source_bits.append(name)
    if runtime_rows:
        source_bits.append("sqlite_or_recorder_rows")

    contract_present: Any = ""
    ticker_present: Any = ""
    state_present: Any = ""
    usable_price: Any = ""
    first_price_init: Any = ""
    first5_init: Any = ""
    first15_init: Any = ""
    ready: Any = ""
    signal_sent: Any = ""
    if health:
        contract_present = _truth(health.get("contract_present"))
        ticker_present = _truth(health.get("ticker_present"))
        usable_price = _truth(health.get("usable_price"))
        state_present = _truth(health.get("state_present"))
        first_price_init = _truth(health.get("first_price_initialized"))
        first5_init = _truth(health.get("first5_initialized"))
        first15_init = _truth(health.get("first15_initialized"))
        ready = _truth(health.get("ready"))
        signal_sent = _truth(health.get("signal_sent"))
    if subscription:
        contract_present = contract_present if contract_present != "" else 1
        ticker_present = ticker_present if ticker_present != "" else 1
    if contract_failed:
        contract_present = 0
    if no_price:
        usable_price = 0

    latest_live_ts = ""
    if live_line:
        latest_live_ts = iso_ts(line_time(live_line) or _ts(live_line[:32]))
    elif health_line:
        latest_live_ts = iso_ts(line_time(health_line) or _ts(health_line[:32]))

    values = {
        "runtime_symbol_present_in_top100": int(symbol in top100_symbols) if top100_symbols else "",
        "runtime_contract_present": contract_present,
        "runtime_ticker_present": ticker_present,
        "runtime_usable_price_present": usable_price,
        "runtime_latest_live_update_timestamp": latest_live_ts,
        "runtime_state_present": state_present,
        "runtime_first_price_initialized": first_price_init,
        "runtime_first5_initialized": first5_init,
        "runtime_first15_initialized": first15_init,
        "runtime_open_price": live.get("open_price") or live.get("first_price") or "",
        "runtime_open_price_source": "LIVE_FEATURE_DEBUG" if live else "",
        "runtime_first_5m_high": live.get("first_5m_high") or "",
        "runtime_first_15m_high": live.get("first_15m_high") or "",
        "runtime_or_high": live.get("or_high") or "",
        "runtime_or_low": live.get("or_low") or "",
        "runtime_or_range_pct": live.get("or_range_pct") or "",
        "runtime_live_entry_score": live.get("score") or "",
        "runtime_ready": ready,
        "runtime_signal_sent": signal_sent,
        "runtime_ready_since": live.get("ready_since") or "",
        "runtime_candidate_age_seconds": live.get("candidate_age_seconds") or "",
        "runtime_spread_bps": live.get("spread_bps") or no_price.get("spread_bps") or "",
        "runtime_price": live.get("current_price") or live.get("last_price") or no_price.get("last") or no_price.get("midpoint") or "",
        "runtime_price_eligibility": "no_usable_price" if no_price else ("observed_price" if live else ""),
        "runtime_symbol_eligibility": ineligible.get("reason") if ineligible else ("contract_failed" if contract_failed else ""),
        "runtime_stale_or_backfill_status": stale.get("reason") if stale else "",
        "runtime_entries_blocked_status": "",
        "runtime_candidate_rejection_reason": live.get("reason") or ineligible.get("reason") or stale.get("reason") or "",
        "runtime_subscription_state": "contract_failed" if contract_failed else ("subscribed" if subscription else ""),
        "runtime_last_restart_unblock_time": iso_ts(extract_last_restart_unblock_time(evidence.journal_lines)),
        "runtime_session_boundary_state": "",
    }

    heartbeat_lines = [line for line in evidence.journal_lines if "heartbeat" in line.lower()]
    if heartbeat_lines and center is not None:
        nearest = min(
            heartbeat_lines,
            key=lambda line: abs(((line_time(line) or center) - center).total_seconds()),
        )
        hb = parse_key_values(nearest)
        values["runtime_entries_blocked_status"] = hb.get("entries_blocked") or ""
        values["runtime_session_boundary_state"] = (
            f"top100_block={hb.get('top100_block', '')};restart_block={hb.get('restart_block', '')};"
            f"subscription_cap_block={hb.get('subscription_cap_block', '')};managed_open={hb.get('managed_open', '')}"
        )

    return RuntimeSnapshot(
        values=values,
        source=";".join(source_bits) if source_bits else "",
        observed=observed,
        raw_line=(health_line or live_line or no_price_line or contract_failed_line or subscription_line or ineligible_line or stale_line)[:1000],
    )


def divergence_for_row(offline: dict[str, Any], runtime: RuntimeSnapshot) -> dict[str, Any]:
    values = runtime.values
    if values.get("runtime_symbol_present_in_top100") == 0:
        return {
            "first_divergence_stage": "top100_mismatch",
            "first_divergence_reason": "symbol_not_in_runtime_top100_file",
            "offline_value": "top100_expected",
            "runtime_value": "not_in_top100",
            "confidence": "high",
            "likely_live_impact": "symbol would not be considered for entry",
            "restart_could_explain": "uncertain",
            "restart_mechanism": "restart reloads latest Top100; continuous runtime may have stale Top100 if refresh failed",
        }
    if not runtime.observed:
        return {
            "first_divergence_stage": "runtime evidence missing",
            "first_divergence_reason": "no symbol-specific recorder/sqlite/journal evidence near offline signal time",
            "offline_value": offline.get("possible_signal_time", ""),
            "runtime_value": "missing",
            "confidence": "medium",
            "likely_live_impact": "cannot prove subscription/state existed; runtime may never have processed symbol",
            "restart_could_explain": "yes",
            "restart_mechanism": "restart rebuilds subscriptions and symbol states; missing evidence can indicate a stale reload/subscription gap",
        }
    if values.get("runtime_contract_present") == 0 or values.get("runtime_ticker_present") == 0 or values.get("runtime_subscription_state") == "contract_failed":
        return {
            "first_divergence_stage": "subscription",
            "first_divergence_reason": values.get("runtime_subscription_state") or "contract_or_ticker_missing",
            "offline_value": "top100_symbol",
            "runtime_value": f"contract={values.get('runtime_contract_present')} ticker={values.get('runtime_ticker_present')}",
            "confidence": "high",
            "likely_live_impact": "no ticker means no live features and no SIGNAL_READY",
            "restart_could_explain": "yes",
            "restart_mechanism": "startup qualification/subscription path may differ from Top100 refresh reconciliation",
        }
    if values.get("runtime_usable_price_present") == 0 or values.get("runtime_price_eligibility") == "no_usable_price":
        return {
            "first_divergence_stage": "ticker/price",
            "first_divergence_reason": "runtime_no_usable_price",
            "offline_value": offline.get("offline_current_price", ""),
            "runtime_value": values.get("runtime_price") or "missing_price",
            "confidence": "high",
            "likely_live_impact": "compute_live_safe_features cannot become ready without price",
            "restart_could_explain": "yes",
            "restart_mechanism": "restart can recreate ticker subscriptions and clear stale tickers without usable price",
        }
    if values.get("runtime_state_present") == 0:
        return {
            "first_divergence_stage": "runtime_state_missing",
            "first_divergence_reason": "ticker/contract existed but SymbolState missing",
            "offline_value": "offline_features_ready",
            "runtime_value": "state_missing",
            "confidence": "high",
            "likely_live_impact": "without state, first5/first15/OR cannot initialize",
            "restart_could_explain": "yes",
            "restart_mechanism": "startup creates states for subscribed symbols; reload path may miss state creation",
        }
    if values.get("runtime_signal_sent") == 1:
        return {
            "first_divergence_stage": "signal_sent carryover",
            "first_divergence_reason": "runtime state already had signal_sent=1 before offline expected signal",
            "offline_value": "new_day_signal_expected",
            "runtime_value": "signal_sent=1",
            "confidence": "high",
            "likely_live_impact": "entry candidate gate requires not state.signal_sent",
            "restart_could_explain": "yes",
            "restart_mechanism": "restart clears in-memory signal_sent unless restored from stale state",
        }
    if values.get("runtime_first_price_initialized") == 0:
        return {
            "first_divergence_stage": "session/open-price initialization",
            "first_divergence_reason": "runtime first/open price not initialized",
            "offline_value": offline.get("offline_open_price", ""),
            "runtime_value": values.get("runtime_open_price") or "missing",
            "confidence": "high",
            "likely_live_impact": "feature percentages are None without first/open price",
            "restart_could_explain": "yes",
            "restart_mechanism": "restart may rebuild state from fresh post-open ticks/candles",
        }
    if values.get("runtime_first5_initialized") == 0:
        return {
            "first_divergence_stage": "first5",
            "first_divergence_reason": "runtime first_5m_high not initialized",
            "offline_value": offline.get("offline_first_5m_high_pct", ""),
            "runtime_value": "first5_missing",
            "confidence": "high",
            "likely_live_impact": "ready requires first_5m_high_pct",
            "restart_could_explain": "uncertain",
            "restart_mechanism": "restart after first 5m may also miss first5 unless state rebuild supplies candles",
        }
    if values.get("runtime_first15_initialized") == 0:
        return {
            "first_divergence_stage": "first15",
            "first_divergence_reason": "runtime first_15m_high not initialized",
            "offline_value": offline.get("offline_first_15m_high_pct", ""),
            "runtime_value": "first15_missing",
            "confidence": "high",
            "likely_live_impact": "ready requires first_15m_high_pct",
            "restart_could_explain": "uncertain",
            "restart_mechanism": "restart after first 15m depends on state rebuild from candles",
        }
    reason = str(values.get("runtime_candidate_rejection_reason") or "")
    if "spread" in reason.lower():
        return {
            "first_divergence_stage": "spread",
            "first_divergence_reason": reason,
            "offline_value": "offline replay has no reliable live spread",
            "runtime_value": values.get("runtime_spread_bps", ""),
            "confidence": "medium",
            "likely_live_impact": "ready requires spread <= max_spread_bps when spread is present",
            "restart_could_explain": "no",
            "restart_mechanism": "spread is market-state dependent, not reset-state dependent",
        }
    if values.get("runtime_stale_or_backfill_status"):
        return {
            "first_divergence_stage": "stale state",
            "first_divergence_reason": values.get("runtime_stale_or_backfill_status"),
            "offline_value": offline.get("possible_signal_time", ""),
            "runtime_value": values.get("runtime_ready_since", ""),
            "confidence": "medium",
            "likely_live_impact": "stale/backfill candidates are skipped before order dispatch",
            "restart_could_explain": "yes",
            "restart_mechanism": "last_restart_unblock_time can make earlier ready states stale after restart",
        }
    if values.get("runtime_live_entry_score") == "" and values.get("runtime_first_5m_high") == "":
        return {
            "first_divergence_stage": "runtime feature snapshot missing",
            "first_divergence_reason": "no observed live feature values for symbol near offline signal time",
            "offline_value": json.dumps({
                "first5": offline.get("offline_first_5m_high_pct"),
                "first15": offline.get("offline_first_15m_high_pct"),
                "or": offline.get("offline_or_range_pct"),
            }, default=str),
            "runtime_value": "missing_feature_snapshot",
            "confidence": "medium",
            "likely_live_impact": "runtime may have processed symbol but did not persist enough pre-signal state to explain rejection",
            "restart_could_explain": "uncertain",
            "restart_mechanism": "missing diagnostics may hide reload/state issue that restart resets",
        }
    return {
        "first_divergence_stage": "runtime rejection reason not logged",
        "first_divergence_reason": reason or "runtime_values_observed_but_no_SIGNAL_READY_or_explicit_rejection",
        "offline_value": "offline_ready",
        "runtime_value": json.dumps({
            "ready": values.get("runtime_ready"),
            "score": values.get("runtime_live_entry_score"),
            "reason": reason,
        }, default=str),
        "confidence": "low",
        "likely_live_impact": "need pre-SIGNAL_READY diagnostics to distinguish missing rejection from logic divergence",
        "restart_could_explain": "uncertain",
        "restart_mechanism": "observed evidence is insufficient; restart hypothesis remains plausible but unproven",
    }


def analyze_case(
    *,
    target: dict[str, Any],
    session_date: str,
    history_dir: Path,
    evidence: EvidenceBundle,
    top100: pd.DataFrame,
    min_first_5m_high_pct: float,
    min_first_15m_high_pct: float,
    min_or_range_pct: float,
) -> dict[str, Any]:
    symbol = normalize_symbol(target.get("symbol"))
    offline = offline_features_for_target(
        target=target,
        session_date=session_date,
        history_dir=history_dir,
        min_first_5m_high_pct=min_first_5m_high_pct,
        min_first_15m_high_pct=min_first_15m_high_pct,
        min_or_range_pct=min_or_range_pct,
    )
    center = _ts(offline.get("possible_signal_time") or target.get("possible_signal_time"))
    runtime = runtime_snapshot_from_evidence(symbol=symbol, center=center, evidence=evidence, top100=top100)
    divergence = divergence_for_row(offline, runtime)
    return {
        "date": session_date,
        "symbol": symbol,
        "top100_rank": target.get("top100_rank"),
        "top100_score": target.get("top100_score"),
        **offline,
        **runtime.values,
        "runtime_evidence_source": runtime.source,
        "runtime_evidence_observed": int(runtime.observed),
        **divergence,
    }


def summary_for_cases(cases: pd.DataFrame, session_date: str) -> pd.DataFrame:
    counts = Counter(cases.get("first_divergence_stage", pd.Series(dtype=str)).fillna("runtime evidence unavailable").astype(str)) if not cases.empty else Counter()
    row: dict[str, Any] = {"date": session_date, "total_cases": int(len(cases))}
    for stage in DIVERGENCE_STAGES:
        row[stage] = int(counts.get(stage, 0))
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def investigate_date(
    *,
    session_date: str,
    cases_csv: Path | None,
    sqlite_path: Path,
    recorder_dir: Path,
    history_dir: Path,
    top100_path: Path | None,
    journal_log: Path | None,
    output_dir: Path,
    force: bool = False,
    max_cases: int | None = None,
    min_first_5m_high_pct: float = 0.5,
    min_first_15m_high_pct: float = 1.0,
    min_or_range_pct: float = 0.5,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / f"offline_runtime_pre_signal_cases_{session_date}.csv"
    summary_path = output_dir / f"offline_runtime_pre_signal_summary_{session_date}.csv"
    if cases_path.exists() and summary_path.exists() and not force:
        print(f"PRE_SIGNAL_SKIPPED_EXISTING date={session_date} output={cases_path}", flush=True)
        return cases_path, summary_path

    started = time.monotonic()
    targets = load_targets_for_pre_signal(session_date, cases_csv, max_cases)
    target_symbols = set(targets["symbol"].map(normalize_symbol).dropna().tolist()) if not targets.empty and "symbol" in targets.columns else set()
    print(f"PRE_SIGNAL_START date={session_date} targets={len(targets)}", flush=True)
    evidence = load_evidence_bundle(
        sqlite_path=sqlite_path,
        recorder_dir=recorder_dir,
        session_date=session_date,
        journal_log=journal_log,
        target_symbols=target_symbols,
    )
    top100 = load_top100(top100_path or Path(f"data/universe/daily_top100_{session_date}.csv"))
    if top100.empty:
        top100 = load_top100(Path("data/universe/daily_top100_latest.csv"))
    print(
        f"PRE_SIGNAL_LOAD_EVIDENCE_DONE date={session_date} elapsed={time.monotonic() - started:.1f} "
        f"sqlite_rows={sum(len(df) for df in evidence.sqlite_sources.values())} "
        f"recorder_rows={sum(len(df) for df in evidence.recorder_sources.values())} "
        f"journal_lines={len(evidence.journal_lines)} top100_symbols={len(top100)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets.to_dict("records"), start=1):
        rows.append(
            analyze_case(
                target=target,
                session_date=session_date,
                history_dir=history_dir,
                evidence=evidence,
                top100=top100,
                min_first_5m_high_pct=min_first_5m_high_pct,
                min_first_15m_high_pct=min_first_15m_high_pct,
                min_or_range_pct=min_or_range_pct,
            )
        )
        if idx % 10 == 0 or idx == len(targets):
            print(f"PRE_SIGNAL_PROGRESS date={session_date} processed={idx}/{len(targets)} elapsed={time.monotonic() - started:.1f}", flush=True)
    cases = pd.DataFrame(rows, columns=CASE_COLUMNS) if rows else pd.DataFrame(columns=CASE_COLUMNS)
    summary = summary_for_cases(cases, session_date)
    cases.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"PRE_SIGNAL_DONE date={session_date} elapsed_seconds={time.monotonic() - started:.1f} output={cases_path}", flush=True)
    return cases_path, summary_path


def update_all_summary(output_dir: Path, summaries: list[Path]) -> Path:
    frames = [pd.read_csv(path) for path in summaries if path.exists()]
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    else:
        out = pd.DataFrame(columns=SUMMARY_COLUMNS)
    path = output_dir / "offline_runtime_pre_signal_summary_ALL.csv"
    out.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare offline should-have-signaled cases with observed runtime pre-SIGNAL_READY evidence.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Session date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for --start-date range, YYYY-MM-DD.")
    parser.add_argument("--cases-csv", type=Path, default=None, help="Optional NBAS/SHS cases CSV for a single date.")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--journal-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-first-5m-high-pct", type=float, default=0.5)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=1.0)
    parser.add_argument("--min-or-range-pct", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dates = [args.date] if args.date else list(iter_dates(args.start_date, args.end_date or args.start_date))
    summaries: list[Path] = []
    for session_date in dates:
        _cases, summary = investigate_date(
            session_date=session_date,
            cases_csv=args.cases_csv if args.date else None,
            sqlite_path=args.sqlite_path,
            recorder_dir=args.recorder_dir,
            history_dir=args.history_dir,
            top100_path=args.top100 if args.date else None,
            journal_log=args.journal_log if args.date else None,
            output_dir=args.output_dir,
            force=args.force,
            max_cases=args.max_cases,
            min_first_5m_high_pct=args.min_first_5m_high_pct,
            min_first_15m_high_pct=args.min_first_15m_high_pct,
            min_or_range_pct=args.min_or_range_pct,
        )
        summaries.append(summary)
    all_path = update_all_summary(args.output_dir, summaries)
    print(f"PRE_SIGNAL_SUMMARY_ALL output={all_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
