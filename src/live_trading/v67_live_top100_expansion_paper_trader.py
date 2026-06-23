from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import signal
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.control.control_api import process_control_api_commands, process_history_collector_commands, start_control_api
from src.live_trading.market_calendar import get_us_equity_session, is_us_equity_trading_day, previous_us_equity_trading_day
from src.live_trading.ineligible_symbols import (
    DEFAULT_RUNTIME_INELIGIBLE,
    DEFAULT_SYMBOL_DENYLIST,
    combined_ineligible_symbols,
    contract_ineligible_reason,
    contract_metadata,
    record_runtime_ineligible,
)
from src.live_trading.v66_ibkr_account_recorder import (
    install_commission_report_handler,
    record_account_snapshot,
    record_recent_fills,
)
from src.live_trading.order_lifecycle.models import (
    ExecutionRecord,
    LifecycleEvent,
    LifecycleEventType,
    OrderSide,
    OrderState,
    PositionRecord,
    PositionState,
)
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore
from src.live_trading.order_lifecycle.reducer import reduce_lifecycle_events
from src.live_trading.order_lifecycle.reconciliation import build_reconciliation_report, log_reconciliation_report
from src.live_trading.storage.sqlite_store import open_sqlite_store, safe_sqlite_call
from src.live_trading.unified_logger import (
    current_git_commit,
    emit_unified_log_line,
    format_traceback,
    install_unified_logger,
    log_event,
    monitor_disk_usage,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 65
DEFAULT_ALPHA_RANK = "data/universe/daily_top100_latest.csv"
DEFAULT_UNIVERSE = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_HISTORY_DIR = "data/history/universe_1m"
DEFAULT_RECORDER_DIR = "data/live/recorder"
STRATEGY_NAME = "v67_top100_live_safe_expansion_v46_wide_trail"
_ACTIVE_SHUTDOWN_DIAGNOSTICS: "ShutdownDiagnostics | None" = None


@dataclass
class SymbolState:
    symbol: str
    open_price: float | None = None
    first_price: float | None = None
    high: float | None = None
    low: float | None = None
    last_price: float | None = None
    first_5m_high: float | None = None
    first_15m_high: float | None = None
    or_high: float | None = None
    or_low: float | None = None
    first_seen_ts: float | None = None
    first_seen_utc: str | None = None
    latest_seen_utc: str | None = None
    latest_volume: float | None = None
    signal_sent: bool = False
    ready_since_ts: float | None = None
    ready_since_utc: str | None = None
    signal_source: str = ""
    last_update_source: str = ""
    last_live_update_ts: float | None = None
    last_live_update_utc: str | None = None
    stale_ready_logged: bool = False
    bars: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ManagedPosition:
    symbol: str
    contract: Any
    quantity: int
    entry_price: float
    entry_time: str
    peak_price: float
    low_price: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    peak_unrealized_pnl: float | None = None
    max_adverse_unrealized_pnl: float | None = None
    last_update_time: str | None = None
    active: bool = True
    exit_sent: bool = False
    source: str = "live_buy"
    exit_order_id: int | None = None
    last_exit_order_ts: float | None = None
    eod_retry_count: int = 0
    entry_fill_verified: bool = False


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_us_equity_session_active_now(args: argparse.Namespace | None = None, *, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    session = get_us_equity_session(now.date())
    if not session.is_trading_day:
        return False
    if session.open_utc and session.close_utc:
        return session.open_utc <= now < session.close_utc
    market_open = parse_utc_hhmm(getattr(args, "market_open_utc", "13:30") if args is not None else "13:30")
    market_close = parse_utc_hhmm(getattr(args, "market_close_utc", "20:00") if args is not None else "20:00")
    now_min = now.hour * 60 + now.minute
    return market_open <= now_min < market_close


class ShutdownDiagnostics:
    def __init__(self, *, log_dir: str | Path | None, args: argparse.Namespace | None = None) -> None:
        self.log_dir = log_dir
        self.args = args
        self.start_monotonic = time.monotonic()
        self.reason = "unknown"
        self.exit_code: int | None = None
        self.signal_name = ""
        self.exit_logged = False
        self.atexit_logged = False

    def uptime_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.start_monotonic)

    def set_reason(self, reason: str, *, exit_code: int | None = None) -> None:
        self.reason = reason or self.reason or "unknown"
        if exit_code is not None:
            self.exit_code = int(exit_code)

    def log_signal(self, signum: int) -> None:
        try:
            self.signal_name = signal.Signals(signum).name
        except Exception:
            self.signal_name = f"SIG{signum}"
        self.set_reason(f"signal_{self.signal_name.lower()}", exit_code=0)
        log_event(
            "BOT",
            "BOT_SIGNAL_RECEIVED",
            "WARN",
            log_dir=self.log_dir,
            signal=self.signal_name,
            uptime_seconds=round(self.uptime_seconds(), 3),
        )

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self.log_signal(int(signum))
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(0)

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(signum, _handler)
            except Exception:
                pass

    def log_main_loop_exit(self, *, reason: str, exit_code: int = 0) -> None:
        self.set_reason(reason, exit_code=exit_code)
        log_event(
            "BOT",
            "MAIN_LOOP_EXIT",
            log_dir=self.log_dir,
            reason=reason,
            exit_code=exit_code,
            uptime_seconds=round(self.uptime_seconds(), 3),
        )

    def log_exit(self, *, reason: str | None = None, exit_code: int | None = None, recorder_dir: Any = None) -> None:
        if reason:
            self.set_reason(reason)
        if exit_code is not None:
            self.exit_code = int(exit_code)
        if self.exit_logged:
            return
        self.exit_logged = True
        code = 0 if self.exit_code is None else int(self.exit_code)
        log_event(
            "BOT",
            "BOT_EXIT",
            log_dir=self.log_dir,
            reason=self.reason or "unknown",
            exit_code=code,
            signal=self.signal_name,
            uptime_seconds=round(self.uptime_seconds(), 3),
            recorder_dir=recorder_dir or "",
        )
        log_event(
            "BOT",
            "BOT_STOP",
            log_dir=self.log_dir,
            reason=self.reason or "unknown",
            exit_code=code,
            signal=self.signal_name,
            recorder_dir=recorder_dir or "",
        )
        if code == 0 and is_us_equity_session_active_now(self.args):
            log_event(
                "BOT",
                "UNEXPECTED_CLEAN_EXIT_DURING_SESSION",
                "WARN",
                log_dir=self.log_dir,
                reason=self.reason or "unknown",
                uptime_seconds=round(self.uptime_seconds(), 3),
            )

    def atexit(self) -> None:
        if self.atexit_logged:
            return
        self.atexit_logged = True
        if not self.exit_logged:
            self.log_exit(reason=f"atexit_{self.reason or 'unknown'}", exit_code=self.exit_code)
        else:
            log_event(
                "BOT",
                "BOT_EXIT_ATEXIT",
                log_dir=self.log_dir,
                reason=self.reason or "unknown",
                exit_code=0 if self.exit_code is None else self.exit_code,
                signal=self.signal_name,
                uptime_seconds=round(self.uptime_seconds(), 3),
            )


def log_ibkr_disconnect_source(
    runtime_state: dict[str, Any] | None,
    *,
    source: str,
    reason: str,
    log_dir: str | Path | None = None,
    **fields: Any,
) -> None:
    if runtime_state is not None:
        runtime_state["ibkr_disconnect_source"] = source
        runtime_state["ibkr_disconnect_reason"] = reason
        runtime_state["ibkr_disconnect_at"] = now_utc()
    log_event("IBKR", "IBKR_DISCONNECT_SOURCE", log_dir=log_dir, source=source, reason=reason, **fields)


def emit_heartbeat(line: str, runtime_state: dict[str, Any], log_dir: str | Path | None = None) -> None:
    now_ts = time.monotonic()
    previous_ts = runtime_state.get("unified_log_last_heartbeat_monotonic")
    if previous_ts is not None:
        try:
            gap = now_ts - float(previous_ts)
        except Exception:
            gap = 0.0
        if gap > 30.0:
            try:
                log_event(
                    "LOG",
                    "LOG_GAP_WARNING",
                    "WARN",
                    log_dir=log_dir,
                    monitored_event="heartbeat",
                    gap_seconds=round(gap, 3),
                )
            except Exception:
                pass
    runtime_state["unified_log_last_heartbeat_monotonic"] = now_ts
    emit_unified_log_line(line, log_dir=log_dir)


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def append_dict_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def record_contract_metadata(recorder: LiveDataRecorder, contract: Any, *, source: str) -> None:
    metadata = contract_metadata(contract)
    metadata["recorded_at"] = now_utc()
    metadata["source"] = source
    append_dict_csv(
        recorder.path("contract_metadata.csv"),
        metadata,
        [
            "recorded_at",
            "source",
            "symbol",
            "conId",
            "secType",
            "longName",
            "category",
            "industry",
            "primaryExchange",
            "tradingClass",
        ],
    )


def load_top_symbols(alpha_rank_csv: str, top_n: int, min_price: float | None = None) -> list[str]:
    p = Path(alpha_rank_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing alpha rank file: {alpha_rank_csv}")
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("alpha rank csv must contain symbol column")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if "alpha_score" in df.columns:
        df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("alpha_score", ascending=False)
    if min_price is not None and "last_close" in df.columns:
        df["last_close"] = pd.to_numeric(df["last_close"], errors="coerce")
        df = df[df["last_close"] >= min_price]
    return df["symbol"].dropna().drop_duplicates().head(top_n).tolist()


def load_tradeable_top_symbols(
    alpha_rank_csv: str,
    top_n: int,
    *,
    min_price: float | None = None,
    symbol_denylist_path: str | Path | None = DEFAULT_SYMBOL_DENYLIST,
    runtime_ineligible_path: str | Path | None = DEFAULT_RUNTIME_INELIGIBLE,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    symbols = load_top_symbols(alpha_rank_csv, top_n, min_price=min_price)
    ineligible = combined_ineligible_symbols(symbol_denylist_path, runtime_ineligible_path)
    selected: list[str] = []
    skipped: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        info = ineligible.get(symbol)
        if info:
            skipped[symbol] = info
            continue
        selected.append(symbol)
    return selected, skipped


def runtime_ineligible_info(runtime_state: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbol = str(symbol or "").upper().strip()
    reasons = _runtime_dict(runtime_state, "ineligible_symbol_reasons")
    if symbol in reasons and isinstance(reasons[symbol], dict):
        return reasons[symbol]
    if symbol in _runtime_set(runtime_state, "ineligible_symbols"):
        return {"reason": "kid_priip_ineligible", "source": "runtime_cache"}
    return {}


def mark_runtime_symbol_ineligible(
    runtime_state: dict[str, Any],
    symbol: str,
    *,
    reason: str,
    source: str,
    con_id: Any = None,
    ibkr_error_code: Any = None,
    raw_message: str = "",
    persist: bool = True,
) -> None:
    symbol = str(symbol or "").upper().strip()
    if not symbol:
        return
    _runtime_set(runtime_state, "ineligible_symbols").add(symbol)
    info = {
        "symbol": symbol,
        "reason": reason,
        "source": source,
        "conId": con_id,
        "ibkr_error_code": ibkr_error_code,
        "raw_message": raw_message,
    }
    _runtime_dict(runtime_state, "ineligible_symbol_reasons")[symbol] = info
    if persist:
        try:
            record_runtime_ineligible(
                symbol,
                con_id=con_id,
                reason=reason,
                ibkr_error_code=ibkr_error_code,
                raw_message=raw_message,
                source=source,
                path=runtime_state.get("runtime_ineligible_path") or DEFAULT_RUNTIME_INELIGIBLE,
            )
        except Exception as exc:
            print(f"{now_utc()} RUNTIME_INELIGIBLE_PERSIST_FAILED symbol={symbol} error={exc!r}", flush=True)


def snapshot_from_ticker(symbol: str, ticker: Any) -> dict[str, Any]:
    bid = safe_float(getattr(ticker, "bid", None))
    ask = safe_float(getattr(ticker, "ask", None))
    last = safe_float(getattr(ticker, "last", None))
    close = safe_float(getattr(ticker, "close", None))
    volume = safe_float(getattr(ticker, "volume", None))
    bid_size = safe_float(getattr(ticker, "bidSize", None))
    ask_size = safe_float(getattr(ticker, "askSize", None))

    mid = None
    spread = None
    spread_bps = None
    if bid is not None and ask is not None and ask >= bid and bid > 0:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10_000.0 if mid else None

    price = last or mid or close or bid or ask
    return {
        "symbol": symbol,
        "price": price,
        "observed_at": now_utc(),
        "bid": bid,
        "ask": ask,
        "last": last,
        "close": close,
        "volume": volume,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid_price": mid,
        "spread": spread,
        "spread_bps": spread_bps,
    }


def market_open_datetime_utc(args: argparse.Namespace, now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
    return now.replace(hour=hh, minute=mm, second=0, microsecond=0)


def update_state(
    state: SymbolState,
    snap: dict[str, Any],
    session_elapsed: float,
    opening_range_seconds: int,
    *,
    observed_at: datetime | None = None,
    source: str = "live",
) -> None:
    price = safe_float(snap.get("price"))
    if price is None or price <= 0:
        return
    observed_at = observed_at or datetime.now(timezone.utc)
    now_ts = observed_at.timestamp()
    observed_iso = observed_at.isoformat()
    state.last_price = price
    state.latest_seen_utc = observed_iso
    state.latest_volume = safe_float(snap.get("volume"))
    state.last_update_source = source
    if source == "live":
        state.last_live_update_ts = now_ts
        state.last_live_update_utc = observed_iso
    state.bars.append(
        {
            "bar_time_utc": observed_iso,
            "price": price,
            "session_elapsed_seconds": round(float(session_elapsed), 3),
            "source": "live_ticker_snapshot" if source == "live" else source,
        }
    )
    if len(state.bars) > 500:
        del state.bars[:-500]

    if session_elapsed < 0:
        return

    if state.first_seen_ts is None:
        state.first_seen_ts = now_ts
        state.first_seen_utc = observed_iso
        state.first_price = price
        state.open_price = price
        state.high = price
        state.low = price
    state.high = max(state.high or price, price)
    state.low = min(state.low or price, price)
    if 0 <= session_elapsed < 5 * 60:
        state.first_5m_high = max(state.first_5m_high or price, price)
    if 0 <= session_elapsed < 15 * 60:
        state.first_15m_high = max(state.first_15m_high or price, price)
    if 0 <= session_elapsed < opening_range_seconds:
        state.or_high = max(state.or_high or price, price)
        state.or_low = min(state.or_low or price, price)


def pct_from(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base <= 0:
        return None
    return (value / base - 1.0) * 100.0


def compute_live_safe_features(state: SymbolState, snap: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    first = state.first_price or state.open_price
    last = state.last_price or safe_float(snap.get("price"))
    first_5m_high_pct = pct_from(first, state.first_5m_high)
    first_15m_high_pct = pct_from(first, state.first_15m_high)
    or_range_pct = None
    if state.or_low is not None and state.or_high is not None and state.or_low > 0:
        or_range_pct = (state.or_high / state.or_low - 1.0) * 100.0

    price = last
    spread_bps = safe_float(snap.get("spread_bps"))
    ready = (
        first_5m_high_pct is not None
        and first_15m_high_pct is not None
        and or_range_pct is not None
        and first_5m_high_pct >= args.min_first_5m_high_pct
        and first_15m_high_pct >= args.min_first_15m_high_pct
        and or_range_pct >= args.min_or_range_pct
        and price is not None
        and price >= args.min_price
        and (spread_bps is None or spread_bps <= args.max_spread_bps)
    )

    score = 0.0
    for value, weight in [(first_5m_high_pct, 2.0), (first_15m_high_pct, 2.0), (or_range_pct, 1.0)]:
        if value is not None:
            score += value * weight
    if spread_bps is not None and args.max_spread_bps > 0:
        score += max(0.0, args.max_spread_bps - spread_bps) / args.max_spread_bps * 5.0

    reasons: list[str] = []
    if first_5m_high_pct is None or first_5m_high_pct < args.min_first_5m_high_pct:
        reasons.append("first_5m_high_too_low")
    if first_15m_high_pct is None or first_15m_high_pct < args.min_first_15m_high_pct:
        reasons.append("first_15m_high_too_low")
    if or_range_pct is None or or_range_pct < args.min_or_range_pct:
        reasons.append("or_range_too_low")
    if price is None or price < args.min_price:
        reasons.append("price_too_low")
    if spread_bps is not None and spread_bps > args.max_spread_bps:
        reasons.append("spread_too_wide")

    return {
        "ready": ready,
        "score": round(score, 4),
        "reason": ";".join(reasons) if reasons else "live_safe_expansion_ready",
        "first_5m_high_pct": first_5m_high_pct,
        "first_15m_high_pct": first_15m_high_pct,
        "or_range_pct": or_range_pct,
        "entry_price": price,
        "spread_bps": spread_bps,
    }


def _feature_value_is_zeroish(value: Any) -> bool:
    numeric = safe_float(value)
    return numeric is None or abs(numeric) <= 1e-9


def features_are_all_zeroish(features: dict[str, Any]) -> bool:
    return (
        _feature_value_is_zeroish(features.get("first_5m_high_pct"))
        and _feature_value_is_zeroish(features.get("first_15m_high_pct"))
        and _feature_value_is_zeroish(features.get("or_range_pct"))
        and _feature_value_is_zeroish(features.get("score"))
    )


def mark_entry_block_state(runtime_state: dict[str, Any], entries_blocked: bool, now_ts: float | None = None) -> None:
    """Track the moment entries move from blocked to unblocked."""
    now_ts = time.time() if now_ts is None else float(now_ts)
    previous = runtime_state.get("entry_previous_entries_blocked")
    if previous is None:
        runtime_state["entry_previous_entries_blocked"] = bool(entries_blocked)
        if not entries_blocked:
            runtime_state.setdefault("last_unblock_timestamp", now_ts)
            runtime_state.setdefault("last_unblock_utc", now_utc())
            runtime_state.setdefault("last_restart_unblock_timestamp", now_ts)
            runtime_state.setdefault("last_restart_unblock_utc", runtime_state.get("last_unblock_utc") or now_utc())
        return
    if bool(previous) and not entries_blocked:
        runtime_state["last_unblock_timestamp"] = now_ts
        runtime_state["last_unblock_utc"] = now_utc()
        runtime_state["last_restart_unblock_timestamp"] = now_ts
        runtime_state["last_restart_unblock_utc"] = runtime_state["last_unblock_utc"]
    runtime_state["entry_previous_entries_blocked"] = bool(entries_blocked)


def ready_candidate_age_seconds(state: SymbolState, now_ts: float | None = None) -> float | None:
    if state.ready_since_ts is None:
        return None
    now_ts = time.time() if now_ts is None else float(now_ts)
    return max(0.0, now_ts - float(state.ready_since_ts))


def is_stale_ready_candidate(
    state: SymbolState,
    runtime_state: dict[str, Any],
    *,
    max_age_seconds: float,
    now_ts: float | None = None,
) -> bool:
    if state.ready_since_ts is None:
        return False
    now_ts = time.time() if now_ts is None else float(now_ts)
    age = ready_candidate_age_seconds(state, now_ts) or 0.0
    last_unblock = float(runtime_state.get("last_unblock_timestamp") or 0.0)
    return float(state.ready_since_ts) < last_unblock or age > float(max_age_seconds)


def ready_candidate_rejection_reason(
    state: SymbolState,
    runtime_state: dict[str, Any],
    *,
    max_age_seconds: float,
    now_ts: float | None = None,
) -> str:
    now_ts = time.time() if now_ts is None else float(now_ts)
    source = state.signal_source or state.last_update_source or "unknown"
    last_unblock = float(runtime_state.get("last_restart_unblock_timestamp") or runtime_state.get("last_unblock_timestamp") or 0.0)
    age = ready_candidate_age_seconds(state, now_ts)
    if source != "live":
        return f"signal_source_{source}"
    if state.ready_since_ts is None:
        return "missing_signal_time"
    if float(state.ready_since_ts) < last_unblock:
        return "signal_before_last_unblock"
    if state.last_live_update_ts is None:
        return "missing_live_update"
    if float(state.last_live_update_ts) < last_unblock:
        return "live_update_before_last_unblock"
    if age is not None and age > float(max_age_seconds):
        return "candidate_age_exceeded"
    return ""


def prune_entry_submit_timestamps(
    runtime_state: dict[str, Any],
    now_ts: float | None = None,
    *,
    window_seconds: float = 60.0,
) -> list[float]:
    now_ts = time.time() if now_ts is None else float(now_ts)
    kept: list[float] = []
    for value in runtime_state.get("entry_submit_timestamps") or []:
        try:
            ts = float(value)
        except Exception:
            continue
        if now_ts - ts <= float(window_seconds):
            kept.append(ts)
    runtime_state["entry_submit_timestamps"] = kept
    return kept


def entry_minute_capacity(runtime_state: dict[str, Any], max_entries_per_minute: int, now_ts: float | None = None) -> int:
    if int(max_entries_per_minute or 0) <= 0:
        return 10**9
    recent = prune_entry_submit_timestamps(runtime_state, now_ts, window_seconds=60.0)
    return max(0, int(max_entries_per_minute) - len(recent))


def record_entry_submission(runtime_state: dict[str, Any], now_ts: float | None = None) -> int:
    now_ts = time.time() if now_ts is None else float(now_ts)
    recent = prune_entry_submit_timestamps(runtime_state, now_ts, window_seconds=60.0)
    recent.append(now_ts)
    runtime_state["entry_submit_timestamps"] = recent
    return len(recent)


def ready_candidate_diagnostics(
    state: SymbolState,
    features: dict[str, Any],
    runtime_state: dict[str, Any],
    *,
    now_ts: float,
    ranking_position: int | None = None,
) -> dict[str, Any]:
    age = ready_candidate_age_seconds(state, now_ts)
    last_unblock = float(runtime_state.get("last_unblock_timestamp") or 0.0)
    return {
        "signal_time": state.ready_since_utc,
        "ready_since": state.ready_since_utc,
        "signal_source": state.signal_source or state.last_update_source or "unknown",
        "last_live_update_at": state.last_live_update_utc,
        "candidate_age_seconds": None if age is None else round(age, 3),
        "entry_decision_time": now_utc(),
        "score": features.get("score"),
        "ranking_position": ranking_position,
        "last_unblock_time": runtime_state.get("last_unblock_utc"),
        "last_restart_unblock_time": runtime_state.get("last_restart_unblock_utc") or runtime_state.get("last_unblock_utc"),
        "ready_before_last_unblock": bool(state.ready_since_ts is not None and float(state.ready_since_ts) < last_unblock),
    }


def log_live_feature_debug(
    *,
    runtime_state: dict[str, Any],
    symbol: str,
    state: SymbolState,
    snap: dict[str, Any],
    features: dict[str, Any],
    market_open: datetime,
    session_elapsed: float,
    reason: str,
    interval_seconds: float = 60.0,
) -> None:
    now_ts = time.time()
    key = f"live_feature_debug_last_{reason}"
    if now_ts - float(runtime_state.get(key) or 0.0) < interval_seconds:
        return
    runtime_state[key] = now_ts
    first_bar = state.bars[0] if state.bars else {}
    last_bar = state.bars[-1] if state.bars else {}
    print(
        f"{now_utc()} LIVE_FEATURE_DEBUG reason={reason} symbol={symbol} "
        f"session_start={market_open.isoformat()} session_elapsed_seconds={session_elapsed:.1f} "
        f"open_price={state.open_price} first_price={state.first_price} current_price={snap.get('price')} "
        f"last_price={state.last_price} first_5m_high={state.first_5m_high} "
        f"first_15m_high={state.first_15m_high} or_high={state.or_high} or_low={state.or_low} "
        f"first_5m_high_pct={features.get('first_5m_high_pct')} "
        f"first_15m_high_pct={features.get('first_15m_high_pct')} "
        f"or_range_pct={features.get('or_range_pct')} score={features.get('score')} "
        f"spread_bps={features.get('spread_bps')} first_seen_utc={state.first_seen_utc} "
        f"latest_seen_utc={state.latest_seen_utc} first_candle_ts={first_bar.get('bar_time_utc')} "
        f"latest_candle_ts={last_bar.get('bar_time_utc')} candle_samples={len(state.bars)}",
        flush=True,
    )


def record_lifecycle(recorder: LiveDataRecorder, event: str, symbol: str, **kwargs: Any) -> None:
    fields = [
        "recorded_at", "strategy", "event", "symbol", "action", "quantity", "price", "order_id",
        "execution_id", "reason", "entry_price", "peak_price", "pnl_pct",
        "decision_bid", "decision_ask", "decision_mid", "decision_last",
        "spread_pct", "fill_price", "fill_latency_ms",
        "estimated_commission", "realized_slippage_bps",
        "close_source", "fill_verified", "entry_fill_verified", "close_fill_verified",
        "raw_json",
    ]
    row = {"recorded_at": now_utc(), "strategy": STRATEGY_NAME, "event": event, "symbol": symbol, **kwargs}
    raw = row.get("raw_json")
    if raw and not isinstance(raw, str):
        row["raw_json"] = json.dumps(raw, ensure_ascii=False, default=str)
    append_dict_csv(recorder.path("trade_lifecycle.csv"), row, fields)
    safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "record_runtime_event",
        event_time=row.get("recorded_at"),
        severity="WARN" if "FAILED" in event or "REJECTED" in event or "DRIFT" in event or "ORPHAN" in event else "INFO",
        event_type=event,
        strategy_name=STRATEGY_NAME,
        session_date=getattr(recorder, "session_date", None),
        symbol=symbol,
        order_id=kwargs.get("order_id"),
        execution_id=kwargs.get("execution_id"),
        source="v67_live_runtime",
        reason=kwargs.get("reason"),
        action_required=1 if "MANUAL_REQUIRED" in event or "FAILED" in event else 0,
        raw_json={**kwargs, "legacy_event": event},
    )


def formal_event_type_for_legacy_event(event: str) -> LifecycleEventType | None:
    mapping = {
        "SIGNAL_READY": LifecycleEventType.ENTRY_SIGNAL,
        "BUY_ORDER_SENT": LifecycleEventType.ENTRY_ORDER_SUBMITTED,
        "SELL_ORDER_SENT": LifecycleEventType.EXIT_ORDER_SUBMITTED,
        "EOD_FLATTEN_SUBMIT": LifecycleEventType.EXIT_ORDER_SUBMITTED,
        "MANUAL_FLATTEN_SENT": LifecycleEventType.EXIT_ORDER_SUBMITTED,
        "MANUAL_FLATTEN_QUEUED": LifecycleEventType.EXIT_ORDER_PREPARED,
        "MANUAL_FLATTEN_DRY_RUN": LifecycleEventType.EXIT_ORDER_PREPARED,
        "MANUAL_FLATTEN_FAILED": LifecycleEventType.EXIT_ORDER_REJECTED,
        "EOD_FLATTEN_FAILED": LifecycleEventType.EXIT_ORDER_REJECTED,
        "EOD_FLATTEN_RETRY": LifecycleEventType.EXIT_ORDER_PREPARED,
        "EOD_FLATTEN_SUCCESS": LifecycleEventType.POSITION_CLOSED,
        "EOD_FLATTEN_GIVEUP": LifecycleEventType.EXIT_ORDER_REJECTED,
        "EXIT_ORDER_CANCEL_REQUESTED": LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED,
        "EXIT_ORDER_CANCELLED": LifecycleEventType.EXIT_ORDER_CANCELLED,
        "ORDER_CANCEL_CONFIRMED": LifecycleEventType.EXIT_ORDER_CANCELLED,
        "ORDER_STALE": LifecycleEventType.EXIT_ORDER_STALE,
        "ENTRY_NOT_FILLED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "ENTRY_EXPIRED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "ORDER_NOT_FILLED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "EXIT_ORDER_BLOCKED_NO_ENTRY_FILL": LifecycleEventType.EXIT_ORDER_REJECTED,
        "ADOPTED_POSITION": LifecycleEventType.POSITION_ADOPTED,
        "RESTORED_MANAGED_POSITION": LifecycleEventType.POSITION_ADOPTED,
        "POSITION_VERIFIED_CLOSED": LifecycleEventType.POSITION_CLOSED,
        "POSITION_CLOSED_UNVERIFIED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "RECONCILIATION_CLOSE_WITHOUT_FILL": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "POSITION_QUANTITY_DRIFT": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "POSITION_MISSING_IN_IBKR": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "ORPHAN_IBKR_POSITION_OBSERVED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "FRACTIONAL_ORPHAN_MANUAL_REQUIRED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "ENTRY_ORDER_REJECTED": LifecycleEventType.ENTRY_ORDER_REJECTED,
    }
    return mapping.get(event)


def record_formal_lifecycle(
    recorder: LiveDataRecorder,
    event_type: LifecycleEventType,
    symbol: str,
    *,
    strategy: str = STRATEGY_NAME,
    order_state: OrderState | None = None,
    position_state: PositionState | None = None,
    state_before: PositionState | None = None,
    state_after: PositionState | None = None,
    client_order_id: str = "",
    order_id: Any = "",
    execution_id: str = "",
    quantity: Any = None,
    price: Any = None,
    reason: str = "",
    raw_json: dict[str, Any] | None = None,
) -> None:
    try:
        store = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl"))
        store.append_event(
            LifecycleEvent(
                event_type=event_type,
                symbol=symbol,
                strategy=strategy,
                state_before=state_before,
                state_after=state_after,
                client_order_id=client_order_id,
                ib_order_id=str(order_id or ""),
                execution_id=execution_id,
                order_state=order_state,
                position_state=position_state,
                quantity=safe_float(quantity),
                price=safe_float(price),
                reason=reason,
                raw_json=raw_json or {},
            )
        )
    except Exception as exc:
        print(f"{now_utc()} formal_lifecycle_record_error event={event_type} symbol={symbol} error={exc!r}", flush=True)


def record_lifecycle_with_formal(recorder: LiveDataRecorder, event: str, symbol: str, **kwargs: Any) -> None:
    record_lifecycle(recorder, event, symbol, **kwargs)
    event_type = formal_event_type_for_legacy_event(event)
    if event_type is None:
        return
    order_state = None
    position_state = None
    state_before = None
    state_after = None
    if event_type == LifecycleEventType.ENTRY_SIGNAL:
        position_state = PositionState.NONE
    elif event_type == LifecycleEventType.ENTRY_ORDER_SUBMITTED:
        order_state = OrderState.SUBMITTED
        state_before = PositionState.NONE
        state_after = PositionState.ENTRY_PENDING
        position_state = PositionState.ENTRY_PENDING
    elif event_type == LifecycleEventType.ENTRY_ORDER_REJECTED:
        order_state = OrderState.REJECTED
        state_before = PositionState.ENTRY_PENDING
        state_after = PositionState.CLOSED
        position_state = PositionState.CLOSED
    elif event_type == LifecycleEventType.EXIT_ORDER_SUBMITTED:
        order_state = OrderState.SUBMITTED
        state_before = PositionState.OPEN
        state_after = PositionState.EXIT_PENDING
        position_state = PositionState.EXIT_PENDING
    elif event_type == LifecycleEventType.EXIT_ORDER_REJECTED:
        order_state = OrderState.REJECTED
        state_before = PositionState.OPEN
        state_after = PositionState.RECONCILING
        position_state = PositionState.RECONCILING
    elif event_type in {LifecycleEventType.ENTRY_ORDER_CANCELLED, LifecycleEventType.EXIT_ORDER_CANCELLED}:
        order_state = OrderState.CANCELLED
        state_after = PositionState.RECONCILING
        position_state = PositionState.RECONCILING
    elif event_type in {LifecycleEventType.POSITION_ADOPTED, LifecycleEventType.POSITION_DRIFT_DETECTED}:
        position_state = PositionState.RECONCILING if event_type == LifecycleEventType.POSITION_DRIFT_DETECTED else PositionState.OPEN
    elif event_type == LifecycleEventType.POSITION_CLOSED:
        state_after = PositionState.CLOSED
        position_state = PositionState.CLOSED
    record_formal_lifecycle(
        recorder,
        event_type,
        symbol,
        order_state=order_state,
        position_state=position_state,
        state_before=state_before,
        state_after=state_after,
        order_id=kwargs.get("order_id", ""),
        execution_id=str(kwargs.get("execution_id", "") or ""),
        quantity=kwargs.get("quantity"),
        price=kwargs.get("price"),
        reason=str(kwargs.get("reason", "") or ""),
        raw_json=kwargs.get("raw_json") if isinstance(kwargs.get("raw_json"), dict) else {"legacy_event": event},
    )


def load_existing_fill_keys(recorder: LiveDataRecorder) -> set[str]:
    path = recorder.path("fills.csv")
    seen: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return seen
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("execution_id") or f"{row.get('order_id','')}-{row.get('symbol','')}-{row.get('recorded_at','')}"
                if key.strip("-"):
                    seen.add(key)
    except Exception:
        return seen
    return seen


def load_exit_sent_symbols(recorder: LiveDataRecorder) -> set[str]:
    path = recorder.path("trade_lifecycle.csv")
    symbols: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return symbols
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("event", "")).strip() == "SELL_ORDER_SENT":
                    sym = str(row.get("symbol", "")).upper().strip()
                    if sym and sym != "__ALL__":
                        symbols.add(sym)
    except Exception:
        return symbols
    return symbols


def update_managed_position_excursion(pos: ManagedPosition, market_price: float | None = None, *, observed_at: str | None = None) -> None:
    price = safe_float(market_price)
    if pos.low_price is None or pos.low_price <= 0:
        pos.low_price = pos.entry_price
    if pos.peak_price <= 0:
        pos.peak_price = pos.entry_price
    if price is not None and price > 0:
        pos.peak_price = max(pos.peak_price, price)
        pos.low_price = min(pos.low_price or price, price)
    if pos.entry_price > 0:
        pos.mfe_pct = (pos.peak_price / pos.entry_price - 1.0) * 100.0
        if pos.low_price is not None and pos.low_price > 0:
            pos.mae_pct = (pos.low_price / pos.entry_price - 1.0) * 100.0
        pos.peak_unrealized_pnl = (pos.peak_price - pos.entry_price) * pos.quantity
        if pos.low_price is not None:
            pos.max_adverse_unrealized_pnl = (pos.low_price - pos.entry_price) * pos.quantity
    pos.last_update_time = observed_at or now_utc()


def managed_position_payload(pos: ManagedPosition, market_price_info: dict[str, Any] | None = None) -> dict[str, Any]:
    market_price = safe_float((market_price_info or {}).get("market_price"))
    update_managed_position_excursion(pos, market_price, observed_at=(market_price_info or {}).get("market_price_at"))
    unrealized_pnl = None
    unrealized_pct = None
    peak_pct = None
    drop_from_peak_pct = None
    if pos.entry_price:
        peak_pct = pos.mfe_pct if pos.mfe_pct is not None else (pos.peak_price / pos.entry_price - 1.0) * 100.0
        if market_price is not None:
            unrealized_pnl = (market_price - pos.entry_price) * pos.quantity
            unrealized_pct = (market_price / pos.entry_price - 1.0) * 100.0
            drop_from_peak_pct = peak_pct - unrealized_pct
    payload = {
        "symbol": pos.symbol,
        "quantity": pos.quantity,
        "entry_price": pos.entry_price,
        "entry_time": pos.entry_time,
        "peak_price": pos.peak_price,
        "low_price": pos.low_price,
        "mfe_pct": pos.mfe_pct,
        "mae_pct": pos.mae_pct,
        "peak_pct": peak_pct,
        "peak_unrealized_pct": peak_pct,
        "peak_unrealized_pnl": pos.peak_unrealized_pnl,
        "max_adverse_unrealized_pnl": pos.max_adverse_unrealized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pct": unrealized_pct,
        "drop_from_peak_pct": drop_from_peak_pct,
        "active": pos.active,
        "exit_sent": pos.exit_sent,
        "source": pos.source,
        "exit_order_id": pos.exit_order_id,
        "last_exit_order_ts": pos.last_exit_order_ts,
        "eod_retry_count": pos.eod_retry_count,
        "entry_fill_verified": pos.entry_fill_verified,
        "last_update": pos.last_update_time or (market_price_info or {}).get("market_price_at") or now_utc(),
        "last_update_time": pos.last_update_time,
    }
    if market_price_info:
        payload.update({k: v for k, v in market_price_info.items() if v is not None})
    return payload


def managed_position_market_price_info(
    pos: ManagedPosition,
    latest_snapshots: dict[str, dict[str, Any]] | None = None,
    portfolio_rows: list[dict[str, Any]] | None = None,
    *,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    symbol = str(pos.symbol).upper()
    recorded_at = recorded_at or now_utc()
    snap = (latest_snapshots or {}).get(symbol) or {}
    snap_price = safe_float(snap.get("price") or snap.get("last") or snap.get("mid_price") or snap.get("close"))
    if snap_price is not None and snap_price > 0:
        return {
            "market_price": snap_price,
            "market_price_at": snap.get("observed_at") or snap.get("latest_seen_utc") or recorded_at,
            "market_price_source": "live_ticker",
        }
    for row in portfolio_rows or []:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        portfolio_price = safe_float(row.get("market_price"))
        if portfolio_price is None:
            qty = safe_float(row.get("quantity"))
            market_value = safe_float(row.get("market_value"))
            if qty and abs(qty) > 0 and market_value is not None:
                portfolio_price = abs(market_value / qty)
        if portfolio_price is not None and portfolio_price > 0:
            return {
                "market_price": portfolio_price,
                "market_price_at": recorded_at,
                "market_price_source": "portfolio_snapshot",
            }
    return {
        "market_price": None,
        "market_price_at": None,
        "market_price_source": "missing",
    }


def open_position_price_diagnostics(
    positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]] | None = None,
    portfolio_rows: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    ok = 0
    missing = 0
    for pos in positions.values():
        if not pos.active:
            continue
        info = managed_position_market_price_info(pos, latest_snapshots, portfolio_rows)
        if safe_float(info.get("market_price")) is not None:
            ok += 1
        else:
            missing += 1
    return ok, missing


def persist_managed_positions(
    recorder: LiveDataRecorder,
    positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]] | None = None,
    portfolio_rows: list[dict[str, Any]] | None = None,
) -> None:
    recorded_at = now_utc()
    active_payloads = {
        symbol: managed_position_payload(
            pos,
            managed_position_market_price_info(pos, latest_snapshots, portfolio_rows, recorded_at=recorded_at),
        )
        for symbol, pos in positions.items()
        if pos.active
    }
    payload = {
        "recorded_at": recorded_at,
        "strategy": STRATEGY_NAME,
        "positions": active_payloads,
    }
    recorder.path("managed_positions.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    store = getattr(recorder, "sqlite_store", None)
    for symbol, pos in positions.items():
        raw_payload = managed_position_payload(
            pos,
            managed_position_market_price_info(pos, latest_snapshots, portfolio_rows, recorded_at=recorded_at),
        )
        safe_sqlite_call(
            store,
            "upsert_position",
            {
                "strategy_name": STRATEGY_NAME,
                "session_date": getattr(recorder, "session_date", None),
                "symbol": symbol,
                "status": "OPEN" if pos.active else "CLOSED",
                "quantity": pos.quantity,
                "avg_price": pos.entry_price,
                "source": pos.source,
                "active": pos.active,
                "exit_sent": pos.exit_sent,
                "updated_at": payload["recorded_at"],
                "raw_json": raw_payload,
            },
        )


def restore_managed_positions(
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    broker_qty_by_symbol: dict[str, float] | None = None,
    runtime_state: dict[str, Any] | None = None,
    restore_enabled: bool = True,
    disabled_reason: str | None = None,
) -> dict[str, ManagedPosition]:
    path = recorder.path("managed_positions.json")
    restored: dict[str, ManagedPosition] = {}
    if not path.exists():
        return restored
    candidate_count = 0
    rejected_count = 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for symbol, row in payload.get("positions", {}).items():
            symbol = str(symbol).upper()
            if symbol not in contract_by_symbol:
                continue
            qty = int(float(row.get("quantity", 0)))
            entry = safe_float(row.get("entry_price"))
            peak = safe_float(row.get("peak_price")) or entry
            if qty <= 0 or entry is None or entry <= 0:
                continue
            exit_sent = safe_bool(row.get("exit_sent"), False)
            entry_verified = safe_bool(row.get("entry_fill_verified"), False)
            if "entry_fill_verified" not in row and exit_sent:
                entry_verified = True
            candidate_count += 1
            if not restore_enabled:
                rejected_count += 1
                continue
            if broker_qty_by_symbol is not None:
                broker_qty = float(broker_qty_by_symbol.get(symbol, 0.0) or 0.0)
                if abs(broker_qty) <= 1e-9:
                    rejected_count += 1
                    continue
                if same_position_direction(float(qty), broker_qty) and whole_share_quantity(broker_qty):
                    qty = int(abs(round(broker_qty)))
            restored[symbol] = ManagedPosition(
                symbol=symbol,
                contract=contract_by_symbol[symbol],
                quantity=qty,
                entry_price=float(entry),
                entry_time=str(row.get("entry_time") or f"restored:{now_utc()}"),
                peak_price=float(peak or entry),
                low_price=safe_float(row.get("low_price")) or float(entry),
                mfe_pct=safe_float(row.get("mfe_pct") or row.get("peak_pct") or row.get("peak_unrealized_pct")),
                mae_pct=safe_float(row.get("mae_pct")),
                peak_unrealized_pnl=safe_float(row.get("peak_unrealized_pnl")),
                max_adverse_unrealized_pnl=safe_float(row.get("max_adverse_unrealized_pnl")),
                last_update_time=str(row.get("last_update_time") or row.get("last_update") or "") or None,
                active=safe_bool(row.get("active"), True),
                exit_sent=exit_sent,
                source=str(row.get("source") or "restored"),
                exit_order_id=int(float(row["exit_order_id"])) if row.get("exit_order_id") not in (None, "", "None") else None,
                last_exit_order_ts=safe_float(row.get("last_exit_order_ts")),
                eod_retry_count=int(float(row.get("eod_retry_count") or 0)),
                entry_fill_verified=entry_verified,
            )
    except Exception as exc:
        print(f"{now_utc()} managed_positions_restore_error={exc!r}", flush=True)
    if runtime_state is not None:
        runtime_state["startup_restore_broker_snapshot_count"] = len(broker_qty_by_symbol or {})
        runtime_state["startup_restore_candidate_count"] = candidate_count
        runtime_state["startup_restore_open_count"] = len(restored)
        runtime_state["startup_restore_rejected_count"] = rejected_count
        runtime_state["startup_restore_enabled"] = bool(restore_enabled)
        if disabled_reason:
            runtime_state["startup_restore_disabled_reason"] = disabled_reason
    print(
        f"{now_utc()} STARTUP_RESTORE_DIAGNOSTICS "
        f"restore_enabled={int(bool(restore_enabled))} "
        f"disabled_reason={disabled_reason or 'none'} "
        f"broker_snapshot_count={len(broker_qty_by_symbol or {})} "
        f"restored_candidate_count={candidate_count} "
        f"restored_open_count={len(restored)} "
        f"restored_rejected_count={rejected_count}",
        flush=True,
    )
    return restored


def record_strategy_equity(recorder: LiveDataRecorder, positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]]) -> None:
    unrealized = 0.0
    gross = 0.0
    active_count = 0
    rows = []
    for symbol, pos in positions.items():
        if not pos.active or pos.exit_sent or not pos.entry_fill_verified:
            continue
        price = safe_float((latest_snapshots.get(symbol) or {}).get("price"))
        if price is None:
            continue
        active_count += 1
        market_value = price * pos.quantity
        cost = pos.entry_price * pos.quantity
        pnl = market_value - cost
        unrealized += pnl
        gross += abs(market_value)
        rows.append({
            "symbol": symbol,
            "qty": pos.quantity,
            "entry": pos.entry_price,
            "price": price,
            "peak": pos.peak_price,
            "unrealized_pnl": pnl,
            "unrealized_pct": (price / pos.entry_price - 1.0) * 100.0 if pos.entry_price else None,
        })
    append_dict_csv(
        recorder.path("strategy_equity.csv"),
        {
            "recorded_at": now_utc(),
            "strategy": STRATEGY_NAME,
            "active_positions": active_count,
            "gross_exposure": gross,
            "unrealized_pnl": unrealized,
            "positions_json": json.dumps(rows, ensure_ascii=False, default=str),
        },
        ["recorded_at", "strategy", "active_positions", "gross_exposure", "unrealized_pnl", "positions_json"],
    )


def existing_candle_keys(recorder: LiveDataRecorder) -> set[tuple[str, str]]:
    path = recorder.path("candles_1m.csv")
    keys: set[tuple[str, str]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym = str(row.get("symbol", "")).upper()
                ts = str(row.get("bar_time", ""))
                if sym and ts:
                    keys.add((sym, ts))
    except Exception:
        return keys
    return keys


def backfill_recent_1m(ib: IB, recorder: LiveDataRecorder, contracts: list[tuple[str, Any]], args: argparse.Namespace) -> int:
    if not args.backfill_1m_on_start:
        return 0
    keys = existing_candle_keys(recorder)
    total = 0
    subset = contracts[: max(0, args.backfill_top_n)]
    for symbol, contract in subset:
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=args.backfill_duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            print(f"{now_utc()} backfill_1m_error symbol={symbol} error={exc!r}", flush=True)
            continue
        rows = []
        for bar in bars:
            bar_time = str(getattr(bar, "date", ""))
            key = (symbol, bar_time)
            if key in keys:
                continue
            keys.add(key)
            rows.append({
                "symbol": symbol,
                "bar_time": bar_time,
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "source": "ibkr_backfill_1m",
                "recorded_at": now_utc(),
            })
        if rows:
            total += recorder.record_candles_1m(rows)
        ib.sleep(args.backfill_pause_seconds)
    print(f"{now_utc()} backfill_1m_done symbols={len(subset)} rows={total}", flush=True)
    return total


def current_session_candle_count(recorder: LiveDataRecorder, args: argparse.Namespace) -> int:
    path = recorder.path("candles_1m.csv")
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        today = datetime.now(timezone.utc).date()
        market_open = datetime(today.year, today.month, today.day, hh, mm, tzinfo=timezone.utc)
    except Exception:
        return 0
    count = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_bar_time_utc(row.get("bar_time", ""))
                if ts is not None and ts >= market_open:
                    count += 1
    except Exception:
        return 0
    return count


def backfill_current_session_1m(
    ib: IB,
    recorder: LiveDataRecorder,
    contracts: list[tuple[str, Any]],
    args: argparse.Namespace,
) -> int:
    if not getattr(args, "backfill_current_session_on_rebuild_miss", True):
        return 0

    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        now = datetime.now(timezone.utc)
        market_open = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception as exc:
        print(f"{now_utc()} current_session_backfill_skipped reason=bad_market_open error={exc!r}", flush=True)
        return 0

    if now < market_open:
        print(f"{now_utc()} current_session_backfill_skipped reason=before_market_open", flush=True)
        return 0

    duration_seconds = max(60, int((now - market_open).total_seconds()) + 300)
    duration = f"{duration_seconds} S"
    keys = existing_candle_keys(recorder)
    total = 0
    subset = contracts[: max(0, int(getattr(args, "backfill_top_n", len(contracts))))]
    print(f"{now_utc()} current_session_backfill_start symbols={len(subset)} duration={duration}", flush=True)

    for symbol, contract in subset:
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            print(f"{now_utc()} current_session_backfill_error symbol={symbol} error={exc!r}", flush=True)
            continue

        rows = []
        for bar in bars:
            bar_time = str(getattr(bar, "date", ""))
            ts = _parse_bar_time_utc(bar_time)
            if ts is None or ts < market_open:
                continue
            key = (symbol, bar_time)
            if key in keys:
                continue
            keys.add(key)
            rows.append({
                "symbol": symbol,
                "bar_time": bar_time,
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "source": "ibkr_current_session_backfill_1m",
                "recorded_at": now_utc(),
            })
        if rows:
            total += recorder.record_candles_1m(rows)
        ib.sleep(float(getattr(args, "backfill_pause_seconds", 0.15)))

    print(f"{now_utc()} current_session_backfill_done symbols={len(subset)} rows={total}", flush=True)
    return total


def is_eod_flatten_time(flatten_utc: str) -> bool:
    try:
        hh, mm = [int(x) for x in flatten_utc.split(":", 1)]
        return datetime.now(timezone.utc).time() >= dtime(hour=hh, minute=mm, tzinfo=timezone.utc)
    except Exception:
        return False


def send_exit_order(ib: IB, recorder: LiveDataRecorder, pos: ManagedPosition, reason: str, price: float | None) -> bool:
    if not pos.active or pos.quantity <= 0:
        return False
    if not pos.entry_fill_verified:
        record_lifecycle_with_formal(
            recorder,
            "EXIT_ORDER_BLOCKED_NO_ENTRY_FILL",
            pos.symbol,
            action="SELL",
            quantity=pos.quantity,
            price=price,
            reason="entry_fill_not_verified",
            entry_fill_verified="false",
            raw_json={"requested_exit_reason": reason, "entry_fill_verified": False},
        )
        print(
            f"{now_utc()} EXIT_ORDER_BLOCKED_NO_ENTRY_FILL symbol={pos.symbol} "
            f"quantity={pos.quantity} reason={reason}",
            flush=True,
        )
        return False
    order = MarketOrder("SELL", pos.quantity)
    order.tif = "DAY"
    order.outsideRth = False
    trade = ib.placeOrder(pos.contract, order)
    order_id = trade.order.orderId
    submitted_at = now_utc()
    position_key = f"{STRATEGY_NAME}:{getattr(recorder, 'session_date', '')}:{pos.symbol}"
    safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "record_exit_order_intent",
        order_id=order_id,
        symbol=pos.symbol,
        exit_reason=reason,
        quantity=pos.quantity,
        submitted_at=submitted_at,
        position_key=position_key,
        strategy_name=STRATEGY_NAME,
        session_date=getattr(recorder, "session_date", None),
        raw_json={
            "source": "send_exit_order",
            "entry_price": pos.entry_price,
            "peak_price": pos.peak_price,
            "entry_time": pos.entry_time,
        },
    )
    pnl_pct = ((price / pos.entry_price - 1.0) * 100.0) if price and pos.entry_price > 0 else None
    record_lifecycle_with_formal(
        recorder,
        "SELL_ORDER_SENT",
        pos.symbol,
        action="SELL",
        quantity=pos.quantity,
        price=price,
        order_id=order_id,
        reason=reason,
        entry_price=pos.entry_price,
        peak_price=pos.peak_price,
        pnl_pct=pnl_pct,
        decision_bid=None,
        decision_ask=None,
        decision_mid=None,
        decision_last=price,
        spread_pct=None,
        entry_fill_verified="true",
    )
    pos.exit_sent = True
    pos.exit_order_id = order_id
    pos.last_exit_order_ts = time.time()
    pnl_txt = f" pnl_pct={pnl_pct:.2f}" if pnl_pct is not None else ""
    print(
        f"PAPER SELL SENT symbol={pos.symbol} qty={pos.quantity} "
        f"reason={reason} entry={pos.entry_price:.2f} price={price if price else 0:.2f}"
        f"{pnl_txt} orderId={order_id} tif={order.tif} outsideRth={order.outsideRth}",
        flush=True,
    )
    return True


def ibkr_position_quantities(ib: IB) -> dict[str, float]:
    quantities: dict[str, float] = {}
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = safe_float(getattr(item, "position", None)) or 0.0
        if abs(qty) > 0:
            quantities[symbol] = quantities.get(symbol, 0.0) + qty
    return quantities


def ibkr_portfolio_position_rows(ib: IB) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ib.portfolio():
        contract = getattr(item, "contract", None)
        symbol = str(getattr(contract, "symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = safe_float(getattr(item, "position", None)) or 0.0
        if abs(qty) <= 0:
            continue
        rows.append({
            "symbol": symbol,
            "quantity": qty,
            "contract": contract,
            "market_price": safe_float(getattr(item, "marketPrice", None)),
            "market_value": safe_float(getattr(item, "marketValue", None)),
            "average_cost": safe_float(getattr(item, "averageCost", None)),
        })
    return rows


def same_position_direction(managed_qty: float, ibkr_qty: float) -> bool:
    return (managed_qty > 0 and ibkr_qty > 0) or (managed_qty < 0 and ibkr_qty < 0)


def whole_share_quantity(quantity: float) -> bool:
    return not is_fractional_position_quantity(quantity)


def open_ibkr_order_trades(ib: IB) -> list[Any]:
    out: list[Any] = []
    try:
        out.extend(list(ib.openTrades()))
    except Exception:
        pass
    seen_order_ids = {order_trade_id(trade) for trade in out if order_trade_id(trade) is not None}
    try:
        for order in ib.openOrders():
            order_id = getattr(order, "orderId", None)
            if order_id in seen_order_ids:
                continue
            out.append(type("OpenOrderTrade", (), {"order": order, "contract": None})())
    except Exception:
        pass
    return out


def order_trade_symbol(trade: Any) -> str:
    return str(getattr(getattr(trade, "contract", None), "symbol", "") or "").upper().strip()


def order_trade_id(trade: Any) -> Any:
    return getattr(getattr(trade, "order", None), "orderId", None)


def order_trade_action(trade: Any) -> str:
    return str(getattr(getattr(trade, "order", None), "action", "") or "").upper().strip()


def order_trade_quantity(trade: Any) -> float | None:
    return safe_float(getattr(getattr(trade, "order", None), "totalQuantity", None))


def sqlite_active_position_count(recorder: LiveDataRecorder) -> int | None:
    rows = safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "query",
        "SELECT COUNT(*) AS count FROM positions WHERE COALESCE(active, 0) = 1",
    )
    if not rows:
        return None
    try:
        return int(rows[0].get("count") or 0)
    except Exception:
        return None


def sqlite_active_position_symbols(recorder: LiveDataRecorder) -> set[str]:
    rows = safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "query",
        """
        SELECT DISTINCT UPPER(symbol) AS symbol
        FROM positions
        WHERE COALESCE(active, 0) = 1
          AND UPPER(COALESCE(status, '')) IN ('OPEN', 'EXIT_ORDER', 'RECONCILING')
          AND COALESCE(symbol, '') != ''
        """,
    )
    if not rows:
        return set()
    return {str(row.get("symbol") or "").upper() for row in rows if str(row.get("symbol") or "").strip()}


def startup_reconcile_runtime_state(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    contract_by_symbol: dict[str, Any],
    runtime_state: dict[str, Any],
    *,
    cancel_stale_orders: bool = True,
    submit_orphan_flatten: bool = False,
    log_prefix: str = "STARTUP_RECONCILIATION",
    reason_prefix: str = "startup_reconciliation",
) -> dict[str, Any]:
    print(f"{now_utc()} {log_prefix}_START", flush=True)
    runtime_state["entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = reason_prefix

    lifecycle_events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
    reducer_snapshot = reduce_lifecycle_events(lifecycle_events)
    reducer_positions = reducer_snapshot.positions
    ibkr_rows = ibkr_portfolio_position_rows(ib)
    ibkr_qty_by_symbol = {row["symbol"]: float(row["quantity"]) for row in ibkr_rows}
    ibkr_row_by_symbol = {row["symbol"]: row for row in ibkr_rows}
    sqlite_active_count = sqlite_active_position_count(recorder)
    sqlite_active_symbols = sqlite_active_position_symbols(recorder)
    runtime_state["startup_reconciliation_broker_open_count"] = len(ibkr_rows)
    runtime_state["startup_reconciliation_sqlite_active_count"] = sqlite_active_count
    runtime_state["startup_reconciliation_sqlite_active_symbols"] = sorted(sqlite_active_symbols)
    if not managed_positions and not ibkr_rows and sqlite_active_count == 0 and reducer_positions:
        print(
            f"{now_utc()} {log_prefix}_STALE_LOCAL_STATE_IGNORED "
            f"lifecycle_positions={len(reducer_positions)} broker_open_count=0 sqlite_active_count=0",
            flush=True,
        )
        reducer_positions = {}

    closed_local: list[str] = []
    drift_symbols: list[str] = []
    orphans: list[str] = []
    untouched_orphans: list[str] = []
    fractional_orphans: list[str] = []
    whole_share_orphans: list[str] = []
    pending_orders: list[str] = []
    submitted_flatten_order_ids: set[Any] = set()

    candidate_symbols = sorted(set(managed_positions) | set(reducer_positions))
    for symbol in candidate_symbols:
        pos = managed_positions.get(symbol)
        reduced = reducer_positions.get(symbol)
        lifecycle_open = reduced is not None and reduced.state in {PositionState.OPEN, PositionState.ENTRY_PENDING, PositionState.EXIT_PENDING, PositionState.RECONCILING}
        local_open = pos is not None and pos.active
        ibkr_qty = float(ibkr_qty_by_symbol.get(symbol, 0.0))
        if (local_open or lifecycle_open) and abs(ibkr_qty) <= 0:
            quantity = getattr(pos, "quantity", None) if pos is not None else getattr(reduced, "open_quantity", None)
            backfill_recent_fills_for_verification(ib, recorder, runtime_state)
            has_entry_fill = entry_fill_verified(recorder, symbol, quantity)
            if pos is not None and pos.active:
                pos.active = False
                pos.exit_sent = False
            closed_local.append(symbol)
            if not has_entry_fill:
                record_entry_not_filled(
                    recorder,
                    symbol,
                    quantity=quantity,
                    reason=f"{reason_prefix}_ibkr_flat_entry_not_filled",
                    ibkr_quantity=ibkr_qty,
                    runtime_state=runtime_state,
                    raw_json={
                        "managed_active": bool(local_open),
                        "reducer_state": getattr(getattr(reduced, "state", None), "value", None),
                    },
                )
                print(f"{now_utc()} {log_prefix}_ENTRY_NOT_FILLED symbol={symbol}", flush=True)
                continue
            if not close_fill_verified(recorder, symbol, quantity):
                record_unverified_reconciliation_close(
                    recorder,
                    symbol,
                    quantity=quantity,
                    reason=f"{reason_prefix}_ibkr_flat_without_fill",
                    ibkr_quantity=ibkr_qty,
                    runtime_state=runtime_state,
                    raw_json={
                        "managed_active": bool(local_open),
                        "reducer_state": getattr(getattr(reduced, "state", None), "value", None),
                        "entry_fill_verified": True,
                    },
                )
                print(f"{now_utc()} {log_prefix}_LOCAL_CLOSED_UNVERIFIED symbol={symbol}", flush=True)
                continue
            record_lifecycle_with_formal(
                recorder,
                "POSITION_VERIFIED_CLOSED",
                symbol,
                action="VERIFY",
                quantity=quantity,
                reason=f"{reason_prefix}_ibkr_flat",
                close_source="fill",
                fill_verified="true",
                entry_fill_verified="true",
                close_fill_verified="true",
                raw_json={
                    "managed_active": bool(local_open),
                    "reducer_state": getattr(getattr(reduced, "state", None), "value", None),
                    "ibkr_quantity": ibkr_qty,
                    "close_source": "fill",
                    "fill_verified": True,
                    "entry_fill_verified": True,
                    "close_fill_verified": True,
                },
            )
            safe_sqlite_call(
                getattr(recorder, "sqlite_store", None),
                "mark_position_flat",
                symbol=symbol,
                strategy_name=STRATEGY_NAME,
                reason=f"{reason_prefix}_ibkr_flat",
                status="CLOSED",
            )
            print(f"{now_utc()} {log_prefix}_LOCAL_CLOSED symbol={symbol}", flush=True)
            continue
        if pos is not None and pos.active and abs(ibkr_qty) > 0 and abs(float(pos.quantity) - abs(ibkr_qty)) > 1e-9:
            drift_symbols.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_QUANTITY_DRIFT",
                symbol,
                action="VERIFY",
                quantity=pos.quantity,
                reason=f"{reason_prefix}_quantity_drift",
                raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ibkr_qty},
            )
            if same_position_direction(float(pos.quantity), ibkr_qty) and whole_share_quantity(ibkr_qty):
                old_qty = pos.quantity
                pos.quantity = int(abs(round(ibkr_qty)))
                pos.source = f"{pos.source}:{reason_prefix}_qty"
                print(
                    f"{now_utc()} {log_prefix}_DRIFT symbol={symbol} managed_quantity={old_qty} "
                    f"ibkr_quantity={ibkr_qty:.4f} action=managed_quantity_updated",
                    flush=True,
                )
            else:
                print(
                    f"{now_utc()} {log_prefix}_DRIFT symbol={symbol} managed_quantity={pos.quantity} "
                    f"ibkr_quantity={ibkr_qty:.4f} action=reconciling_manual_review",
                    flush=True,
                )

    local_known_symbols = {s for s, p in managed_positions.items() if p.active} | sqlite_active_symbols
    for symbol in sorted(set(ibkr_qty_by_symbol) - local_known_symbols):
        row = ibkr_row_by_symbol[symbol]
        ibkr_qty = float(row["quantity"])
        orphans.append(symbol)
        if is_fractional_position_quantity(ibkr_qty):
            fractional_orphans.append(symbol)
            record_fractional_orphan_manual_required(
                recorder,
                runtime_state,
                symbol=symbol,
                quantity=ibkr_qty,
                price=row.get("market_price"),
                reason=f"{reason_prefix}_fractional_orphan_manual_required",
                raw_json={"average_cost": row.get("average_cost"), "market_value": row.get("market_value")},
                log_prefix=f"{log_prefix}_FRACTIONAL_ORPHAN_MANUAL_REQUIRED",
            )
        else:
            whole_share_orphans.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "ORPHAN_IBKR_POSITION_OBSERVED",
                symbol,
                action="ALERT",
                quantity=ibkr_qty,
                price=row.get("market_price"),
                reason=f"{reason_prefix}_orphan_ibkr_position",
                raw_json={"ibkr_quantity": ibkr_qty, "average_cost": row.get("average_cost"), "market_value": row.get("market_value")},
            )
            print(f"{now_utc()} {log_prefix}_ORPHAN_DETECTED symbol={symbol} quantity={ibkr_qty:.4f}", flush=True)
        if submit_orphan_flatten:
            if whole_share_quantity(ibkr_qty):
                submitted_order_id = submit_eod_flatten_order(
                    ib,
                    recorder,
                    symbol=symbol,
                    contract=row["contract"],
                    ibkr_quantity=ibkr_qty,
                    reason=f"{reason_prefix}_orphan_flatten",
                    attempt=1,
                    runtime_state=runtime_state,
                )
                if submitted_order_id is not None:
                    submitted_flatten_order_ids.add(submitted_order_id)
            else:
                runtime_state["startup_reconciliation_fractional_manual_required"] = True
        else:
            untouched_orphans.append(symbol)
            runtime_state["startup_reconciliation_orphan_flatten_blocked"] = True
            print(
                f"{now_utc()} {log_prefix}_ORPHAN_LEFT_UNTOUCHED symbol={symbol} "
                f"quantity={ibkr_qty:.4f} reason=restart_safe_no_auto_flatten",
                flush=True,
            )

    for trade in open_ibkr_order_trades(ib):
        order_id = order_trade_id(trade)
        symbol = order_trade_symbol(trade)
        action = order_trade_action(trade)
        quantity = order_trade_quantity(trade)
        pending_orders.append(str(order_id or ""))
        if order_id in submitted_flatten_order_ids:
            print(f"{now_utc()} {log_prefix}_PENDING_FLATTEN_ORDER order_id={order_id} symbol={symbol}", flush=True)
            continue
        print(f"{now_utc()} {log_prefix}_PENDING_ORDER order_id={order_id} symbol={symbol}", flush=True)
        record_lifecycle_with_formal(
            recorder,
            "ORDER_STALE",
            symbol or "__UNKNOWN__",
            action=action,
            quantity=quantity,
            order_id=order_id,
            reason=f"{reason_prefix}_stale_open_order",
            raw_json={"order_id": order_id, "symbol": symbol, "action": action, "quantity": quantity},
        )
        if cancel_stale_orders:
            record_lifecycle_with_formal(
                recorder,
                "EXIT_ORDER_CANCEL_REQUESTED",
                symbol or "__UNKNOWN__",
                action=action,
                quantity=quantity,
                order_id=order_id,
                reason=f"{reason_prefix}_cancel_stale_open_order",
            )
            try:
                ib.cancelOrder(getattr(trade, "order", trade))
                print(f"{now_utc()} ORDER_CANCEL_CONFIRMED order_id={order_id} symbol={symbol}", flush=True)
                record_lifecycle_with_formal(
                    recorder,
                    "EXIT_ORDER_CANCELLED",
                    symbol or "__UNKNOWN__",
                    action=action,
                    quantity=quantity,
                    order_id=order_id,
                    reason=f"{reason_prefix}_cancel_sent",
                )
                record_lifecycle_with_formal(
                    recorder,
                    "ORDER_CANCEL_CONFIRMED",
                    symbol or "__UNKNOWN__",
                    action=action,
                    quantity=quantity,
                    order_id=order_id,
                    reason=f"{reason_prefix}_cancel_confirmed",
                )
            except Exception as exc:
                record_lifecycle_with_formal(
                    recorder,
                    "EXIT_ORDER_REJECTED",
                    symbol or "__UNKNOWN__",
                    action=action,
                    quantity=quantity,
                    order_id=order_id,
                    reason=f"{reason_prefix}_cancel_failed:{exc!r}",
                )

    clean = not closed_local and not drift_symbols and not orphans and not pending_orders and not reducer_snapshot.anomalies
    runtime_state["startup_reconciliation_done"] = True
    runtime_state["startup_reconciliation_clean"] = bool(clean)
    runtime_state["startup_reconciliation_orphans"] = sorted(orphans)
    runtime_state["startup_reconciliation_untouched_orphans"] = sorted(untouched_orphans)
    runtime_state["startup_reconciliation_fractional_orphans"] = sorted(fractional_orphans)
    runtime_state["startup_reconciliation_whole_share_orphans"] = sorted(whole_share_orphans)
    runtime_state["startup_reconciliation_closed_local"] = sorted(closed_local)
    runtime_state["startup_reconciliation_pending_orders"] = sorted([p for p in pending_orders if p])
    runtime_state["startup_reconciliation_drift_symbols"] = sorted(drift_symbols)
    runtime_state["startup_reconciliation_anomalies"] = list(reducer_snapshot.anomalies)
    runtime_state["entries_blocked"] = False
    if runtime_state.get("entries_blocked_reason") == reason_prefix:
        runtime_state["entries_blocked_reason"] = ""
    persist_managed_positions(recorder, managed_positions)
    print(
        f"{now_utc()} {log_prefix}_DONE clean={int(clean)} "
        f"closed_local={len(closed_local)} orphans={len(orphans)} fractional_orphans={len(fractional_orphans)} "
        f"whole_share_orphans={len(whole_share_orphans)} drift={len(drift_symbols)} "
        f"pending_orders={len(pending_orders)} anomalies={len(reducer_snapshot.anomalies)}",
        flush=True,
    )
    if not ibkr_rows:
        safe_sqlite_call(
            getattr(recorder, "sqlite_store", None),
            "mark_all_positions_flat",
            reason="reconciliation_clean" if clean else "reconciliation_ibkr_flat",
            status="FLAT_CONFIRMED",
            updated_at=now_utc(),
        )
    safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "record_reconciliation_run",
        run_id=f"{reason_prefix}:{getattr(recorder, 'session_date', '')}:{int(time.time())}",
        started_at=now_utc(),
        finished_at=now_utc(),
        mode=reason_prefix.replace("_reconciliation", "") or "startup",
        clean=clean,
        ibkr_positions_count=len(ibkr_rows),
        managed_positions_count=len([p for p in managed_positions.values() if p.active]),
        orphan_count=len(orphans),
        fractional_orphan_count=len(fractional_orphans),
        drift_count=len(drift_symbols),
        pending_orders_count=len(pending_orders),
        details_json={
            "closed_local": sorted(closed_local),
            "orphans": sorted(orphans),
            "fractional_orphans": sorted(fractional_orphans),
            "whole_share_orphans": sorted(whole_share_orphans),
            "drift_symbols": sorted(drift_symbols),
            "pending_orders": sorted([p for p in pending_orders if p]),
            "anomalies": list(reducer_snapshot.anomalies),
        },
    )
    return {
        "clean": clean,
        "closed_local": sorted(closed_local),
        "orphans": sorted(orphans),
        "untouched_orphans": sorted(untouched_orphans),
        "fractional_orphans": sorted(fractional_orphans),
        "whole_share_orphans": sorted(whole_share_orphans),
        "drift_symbols": sorted(drift_symbols),
        "pending_orders": sorted([p for p in pending_orders if p]),
        "anomalies": list(reducer_snapshot.anomalies),
    }


def _flatten_action_for_quantity(quantity: float) -> str:
    return "SELL" if quantity > 0 else "BUY"


def is_fractional_position_quantity(quantity: float) -> bool:
    qty = abs(float(quantity))
    return not math.isclose(qty, round(qty), rel_tol=0.0, abs_tol=1e-9)


def record_fractional_orphan_manual_required(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    symbol: str,
    quantity: float,
    price: float | None = None,
    reason: str,
    raw_json: dict[str, Any] | None = None,
    log_prefix: str = "FRACTIONAL_ORPHAN_MANUAL_REQUIRED",
) -> bool:
    seen = runtime_state.setdefault("fractional_orphan_manual_required_seen", {})
    suppressed = runtime_state.setdefault("fractional_orphan_manual_required_suppressed", {})
    if not isinstance(seen, dict):
        seen = {}
        runtime_state["fractional_orphan_manual_required_seen"] = seen
    if not isinstance(suppressed, dict):
        suppressed = {}
        runtime_state["fractional_orphan_manual_required_suppressed"] = suppressed

    key = f"{symbol}:{float(quantity):.8f}"
    if seen.get(key):
        suppressed[key] = int(suppressed.get(key, 0) or 0) + 1
        runtime_state["fractional_orphan_manual_required_suppressed_total"] = sum(int(v or 0) for v in suppressed.values())
        return False

    seen[key] = now_utc()
    payload = {
        "ibkr_quantity": quantity,
        "manual_action_required": "close_fractional_position_in_ibkr_desktop",
        **(raw_json or {}),
    }
    record_lifecycle_with_formal(
        recorder,
        "FRACTIONAL_ORPHAN_MANUAL_REQUIRED",
        symbol,
        action="ALERT",
        quantity=quantity,
        price=price,
        reason=reason,
        raw_json=payload,
    )
    print(
        f"{now_utc()} {log_prefix} symbol={symbol} quantity={quantity:.4f} "
        f"reason={reason} manual_action_required=ibkr_desktop",
        flush=True,
    )
    return True


def smart_stock_contract_for_flatten(symbol: str, contract: Any | None) -> Any:
    currency = str(getattr(contract, "currency", "") or "USD")
    return Stock(symbol, "SMART", currency)


def _order_quantity_from_position(quantity: float) -> int | float:
    qty = abs(float(quantity))
    rounded = round(qty)
    if abs(qty - rounded) <= 1e-6:
        return int(rounded)
    return qty


def open_flatten_order_keys(ib: IB) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    done_statuses = {"Cancelled", "ApiCancelled", "Filled", "Inactive"}
    try:
        trades = list(ib.openTrades())
    except Exception:
        trades = []
    for trade in trades:
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        status = getattr(getattr(trade, "orderStatus", None), "status", "")
        if status in done_statuses:
            continue
        symbol = str(getattr(contract, "symbol", "") or "").upper().strip()
        action = str(getattr(order, "action", "") or "").upper().strip()
        if symbol and action in {"BUY", "SELL"}:
            keys.add((symbol, action))
    return keys


def write_eod_final_status(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    pending_orders: int,
    managed_positions: dict[str, ManagedPosition],
    reason: str,
) -> dict[str, Any]:
    managed_open_symbols = sorted(symbol for symbol, pos in managed_positions.items() if pos.active)
    managed_open_set = set(managed_open_symbols)
    fractional_orphans = sorted(
        row["symbol"] for row in rows
        if is_fractional_position_quantity(float(row["quantity"]))
    )
    whole_share_orphans = sorted(
        row["symbol"] for row in rows
        if not is_fractional_position_quantity(float(row["quantity"])) and row["symbol"] not in managed_open_set
    )
    summary = {
        "recorded_at": now_utc(),
        "reason": reason,
        "clean": not rows and pending_orders == 0 and not managed_open_symbols,
        "open_positions": len(rows),
        "open_symbols": sorted(row["symbol"] for row in rows),
        "fractional_orphans": fractional_orphans,
        "whole_share_orphans": whole_share_orphans,
        "pending_orders": pending_orders,
        "managed_open": len(managed_open_symbols),
        "managed_open_symbols": managed_open_symbols,
        "pending_eod_flatten": bool(rows),
    }
    runtime_state["eod_final_status"] = summary
    recorder.path("eod_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if summary["clean"]:
        safe_sqlite_call(
            getattr(recorder, "sqlite_store", None),
            "mark_all_positions_flat",
            reason="eod_success",
            status="FLAT_CONFIRMED",
            updated_at=summary["recorded_at"],
        )
    safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "record_runtime_event",
        event_time=summary["recorded_at"],
        severity="INFO" if summary["clean"] else "WARN",
        event_type="EOD_FINAL_STATUS",
        strategy_name=STRATEGY_NAME,
        session_date=getattr(recorder, "session_date", None),
        source="v67_live_runtime",
        reason=reason,
        action_required=0 if summary["clean"] else 1,
        raw_json=summary,
    )
    safe_sqlite_call(
        getattr(recorder, "sqlite_store", None),
        "record_reconciliation_run",
        run_id=f"eod:{getattr(recorder, 'session_date', '')}:{reason}",
        started_at=summary["recorded_at"],
        finished_at=summary["recorded_at"],
        mode="eod",
        clean=summary["clean"],
        ibkr_positions_count=summary["open_positions"],
        managed_positions_count=summary["managed_open"],
        orphan_count=len(summary["whole_share_orphans"]) + len(summary["fractional_orphans"]),
        fractional_orphan_count=len(summary["fractional_orphans"]),
        drift_count=0,
        pending_orders_count=summary["pending_orders"],
        details_json=summary,
    )
    print(
        f"{now_utc()} EOD_FINAL_STATUS clean={int(summary['clean'])} "
        f"open_positions={summary['open_positions']} fractional_orphans={len(fractional_orphans)} "
        f"whole_share_orphans={len(whole_share_orphans)} pending_orders={pending_orders} "
        f"managed_open={summary['managed_open']}",
        flush=True,
    )
    return summary


def persist_pending_eod_flatten(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    pending: bool,
    reason: str,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    runtime_state["pending_eod_flatten"] = bool(pending)
    runtime_state["pending_eod_flatten_reason"] = reason if pending else ""
    runtime_state["pending_eod_flatten_updated_at"] = now_utc()
    runtime_state["pending_eod_flatten_symbols"] = sorted(row["symbol"] for row in (rows or [])) if pending else []
    payload = {
        "recorded_at": runtime_state["pending_eod_flatten_updated_at"],
        "pending_eod_flatten": bool(pending),
        "reason": reason,
        "symbols": runtime_state["pending_eod_flatten_symbols"],
    }
    try:
        recorder.path("eod_pending.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as exc:
        print(f"{now_utc()} EOD_PENDING_PERSIST_FAILED reason={reason} error={exc!r}", flush=True)


def load_pending_eod_flatten(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    broker_open_count: int | None = None,
    sqlite_active_count: int | None = None,
) -> bool:
    path = recorder.path("eod_pending.json")
    runtime_state["eod_pending_file"] = str(path)
    if not path.exists():
        runtime_state["eod_pending_symbols_count"] = 0
        runtime_state["eod_pending_ignored_count"] = 0
        runtime_state["eod_pending_pending_restored"] = 0
        runtime_state["eod_pending_ignored_reason"] = ""
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{now_utc()} EOD_PENDING_LOAD_FAILED error={exc!r}", flush=True)
        return False
    pending = bool(payload.get("pending_eod_flatten"))
    symbols = list(payload.get("symbols") or [])
    runtime_state["eod_pending_broker_open_count"] = broker_open_count
    runtime_state["eod_pending_sqlite_active_count"] = sqlite_active_count
    runtime_state["eod_pending_symbols_count"] = len(symbols)
    if pending and broker_open_count == 0 and sqlite_active_count == 0:
        runtime_state["eod_pending_ignored_count"] = len(symbols)
        runtime_state["eod_pending_pending_restored"] = 0
        runtime_state["eod_pending_ignored_reason"] = "broker_sqlite_flat_on_startup"
        persist_pending_eod_flatten(recorder, runtime_state, pending=False, reason="broker_sqlite_flat_on_startup")
        print(
            f"{now_utc()} EOD_FLATTEN_PENDING_IGNORED_BROKER_FLAT "
            f"eod_pending_file={path} "
            f"broker_open_count=0 sqlite_active_count=0 "
            f"eod_pending_symbols_count={len(symbols)} eod_pending_ignored_count={len(symbols)} "
            f"pending_restored=0 ignored_reason=broker_sqlite_flat_on_startup",
            flush=True,
        )
        return False
    runtime_state["eod_pending_ignored_count"] = 0
    runtime_state["eod_pending_pending_restored"] = int(pending)
    runtime_state["eod_pending_ignored_reason"] = ""
    runtime_state["pending_eod_flatten"] = pending
    runtime_state["pending_eod_flatten_reason"] = str(payload.get("reason") or "")
    runtime_state["pending_eod_flatten_updated_at"] = str(payload.get("recorded_at") or "")
    runtime_state["pending_eod_flatten_symbols"] = symbols
    if pending:
        print(
            f"{now_utc()} EOD_FLATTEN_PENDING_RESTORED reason={runtime_state['pending_eod_flatten_reason']} "
            f"eod_pending_file={path} "
            f"broker_open_count={broker_open_count if broker_open_count is not None else 'unknown'} "
            f"sqlite_active_count={sqlite_active_count if sqlite_active_count is not None else 'unknown'} "
            f"eod_pending_symbols_count={len(symbols)} pending_restored=1 ignored_reason=none "
            f"symbols={','.join(runtime_state['pending_eod_flatten_symbols'])}",
            flush=True,
        )
    return pending


def apply_pending_eod_entry_block(runtime_state: dict[str, Any]) -> bool:
    pending = bool(runtime_state.get("pending_eod_flatten"))
    if pending:
        runtime_state["entries_blocked_reason"] = "pending_eod_flatten"
    elif runtime_state.get("entries_blocked_reason") == "pending_eod_flatten":
        runtime_state["entries_blocked_reason"] = ""
    return pending


def pending_eod_retry_age_seconds(runtime_state: dict[str, Any], now_ts: float | None = None) -> float | None:
    last_retry = safe_float(runtime_state.get("pending_eod_flatten_last_retry_ts"))
    if last_retry is None or last_retry <= 0:
        return None
    now_ts = time.time() if now_ts is None else now_ts
    return max(0.0, now_ts - last_retry)


def record_eod_flatten_giveup_once(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    symbol: str,
    quantity: float,
    reason: str,
    attempt: int,
) -> None:
    seen = runtime_state.setdefault("eod_flatten_giveup_seen", set())
    if not isinstance(seen, set):
        seen = set(seen or [])
        runtime_state["eod_flatten_giveup_seen"] = seen
    key = f"{symbol}:{reason}"
    if key in seen:
        return
    seen.add(key)
    record_lifecycle_with_formal(
        recorder,
        "EOD_FLATTEN_GIVEUP",
        symbol,
        action=_flatten_action_for_quantity(quantity),
        quantity=abs(float(quantity)),
        reason=reason,
        raw_json={"ibkr_quantity": quantity, "attempt": attempt},
    )
    print(
        f"{now_utc()} EOD_FLATTEN_GIVEUP symbol={symbol} quantity={quantity:.4f} "
        f"attempt={attempt} reason={reason}",
        flush=True,
    )


def submit_eod_flatten_order(
    ib: IB,
    recorder: LiveDataRecorder,
    *,
    symbol: str,
    contract: Any,
    ibkr_quantity: float,
    reason: str,
    attempt: int,
    runtime_state: dict[str, Any] | None = None,
) -> int | None:
    if abs(ibkr_quantity) <= 0:
        return None
    if is_fractional_position_quantity(ibkr_quantity):
        action = _flatten_action_for_quantity(ibkr_quantity)
        if runtime_state is not None:
            record_fractional_orphan_manual_required(
                recorder,
                runtime_state,
                symbol=symbol,
                quantity=ibkr_quantity,
                reason=f"{reason}_fractional_quantity_api_unsupported",
                raw_json={"attempt": attempt, "action": action},
            )
            record_eod_flatten_giveup_once(
                recorder,
                runtime_state,
                symbol=symbol,
                quantity=ibkr_quantity,
                reason="fractional_quantity_manual_required",
                attempt=attempt,
            )
        else:
            record_lifecycle_with_formal(
                recorder,
                "EOD_FLATTEN_FAILED",
                symbol,
                action=action,
                quantity=abs(float(ibkr_quantity)),
                reason="fractional_quantity_api_unsupported",
                raw_json={
                    "ibkr_quantity": ibkr_quantity,
                    "attempt": attempt,
                    "manual_action_required": "close_fractional_position_in_ibkr_desktop",
                },
            )
            print(
                f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} quantity={ibkr_quantity:.4f} "
                "reason=fractional_quantity_api_unsupported manual_action_required=ibkr_desktop",
                flush=True,
            )
        return None
    action = _flatten_action_for_quantity(ibkr_quantity)
    quantity = _order_quantity_from_position(ibkr_quantity)
    if quantity <= 0:
        return None
    order_contract = smart_stock_contract_for_flatten(symbol, contract)
    try:
        qualified = ib.qualifyContracts(order_contract)
        if qualified:
            order_contract = qualified[0]
    except Exception as exc:
        record_lifecycle_with_formal(
            recorder,
            "EOD_FLATTEN_FAILED",
            symbol,
            action=action,
            quantity=quantity,
            reason=f"qualify_failed:{exc!r}",
            raw_json={"ibkr_quantity": ibkr_quantity, "attempt": attempt},
        )
        print(f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} reason=qualify_failed error={exc!r}", flush=True)
        return None
    order = MarketOrder(action, quantity)
    order.tif = "DAY"
    order.outsideRth = False
    trade = ib.placeOrder(order_contract, order)
    try:
        ib.sleep(0.5)
    except Exception:
        pass
    order_id = getattr(getattr(trade, "order", None), "orderId", None)
    status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
    if status.lower() == "cancelled":
        log = getattr(trade, "log", None)
        error_message = str(log[-1].message) if log else "cancelled"
        record_lifecycle_with_formal(
            recorder,
            "EOD_FLATTEN_FAILED",
            symbol,
            action=action,
            quantity=quantity,
            order_id=order_id,
            reason=f"ibkr_cancelled:{error_message}",
            raw_json={"ibkr_quantity": ibkr_quantity, "attempt": attempt, "status": status},
        )
        print(
            f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} action={action} quantity={quantity} "
            f"orderId={order_id} reason=ibkr_cancelled error={error_message}",
            flush=True,
        )
        return None
    record_lifecycle_with_formal(
        recorder,
        "EOD_FLATTEN_SUBMIT",
        symbol,
        action=action,
        quantity=quantity,
        order_id=order_id,
        reason=reason,
        raw_json={"ibkr_quantity": ibkr_quantity, "attempt": attempt, "tif": order.tif, "outsideRth": order.outsideRth, "status": status},
    )
    print(
        f"{now_utc()} EOD_FLATTEN_SUBMIT symbol={symbol} action={action} "
        f"quantity={quantity} ibkr_quantity={ibkr_quantity:.4f} attempt={attempt} "
        f"orderId={order_id} reason={reason}",
        flush=True,
    )
    return int(order_id) if order_id is not None else 0


def hard_eod_flatten_portfolio(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
    *,
    reason: str,
) -> int:
    try:
        if hasattr(ib, "isConnected") and not ib.isConnected():
            print(f"{now_utc()} EOD_FLATTEN_FAILED reason=ibkr_not_connected", flush=True)
            persist_pending_eod_flatten(recorder, runtime_state, pending=True, reason="ibkr_not_connected")
            return 0
    except Exception as exc:
        print(f"{now_utc()} EOD_FLATTEN_FAILED reason=ibkr_connection_check_error error={exc!r}", flush=True)
        persist_pending_eod_flatten(recorder, runtime_state, pending=True, reason="ibkr_connection_check_error")
        return 0

    now_ts = time.time()
    rows = ibkr_portfolio_position_rows(ib)
    open_keys = open_flatten_order_keys(ib)
    pending_order_count = len(open_ibkr_order_trades(ib))
    attempt_by_symbol = runtime_state.setdefault("eod_flatten_attempts_by_symbol", {})
    last_submit_by_symbol = runtime_state.setdefault("eod_flatten_last_submit_ts_by_symbol", {})
    if not isinstance(attempt_by_symbol, dict):
        attempt_by_symbol = {}
        runtime_state["eod_flatten_attempts_by_symbol"] = attempt_by_symbol
    if not isinstance(last_submit_by_symbol, dict):
        last_submit_by_symbol = {}
        runtime_state["eod_flatten_last_submit_ts_by_symbol"] = last_submit_by_symbol

    print(
        f"{now_utc()} EOD_FLATTEN_VERIFY open_positions={len(rows)} open_flatten_orders={len(open_keys)} reason={reason}",
        flush=True,
    )
    runtime_state["eod_last_verification"] = {
        "recorded_at": now_utc(),
        "reason": reason,
        "ibkr_open_symbols": sorted(row["symbol"] for row in rows),
        "managed_open_symbols": sorted(symbol for symbol, pos in managed_positions.items() if pos.active),
        "open_flatten_orders": sorted(f"{symbol}:{action}" for symbol, action in open_keys),
    }

    if not rows:
        for pos in managed_positions.values():
            if pos.active:
                pos.active = False
        runtime_state["manual_eod_flatten_requested"] = False
        runtime_state["manual_eod_flatten_force"] = False
        runtime_state["eod_flatten_attempts_by_symbol"] = {}
        runtime_state["eod_flatten_last_submit_ts_by_symbol"] = {}
        runtime_state["pending_eod_flatten"] = False
        persist_pending_eod_flatten(recorder, runtime_state, pending=False, reason="ibkr_flat")
        print(f"{now_utc()} EOD_FLATTEN_SUCCESS open_positions=0", flush=True)
        record_lifecycle_with_formal(
            recorder,
            "EOD_FLATTEN_SUCCESS",
            "__ALL__",
            action="VERIFY",
            quantity=0,
            reason=reason,
            raw_json={"open_positions": 0},
        )
        write_eod_final_status(
            recorder,
            runtime_state,
            rows=[],
            pending_orders=pending_order_count,
            managed_positions=managed_positions,
            reason=reason,
        )
        return 0

    submitted = 0
    force = bool(runtime_state.get("manual_eod_flatten_force", False))
    retry_seconds = float(getattr(args, "eod_retry_seconds", 60))
    max_retries = int(getattr(args, "eod_max_retries", 5))

    for row in rows:
        symbol = row["symbol"]
        ibkr_qty = float(row["quantity"])
        action = _flatten_action_for_quantity(ibkr_qty)
        key = (symbol, action)
        print(
            f"{now_utc()} EOD_POSITION_STILL_OPEN symbol={symbol} quantity={ibkr_qty:.4f} "
            f"action={action} has_open_flatten_order={int(key in open_keys)}",
            flush=True,
        )
        if key in open_keys:
            continue

        attempts = int(attempt_by_symbol.get(symbol, 0) or 0)
        last_submit = safe_float(last_submit_by_symbol.get(symbol))
        retry_due = last_submit is None or (now_ts - last_submit) >= retry_seconds
        if attempts > 0 and not retry_due and not force:
            continue
        if attempts > 0:
            print(f"{now_utc()} EOD_FLATTEN_RETRY symbol={symbol} attempt={attempts + 1}", flush=True)
            record_lifecycle_with_formal(
                recorder,
                "EOD_FLATTEN_RETRY",
                symbol,
                action=action,
                quantity=abs(ibkr_qty),
                reason=reason,
                raw_json={"attempt": attempts + 1, "ibkr_quantity": ibkr_qty},
            )
        if attempts >= max_retries and not force:
            record_eod_flatten_giveup_once(
                recorder,
                runtime_state,
                symbol=symbol,
                quantity=ibkr_qty,
                reason="max_retries_exceeded_retrying",
                attempt=attempts,
            )
        order_id = submit_eod_flatten_order(
            ib,
            recorder,
            symbol=symbol,
            contract=row["contract"],
            ibkr_quantity=ibkr_qty,
            reason=reason,
            attempt=attempts + 1,
            runtime_state=runtime_state,
        )
        attempt_by_symbol[symbol] = attempts + 1
        last_submit_by_symbol[symbol] = now_ts
        if order_id:
            submitted += 1
            pos = managed_positions.get(symbol)
            if pos is not None:
                pos.exit_sent = True
                pos.last_exit_order_ts = now_ts
                pos.eod_retry_count = attempts + 1

    runtime_state["manual_eod_flatten_force"] = False
    try:
        pending_order_count = len(open_ibkr_order_trades(ib))
    except Exception:
        pass
    write_eod_final_status(
        recorder,
        runtime_state,
        rows=rows,
        pending_orders=pending_order_count,
        managed_positions=managed_positions,
        reason=reason,
    )
    persist_pending_eod_flatten(recorder, runtime_state, pending=True, reason=reason, rows=rows)
    return submitted


def process_pending_eod_flatten_retry(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
    *,
    reason: str,
    force: bool = False,
    cooldown_seconds: float | None = None,
) -> int:
    if not bool(runtime_state.get("pending_eod_flatten")) and not force:
        return 0
    if runtime_state.get("eod_recovery_active") and not force:
        return 0
    now_ts = time.time()
    retry_seconds = float(cooldown_seconds if cooldown_seconds is not None else getattr(args, "eod_retry_seconds", 60.0))
    retry_age = pending_eod_retry_age_seconds(runtime_state, now_ts)
    if not force and retry_age is not None and retry_age < retry_seconds:
        return 0
    runtime_state["eod_recovery_active"] = True
    try:
        rows = ibkr_portfolio_position_rows(ib)
        if rows:
            print(
                f"{now_utc()} EOD_FLATTEN_RETRY reason={reason} open_positions={len(rows)} "
                f"symbols={','.join(sorted(row['symbol'] for row in rows))}",
                flush=True,
            )
            for row in rows:
                ibkr_qty = float(row["quantity"])
                record_lifecycle_with_formal(
                    recorder,
                    "EOD_FLATTEN_RETRY",
                    row["symbol"],
                    action=_flatten_action_for_quantity(ibkr_qty),
                    quantity=abs(ibkr_qty),
                    reason=reason,
                    raw_json={
                        "ibkr_quantity": ibkr_qty,
                        "source": "pending_eod_flatten_retry",
                    },
                )
        runtime_state["pending_eod_flatten_last_retry_ts"] = now_ts
        return hard_eod_flatten_portfolio(
            ib,
            recorder,
            managed_positions,
            args,
            runtime_state,
            reason=reason,
        )
    finally:
        runtime_state["eod_recovery_active"] = False


def process_portfolio_sync_pending_eod_retry(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
) -> int:
    return process_pending_eod_flatten_retry(
        ib,
        recorder,
        managed_positions,
        args,
        runtime_state,
        reason="portfolio_sync_pending_eod",
        cooldown_seconds=5.0,
    )


def enforce_eod_flatten_if_due(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
    *,
    eod_active: bool | None = None,
    reason: str = "eod_active_failsafe",
    cooldown_seconds: float = 5.0,
) -> int:
    if eod_active is None:
        eod_active = bool(getattr(args, "enable_eod_flatten", False)) and (
            is_after_utc(getattr(args, "eod_flatten_utc", "19:45"))
            or bool(runtime_state.get("manual_eod_flatten_requested", False))
        )
    if not eod_active:
        return 0

    runtime_state["entries_blocked"] = True
    if runtime_state.get("pending_eod_flatten"):
        runtime_state["entries_blocked_reason"] = "pending_eod_flatten"
    elif not runtime_state.get("entries_blocked_reason"):
        runtime_state["entries_blocked_reason"] = "eod_active"

    active_managed = [symbol for symbol, pos in managed_positions.items() if pos.active]
    try:
        ibkr_rows = ibkr_portfolio_position_rows(ib)
    except Exception as exc:
        print(f"{now_utc()} EOD_FLATTEN_FAILSAFE_PORTFOLIO_ERROR reason={reason} error={exc!r}", flush=True)
        persist_pending_eod_flatten(recorder, runtime_state, pending=True, reason="eod_failsafe_portfolio_error")
        apply_pending_eod_entry_block(runtime_state)
        return 0

    sqlite_active_count = sqlite_active_position_count(recorder)
    if not active_managed and not ibkr_rows and sqlite_active_count == 0 and runtime_state.get("pending_eod_flatten"):
        ignored_count = len(runtime_state.get("pending_eod_flatten_symbols") or [])
        runtime_state["eod_pending_ignored_count"] = ignored_count
        runtime_state["eod_pending_pending_restored"] = 0
        runtime_state["eod_pending_ignored_reason"] = "broker_sqlite_flat_eod_failsafe"
        persist_pending_eod_flatten(recorder, runtime_state, pending=False, reason="broker_sqlite_flat_eod_failsafe")
        print(
            f"{now_utc()} EOD_FLATTEN_PENDING_IGNORED_BROKER_FLAT reason={reason} "
            f"eod_pending_file={recorder.path('eod_pending.json')} "
            f"broker_open_count=0 sqlite_active_count=0 "
            f"eod_pending_symbols_count={ignored_count} eod_pending_ignored_count={ignored_count} "
            f"pending_restored=0 ignored_reason=broker_sqlite_flat_eod_failsafe",
            flush=True,
        )
        return 0

    if not active_managed and not ibkr_rows and not runtime_state.get("pending_eod_flatten"):
        return 0

    now_ts = time.time()
    last_enforced = safe_float(runtime_state.get("eod_failsafe_last_enforced_ts"))
    if last_enforced is not None and (now_ts - last_enforced) < cooldown_seconds:
        return 0

    runtime_state["eod_failsafe_last_enforced_ts"] = now_ts
    print(
        f"{now_utc()} EOD_FLATTEN_FAILSAFE_TRIGGER reason={reason} "
        f"managed_open={len(active_managed)} ibkr_open={len(ibkr_rows)} "
        f"sqlite_active_count={sqlite_active_count if sqlite_active_count is not None else 'unknown'} "
        f"pending_eod_flatten={int(bool(runtime_state.get('pending_eod_flatten')))}",
        flush=True,
    )
    submitted = hard_eod_flatten_portfolio(
        ib,
        recorder,
        managed_positions,
        args,
        runtime_state,
        reason=reason,
    )
    apply_pending_eod_entry_block(runtime_state)
    return submitted


def verify_managed_positions_against_ibkr(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    *,
    reason: str,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_state = runtime_state if runtime_state is not None else {}
    quantities = ibkr_position_quantities(ib)
    open_symbols: list[str] = []
    closed_symbols: list[str] = []
    drift_symbols: list[str] = []

    for symbol, pos in managed_positions.items():
        if not pos.active:
            continue
        ib_qty = quantities.get(symbol, 0.0)
        if abs(ib_qty) <= 0:
            backfill_recent_fills_for_verification(ib, recorder, runtime_state)
            if not pos.entry_fill_verified and entry_fill_verified(recorder, symbol, pos.quantity):
                pos.entry_fill_verified = True
            if not pos.entry_fill_verified:
                pos.active = False
                closed_symbols.append(symbol)
                record_entry_not_filled(
                    recorder,
                    symbol,
                    quantity=pos.quantity,
                    reason=reason,
                    ibkr_quantity=ib_qty,
                    runtime_state=runtime_state,
                    raw_json={
                        "entry_price": pos.entry_price,
                        "peak_price": pos.peak_price,
                        "exit_sent": pos.exit_sent,
                    },
                )
                continue
            if not pos.exit_sent:
                drift_symbols.append(symbol)
                record_lifecycle_with_formal(
                    recorder,
                    "POSITION_MISSING_IN_IBKR",
                    symbol,
                    action="VERIFY",
                    quantity=pos.quantity,
                    reason=reason,
                    raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ib_qty},
                )
                continue
            if not close_fill_verified(recorder, symbol, pos.quantity):
                pos.active = False
                closed_symbols.append(symbol)
                record_unverified_reconciliation_close(
                    recorder,
                    symbol,
                    quantity=pos.quantity,
                    reason=reason,
                    ibkr_quantity=ib_qty,
                    runtime_state=runtime_state,
                    raw_json={
                        "entry_price": pos.entry_price,
                        "peak_price": pos.peak_price,
                        "exit_sent": pos.exit_sent,
                        "entry_fill_verified": True,
                    },
                )
                continue
            pos.active = False
            closed_symbols.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_VERIFIED_CLOSED",
                symbol,
                action="VERIFY",
                quantity=pos.quantity,
                reason=reason,
                entry_price=pos.entry_price,
                peak_price=pos.peak_price,
                close_source="fill",
                fill_verified="true",
                entry_fill_verified="true",
                close_fill_verified="true",
                raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ib_qty, "fill_verified": True, "entry_fill_verified": True, "close_fill_verified": True, "close_source": "fill"},
            )
            safe_sqlite_call(
                getattr(recorder, "sqlite_store", None),
                "mark_position_flat",
                symbol=symbol,
                strategy_name=STRATEGY_NAME,
                reason=reason,
                status="CLOSED",
            )
            continue
        open_symbols.append(symbol)
        if int(abs(ib_qty)) != int(abs(pos.quantity)):
            drift_symbols.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_QUANTITY_DRIFT",
                symbol,
                action="VERIFY",
                quantity=pos.quantity,
                reason=reason,
                raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ib_qty},
            )

    result = {
        "recorded_at": now_utc(),
        "reason": reason,
        "ibkr_open_symbols": sorted(quantities.keys()),
        "managed_open_symbols": sorted(open_symbols),
        "closed_symbols": sorted(closed_symbols),
        "drift_symbols": sorted(drift_symbols),
    }
    return result


def managed_positions_to_lifecycle_positions(
    managed_positions: dict[str, ManagedPosition],
) -> dict[str, PositionRecord]:
    positions: dict[str, PositionRecord] = {}
    session_date = datetime.now(timezone.utc).date().isoformat()
    for symbol, pos in managed_positions.items():
        if not pos.active:
            continue
        fill_verified_for_state = bool(pos.entry_fill_verified or pos.exit_sent)
        if not fill_verified_for_state:
            open_qty = 0.0
            state = PositionState.ENTRY_PENDING
            entry_filled_quantity = 0.0
        else:
            open_qty = 0.0 if pos.exit_sent else float(pos.quantity)
            state = PositionState.EXIT_PENDING if pos.exit_sent else PositionState.OPEN
            entry_filled_quantity = float(pos.quantity)
        positions[symbol] = PositionRecord(
            symbol=symbol,
            strategy=STRATEGY_NAME,
            session_date=session_date,
            state=state,
            target_quantity=float(pos.quantity),
            entry_filled_quantity=entry_filled_quantity,
            exit_filled_quantity=0.0,
            avg_entry_price=pos.entry_price,
            open_quantity=open_qty,
            peak_price=pos.peak_price,
        )
    return positions


def ibkr_open_order_rows(ib: IB) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        trades = list(ib.openTrades())
    except Exception:
        trades = []
    for trade in trades:
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        status = getattr(trade, "orderStatus", None)
        rows.append({
            "symbol": str(getattr(contract, "symbol", "") or "").upper(),
            "ib_order_id": str(getattr(order, "orderId", "") or ""),
            "perm_id": str(getattr(order, "permId", "") or ""),
            "action": str(getattr(order, "action", "") or ""),
            "total_quantity": safe_float(getattr(order, "totalQuantity", None)),
            "status": str(getattr(status, "status", "") or ""),
            "filled": safe_float(getattr(status, "filled", None)),
            "remaining": safe_float(getattr(status, "remaining", None)),
        })
    if rows:
        return rows
    try:
        orders = list(ib.openOrders())
    except Exception:
        orders = []
    for order in orders:
        rows.append({
            "symbol": "",
            "ib_order_id": str(getattr(order, "orderId", "") or ""),
            "perm_id": str(getattr(order, "permId", "") or ""),
            "action": str(getattr(order, "action", "") or ""),
            "total_quantity": safe_float(getattr(order, "totalQuantity", None)),
            "status": "",
            "filled": None,
            "remaining": None,
        })
    return rows


def run_dry_run_reconciliation_report(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
    *,
    reason: str,
) -> None:
    try:
        positions = managed_positions_to_lifecycle_positions(managed_positions)
        ibkr_quantities = ibkr_position_quantities(ib)
        open_orders = ibkr_open_order_rows(ib)
        lifecycle_events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
        report = build_reconciliation_report(
            positions,
            ibkr_quantities,
            lifecycle_events=lifecycle_events,
            open_orders=open_orders,
        )
        runtime_state["reconciliation_last_report"] = report.to_dict()
        runtime_state["reconciliation_last_reason"] = reason
        signature = json.dumps(
            {
                "missing": report.missing_in_ibkr,
                "orphan": report.orphan_in_ibkr,
                "drift": report.quantity_drift,
                "pending": report.pending_order_ids,
            },
            sort_keys=True,
            default=str,
        )
        last_signature = runtime_state.get("reconciliation_last_logged_signature")
        if signature == last_signature:
            suppressed = int(runtime_state.get("reconciliation_repeat_suppressed", 0) or 0) + 1
            runtime_state["reconciliation_repeat_suppressed"] = suppressed
            last_summary = float(runtime_state.get("reconciliation_last_suppressed_summary_at", 0.0) or 0.0)
            now_mono = time.monotonic()
            if now_mono - last_summary >= 60.0:
                print(
                    f"{now_utc()} RECONCILIATION_REPEAT_SUPPRESSED count={suppressed} clean={int(report.clean)} "
                    f"missing={len(report.missing_in_ibkr)} orphan={len(report.orphan_in_ibkr)} "
                    f"drift={len(report.quantity_drift)} pending_orders={len(report.pending_order_ids)}",
                    flush=True,
                )
                runtime_state["reconciliation_repeat_suppressed"] = 0
                runtime_state["reconciliation_last_suppressed_summary_at"] = now_mono
            return
        runtime_state["reconciliation_last_logged_signature"] = signature
        runtime_state["reconciliation_repeat_suppressed"] = 0
        runtime_state["reconciliation_last_suppressed_summary_at"] = time.monotonic()
        log_reconciliation_report(report, prefix=now_utc())
    except Exception as exc:
        print(f"{now_utc()} RECONCILIATION_ERROR reason={reason} error={exc!r}", flush=True)


def adopt_existing_long_positions(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    latest_snapshots: dict[str, dict[str, Any]],
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any] | None = None,
) -> int:
    adopted = 0
    exit_sent_symbols = load_exit_sent_symbols(recorder)
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper().strip()
        if not symbol or symbol in managed_positions:
            continue
        if symbol in exit_sent_symbols:
            record_lifecycle_with_formal(
                recorder,
                "ADOPT_DESPITE_EXIT_SENT",
                symbol,
                action="ADOPT",
                reason="ibkr_portfolio_still_has_position_after_lifecycle_sell_order",
            )
        quantity = safe_float(getattr(item, "position", None))
        avg_cost = safe_float(getattr(item, "averageCost", None))
        market_price = safe_float(getattr(item, "marketPrice", None))
        if quantity is None or quantity <= 0:
            continue
        if is_fractional_position_quantity(quantity):
            state = runtime_state if runtime_state is not None else {}
            record_fractional_orphan_manual_required(
                recorder,
                state,
                symbol=symbol,
                quantity=quantity,
                price=market_price,
                reason="fractional_ibkr_position_requires_manual_desktop_close",
                raw_json={"average_cost": avg_cost, "market_price": market_price},
            )
            continue
        in_runtime_universe = symbol in contract_by_symbol
        if not in_runtime_universe:
            record_lifecycle_with_formal(
                recorder,
                "ORPHAN_IBKR_POSITION_OBSERVED",
                symbol,
                action="ALERT",
                quantity=quantity,
                price=market_price,
                reason="external_ibkr_position_not_adopted_into_strategy",
                raw_json={"ibkr_quantity": quantity, "average_cost": avg_cost, "market_price": market_price},
            )
            print(f"{now_utc()} ORPHAN_IBKR_POSITION_OBSERVED symbol={symbol} quantity={quantity:.4f} reason=external_not_adopted", flush=True)
            continue
        entry_price = avg_cost or market_price
        if entry_price is None or entry_price <= 0:
            continue
        contract = contract_by_symbol.get(symbol) or item.contract
        if contract is None:
            contract = Stock(symbol, "SMART", "USD")
        contract_by_symbol.setdefault(symbol, contract)
        snap_price = safe_float((latest_snapshots.get(symbol) or {}).get("price"))
        peak_price = max(entry_price, snap_price or market_price or entry_price)
        managed_positions[symbol] = ManagedPosition(
            symbol=symbol,
            contract=contract,
            quantity=int(quantity),
            entry_price=float(entry_price),
            entry_time=f"adopted_on_restart:{now_utc()}",
            peak_price=float(peak_price),
            source="adopted_from_ibkr_portfolio_top100" if in_runtime_universe else "adopted_from_ibkr_portfolio_external",
            entry_fill_verified=True,
        )
        record_lifecycle_with_formal(
            recorder,
            "ADOPTED_POSITION",
            symbol,
            action="ADOPT",
            quantity=int(quantity),
            price=snap_price or market_price,
            entry_price=entry_price,
            peak_price=peak_price,
            reason="restart_recovery_from_ibkr_portfolio",
            entry_fill_verified="true",
        )
        print(f"ADOPTED EXISTING POSITION symbol={symbol} qty={int(quantity)} entry={entry_price:.2f} peak={peak_price:.2f}", flush=True)
        adopted += 1
    return adopted


def manage_exits(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
) -> int:
    exits = 0
    manual_eod = bool(runtime_state.get("manual_eod_flatten_requested", False))
    manual_eod_force = bool(runtime_state.get("manual_eod_flatten_force", False))
    eod = args.enable_eod_flatten and (manual_eod or is_eod_flatten_time(args.eod_flatten_utc))
    if not eod and not is_after_utc(getattr(args, "manage_exits_start_utc", "13:30")):
        return 0

    if eod:
        runtime_state["entries_blocked"] = True
        try:
            exits += hard_eod_flatten_portfolio(
                ib,
                recorder,
                managed_positions,
                args,
                runtime_state,
                reason="manual_eod_flatten" if manual_eod else "scheduled_eod_flatten",
            )
            persist_managed_positions(recorder, managed_positions)
            return exits
        except Exception as exc:
            print(f"{now_utc()} EOD_FLATTEN_FAILED reason=portfolio_flatten_exception error={exc!r}", flush=True)

    for symbol, pos in list(managed_positions.items()):
        if not pos.active or pos.exit_sent or not pos.entry_fill_verified:
            continue
        snap = latest_snapshots.get(symbol) or {}
        price = safe_float(snap.get("price"))

        if price is None or price <= 0:
            continue

        old_peak = pos.peak_price
        if pos.low_price is None or pos.low_price <= 0:
            pos.low_price = pos.entry_price
        pos.peak_price = max(pos.peak_price, price)
        pos.low_price = min(pos.low_price, price)
        update_managed_position_excursion(pos, price, observed_at=now_utc())
        if pos.peak_price != old_peak:
            record_lifecycle(recorder, "PEAK_UPDATED", symbol, price=price, entry_price=pos.entry_price, peak_price=pos.peak_price)

        stop_price = pos.entry_price * (1.0 - args.exit_stop_loss_pct / 100.0)
        peak_pnl_pct = (pos.peak_price / pos.entry_price - 1.0) * 100.0
        reason = None
        if price <= stop_price:
            reason = "v46_wide_trail_stop_loss"
        elif peak_pnl_pct >= args.exit_trailing_activation_pct:
            trail_price = pos.peak_price * (1.0 - args.exit_trailing_stop_pct / 100.0)
            if price <= trail_price:
                reason = "v46_wide_trail_trailing_stop"
        if reason is None and eod:
            reason = "v46_wide_trail_close_exit_eod"
        if reason:
            if not pos.exit_sent and send_exit_order(ib, recorder, pos, reason, price):
                exits += 1
    return exits


def utc_minutes_now():
    from datetime import datetime, timezone
    t = datetime.now(timezone.utc).time()
    return t.hour * 60 + t.minute


def parse_utc_hhmm(value):
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    if not 0 <= hh <= 23 or not 0 <= mm <= 59:
        raise ValueError(f"Invalid UTC time HH:MM: {value}")
    return hh * 60 + mm


def is_after_utc(value):
    return utc_minutes_now() >= parse_utc_hhmm(value)


def parse_utc_schedule(value: str) -> list[str]:
    slots: list[str] = []
    for raw in str(value or "").split(","):
        slot = raw.strip()
        if not slot:
            continue
        parse_utc_hhmm(slot)
        slots.append(slot)
    return slots


def is_utc_slot_due(now: datetime, slot: str, *, window_minutes: int = 10) -> bool:
    now_min = now.hour * 60 + now.minute
    slot_min = parse_utc_hhmm(slot)
    return slot_min <= now_min < slot_min + max(1, int(window_minutes))


def latest_completed_trading_day(now: datetime, market_close_utc: str) -> date:
    close_min = parse_utc_hhmm(market_close_utc)
    now_min = now.hour * 60 + now.minute
    cur = now.date()
    if is_us_equity_trading_day(cur) and now_min >= close_min:
        return cur
    return previous_us_equity_trading_day(cur)


def _runtime_set(runtime_state: dict[str, Any], key: str) -> set[str]:
    value = runtime_state.setdefault(key, set())
    if isinstance(value, set):
        return value
    if isinstance(value, list):
        out = {str(item) for item in value}
        runtime_state[key] = out
        return out
    out: set[str] = set()
    runtime_state[key] = out
    return out


def runtime_rate_limited_log(
    runtime_state: dict[str, Any],
    bucket: str,
    message: str,
    *,
    key: str,
    max_unique: int = 20,
    window_seconds: float = 60.0,
) -> bool:
    state = runtime_state.setdefault("rate_limited_log_state", {})
    if not isinstance(state, dict):
        state = {}
        runtime_state["rate_limited_log_state"] = state
    now = time.monotonic()
    item = state.setdefault(bucket, {"window_start": now, "keys": set(), "suppressed": 0})
    if now - float(item.get("window_start", now)) >= window_seconds:
        suppressed = int(item.get("suppressed", 0) or 0)
        if suppressed:
            print(f"{now_utc()} {bucket}_SUPPRESSED count={suppressed}", flush=True)
        item["window_start"] = now
        item["keys"] = set()
        item["suppressed"] = 0
    keys = item.setdefault("keys", set())
    if not isinstance(keys, set):
        keys = set()
        item["keys"] = keys
    if key in keys:
        return False
    if len(keys) >= max_unique:
        item["suppressed"] = int(item.get("suppressed", 0) or 0) + 1
        return False
    keys.add(key)
    print(message, flush=True)
    return True


def overnight_backlog_start_date(end_date: str, args: argparse.Namespace) -> str:
    lookback_days = int(getattr(args, "overnight_backlog_lookback_days", 30) or 0)
    if lookback_days > 0:
        try:
            end = date.fromisoformat(str(end_date))
            return (end - timedelta(days=lookback_days)).isoformat()
        except Exception:
            pass
    return str(getattr(args, "overnight_collector_start_date", "2026-01-01"))


def history_parquet_path(history_dir: str | Path, symbol: str, session_date: date, session_type: str = "RTH") -> Path:
    return (
        Path(history_dir)
        / f"session_type={session_type.upper()}"
        / f"symbol={symbol.upper()}"
        / f"year={session_date.year:04d}"
        / f"month={session_date.month:02d}"
        / f"day={session_date.day:02d}.parquet"
    )


def history_task_key(symbol: str, session_date: date, session_type: str = "RTH") -> str:
    return f"{symbol.upper()}_{session_date.isoformat()}_{session_type.upper()}"


def load_history_status(history_dir: str | Path) -> dict[str, Any]:
    status_path = Path(history_dir).parent / "collector_status.json"
    try:
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception as exc:
        print(f"{now_utc()} STARTUP_HISTORY_REPAIR_STATUS_LOAD_FAILED path={status_path} error={exc!r}", flush=True)
    return {}


def load_history_universe_symbols(universe_csv: str | Path) -> list[str]:
    try:
        df = pd.read_csv(universe_csv)
    except Exception as exc:
        print(f"{now_utc()} STARTUP_HISTORY_REPAIR_UNIVERSE_LOAD_FAILED path={universe_csv} error={exc!r}", flush=True)
        return []
    if "symbol" not in df.columns:
        print(f"{now_utc()} STARTUP_HISTORY_REPAIR_UNIVERSE_LOAD_FAILED path={universe_csv} reason=missing_symbol_column", flush=True)
        return []
    symbols = [str(symbol).upper().strip() for symbol in df["symbol"].tolist()]
    return [symbol for symbol in symbols if symbol]


def assess_history_completion(
    *,
    history_dir: str | Path,
    universe_csv: str | Path,
    session_date: date,
    session_type: str = "RTH",
) -> dict[str, Any]:
    symbols = load_history_universe_symbols(universe_csv)
    status = load_history_status(history_dir)
    parquet_files = 0
    complete_symbols = 0
    partial_symbols = 0
    no_data_symbols = 0
    failed = 0
    missing = 0
    for symbol in symbols:
        path = history_parquet_path(history_dir, symbol, session_date, session_type)
        has_parquet = path.exists() and path.stat().st_size > 0
        if has_parquet:
            parquet_files += 1
        row = status.get(history_task_key(symbol, session_date, session_type))
        row_status = str(row.get("status") or "").lower() if isinstance(row, dict) else ""
        if has_parquet or row_status == "complete":
            complete_symbols += 1
        elif row_status == "partial":
            partial_symbols += 1
        elif row_status in {"no_data", "no_data_permanent"}:
            no_data_symbols += 1
        elif row_status in {"failed", "failed_permanent"}:
            failed += 1
        if not has_parquet and row_status not in {"complete", "partial", "no_data", "no_data_permanent", "failed", "failed_permanent"}:
            missing += 1
    expected = len(symbols)
    terminal = complete_symbols + no_data_symbols
    completion_pct_value = round((terminal / expected) * 100.0, 2) if expected else 100.0
    ready = expected > 0 and missing == 0 and partial_symbols == 0 and failed == 0
    readiness_status = "OK" if ready else ("PARTIAL" if terminal or partial_symbols else "NOT_READY")
    return {
        "date": session_date.isoformat(),
        "session_type": session_type.upper(),
        "expected_symbols": expected,
        "parquet_files": parquet_files,
        "status_done": terminal,
        "complete_symbols": complete_symbols,
        "partial_symbols": partial_symbols,
        "no_data_symbols": no_data_symbols,
        "failed": failed,
        "missing": missing,
        "failed_symbols": failed,
        "missing_symbols": missing,
        "completed": terminal,
        "completion_pct": completion_pct_value,
        "readiness_status": readiness_status,
        "ready": ready,
    }


def latest_history_is_complete(
    *,
    args: argparse.Namespace,
    session_date: date,
    session_type: str = "RTH",
    min_completion_pct: float = 100.0,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assessment = assess_history_completion(
        history_dir=getattr(args, "daily_top100_history_dir", DEFAULT_HISTORY_DIR),
        universe_csv=getattr(args, "daily_top100_universe", DEFAULT_UNIVERSE),
        session_date=session_date,
        session_type=session_type,
    )
    expected = int(assessment.get("expected_symbols") or 0)
    completion_pct_value = float(assessment.get("completion_pct") or 0.0)
    failed = int(assessment.get("failed") or 0)
    complete = (
        bool(assessment.get("ready"))
        and expected > 0
        and completion_pct_value >= min_completion_pct
        and failed == 0
    )
    latest_output = Path(getattr(args, "daily_top100_latest_output", "data/universe/daily_top100_latest.csv"))
    latest_age = None
    try:
        if latest_output.exists():
            latest_age = round(time.time() - latest_output.stat().st_mtime, 1)
    except Exception:
        latest_age = None
    result = {**assessment, "complete": complete, "min_completion_pct": min_completion_pct, "latest_top100_file_age": latest_age}
    last_collector_run = ""
    last_successful_collector_run = ""
    if runtime_state is not None:
        last_collector_run = str(runtime_state.get("history_collector_last_run_at") or "")
        last_successful_collector_run = str(runtime_state.get("history_collector_last_successful_run_at") or "")
    print(
        f"{now_utc()} HISTORY_READINESS_CHECK ranking_date={session_date.isoformat()} "
        f"expected_symbols={result.get('expected_symbols')} complete_symbols={result.get('complete_symbols')} "
        f"partial_symbols={result.get('partial_symbols')} missing_symbols={result.get('missing_symbols')} "
        f"no_data_symbols={result.get('no_data_symbols')} failed_symbols={result.get('failed_symbols')} "
        f"completion_pct={result.get('completion_pct')} readiness_status={result.get('readiness_status')} "
        f"last_collector_run={last_collector_run} last_successful_collector_run={last_successful_collector_run} "
        f"latest_top100_file_age={latest_age}",
        flush=True,
    )
    return result


def _history_command_priority(command: dict[str, Any]) -> int:
    mode = str(command.get("collector_mode") or "")
    if mode in {"daily", "startup_repair"}:
        return 0
    if mode == "backlog":
        return 10
    return 5


def top100_freshness_state(args: argparse.Namespace, ranking_date: date) -> dict[str, Any]:
    latest = Path(getattr(args, "daily_top100_latest_output", "data/universe/daily_top100_latest.csv"))
    output_dir = Path(getattr(args, "daily_top100_output_dir", "data/universe"))
    dated = output_dir / f"daily_top100_{ranking_date.isoformat()}.csv"
    latest_exists = latest.exists()
    dated_exists = dated.exists()
    rows = 0
    latest_matches_dated = False
    if dated_exists:
        try:
            rows = len(pd.read_csv(dated))
        except Exception:
            rows = 0
    if latest_exists and dated_exists:
        try:
            latest_matches_dated = latest.read_bytes() == dated.read_bytes()
        except Exception:
            latest_matches_dated = False
    latest_age = None
    try:
        if latest_exists:
            latest_age = round(time.time() - latest.stat().st_mtime, 1)
    except Exception:
        latest_age = None
    required_rows = int(getattr(args, "daily_top100_top_n", 100) or 100)
    ready = latest_exists and dated_exists and rows >= required_rows and latest_matches_dated
    reason = "ok"
    if not latest_exists:
        reason = "latest_missing"
    elif not dated_exists:
        reason = "dated_missing"
    elif rows < required_rows:
        reason = "too_few_rows"
    elif not latest_matches_dated:
        reason = "latest_not_matching_ranking_date"
    return {
        "ready": ready,
        "reason": reason,
        "ranking_date": ranking_date.isoformat(),
        "latest_output": str(latest),
        "dated_output": str(dated),
        "latest_exists": latest_exists,
        "dated_exists": dated_exists,
        "rows": rows,
        "required_rows": required_rows,
        "latest_matches_dated": latest_matches_dated,
        "latest_top100_file_age": latest_age,
    }


def apply_top100_freshness_gate(runtime_state: dict[str, Any], args: argparse.Namespace, ranking_date: date) -> dict[str, Any]:
    state = top100_freshness_state(args, ranking_date)
    runtime_state["top100_freshness"] = state
    allow_stale = bool(getattr(args, "allow_stale_top100", False))
    if state["ready"]:
        runtime_state["top100_entries_blocked"] = False
        if runtime_state.get("entries_blocked_reason") == "stale_top100":
            runtime_state["entries_blocked_reason"] = ""
        return state
    if allow_stale:
        key = f"{state['ranking_date']}_{state['reason']}_allow"
        logged = _runtime_set(runtime_state, "top100_stale_allow_logged")
        if key not in logged:
            logged.add(key)
            print(
                f"{now_utc()} DAILY_TOP100_USING_STALE_ALLOWED ranking_date={state['ranking_date']} "
                f"reason={state['reason']} latest_output={state['latest_output']} dated_output={state['dated_output']} "
                f"latest_top100_file_age={state['latest_top100_file_age']}",
                flush=True,
            )
        runtime_state["top100_entries_blocked"] = False
        return state
    runtime_state["top100_entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = "stale_top100"
    key = f"{state['ranking_date']}_{state['reason']}_blocked"
    logged = _runtime_set(runtime_state, "top100_stale_block_logged")
    if key not in logged:
        logged.add(key)
        print(
            f"{now_utc()} DAILY_TOP100_USING_STALE_BLOCKED ranking_date={state['ranking_date']} "
            f"reason={state['reason']} latest_output={state['latest_output']} dated_output={state['dated_output']} "
            f"rows={state['rows']} required_rows={state['required_rows']} "
            f"latest_matches_dated={int(bool(state['latest_matches_dated']))} "
            f"latest_top100_file_age={state['latest_top100_file_age']}",
            flush=True,
        )
    return state


def enqueue_startup_history_repair_if_needed(
    runtime_state: dict[str, Any],
    args: argparse.Namespace,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not bool(getattr(args, "startup_history_repair", True)):
        return {"queued": False, "reason": "disabled"}
    now = now or datetime.now(timezone.utc)
    target_date = previous_us_equity_trading_day(now.date())
    session_type = "RTH"
    assessment = assess_history_completion(
        history_dir=getattr(args, "daily_top100_history_dir", DEFAULT_HISTORY_DIR),
        universe_csv=getattr(args, "daily_top100_universe", DEFAULT_UNIVERSE),
        session_date=target_date,
        session_type=session_type,
    )
    min_pct = float(getattr(args, "startup_history_repair_min_completion_pct", 100.0))
    retry_failed = bool(getattr(args, "startup_history_repair_retry_failed", True))
    incomplete = assessment["expected_symbols"] > 0 and assessment["completion_pct"] < min_pct
    has_failed = int(assessment.get("failed") or 0) > 0
    should_queue = incomplete or (retry_failed and has_failed)
    print(
        f"{now_utc()} STARTUP_HISTORY_REPAIR_CHECK date={assessment['date']} "
        f"completion_pct={assessment['completion_pct']} expected_symbols={assessment['expected_symbols']} "
        f"parquet_files={assessment['parquet_files']} status_done={assessment['status_done']} "
        f"missing={assessment['missing']} failed={assessment['failed']} min_completion_pct={min_pct}",
        flush=True,
    )
    if not should_queue:
        return {**assessment, "queued": False, "reason": "complete"}
    queue = runtime_state.setdefault("history_collector_commands", [])
    if not isinstance(queue, list):
        queue = []
        runtime_state["history_collector_commands"] = queue
    if runtime_state.get("history_collector_process") is not None:
        print(
            f"{now_utc()} STARTUP_HISTORY_REPAIR_SKIPPED date={assessment['date']} "
            f"reason=collector_already_running pending={len(queue)}",
            flush=True,
        )
        return {**assessment, "queued": False, "reason": "collector_already_running"}
    existing_latest = any(
        str(cmd.get("end_date")) == str(assessment["date"]) and str(cmd.get("collector_mode")) in {"daily", "startup_repair"}
        for cmd in queue
        if isinstance(cmd, dict)
    )
    if existing_latest:
        print(
            f"{now_utc()} STARTUP_HISTORY_REPAIR_SKIPPED date={assessment['date']} "
            f"reason=latest_day_already_queued pending={len(queue)}",
            flush=True,
        )
        return {**assessment, "queued": False, "reason": "latest_day_already_queued"}
    command_id = f"startup_history_repair_{assessment['date'].replace('-', '')}"
    command = {
        "id": command_id,
        "type": "history_collector",
        "source": "startup_history_repair",
        "collector_mode": "startup_repair",
        "schedule_slot_utc": "startup",
        "start_date": assessment["date"],
        "end_date": assessment["date"],
        "session_type": session_type,
        "max_tasks": int(getattr(args, "startup_history_repair_max_tasks", getattr(args, "overnight_daily_collector_max_tasks", 3000))),
        "max_attempts": int(getattr(args, "overnight_collector_max_attempts", 5)),
        "limit_symbols": 0,
        "client_id": int(getattr(args, "history_collector_client_id", 168)),
        "force": True,
        "allow_live_session": False,
        "plan_only": False,
        "include_weekends": False,
        "retry_failed": retry_failed,
    }
    queue.insert(0, command)
    queue.sort(key=_history_command_priority)
    print(
        f"{now_utc()} STARTUP_HISTORY_REPAIR_QUEUED command_id={command_id} "
        f"start={command['start_date']} end={command['end_date']} missing={assessment['missing']} "
        f"failed={assessment['failed']} completion_pct={assessment['completion_pct']}",
        flush=True,
    )
    return {**assessment, "queued": True, "command_id": command_id}


def enqueue_overnight_collector_if_due(runtime_state: dict[str, Any], args: argparse.Namespace, now: datetime | None = None) -> None:
    if not bool(getattr(args, "enable_overnight_automation", True)):
        return
    now = now or datetime.now(timezone.utc)
    slots = parse_utc_schedule(getattr(args, "overnight_collector_times_utc", ""))
    if not slots:
        return

    run_keys = _runtime_set(runtime_state, "overnight_collector_run_keys")
    skip_keys = _runtime_set(runtime_state, "overnight_collector_skip_logged_keys")
    queue = runtime_state.setdefault("history_collector_commands", [])
    if not isinstance(queue, list):
        queue = []
        runtime_state["history_collector_commands"] = queue

    backlog_slots = set(parse_utc_schedule(getattr(args, "overnight_backlog_collector_times_utc", "")))
    prioritize_previous_day = bool(getattr(args, "overnight_prioritize_previous_day", True))

    for slot in slots:
        if not is_utc_slot_due(now, slot):
            continue
        latest_day = latest_completed_trading_day(now, getattr(args, "market_close_utc", "20:00"))
        end_date = latest_day.isoformat()
        min_pct = float(getattr(args, "startup_history_repair_min_completion_pct", 100.0))
        latest_assessment = latest_history_is_complete(args=args, session_date=latest_day, min_completion_pct=min_pct, runtime_state=runtime_state)
        latest_complete = bool(latest_assessment.get("complete"))
        modes: list[str] = []
        if not latest_complete:
            modes.append("daily")
        elif slot in backlog_slots:
            modes.append("backlog")
        if not modes:
            continue
        if not prioritize_previous_day:
            modes = ["backlog"]
        key = f"{end_date}_{slot}_{'+'.join(modes)}"
        if key in run_keys:
            continue
        if (
            runtime_state.get("history_collector_process") is not None
            and not latest_complete
            and str((runtime_state.get("history_collector_running_command") or {}).get("collector_mode") or "") == "backlog"
        ):
            running = runtime_state.get("history_collector_running_command") or {}
            proc = runtime_state.get("history_collector_process")
            print(
                f"{now_utc()} OVERNIGHT_COLLECTOR_BACKLOG_INTERRUPTED_FOR_LATEST_DAY "
                f"latest_date={end_date} running_command_id={running.get('id')} "
                f"completion_pct={latest_assessment.get('completion_pct')} missing={latest_assessment.get('missing')}",
                flush=True,
            )
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            runtime_state["history_collector_process"] = None
            runtime_state["history_collector_running_command"] = None
            runtime_state["history_collector_started_monotonic"] = None
            runtime_state["history_collector_last_returncode"] = -9
        if runtime_state.get("history_collector_process") is not None or queue:
            if key not in skip_keys:
                print(
                    f"{now_utc()} OVERNIGHT_COLLECTOR_SKIPPED slot={slot} end={end_date} "
                    f"reason=collector_already_running_or_queued pending={len(queue)}",
                    flush=True,
                )
                skip_keys.add(key)
            continue
        queued_commands = []
        for mode in modes:
            command_id = f"overnight_{mode}_{end_date}_{slot.replace(':', '')}"
            start_date = end_date if mode == "daily" else overnight_backlog_start_date(end_date, args)
            max_tasks = (
                int(getattr(args, "overnight_daily_collector_max_tasks", 3000))
                if mode == "daily"
                else int(getattr(args, "overnight_collector_max_tasks", 3000))
            )
            command = {
                "id": command_id,
                "type": "history_collector",
                "source": "overnight_scheduler",
                "collector_mode": mode,
                "schedule_slot_utc": slot,
                "start_date": start_date,
                "end_date": end_date,
                "session_type": "RTH",
                "max_tasks": max_tasks,
                "max_attempts": int(getattr(args, "overnight_collector_max_attempts", 5)),
                "limit_symbols": 0,
                "client_id": int(getattr(args, "history_collector_client_id", 168)),
                "force": False,
                "allow_live_session": False,
                "plan_only": False,
                "include_weekends": False,
                "retry_failed": True if mode == "daily" else bool(getattr(args, "overnight_collector_retry_failed", False)),
                "priority_recent_catchup": True,
                "recent_sessions": int(getattr(args, "history_collector_recent_sessions", 5) or 5),
            }
            queue.append(command)
            queued_commands.append(command)
        queue.sort(key=_history_command_priority)
        run_keys.add(key)
        skip_keys.discard(key)
        for command in queued_commands:
            print(
                f"{now_utc()} OVERNIGHT_COLLECTOR_QUEUED command_id={command['id']} mode={command['collector_mode']} "
                f"slot={slot} start={command['start_date']} end={end_date} max_tasks={command['max_tasks']} "
                f"latest_day_complete={int(latest_complete)} latest_day_completion_pct={latest_assessment.get('completion_pct')} "
                f"latest_day_missing={latest_assessment.get('missing')} latest_day_failed={latest_assessment.get('failed')}",
                flush=True,
            )
            if command["collector_mode"] == "daily":
                print(
                    f"{now_utc()} HISTORY_CATCHUP_START command_id={command['id']} ranking_date={end_date} "
                    f"expected_symbols={latest_assessment.get('expected_symbols')} "
                    f"complete_symbols={latest_assessment.get('complete_symbols')} "
                    f"partial_symbols={latest_assessment.get('partial_symbols')} "
                    f"missing_symbols={latest_assessment.get('missing_symbols')} "
                    f"no_data_symbols={latest_assessment.get('no_data_symbols')} "
                    f"failed_symbols={latest_assessment.get('failed_symbols')}",
                    flush=True,
                )


def process_daily_top100_build(runtime_state: dict[str, Any], args: argparse.Namespace, now: datetime | None = None) -> None:
    if not bool(getattr(args, "enable_overnight_automation", True)):
        return
    proc = runtime_state.get("daily_top100_process")
    if proc is not None:
        rc = proc.poll()
        if rc is None:
            return
        command = runtime_state.get("daily_top100_running_command") or {}
        if rc == 0:
            print(
                f"{now_utc()} DAILY_TOP100_BUILD_DONE ranking_date={command.get('ranking_date')} "
                f"returncode={rc} latest_output={command.get('latest_output')}",
                flush=True,
            )
            runtime_state["top100_reload_requested"] = True
            runtime_state["top100_reload_path"] = command.get("latest_output")
            runtime_state["top100_reload_ranking_date"] = command.get("ranking_date")
            try:
                apply_top100_freshness_gate(runtime_state, args, date.fromisoformat(str(command.get("ranking_date"))))
            except Exception as exc:
                print(f"{now_utc()} DAILY_TOP100_FRESHNESS_CHECK_FAILED ranking_date={command.get('ranking_date')} error={exc!r}", flush=True)
        else:
            print(
                f"{now_utc()} DAILY_TOP100_BUILD_FAILED ranking_date={command.get('ranking_date')} "
                f"returncode={rc} latest_output={command.get('latest_output')}",
                flush=True,
            )
        runtime_state["daily_top100_process"] = None
        runtime_state["daily_top100_running_command"] = None

    now = now or datetime.now(timezone.utc)
    slot = str(getattr(args, "daily_top100_build_utc", "12:45"))
    if not is_utc_slot_due(now, slot):
        return

    ranking_date = latest_completed_trading_day(now, getattr(args, "market_close_utc", "20:00")).isoformat()
    key = f"{ranking_date}_{slot}"
    run_keys = _runtime_set(runtime_state, "daily_top100_build_run_keys")
    if key in run_keys:
        return
    if runtime_state.get("daily_top100_process") is not None:
        return
    min_pct = float(getattr(args, "startup_history_repair_min_completion_pct", 100.0))
    assessment = latest_history_is_complete(args=args, session_date=date.fromisoformat(ranking_date), min_completion_pct=min_pct, runtime_state=runtime_state)
    if not bool(assessment.get("complete")):
        last_key = f"{ranking_date}_{slot}_{assessment.get('completion_pct')}_{assessment.get('missing')}_{assessment.get('failed')}"
        wait_keys = _runtime_set(runtime_state, "daily_top100_build_wait_logged_keys")
        if last_key not in wait_keys:
            wait_keys.add(last_key)
            print(
                f"{now_utc()} DAILY_TOP100_BUILD_SKIPPED reason=latest_history_incomplete "
                f"ranking_date={ranking_date} completion_pct={assessment.get('completion_pct')} "
                f"expected_symbols={assessment.get('expected_symbols')} parquet_files={assessment.get('parquet_files')} "
                f"missing={assessment.get('missing')} failed={assessment.get('failed')} min_completion_pct={min_pct}",
                flush=True,
            )
            print(
                f"{now_utc()} DAILY_TOP100_BLOCKED_HISTORY_NOT_READY ranking_date={ranking_date} "
                f"expected_symbols={assessment.get('expected_symbols')} complete_symbols={assessment.get('complete_symbols')} "
                f"partial_symbols={assessment.get('partial_symbols')} missing_symbols={assessment.get('missing_symbols')} "
                f"no_data_symbols={assessment.get('no_data_symbols')} failed_symbols={assessment.get('failed_symbols')} "
                f"readiness_status={assessment.get('readiness_status')} latest_top100_file_age={assessment.get('latest_top100_file_age')}",
                flush=True,
            )
        return

    output_dir = Path(getattr(args, "daily_top100_output_dir", "data/universe"))
    dated_output = output_dir / f"daily_top100_{ranking_date}.csv"
    diagnostics_output = output_dir / f"daily_top100_{ranking_date}_diagnostics.csv"
    latest_output = Path(getattr(args, "daily_top100_latest_output", "data/universe/daily_top100_latest.csv"))
    command = [
        sys.executable,
        "-m",
        "src.live_trading.ranking.daily_top100_builder",
        "--date", ranking_date,
        "--universe", str(getattr(args, "daily_top100_universe", DEFAULT_UNIVERSE)),
        "--history-dir", str(getattr(args, "daily_top100_history_dir", DEFAULT_HISTORY_DIR)),
        "--output", str(dated_output),
        "--latest-output", str(latest_output),
        "--diagnostics-output", str(diagnostics_output),
        "--top-n", str(int(getattr(args, "daily_top100_top_n", 100))),
        "--disable-runtime-sqlite",
    ]
    sqlite_path = str(getattr(args, "daily_top100_sqlite_path", "data/runtime/rankings.sqlite"))
    if sqlite_path:
        command.extend(["--sqlite-path", sqlite_path])
    print(
        f"{now_utc()} DAILY_TOP100_BUILD_START ranking_date={ranking_date} "
        f"output={dated_output} latest_output={latest_output}",
        flush=True,
    )
    try:
        proc = subprocess.Popen(command)
    except Exception as exc:
        print(f"{now_utc()} DAILY_TOP100_BUILD_FAILED ranking_date={ranking_date} error={exc!r}", flush=True)
        run_keys.add(key)
        return
    runtime_state["daily_top100_process"] = proc
    runtime_state["daily_top100_running_command"] = {
        "ranking_date": ranking_date,
        "latest_output": str(latest_output),
        "command": command,
    }
    run_keys.add(key)


def process_overnight_automation(runtime_state: dict[str, Any], args: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc)
    enqueue_overnight_collector_if_due(runtime_state, args, now)
    process_daily_top100_build(runtime_state, args, now)


def reload_top100_universe_if_requested(
    ib: IB,
    recorder: LiveDataRecorder,
    states: dict[str, SymbolState],
    contracts: list[tuple[str, Any]],
    contract_by_symbol: dict[str, Any],
    tickers: dict[str, Any],
    latest_snapshots: dict[str, dict[str, Any]],
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not bool(runtime_state.get("top100_reload_requested", False)):
        return False

    reload_path = str(runtime_state.get("top100_reload_path") or getattr(args, "daily_top100_latest_output", args.alpha_rank_csv))
    ranking_date = runtime_state.get("top100_reload_ranking_date")
    print(f"{now_utc()} TOP100_RELOAD_START ranking_date={ranking_date} path={reload_path}", flush=True)
    runtime_state["entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = "top100_reload"

    try:
        entry_symbols, skipped_ineligible = load_tradeable_top_symbols(
            reload_path,
            int(args.top_n),
            min_price=args.min_price,
            symbol_denylist_path=getattr(args, "symbol_denylist", DEFAULT_SYMBOL_DENYLIST),
            runtime_ineligible_path=getattr(args, "runtime_ineligible_path", DEFAULT_RUNTIME_INELIGIBLE),
        )
        for symbol, info in skipped_ineligible.items():
            reason = str(info.get("reason") or "ineligible")
            mark_runtime_symbol_ineligible(
                runtime_state,
                symbol,
                reason=reason,
                source=str(info.get("source") or "top100_reload_filter"),
                persist=False,
            )
            print(f"{now_utc()} ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol={symbol} reason={reason} source=top100_reload", flush=True)
        if not entry_symbols:
            raise RuntimeError("top100_reload_no_tradeable_symbols")
        if len(entry_symbols) < int(args.top_n):
            print(
                f"{now_utc()} TOP100_RELOAD_WARNING reason=fewer_tradeable_symbols_after_ineligible_filter "
                f"rows={len(entry_symbols)} requested={int(args.top_n)} excluded_ineligible={len(skipped_ineligible)}",
                flush=True,
            )

        active_symbols = sorted(symbol for symbol, pos in managed_positions.items() if pos.active)
        max_subscriptions = max(0, int(getattr(args, "max_market_data_subscriptions", 100) or 0))
        active_symbol_set = set(active_symbols)
        selected_symbols: list[str] = []
        for symbol in active_symbols:
            if symbol not in selected_symbols:
                selected_symbols.append(symbol)
        top100_slots = max(0, max_subscriptions - len(selected_symbols)) if max_subscriptions > 0 else len(entry_symbols)
        subscribed_top100_candidates: list[str] = []
        skipped_symbols_due_to_cap: list[str] = []
        for symbol in entry_symbols:
            if symbol in selected_symbols:
                subscribed_top100_candidates.append(symbol)
                continue
            if max_subscriptions > 0 and len([s for s in selected_symbols if s not in active_symbol_set]) >= top100_slots:
                skipped_symbols_due_to_cap.append(symbol)
                continue
            selected_symbols.append(symbol)
            subscribed_top100_candidates.append(symbol)
        subscription_symbols = selected_symbols
        previous_symbols = [symbol for symbol, _ in contracts]
        previous_symbol_set = set(previous_symbols)
        subscription_symbol_set = set(subscription_symbols)
        skipped_due_to_subscription_cap = len(skipped_symbols_due_to_cap)

        for symbol in sorted(previous_symbol_set - subscription_symbol_set):
            ticker = tickers.pop(symbol, None)
            contract = getattr(ticker, "contract", None) or contract_by_symbol.get(symbol)
            if contract is not None:
                try:
                    ib.cancelMktData(contract)
                except Exception as exc:
                    print(f"{now_utc()} TOP100_RELOAD_CANCEL_MKTDATA_FAILED symbol={symbol} error={exc!r}", flush=True)
            latest_snapshots.pop(symbol, None)

        new_contracts: list[tuple[str, Any]] = []
        new_contract_by_symbol: dict[str, Any] = {}
        subscribed = 0
        reused = 0
        ibkr_error_101_count = 0
        failed_symbols: list[str] = []
        for symbol in subscription_symbols:
            contract = contract_by_symbol.get(symbol)
            if contract is None:
                contract = Stock(symbol, "SMART", "USD")
                try:
                    qualified = ib.qualifyContracts(contract)
                    if not qualified:
                        failed_symbols.append(symbol)
                        print(f"{now_utc()} TOP100_RELOAD_CONTRACT_FAILED symbol={symbol} reason=not_qualified", flush=True)
                        continue
                    contract = qualified[0]
                    record_contract_metadata(recorder, contract, source="top100_reload")
                    metadata_reason = None if symbol in active_symbol_set else contract_ineligible_reason(contract)
                    if metadata_reason:
                        failed_symbols.append(symbol)
                        mark_runtime_symbol_ineligible(
                            runtime_state,
                            symbol,
                            reason=metadata_reason,
                            source="contract_metadata",
                            con_id=getattr(contract, "conId", None),
                            raw_message=json.dumps(contract_metadata(contract), sort_keys=True),
                        )
                        print(
                            f"{now_utc()} ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol={symbol} "
                            f"reason={metadata_reason} source=contract_metadata",
                            flush=True,
                        )
                        continue
                except Exception as exc:
                    failed_symbols.append(symbol)
                    print(f"{now_utc()} TOP100_RELOAD_CONTRACT_FAILED symbol={symbol} error={exc!r}", flush=True)
                    continue

            states.setdefault(symbol, SymbolState(symbol=symbol))
            if symbol not in tickers:
                print(f"{now_utc()} TOP100_RELOAD_REQUESTED symbol={symbol} conId={getattr(contract, 'conId', '')}", flush=True)
                try:
                    tickers[symbol] = ib.reqMktData(contract, "", False, False)
                    subscribed += 1
                    print(f"{now_utc()} TOP100_RELOAD_SUBSCRIBED symbol={symbol} conId={getattr(contract, 'conId', '')}", flush=True)
                except Exception as exc:
                    failed_symbols.append(symbol)
                    error_text = repr(exc)
                    if "101" in error_text or "Max number of tickers" in error_text:
                        ibkr_error_101_count += 1
                        safe_sqlite_call(
                            getattr(recorder, "sqlite_store", None),
                            "record_runtime_event",
                            event_type="TOP100_RELOAD_SUBSCRIBE_ERROR",
                            severity="WARN",
                            strategy_name=STRATEGY_NAME,
                            session_date=getattr(recorder, "session_date", None),
                            symbol=symbol,
                            source="v67_live_runtime",
                            reason="ibkr_error_101",
                            raw_json={"error": error_text},
                        )
                    print(f"{now_utc()} TOP100_RELOAD_SUBSCRIBE_ERROR symbol={symbol} error={exc!r}", flush=True)
                    continue
            else:
                reused += 1
            new_contracts.append((symbol, contract))
            new_contract_by_symbol[symbol] = contract

        for symbol in list(states):
            if symbol not in subscription_symbol_set and symbol not in active_symbols:
                states.pop(symbol, None)

        contracts[:] = new_contracts
        contract_by_symbol.clear()
        contract_by_symbol.update(new_contract_by_symbol)
        runtime_state["entry_symbols"] = set(entry_symbols)
        runtime_state["top100_reload_requested"] = False
        runtime_state["top100_reload_done_at"] = now_utc()
        runtime_state["top100_reload_last_error"] = ""
        runtime_state["top100_reload_symbols"] = list(entry_symbols)
        runtime_state["top100_reload_diagnostics"] = {
            "max_subscriptions": max_subscriptions,
            "active_carried": len(active_symbols),
            "active_position_symbols_count": len(active_symbols),
            "top100_requested": len(entry_symbols),
            "subscribed_total": len(contracts),
            "subscribed_top100": len([s for s, _ in contracts if s in set(entry_symbols)]),
            "subscribed_active": len([s for s, _ in contracts if s in active_symbol_set]),
            "skipped_due_to_subscription_cap": skipped_due_to_subscription_cap,
            "skipped_symbols_due_to_cap": skipped_symbols_due_to_cap,
            "excluded_ineligible_count": len(skipped_ineligible),
            "excluded_ineligible_symbols": list(skipped_ineligible),
            "ibkr_error_101_count": ibkr_error_101_count,
        }

        traded_symbols_today = load_traded_symbols_today(recorder)
        if traded_symbols_today and args.max_one_trade_per_symbol_per_day:
            for symbol in traded_symbols_today:
                if symbol in states:
                    states[symbol].signal_sent = True
        rebuilt = rebuild_symbol_states_from_1m_candles(recorder, states, args)

        print(
            f"{now_utc()} TOP100_RELOAD_DONE ranking_date={ranking_date} entry_symbols={len(entry_symbols)} "
            f"subscriptions={len(contracts)} added={subscribed} reused={reused} removed={len(previous_symbol_set - subscription_symbol_set)} "
            f"max_subscriptions={max_subscriptions} active_carried={len(active_symbols)} "
            f"active_position_symbols_count={len(active_symbols)} top100_requested={len(entry_symbols)} "
            f"subscribed_total={len(contracts)} subscribed_top100={len([s for s, _ in contracts if s in set(entry_symbols)])} "
            f"subscribed_active={len([s for s, _ in contracts if s in active_symbol_set])} "
            f"skipped_due_to_subscription_cap={skipped_due_to_subscription_cap} "
            f"skipped_symbols_due_to_cap={','.join(skipped_symbols_due_to_cap[:20])} "
            f"excluded_ineligible_count={len(skipped_ineligible)} "
            f"excluded_ineligible_symbols={','.join(list(skipped_ineligible)[:20])} "
            f"ibkr_error_101_count={ibkr_error_101_count} failed={len(failed_symbols)} state_rebuilt={rebuilt}",
            flush=True,
        )
        return True
    except Exception as exc:
        runtime_state["top100_reload_requested"] = False
        runtime_state["top100_reload_last_error"] = repr(exc)
        print(f"{now_utc()} TOP100_RELOAD_FAILED ranking_date={ranking_date} path={reload_path} error={exc!r}", flush=True)
        return False
    finally:
        runtime_state["entries_blocked"] = False
        if runtime_state.get("entries_blocked_reason") == "top100_reload":
            runtime_state["entries_blocked_reason"] = ""


def enrich_lifecycle_with_fills(recorder: LiveDataRecorder) -> int:
    lifecycle_path = recorder.path("trade_lifecycle.csv")
    fills_path = recorder.path("fills.csv")
    if not lifecycle_path.exists() or not fills_path.exists():
        return 0

    try:
        with lifecycle_path.open("r", newline="", encoding="utf-8") as f:
            lifecycle_rows = list(csv.DictReader(f))
            lifecycle_fields = list(f.reader.fieldnames or [])
        with fills_path.open("r", newline="", encoding="utf-8") as f:
            fills_rows = list(csv.DictReader(f))
    except Exception:
        return 0

    if not lifecycle_rows or not fills_rows:
        return 0

    needed = ["fill_price", "fill_latency_ms"]
    for field in needed:
        if field not in lifecycle_fields:
            lifecycle_fields.append(field)

    fills_by_order = {}
    for fill in fills_rows:
        oid = str(fill.get("order_id", "")).strip()
        if oid:
            fills_by_order[oid] = fill

    updated = 0
    for row in lifecycle_rows:
        event = str(row.get("event", ""))
        if event not in {"BUY_ORDER_SENT", "SELL_ORDER_SENT"}:
            continue
        if str(row.get("fill_price", "")).strip():
            continue

        oid = str(row.get("order_id", "")).strip()
        fill = fills_by_order.get(oid)
        if not fill:
            continue

        fill_price = (
            fill.get("price")
            or fill.get("avg_price")
            or fill.get("avgPrice")
            or fill.get("fill_price")
        )
        if fill_price:
            row["fill_price"] = fill_price
            side = str(row.get("action") or "").upper()
            event_type = LifecycleEventType.ENTRY_ORDER_FILLED if side == "BUY" else LifecycleEventType.EXIT_ORDER_FILLED
            position_state = PositionState.OPEN if side == "BUY" else PositionState.CLOSED
            try:
                execution_id = str(fill.get("execution_id") or fill.get("execId") or f"{oid}-{row.get('symbol','')}-{fill_price}")
                execution = ExecutionRecord(
                    execution_id=execution_id,
                    client_order_id="",
                    symbol=str(row.get("symbol") or ""),
                    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                    quantity=abs(float(row.get("quantity") or fill.get("quantity") or 0)),
                    price=float(fill_price),
                    ib_order_id=oid,
                    commission=safe_float(fill.get("commission")),
                    raw_json={"source": "enrich_lifecycle_with_fills"},
                )
                store = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl"))
                store.append_execution_once(
                    execution,
                    LifecycleEvent(
                        event_type=event_type,
                        symbol=execution.symbol,
                        strategy=STRATEGY_NAME,
                        state_after=position_state,
                        ib_order_id=oid,
                        execution_id=execution.execution_id,
                        order_state=OrderState.FILLED,
                        position_state=position_state,
                        quantity=execution.quantity,
                        price=execution.price,
                        reason="ibkr_execution_recorded",
                        raw_json={"legacy_order_event": event},
                    ),
                )
                entry_verified = side == "BUY"
                close_verified = side != "BUY"
                record_formal_lifecycle(
                    recorder,
                    LifecycleEventType.POSITION_OPENED if side == "BUY" else LifecycleEventType.POSITION_CLOSED,
                    execution.symbol,
                    position_state=position_state,
                    state_after=position_state,
                    order_id=oid,
                    execution_id=execution.execution_id,
                    quantity=execution.quantity,
                    price=execution.price,
                    reason="fill_enriched_lifecycle",
                    raw_json={
                        "entry_fill_verified": entry_verified,
                        "close_fill_verified": close_verified,
                        "source": "enrich_lifecycle_with_fills",
                    },
                )
            except Exception as exc:
                print(f"{now_utc()} formal_fill_record_error order_id={oid} error={exc!r}", flush=True)

            try:
                fill_px = float(fill_price)
                qty = abs(float(row.get("quantity") or 0))
                decision_mid = row.get("decision_mid")

                # IBKR-style rough estimate
                est_commission = max(0.35, qty * 0.0035)
                row["estimated_commission"] = round(est_commission, 4)

                if decision_mid not in (None, "", "None"):
                    decision_mid = float(decision_mid)
                    if decision_mid > 0:
                        side = str(row.get("action") or "").upper()

                        if side == "BUY":
                            slippage_bps = ((fill_px - decision_mid) / decision_mid) * 10000.0
                        else:
                            slippage_bps = ((decision_mid - fill_px) / decision_mid) * 10000.0

                        row["realized_slippage_bps"] = round(slippage_bps, 2)
            except Exception:
                pass

        try:
            order_ts = datetime.fromisoformat(str(row.get("recorded_at")).replace("Z", "+00:00"))
            fill_ts_raw = fill.get("recorded_at") or fill.get("time") or fill.get("execution_time")
            fill_ts = datetime.fromisoformat(str(fill_ts_raw).replace("Z", "+00:00"))
            row["fill_latency_ms"] = int((fill_ts - order_ts).total_seconds() * 1000)
        except Exception:
            row["fill_latency_ms"] = ""

        updated += 1

    if updated:
        with lifecycle_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=lifecycle_fields)
            writer.writeheader()
            writer.writerows(lifecycle_rows)

    return updated


def load_traded_symbols_today(recorder: LiveDataRecorder) -> set[str]:
    path = recorder.path("trade_lifecycle.csv")
    symbols: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return symbols
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("event", "")).strip() in {"BUY_ORDER_SENT", "SIGNAL_READY"}:
                    sym = str(row.get("symbol", "")).upper().strip()
                    if sym:
                        symbols.add(sym)
    except Exception:
        return symbols
    return symbols


def count_entry_orders_today(recorder: LiveDataRecorder) -> int:
    path = recorder.path("trade_lifecycle.csv")
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("event", "")).strip() == "BUY_ORDER_SENT":
                    count += 1
    except Exception:
        return 0
    return count


def latest_strategy_equity_metrics(recorder: LiveDataRecorder) -> dict[str, float]:
    path = recorder.path("strategy_equity.csv")
    if not path.exists() or path.stat().st_size == 0:
        return {"unrealized_pnl": 0.0, "gross_exposure": 0.0}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {"unrealized_pnl": 0.0, "gross_exposure": 0.0}
    row = rows[-1] if rows else {}
    return {
        "unrealized_pnl": safe_float(row.get("unrealized_pnl")) or 0.0,
        "gross_exposure": safe_float(row.get("gross_exposure")) or 0.0,
        "active_positions": safe_float(row.get("active_positions")) or 0.0,
    }


def latest_portfolio_pnl_metrics(recorder: LiveDataRecorder) -> dict[str, float]:
    path = recorder.path("portfolio_snapshots.csv")
    if not path.exists() or path.stat().st_size == 0:
        return {"realized_pnl": 0.0, "unrealized_pnl": 0.0}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {"realized_pnl": 0.0, "unrealized_pnl": 0.0}
    row = rows[-1] if rows else {}
    return {
        "realized_pnl": safe_float(row.get("realized_pnl")) or 0.0,
        "unrealized_pnl": safe_float(row.get("unrealized_pnl")) or 0.0,
    }


def active_position_count(positions: dict[str, ManagedPosition]) -> int:
    return sum(1 for pos in positions.values() if bool(pos.active) and bool(pos.entry_fill_verified or pos.exit_sent))


def managed_gross_exposure(positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]]) -> float:
    gross = 0.0
    for symbol, pos in positions.items():
        if not bool(pos.active) or not bool(pos.entry_fill_verified or pos.exit_sent):
            continue
        price = safe_float((latest_snapshots.get(symbol) or {}).get("price")) or safe_float(getattr(pos, "entry_price", None))
        qty = safe_float(getattr(pos, "quantity", None)) or 0.0
        if price and qty:
            gross += abs(price * qty)
    return gross


def evaluate_risk_guard(
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    symbol: str = "",
    candidate_notional: float = 0.0,
) -> dict[str, Any]:
    enabled = bool(getattr(args, "risk_guard_enabled", True))
    equity = latest_strategy_equity_metrics(recorder)
    portfolio = latest_portfolio_pnl_metrics(recorder)
    active_positions = active_position_count(managed_positions)
    gross_exposure = managed_gross_exposure(managed_positions, latest_snapshots) or safe_float(equity.get("gross_exposure")) or 0.0
    trades_today = count_entry_orders_today(recorder)
    realized_pnl = safe_float(portfolio.get("realized_pnl")) or 0.0
    unrealized_pnl = safe_float(portfolio.get("unrealized_pnl")) or safe_float(equity.get("unrealized_pnl")) or 0.0
    total_pnl = realized_pnl + unrealized_pnl
    effective_single_cap = float(getattr(args, "max_single_position_usd", 0.0) or 0.0)
    if effective_single_cap <= 0:
        effective_single_cap = float(getattr(args, "position_usd", 0.0) or 0.0)

    metrics: dict[str, Any] = {
        "enabled": enabled,
        "blocked": False,
        "reason": "",
        "symbol": symbol,
        "candidate_notional": round(float(candidate_notional or 0.0), 4),
        "daily_pnl": round(total_pnl, 4),
        "realized_pnl": round(realized_pnl, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "trades_today": trades_today,
        "active_positions": active_positions,
        "gross_exposure": round(gross_exposure, 4),
        "max_daily_loss_usd": float(getattr(args, "max_daily_loss_usd", 0.0) or 0.0),
        "max_trades_per_day": int(getattr(args, "max_trades_per_day", 0) or 0),
        "max_open_positions": int(getattr(args, "max_open_positions", 0) or 0),
        "max_gross_exposure_usd": float(getattr(args, "max_gross_exposure_usd", 0.0) or 0.0),
        "max_single_position_usd": effective_single_cap,
    }
    if not enabled:
        return metrics

    checks = [
        ("max_daily_loss", metrics["max_daily_loss_usd"] > 0 and total_pnl <= -metrics["max_daily_loss_usd"]),
        ("max_trades_per_day", metrics["max_trades_per_day"] > 0 and trades_today >= metrics["max_trades_per_day"]),
        ("max_open_positions", metrics["max_open_positions"] > 0 and active_positions >= metrics["max_open_positions"]),
        ("max_gross_exposure", metrics["max_gross_exposure_usd"] > 0 and gross_exposure >= metrics["max_gross_exposure_usd"]),
        ("max_single_position", effective_single_cap > 0 and candidate_notional > effective_single_cap),
    ]
    for reason, blocked in checks:
        if blocked:
            metrics["blocked"] = True
            metrics["reason"] = reason
            break
    return metrics


def _read_lifecycle_rows(recorder: LiveDataRecorder) -> list[dict[str, Any]]:
    path = recorder.path("trade_lifecycle.csv")
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _read_fill_rows(recorder: LiveDataRecorder) -> list[dict[str, Any]]:
    path = recorder.path("fills.csv")
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def exit_fill_quantity_for_symbol(recorder: LiveDataRecorder, symbol: str) -> float:
    symbol = str(symbol or "").upper().strip()
    total = 0.0
    for row in _read_fill_rows(recorder):
        if str(row.get("symbol") or "").upper().strip() != symbol:
            continue
        action = str(row.get("action") or "").upper().strip()
        if action not in {"SLD", "SELL"}:
            continue
        total += abs(safe_float(row.get("quantity")) or 0.0)
    return total


def entry_fill_quantity_for_symbol(recorder: LiveDataRecorder, symbol: str) -> float:
    symbol = str(symbol or "").upper().strip()
    total = 0.0
    for row in _read_fill_rows(recorder):
        if str(row.get("symbol") or "").upper().strip() != symbol:
            continue
        action = str(row.get("action") or "").upper().strip()
        if action not in {"BOT", "BUY"}:
            continue
        total += abs(safe_float(row.get("quantity")) or 0.0)
    return total


def entry_fill_verified(recorder: LiveDataRecorder, symbol: str, managed_quantity: Any) -> bool:
    required = abs(safe_float(managed_quantity) or 0.0)
    if required <= 0:
        return False
    return entry_fill_quantity_for_symbol(recorder, symbol) + 1e-9 >= required


def sync_managed_entry_fill_verification(
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
) -> int:
    updated = 0
    for symbol, pos in managed_positions.items():
        if not pos.active or pos.entry_fill_verified:
            continue
        if not entry_fill_verified(recorder, symbol, pos.quantity):
            continue
        pos.entry_fill_verified = True
        record_lifecycle_with_formal(
            recorder,
            "POSITION_OPENED",
            symbol,
            action="VERIFY",
            quantity=pos.quantity,
            price=pos.entry_price,
            entry_price=pos.entry_price,
            peak_price=pos.peak_price,
            reason="entry_fill_verified_from_ibkr_execution",
            entry_fill_verified="true",
            raw_json={"entry_fill_verified": True, "source": "fills_csv"},
        )
        print(f"{now_utc()} ENTRY_FILL_VERIFIED symbol={symbol} quantity={pos.quantity}", flush=True)
        updated += 1
    return updated


def backfill_recent_fills_for_verification(ib: IB, recorder: LiveDataRecorder, runtime_state: dict[str, Any]) -> int:
    try:
        return int(record_recent_fills(ib, recorder, set()) or 0)
    except Exception as exc:
        runtime_state["reconciliation_fill_backfill_error"] = repr(exc)
        print(f"{now_utc()} RECONCILIATION_FILL_BACKFILL_FAILED error={exc!r}", flush=True)
        return 0


def close_fill_verified(recorder: LiveDataRecorder, symbol: str, managed_quantity: Any) -> bool:
    required = abs(safe_float(managed_quantity) or 0.0)
    if required <= 0:
        return False
    return exit_fill_quantity_for_symbol(recorder, symbol) + 1e-9 >= required


def record_entry_not_filled(
    recorder: LiveDataRecorder,
    symbol: str,
    *,
    quantity: Any,
    reason: str,
    ibkr_quantity: float,
    runtime_state: dict[str, Any],
    raw_json: dict[str, Any] | None = None,
) -> None:
    payload = {
        "managed_quantity": quantity,
        "ibkr_quantity": ibkr_quantity,
        "entry_fill_verified": False,
        "close_fill_verified": False,
        **(raw_json or {}),
    }
    record_lifecycle_with_formal(
        recorder,
        "ENTRY_NOT_FILLED",
        symbol,
        action="VERIFY",
        quantity=quantity,
        reason=reason,
        entry_fill_verified="false",
        close_fill_verified="false",
        raw_json=payload,
    )
    runtime_state["entry_not_filled_count"] = int(runtime_state.get("entry_not_filled_count") or 0) + 1
    print(
        f"{now_utc()} ENTRY_NOT_FILLED symbol={symbol} managed_quantity={quantity} "
        f"ibkr_quantity={ibkr_quantity} reason={reason}",
        flush=True,
    )


def ibkr_entry_reject_reason(error_code: Any, message: Any = "") -> str:
    try:
        code = int(error_code)
    except Exception:
        code = 0
    msg = str(message or "").lower()
    if code == 201 and ("no trading permission" in msg or "kid" in msg or "customer ineligible" in msg):
        return "no_trading_permission_kid"
    if code == 201:
        return "order_rejected_201"
    return ""


def _runtime_dict(runtime_state: dict[str, Any], key: str) -> dict[Any, Any]:
    value = runtime_state.setdefault(key, {})
    if not isinstance(value, dict):
        value = {}
        runtime_state[key] = value
    return value


def _runtime_order_id(value: Any) -> str:
    try:
        if value in (None, ""):
            return ""
        return str(int(float(value)))
    except Exception:
        return str(value or "").strip()


def _runtime_symbol_from_contract(contract: Any) -> str:
    return str(getattr(contract, "symbol", "") or "").upper().strip()


def record_entry_order_rejected(
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
    *,
    symbol: str,
    order_id: Any,
    quantity: Any = None,
    price: Any = None,
    reason: str,
    ibkr_error_code: Any = None,
    message: str = "",
) -> bool:
    symbol = str(symbol or "").upper().strip()
    order_key = _runtime_order_id(order_id)
    if not symbol:
        return False
    processed_key = f"{order_key}:{symbol}:{ibkr_error_code}:{reason}"
    processed = _runtime_set(runtime_state, "entry_rejection_processed")
    if processed_key in processed:
        return False
    processed.add(processed_key)

    order_meta = _runtime_dict(runtime_state, "entry_order_by_order_id").get(order_key, {})
    qty = quantity if quantity not in (None, "") else order_meta.get("quantity")
    entry_price = price if price not in (None, "") else order_meta.get("price")
    pos = managed_positions.get(symbol)
    if pos is not None:
        qty = qty if qty not in (None, "") else pos.quantity
        entry_price = entry_price if entry_price not in (None, "") else pos.entry_price
        pos.active = False
        pos.entry_fill_verified = False
        safe_sqlite_call(
            getattr(recorder, "sqlite_store", None),
            "upsert_position",
            {
                "strategy_name": STRATEGY_NAME,
                "session_date": getattr(recorder, "session_date", None),
                "symbol": symbol,
                "status": "ENTRY_REJECTED",
                "quantity": pos.quantity,
                "avg_price": pos.entry_price,
                "source": pos.source,
                "active": 0,
                "exit_sent": 0,
                "updated_at": now_utc(),
                "raw_json": {
                    **managed_position_payload(pos),
                    "entry_fill_verified": False,
                    "ibkr_entry_confirmed": False,
                    "reject_reason": reason,
                    "ibkr_error_code": ibkr_error_code,
                    "ibkr_error_message": message,
                    "order_id": order_key,
                },
            },
        )
        managed_positions.pop(symbol, None)
        persist_managed_positions(recorder, managed_positions)

    mark_runtime_symbol_ineligible(
        runtime_state,
        symbol,
        reason=reason,
        source="ibkr_error_201",
        con_id=order_meta.get("conId") or order_meta.get("con_id"),
        ibkr_error_code=ibkr_error_code,
        raw_message=str(message or ""),
    )
    rejected_entries = _runtime_dict(runtime_state, "rejected_entries")
    rejected_entries[symbol] = {
        "symbol": symbol,
        "quantity": qty,
        "price": entry_price,
        "order_id": order_key,
        "reason": reason,
        "ibkr_error_code": ibkr_error_code,
        "message": message,
        "time": now_utc(),
    }
    runtime_state["entry_rejected_count"] = int(runtime_state.get("entry_rejected_count") or 0) + 1
    record_lifecycle_with_formal(
        recorder,
        "ENTRY_ORDER_REJECTED",
        symbol,
        action="BUY",
        quantity=qty,
        price=entry_price,
        order_id=order_key,
        reason=reason,
        entry_fill_verified="false",
        close_fill_verified="false",
        raw_json={
            "status": "ENTRY_REJECTED",
            "active": False,
            "entry_fill_verified": False,
            "ibkr_entry_confirmed": False,
            "reject_reason": reason,
            "ibkr_error_code": ibkr_error_code,
            "ibkr_error_message": message,
            "order_id": order_key,
        },
    )
    print(
        f"{now_utc()} ENTRY_ORDER_REJECTED symbol={symbol} order_id={order_key} "
        f"reason={reason} ibkr_error_code={ibkr_error_code}",
        flush=True,
    )
    return True


def handle_ibkr_order_rejection_event(
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
    *,
    order_id: Any,
    error_code: Any,
    message: Any = "",
    contract: Any = None,
    status: str = "",
) -> bool:
    reason = ibkr_entry_reject_reason(error_code, message)
    if not reason:
        return False
    order_key = _runtime_order_id(order_id)
    order_meta = _runtime_dict(runtime_state, "entry_order_by_order_id").get(order_key, {})
    symbol = str(order_meta.get("symbol") or _runtime_symbol_from_contract(contract)).upper().strip()
    if not symbol:
        return False
    return record_entry_order_rejected(
        recorder,
        managed_positions,
        runtime_state,
        symbol=symbol,
        order_id=order_key,
        quantity=order_meta.get("quantity"),
        price=order_meta.get("price"),
        reason=reason,
        ibkr_error_code=error_code,
        message=str(message or status or ""),
    )


def install_ibkr_order_rejection_handler(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
) -> None:
    if getattr(ib, "_v67_order_rejection_handler_installed", False):
        return

    def _on_error(req_id: Any, error_code: Any, error_string: Any, contract: Any = None, *args: Any) -> None:
        try:
            handled = handle_ibkr_order_rejection_event(
                recorder,
                managed_positions,
                runtime_state,
                order_id=req_id,
                error_code=error_code,
                message=error_string,
                contract=contract,
            )
            if handled:
                runtime_state["ibkr_error_201_count"] = int(runtime_state.get("ibkr_error_201_count") or 0) + 1
        except Exception as exc:
            print(f"{now_utc()} ENTRY_ORDER_REJECT_HANDLER_FAILED source=errorEvent error={exc!r}", flush=True)

    def _on_order_status(*event_args: Any) -> None:
        try:
            trade = event_args[0] if event_args else None
            status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
            if status not in {"Cancelled", "Inactive", "ApiCancelled"}:
                return
            order = getattr(trade, "order", None)
            order_id = getattr(order, "orderId", "")
            log_rows = list(getattr(trade, "log", None) or [])
            message = " ".join(str(getattr(row, "message", "") or "") for row in log_rows)
            error_code = ""
            for row in reversed(log_rows):
                value = getattr(row, "errorCode", None)
                if value not in (None, ""):
                    error_code = value
                    break
            if not error_code and "201" in message:
                error_code = 201
            handle_ibkr_order_rejection_event(
                recorder,
                managed_positions,
                runtime_state,
                order_id=order_id,
                error_code=error_code,
                message=message,
                contract=getattr(trade, "contract", None),
                status=status,
            )
        except Exception as exc:
            print(f"{now_utc()} ENTRY_ORDER_REJECT_HANDLER_FAILED source=orderStatusEvent error={exc!r}", flush=True)

    try:
        if hasattr(ib, "errorEvent"):
            ib.errorEvent += _on_error
        if hasattr(ib, "orderStatusEvent"):
            ib.orderStatusEvent += _on_order_status
        setattr(ib, "_v67_order_rejection_handler_installed", True)
    except Exception as exc:
        print(f"{now_utc()} ENTRY_ORDER_REJECT_HANDLER_INSTALL_FAILED error={exc!r}", flush=True)


def record_unverified_reconciliation_close(
    recorder: LiveDataRecorder,
    symbol: str,
    *,
    quantity: Any,
    reason: str,
    ibkr_quantity: float,
    runtime_state: dict[str, Any],
    close_source: str = "reconciliation",
    raw_json: dict[str, Any] | None = None,
) -> None:
    payload = {
        "managed_quantity": quantity,
        "ibkr_quantity": ibkr_quantity,
        "close_source": close_source,
        "fill_verified": False,
        "entry_fill_verified": True,
        "close_fill_verified": False,
        **(raw_json or {}),
    }
    record_lifecycle_with_formal(
        recorder,
        "RECONCILIATION_CLOSE_WITHOUT_FILL",
        symbol,
        action="VERIFY",
        quantity=quantity,
        reason=reason,
        close_source=close_source,
        fill_verified="false",
        entry_fill_verified="true",
        close_fill_verified="false",
        raw_json=payload,
    )
    record_lifecycle_with_formal(
        recorder,
        "POSITION_CLOSED_UNVERIFIED",
        symbol,
        action="VERIFY",
        quantity=quantity,
        reason=reason,
        close_source=close_source,
        fill_verified="false",
        entry_fill_verified="true",
        close_fill_verified="false",
        raw_json=payload,
    )
    runtime_state["reconciliation_close_without_fill_count"] = int(runtime_state.get("reconciliation_close_without_fill_count") or 0) + 1
    print(
        f"{now_utc()} RECONCILIATION_CLOSE_WITHOUT_FILL symbol={symbol} "
        f"managed_quantity={quantity} ibkr_quantity={ibkr_quantity} reason={reason}",
        flush=True,
    )


def process_fill_lifecycle_diagnostics(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
) -> int:
    fills = _read_fill_rows(recorder)
    if not fills:
        return 0
    lifecycle_rows = _read_lifecycle_rows(recorder)
    orders: dict[str, dict[str, Any]] = {}
    cancelled_order_ids: set[str] = set()
    for row in lifecycle_rows:
        order_id = str(row.get("order_id") or "").strip()
        if not order_id:
            continue
        event = str(row.get("event") or "").strip()
        if event in {"BUY_ORDER_SENT", "SELL_ORDER_SENT", "EOD_FLATTEN_SUBMIT", "MANUAL_FLATTEN_SENT"}:
            orders.setdefault(order_id, row)
        if event in {"ENTRY_ORDER_CANCELLED", "EXIT_ORDER_CANCELLED", "ORDER_CANCEL_CONFIRMED", "ORDER_STALE"}:
            cancelled_order_ids.add(order_id)

    processed = runtime_state.setdefault("fill_diagnostic_execution_ids", set())
    if not isinstance(processed, set):
        processed = set(processed or [])
        runtime_state["fill_diagnostic_execution_ids"] = processed
    portfolio_qty = ibkr_position_quantities(ib)
    emitted = 0
    for fill in fills:
        execution_id = str(fill.get("execution_id") or "").strip()
        if not execution_id or execution_id in processed:
            continue
        processed.add(execution_id)
        order_id = str(fill.get("order_id") or "").strip()
        symbol = str(fill.get("symbol") or "").upper().strip()
        action = str(fill.get("action") or "").upper().strip()
        fill_qty = abs(safe_float(fill.get("quantity")) or 0.0)
        order_row = orders.get(order_id) or {}
        order_qty = abs(safe_float(order_row.get("quantity")) or 0.0)
        fill_price = safe_float(fill.get("fill_price"))

        if order_id and order_id in cancelled_order_ids:
            record_lifecycle(
                recorder,
                "DELAYED_FILL_AFTER_CANCEL",
                symbol,
                action=action,
                quantity=fill_qty,
                price=fill_price,
                order_id=order_id,
                execution_id=execution_id,
                reason="fill_arrived_after_cancelled_order",
                raw_json={"fill": fill, "order": order_row},
            )
            record_formal_lifecycle(
                recorder,
                LifecycleEventType.EXECUTION_RECORDED,
                symbol,
                order_id=order_id,
                execution_id=execution_id,
                quantity=fill_qty,
                price=fill_price,
                reason="delayed_fill_after_cancel",
                raw_json={"fill": fill, "order": order_row},
            )
            remaining = portfolio_qty.get(symbol, 0.0)
            if abs(remaining) > 0:
                record_lifecycle(
                    recorder,
                    "ORDER_CANCEL_BUT_POSITION_EXISTS",
                    symbol,
                    action=action,
                    quantity=abs(remaining),
                    order_id=order_id,
                    execution_id=execution_id,
                    reason="ibkr_position_exists_after_cancelled_order_fill",
                    raw_json={"ibkr_quantity": remaining},
                )
                runtime_state["cancel_but_position_exists_count"] = int(runtime_state.get("cancel_but_position_exists_count") or 0) + 1
            runtime_state["delayed_fill_after_cancel_count"] = int(runtime_state.get("delayed_fill_after_cancel_count") or 0) + 1
            try:
                runtime_state["delayed_fill_after_cancel_verification"] = verify_managed_positions_against_ibkr(
                    ib,
                    recorder,
                    managed_positions,
                    reason="delayed_fill_after_cancel",
                    runtime_state=runtime_state,
                )
            except Exception as exc:
                runtime_state["delayed_fill_after_cancel_verification_error"] = repr(exc)
            print(f"{now_utc()} DELAYED_FILL_AFTER_CANCEL symbol={symbol} order_id={order_id} execution_id={execution_id} quantity={fill_qty}", flush=True)
            emitted += 1

        if order_qty > 0 and fill_qty > 0 and fill_qty + 1e-9 < order_qty:
            partial_event = "ENTRY_ORDER_PARTIAL" if action in {"BOT", "BUY"} else "EXIT_ORDER_PARTIAL"
            record_lifecycle(
                recorder,
                partial_event,
                symbol,
                action=action,
                quantity=fill_qty,
                price=fill_price,
                order_id=order_id,
                execution_id=execution_id,
                reason="ibkr_partial_fill",
                raw_json={"order_quantity": order_qty, "fill": fill, "order": order_row},
            )
            record_formal_lifecycle(
                recorder,
                LifecycleEventType.ENTRY_ORDER_PARTIAL if partial_event == "ENTRY_ORDER_PARTIAL" else LifecycleEventType.EXIT_ORDER_PARTIAL,
                symbol,
                order_state=OrderState.PARTIAL,
                position_state=PositionState.ENTRY_PENDING if partial_event == "ENTRY_ORDER_PARTIAL" else PositionState.EXIT_PENDING,
                order_id=order_id,
                execution_id=execution_id,
                quantity=fill_qty,
                price=fill_price,
                reason="ibkr_partial_fill",
                raw_json={"order_quantity": order_qty},
            )
            key = "partial_entry_count" if partial_event == "ENTRY_ORDER_PARTIAL" else "partial_exit_count"
            runtime_state[key] = int(runtime_state.get(key) or 0) + 1
            partial_states = runtime_state.setdefault("partial_fill_states", {})
            if isinstance(partial_states, dict):
                partial_states[str(order_id or execution_id)] = {
                    "state": "ENTRY_PARTIAL" if partial_event == "ENTRY_ORDER_PARTIAL" else "EXIT_PARTIAL",
                    "event": partial_event,
                    "symbol": symbol,
                    "order_id": order_id,
                    "execution_id": execution_id,
                    "fill_quantity": fill_qty,
                    "order_quantity": order_qty,
                }
            if partial_event == "EXIT_ORDER_PARTIAL" and symbol in managed_positions:
                remaining = portfolio_qty.get(symbol, 0.0)
                pos = managed_positions[symbol]
                if abs(remaining) > 0:
                    pos.quantity = int(round(abs(remaining)))
                    pos.active = True
                else:
                    pos.active = False
            print(f"{now_utc()} {partial_event} symbol={symbol} order_id={order_id} execution_id={execution_id} fill_quantity={fill_qty} order_quantity={order_qty}", flush=True)
            emitted += 1
    return emitted


def _parse_bar_time_utc(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None

    raw = raw.replace("US/Eastern", "").replace("America/New_York", "").strip()

    for fmt in (
        "%Y%m%d  %H:%M:%S",
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def rebuild_symbol_states_from_1m_candles(
    recorder: LiveDataRecorder,
    states: dict[str, SymbolState],
    args: argparse.Namespace,
) -> int:
    path = recorder.path("candles_1m.csv")
    if not path.exists() or path.stat().st_size == 0:
        print(f"{now_utc()} state_rebuild_skipped reason=no_candles_1m", flush=True)
        return 0

    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        now = datetime.now(timezone.utc)
        market_open = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception as exc:
        print(f"{now_utc()} state_rebuild_skipped reason=bad_market_open error={exc!r}", flush=True)
        return 0

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                symbol = str(row.get("symbol", "")).upper().strip()
                if symbol not in states:
                    continue
                ts = _parse_bar_time_utc(row.get("bar_time", ""))
                if ts is None or ts < market_open:
                    continue
                by_symbol.setdefault(symbol, []).append(row)
    except Exception as exc:
        print(f"{now_utc()} state_rebuild_error error={exc!r}", flush=True)
        return 0

    rebuilt = 0
    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: str(r.get("bar_time", "")))
        if not rows:
            continue

        first = safe_float(rows[0].get("open")) or safe_float(rows[0].get("close"))
        if first is None or first <= 0:
            continue

        st = states[symbol]
        st.signal_source = "reconstructed"
        st.last_update_source = "reconstructed"
        st.last_live_update_ts = None
        st.last_live_update_utc = None
        st.ready_since_ts = None
        st.ready_since_utc = None
        st.stale_ready_logged = False
        st.first_seen_ts = time.time()
        first_ts = _parse_bar_time_utc(rows[0].get("bar_time", ""))
        st.first_seen_utc = first_ts.isoformat() if first_ts is not None else None
        st.latest_seen_utc = st.first_seen_utc
        st.first_price = first
        st.open_price = first
        st.high = first
        st.low = first
        st.first_5m_high = None
        st.first_15m_high = None
        st.or_high = None
        st.or_low = None
        st.bars = []

        for row in rows:
            ts = _parse_bar_time_utc(row.get("bar_time", ""))
            if ts is None:
                continue

            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            close = safe_float(row.get("close"))
            vol = safe_float(row.get("volume"))

            if high is None or low is None:
                continue

            minutes = (ts - market_open).total_seconds() / 60.0

            st.high = max(st.high or high, high)
            st.low = min(st.low or low, low)
            st.last_price = close or st.last_price
            st.latest_volume = vol or st.latest_volume
            st.latest_seen_utc = ts.isoformat()
            st.bars.append(
                {
                    "bar_time_utc": ts.isoformat(),
                    "open": safe_float(row.get("open")),
                    "high": high,
                    "low": low,
                    "close": close,
                    "session_elapsed_seconds": round(minutes * 60.0, 3),
                    "source": "candles_1m_rebuild",
                }
            )

            if 0 <= minutes < 5:
                st.first_5m_high = max(st.first_5m_high or high, high)
            if 0 <= minutes < 15:
                st.first_15m_high = max(st.first_15m_high or high, high)
            if 0 <= minutes < (args.opening_range_seconds / 60.0):
                st.or_high = max(st.or_high or high, high)
                st.or_low = min(st.or_low or low, low)

        rebuilt += 1

    print(f"{now_utc()} state_rebuild_done symbols={rebuilt} source=candles_1m market_open_utc={args.market_open_utc}", flush=True)
    return rebuilt


def connect_ibkr_with_retry(ib: IB, args: argparse.Namespace) -> None:
    attempts = 0
    while True:
        attempts += 1
        try:
            if ib.isConnected():
                return
            print(f"{now_utc()} RECONNECT_START attempt={attempts}", flush=True)
            ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
            ib.reqMarketDataType(args.market_data_type)
            print(f"{now_utc()} RECONNECT_DONE attempt={attempts}", flush=True)
            return
        except Exception as exc:
            print(f"{now_utc()} RECONNECT_FAILED attempt={attempts} error={exc!r}", flush=True)
            try:
                ib.disconnect()
            except Exception:
                pass
            if attempts >= getattr(args, "reconnect_max_attempts", 999999):
                raise
            time.sleep(getattr(args, "reconnect_wait_seconds", 15.0))


def resubscribe_market_data(ib: IB, contracts: list[tuple[str, Any]]) -> dict[str, Any]:
    tickers: dict[str, Any] = {}
    for symbol, contract in contracts:
        tickers[symbol] = ib.reqMktData(contract, "", False, False)
        print(f"RESUBSCRIBED {symbol} conId={getattr(contract, 'conId', '')}", flush=True)
    print(f"{now_utc()} RESUBSCRIBE_DONE symbols={len(tickers)}", flush=True)
    return tickers


def ibkr_connection_alive(ib: IB) -> bool:
    try:
        return bool(ib.isConnected())
    except Exception:
        return False


def handle_ibkr_disconnect_and_recover(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    contract_by_symbol: dict[str, Any],
    contracts: list[tuple[str, Any]],
    runtime_state: dict[str, Any],
    args: argparse.Namespace,
    *,
    reason: str,
    seen_fills: set[str] | None = None,
    connect_fn: Any | None = None,
    resubscribe_fn: Any | None = None,
    reconcile_fn: Any | None = None,
    record_account_snapshot_fn: Any | None = None,
    record_recent_fills_fn: Any | None = None,
) -> dict[str, Any]:
    print(f"{now_utc()} RECONNECT_SUPERVISOR_START reason={reason}", flush=True)
    runtime_state["ibkr_connected"] = False
    runtime_state["reconnect_active"] = True
    runtime_state["post_reconnect_reconciliation_done"] = False
    runtime_state["entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = "ibkr_reconnect"
    source = "exception" if "exception" in str(reason).lower() else "api_disconnect"
    log_ibkr_disconnect_source(runtime_state, source=source, reason=reason)

    if ibkr_connection_alive(ib):
        try:
            ib.disconnect()
        except Exception:
            pass
    print(f"{now_utc()} RECONNECT_SUPERVISOR_DISCONNECTED reason={reason}", flush=True)

    connect_impl = connect_fn or connect_ibkr_with_retry
    resubscribe_impl = resubscribe_fn or resubscribe_market_data
    reconcile_impl = reconcile_fn or startup_reconcile_runtime_state
    account_snapshot_impl = record_account_snapshot_fn or record_account_snapshot
    recent_fills_impl = record_recent_fills_fn or record_recent_fills

    attempt = int(runtime_state.get("reconnect_attempts") or 0) + 1
    runtime_state["reconnect_attempts"] = attempt
    print(f"{now_utc()} RECONNECT_SUPERVISOR_ATTEMPT attempt={attempt} reason={reason}", flush=True)

    try:
        connect_impl(ib, args)
        install_commission_report_handler(ib, recorder)
        runtime_state["ibkr_connected"] = True
        runtime_state["reconnect_last_error"] = ""
        runtime_state["reconnect_last_success_at"] = now_utc()
        print(f"{now_utc()} RECONNECT_SUPERVISOR_CONNECTED attempt={attempt}", flush=True)

        tickers = resubscribe_impl(ib, contracts)
        print(f"{now_utc()} RECONNECT_SUPERVISOR_RESUBSCRIBED symbols={len(tickers)}", flush=True)

        try:
            account_snapshot_impl(ib, recorder)
        except Exception as exc:
            print(f"{now_utc()} RECONNECT_SUPERVISOR_ACCOUNT_RECORD_FAILED error={exc!r}", flush=True)
        if seen_fills is not None:
            try:
                recent_fills_impl(ib, recorder, seen_fills)
            except Exception as exc:
                print(f"{now_utc()} RECONNECT_SUPERVISOR_FILLS_RECORD_FAILED error={exc!r}", flush=True)

        reconciliation = reconcile_impl(
            ib,
            recorder,
            managed_positions,
            contract_by_symbol,
            runtime_state,
            log_prefix="POST_RECONNECT_RECONCILIATION",
            reason_prefix="post_reconnect_reconciliation",
        )
        runtime_state["post_reconnect_reconciliation_done"] = True
        runtime_state["post_reconnect_reconciliation_clean"] = bool(reconciliation.get("clean")) if isinstance(reconciliation, dict) else False
        runtime_state["post_reconnect_reconciliation_orphans"] = list(reconciliation.get("orphans", [])) if isinstance(reconciliation, dict) else []
        runtime_state["post_reconnect_reconciliation_pending_orders"] = list(reconciliation.get("pending_orders", [])) if isinstance(reconciliation, dict) else []
        if runtime_state.get("pending_eod_flatten"):
            process_pending_eod_flatten_retry(
                ib,
                recorder,
                managed_positions,
                args,
                runtime_state,
                reason="reconnect_pending_eod_flatten",
                force=True,
            )
        runtime_state["reconnect_active"] = False
        runtime_state["entries_blocked"] = False
        if runtime_state.get("entries_blocked_reason") in {"ibkr_reconnect", "post_reconnect_reconciliation"}:
            runtime_state["entries_blocked_reason"] = ""
        return {"ok": True, "tickers": tickers, "reconciliation": reconciliation}
    except Exception as exc:
        runtime_state["ibkr_connected"] = False
        runtime_state["reconnect_active"] = True
        runtime_state["reconnect_last_error"] = repr(exc)
        runtime_state["post_reconnect_reconciliation_done"] = False
        runtime_state["entries_blocked"] = True
        runtime_state["entries_blocked_reason"] = "ibkr_reconnect"
        print(f"{now_utc()} RECONNECT_SUPERVISOR_FAILED attempt={attempt} error={exc!r}", flush=True)
        return {"ok": False, "tickers": {}, "error": repr(exc)}


def main() -> int:
    global _ACTIVE_SHUTDOWN_DIAGNOSTICS
    parser = argparse.ArgumentParser(description="v67 live top100 expansion paper trader")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--alpha-rank-csv", default=DEFAULT_ALPHA_RANK)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--max-market-data-subscriptions", type=int, default=100)
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--disable-sqlite", action="store_true")
    parser.add_argument("--sqlite-writer-queue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--restore-managed-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy fallback: allow managed_positions.json to recreate active in-memory positions on startup.",
    )
    parser.add_argument("--duration-seconds", type=int, default=28800)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--portfolio-interval-seconds", type=float, default=10.0)
    parser.add_argument("--market-data-type", type=int, default=1)
    parser.add_argument("--opening-range-seconds", type=int, default=15 * 60)
    parser.add_argument("--min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-spread-bps", type=float, default=50.0)
    parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--position-usd", type=float, default=1000.0)
    parser.add_argument("--risk-guard-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-daily-loss-usd", type=float, default=0.0)
    parser.add_argument("--max-trades-per-day", type=int, default=0)
    parser.add_argument("--max-open-positions", type=int, default=0)
    parser.add_argument("--max-gross-exposure-usd", type=float, default=0.0)
    parser.add_argument("--max-single-position-usd", type=float, default=0.0)
    parser.add_argument("--max-entry-candidate-age-seconds", type=float, default=60.0)
    parser.add_argument("--max-entries-per-cycle", type=int, default=5)
    parser.add_argument("--max-entries-per-minute", type=int, default=5)
    parser.add_argument("--entry-backlog-window-seconds", type=float, default=60.0)
    parser.add_argument("--entry-backlog-threshold", type=int, default=5)
    parser.add_argument("--exit-stop-loss-pct", type=float, default=8.0)
    parser.add_argument("--exit-trailing-activation-pct", type=float, default=3.0)
    parser.add_argument("--exit-trailing-stop-pct", type=float, default=3.0)
    parser.add_argument("--adopt-existing-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-eod-flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eod-flatten-utc", default="19:45")
    parser.add_argument("--eod-retry-seconds", type=float, default=60.0)
    parser.add_argument("--eod-max-retries", type=int, default=5)
    parser.add_argument("--no-new-entries-after-utc", default="19:30")
    parser.add_argument("--market-open-utc", default="13:30")
    parser.add_argument("--market-close-utc", default="20:00")
    parser.add_argument("--new-entries-start-utc", default="13:35")
    parser.add_argument("--manage-exits-start-utc", default="13:30")
    parser.add_argument("--restart-cooldown-seconds", type=float, default=300.0)
    parser.add_argument("--backfill-1m-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backfill-duration", default="1 D")
    parser.add_argument("--backfill-top-n", type=int, default=100)
    parser.add_argument("--backfill-pause-seconds", type=float, default=0.15)
    parser.add_argument("--backfill-current-session-on-rebuild-miss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reconnect-wait-seconds", type=float, default=15.0)
    parser.add_argument("--reconnect-max-attempts", type=int, default=999999)
    parser.add_argument("--enable-overnight-automation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overnight-collector-times-utc", default="20:15,23:00,03:00,07:00,10:30")
    parser.add_argument("--overnight-backlog-collector-times-utc", default="07:00")
    parser.add_argument("--overnight-prioritize-previous-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overnight-collector-start-date", default="2026-01-01")
    parser.add_argument("--overnight-backlog-lookback-days", type=int, default=30)
    parser.add_argument("--overnight-daily-collector-max-tasks", type=int, default=3000)
    parser.add_argument("--overnight-collector-max-tasks", type=int, default=3000)
    parser.add_argument("--overnight-collector-max-attempts", type=int, default=5)
    parser.add_argument("--overnight-collector-retry-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-collector-recent-sessions", type=int, default=5)
    parser.add_argument("--history-collector-client-id", type=int, default=168)
    parser.add_argument("--history-collector-max-runtime-minutes", type=float, default=120.0)
    parser.add_argument("--startup-history-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-history-repair-min-completion-pct", type=float, default=100.0)
    parser.add_argument("--startup-history-repair-max-tasks", type=int, default=3000)
    parser.add_argument("--startup-history-repair-retry-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--daily-top100-build-utc", default="11:30")
    parser.add_argument("--daily-top100-universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--daily-top100-history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--daily-top100-output-dir", default="data/universe")
    parser.add_argument("--daily-top100-latest-output", default="data/universe/daily_top100_latest.csv")
    parser.add_argument("--daily-top100-sqlite-path", default="data/runtime/rankings.sqlite")
    parser.add_argument("--daily-top100-top-n", type=int, default=100)
    parser.add_argument("--allow-stale-top100", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--symbol-denylist", default=DEFAULT_SYMBOL_DENYLIST)
    parser.add_argument("--runtime-ineligible-path", default=DEFAULT_RUNTIME_INELIGIBLE)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    symbols, startup_skipped_ineligible = load_tradeable_top_symbols(
        args.alpha_rank_csv,
        args.top_n,
        min_price=args.min_price,
        symbol_denylist_path=args.symbol_denylist,
        runtime_ineligible_path=args.runtime_ineligible_path,
    )
    recorder = LiveDataRecorder(args.recorder_dir)
    log_dir = install_unified_logger(args.log_dir)
    shutdown = ShutdownDiagnostics(log_dir=log_dir, args=args)
    _ACTIVE_SHUTDOWN_DIAGNOSTICS = shutdown
    shutdown.install_signal_handlers()
    atexit.register(shutdown.atexit)
    sqlite_store = None if args.disable_sqlite else open_sqlite_store(args.sqlite_path, use_writer_queue=bool(args.sqlite_writer_queue))
    setattr(recorder, "sqlite_store", sqlite_store)
    disk = shutil.disk_usage(str(Path(args.recorder_dir).parent if Path(args.recorder_dir).parent else Path(".")))
    log_event(
        "BOT",
        "BOT_START",
        log_dir=log_dir,
        git_commit=current_git_commit(Path.cwd()),
        cli_args=" ".join(sys.argv[1:]),
        sqlite_path=args.sqlite_path or "data/runtime/trading_runtime.sqlite",
        recorder_dir=recorder.session_dir,
        disk_free_bytes=disk.free,
    )

    print("=== v67 live top100 expansion paper trader ===", flush=True)
    print(f"Symbols loaded: {len(symbols)}", flush=True)
    if startup_skipped_ineligible:
        for symbol, info in startup_skipped_ineligible.items():
            print(
                f"{now_utc()} ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol={symbol} "
                f"reason={info.get('reason') or 'ineligible'} source=startup_filter",
                flush=True,
            )
        print(
            f"{now_utc()} STARTUP_INELIGIBLE_SYMBOLS_SKIPPED count={len(startup_skipped_ineligible)} "
            f"symbols={','.join(list(startup_skipped_ineligible)[:20])}",
            flush=True,
        )
    print(f"Recorder dir: {recorder.session_dir}", flush=True)
    print(f"Portfolio/fills recorder: integrated every {args.portfolio_interval_seconds}s", flush=True)
    print(
        "Exit: v46 wide_trail "
        f"stop_loss={args.exit_stop_loss_pct}% trail_activation={args.exit_trailing_activation_pct}% "
        f"trail_stop={args.exit_trailing_stop_pct}% adopt_existing={args.adopt_existing_positions} "
        f"eod_flatten={args.enable_eod_flatten} at {args.eod_flatten_utc} UTC",
        flush=True,
    )
    print(f"Backfill 1m: {args.backfill_1m_on_start} duration={args.backfill_duration} top_n={args.backfill_top_n}", flush=True)
    print(
        f"Overnight automation: {args.enable_overnight_automation} "
        f"collector_slots={args.overnight_collector_times_utc} "
        f"backlog_slots={args.overnight_backlog_collector_times_utc} "
        f"prioritize_previous_day={args.overnight_prioritize_previous_day} "
        f"backlog_lookback_days={args.overnight_backlog_lookback_days} "
        f"startup_history_repair={args.startup_history_repair} "
        f"daily_top100_build={args.daily_top100_build_utc}",
        flush=True,
    )

    ib = IB()
    connect_ibkr_with_retry(ib, args)
    install_commission_report_handler(ib, recorder)

    tickers: dict[str, Any] = {}
    states = {symbol: SymbolState(symbol=symbol) for symbol in symbols}
    contracts: list[tuple[str, Any]] = []
    contract_by_symbol: dict[str, Any] = {}
    seen_fills: set[str] = load_existing_fill_keys(recorder)
    managed_positions: dict[str, ManagedPosition] = {}
    runtime_state = {
        "entries_blocked": False,
        "entries_blocked_until": time.time() + max(0.0, args.restart_cooldown_seconds),
        "entries_blocked_reason": "restart_cooldown" if args.restart_cooldown_seconds > 0 else "",
        "entry_previous_entries_blocked": None,
        "last_unblock_timestamp": 0.0,
        "last_unblock_utc": "",
        "last_restart_unblock_timestamp": 0.0,
        "last_restart_unblock_utc": "",
        "entry_submit_timestamps": [],
        "entry_order_by_order_id": {},
        "entry_rejection_processed": set(),
        "rejected_entries": {},
        "ineligible_symbols": set(startup_skipped_ineligible),
        "ineligible_symbol_reasons": dict(startup_skipped_ineligible),
        "runtime_ineligible_path": args.runtime_ineligible_path,
        "entry_rejected_count": 0,
        "ready_candidates": 0,
        "live_ready_candidates": 0,
        "backfill_context_candidates": 0,
        "stale_ready_candidates": 0,
        "oldest_ready_candidate_age_seconds": None,
        "control_api_commands": [],
        "history_collector_start_utc": "20:15",
        "history_collector_end_utc": str(args.market_open_utc),
        "history_collector_max_tasks": int(args.overnight_collector_max_tasks),
        "history_collector_max_attempts": int(args.overnight_collector_max_attempts),
        "history_collector_recent_sessions": int(args.history_collector_recent_sessions),
        "history_collector_client_id": int(args.history_collector_client_id),
        "history_collector_max_runtime_minutes": float(args.history_collector_max_runtime_minutes),
        "market_open_utc": str(args.market_open_utc),
        "market_close_utc": str(args.market_close_utc),
        "manual_eod_flatten_requested": False,
        "manual_eod_flatten_force": False,
        "pending_eod_flatten": False,
        "pending_eod_flatten_reason": "",
        "pending_eod_flatten_symbols": [],
        "pending_eod_flatten_last_retry_ts": 0.0,
        "eod_recovery_active": False,
        "startup_reconciliation_done": False,
        "startup_reconciliation_clean": False,
        "startup_reconciliation_orphans": [],
        "startup_reconciliation_fractional_orphans": [],
        "startup_reconciliation_whole_share_orphans": [],
        "startup_reconciliation_closed_local": [],
        "startup_reconciliation_pending_orders": [],
        "fractional_orphan_manual_required_seen": {},
        "fractional_orphan_manual_required_suppressed": {},
        "fractional_orphan_manual_required_suppressed_total": 0,
        "eod_final_status": {},
        "ibkr_connected": True,
        "reconnect_active": False,
        "reconnect_attempts": 0,
        "reconnect_last_error": "",
        "reconnect_last_success_at": "",
        "post_reconnect_reconciliation_done": False,
        "post_reconnect_reconciliation_clean": False,
        "post_reconnect_reconciliation_orphans": [],
        "post_reconnect_reconciliation_pending_orders": [],
        "entry_symbols": set(symbols),
        "top100_reload_requested": False,
        "top100_reload_path": "",
        "top100_reload_ranking_date": "",
        "top100_reload_last_error": "",
        "top100_reload_symbols": list(symbols),
        "top100_reload_diagnostics": {},
        "top100_entries_blocked": False,
        "top100_freshness": {},
        "sqlite_writer_status": {},
        "market_closed_logged_dates": set(),
        "risk_guard_last_status": {
            "enabled": bool(args.risk_guard_enabled),
            "blocked": False,
            "reason": "",
        },
        "disk_usage_pct": 0.0,
        "disk_full_entries_blocked": False,
        "disk_usage_last_check_ts": 0.0,
        "partial_entry_count": 0,
        "partial_exit_count": 0,
        "partial_fill_states": {},
        "delayed_fill_after_cancel_count": 0,
        "cancel_but_position_exists_count": 0,
        "process_start_monotonic": shutdown.start_monotonic,
        "shutdown_reason": "",
        "ibkr_disconnect_source": "",
        "ibkr_disconnect_reason": "",
        "ibkr_disconnect_at": "",
    }
    startup_ranking_date = latest_completed_trading_day(datetime.now(timezone.utc), getattr(args, "market_close_utc", "20:00"))
    apply_top100_freshness_gate(runtime_state, args, startup_ranking_date)
    enqueue_startup_history_repair_if_needed(runtime_state, args)
    latest_snapshots: dict[str, dict[str, Any]] = {}
    last_portfolio_record = 0.0
    adopted_once = False
    normal_exit = False

    try:
        for symbol in symbols:
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                continue
            q = qualified[0]
            record_contract_metadata(recorder, q, source="startup")
            metadata_reason = contract_ineligible_reason(q)
            if metadata_reason:
                mark_runtime_symbol_ineligible(
                    runtime_state,
                    symbol,
                    reason=metadata_reason,
                    source="contract_metadata",
                    con_id=getattr(q, "conId", None),
                    raw_message=json.dumps(contract_metadata(q), sort_keys=True),
                )
                print(
                    f"{now_utc()} ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol={symbol} "
                    f"reason={metadata_reason} source=contract_metadata",
                    flush=True,
                )
                continue
            contracts.append((symbol, q))
            contract_by_symbol[symbol] = q
            tickers[symbol] = ib.reqMktData(q, "", False, False)
            print(f"Subscribed {symbol} conId={q.conId}", flush=True)

        startup_broker_rows = ibkr_portfolio_position_rows(ib)
        startup_broker_qty_by_symbol = {row["symbol"]: float(row["quantity"]) for row in startup_broker_rows}
        safe_sqlite_call(
            getattr(recorder, "sqlite_store", None),
            "set_broker_net_positions",
            startup_broker_qty_by_symbol,
        )
        startup_sqlite_active_count = sqlite_active_position_count(recorder)
        load_pending_eod_flatten(
            recorder,
            runtime_state,
            broker_open_count=len(startup_broker_rows),
            sqlite_active_count=startup_sqlite_active_count,
        )
        apply_pending_eod_entry_block(runtime_state)
        restored = restore_managed_positions(
            recorder,
            contract_by_symbol,
            broker_qty_by_symbol=startup_broker_qty_by_symbol,
            runtime_state=runtime_state,
            restore_enabled=bool(args.restore_managed_json),
            disabled_reason=None if args.restore_managed_json else "sqlite_broker_source_of_truth",
        )
        if restored:
            managed_positions.update(restored)
            for symbol, pos in restored.items():
                states.get(symbol, SymbolState(symbol)).signal_sent = True
                record_lifecycle_with_formal(recorder, "RESTORED_MANAGED_POSITION", symbol, quantity=pos.quantity, entry_price=pos.entry_price, peak_price=pos.peak_price, reason="managed_positions_json")
            print(f"{now_utc()} restored_managed_positions={len(restored)}", flush=True)

        backfilled_rows = backfill_recent_1m(ib, recorder, contracts, args)
        state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        current_session_backfilled_rows = 0
        today_session = get_us_equity_session(datetime.now(timezone.utc).date())
        if not today_session.is_trading_day:
            print(
                f"{now_utc()} MARKET_CLOSED_HOLIDAY date={today_session.session_date.isoformat()} "
                f"reason={today_session.reason}",
                flush=True,
            )
            _runtime_set(runtime_state, "market_closed_logged_dates").add(today_session.session_date.isoformat())
        if today_session.is_trading_day and state_rebuild_count == 0 and current_session_candle_count(recorder, args) == 0:
            current_session_backfilled_rows = backfill_current_session_1m(ib, recorder, contracts, args)
            if current_session_backfilled_rows:
                state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        traded_symbols_today = load_traded_symbols_today(recorder)
        if traded_symbols_today:
            for symbol in traded_symbols_today:
                if symbol in states and args.max_one_trade_per_symbol_per_day:
                    states[symbol].signal_sent = True
            print(f"{now_utc()} traded_symbols_today_loaded={len(traded_symbols_today)} max_one_trade_per_symbol_per_day={args.max_one_trade_per_symbol_per_day}", flush=True)

        control_api_server = start_control_api(
            ib=ib,
            recorder=recorder,
            managed_positions=managed_positions,
            runtime_state=runtime_state,
            record_lifecycle_fn=record_lifecycle_with_formal,
            persist_managed_positions_fn=persist_managed_positions,
            host="127.0.0.1",
            port=8767,
        )
        install_ibkr_order_rejection_handler(ib, recorder, managed_positions, runtime_state)

        startup_reconciliation = startup_reconcile_runtime_state(
            ib,
            recorder,
            managed_positions,
            contract_by_symbol,
            runtime_state,
        )
        post_startup_broker_rows = ibkr_portfolio_position_rows(ib)
        post_startup_broker_qty_by_symbol = {row["symbol"]: float(row["quantity"]) for row in post_startup_broker_rows}
        safe_sqlite_call(
            getattr(recorder, "sqlite_store", None),
            "set_broker_net_positions",
            post_startup_broker_qty_by_symbol,
        )
        sqlite_rebuild_result = safe_sqlite_call(
            getattr(recorder, "sqlite_store", None),
            "rebuild_positions_from_executions",
            broker_net_positions=post_startup_broker_qty_by_symbol,
        )
        print(
            f"{now_utc()} STARTUP_SQLITE_POSITION_REBUILD "
            f"broker_snapshot_count={len(post_startup_broker_qty_by_symbol)} "
            f"result={json.dumps(sqlite_rebuild_result or {}, sort_keys=True, default=str)}",
            flush=True,
        )
        if runtime_state.get("pending_eod_flatten"):
            process_pending_eod_flatten_retry(
                ib,
                recorder,
                managed_positions,
                args,
                runtime_state,
                reason="startup_pending_eod_flatten",
                force=True,
            )

        recorder.record_run_metadata({
            "module": "v67_live_top100_expansion_paper_trader",
            "strategy": STRATEGY_NAME,
            "client_id": args.client_id,
            "top_n": args.top_n,
            "seen_fills_loaded": len(seen_fills),
            "restored_positions": len(restored),
            "backfilled_1m_rows": backfilled_rows,
            "state_rebuild_count": state_rebuild_count,
            "current_session_backfilled_1m_rows": current_session_backfilled_rows,
            "startup_reconciliation": startup_reconciliation,
        })
        run_dry_run_reconciliation_report(
            ib,
            recorder,
            managed_positions,
            runtime_state,
            reason="startup_after_restore_and_control_api_start",
        )

        start = time.time()
        while time.time() - start < args.duration_seconds:
            try:
                if not ibkr_connection_alive(ib):
                    recovery = handle_ibkr_disconnect_and_recover(
                        ib,
                        recorder,
                        managed_positions,
                        contract_by_symbol,
                        contracts,
                        runtime_state,
                        args,
                        reason="connection_health_check",
                        seen_fills=seen_fills,
                    )
                    if recovery.get("ok"):
                        tickers = recovery.get("tickers") or tickers
                    else:
                        time.sleep(float(args.reconnect_wait_seconds))
                    continue

                process_overnight_automation(runtime_state, args)

                reload_top100_universe_if_requested(
                    ib,
                    recorder,
                    states,
                    contracts,
                    contract_by_symbol,
                    tickers,
                    latest_snapshots,
                    managed_positions,
                    runtime_state,
                    args,
                )

                process_control_api_commands(
                    ib=ib,
                    recorder=recorder,
                    managed_positions=managed_positions,
                    runtime_state=runtime_state,
                    record_lifecycle_fn=record_lifecycle_with_formal,
                    persist_managed_positions_fn=persist_managed_positions,
                )

                process_history_collector_commands(
                    runtime_state=runtime_state,
                )

                if runtime_state.get("pending_eod_flatten"):
                    process_pending_eod_flatten_retry(
                        ib,
                        recorder,
                        managed_positions,
                        args,
                        runtime_state,
                        reason="periodic_pending_eod_flatten",
                    )

                ib.sleep(args.interval_seconds)
            except Exception as exc:
                print(f"{now_utc()} IBKR_DISCONNECTED during=sleep error={exc!r}", flush=True)
                recovery = handle_ibkr_disconnect_and_recover(
                    ib,
                    recorder,
                    managed_positions,
                    contract_by_symbol,
                    contracts,
                    runtime_state,
                    args,
                    reason="main_loop_exception",
                    seen_fills=seen_fills,
                )
                if recovery.get("ok"):
                    tickers = recovery.get("tickers") or tickers
                else:
                    time.sleep(float(args.reconnect_wait_seconds))
                continue
            loop_now = time.time()
            observed_at = datetime.now(timezone.utc)
            if loop_now - float(runtime_state.get("disk_usage_last_check_ts") or 0.0) >= 60.0:
                runtime_state["disk_usage_last_check_ts"] = loop_now
                monitor_disk_usage(args.recorder_dir, runtime_state, log_dir=log_dir)
            market_open = market_open_datetime_utc(args, observed_at)
            session_elapsed = (observed_at - market_open).total_seconds()
            today_session = get_us_equity_session(observed_at.date())
            market_closed_today = not today_session.is_trading_day
            if market_closed_today:
                runtime_state["entries_blocked_reason"] = "market_closed_holiday"
                closed_key = today_session.session_date.isoformat()
                logged_dates = _runtime_set(runtime_state, "market_closed_logged_dates")
                if closed_key not in logged_dates:
                    print(
                        f"{now_utc()} MARKET_CLOSED_HOLIDAY date={closed_key} reason={today_session.reason}",
                        flush=True,
                    )
                    logged_dates.add(closed_key)
            elif runtime_state.get("entries_blocked_reason") == "market_closed_holiday":
                runtime_state["entries_blocked_reason"] = ""
            pending_eod_entries_blocked = apply_pending_eod_entry_block(runtime_state)
            ready_count = 0
            data_count = 0
            best_symbol = None
            best_score = -999999.0
            rejection_counter = Counter()
            ranked = []
            debug_symbol = contracts[0][0] if contracts else None
            debug_payload: tuple[str, SymbolState, dict[str, Any], dict[str, Any]] | None = None
            zeroish_feature_count = 0
            time_entries_blocked = (
                market_closed_today
                or
                not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            restart_entries_blocked = loop_now < float(runtime_state.get("entries_blocked_until") or 0.0)
            if not restart_entries_blocked and runtime_state.get("entries_blocked_reason") == "restart_cooldown":
                runtime_state["entries_blocked_reason"] = ""
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            reconnect_entries_blocked = bool(runtime_state.get("reconnect_active", False))
            disk_entries_blocked = bool(runtime_state.get("disk_full_entries_blocked", False))
            top100_entries_blocked = bool(runtime_state.get("top100_entries_blocked", False))
            if pending_eod_entries_blocked:
                runtime_state["entries_blocked_reason"] = "pending_eod_flatten"
            elif disk_entries_blocked:
                runtime_state["entries_blocked_reason"] = "disk_full_risk"
            elif runtime_state.get("entries_blocked_reason") == "disk_full_risk":
                runtime_state["entries_blocked_reason"] = ""
            entries_blocked = time_entries_blocked or manual_entries_blocked or restart_entries_blocked or reconnect_entries_blocked or disk_entries_blocked or pending_eod_entries_blocked or top100_entries_blocked
            eod_active = args.enable_eod_flatten and (is_after_utc(args.eod_flatten_utc) or bool(runtime_state.get("manual_eod_flatten_requested", False)))
            if eod_active:
                enforce_eod_flatten_if_due(
                    ib,
                    recorder,
                    managed_positions,
                    args,
                    runtime_state,
                    eod_active=True,
                    reason="main_loop_eod_active_failsafe",
                )
                pending_eod_entries_blocked = apply_pending_eod_entry_block(runtime_state)
                manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
                entries_blocked = (
                    time_entries_blocked
                    or manual_entries_blocked
                    or restart_entries_blocked
                    or reconnect_entries_blocked
                    or disk_entries_blocked
                    or pending_eod_entries_blocked
                    or top100_entries_blocked
                )

            mark_entry_block_state(runtime_state, entries_blocked, loop_now)
            entry_candidates: list[dict[str, Any]] = []

            for symbol, q in contracts:
                snap = snapshot_from_ticker(symbol, tickers[symbol])
                if snap.get("price") is None:
                    continue
                latest_snapshots[symbol] = snap
                data_count += 1
                state = states[symbol]
                update_state(state, snap, session_elapsed, args.opening_range_seconds, observed_at=observed_at)
                features = compute_live_safe_features(state, snap, args)
                if features_are_all_zeroish(features):
                    zeroish_feature_count += 1
                if symbol == debug_symbol:
                    debug_payload = (symbol, state, snap, features)
                ranked.append((symbol, features["score"], features))
                if features["score"] > best_score:
                    best_score = features["score"]
                    best_symbol = symbol
                if not features["ready"]:
                    for reason in features["reason"].split(";"):
                        rejection_counter[reason] += 1
                    state.ready_since_ts = None
                    state.ready_since_utc = None
                    state.signal_source = ""
                    state.stale_ready_logged = False

                has_active_position = symbol in managed_positions and managed_positions[symbol].active and not managed_positions[symbol].exit_sent
                entry_symbol_allowed = symbol in _runtime_set(runtime_state, "entry_symbols")
                symbol_ineligible = symbol in _runtime_set(runtime_state, "ineligible_symbols")
                if symbol_ineligible and features["ready"]:
                    info = runtime_ineligible_info(runtime_state, symbol)
                    reason = str(info.get("reason") or "ineligible_no_trading_permission_kid")
                    rejection_counter[reason] += 1
                    if not state.stale_ready_logged:
                        print(
                            f"{now_utc()} ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol={symbol} "
                            f"reason={reason} source={info.get('source') or 'runtime_cache'}",
                            flush=True,
                        )
                        record_lifecycle_with_formal(
                            recorder,
                            "ENTRY_SYMBOL_INELIGIBLE_SKIPPED",
                            symbol,
                            action="BUY",
                            price=features.get("entry_price"),
                            reason=reason,
                            raw_json={**features, "blocked_by": info.get("source") or "ineligible_cache"},
                        )
                        state.stale_ready_logged = True
                    continue
                if features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed:
                    if state.ready_since_ts is None:
                        state.ready_since_ts = loop_now
                        state.ready_since_utc = now_utc()
                        state.signal_source = state.last_update_source or "unknown"
                        state.stale_ready_logged = False
                    candidate_age = ready_candidate_age_seconds(state, loop_now)
                    entry_candidates.append(
                        {
                            "symbol": symbol,
                            "contract": q,
                            "state": state,
                            "snap": snap,
                            "features": features,
                            "candidate_age_seconds": candidate_age,
                        }
                    )
                if features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed and entries_blocked:
                    record_lifecycle_with_formal(
                        recorder,
                        "BUY_BLOCKED",
                        symbol,
                        action="BUY",
                        price=features.get("entry_price"),
                        reason="entries_blocked_manual_or_time_window",
                        raw_json={
                            **features,
                            "ready_since": state.ready_since_utc,
                            "candidate_age_seconds": candidate_age,
                            "manual_entries_blocked": manual_entries_blocked,
                            "time_entries_blocked": time_entries_blocked,
                            "restart_entries_blocked": restart_entries_blocked,
                            "reconnect_entries_blocked": reconnect_entries_blocked,
                            "entries_blocked_reason": runtime_state.get("entries_blocked_reason"),
                        },
                    )

            ranking_position_by_symbol = {
                symbol: idx + 1 for idx, (symbol, _score, _features) in enumerate(sorted(ranked, key=lambda item: item[1], reverse=True))
            }
            ready_candidates_total = len(entry_candidates)
            candidate_rejection_reasons = {
                candidate["symbol"]: ready_candidate_rejection_reason(
                    candidate["state"],
                    runtime_state,
                    max_age_seconds=float(args.max_entry_candidate_age_seconds),
                    now_ts=loop_now,
                )
                for candidate in entry_candidates
            }
            stale_ready_candidates = sum(1 for reason in candidate_rejection_reasons.values() if reason)
            live_ready_candidates = sum(1 for candidate in entry_candidates if (candidate["state"].signal_source or candidate["state"].last_update_source) == "live")
            backfill_context_candidates = sum(1 for candidate in entry_candidates if (candidate["state"].signal_source or candidate["state"].last_update_source) in {"backfill", "reconstructed"})
            candidate_ages = [age for age in (candidate.get("candidate_age_seconds") for candidate in entry_candidates) if age is not None]
            oldest_ready_candidate_age = max(candidate_ages) if candidate_ages else None
            runtime_state["ready_candidates"] = ready_candidates_total
            runtime_state["live_ready_candidates"] = live_ready_candidates
            runtime_state["backfill_context_candidates"] = backfill_context_candidates
            runtime_state["stale_ready_candidates"] = stale_ready_candidates
            runtime_state["oldest_ready_candidate_age_seconds"] = oldest_ready_candidate_age

            entries_submitted_this_cycle = 0
            if not entries_blocked:
                ordered_entry_candidates = sorted(
                    entry_candidates,
                    key=lambda candidate: float(candidate["features"].get("score") or 0.0),
                    reverse=True,
                )
                for candidate in ordered_entry_candidates:
                    symbol = candidate["symbol"]
                    q = candidate["contract"]
                    state = candidate["state"]
                    snap = candidate["snap"]
                    features = candidate["features"]
                    ranking_position = ranking_position_by_symbol.get(symbol)
                    diagnostics = ready_candidate_diagnostics(
                        state,
                        features,
                        runtime_state,
                        now_ts=loop_now,
                        ranking_position=ranking_position,
                    )
                    skip_reason = candidate_rejection_reasons.get(symbol) or ""
                    if skip_reason:
                        if not state.stale_ready_logged:
                            record_lifecycle_with_formal(
                                recorder,
                                "STALE_OR_BACKFILL_READY_SKIPPED",
                                symbol,
                                action="BUY",
                                price=features.get("entry_price"),
                                reason=skip_reason,
                                raw_json={**features, **diagnostics, "skip_reason": skip_reason},
                            )
                            print(
                                f"{now_utc()} STALE_OR_BACKFILL_READY_SKIPPED symbol={symbol} "
                                f"signal_source={diagnostics.get('signal_source')} ready_since={diagnostics.get('ready_since') or ''} "
                                f"signal_time={diagnostics.get('signal_time') or ''} "
                                f"last_unblock_time={diagnostics.get('last_restart_unblock_time') or ''} "
                                f"candidate_age_seconds={diagnostics.get('candidate_age_seconds')} reason={skip_reason}",
                                flush=True,
                            )
                            state.stale_ready_logged = True
                        state.ready_since_ts = None
                        state.ready_since_utc = None
                        state.signal_source = ""
                        continue
                    max_per_cycle = int(args.max_entries_per_cycle or 0)
                    if max_per_cycle > 0 and entries_submitted_this_cycle >= max_per_cycle:
                        break
                    minute_capacity = entry_minute_capacity(runtime_state, int(args.max_entries_per_minute or 0), loop_now)
                    if minute_capacity <= 0:
                        runtime_rate_limited_log(
                            runtime_state,
                            "ENTRY_RATE_LIMIT_BLOCK",
                            f"{now_utc()} ENTRY_RATE_LIMIT_BLOCK reason=max_entries_per_minute "
                            f"max_entries_per_minute={int(args.max_entries_per_minute or 0)} ready_candidates={ready_candidates_total}",
                            key="max_entries_per_minute",
                            max_unique=1,
                            window_seconds=60.0,
                        )
                        break
                    ready_count += 1
                    price = features.get("entry_price")
                    qty = max(1, int(args.position_usd // price)) if price and price > 0 else 0
                    signal_payload = {**features, **diagnostics}
                    record_lifecycle_with_formal(
                        recorder,
                        "SIGNAL_READY",
                        symbol,
                        action="BUY",
                        quantity=qty,
                        price=price,
                        reason=features["reason"],
                        raw_json=signal_payload,
                    )
                    candidate_notional = float(qty) * float(price) if qty and price else 0.0
                    risk_status = evaluate_risk_guard(
                        recorder,
                        managed_positions,
                        latest_snapshots,
                        args,
                        symbol=symbol,
                        candidate_notional=candidate_notional,
                    )
                    runtime_state["risk_guard_last_status"] = risk_status
                    if risk_status.get("blocked"):
                        reason = str(risk_status.get("reason") or "risk_guard")
                        runtime_rate_limited_log(
                            runtime_state,
                            "RISK_GUARD_BLOCK_ENTRY",
                            f"{now_utc()} RISK_GUARD_BLOCK_ENTRY symbol={symbol} reason={reason} "
                            f"daily_pnl={risk_status.get('daily_pnl')} trades_today={risk_status.get('trades_today')} "
                            f"active_positions={risk_status.get('active_positions')} gross_exposure={risk_status.get('gross_exposure')} "
                            f"candidate_notional={risk_status.get('candidate_notional')}",
                            key=f"{reason}:{symbol}",
                            max_unique=10,
                            window_seconds=60.0,
                        )
                        record_lifecycle_with_formal(
                            recorder,
                            "RISK_GUARD_BLOCK_ENTRY",
                            symbol,
                            action="BUY",
                            quantity=qty,
                            price=price,
                            reason=reason,
                            raw_json={**risk_status, **diagnostics},
                        )
                        safe_sqlite_call(
                            getattr(recorder, "sqlite_store", None),
                            "record_risk_event",
                            event_type="RISK_GUARD_BLOCK_ENTRY",
                            category="entry_guard",
                            severity="WARN",
                            strategy_name=STRATEGY_NAME,
                            session_date=getattr(recorder, "session_date", None),
                            symbol=symbol,
                            blocked=1,
                            reason=reason,
                            daily_pnl=risk_status.get("daily_pnl"),
                            gross_exposure=risk_status.get("gross_exposure"),
                            active_positions=risk_status.get("active_positions"),
                            trades_today=risk_status.get("trades_today"),
                            raw_json={**risk_status, **diagnostics},
                        )
                        continue
                    order = MarketOrder("BUY", qty)
                    order.tif = "DAY"
                    order.outsideRth = False
                    trade = ib.placeOrder(q, order)
                    order_id_for_entry = getattr(getattr(trade, "order", None), "orderId", "")
                    _runtime_dict(runtime_state, "entry_order_by_order_id")[_runtime_order_id(order_id_for_entry)] = {
                        "symbol": symbol,
                        "quantity": qty,
                        "price": price,
                        "submitted_at": now_utc(),
                        "ranking_position": ranking_position,
                        "score": features.get("score"),
                        "conId": getattr(q, "conId", None),
                    }
                    submission_count = record_entry_submission(runtime_state, loop_now)
                    entries_submitted_this_cycle += 1
                    backlog_window = float(args.entry_backlog_window_seconds or 60.0)
                    backlog_threshold = int(args.entry_backlog_threshold or 0)
                    if backlog_threshold > 0:
                        recent_for_backlog = prune_entry_submit_timestamps(runtime_state, loop_now, window_seconds=backlog_window)
                        if len(recent_for_backlog) > backlog_threshold:
                            backlog_key = int(loop_now // max(1.0, backlog_window))
                            if runtime_state.get("entry_backlog_last_key") != backlog_key:
                                runtime_state["entry_backlog_last_key"] = backlog_key
                                print(
                                    f"{now_utc()} ENTRY_BACKLOG_DETECTED count={len(recent_for_backlog)} "
                                    f"window_seconds={backlog_window:.1f} threshold={backlog_threshold} "
                                    f"latest_symbol={symbol}",
                                    flush=True,
                                )
                                safe_sqlite_call(
                                    getattr(recorder, "sqlite_store", None),
                                    "record_runtime_event",
                                    event_type="ENTRY_BACKLOG_DETECTED",
                                    severity="WARN",
                                    strategy_name=STRATEGY_NAME,
                                    session_date=getattr(recorder, "session_date", None),
                                    symbol=symbol,
                                    source="v67_live_runtime",
                                    reason="entry_rate_burst",
                                    raw_json={
                                        "count": len(recent_for_backlog),
                                        "window_seconds": backlog_window,
                                        "threshold": backlog_threshold,
                                        "latest_symbol": symbol,
                                    },
                                )
                    if qty > 0 and price and price > 0:
                        managed_positions[symbol] = ManagedPosition(
                            symbol=symbol,
                            contract=q,
                            quantity=qty,
                            entry_price=float(price),
                            entry_time=now_utc(),
                            peak_price=float(price),
                            entry_fill_verified=False,
                        )
                        persist_managed_positions(recorder, managed_positions)
                    order_payload = {
                        **diagnostics,
                        "entries_submitted_last_minute": submission_count,
                        "max_entries_per_cycle": int(args.max_entries_per_cycle or 0),
                        "max_entries_per_minute": int(args.max_entries_per_minute or 0),
                    }
                    record_lifecycle_with_formal(
                        recorder,
                        "BUY_ORDER_SENT",
                        symbol,
                        action="BUY",
                        quantity=qty,
                        price=price,
                        order_id=trade.order.orderId,
                        entry_price=price,
                        peak_price=price,
                        decision_bid=snap.get("bid"),
                        decision_ask=snap.get("ask"),
                        decision_mid=snap.get("mid_price"),
                        decision_last=snap.get("last"),
                        spread_pct=((snap.get("spread") / snap.get("mid_price") * 100.0) if snap.get("spread") and snap.get("mid_price") else None),
                        entry_fill_verified="false",
                        raw_json=order_payload,
                    )
                    print(
                        f"PAPER BUY SENT symbol={symbol} qty={qty} price={price:.2f} "
                        f"score={features['score']:.2f} ranking_position={ranking_position or ''} "
                        f"signal_source={diagnostics.get('signal_source') or ''} "
                        f"signal_time={diagnostics.get('signal_time') or ''} ready_since={diagnostics.get('ready_since') or ''} "
                        f"last_live_update_at={diagnostics.get('last_live_update_at') or ''} "
                        f"candidate_age_seconds={diagnostics.get('candidate_age_seconds')} "
                        f"last_restart_unblock_time={diagnostics.get('last_restart_unblock_time') or ''} "
                        f"entry_decision_time={diagnostics.get('entry_decision_time')} "
                        f"orderId={order_id_for_entry} tif={order.tif} outsideRth={order.outsideRth}",
                        flush=True,
                    )
                    state.signal_sent = True

            adopted_count = 0
            if args.adopt_existing_positions and not adopted_once and data_count > 0:
                adopted_count = adopt_existing_long_positions(ib, recorder, contract_by_symbol, latest_snapshots, managed_positions, runtime_state)
                for symbol in managed_positions:
                    if symbol in states:
                        states[symbol].signal_sent = True
                adopted_once = True
                if adopted_count:
                    persist_managed_positions(recorder, managed_positions)

            exit_count = manage_exits(ib, recorder, managed_positions, latest_snapshots, args, runtime_state)
            if exit_count:
                persist_managed_positions(recorder, managed_positions)

            new_fills = None
            latest_portfolio_rows: list[dict[str, Any]] = []
            if loop_now - last_portfolio_record >= args.portfolio_interval_seconds:
                try:
                    latest_portfolio_rows = ibkr_portfolio_position_rows(ib)
                    latest_broker_qty_by_symbol = {row["symbol"]: float(row["quantity"]) for row in latest_portfolio_rows}
                    safe_sqlite_call(
                        getattr(recorder, "sqlite_store", None),
                        "set_broker_net_positions",
                        latest_broker_qty_by_symbol,
                    )
                    record_account_snapshot(ib, recorder)
                    new_fills = record_recent_fills(ib, recorder, seen_fills)
                    position_reconcile_started_at = now_utc()
                    safe_sqlite_call(
                        getattr(recorder, "sqlite_store", None),
                        "mark_operation_status",
                        "position_reconcile",
                        "running",
                        started_at=position_reconcile_started_at,
                    )
                    sqlite_position_reconcile = safe_sqlite_call(
                        getattr(recorder, "sqlite_store", None),
                        "reconcile_active_positions_to_broker_snapshot",
                        latest_broker_qty_by_symbol,
                    )
                    safe_sqlite_call(
                        getattr(recorder, "sqlite_store", None),
                        "mark_operation_status",
                        "position_reconcile",
                        "idle",
                        started_at=position_reconcile_started_at,
                        result=sqlite_position_reconcile or {},
                    )
                    if sqlite_position_reconcile and sqlite_position_reconcile.get("suppressed_historical_open_symbols_count"):
                        print(
                            f"{now_utc()} SQLITE_POSITION_BROKER_SNAPSHOT_RECONCILE "
                            f"result={json.dumps(sqlite_position_reconcile, sort_keys=True, default=str)}",
                            flush=True,
                        )
                    lifecycle_fills_updated = enrich_lifecycle_with_fills(recorder)
                    entry_fills_verified = sync_managed_entry_fill_verification(recorder, managed_positions)
                    fill_diagnostics_updated = process_fill_lifecycle_diagnostics(
                        ib,
                        recorder,
                        managed_positions,
                        runtime_state,
                    )
                    if lifecycle_fills_updated:
                        print(f"{now_utc()} lifecycle_fills_updated={lifecycle_fills_updated}", flush=True)
                    if fill_diagnostics_updated:
                        print(f"{now_utc()} fill_lifecycle_diagnostics_updated={fill_diagnostics_updated}", flush=True)
                    if entry_fills_verified:
                        print(f"{now_utc()} entry_fills_verified={entry_fills_verified}", flush=True)
                    verification = verify_managed_positions_against_ibkr(
                        ib,
                        recorder,
                        managed_positions,
                        reason="portfolio_recorder_verification",
                        runtime_state=runtime_state,
                    )
                    runtime_state["portfolio_last_verification"] = verification
                    run_dry_run_reconciliation_report(
                        ib,
                        recorder,
                        managed_positions,
                        runtime_state,
                        reason="periodic_portfolio_record",
                    )
                    record_strategy_equity(recorder, managed_positions, latest_snapshots)
                    persist_managed_positions(recorder, managed_positions, latest_snapshots, latest_portfolio_rows)
                    last_portfolio_record = loop_now
                    if runtime_state.get("pending_eod_flatten"):
                        process_portfolio_sync_pending_eod_retry(
                            ib,
                            recorder,
                            managed_positions,
                            args,
                            runtime_state,
                        )
                except Exception as exc:
                    print(f"{now_utc()} portfolio_recorder_error={exc!r}", flush=True)
                    if "disconnect" in repr(exc).lower() or "connection" in repr(exc).lower() or "socket" in repr(exc).lower():
                        print(f"{now_utc()} IBKR_DISCONNECTED during=portfolio_recorder error={exc!r}", flush=True)
                        recovery = handle_ibkr_disconnect_and_recover(
                            ib,
                            recorder,
                            managed_positions,
                            contract_by_symbol,
                            contracts,
                            runtime_state,
                            args,
                            reason="portfolio_recorder_exception",
                            seen_fills=seen_fills,
                        )
                        if recovery.get("ok"):
                            tickers = recovery.get("tickers") or tickers
                        else:
                            time.sleep(float(args.reconnect_wait_seconds))

            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            if debug_payload is not None:
                symbol, state, snap, features = debug_payload
                log_live_feature_debug(
                    runtime_state=runtime_state,
                    symbol=symbol,
                    state=state,
                    snap=snap,
                    features=features,
                    market_open=market_open,
                    session_elapsed=session_elapsed,
                    reason="top_ranked_symbol",
                )
            if data_count > 0 and zeroish_feature_count == data_count and not market_closed_today:
                if debug_payload is not None:
                    symbol, state, snap, features = debug_payload
                    log_live_feature_debug(
                        runtime_state=runtime_state,
                        symbol=symbol,
                        state=state,
                        snap=snap,
                        features=features,
                        market_open=market_open,
                        session_elapsed=session_elapsed,
                        reason="all_features_zero",
                    )
                warning_key = "live_features_all_zero_warning_last"
                if loop_now - float(runtime_state.get(warning_key) or 0.0) >= 60.0:
                    runtime_state[warning_key] = loop_now
                    print(
                        f"{now_utc()} LIVE_FEATURES_ALL_ZERO_WARNING with_data={data_count} "
                        f"zeroish_features={zeroish_feature_count} session_start={market_open.isoformat()} "
                        f"session_elapsed_seconds={session_elapsed:.1f} top_symbol={debug_symbol}",
                        flush=True,
                    )
            top5_str = " | ".join([f"{s}:{score:.1f}" for s, score, _ in ranked[:5]])
            rejection_summary = ", ".join([f"{k}={v}" for k, v in rejection_counter.most_common(5)])
            portfolio_part = f" portfolio_recorded=1 new_fills={new_fills}" if new_fills is not None else ""
            active_managed = sum(1 for p in managed_positions.values() if p.active)
            open_price_ok, open_price_missing = open_position_price_diagnostics(
                managed_positions,
                latest_snapshots,
                latest_portfolio_rows,
            )
            if active_managed:
                persist_managed_positions(recorder, managed_positions, latest_snapshots, latest_portfolio_rows)
            time_entries_blocked = (
                market_closed_today
                or not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            restart_entries_blocked = loop_now < float(runtime_state.get("entries_blocked_until") or 0.0)
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            reconnect_entries_blocked = bool(runtime_state.get("reconnect_active", False))
            disk_entries_blocked = bool(runtime_state.get("disk_full_entries_blocked", False))
            top100_entries_blocked = bool(runtime_state.get("top100_entries_blocked", False))
            pending_eod_entries_blocked = bool(runtime_state.get("pending_eod_flatten"))
            entries_blocked = time_entries_blocked or manual_entries_blocked or restart_entries_blocked or reconnect_entries_blocked or disk_entries_blocked or pending_eod_entries_blocked or top100_entries_blocked
            eod_active = args.enable_eod_flatten and (is_after_utc(args.eod_flatten_utc) or bool(runtime_state.get("manual_eod_flatten_requested", False)))
            risk_status = runtime_state.get("risk_guard_last_status") if isinstance(runtime_state.get("risk_guard_last_status"), dict) else {}
            risk_guard_block = bool((risk_status or {}).get("blocked"))
            risk_guard_reason = str((risk_status or {}).get("reason") or "")
            subscriptions_cap = max(0, int(getattr(args, "max_market_data_subscriptions", 100) or 0))
            subscription_diag = runtime_state.get("top100_reload_diagnostics") if isinstance(runtime_state.get("top100_reload_diagnostics"), dict) else {}
            subscription_cap_block = bool(subscription_diag.get("skipped_due_to_subscription_cap")) or (subscriptions_cap > 0 and len(tickers) >= subscriptions_cap)
            sqlite_writer_status = {}
            if sqlite_store is not None and hasattr(sqlite_store, "status"):
                try:
                    sqlite_writer_status = sqlite_store.status()
                    runtime_state["sqlite_writer_status"] = sqlite_writer_status
                except Exception as exc:
                    sqlite_writer_status = {"last_write_error": repr(exc)}
                    runtime_state["sqlite_writer_status"] = sqlite_writer_status
            sqlite_last_error = str(sqlite_writer_status.get("last_write_error") or "").replace(" ", "_")
            last_eod_retry_age = pending_eod_retry_age_seconds(runtime_state, loop_now)
            last_eod_retry_age_text = "" if last_eod_retry_age is None else f"{last_eod_retry_age:.1f}"
            oldest_ready_age = runtime_state.get("oldest_ready_candidate_age_seconds")
            oldest_ready_age_text = "" if oldest_ready_age is None else f"{float(oldest_ready_age):.1f}"
            process_uptime_seconds = shutdown.uptime_seconds()
            heartbeat_line = (
                f"{now_utc()} heartbeat scanned={len(contracts)} with_data={data_count} ready_new={ready_count} "
                f"ready_candidates={int(runtime_state.get('ready_candidates') or 0)} "
                f"live_ready_candidates={int(runtime_state.get('live_ready_candidates') or 0)} "
                f"backfill_context_candidates={int(runtime_state.get('backfill_context_candidates') or 0)} "
                f"stale_ready_candidates={int(runtime_state.get('stale_ready_candidates') or 0)} "
                f"oldest_ready_candidate_age_seconds={oldest_ready_age_text} "
                f"last_restart_unblock_time={runtime_state.get('last_restart_unblock_utc') or ''} "
                f"adopted={adopted_count} exits_sent={exit_count} managed_open={active_managed} entries_blocked={int(entries_blocked)} "
                f"entries_blocked_reason={runtime_state.get('entries_blocked_reason') or ''} "
                f"manual_block={int(manual_entries_blocked)} restart_block={int(restart_entries_blocked)} reconnect_block={int(reconnect_entries_blocked)} disk_block={int(disk_entries_blocked)} "
                f"top100_block={int(top100_entries_blocked)} "
                f"pending_eod_flatten={int(pending_eod_entries_blocked)} eod_recovery_active={int(bool(runtime_state.get('eod_recovery_active')))} "
                f"last_eod_retry_age_seconds={last_eod_retry_age_text} eod_active={int(eod_active)} "
                f"risk_guard_block={int(risk_guard_block)} risk_guard_reason={risk_guard_reason} "
                f"open_price_ok={open_price_ok} open_price_missing={open_price_missing} "
                f"subscriptions_active={len(tickers)} subscriptions_cap={subscriptions_cap} subscription_cap_block={int(subscription_cap_block)} "
                f"sqlite_queue_depth={int(sqlite_writer_status.get('queue_depth') or 0)} "
                f"sqlite_dropped_writes={int(sqlite_writer_status.get('dropped_writes') or 0)} "
                f"sqlite_last_write_latency_ms={sqlite_writer_status.get('last_write_latency_ms') or ''} "
                f"sqlite_last_write_error={sqlite_last_error} "
                f"ineligible_symbols={len(_runtime_set(runtime_state, 'ineligible_symbols'))} "
                f"entry_rejected_count={int(runtime_state.get('entry_rejected_count') or 0)} "
                f"process_uptime_seconds={process_uptime_seconds:.1f} "
                f"best={best_symbol}:{best_score:.2f} top5=[{top5_str}] rejects=[{rejection_summary}]"
                f"{portfolio_part}"
            )
            emit_heartbeat(heartbeat_line, runtime_state, log_dir)
        normal_exit = True
        shutdown.log_main_loop_exit(reason="duration_elapsed", exit_code=0)

    except SystemExit as exc:
        if shutdown.reason == "unknown":
            code = exc.code if isinstance(exc.code, int) else 0
            shutdown.set_reason("system_exit", exit_code=code)
        raise
    except KeyboardInterrupt:
        shutdown.set_reason("keyboard_interrupt", exit_code=130)
        raise SystemExit(130)
    except Exception:
        shutdown.set_reason("exception", exit_code=1)
        raise
    finally:
        if not normal_exit and shutdown.reason == "unknown":
            shutdown.set_reason("main_loop_interrupted")
        persist_managed_positions(recorder, managed_positions)
        if sqlite_store is not None:
            sqlite_store.close()
        for ticker in tickers.values():
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        log_ibkr_disconnect_source(
            runtime_state,
            source="shutdown",
            reason=shutdown.reason or ("normal_exit" if normal_exit else "shutdown"),
            log_dir=log_dir,
            connected_before=int(ibkr_connection_alive(ib)),
        )
        ib.disconnect()
        print("Disconnected", flush=True)
        shutdown.log_exit(
            reason=shutdown.reason or ("duration_elapsed" if normal_exit else "shutdown"),
            exit_code=0 if shutdown.exit_code is None else shutdown.exit_code,
            recorder_dir=recorder.session_dir,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        if _ACTIVE_SHUTDOWN_DIAGNOSTICS is not None:
            _ACTIVE_SHUTDOWN_DIAGNOSTICS.log_exit(reason="keyboard_interrupt", exit_code=130)
        else:
            log_event("BOT", "BOT_STOP", reason="keyboard_interrupt", exit_code=130)
        raise SystemExit(130)
    except Exception as exc:
        if _ACTIVE_SHUTDOWN_DIAGNOSTICS is not None:
            _ACTIVE_SHUTDOWN_DIAGNOSTICS.log_exit(reason="exception", exit_code=1)
        log_event("BOT", "BOT_CRASH", "CRITICAL", exception=repr(exc), traceback=format_traceback(exc))
        raise
