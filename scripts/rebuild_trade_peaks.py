#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.analysis.common import fnum, parse_dt, load_trade_candles as load_shared_trade_candles
from src.live_trading.ranking.daily_top100_builder import normalize_history_df, parquet_path


DEFAULT_DB = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
PEAK_REBUILD_VERSION = 2
VALID_PEAK_QUALITIES = {"EXACT", "PARTIAL", "PARTIAL_COVERAGE"}
RETRY_PEAK_QUALITIES = {"HISTORY_NOT_FINALIZED", "RETRY_PENDING"}
FINAL_MISSING_PEAK_QUALITIES = {"MISSING_CANDLES_FINAL", "OUTSIDE_CANDLE_RANGE", "MISSING_ENTRY_TIME", "MISSING_EXIT_TIME", "NEEDS_REBUILD"}
PEAK_VALIDATION_TOLERANCE_PCT = 0.05


@dataclass(frozen=True)
class PeakResult:
    peak_price: float | None
    peak_time: str
    peak_pct: float | None
    low_price: float | None
    low_time: str
    mae_pct: float | None
    drop_from_peak_pct: float | None
    giveback_usd: float | None
    peak_unrealized_pnl: float | None
    max_adverse_unrealized_pnl: float | None
    candle_count: int
    expected_candle_count: int
    candles_min_time_utc: str
    candles_max_time_utc: str
    peak_data_quality: str
    validation_status: str
    validation_reason: str


def pct(price: float | None, base: float | None) -> float | None:
    if price is None or base is None or base <= 0:
        return None
    return ((float(price) / float(base)) - 1.0) * 100.0


def iso_ts(value: Any) -> str:
    parsed = parse_dt(value)
    return "" if parsed is None else parsed.isoformat()


def parse_raw_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def connect_sqlite(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    else:
        conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def production_db_path() -> Path:
    return REPO_ROOT / DEFAULT_DB


def is_production_db_path(sqlite_path: Path) -> bool:
    return resolved_path(sqlite_path) == resolved_path(production_db_path())


def trader_process_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "v67_live_top100_expansion_paper_trader|v67-trader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def enforce_apply_guard(sqlite_path: Path) -> None:
    absolute_path = resolved_path(sqlite_path)
    if is_production_db_path(sqlite_path):
        if trader_process_running():
            raise RuntimeError("v67 trader process appears active on production DB; refusing --apply")
        return
    print(f"NON_PRODUCTION_DATABASE_APPLY sqlite_path={absolute_path}", flush=True)




def run_integrity_check(sqlite_path: Path) -> str:
    try:
        with connect_sqlite(sqlite_path, read_only=True) as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
    except Exception as exc:
        return f"ERROR: {exc!r}"
    if row is None:
        return "ERROR: integrity_check returned no rows"
    return str(row[0])


def assert_sqlite_integrity_ok(sqlite_path: Path, *, label: str) -> None:
    result = run_integrity_check(sqlite_path)
    if result.lower() != "ok":
        raise RuntimeError(
            f"SQLITE_INTEGRITY_CHECK_FAILED label={label} "
            f"sqlite_path={resolved_path(sqlite_path)} result={result}"
        )


def create_online_backup(source_path: Path, target_path: Path) -> Path:
    source = resolved_path(source_path)
    target = resolved_path(target_path)
    if source == target:
        raise RuntimeError(f"SQLITE_BACKUP_REFUSES_SAME_PATH sqlite_path={source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"SQLITE_BACKUP_TARGET_EXISTS backup_path={target}")
    src_conn = connect_sqlite(source, read_only=True)
    dst_conn = sqlite3.connect(target, timeout=30)
    try:
        src_conn.backup(dst_conn)
        dst_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    finally:
        dst_conn.close()
        src_conn.close()
    assert_sqlite_integrity_ok(target, label="backup")
    print(f"SQLITE_ONLINE_BACKUP_DONE source_path={source} backup_path={target}", flush=True)
    return target

def selected_trade_rows(conn: sqlite3.Connection, *, date: str | None, trade_id: str | None) -> list[dict[str, Any]]:
    columns = table_columns(conn, "trades")
    if not columns:
        return []
    wanted = [
        "trade_id",
        "strategy_name",
        "session_date",
        "symbol",
        "status",
        "entry_fill_time",
        "exit_fill_time",
        "closed_at",
        "entry_price",
        "exit_price",
        "quantity",
        "gross_pnl",
        "commission",
        "net_pnl",
        "mfe_pct",
        "mae_pct",
        "peak_price",
        "low_price",
        "peak_unrealized_pnl",
        "max_adverse_unrealized_pnl",
        "giveback_from_peak",
        "raw_json",
    ]
    selected = [col for col in wanted if col in columns]
    clauses = [
        "UPPER(COALESCE(status, '')) IN ('CLOSED', 'COMMISSION_PENDING', 'PNL_PENDING')",
        "entry_fill_time IS NOT NULL",
        "exit_fill_time IS NOT NULL",
        "entry_price IS NOT NULL",
        "exit_price IS NOT NULL",
    ]
    params: list[Any] = []
    if date:
        clauses.append("(session_date = ? OR substr(COALESCE(exit_fill_time, closed_at, ''), 1, 10) = ?)")
        params.extend([date, date])
    if trade_id:
        clauses.append("trade_id = ?")
        params.append(trade_id)
    return [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT {', '.join(selected)}
            FROM trades
            WHERE {' AND '.join(clauses)}
            ORDER BY UPPER(symbol), entry_fill_time, exit_fill_time, trade_id
            """,
            params,
        ).fetchall()
    ]


def expected_candles(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    seconds = max(0.0, (exit_time - entry_time).total_seconds())
    return int(seconds // 60) + 1


def load_session_candles(history_dir: Path, symbol: str, session_date: str, session_type: str) -> pd.DataFrame:
    path = parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), session_type)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=["timestamp", "high", "low"])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    try:
        out = normalize_history_df(df)
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in out.columns or "high" not in out.columns or "low" not in out.columns:
        return pd.DataFrame()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    return out.dropna(subset=["timestamp", "high", "low"]).sort_values("timestamp").reset_index(drop=True)


def load_trade_candles(history_dir: Path, symbol: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp, session_type: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in pd.date_range(entry_time.date(), exit_time.date(), freq="D"):
        frame = load_session_candles(history_dir, symbol, day.date().isoformat(), session_type)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def validate_peak(
    *,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    peak_price: float | None,
    peak_time: pd.Timestamp | None,
    peak_pct: float | None,
    drop_from_peak_pct: float | None,
    giveback_usd: float | None,
    gross_return_pct: float | None,
) -> tuple[str, str]:
    reasons: list[str] = []
    if peak_price is None:
        if peak_pct is not None or drop_from_peak_pct is not None or giveback_usd is not None:
            reasons.append("null_peak_has_non_null_derived_values")
        return ("VALID" if not reasons else "INVALID", ";".join(reasons))
    if peak_time is None:
        reasons.append("missing_peak_time")
    else:
        if peak_time < entry_time:
            reasons.append("peak_time_before_entry")
        if peak_time > exit_time:
            reasons.append("peak_time_after_exit")
    expected_peak_pct = pct(peak_price, entry_price)
    expected_drop_from_peak_pct = pct(exit_price, peak_price)
    if peak_pct is None or expected_peak_pct is None or abs(float(peak_pct) - float(expected_peak_pct)) > PEAK_VALIDATION_TOLERANCE_PCT:
        reasons.append("PEAK_DROP_ALGEBRA_MISMATCH")
    if (
        drop_from_peak_pct is None
        or expected_drop_from_peak_pct is None
        or abs(float(drop_from_peak_pct) - float(expected_drop_from_peak_pct)) > PEAK_VALIDATION_TOLERANCE_PCT
    ):
        reasons.append("PEAK_DROP_ALGEBRA_MISMATCH")
    if (
        peak_pct is not None
        and gross_return_pct is not None
        and abs(float(peak_pct)) <= PEAK_VALIDATION_TOLERANCE_PCT
        and drop_from_peak_pct is not None
        and abs(float(drop_from_peak_pct) - float(gross_return_pct)) > PEAK_VALIDATION_TOLERANCE_PCT
    ):
        reasons.append("PEAK_ZERO_DROP_INCONSISTENT")
    if gross_return_pct is not None and gross_return_pct > 0:
        if peak_pct is None or peak_pct + 0.01 < gross_return_pct:
            reasons.append("profitable_long_peak_pct_below_gross_return")
        if peak_price + 1e-9 < entry_price:
            reasons.append("profitable_long_peak_price_below_entry")
    if peak_price + 1e-9 < entry_price:
        reasons.append("long_peak_price_below_entry_floor")
    if peak_price == 0:
        reasons.append("zero_peak_price_is_not_missing_data")
    unique_reasons = sorted(set(reasons))
    return ("VALID" if not unique_reasons else "INVALID", ";".join(unique_reasons))


def calculate_peak(candles: pd.DataFrame, *, trade: dict[str, Any], history_finalized: bool = True) -> PeakResult:
    entry_time = parse_dt(trade.get("entry_fill_time"))
    exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
    entry_price = fnum(trade.get("entry_price"))
    exit_price = fnum(trade.get("exit_price"))
    qty = abs(fnum(trade.get("quantity")) or 0.0)
    gross_return_pct = pct(exit_price, entry_price)
    if entry_time is None:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, 0, "", "", "MISSING_ENTRY_TIME", "FAIL", "missing_entry_time")
    if exit_time is None:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, 0, "", "", "MISSING_EXIT_TIME", "FAIL", "missing_exit_time")
    if entry_price is None or entry_price <= 0 or exit_price is None:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, 0, "", "", "NEEDS_REBUILD", "FAIL", "missing_trade_prices")
    expected = expected_candles(entry_time, exit_time)
    if candles.empty:
        quality = "MISSING_CANDLES_FINAL" if history_finalized else "RETRY_PENDING"
        reason = "missing_candles_after_history_finalized" if history_finalized else "history_not_finalized"
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, expected, "", "", quality, "OK", reason)
    rows = candles.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    rows["high"] = pd.to_numeric(rows["high"], errors="coerce")
    rows["low"] = pd.to_numeric(rows["low"], errors="coerce")
    rows = rows.dropna(subset=["timestamp", "high", "low"]).sort_values("timestamp").reset_index(drop=True)
    if rows.empty:
        quality = "MISSING_CANDLES_FINAL" if history_finalized else "RETRY_PENDING"
        reason = "empty_candles_after_history_finalized" if history_finalized else "history_not_finalized"
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, expected, "", "", quality, "OK", reason)
    min_ts = rows["timestamp"].min()
    max_ts = rows["timestamp"].max()
    window = rows[(rows["timestamp"] >= entry_time) & (rows["timestamp"] <= exit_time)].copy()
    if window.empty:
        quality = "OUTSIDE_CANDLE_RANGE" if history_finalized else "RETRY_PENDING"
        reason = "outside_candle_range" if history_finalized else "history_not_finalized"
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, expected, min_ts.isoformat(), max_ts.isoformat(), quality, "OK", reason)

    if max_ts < exit_time and not history_finalized:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, len(window), expected, min_ts.isoformat(), max_ts.isoformat(), "RETRY_PENDING", "OK", "history_not_finalized")
    quality = "EXACT" if min_ts <= entry_time and max_ts >= exit_time and len(window) >= expected else "PARTIAL_COVERAGE"
    peak_idx = pd.to_numeric(window["high"], errors="coerce").idxmax()
    low_idx = pd.to_numeric(window["low"], errors="coerce").idxmin()
    raw_peak_price = fnum(window.loc[peak_idx, "high"])
    low_price = fnum(window.loc[low_idx, "low"])
    raw_peak_time = parse_dt(window.loc[peak_idx, "timestamp"])
    low_time = parse_dt(window.loc[low_idx, "timestamp"])
    if raw_peak_price is None:
        peak_price = None
        peak_time = None
    elif raw_peak_price < entry_price:
        peak_price = entry_price
        peak_time = entry_time
    else:
        peak_price = raw_peak_price
        peak_time = raw_peak_time
    peak_pct = pct(peak_price, entry_price)
    mae_pct = pct(low_price, entry_price)
    drop_from_peak = pct(exit_price, peak_price)
    giveback_usd = (float(peak_price) - exit_price) * qty if peak_price is not None else None
    peak_upnl = (float(peak_price) - entry_price) * qty if peak_price is not None else None
    adverse_upnl = (float(low_price) - entry_price) * qty if low_price is not None else None
    validation_status, validation_reason = validate_peak(
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        peak_price=peak_price,
        peak_time=peak_time,
        peak_pct=peak_pct,
        drop_from_peak_pct=drop_from_peak,
        giveback_usd=giveback_usd,
        gross_return_pct=gross_return_pct,
    )
    if validation_status != "VALID":
        quality = "NEEDS_REBUILD"
    return PeakResult(
        peak_price=peak_price,
        peak_time=iso_ts(peak_time),
        peak_pct=peak_pct,
        low_price=low_price,
        low_time=iso_ts(low_time),
        mae_pct=mae_pct,
        drop_from_peak_pct=drop_from_peak,
        giveback_usd=giveback_usd,
        peak_unrealized_pnl=peak_upnl,
        max_adverse_unrealized_pnl=adverse_upnl,
        candle_count=len(window),
        expected_candle_count=expected,
        candles_min_time_utc=min_ts.isoformat(),
        candles_max_time_utc=max_ts.isoformat(),
        peak_data_quality=quality,
        validation_status=validation_status,
        validation_reason=validation_reason,
    )


def build_peak_rows(sqlite_path: Path, *, date: str | None, trade_id: str | None, history_dir: Path, recorder_dir: Path, session_type: str, history_finalized: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with connect_sqlite(sqlite_path, read_only=True) as conn:
        trades = selected_trade_rows(conn, date=date, trade_id=trade_id)
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for trade in trades:
        symbol = str(trade.get("symbol") or "").upper()
        entry_time = parse_dt(trade.get("entry_fill_time"))
        exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
        if entry_time is None or exit_time is None:
            candles = pd.DataFrame()
        else:
            key = (symbol, entry_time.date().isoformat(), exit_time.date().isoformat())
            if key not in cache:
                session = str(trade.get("session_date") or entry_time.date().isoformat())
                cache[key] = load_shared_trade_candles(history_dir, recorder_dir, symbol, session, entry_time, exit_time, session_type)
            candles = cache[key]
        peak = calculate_peak(candles, trade=trade, history_finalized=history_finalized)
        raw = parse_raw_json(trade.get("raw_json"))
        old_peak_price = fnum(trade.get("peak_price"))
        old_peak_pct = fnum(trade.get("mfe_pct"))
        old_giveback = fnum(trade.get("giveback_from_peak"))
        old_drop = fnum(raw.get("drop_from_peak_pct"))
        gross_return_pct = pct(fnum(trade.get("exit_price")), fnum(trade.get("entry_price")))
        stored_peak_pct = old_peak_pct
        stored_drop = old_drop
        calculated_peak_pct = pct(peak.peak_price, fnum(trade.get("entry_price")))
        calculated_drop = pct(fnum(trade.get("exit_price")), peak.peak_price)
        peak_drop_delta = None
        if stored_drop is not None and calculated_drop is not None:
            peak_drop_delta = stored_drop - calculated_drop
        zero_peak_consistent = None
        if calculated_peak_pct is not None and gross_return_pct is not None and calculated_drop is not None and abs(calculated_peak_pct) <= PEAK_VALIDATION_TOLERANCE_PCT:
            zero_peak_consistent = int(abs(calculated_drop - gross_return_pct) <= PEAK_VALIDATION_TOLERANCE_PCT)
        stale_field_mismatch = int(
            (stored_peak_pct is not None and calculated_peak_pct is not None and abs(stored_peak_pct - calculated_peak_pct) > PEAK_VALIDATION_TOLERANCE_PCT)
            or (stored_drop is not None and calculated_drop is not None and abs(stored_drop - calculated_drop) > PEAK_VALIDATION_TOLERANCE_PCT)
        )
        suspicious_zero = int((old_peak_price == 0 or old_peak_pct == 0) and raw.get("peak_data_quality") not in VALID_PEAK_QUALITIES)
        rows.append(
            {
                "trade_id": trade.get("trade_id"),
                "symbol": symbol,
                "entry_time": trade.get("entry_fill_time"),
                "exit_time": trade.get("exit_fill_time") or trade.get("closed_at"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "quantity": trade.get("quantity"),
                "gross_return_pct": gross_return_pct,
                "calculated_peak_pct": calculated_peak_pct,
                "stored_peak_pct": stored_peak_pct,
                "calculated_drop_from_peak_pct": calculated_drop,
                "stored_drop_from_peak_pct": stored_drop,
                "peak_drop_delta": peak_drop_delta,
                "zero_peak_drop_consistent": zero_peak_consistent,
                "stale_field_mismatch": stale_field_mismatch,
                "old_peak_price": old_peak_price,
                "new_peak_price": peak.peak_price,
                "peak_time": peak.peak_time,
                "peak_pct": peak.peak_pct,
                "drop_from_peak_pct": peak.drop_from_peak_pct,
                "giveback_usd": peak.giveback_usd,
                "low_price": peak.low_price,
                "mae_pct": peak.mae_pct,
                "peak_unrealized_pnl": peak.peak_unrealized_pnl,
                "max_adverse_unrealized_pnl": peak.max_adverse_unrealized_pnl,
                "candle_count": peak.candle_count,
                "expected_candle_count": peak.expected_candle_count,
                "candles_min_time_utc": peak.candles_min_time_utc,
                "candles_max_time_utc": peak.candles_max_time_utc,
                "peak_data_quality": peak.peak_data_quality,
                "validation_status": peak.validation_status,
                "validation_reason": peak.validation_reason,
                "old_peak_pct": old_peak_pct,
                "old_giveback_from_peak": old_giveback,
                "old_peak_position_key": raw.get("peak_position_key") or raw.get("position_key") or "",
                "suspicious_peak_zero": suspicious_zero,
                "would_change": int(
                    old_peak_price != peak.peak_price
                    or old_peak_pct != peak.peak_pct
                    or old_giveback != peak.giveback_usd
                    or raw.get("peak_data_quality") != peak.peak_data_quality
                ),
            }
        )
    net_values = [fnum(trade.get("net_pnl"), 0.0) or 0.0 for trade in trades]
    summary = {
        "session_date": date or "",
        "canonical_trade_count": len(rows),
        "trades_scanned": len(rows),
        "net_pnl_canonical": sum(net_values),
        "exact": sum(1 for row in rows if row["peak_data_quality"] == "EXACT"),
        "partial": sum(1 for row in rows if row["peak_data_quality"] in {"PARTIAL", "PARTIAL_COVERAGE"}),
        "peak_exact_count": sum(1 for row in rows if row["peak_data_quality"] == "EXACT"),
        "peak_partial_count": sum(1 for row in rows if row["peak_data_quality"] in {"PARTIAL", "PARTIAL_COVERAGE"}),
        "peak_retry_pending_count": sum(1 for row in rows if row["peak_data_quality"] in RETRY_PEAK_QUALITIES),
        "peak_missing_final_count": sum(1 for row in rows if row["peak_data_quality"] in FINAL_MISSING_PEAK_QUALITIES),
        "retry_pending": sum(1 for row in rows if row["peak_data_quality"] in RETRY_PEAK_QUALITIES),
        "missing_final": sum(1 for row in rows if row["peak_data_quality"] in FINAL_MISSING_PEAK_QUALITIES),
        "missing": sum(1 for row in rows if row["peak_data_quality"] in (RETRY_PEAK_QUALITIES | FINAL_MISSING_PEAK_QUALITIES)),
        "mfe_missing_count": sum(1 for row in rows if row.get("peak_pct") is None),
        "mae_missing_count": sum(1 for row in rows if row.get("mae_pct") is None),
        "peak_price_missing_count": sum(1 for row in rows if row.get("new_peak_price") is None),
        "profitable_peak_below_final_return": sum(1 for row in rows if row["validation_reason"] and "profitable_long_peak_pct_below_gross_return" in row["validation_reason"]),
        "exact_valid": sum(1 for row in rows if row["peak_data_quality"] == "EXACT" and row["validation_status"] == "VALID"),
        "exact_invalid": sum(1 for row in rows if row["peak_data_quality"] == "EXACT" and row["validation_status"] != "VALID"),
        "zero_peak_consistent": sum(1 for row in rows if row["zero_peak_drop_consistent"] == 1),
        "zero_peak_inconsistent": sum(1 for row in rows if row["zero_peak_drop_consistent"] == 0),
        "algebra_mismatch": sum(1 for row in rows if row["validation_reason"] and "PEAK_DROP_ALGEBRA_MISMATCH" in row["validation_reason"]),
        "stale_field_mismatch": sum(1 for row in rows if row["stale_field_mismatch"]),
        "old_peak_values_that_would_change": sum(int(row["would_change"]) for row in rows),
        "suspicious_peak_zero_values": sum(int(row["suspicious_peak_zero"]) for row in rows),
    }
    return rows, summary


def write_report(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, suffix: str, *, audit: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "trade_peak_audit" if audit else "trade_peak_rebuild_dry_run"
    path = output_dir / f"{name}_{suffix}.csv"
    fieldnames = [
        "trade_id",
        "symbol",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "gross_return_pct",
        "calculated_peak_pct",
        "stored_peak_pct",
        "calculated_drop_from_peak_pct",
        "stored_drop_from_peak_pct",
        "peak_drop_delta",
        "zero_peak_drop_consistent",
        "stale_field_mismatch",
        "old_peak_price",
        "new_peak_price",
        "peak_time",
        "peak_pct",
        "drop_from_peak_pct",
        "giveback_usd",
        "candle_count",
        "expected_candle_count",
        "peak_data_quality",
        "validation_status",
        "validation_reason",
        "old_peak_pct",
        "old_giveback_from_peak",
        "old_peak_position_key",
        "suspicious_peak_zero",
        "would_change",
        "candles_min_time_utc",
        "candles_max_time_utc",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    summary_path = output_dir / f"{name}_{suffix}_summary.md"
    summary_path.write_text("```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
    return path


def backup_path(sqlite_path: Path) -> Path:
    return sqlite_path.with_suffix(sqlite_path.suffix + f".backup_peaks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")


def update_trade_peak(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    trade_id = str(row.get("trade_id") or "")
    existing = conn.execute("SELECT raw_json FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    raw = parse_raw_json(existing["raw_json"] if existing else None)
    for key in (
        "peak_gain_pct",
        "max_gain_pct",
        "peak_unrealized_pct",
        "giveback_pct",
        "peak_position_key",
        "position_key",
    ):
        raw.pop(key, None)
    if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID":
        rebuild_status = "rebuilt_from_candles"
    elif row["peak_data_quality"] in RETRY_PEAK_QUALITIES:
        rebuild_status = "retry_pending"
    else:
        rebuild_status = "needs_rebuild"
    raw.update(
        {
            "peak_rebuild_status": rebuild_status,
            "peak_rebuild_version": PEAK_REBUILD_VERSION,
            "peak_version": PEAK_REBUILD_VERSION,
            "peak_data_quality": row["peak_data_quality"],
            "peak_source": "canonical_trade_candles_1m" if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else "unavailable",
            "peak_time": row["peak_time"] if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else "",
            "peak_pct": row["peak_pct"] if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else None,
            "mfe_pct": row["peak_pct"] if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else None,
            "mae_pct": row["mae_pct"] if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else None,
            "drop_from_peak_pct": row["drop_from_peak_pct"] if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else None,
            "giveback_usd": row["giveback_usd"] if row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID" else None,
            "peak_validation_status": row["validation_status"],
            "peak_validation_reason": row["validation_reason"],
            "stale_peak_position_key_ignored": True,
            "peak_calculated_at": datetime.now(timezone.utc).isoformat(),
            "peak_rebuilt_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    peak_ok = row["peak_data_quality"] in VALID_PEAK_QUALITIES and row["validation_status"] == "VALID"
    conn.execute(
        """
        UPDATE trades
        SET mfe_pct = ?,
            mae_pct = ?,
            peak_price = ?,
            low_price = ?,
            peak_unrealized_pnl = ?,
            max_adverse_unrealized_pnl = ?,
            giveback_from_peak = ?,
            raw_json = ?,
            updated_at = ?
        WHERE trade_id = ?
        """,
        (
            row["peak_pct"] if peak_ok else None,
            row["mae_pct"] if peak_ok else None,
            row["new_peak_price"] if peak_ok else None,
            row["low_price"] if peak_ok else None,
            row["peak_unrealized_pnl"] if peak_ok else None,
            row["max_adverse_unrealized_pnl"] if peak_ok else None,
            row["giveback_usd"] if peak_ok else None,
            json.dumps(raw, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
            trade_id,
        ),
    )


def apply_rows(sqlite_path: Path, rows: list[dict[str, Any]], *, backup_to: Path | None = None) -> dict[str, Any]:
    enforce_apply_guard(sqlite_path)
    assert_sqlite_integrity_ok(sqlite_path, label="target_before_apply")
    backup = backup_to or backup_path(sqlite_path)
    create_online_backup(sqlite_path, backup)
    updated = 0
    with connect_sqlite(sqlite_path, read_only=False) as conn:
        try:
            conn.execute("BEGIN")
            for row in rows:
                update_trade_peak(conn, row)
                updated += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {"backup_path": str(backup), "updated_trades": updated}


def run(args: argparse.Namespace, *, audit: bool = False) -> int:
    if not args.date and not args.trade_id:
        raise SystemExit("--date or --trade-id is required")
    sqlite_path = Path(args.sqlite_path)
    rows, summary = build_peak_rows(
        sqlite_path,
        date=args.date,
        trade_id=args.trade_id,
        history_dir=Path(args.history_dir),
        recorder_dir=Path(args.recorder_dir),
        session_type=args.session_type,
        history_finalized=bool(args.history_finalized),
    )
    suffix = args.trade_id or args.date or "all"
    output = write_report(rows, summary, Path(args.output_dir), suffix, audit=audit)
    event = "TRADE_PEAK_AUDIT_DONE" if audit else "TRADE_PEAK_REBUILD_DRY_RUN"
    print(
        f"{event} trades_scanned={summary['trades_scanned']} exact={summary['exact']} "
        f"partial={summary['partial']} retry_pending={summary.get('retry_pending', 0)} "
        f"missing_final={summary.get('missing_final', 0)} missing={summary['missing']} "
        f"profitable_peak_below_final_return={summary['profitable_peak_below_final_return']} "
        f"suspicious_peak_zero_values={summary['suspicious_peak_zero_values']} output={output}",
        flush=True,
    )
    if getattr(args, "apply", False):
        result = apply_rows(sqlite_path, rows, backup_to=Path(args.backup_to) if args.backup_to else None)
        print("TRADE_PEAK_REBUILD_APPLIED " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--date")
    parser.add_argument("--trade-id")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_DB))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--recorder-dir", default="data/live/recorder")
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Explicit no-op alias; dry-run is the default.")
    parser.add_argument("--history-finalized", action="store_true", help="Allow final MISSING_CANDLES_FINAL/OUTSIDE states after history has been finalized.")
    parser.add_argument("--apply", action="store_true", help="Apply changes to SQLite. Default is dry-run. Refuses production DB while v67 trader is active.")
    parser.add_argument("--backup-to", help="Create a consistent SQLite online backup at this path before --apply. Never use plain cp on a live WAL database.")
    parser.epilog = (
        'Safe live-copy example: sqlite3 data/runtime/trading_runtime.sqlite "'
        '.backup data/runtime/trading_runtime.peak_test.sqlite"\n'
        "Python/apply example: python scripts/rebuild_trade_peaks.py --date 2026-07-20 "
        "--sqlite-path data/runtime/trading_runtime.peak_test.sqlite --history-finalized --apply "
        "--backup-to data/runtime/trading_runtime.peak_test.pre_peak_apply.sqlite"
    )
    return parser


def main() -> int:
    return run(build_parser("Rebuild canonical trade peak/giveback metrics from 1m candles.").parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
