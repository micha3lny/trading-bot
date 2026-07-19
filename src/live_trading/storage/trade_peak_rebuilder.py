from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_HISTORY_DIR = Path(os.environ.get("TRADING_BOT_TRADE_PEAK_HISTORY_DIR", "data/history/universe_1m"))
DEFAULT_SESSION_TYPE = os.environ.get("TRADING_BOT_TRADE_PEAK_SESSION_TYPE", "RTH")
PEAK_VERSION = 2
VALID_PEAK_QUALITIES = {"EXACT", "PARTIAL"}
MISSING_PEAK_QUALITIES = {"MISSING_CANDLES", "OUTSIDE_CANDLE_RANGE", "NEEDS_REBUILD"}
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


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_dt(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


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


def history_parquet_path(history_dir: Path, symbol: str, session_date: Any, session_type: str) -> Path:
    day = pd.Timestamp(session_date).date()
    return (
        history_dir
        / f"session_type={session_type}"
        / f"symbol={str(symbol or '').upper()}"
        / f"year={day.year:04d}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}.parquet"
    )


def normalize_candle_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    lower = {str(col).strip().lower(): col for col in df.columns}
    timestamp_col = lower.get("timestamp") or lower.get("bar_time") or lower.get("time") or lower.get("datetime")
    high_col = lower.get("high")
    low_col = lower.get("low")
    if not timestamp_col or not high_col or not low_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    out["high"] = pd.to_numeric(df[high_col], errors="coerce")
    out["low"] = pd.to_numeric(df[low_col], errors="coerce")
    return out.dropna(subset=["timestamp", "high", "low"]).sort_values("timestamp").reset_index(drop=True)


def expected_candles(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    seconds = max(0.0, (exit_time - entry_time).total_seconds())
    return int(seconds // 60) + 1


def load_session_candles(history_dir: Path, symbol: str, session_date: str, session_type: str = DEFAULT_SESSION_TYPE) -> pd.DataFrame:
    path = history_parquet_path(history_dir, symbol, session_date, session_type)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=["timestamp", "high", "low"])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return normalize_candle_frame(df)


def load_trade_candles(history_dir: Path, symbol: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp, session_type: str = DEFAULT_SESSION_TYPE) -> pd.DataFrame:
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


def calculate_peak(candles: pd.DataFrame, *, trade: dict[str, Any]) -> PeakResult:
    entry_time = parse_dt(trade.get("entry_fill_time"))
    exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
    entry_price = fnum(trade.get("entry_price"))
    exit_price = fnum(trade.get("exit_price"))
    qty = abs(fnum(trade.get("quantity")) or 0.0)
    gross_return_pct = pct(exit_price, entry_price)
    if entry_time is None or exit_time is None or entry_price is None or entry_price <= 0 or exit_price is None:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, 0, "", "", "NEEDS_REBUILD", "FAIL", "missing_trade_inputs")
    expected = expected_candles(entry_time, exit_time)
    if candles.empty:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, expected, "", "", "MISSING_CANDLES", "OK", "")
    rows = candles.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    rows["high"] = pd.to_numeric(rows["high"], errors="coerce")
    rows["low"] = pd.to_numeric(rows["low"], errors="coerce")
    rows = rows.dropna(subset=["timestamp", "high", "low"]).sort_values("timestamp").reset_index(drop=True)
    if rows.empty:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, expected, "", "", "MISSING_CANDLES", "OK", "")
    min_ts = rows["timestamp"].min()
    max_ts = rows["timestamp"].max()
    window = rows[(rows["timestamp"] >= entry_time) & (rows["timestamp"] <= exit_time)].copy()
    if window.empty:
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, expected, min_ts.isoformat(), max_ts.isoformat(), "OUTSIDE_CANDLE_RANGE", "OK", "")

    quality = "EXACT" if min_ts <= entry_time and max_ts >= exit_time and len(window) >= expected else "PARTIAL"
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


def calculate_peak_for_trade(trade: dict[str, Any], *, history_dir: Path = DEFAULT_HISTORY_DIR, session_type: str = DEFAULT_SESSION_TYPE) -> PeakResult:
    entry_time = parse_dt(trade.get("entry_fill_time"))
    exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
    symbol = str(trade.get("symbol") or "").upper()
    if entry_time is None or exit_time is None or not symbol:
        return calculate_peak(pd.DataFrame(), trade=trade)
    return calculate_peak(load_trade_candles(history_dir, symbol, entry_time, exit_time, session_type), trade=trade)


def peak_raw_payload(result: PeakResult) -> dict[str, Any]:
    peak_ok = result.peak_data_quality in VALID_PEAK_QUALITIES and result.validation_status == "VALID"
    return {
        "peak_rebuild_status": "rebuilt_from_candles" if result.peak_data_quality in VALID_PEAK_QUALITIES else "needs_rebuild",
        "peak_data_quality": result.peak_data_quality,
        "peak_source": "canonical_trade_candles_1m" if peak_ok else "unavailable",
        "peak_version": PEAK_VERSION,
        "peak_rebuild_version": PEAK_VERSION,
        "peak_calculated_at": datetime.now(timezone.utc).isoformat(),
        "peak_time": result.peak_time if peak_ok else "",
        "low_time": result.low_time if peak_ok else "",
        "peak_pct": result.peak_pct if peak_ok else None,
        "mfe_pct": result.peak_pct if peak_ok else None,
        "mae_pct": result.mae_pct if peak_ok else None,
        "drop_from_peak_pct": result.drop_from_peak_pct if peak_ok else None,
        "giveback_usd": result.giveback_usd if peak_ok else None,
        "peak_validation_status": result.validation_status,
        "peak_validation_reason": result.validation_reason,
        "stale_peak_position_key_ignored": True,
    }


def update_trade_peak(conn: sqlite3.Connection, trade_id: str, result: PeakResult) -> None:
    row = conn.execute("SELECT raw_json FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    raw = parse_raw_json(row["raw_json"] if row else None)
    for key in (
        "peak_gain_pct",
        "max_gain_pct",
        "peak_unrealized_pct",
        "giveback_pct",
        "peak_position_key",
        "position_key",
    ):
        raw.pop(key, None)
    raw.update(peak_raw_payload(result))
    peak_ok = result.peak_data_quality in VALID_PEAK_QUALITIES and result.validation_status == "VALID"
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
            result.peak_pct if peak_ok else None,
            result.mae_pct if peak_ok else None,
            result.peak_price if peak_ok else None,
            result.low_price if peak_ok else None,
            result.peak_unrealized_pnl if peak_ok else None,
            result.max_adverse_unrealized_pnl if peak_ok else None,
            result.giveback_usd if peak_ok else None,
            json.dumps(raw, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
            trade_id,
        ),
    )


def mark_trade_peak_needs_rebuild(conn: sqlite3.Connection, trade_id: str, reason: str) -> None:
    row = conn.execute("SELECT raw_json FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    raw = parse_raw_json(row["raw_json"] if row else None)
    for key in (
        "peak_gain_pct",
        "max_gain_pct",
        "peak_unrealized_pct",
        "giveback_pct",
        "peak_position_key",
        "position_key",
        "peak_pct",
        "mfe_pct",
        "mae_pct",
        "drop_from_peak_pct",
        "giveback_usd",
    ):
        raw.pop(key, None)
    raw.update(
        {
            "peak_rebuild_status": "needs_rebuild",
            "peak_data_quality": "NEEDS_REBUILD",
            "peak_source": "unavailable",
            "peak_version": PEAK_VERSION,
            "peak_rebuild_version": PEAK_VERSION,
            "peak_calculated_at": datetime.now(timezone.utc).isoformat(),
            "peak_error": reason,
        }
    )
    conn.execute(
        """
        UPDATE trades
        SET mfe_pct = NULL,
            mae_pct = NULL,
            peak_price = NULL,
            low_price = NULL,
            peak_unrealized_pnl = NULL,
            max_adverse_unrealized_pnl = NULL,
            giveback_from_peak = NULL,
            raw_json = ?,
            updated_at = ?
        WHERE trade_id = ?
        """,
        (json.dumps(raw, sort_keys=True), datetime.now(timezone.utc).isoformat(), trade_id),
    )


def trade_row(conn: sqlite3.Connection, trade_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT trade_id, session_date, symbol, entry_fill_time, exit_fill_time, closed_at,
               entry_price, exit_price, quantity, raw_json
        FROM trades
        WHERE trade_id = ?
        """,
        (trade_id,),
    ).fetchone()
    return dict(row) if row else None


def calculate_and_store_trade_peak(
    conn: sqlite3.Connection,
    trade_id: str,
    *,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    session_type: str = DEFAULT_SESSION_TYPE,
) -> PeakResult:
    row = trade_row(conn, trade_id)
    if not row:
        raise ValueError(f"trade_id not found: {trade_id}")
    try:
        result = calculate_peak_for_trade(row, history_dir=history_dir, session_type=session_type)
        update_trade_peak(conn, trade_id, result)
        return result
    except Exception as exc:
        mark_trade_peak_needs_rebuild(conn, trade_id, repr(exc))
        return PeakResult(None, "", None, None, "", None, None, None, None, None, 0, 0, "", "", "NEEDS_REBUILD", "FAIL", repr(exc))
