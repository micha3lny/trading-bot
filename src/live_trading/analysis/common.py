from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.live_trading.ranking.daily_top100_builder import normalize_history_df, parquet_path


RTH_OPEN_UTC = time(13, 30)
EOD_FLATTEN_UTC = time(19, 45)
NY_TZ = ZoneInfo("America/New_York")


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


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def pct(price: float | None, base: float | None) -> float | None:
    if price is None or base is None or base <= 0:
        return None
    return ((float(price) / float(base)) - 1.0) * 100.0


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


def sqlite_connect_readonly(sqlite_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(sqlite_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def read_sql_table(
    sqlite_path: str | Path,
    table: str,
    *,
    columns: Iterable[str] | None = None,
    where: str = "",
    params: Iterable[Any] = (),
    order_by: str = "",
) -> pd.DataFrame:
    path = Path(sqlite_path)
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite_connect_readonly(path)
    try:
        available = table_columns(conn, table)
        if not available:
            return pd.DataFrame()
        selected = [col for col in (columns or sorted(available)) if col in available]
        if not selected:
            selected = sorted(available)
        sql = f"SELECT {', '.join(selected)} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        return pd.read_sql_query(sql, conn, params=list(params))
    finally:
        conn.close()


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def load_universe_symbols(path: str | Path) -> list[str]:
    df = safe_read_csv(path)
    if df.empty or "symbol" not in df.columns:
        return []
    symbols = df["symbol"].map(normalize_symbol).dropna().drop_duplicates().tolist()
    return [symbol for symbol in symbols if symbol and symbol != "NAN"]


def load_top100(path: str | Path) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "top100_rank", "top100_score"])
    out = df.copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    if "top100_rank" not in out.columns:
        rank_col = next((col for col in ["rank", "ranking_position", "position"] if col in out.columns), None)
        out["top100_rank"] = pd.to_numeric(out[rank_col], errors="coerce") if rank_col else range(1, len(out) + 1)
    score_col = next((col for col in ["top100_score", "score", "final_score", "alpha_score"] if col in out.columns), None)
    out["top100_score"] = pd.to_numeric(out[score_col], errors="coerce") if score_col else pd.NA
    return out.drop_duplicates("symbol")


def load_session_candles(history_dir: str | Path, symbol: str, session_date: str | date, session_type: str = "RTH") -> pd.DataFrame:
    session = pd.Timestamp(session_date).date() if not isinstance(session_date, date) else session_date
    path = parquet_path(history_dir, symbol, session, session_type)
    if not path.exists():
        return pd.DataFrame()
    try:
        return normalize_history_df(pd.read_parquet(path))
    except Exception:
        return pd.DataFrame()


def load_recorder_candles(recorder_dir: str | Path, session_date: str, symbol: str | None = None) -> pd.DataFrame:
    root = Path(recorder_dir) / session_date
    candidates = [
        root / "candles_1m.csv",
        root / "market_data_1m.csv",
        root / "bars_1m.csv",
    ]
    for path in candidates:
        df = safe_read_csv(path)
        if df.empty:
            continue
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
        try:
            if "timestamp" not in {str(c).strip().lower() for c in df.columns}:
                lower = {str(c).strip().lower(): c for c in df.columns}
                for candidate in ("bar_time", "bar_time_utc", "time", "datetime"):
                    if candidate in lower:
                        df = df.rename(columns={lower[candidate]: "timestamp"})
                        break
            return normalize_history_df(df)
        except Exception:
            continue
    return pd.DataFrame()


def load_trade_candles(
    history_dir: str | Path,
    recorder_dir: str | Path,
    symbol: str,
    session_date: str,
    entry_time: pd.Timestamp | None,
    exit_time: pd.Timestamp | None,
    session_type: str = "RTH",
) -> pd.DataFrame:
    recorder = load_recorder_candles(recorder_dir, session_date, symbol)
    candles = recorder if not recorder.empty else load_session_candles(history_dir, symbol, session_date, session_type)
    if candles.empty:
        return candles
    if entry_time is not None:
        candles = candles[candles["timestamp"] >= entry_time]
    if exit_time is not None:
        candles = candles[candles["timestamp"] <= exit_time]
    return candles.sort_values("timestamp").reset_index(drop=True)


def rth_open_timestamp(session_date: str | date) -> pd.Timestamp:
    d = pd.Timestamp(session_date).date() if not isinstance(session_date, date) else session_date
    return pd.Timestamp(datetime.combine(d, time(9, 30), tzinfo=NY_TZ)).tz_convert("UTC")


def eod_timestamp(session_date: str | date) -> pd.Timestamp:
    d = pd.Timestamp(session_date).date() if not isinstance(session_date, date) else session_date
    return pd.Timestamp(datetime.combine(d, EOD_FLATTEN_UTC, tzinfo=timezone.utc))


@dataclass(frozen=True)
class RunnerStats:
    open_price: float
    high_price: float
    high_time: pd.Timestamp
    open_to_high_pct: float
    first_5m_high_pct: float | None
    first_15m_high_pct: float | None
    or_range_pct: float | None


def calculate_runner_stats(candles: pd.DataFrame) -> RunnerStats | None:
    if candles.empty:
        return None
    rows = candles.sort_values("timestamp").reset_index(drop=True)
    open_price = fnum(rows.iloc[0].get("open"))
    if open_price is None or open_price <= 0:
        return None
    high_idx = pd.to_numeric(rows["high"], errors="coerce").idxmax()
    high_price = float(rows.loc[high_idx, "high"])
    high_time = rows.loc[high_idx, "timestamp"]
    start = rows.iloc[0]["timestamp"]
    first5 = rows[rows["timestamp"] < start + pd.Timedelta(minutes=5)]
    first15 = rows[rows["timestamp"] < start + pd.Timedelta(minutes=15)]
    or_high = fnum(first15["high"].max()) if not first15.empty else None
    or_low = fnum(first15["low"].min()) if not first15.empty else None
    return RunnerStats(
        open_price=open_price,
        high_price=high_price,
        high_time=high_time,
        open_to_high_pct=pct(high_price, open_price) or 0.0,
        first_5m_high_pct=pct(fnum(first5["high"].max()) if not first5.empty else None, open_price),
        first_15m_high_pct=pct(or_high, open_price),
        or_range_pct=((or_high - or_low) / open_price * 100.0) if or_high is not None and or_low is not None else None,
    )


@dataclass(frozen=True)
class PathStats:
    peak_price: float | None
    low_price: float | None
    peak_time: pd.Timestamp | None
    low_time: pd.Timestamp | None
    mfe_pct: float | None
    mae_pct: float | None
    max_drawdown_from_peak_pct: float | None
    time_to_peak_seconds: float | None
    time_to_low_seconds: float | None


def calculate_path_stats(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None = None) -> PathStats:
    if candles.empty or entry_price <= 0:
        return PathStats(None, None, None, None, None, None, None, None, None)
    rows = candles.sort_values("timestamp").reset_index(drop=True)
    peak_idx = pd.to_numeric(rows["high"], errors="coerce").idxmax()
    low_idx = pd.to_numeric(rows["low"], errors="coerce").idxmin()
    raw_peak_price = float(rows.loc[peak_idx, "high"])
    low_price = float(rows.loc[low_idx, "low"])
    raw_peak_time = rows.loc[peak_idx, "timestamp"]
    if raw_peak_price < entry_price:
        peak_price = float(entry_price)
        peak_time = entry_time or rows.iloc[0]["timestamp"]
    else:
        peak_price = raw_peak_price
        peak_time = raw_peak_time
    low_time = rows.loc[low_idx, "timestamp"]
    after_peak = rows[rows["timestamp"] >= peak_time]
    post_peak_low = fnum(after_peak["low"].min()) if not after_peak.empty else low_price
    return PathStats(
        peak_price=peak_price,
        low_price=low_price,
        peak_time=peak_time,
        low_time=low_time,
        mfe_pct=pct(peak_price, entry_price),
        mae_pct=pct(low_price, entry_price),
        max_drawdown_from_peak_pct=pct(post_peak_low, peak_price),
        time_to_peak_seconds=((peak_time - entry_time).total_seconds() if entry_time is not None else None),
        time_to_low_seconds=((low_time - entry_time).total_seconds() if entry_time is not None else None),
    )


def min_after_pct(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, minutes: int) -> float | None:
    if candles.empty or entry_time is None or entry_price <= 0:
        return None
    window = candles[(candles["timestamp"] >= entry_time) & (candles["timestamp"] <= entry_time + pd.Timedelta(minutes=minutes))]
    if window.empty:
        return None
    return pct(float(window["low"].min()), entry_price)


def entry_time_bucket(entry_time: pd.Timestamp | None, session_date: str | date | None = None) -> str:
    if entry_time is None:
        return "missing"
    open_ts = rth_open_timestamp(session_date or entry_time.date())
    minutes = (entry_time - open_ts).total_seconds() / 60.0
    if minutes < 5:
        return "before_09:35 ET"
    if minutes < 10:
        return "09:35-09:40 ET"
    if minutes < 15:
        return "09:40-09:45 ET"
    if minutes < 30:
        return "09:45-10:00 ET"
    return "10:00+ ET"


def entry_minutes_after_open(entry_time: pd.Timestamp | None, session_date: str | date | None = None) -> float | None:
    if entry_time is None:
        return None
    return (entry_time - rth_open_timestamp(session_date or entry_time.date())).total_seconds() / 60.0


@dataclass(frozen=True)
class TPSLSimulation:
    exit_reason: str
    exit_time: pd.Timestamp | None
    exit_price: float | None
    pnl_pct: float | None


def simulate_tp_sl(
    candles: pd.DataFrame,
    *,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    fallback_exit_time: pd.Timestamp | None = None,
    fallback_exit_price: float | None = None,
) -> TPSLSimulation:
    if entry_price <= 0:
        return TPSLSimulation("missing_entry", None, None, None)
    if candles.empty or "timestamp" not in candles.columns:
        if fallback_exit_price is not None:
            return TPSLSimulation("actual_exit", fallback_exit_time, fallback_exit_price, pct(fallback_exit_price, entry_price))
        return TPSLSimulation("missing_candles", fallback_exit_time, None, None)
    tp_price = entry_price * (1.0 + tp_pct / 100.0)
    sl_price = entry_price * (1.0 + sl_pct / 100.0)
    for _, row in candles.sort_values("timestamp").iterrows():
        low = fnum(row.get("low"))
        high = fnum(row.get("high"))
        ts = row.get("timestamp")
        # Conservative assumption for long trades: if both happen in one minute,
        # count the stop first because intraminute order is unknown.
        if low is not None and low <= sl_price:
            return TPSLSimulation(f"SL {sl_pct:g}%", ts, sl_price, pct(sl_price, entry_price))
        if high is not None and high >= tp_price:
            return TPSLSimulation(f"TP {tp_pct:g}%", ts, tp_price, pct(tp_price, entry_price))
    if fallback_exit_price is not None:
        return TPSLSimulation("actual_exit", fallback_exit_time, fallback_exit_price, pct(fallback_exit_price, entry_price))
    return TPSLSimulation("no_exit", fallback_exit_time, None, None)


def nearest_row(df: pd.DataFrame, timestamp: pd.Timestamp | None, symbol: str | None = None, tolerance_seconds: int = 300) -> dict[str, Any]:
    if df.empty or timestamp is None:
        return {}
    rows = df.copy()
    if symbol and "symbol" in rows.columns:
        rows = rows[rows["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
    time_col = next((col for col in ["timestamp", "time", "event_time", "recorded_at", "created_at", "ts"] if col in rows.columns), None)
    if time_col is None or rows.empty:
        return {}
    rows["_ts"] = pd.to_datetime(rows[time_col], errors="coerce", utc=True)
    rows["_delta"] = (rows["_ts"] - timestamp).abs().dt.total_seconds()
    rows = rows.dropna(subset=["_delta"]).sort_values("_delta")
    if rows.empty or float(rows.iloc[0]["_delta"]) > tolerance_seconds:
        return {}
    return rows.iloc[0].to_dict()


def first_existing_column(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, "") and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None
