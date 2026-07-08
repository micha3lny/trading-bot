from __future__ import annotations

import argparse
import glob
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_runner_stats,
    first_existing_column,
    fnum,
    iso_ts,
    load_session_candles,
    load_top100,
    load_universe_symbols,
    nearest_row,
    normalize_symbol,
    pct,
    read_sql_table,
    safe_read_csv,
)
from src.live_trading.market_calendar import previous_us_equity_trading_day
from src.live_trading.ranking.daily_top100_builder import normalize_history_df


DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_UNIVERSE = Path("data/universe/v68_final_daytrading_universe.csv")
DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")
NEEDED_PARQUET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "source_bucket",
    "top100_rank",
    "top100_score",
    "open",
    "high",
    "high_time",
    "open_to_high_pct",
    "was_bought",
    "entry_time",
    "entry_price",
    "entry_vs_open_pct",
    "entry_vs_high_pct",
    "missed_reason_group",
    "rejection_reason",
    "blocked_reason",
    "signal_time",
    "ready_since",
    "candidate_age_seconds",
    "live_entry_score",
    "live_entry_rank",
    "spread_bps_near_entry",
    "first_5m_high_pct",
    "first_15m_high_pct",
    "or_range_pct",
    "was_detectable_from_history",
    "detectability_reason",
    "prev_1d_return_pct",
    "prev_2d_return_pct",
    "prev_3d_return_pct",
    "prev_5d_return_pct",
    "prev_1d_intraday_high_pct",
    "prev_3d_max_intraday_high_pct",
    "prev_5d_max_intraday_high_pct",
    "prev_3d_avg_volume",
    "prev_5d_avg_volume",
    "prev_3d_relative_volume_like",
    "prev_5d_relative_volume_like",
    "hypothetical_multiday_score",
    "hypothetical_multiday_rank",
    "would_enter_multiday_top100",
    "top100_no_signal_reason",
    "first_time_above_5pct",
    "first_time_above_8pct",
    "opening_range_high_pct",
    "opening_range_low_pct",
    "opening_range_break_time",
    "did_break_or_high",
    "had_required_first5",
    "had_required_first15",
    "had_required_or_range",
    "possible_signal_time",
]


def dated_history_glob(history_dir: Path, session_date: str, session_type: str = "RTH") -> str:
    d = pd.Timestamp(session_date).date()
    return str(
        Path(history_dir)
        / f"session_type={session_type.upper()}"
        / "symbol=*"
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / f"day={d.day:02d}.parquet"
    )


def symbol_from_parquet_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("symbol="):
            return normalize_symbol(part.split("=", 1)[1])
    return ""


def find_history_files(history_dir: Path, session_date: str, session_type: str = "RTH") -> dict[str, Path]:
    files = sorted(Path(value) for value in glob.glob(dated_history_glob(history_dir, session_date, session_type)))
    out: dict[str, Path] = {}
    for path in files:
        symbol = symbol_from_parquet_path(path)
        if symbol:
            out[symbol] = path
    return out


def read_history_for_missed(path: Path) -> pd.DataFrame:
    try:
        try:
            raw = pd.read_parquet(path, columns=NEEDED_PARQUET_COLUMNS)
        except Exception:
            raw = pd.read_parquet(path)
        return normalize_history_df(raw)
    except Exception:
        return pd.DataFrame()


def load_recorder_table(recorder_dir: Path, session_date: str, names: list[str]) -> pd.DataFrame:
    root = recorder_dir / session_date
    for name in names:
        for path in [root / name, root / f"{name}.csv", root / f"{name}.jsonl"]:
            if path.suffix == ".jsonl" and path.exists():
                try:
                    return pd.read_json(path, lines=True)
                except Exception:
                    return pd.DataFrame()
            df = safe_read_csv(path)
            if not df.empty:
                return df
    return pd.DataFrame()


def load_entries(sqlite_path: str | Path, session_date: str) -> pd.DataFrame:
    trades = read_sql_table(
        sqlite_path,
        "trades",
        where=(
            "session_date = ? OR substr(entry_fill_time, 1, 10) = ? "
            "OR substr(exit_fill_time, 1, 10) = ? OR substr(closed_at, 1, 10) = ?"
        ),
        params=[session_date, session_date, session_date, session_date],
    )
    executions = read_sql_table(
        sqlite_path,
        "executions",
        where="session_date = ? OR substr(executed_at, 1, 10) = ? OR substr(recorded_at, 1, 10) = ?",
        params=[session_date, session_date, session_date],
    )
    rows: list[dict[str, Any]] = []
    if not trades.empty:
        for row in trades.to_dict("records"):
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "entry_time": first_existing_column(row, ["entry_fill_time", "entry_time", "opened_at"]),
                "entry_price": first_existing_column(row, ["entry_price", "avg_price"]),
                "live_entry_score": first_existing_column(row, ["live_entry_score", "entry_score", "score"]),
                "live_entry_rank": first_existing_column(row, ["live_entry_rank", "ranking_position"]),
            })
    if not executions.empty:
        side = executions.get("side", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
        buys = executions[side.isin(["BOT", "BUY", "BOUGHT"])].copy()
        for row in buys.to_dict("records"):
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "entry_time": first_existing_column(row, ["executed_at", "recorded_at"]),
                "entry_price": row.get("price"),
                "live_entry_score": None,
                "live_entry_rank": None,
            })
    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_time", "entry_price", "live_entry_score", "live_entry_rank"])
    out = pd.DataFrame(rows)
    out["entry_ts"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["entry_price_num"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out = out.sort_values(["symbol", "entry_ts"], na_position="last").drop_duplicates("symbol", keep="first")
    return out


def symbol_event_row(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    if df.empty or "symbol" not in df.columns:
        return {}
    rows = df[df["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()


def text_contains(value: Any, needle: str) -> bool:
    return needle.lower() in str(value or "").lower()


def classify_missed_reason(
    *,
    source_bucket: str,
    was_bought: bool,
    entry_time: pd.Timestamp | None,
    high_time: pd.Timestamp | None,
    signal_row: dict[str, Any],
    order_row: dict[str, Any],
) -> str:
    combined = " ".join(str(v) for v in [*signal_row.values(), *order_row.values()] if v not in (None, ""))
    if was_bought:
        if entry_time is not None and high_time is not None and entry_time > high_time:
            return "bought_late"
        return "bought"
    if source_bucket == "outside_top100":
        return "not_in_top100"
    if text_contains(combined, "spread"):
        return "spread_too_wide"
    if text_contains(combined, "risk_guard"):
        return "risk_guard_blocked"
    if text_contains(combined, "max_position") or text_contains(combined, "max positions"):
        return "max_positions_blocked"
    if text_contains(combined, "candidate_age") or text_contains(combined, "stale"):
        return "stale_candidate"
    if text_contains(combined, "failed") or text_contains(combined, "rejected") or text_contains(combined, "error"):
        return "order_failed"
    signal_time = first_existing_column(signal_row, ["signal_time", "ready_since", "timestamp", "event_time"])
    signal_ts = pd.to_datetime(signal_time, errors="coerce", utc=True)
    if high_time is not None and not pd.isna(signal_ts) and signal_ts > high_time:
        return "signal_too_late"
    if not signal_row:
        return "no_signal"
    return "unknown"


def prior_trading_days(session_date: str, count: int) -> list[pd.Timestamp]:
    days: list[pd.Timestamp] = []
    cur = pd.Timestamp(session_date).date()
    for _ in range(count):
        cur = previous_us_equity_trading_day(cur)
        days.append(pd.Timestamp(cur))
    return days


def previous_session_context(history_dir: Path, symbol: str, session_date: str) -> dict[str, Any]:
    days = prior_trading_days(session_date, 5)
    sessions: list[dict[str, Any]] = []
    for day in days:
        candles = load_session_candles(history_dir, symbol, day.date())
        stats = calculate_runner_stats(candles)
        if stats is None or candles.empty:
            sessions.append({})
            continue
        close = fnum(candles.iloc[-1].get("close"))
        volume = fnum(candles.get("volume", pd.Series(dtype=float)).sum())
        sessions.append({
            "date": day.strftime("%F"),
            "open": stats.open_price,
            "close": close,
            "intraday_high_pct": stats.open_to_high_pct,
            "volume": volume,
        })

    def ret(n: int) -> float | None:
        available = [s for s in sessions[:n] if s.get("open") and s.get("close")]
        if not available:
            return None
        newest_close = fnum(available[0].get("close"))
        oldest_open = fnum(available[-1].get("open"))
        return pct(newest_close, oldest_open)

    highs = [fnum(s.get("intraday_high_pct")) for s in sessions if s]
    volumes = [fnum(s.get("volume")) for s in sessions if s]
    prev_3_vol = [v for v in volumes[:3] if v is not None]
    prev_5_vol = [v for v in volumes[:5] if v is not None]
    prev_1d_return = ret(1)
    prev_2d_return = ret(2)
    prev_3d_return = ret(3)
    prev_5d_return = ret(5)
    prev_1d_high = highs[0] if highs else None
    prev_3d_high = max([h for h in highs[:3] if h is not None], default=None)
    prev_5d_high = max([h for h in highs[:5] if h is not None], default=None)
    reasons: list[str] = []
    if prev_1d_return is not None and prev_1d_return >= 3:
        reasons.append("prev_1d_return>=3")
    if prev_3d_return is not None and prev_3d_return >= 5:
        reasons.append("prev_3d_return>=5")
    if prev_5d_return is not None and prev_5d_return >= 8:
        reasons.append("prev_5d_return>=8")
    if prev_3d_high is not None and prev_3d_high >= 6:
        reasons.append("prev_3d_high>=6")
    if prev_5d_high is not None and prev_5d_high >= 8:
        reasons.append("prev_5d_high>=8")
    return {
        "was_detectable_from_history": int(bool(reasons)),
        "detectability_reason": ",".join(reasons),
        "prev_1d_return_pct": prev_1d_return,
        "prev_2d_return_pct": prev_2d_return,
        "prev_3d_return_pct": prev_3d_return,
        "prev_5d_return_pct": prev_5d_return,
        "prev_1d_intraday_high_pct": prev_1d_high,
        "prev_3d_max_intraday_high_pct": prev_3d_high,
        "prev_5d_max_intraday_high_pct": prev_5d_high,
        "prev_3d_avg_volume": (sum(prev_3_vol) / len(prev_3_vol)) if prev_3_vol else None,
        "prev_5d_avg_volume": (sum(prev_5_vol) / len(prev_5_vol)) if prev_5_vol else None,
        "prev_3d_relative_volume_like": (prev_3_vol[0] / (sum(prev_3_vol) / len(prev_3_vol))) if len(prev_3_vol) >= 2 and sum(prev_3_vol) > 0 else None,
        "prev_5d_relative_volume_like": (prev_5_vol[0] / (sum(prev_5_vol) / len(prev_5_vol))) if len(prev_5_vol) >= 2 and sum(prev_5_vol) > 0 else None,
    }


def multiday_score(row: dict[str, Any]) -> float:
    """Compact 0-100 what-if score for prior momentum/liquidity."""
    score = 0.0

    def pos(value: Any, cap: float, weight: float) -> float:
        val = fnum(value)
        if val is None:
            return 0.0
        return max(0.0, min(float(val), cap)) / cap * weight

    score += pos(row.get("prev_1d_return_pct"), 10.0, 20.0)
    score += pos(row.get("prev_3d_return_pct"), 15.0, 20.0)
    score += pos(row.get("prev_5d_return_pct"), 25.0, 20.0)
    score += pos(row.get("prev_3d_max_intraday_high_pct"), 15.0, 15.0)
    score += pos(row.get("prev_5d_max_intraday_high_pct"), 25.0, 15.0)
    score += pos(row.get("prev_3d_relative_volume_like"), 3.0, 5.0)
    score += pos(row.get("prev_5d_relative_volume_like"), 3.0, 5.0)
    return round(max(0.0, min(100.0, score)), 4)


def add_multiday_ranks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["hypothetical_multiday_score"] = [multiday_score(row) for row in out.to_dict("records")]
    out["hypothetical_multiday_rank"] = out["hypothetical_multiday_score"].rank(method="first", ascending=False).astype(int)
    out["would_enter_multiday_top100"] = (out["hypothetical_multiday_rank"] <= 100).astype(int)
    return out


def first_time_above(candles: pd.DataFrame, open_price: float | None, threshold_pct: float) -> pd.Timestamp | None:
    if candles.empty or open_price is None or open_price <= 0:
        return None
    rows = candles[pd.to_numeric(candles.get("high", pd.Series(dtype=float)), errors="coerce") >= open_price * (1.0 + threshold_pct / 100.0)]
    if rows.empty:
        return None
    return rows.sort_values("timestamp").iloc[0]["timestamp"]


def no_signal_diagnostics(
    candles: pd.DataFrame,
    *,
    min_first_5m_high_pct: float,
    min_first_15m_high_pct: float,
    min_or_range_pct: float,
) -> dict[str, Any]:
    stats = calculate_runner_stats(candles)
    if stats is None or candles.empty:
        return {
            "top100_no_signal_reason": "missing_candles",
            "first_time_above_5pct": "",
            "first_time_above_8pct": "",
            "opening_range_high_pct": None,
            "opening_range_low_pct": None,
            "opening_range_break_time": "",
            "did_break_or_high": 0,
            "had_required_first5": 0,
            "had_required_first15": 0,
            "had_required_or_range": 0,
            "possible_signal_time": "",
        }
    rows = candles.sort_values("timestamp").reset_index(drop=True)
    start = rows.iloc[0]["timestamp"]
    first15 = rows[rows["timestamp"] < start + pd.Timedelta(minutes=15)]
    or_high = fnum(first15["high"].max()) if not first15.empty else None
    or_low = fnum(first15["low"].min()) if not first15.empty else None
    or_high_pct = pct(or_high, stats.open_price)
    or_low_pct = pct(or_low, stats.open_price)
    break_time: pd.Timestamp | None = None
    if or_high is not None:
        after_or = rows[rows["timestamp"] >= start + pd.Timedelta(minutes=15)]
        broke = after_or[pd.to_numeric(after_or["high"], errors="coerce") >= or_high]
        if not broke.empty:
            break_time = broke.iloc[0]["timestamp"]
    had_first5 = bool((stats.first_5m_high_pct or -999.0) >= min_first_5m_high_pct)
    had_first15 = bool((stats.first_15m_high_pct or -999.0) >= min_first_15m_high_pct)
    had_or = bool((stats.or_range_pct or -999.0) >= min_or_range_pct)
    possible_signal_time = break_time if had_first5 and had_first15 and had_or and break_time is not None else None
    if not had_first5:
        reason = "failed_first5"
    elif not had_first15:
        reason = "failed_first15"
    elif not had_or:
        reason = "failed_or_range"
    elif break_time is not None and stats.high_time is not None and break_time > stats.high_time:
        reason = "broke_too_late"
    elif possible_signal_time is not None:
        reason = "should_have_signaled"
    else:
        reason = "unknown"
    return {
        "top100_no_signal_reason": reason,
        "first_time_above_5pct": iso_ts(first_time_above(rows, stats.open_price, 5.0)),
        "first_time_above_8pct": iso_ts(first_time_above(rows, stats.open_price, 8.0)),
        "opening_range_high_pct": or_high_pct,
        "opening_range_low_pct": or_low_pct,
        "opening_range_break_time": iso_ts(break_time),
        "did_break_or_high": int(break_time is not None),
        "had_required_first5": int(had_first5),
        "had_required_first15": int(had_first15),
        "had_required_or_range": int(had_or),
        "possible_signal_time": iso_ts(possible_signal_time),
    }


def analyze_missed_runners(
    *,
    session_date: str,
    history_dir: Path,
    universe_path: Path,
    top100_path: Path,
    sqlite_path: Path,
    recorder_dir: Path,
    threshold_pct: float,
    min_first_5m_high_pct: float = 0.5,
    min_first_15m_high_pct: float = 1.0,
    min_or_range_pct: float = 0.5,
    max_symbols: int | None = None,
    progress_every: int = 250,
) -> pd.DataFrame:
    started = time.monotonic()
    symbols = load_universe_symbols(universe_path)
    history_files = find_history_files(history_dir, session_date)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    top100 = load_top100(top100_path)
    top100_by_symbol = top100.set_index("symbol").to_dict("index") if not top100.empty else {}
    entries = load_entries(sqlite_path, session_date)
    entries_by_symbol = entries.set_index("symbol").to_dict("index") if not entries.empty else {}
    signal_rows = load_recorder_table(recorder_dir, session_date, ["signal_snapshots", "selection_events", "signals"])
    order_rows = load_recorder_table(recorder_dir, session_date, ["order_intents", "orders", "entry_orders"])
    spread_rows = load_recorder_table(recorder_dir, session_date, ["spread_snapshots", "market_snapshots"])

    print(
        f"MISSED_START date={session_date} universe_symbols={len(symbols)} "
        f"universe_files_found={len(history_files)} top100_symbols={len(top100_by_symbol)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols, start=1):
        path = history_files.get(symbol)
        candles = read_history_for_missed(path) if path is not None else pd.DataFrame()
        stats = calculate_runner_stats(candles)
        if idx % progress_every == 0:
            elapsed = time.monotonic() - started
            print(f"MISSED_PROGRESS date={session_date} processed={idx}/{total} elapsed={elapsed:.1f}", flush=True)
        if stats is None or stats.open_to_high_pct < threshold_pct:
            continue
        top_row = top100_by_symbol.get(symbol, {})
        source_bucket = "top100" if top_row else "outside_top100"
        entry = entries_by_symbol.get(symbol, {})
        entry_ts = pd.to_datetime(entry.get("entry_ts"), errors="coerce", utc=True) if entry else pd.NaT
        entry_time = None if pd.isna(entry_ts) else entry_ts
        entry_price = fnum(entry.get("entry_price_num")) if entry else None
        sig = symbol_event_row(signal_rows, symbol)
        order = symbol_event_row(order_rows, symbol)
        spread = nearest_row(spread_rows, entry_time or stats.high_time, symbol)
        was_bought = bool(entry)
        prior = previous_session_context(history_dir, symbol, session_date)
        missed_reason = classify_missed_reason(
            source_bucket=source_bucket,
            was_bought=was_bought,
            entry_time=entry_time,
            high_time=stats.high_time,
            signal_row=sig,
            order_row=order,
        )
        no_signal = no_signal_diagnostics(
            candles,
            min_first_5m_high_pct=min_first_5m_high_pct,
            min_first_15m_high_pct=min_first_15m_high_pct,
            min_or_range_pct=min_or_range_pct,
        ) if source_bucket == "top100" and missed_reason == "no_signal" else {
            "top100_no_signal_reason": "",
            "first_time_above_5pct": "",
            "first_time_above_8pct": "",
            "opening_range_high_pct": None,
            "opening_range_low_pct": None,
            "opening_range_break_time": "",
            "did_break_or_high": "",
            "had_required_first5": "",
            "had_required_first15": "",
            "had_required_or_range": "",
            "possible_signal_time": "",
        }
        rows.append({
            "date": session_date,
            "symbol": symbol,
            "source_bucket": source_bucket,
            "top100_rank": top_row.get("top100_rank"),
            "top100_score": top_row.get("top100_score"),
            "open": stats.open_price,
            "high": stats.high_price,
            "high_time": iso_ts(stats.high_time),
            "open_to_high_pct": stats.open_to_high_pct,
            "was_bought": int(was_bought),
            "entry_time": iso_ts(entry_time),
            "entry_price": entry_price,
            "entry_vs_open_pct": pct(entry_price, stats.open_price),
            "entry_vs_high_pct": pct(entry_price, stats.high_price),
            "missed_reason_group": missed_reason,
            "rejection_reason": first_existing_column(order, ["reject_reason", "rejection_reason", "reason", "error"]),
            "blocked_reason": first_existing_column(sig, ["blocked_reason", "entries_blocked_reason", "risk_guard_reason", "reject_reason", "reason"]),
            "signal_time": first_existing_column(sig, ["signal_time", "timestamp", "event_time"]),
            "ready_since": first_existing_column(sig, ["ready_since"]),
            "candidate_age_seconds": first_existing_column(sig, ["candidate_age_seconds"]),
            "live_entry_score": entry.get("live_entry_score") if entry else first_existing_column(sig, ["live_entry_score", "score"]),
            "live_entry_rank": entry.get("live_entry_rank") if entry else first_existing_column(sig, ["live_entry_rank", "ranking_position"]),
            "spread_bps_near_entry": first_existing_column(spread, ["spread_bps", "bid_ask_spread_bps"]),
            "first_5m_high_pct": first_existing_column(sig, ["first_5m_high_pct"]) or stats.first_5m_high_pct,
            "first_15m_high_pct": first_existing_column(sig, ["first_15m_high_pct"]) or stats.first_15m_high_pct,
            "or_range_pct": first_existing_column(sig, ["or_range_pct"]) or stats.or_range_pct,
            **prior,
            **no_signal,
        })
    out = pd.DataFrame(rows)
    if total and total % progress_every:
        elapsed = time.monotonic() - started
        print(f"MISSED_PROGRESS date={session_date} processed={total}/{total} elapsed={elapsed:.1f}", flush=True)
    if out.empty:
        elapsed = time.monotonic() - started
        print(f"MISSED_DONE date={session_date} rows=0 elapsed_seconds={elapsed:.1f}", flush=True)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = add_multiday_ranks(out)
    result = out.sort_values("open_to_high_pct", ascending=False)[OUTPUT_COLUMNS]
    elapsed = time.monotonic() - started
    print(f"MISSED_DONE date={session_date} rows={len(result)} elapsed_seconds={elapsed:.1f}", flush=True)
    return result


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    bought = int(pd.to_numeric(df.get("was_bought", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not df.empty else 0
    in_top100 = int((df.get("source_bucket", pd.Series(dtype=str)) == "top100").sum()) if not df.empty else 0
    print(f"MISSED_RUNNERS total_runners={total} in_top100={in_top100} outside_top100={total - in_top100} bought={bought} missed={total - bought}")
    if df.empty:
        return
    reasons = Counter(df["missed_reason_group"].fillna("unknown").astype(str))
    detectable = int(pd.to_numeric(df.get("was_detectable_from_history", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    not_in_top100 = int((df["missed_reason_group"] == "not_in_top100").sum())
    no_signal = int((df["missed_reason_group"] == "no_signal").sum())
    bought_late = int((df["missed_reason_group"] == "bought_late").sum())
    detectable_not_top100 = int(((df["was_detectable_from_history"] == 1) & (df["missed_reason_group"] == "not_in_top100")).sum())
    detectable_multiday_top100 = int(((df["was_detectable_from_history"] == 1) & (df["missed_reason_group"] == "not_in_top100") & (df["would_enter_multiday_top100"] == 1)).sum())
    print(
        f"missed_breakdown not_in_top100={not_in_top100} no_signal={no_signal} bought_late={bought_late} "
        f"detectable_from_history={detectable} not_detectable_from_history={total - detectable} "
        f"detectable_but_not_in_top100={detectable_not_top100} detectable_multiday_top100={detectable_multiday_top100}"
    )
    print("reason_counts=" + ", ".join(f"{k}:{v}" for k, v in reasons.most_common()))
    cols = ["symbol", "source_bucket", "open_to_high_pct", "was_bought", "missed_reason_group", "top100_rank", "top100_score"]
    print("top20_by_open_to_high_pct:")
    print(df[cols].head(20).to_string(index=False))
    print("top_missed_detectable_runners:")
    print(df[(df["was_bought"] == 0) & (df["was_detectable_from_history"] == 1)].head(20)[["symbol", "open_to_high_pct", "missed_reason_group", "detectability_reason", "top100_rank", "hypothetical_multiday_score", "hypothetical_multiday_rank", "would_enter_multiday_top100"]].to_string(index=False))
    print("top_missed_not_detectable_runners:")
    print(df[(df["was_bought"] == 0) & (df["was_detectable_from_history"] == 0)].head(20)[["symbol", "open_to_high_pct", "missed_reason_group", "top100_rank"]].to_string(index=False))
    top_no_signal = df[(df["source_bucket"] == "top100") & (df["missed_reason_group"] == "no_signal")]
    print(f"top100_no_signal_count={len(top_no_signal)}")
    if not top_no_signal.empty:
        counts = Counter(top_no_signal["top100_no_signal_reason"].fillna("unknown").astype(str))
        print("top100_no_signal_reason_counts=" + ", ".join(f"{k}:{v}" for k, v in counts.most_common()))
        print("top20_top100_no_signal_runners:")
        print(top_no_signal.sort_values("open_to_high_pct", ascending=False)[["symbol", "open_to_high_pct", "top100_rank", "first_5m_high_pct", "first_15m_high_pct", "or_range_pct", "top100_no_signal_reason", "possible_signal_time"]].head(20).to_string(index=False))
        print(f"top100_no_signal_should_have_signaled={int((top_no_signal['top100_no_signal_reason'] == 'should_have_signaled').sum())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find >= threshold intraday runners and explain missed or late entries.")
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--threshold-pct", type=float, default=8.0)
    parser.add_argument("--min-first-5m-high-pct", type=float, default=0.5)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=1.0)
    parser.add_argument("--min-or-range-pct", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit processed universe symbols for quick testing.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output CSV.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    top100 = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    output = args.output or Path(f"data/analysis/missed_runners_{args.date}.csv")
    if output.exists() and not args.force:
        print(f"MISSED_SKIPPED_EXISTING date={args.date} output={output}", flush=True)
        return 0
    started = time.monotonic()
    df = analyze_missed_runners(
        session_date=args.date,
        history_dir=args.history_dir,
        universe_path=args.universe,
        top100_path=top100,
        sqlite_path=args.sqlite_path,
        recorder_dir=args.recorder_dir,
        threshold_pct=args.threshold_pct,
        min_first_5m_high_pct=args.min_first_5m_high_pct,
        min_first_15m_high_pct=args.min_first_15m_high_pct,
        min_or_range_pct=args.min_or_range_pct,
        max_symbols=args.max_symbols,
    )
    elapsed = time.monotonic() - started
    if elapsed > 120.0:
        print(f"MISSED_SLOW_DATE date={args.date} elapsed={elapsed:.1f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print_summary(df)
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
