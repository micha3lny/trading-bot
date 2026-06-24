from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.market_calendar import get_us_equity_session, previous_us_equity_trading_day
from src.live_trading.ineligible_symbols import (
    DEFAULT_RUNTIME_INELIGIBLE,
    DEFAULT_SYMBOL_DENYLIST,
    combined_ineligible_symbols,
)
from src.live_trading.ranking.ranking_store import RankingStore
from src.live_trading.storage.sqlite_store import open_sqlite_store, safe_sqlite_call
from src.live_trading.unified_logger import install_unified_logger


DEFAULT_UNIVERSE = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_HISTORY_DIR = "data/history/universe_1m"
DEFAULT_SQLITE_PATH = "data/runtime/rankings.sqlite"
MIN_LATEST_ROWS = 100
DEFAULT_MAX_MISSING_LOG = 50
DEFAULT_MAX_REJECT_LOG = 50
DEFAULT_MAX_PARTIAL_HISTORY_LOG = 50
DEFAULT_PRIOR_SESSIONS = int(os.getenv("TRADING_BOT_TOP100_PRIOR_SESSIONS", "5") or "5")
DEFAULT_PRIOR_READ_SLOW_SECONDS = float(os.getenv("TRADING_BOT_TOP100_PRIOR_READ_SLOW_SECONDS", "2.0") or "2.0")


@dataclass(frozen=True)
class SymbolRanking:
    symbol: str
    score: float
    momentum_score: float
    liquidity_score: float
    last_close: float
    dollar_volume: float
    day_return_pct: float
    intraday_high_pct: float
    range_pct: float
    volume: float
    gap_pct: float | None
    median_1m_range_bps: float
    avg_abs_1m_return_bps: float
    multi_day_return_pct: float | None
    components: dict[str, Any]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def scaled(value: float | None, low: float, high: float) -> float:
    if value is None or pd.isna(value) or high <= low:
        return 0.0
    return clamp((float(value) - low) / (high - low) * 100.0)


def scaled_log(value: float | None, low: float, high: float) -> float:
    if value is None or value <= 0 or low <= 0 or high <= low:
        return 0.0
    return scaled(math.log10(float(value)), math.log10(low), math.log10(high))


def load_universe(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing universe file: {p}")
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("Universe CSV must contain symbol column")
    if "alpha_score" in df.columns:
        df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("alpha_score", ascending=False)
    symbols = df["symbol"].astype(str).str.upper().str.strip().dropna().drop_duplicates().tolist()
    return [s for s in symbols if s and s != "NAN"]


def junk_symbol_reason(symbol: str) -> str | None:
    s = symbol.upper().strip()
    if len(s) >= 5 and s.endswith("W"):
        return "warrant_suffix"
    if len(s) >= 5 and s.endswith("U"):
        return "unit_suffix"
    if len(s) >= 5 and s.endswith("R"):
        return "rights_suffix"
    if len(s) >= 5 and s.endswith("P"):
        return "preferred_or_special_suffix"
    return None


def parquet_path(history_dir: str | Path, symbol: str, session_date: date, session_type: str = "RTH") -> Path:
    root = Path(history_dir)
    return (
        root
        / f"session_type={session_type.upper()}"
        / f"symbol={symbol.upper()}"
        / f"year={session_date.year:04d}"
        / f"month={session_date.month:02d}"
        / f"day={session_date.day:02d}.parquet"
    )


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "timestamp" not in out.columns and "bar_time_utc" in out.columns:
        out = out.rename(columns={"bar_time_utc": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(out.columns):
        missing = ",".join(sorted(required - set(out.columns)))
        raise ValueError(f"history parquet missing required columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume", "wap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out[(out["open"] > 0) & (out["high"] >= out["low"]) & (out["close"] > 0) & (out["volume"] >= 0)]
    return out.sort_values("timestamp").reset_index(drop=True)


def read_session(history_dir: str | Path, symbol: str, session_date: date, session_type: str = "RTH") -> pd.DataFrame:
    path = parquet_path(history_dir, symbol, session_date, session_type)
    if not path.exists():
        return pd.DataFrame()
    return normalize_history_df(pd.read_parquet(path))


def parse_partition_date(path: Path) -> date | None:
    try:
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
        day_value = str(parts.get("day") or path.stem).replace(".parquet", "")
        return date(int(parts["year"]), int(parts["month"]), int(day_value))
    except Exception:
        return None


def prior_session_paths(
    history_dir: str | Path,
    symbol: str,
    session_date: date,
    session_type: str = "RTH",
    limit: int | None = None,
) -> list[Path]:
    """Return recent prior session files without recursively scanning symbol history.

    The Top100 builder runs across the full universe, so a per-symbol recursive
    glob becomes expensive as the parquet archive grows. Daily ranking only
    needs recent trading sessions for multi-day context, so probe exact
    partition paths for prior US equity trading days.
    """
    target = max(0, int(limit)) if limit is not None else 5
    max_checks = max(target * 6, 30)
    paths: list[Path] = []
    cursor = session_date
    checks = 0
    while checks < max_checks and (limit is None or len(paths) < target):
        cursor = previous_us_equity_trading_day(cursor)
        checks += 1
        path = parquet_path(history_dir, symbol, cursor, session_type)
        if path.exists():
            paths.append(path)
    return paths


@dataclass(frozen=True)
class PriorCloseResult:
    closes: list[float]
    paths_checked: int
    paths_found: int
    seconds: float
    slow: bool = False
    degraded: bool = False


def read_prior_close(path: Path) -> float | None:
    """Read only the final close needed for prior-session context.

    Full parquet normalization is intentionally avoided here. Top100 runs this
    for thousands of symbols, and reading all columns for multiple prior days
    dominates premarket build time on the Raspberry Pi.
    """
    try:
        df = pd.read_parquet(path, columns=["close"])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
    if df.empty:
        return None
    columns = {str(col).strip().lower(): col for col in df.columns}
    close_col = columns.get("close")
    if close_col is None:
        return None
    close = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if close.empty:
        return None
    value = float(close.iloc[-1])
    return value if value > 0 else None


def recent_prior_closes_with_diagnostics(
    history_dir: str | Path,
    symbol: str,
    session_date: date,
    limit: int,
    session_type: str = "RTH",
    *,
    slow_seconds: float | None = DEFAULT_PRIOR_READ_SLOW_SECONDS,
) -> PriorCloseResult:
    closes: list[float] = []
    started = time.perf_counter()
    paths = prior_session_paths(history_dir, symbol, session_date, session_type, limit=max(0, limit))
    slow_limit = float(slow_seconds) if slow_seconds is not None and slow_seconds > 0 else None
    degraded = False
    for idx, path in enumerate(paths, 1):
        if slow_limit is not None and time.perf_counter() - started > slow_limit:
            degraded = True
            break
        close = read_prior_close(path)
        if close is not None:
            closes.append(close)
        if slow_limit is not None and time.perf_counter() - started > slow_limit and idx < len(paths):
            degraded = True
            break
    seconds = time.perf_counter() - started
    return PriorCloseResult(
        closes=closes,
        paths_checked=len(paths),
        paths_found=len(closes),
        seconds=seconds,
        slow=slow_limit is not None and seconds > slow_limit,
        degraded=degraded,
    )


def recent_prior_closes(history_dir: str | Path, symbol: str, session_date: date, limit: int, session_type: str = "RTH") -> list[float]:
    return recent_prior_closes_with_diagnostics(history_dir, symbol, session_date, limit, session_type).closes


def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    prior_closes: list[float],
    *,
    min_price: float,
    min_bars: int,
    min_volume: float,
    min_dollar_volume: float,
) -> tuple[SymbolRanking | None, str | None]:
    junk_reason = junk_symbol_reason(symbol)
    if junk_reason:
        return None, junk_reason
    if df.empty:
        return None, "missing_history"
    if len(df) < min_bars:
        return None, f"too_few_bars:{len(df)}"

    open_price = float(df["open"].iloc[0])
    high = float(df["high"].max())
    low = float(df["low"].min())
    last_close = float(df["close"].iloc[-1])
    volume = float(df["volume"].sum())
    dollar_price = df["wap"] if "wap" in df.columns and df["wap"].notna().any() else df["close"]
    dollar_volume = float((dollar_price.fillna(df["close"]) * df["volume"]).sum())

    if last_close < min_price:
        return None, f"price_too_low:{last_close:.2f}"
    if volume < min_volume:
        return None, f"volume_too_low:{volume:.0f}"
    if dollar_volume < min_dollar_volume:
        return None, f"dollar_volume_too_low:{dollar_volume:.0f}"

    day_return_pct = (last_close / open_price - 1.0) * 100.0
    intraday_high_pct = (high / open_price - 1.0) * 100.0
    range_pct = (high / low - 1.0) * 100.0 if low > 0 else 0.0
    range_bps = ((df["high"] - df["low"]) / df["close"].replace(0, pd.NA) * 10_000).dropna()
    ret_bps = (df["close"].pct_change() * 10_000).dropna()
    median_1m_range_bps = float(range_bps.median()) if not range_bps.empty else 0.0
    avg_abs_1m_return_bps = float(ret_bps.abs().mean()) if not ret_bps.empty else 0.0

    prior_close = prior_closes[0] if prior_closes else None
    gap_pct = (open_price / prior_close - 1.0) * 100.0 if prior_close and prior_close > 0 else None
    multi_day_return_pct = None
    if len(prior_closes) >= 2 and prior_closes[-1] > 0:
        multi_day_return_pct = (last_close / prior_closes[-1] - 1.0) * 100.0

    capped_dollar_volume = min(dollar_volume, 50_000_000.0)
    components = {
        "intraday_high": scaled(intraday_high_pct, 1.0, 18.0),
        "close_open": scaled(day_return_pct, -1.0, 10.0),
        "range": scaled(range_pct, 2.0, 18.0),
        "median_1m_range": scaled(median_1m_range_bps, 4.0, 100.0),
        "avg_abs_1m_return": scaled(avg_abs_1m_return_bps, 2.0, 45.0),
        "gap": scaled(gap_pct, -3.0, 10.0) if gap_pct is not None else 35.0,
        "multi_day": scaled(multi_day_return_pct, -5.0, 25.0) if multi_day_return_pct is not None else 35.0,
        "liquidity": scaled_log(capped_dollar_volume, min_dollar_volume, 50_000_000.0),
        "data_completeness": scaled(len(df), min_bars, 390.0),
    }
    momentum_score = (
        0.35 * components["intraday_high"]
        + 0.25 * components["range"]
        + 0.15 * components["close_open"]
        + 0.15 * components["multi_day"]
        + 0.10 * components["gap"]
    )
    volatility_score = (
        0.60 * components["median_1m_range"]
        + 0.40 * components["avg_abs_1m_return"]
    )
    liquidity_score = components["liquidity"]
    score = (
        0.45 * momentum_score
        + 0.25 * components["range"]
        + 0.15 * volatility_score
        + 0.05 * components["close_open"]
        + 0.05 * components["data_completeness"]
        + 0.05 * liquidity_score
    )
    components.update(
        {
            "bars": len(df),
            "capped_dollar_volume": capped_dollar_volume,
            "prior_close": prior_close,
            "momentum_score": momentum_score,
            "volatility_score": volatility_score,
            "liquidity_score": liquidity_score,
            "final_score": score,
            "weights": {
                "momentum_score": 0.45,
                "range": 0.25,
                "volatility_score": 0.15,
                "close_open": 0.05,
                "data_completeness": 0.05,
                "liquidity_score": 0.05,
            },
        }
    )

    return (
        SymbolRanking(
            symbol=symbol,
            score=round(float(score), 4),
            momentum_score=round(float(momentum_score), 4),
            liquidity_score=round(float(liquidity_score), 4),
            last_close=last_close,
            dollar_volume=dollar_volume,
            day_return_pct=day_return_pct,
            intraday_high_pct=intraday_high_pct,
            range_pct=range_pct,
            volume=volume,
            gap_pct=gap_pct,
            median_1m_range_bps=median_1m_range_bps,
            avg_abs_1m_return_bps=avg_abs_1m_return_bps,
            multi_day_return_pct=multi_day_return_pct,
            components=components,
        ),
        None,
    )


def ranking_to_row(rank: int, item: SymbolRanking) -> dict[str, Any]:
    components_json = json.dumps(item.components, sort_keys=True)
    return {
        "rank": rank,
        "symbol": item.symbol,
        "score": item.score,
        "alpha_score": item.score,
        "final_score": item.score,
        "momentum_score": item.momentum_score,
        "liquidity_score": item.liquidity_score,
        "last_close": round(item.last_close, 4),
        "dollar_volume": round(item.dollar_volume, 2),
        "day_return_pct": round(item.day_return_pct, 4),
        "close_open_pct": round(item.day_return_pct, 4),
        "intraday_high_pct": round(item.intraday_high_pct, 4),
        "range_pct": round(item.range_pct, 4),
        "volume": round(item.volume, 2),
        "gap_pct": round(item.gap_pct, 4) if item.gap_pct is not None else "",
        "median_1m_range_bps": round(item.median_1m_range_bps, 4),
        "avg_abs_1m_return_bps": round(item.avg_abs_1m_return_bps, 4),
        "multi_day_return_pct": round(item.multi_day_return_pct, 4) if item.multi_day_return_pct is not None else "",
        "reason": "ranked",
        "components_json": components_json,
    }


def build_daily_top(
    *,
    ranking_date: date,
    universe_path: str | Path,
    history_dir: str | Path,
    top_n: int,
    session_type: str,
    min_price: float,
    min_bars: int,
    min_volume: float,
    min_dollar_volume: float,
    prior_sessions: int,
    max_missing_log: int = DEFAULT_MAX_MISSING_LOG,
    max_reject_log: int = DEFAULT_MAX_REJECT_LOG,
    symbol_denylist_path: str | Path | None = DEFAULT_SYMBOL_DENYLIST,
    runtime_ineligible_path: str | Path | None = DEFAULT_RUNTIME_INELIGIBLE,
    prior_read_slow_seconds: float | None = DEFAULT_PRIOR_READ_SLOW_SECONDS,
    max_partial_history_log: int = DEFAULT_MAX_PARTIAL_HISTORY_LOG,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    symbols = load_universe(universe_path)
    ineligible = combined_ineligible_symbols(symbol_denylist_path, runtime_ineligible_path)
    excluded_rows: list[dict[str, str]] = []
    tradeable_symbols: list[str] = []
    for symbol in symbols:
        info = ineligible.get(symbol)
        if info:
            reason = str(info.get("reason") or "ineligible")
            excluded_rows.append({"symbol": symbol, "reason": reason})
            print(f"TOP100_SYMBOL_EXCLUDED symbol={symbol} reason={reason}", flush=True)
            continue
        tradeable_symbols.append(symbol)
    print(
        f"DAILY_TOP100_START date={ranking_date.isoformat()} symbols={len(symbols)} "
        f"tradeable_symbols={len(tradeable_symbols)} excluded_ineligible={len(excluded_rows)} "
        f"history_dir={history_dir} session_type={session_type} "
        f"prior_sessions={max(0, int(prior_sessions))} "
        f"prior_read_slow_seconds={float(prior_read_slow_seconds or 0.0):.2f}",
        flush=True,
    )
    ranked: list[SymbolRanking] = []
    stats: dict[str, int] = {
        "symbols": len(symbols),
        "valid": 0,
        "missing": 0,
        "rejected": 0,
        "errors": 0,
        "excluded_ineligible": len(excluded_rows),
        "prior_slow_symbols": 0,
        "prior_degraded_symbols": 0,
        "prior_partial_symbols": 0,
        "prior_paths_checked": 0,
        "prior_paths_found": 0,
    }
    missing_symbols: list[str] = []
    rejected_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    started = time.perf_counter()
    current_read_seconds = 0.0
    prior_read_seconds = 0.0
    analyze_seconds = 0.0
    last_progress_at = started
    last_progress_processed = 0
    total = len(tradeable_symbols)

    def emit_progress(processed: int, *, force: bool = False) -> None:
        nonlocal last_progress_at, last_progress_processed
        now = time.perf_counter()
        if processed == last_progress_processed:
            return
        if not force and processed < total and processed % 100 != 0 and now - last_progress_at < 60.0:
            return
        elapsed = max(0.001, now - started)
        last_progress_at = now
        last_progress_processed = processed
        print(
            f"DAILY_TOP100_PROGRESS processed={processed}/{total} valid={stats['valid']} "
            f"missing={stats['missing']} rejected={stats['rejected']} errors={stats['errors']} "
            f"elapsed_seconds={elapsed:.1f} symbols_per_second={processed / elapsed:.2f} "
            f"current_read_seconds={current_read_seconds:.1f} prior_read_seconds={prior_read_seconds:.1f} "
            f"analyze_seconds={analyze_seconds:.1f}",
            flush=True,
        )

    for idx, symbol in enumerate(tradeable_symbols, 1):
        try:
            read_started = time.perf_counter()
            df = read_session(history_dir, symbol, ranking_date, session_type)
            current_read_seconds += time.perf_counter() - read_started
            if df.empty:
                stats["missing"] += 1
                missing_symbols.append(symbol)
                if stats["missing"] <= max(0, max_missing_log):
                    print(f"DAILY_TOP100_MISSING_DATA symbol={symbol} date={ranking_date.isoformat()}", flush=True)
                emit_progress(idx)
                continue
            prior_started = time.perf_counter()
            prior_result = recent_prior_closes_with_diagnostics(
                history_dir,
                symbol,
                ranking_date,
                prior_sessions,
                session_type,
                slow_seconds=prior_read_slow_seconds,
            )
            prior_closes = prior_result.closes
            prior_read_seconds += time.perf_counter() - prior_started
            stats["prior_paths_checked"] += prior_result.paths_checked
            stats["prior_paths_found"] += prior_result.paths_found
            expected_prior_sessions = max(0, int(prior_sessions))
            if expected_prior_sessions > 0 and prior_result.paths_found < expected_prior_sessions:
                stats["prior_partial_symbols"] += 1
                if stats["prior_partial_symbols"] <= max(0, max_partial_history_log):
                    print(
                        f"TOP100_PARTIAL_HISTORY symbol={symbol} requested_prior_sessions={expected_prior_sessions} "
                        f"available_prior_sessions={prior_result.paths_found} paths_checked={prior_result.paths_checked} "
                        f"degraded={1 if prior_result.degraded else 0}",
                        flush=True,
                    )
            if prior_result.slow:
                stats["prior_slow_symbols"] += 1
                print(
                    f"TOP100_PRIOR_READ_SLOW symbol={symbol} seconds={prior_result.seconds:.2f} "
                    f"prior_sessions={max(0, int(prior_sessions))} paths_checked={prior_result.paths_checked} "
                    f"paths_found={prior_result.paths_found} degraded={1 if prior_result.degraded else 0}",
                    flush=True,
                )
            if prior_result.degraded:
                stats["prior_degraded_symbols"] += 1
            analyze_started = time.perf_counter()
            item, reject_reason = analyze_symbol(
                symbol,
                df,
                prior_closes,
                min_price=min_price,
                min_bars=min_bars,
                min_volume=min_volume,
                min_dollar_volume=min_dollar_volume,
            )
            analyze_seconds += time.perf_counter() - analyze_started
            if item is None:
                stats["rejected"] += 1
                rejected_rows.append({"symbol": symbol, "reason": reject_reason or "rejected"})
                if stats["rejected"] <= max(0, max_reject_log):
                    print(f"DAILY_TOP100_REJECTED symbol={symbol} reason={reject_reason}", flush=True)
                emit_progress(idx)
                continue
            ranked.append(item)
            stats["valid"] += 1
        except Exception as exc:
            stats["errors"] += 1
            error_rows.append({"symbol": symbol, "reason": repr(exc)})
            print(f"DAILY_TOP100_SYMBOL_ERROR symbol={symbol} error={exc!r}", flush=True)
        emit_progress(idx)

    total_elapsed = max(0.001, time.perf_counter() - started)
    emit_progress(total, force=True)
    ranked.sort(key=lambda item: (item.score, item.dollar_volume, item.intraday_high_pct), reverse=True)
    rows = [ranking_to_row(rank, item) for rank, item in enumerate(ranked[: max(0, top_n)], 1)]
    stats["elapsed_seconds"] = total_elapsed  # type: ignore[assignment]
    stats["total_seconds"] = total_elapsed  # type: ignore[assignment]
    stats["symbols_per_second"] = total / total_elapsed if total else 0.0  # type: ignore[assignment]
    stats["current_read_seconds"] = current_read_seconds  # type: ignore[assignment]
    stats["current_day_read_seconds"] = current_read_seconds  # type: ignore[assignment]
    stats["prior_read_seconds"] = prior_read_seconds  # type: ignore[assignment]
    stats["prior_sessions_read_seconds"] = prior_read_seconds  # type: ignore[assignment]
    stats["analyze_seconds"] = analyze_seconds  # type: ignore[assignment]
    stats["analysis_seconds"] = analyze_seconds  # type: ignore[assignment]
    stats["_missing_symbols"] = missing_symbols  # type: ignore[assignment]
    stats["_rejected_rows"] = rejected_rows  # type: ignore[assignment]
    stats["_error_rows"] = error_rows  # type: ignore[assignment]
    stats["_excluded_ineligible_rows"] = excluded_rows  # type: ignore[assignment]
    if stats["missing"] > max(0, max_missing_log):
        print(
            f"DAILY_TOP100_MISSING_DATA_SUPPRESSED count={stats['missing'] - max(0, max_missing_log)}",
            flush=True,
        )
    if stats["rejected"] > max(0, max_reject_log):
        print(
            f"DAILY_TOP100_REJECTED_SUPPRESSED count={stats['rejected'] - max(0, max_reject_log)}",
            flush=True,
        )
    if stats["prior_partial_symbols"] > max(0, max_partial_history_log):
        print(
            f"TOP100_PARTIAL_HISTORY_SUPPRESSED count={stats['prior_partial_symbols'] - max(0, max_partial_history_log)}",
            flush=True,
        )
    return rows, stats


CSV_COLUMNS = [
    "rank",
    "symbol",
    "score",
    "alpha_score",
    "final_score",
    "momentum_score",
    "liquidity_score",
    "last_close",
    "dollar_volume",
    "day_return_pct",
    "close_open_pct",
    "intraday_high_pct",
    "range_pct",
    "volume",
    "gap_pct",
    "median_1m_range_bps",
    "avg_abs_1m_return_bps",
    "multi_day_return_pct",
    "reason",
    "components_json",
]


def render_output_csv(rows: list[dict[str, Any]]) -> str:
    return pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(index=False)


def write_text_atomic(path: str | Path, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(output)


def write_output_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_text_atomic(path, render_output_csv(rows))


def write_diagnostics_csv(path: str | Path, ranking_date: date, stats: dict[str, Any]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for symbol in stats.get("_missing_symbols", []):
        rows.append({"date": ranking_date.isoformat(), "symbol": symbol, "status": "missing", "reason": "missing_history"})
    for row in stats.get("_rejected_rows", []):
        rows.append({"date": ranking_date.isoformat(), "symbol": row.get("symbol"), "status": "rejected", "reason": row.get("reason")})
    for row in stats.get("_error_rows", []):
        rows.append({"date": ranking_date.isoformat(), "symbol": row.get("symbol"), "status": "error", "reason": row.get("reason")})
    for row in stats.get("_excluded_ineligible_rows", []):
        rows.append({"date": ranking_date.isoformat(), "symbol": row.get("symbol"), "status": "excluded_ineligible", "reason": row.get("reason")})
    write_text_atomic(output, pd.DataFrame(rows, columns=["date", "symbol", "status", "reason"]).to_csv(index=False))
    return len(rows)


def update_latest_output(
    dated_output: str | Path,
    latest_output: str | Path,
    rows: list[dict[str, Any]],
    *,
    missing_history_count: int = 0,
    max_missing_history_for_latest: int = 0,
) -> bool:
    if len(rows) < MIN_LATEST_ROWS:
        print(
            f"DAILY_TOP100_LATEST_SKIPPED reason=too_few_rows rows={len(rows)} "
            f"required={MIN_LATEST_ROWS} latest_output={latest_output}",
            flush=True,
        )
        return False
    if int(missing_history_count) > int(max_missing_history_for_latest):
        print(
            f"DAILY_TOP100_BLOCKED_HISTORY_NOT_READY missing_history={int(missing_history_count)} "
            f"max_missing_history={int(max_missing_history_for_latest)} latest_output={latest_output}",
            flush=True,
        )
        print(
            f"DAILY_TOP100_LATEST_SKIPPED reason=missing_history missing_history={int(missing_history_count)} "
            f"max_missing_history={int(max_missing_history_for_latest)} latest_output={latest_output}",
            flush=True,
        )
        return False

    dated = Path(dated_output)
    latest = Path(latest_output)
    content = dated.read_text(encoding="utf-8") if dated.exists() else render_output_csv(rows)
    write_text_atomic(latest, content)
    print(
        f"DAILY_TOP100_LATEST_UPDATED latest_output={latest} rows={len(rows)}",
        flush=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v67/v68 daily Top100 CSV from collected 1m parquet history")
    parser.add_argument("--date", required=True, help="RTH session date to rank, e.g. 2026-05-15")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latest-output", default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-bars", type=int, default=180)
    parser.add_argument("--min-volume", type=float, default=100_000.0)
    parser.add_argument("--min-dollar-volume", type=float, default=500_000.0)
    parser.add_argument("--prior-sessions", type=int, default=DEFAULT_PRIOR_SESSIONS)
    parser.add_argument("--prior-read-slow-seconds", type=float, default=DEFAULT_PRIOR_READ_SLOW_SECONDS)
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--runtime-sqlite-path", default=None)
    parser.add_argument("--disable-runtime-sqlite", action="store_true")
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--max-missing-log", type=int, default=DEFAULT_MAX_MISSING_LOG)
    parser.add_argument("--max-reject-log", type=int, default=DEFAULT_MAX_REJECT_LOG)
    parser.add_argument("--max-partial-history-log", type=int, default=DEFAULT_MAX_PARTIAL_HISTORY_LOG)
    parser.add_argument("--max-missing-history-for-latest", type=int, default=0)
    parser.add_argument("--symbol-denylist", default=DEFAULT_SYMBOL_DENYLIST)
    parser.add_argument("--runtime-ineligible-path", default=DEFAULT_RUNTIME_INELIGIBLE)
    parser.add_argument("--no-sqlite", action="store_true")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()
    install_unified_logger(args.log_dir)

    requested_date = parse_date(args.date)
    requested_session = get_us_equity_session(requested_date)
    ranking_date = requested_date
    if not requested_session.is_trading_day:
        ranking_date = previous_us_equity_trading_day(requested_date)
        print(
            f"DAILY_TOP100_MARKET_CLOSED_USING_PREVIOUS requested_date={requested_date.isoformat()} "
            f"effective_date={ranking_date.isoformat()} reason={requested_session.reason}",
            flush=True,
        )
    rows, stats = build_daily_top(
        ranking_date=ranking_date,
        universe_path=args.universe,
        history_dir=args.history_dir,
        top_n=int(args.top_n),
        session_type=str(args.session_type).upper(),
        min_price=float(args.min_price),
        min_bars=int(args.min_bars),
        min_volume=float(args.min_volume),
        min_dollar_volume=float(args.min_dollar_volume),
        prior_sessions=int(args.prior_sessions),
        max_missing_log=int(args.max_missing_log),
        max_reject_log=int(args.max_reject_log),
        symbol_denylist_path=args.symbol_denylist,
        runtime_ineligible_path=args.runtime_ineligible_path,
        prior_read_slow_seconds=float(args.prior_read_slow_seconds),
        max_partial_history_log=int(args.max_partial_history_log),
    )
    write_output_csv(args.output, rows)
    diagnostics_rows = 0
    if args.diagnostics_output:
        diagnostics_rows = write_diagnostics_csv(args.diagnostics_output, ranking_date, stats)
        print(
            f"DAILY_TOP100_DIAGNOSTICS_WRITTEN path={args.diagnostics_output} rows={diagnostics_rows}",
            flush=True,
        )
    latest_ok = None
    if args.latest_output:
        latest_ok = update_latest_output(
            args.output,
            args.latest_output,
            rows,
            missing_history_count=int(stats.get("missing", 0)),
            max_missing_history_for_latest=int(args.max_missing_history_for_latest),
        )
    stored = 0
    if not args.no_sqlite:
        stored = RankingStore(args.sqlite_path).replace_daily_rankings(ranking_date.isoformat(), rows)
    if not args.disable_runtime_sqlite:
        runtime_store = open_sqlite_store(args.runtime_sqlite_path)
        if runtime_store is not None:
            try:
                for row in rows:
                    safe_sqlite_call(
                        runtime_store,
                        "upsert_symbol_daily_feature",
                        {
                            "date": ranking_date.isoformat(),
                            "symbol": row.get("symbol"),
                            "feature_version": "daily_top100_v1",
                            "close": row.get("last_close"),
                            "volume": row.get("volume"),
                            "dollar_volume": row.get("dollar_volume"),
                            "intraday_high_pct": row.get("intraday_high_pct"),
                            "range_pct": row.get("range_pct"),
                            "close_open_pct": row.get("close_open_pct"),
                            "gap_pct": row.get("gap_pct"),
                            "multi_day_return_pct": row.get("multi_day_return_pct"),
                            "median_1m_range_bps": row.get("median_1m_range_bps"),
                            "avg_abs_1m_return_bps": row.get("avg_abs_1m_return_bps"),
                            "momentum_score": row.get("momentum_score"),
                            "liquidity_score": row.get("liquidity_score"),
                            "final_score": row.get("final_score") or row.get("score"),
                            "rank": row.get("rank"),
                            "ranking_version": "daily_top100_builder",
                            "components_json": row.get("components_json"),
                        },
                    )
            finally:
                runtime_store.close()
    print(
        f"DAILY_TOP100_DONE date={ranking_date.isoformat()} output={args.output} rows={len(rows)} "
        f"valid={stats['valid']} missing={stats['missing']} rejected={stats['rejected']} "
        f"errors={stats['errors']} excluded_ineligible_count={stats.get('excluded_ineligible', 0)} "
        f"excluded_ineligible_symbols={','.join([r.get('symbol', '') for r in stats.get('_excluded_ineligible_rows', [])][:20])} "
        f"diagnostics_rows={diagnostics_rows} sqlite_rows={stored} "
        f"elapsed_seconds={float(stats.get('elapsed_seconds', 0.0)):.1f} "
        f"symbols_per_second={float(stats.get('symbols_per_second', 0.0)):.2f} "
        f"current_read_seconds={float(stats.get('current_read_seconds', 0.0)):.1f} "
        f"prior_read_seconds={float(stats.get('prior_read_seconds', 0.0)):.1f} "
        f"analyze_seconds={float(stats.get('analyze_seconds', 0.0)):.1f} "
        f"current_day_read_seconds={float(stats.get('current_day_read_seconds', 0.0)):.1f} "
        f"prior_sessions_read_seconds={float(stats.get('prior_sessions_read_seconds', 0.0)):.1f} "
        f"analysis_seconds={float(stats.get('analysis_seconds', 0.0)):.1f} "
        f"total_seconds={float(stats.get('total_seconds', 0.0)):.1f} "
        f"prior_sessions={int(args.prior_sessions)} "
        f"prior_slow_symbols={stats.get('prior_slow_symbols', 0)} "
        f"prior_degraded_symbols={stats.get('prior_degraded_symbols', 0)} "
        f"prior_partial_symbols={stats.get('prior_partial_symbols', 0)} "
        f"prior_paths_checked={stats.get('prior_paths_checked', 0)} "
        f"prior_paths_found={stats.get('prior_paths_found', 0)}",
        flush=True,
    )
    if len(rows) < int(args.top_n):
        print(f"DAILY_TOP100_WARNING requested_top_n={args.top_n} produced={len(rows)}", flush=True)
    if latest_ok is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
