#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
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

from src.live_trading.analysis.common import fnum, parse_dt
from src.live_trading.ranking.daily_top100_builder import normalize_history_df, parquet_path


DEFAULT_DB = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
PEAK_REBUILD_VERSION = 1


@dataclass(frozen=True)
class PeakMetrics:
    peak_price: float | None
    low_price: float | None
    peak_time: str
    low_time: str
    mfe_pct: float | None
    mae_pct: float | None
    peak_unrealized_pnl: float | None
    max_adverse_unrealized_pnl: float | None
    giveback_from_peak: float | None
    drop_from_peak_pct: float | None
    peak_data_quality: str
    peak_source: str
    validator_status: str
    notes: str
    candles_found: int
    candles_window_rows: int
    candles_min_time_utc: str
    candles_max_time_utc: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def selected_trade_rows(conn: sqlite3.Connection, *, date: str, symbol: str | None = None) -> list[dict[str, Any]]:
    columns = table_columns(conn, "trades")
    if not columns:
        return []
    required = [
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
    selected = [col for col in required if col in columns]
    clauses = [
        "UPPER(COALESCE(status, '')) IN ('CLOSED', 'COMMISSION_PENDING', 'PNL_PENDING')",
        "(session_date = ? OR substr(COALESCE(exit_fill_time, closed_at, ''), 1, 10) = ?)",
        "entry_fill_time IS NOT NULL",
        "exit_fill_time IS NOT NULL",
        "entry_price IS NOT NULL",
        "exit_price IS NOT NULL",
    ]
    params: list[Any] = [date, date]
    if symbol:
        clauses.append("UPPER(symbol) = ?")
        params.append(symbol.upper())
    rows = conn.execute(
        f"""
        SELECT {', '.join(selected)}
        FROM trades
        WHERE {' AND '.join(clauses)}
        ORDER BY UPPER(symbol), entry_fill_time, exit_fill_time, trade_id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def load_session_candles(history_dir: Path, symbol: str, session_date: str, session_type: str) -> pd.DataFrame:
    path = parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), session_type)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    try:
        normalized = normalize_history_df(df)
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in normalized.columns or "high" not in normalized.columns or "low" not in normalized.columns:
        return pd.DataFrame()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    normalized["high"] = pd.to_numeric(normalized["high"], errors="coerce")
    normalized["low"] = pd.to_numeric(normalized["low"], errors="coerce")
    return normalized.dropna(subset=["timestamp", "high", "low"]).sort_values("timestamp").reset_index(drop=True)


def load_trade_candles(history_dir: Path, symbol: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp, session_type: str) -> pd.DataFrame:
    dates = pd.date_range(entry_time.date(), exit_time.date(), freq="D")
    frames = [load_session_candles(history_dir, symbol, day.date().isoformat(), session_type) for day in dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def calculate_peak_metrics(
    candles: pd.DataFrame,
    *,
    entry_time: Any,
    exit_time: Any,
    entry_price: float | None,
    exit_price: float | None,
    quantity: float | None,
    net_pnl: float | None,
) -> PeakMetrics:
    entry_dt = parse_dt(entry_time)
    exit_dt = parse_dt(exit_time)
    entry = fnum(entry_price)
    exit_ = fnum(exit_price)
    qty = abs(fnum(quantity) or 0.0)
    notes: list[str] = []
    if entry_dt is None or exit_dt is None or entry is None or entry <= 0 or exit_ is None:
        return PeakMetrics(None, None, "", "", None, None, None, None, None, None, "MISSING", "unavailable", "MISSING_INPUT", "missing_entry_exit_input", 0, 0, "", "")

    if candles.empty:
        return PeakMetrics(None, None, "", "", None, None, None, None, None, None, "MISSING", "unavailable", "MISSING_CANDLES", "missing_candles", 0, 0, "", "")

    rows = candles.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    rows["high"] = pd.to_numeric(rows["high"], errors="coerce")
    rows["low"] = pd.to_numeric(rows["low"], errors="coerce")
    rows = rows.dropna(subset=["timestamp", "high", "low"]).sort_values("timestamp").reset_index(drop=True)
    if rows.empty:
        return PeakMetrics(None, None, "", "", None, None, None, None, None, None, "MISSING", "unavailable", "MISSING_CANDLES", "missing_candles", 0, 0, "", "")

    candles_min = rows["timestamp"].min()
    candles_max = rows["timestamp"].max()
    window = rows[(rows["timestamp"] >= entry_dt) & (rows["timestamp"] <= exit_dt)].copy()
    quality = "OK"
    if candles_min > entry_dt or candles_max < exit_dt:
        quality = "INCOMPLETE"
        notes.append("candles_do_not_cover_full_entry_exit_window")
    if window.empty:
        return PeakMetrics(
            None,
            None,
            "",
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            "INCOMPLETE",
            "unavailable",
            "MISSING_WINDOW_CANDLES",
            "no_candles_between_entry_exit",
            len(rows),
            0,
            candles_min.isoformat(),
            candles_max.isoformat(),
        )

    high_idx = pd.to_numeric(window["high"], errors="coerce").idxmax()
    low_idx = pd.to_numeric(window["low"], errors="coerce").idxmin()
    candle_peak = fnum(window.loc[high_idx, "high"])
    candle_low = fnum(window.loc[low_idx, "low"])
    candle_peak_time = window.loc[high_idx, "timestamp"]
    candle_low_time = window.loc[low_idx, "timestamp"]

    peak_candidates = [
        ("entry_price", entry, entry_dt),
        ("exit_price", exit_, exit_dt),
        ("candle_high", candle_peak, candle_peak_time),
    ]
    low_candidates = [
        ("entry_price", entry, entry_dt),
        ("exit_price", exit_, exit_dt),
        ("candle_low", candle_low, candle_low_time),
    ]
    peak_source_name, peak_price, peak_time = max(
        [item for item in peak_candidates if item[1] is not None],
        key=lambda item: float(item[1] or 0.0),
    )
    low_source_name, low_price, low_time = min(
        [item for item in low_candidates if item[1] is not None],
        key=lambda item: float(item[1] or 0.0),
    )
    if peak_source_name != "candle_high":
        notes.append(f"peak_from_{peak_source_name}")
        if quality == "OK":
            quality = "INCOMPLETE"
    if low_source_name != "candle_low":
        notes.append(f"low_from_{low_source_name}")

    peak_pct = pct(peak_price, entry)
    low_pct = pct(low_price, entry)
    drop_from_peak = pct(exit_, peak_price)
    peak_unrealized = (float(peak_price) - entry) * qty if peak_price is not None and qty else None
    max_adverse = (float(low_price) - entry) * qty if low_price is not None and qty else None
    giveback = peak_unrealized - float(net_pnl) if peak_unrealized is not None and net_pnl is not None else None
    gross_return = pct(exit_, entry)
    validator = "OK"
    if gross_return is not None and gross_return > 0 and peak_pct is not None and peak_pct + 1e-9 < gross_return:
        validator = "PEAK_LT_GROSS_RETURN"
        notes.append("profitable_trade_peak_below_gross_return")
    return PeakMetrics(
        peak_price=float(peak_price) if peak_price is not None else None,
        low_price=float(low_price) if low_price is not None else None,
        peak_time=iso_ts(peak_time),
        low_time=iso_ts(low_time),
        mfe_pct=peak_pct,
        mae_pct=low_pct,
        peak_unrealized_pnl=peak_unrealized,
        max_adverse_unrealized_pnl=max_adverse,
        giveback_from_peak=giveback,
        drop_from_peak_pct=drop_from_peak,
        peak_data_quality=quality,
        peak_source="canonical_trade_candles_1m",
        validator_status=validator,
        notes=";".join(notes),
        candles_found=len(rows),
        candles_window_rows=len(window),
        candles_min_time_utc=candles_min.isoformat(),
        candles_max_time_utc=candles_max.isoformat(),
    )


def build_peak_plan(sqlite_path: Path, *, date: str, symbol: str | None, history_dir: Path, session_type: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with connect_sqlite(sqlite_path, read_only=True) as conn:
        trades = selected_trade_rows(conn, date=date, symbol=symbol)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    for trade in trades:
        sym = str(trade.get("symbol") or "").upper()
        entry_time = parse_dt(trade.get("entry_fill_time"))
        exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
        if entry_time is not None and exit_time is not None:
            cache_key = (sym, entry_time.date().isoformat(), exit_time.date().isoformat())
            candles = cache.get(cache_key)
            if candles is None:
                candles = load_trade_candles(history_dir, sym, entry_time, exit_time, session_type)
                cache[cache_key] = candles
        else:
            candles = pd.DataFrame()
        metrics = calculate_peak_metrics(
            candles,
            entry_time=trade.get("entry_fill_time"),
            exit_time=trade.get("exit_fill_time") or trade.get("closed_at"),
            entry_price=fnum(trade.get("entry_price")),
            exit_price=fnum(trade.get("exit_price")),
            quantity=fnum(trade.get("quantity")),
            net_pnl=fnum(trade.get("net_pnl")),
        )
        old_peak = fnum(trade.get("peak_price"))
        old_mfe = fnum(trade.get("mfe_pct"))
        gross_return = pct(fnum(trade.get("exit_price")), fnum(trade.get("entry_price")))
        rows.append(
            {
                "trade_id": trade.get("trade_id"),
                "symbol": sym,
                "session_date": trade.get("session_date"),
                "entry_time": trade.get("entry_fill_time"),
                "exit_time": trade.get("exit_fill_time") or trade.get("closed_at"),
                "quantity": trade.get("quantity"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "gross_pnl": trade.get("gross_pnl"),
                "net_pnl": trade.get("net_pnl"),
                "gross_return_pct": gross_return,
                "old_peak_price": old_peak,
                "old_peak_pct": old_mfe,
                "new_peak_price": metrics.peak_price,
                "new_peak_pct": metrics.mfe_pct,
                "new_low_price": metrics.low_price,
                "new_mae_pct": metrics.mae_pct,
                "new_peak_unrealized_pnl": metrics.peak_unrealized_pnl,
                "new_max_adverse_unrealized_pnl": metrics.max_adverse_unrealized_pnl,
                "new_giveback_from_peak": metrics.giveback_from_peak,
                "new_drop_from_peak_pct": metrics.drop_from_peak_pct,
                "peak_time": metrics.peak_time,
                "low_time": metrics.low_time,
                "peak_data_quality": metrics.peak_data_quality,
                "peak_source": metrics.peak_source,
                "validator_status": metrics.validator_status,
                "notes": metrics.notes,
                "candles_found": metrics.candles_found,
                "candles_window_rows": metrics.candles_window_rows,
                "candles_min_time_utc": metrics.candles_min_time_utc,
                "candles_max_time_utc": metrics.candles_max_time_utc,
                "needs_update": int(
                    old_peak != metrics.peak_price
                    or old_mfe != metrics.mfe_pct
                    or parse_raw_json(trade.get("raw_json")).get("peak_rebuild_status") != "rebuilt_from_candles"
                ),
            }
        )
    summary = {
        "date": date,
        "trades": len(rows),
        "ok": sum(1 for row in rows if row["peak_data_quality"] == "OK"),
        "incomplete": sum(1 for row in rows if row["peak_data_quality"] == "INCOMPLETE"),
        "missing": sum(1 for row in rows if row["peak_data_quality"] == "MISSING"),
        "validator_failures": sum(1 for row in rows if row["validator_status"] != "OK"),
        "profitable_peak_missing": sum(1 for row in rows if (fnum(row.get("gross_return_pct")) or 0) > 0 and row.get("new_peak_pct") is None),
        "updates": sum(int(row.get("needs_update") or 0) for row in rows),
    }
    return rows, summary


def write_reports(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, suffix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"trade_peak_rebuild_dry_run_{suffix}.csv"
    summary_path = output_dir / f"trade_peak_rebuild_dry_run_{suffix}.md"
    fieldnames = list(rows[0].keys()) if rows else ["trade_id", "symbol", "peak_data_quality"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Trade Peak Rebuild Dry Run",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Problem Rows",
        "",
    ]
    problems = [row for row in rows if row.get("peak_data_quality") != "OK" or row.get("validator_status") != "OK"]
    if problems:
        for row in problems[:100]:
            lines.append(
                f"- {row.get('symbol')} {row.get('trade_id')}: "
                f"quality={row.get('peak_data_quality')} validator={row.get('validator_status')} notes={row.get('notes')}"
            )
    else:
        lines.append("- none")
    summary_path.write_text("\n".join(lines) + "\n")
    return csv_path, summary_path


def update_trade_peak(conn: sqlite3.Connection, trade_id: str, metrics_row: dict[str, Any]) -> None:
    existing = conn.execute("SELECT raw_json FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    raw = parse_raw_json(existing["raw_json"] if existing else None)
    raw.update(
        {
            "peak_rebuild_status": "rebuilt_from_candles" if metrics_row["peak_data_quality"] == "OK" else "needs_rebuild",
            "peak_rebuild_version": PEAK_REBUILD_VERSION,
            "peak_data_quality": metrics_row["peak_data_quality"],
            "peak_source": metrics_row["peak_source"],
            "peak_time": metrics_row["peak_time"],
            "low_time": metrics_row["low_time"],
            "drop_from_peak_pct": metrics_row["new_drop_from_peak_pct"],
            "peak_validator_status": metrics_row["validator_status"],
            "peak_rebuild_notes": metrics_row["notes"],
            "peak_rebuilt_at": datetime.now(timezone.utc).isoformat(),
            "stale_peak_position_key_ignored": True,
        }
    )
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
            metrics_row["new_peak_pct"],
            metrics_row["new_mae_pct"],
            metrics_row["new_peak_price"],
            metrics_row["new_low_price"],
            metrics_row["new_peak_unrealized_pnl"],
            metrics_row["new_max_adverse_unrealized_pnl"],
            metrics_row["new_giveback_from_peak"],
            json.dumps(raw, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
            trade_id,
        ),
    )


def apply_peak_rebuild(sqlite_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    enforce_apply_guard(sqlite_path)
    backup_path = sqlite_path.with_suffix(sqlite_path.suffix + f".backup_peak_{utc_stamp()}")
    shutil.copy2(sqlite_path, backup_path)
    applied = 0
    with connect_sqlite(sqlite_path, read_only=False) as conn:
        try:
            conn.execute("BEGIN")
            for row in rows:
                if not row.get("trade_id"):
                    continue
                update_trade_peak(conn, str(row["trade_id"]), row)
                applied += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {"backup_path": str(backup_path), "updated_trades": applied}


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild canonical closed-trade peak/giveback metrics from 1m candles.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbol")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_DB))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--apply", action="store_true", help="Apply to SQLite. Default is dry-run.")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path)
    suffix = f"{args.date}_{args.symbol.upper()}" if args.symbol else args.date
    rows, summary = build_peak_plan(
        sqlite_path,
        date=args.date,
        symbol=args.symbol,
        history_dir=Path(args.history_dir),
        session_type=args.session_type,
    )
    csv_path, summary_path = write_reports(rows, summary, Path(args.output_dir), suffix)
    print(
        "TRADE_PEAK_REBUILD_DRY_RUN "
        f"date={args.date} trades={summary['trades']} ok={summary['ok']} "
        f"incomplete={summary['incomplete']} missing={summary['missing']} "
        f"validator_failures={summary['validator_failures']} updates={summary['updates']} "
        f"output={csv_path} summary={summary_path}",
        flush=True,
    )
    if not args.apply:
        return 0
    result = apply_peak_rebuild(sqlite_path, rows)
    print("TRADE_PEAK_REBUILD_APPLIED " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
