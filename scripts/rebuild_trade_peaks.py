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
PEAK_REBUILD_VERSION = 2
VALID_PEAK_QUALITIES = {"EXACT", "PARTIAL"}


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
        return ("OK" if not reasons else "FAIL", ";".join(reasons))
    if peak_time is None:
        reasons.append("missing_peak_time")
    else:
        if peak_time < entry_time:
            reasons.append("peak_time_before_entry")
        if peak_time > exit_time:
            reasons.append("peak_time_after_exit")
    if gross_return_pct is not None and gross_return_pct > 0:
        if peak_pct is None or peak_pct + 0.01 < gross_return_pct:
            reasons.append("profitable_long_peak_pct_below_gross_return")
        if peak_price + 1e-9 < entry_price:
            reasons.append("profitable_long_peak_price_below_entry")
    if peak_price == 0:
        reasons.append("zero_peak_price_is_not_missing_data")
    return ("OK" if not reasons else "FAIL", ";".join(reasons))


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
    peak_price = fnum(window.loc[peak_idx, "high"])
    low_price = fnum(window.loc[low_idx, "low"])
    peak_time = parse_dt(window.loc[peak_idx, "timestamp"])
    low_time = parse_dt(window.loc[low_idx, "timestamp"])
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


def build_peak_rows(sqlite_path: Path, *, date: str | None, trade_id: str | None, history_dir: Path, session_type: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                cache[key] = load_trade_candles(history_dir, symbol, entry_time, exit_time, session_type)
            candles = cache[key]
        peak = calculate_peak(candles, trade=trade)
        old_peak_price = fnum(trade.get("peak_price"))
        old_peak_pct = fnum(trade.get("mfe_pct"))
        old_giveback = fnum(trade.get("giveback_from_peak"))
        gross_return_pct = pct(fnum(trade.get("exit_price")), fnum(trade.get("entry_price")))
        raw = parse_raw_json(trade.get("raw_json"))
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
    summary = {
        "trades_scanned": len(rows),
        "exact": sum(1 for row in rows if row["peak_data_quality"] == "EXACT"),
        "partial": sum(1 for row in rows if row["peak_data_quality"] == "PARTIAL"),
        "missing": sum(1 for row in rows if row["peak_data_quality"] in {"MISSING_CANDLES", "OUTSIDE_CANDLE_RANGE", "NEEDS_REBUILD"}),
        "profitable_peak_below_final_return": sum(1 for row in rows if row["validation_reason"] and "profitable_long_peak_pct_below_gross_return" in row["validation_reason"]),
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
    raw.update(
        {
            "peak_rebuild_status": "rebuilt_from_candles" if row["peak_data_quality"] in VALID_PEAK_QUALITIES else "needs_rebuild",
            "peak_rebuild_version": PEAK_REBUILD_VERSION,
            "peak_version": PEAK_REBUILD_VERSION,
            "peak_data_quality": row["peak_data_quality"],
            "peak_source": "canonical_trade_candles_1m" if row["peak_data_quality"] in VALID_PEAK_QUALITIES else "unavailable",
            "peak_time": row["peak_time"],
            "drop_from_peak_pct": row["drop_from_peak_pct"],
            "giveback_usd": row["giveback_usd"],
            "peak_validation_status": row["validation_status"],
            "peak_validation_reason": row["validation_reason"],
            "stale_peak_position_key_ignored": True,
            "peak_calculated_at": datetime.now(timezone.utc).isoformat(),
            "peak_rebuilt_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    peak_ok = row["peak_data_quality"] in VALID_PEAK_QUALITIES
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


def apply_rows(sqlite_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    enforce_apply_guard(sqlite_path)
    backup = backup_path(sqlite_path)
    shutil.copy2(sqlite_path, backup)
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
        session_type=args.session_type,
    )
    suffix = args.trade_id or args.date or "all"
    output = write_report(rows, summary, Path(args.output_dir), suffix, audit=audit)
    event = "TRADE_PEAK_AUDIT_DONE" if audit else "TRADE_PEAK_REBUILD_DRY_RUN"
    print(
        f"{event} trades_scanned={summary['trades_scanned']} exact={summary['exact']} "
        f"partial={summary['partial']} missing={summary['missing']} "
        f"profitable_peak_below_final_return={summary['profitable_peak_below_final_return']} "
        f"suspicious_peak_zero_values={summary['suspicious_peak_zero_values']} output={output}",
        flush=True,
    )
    if getattr(args, "apply", False):
        result = apply_rows(sqlite_path, rows)
        print("TRADE_PEAK_REBUILD_APPLIED " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--date")
    parser.add_argument("--trade-id")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_DB))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Explicit no-op alias; dry-run is the default.")
    parser.add_argument("--apply", action="store_true", help="Apply changes to SQLite. Default is dry-run.")
    return parser


def main() -> int:
    return run(build_parser("Rebuild canonical trade peak/giveback metrics from 1m candles.").parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
