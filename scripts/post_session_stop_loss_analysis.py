#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.live_trading.ranking.daily_top100_builder import normalize_history_df, parquet_path
from src.live_trading.storage.sqlite_store import resolve_sqlite_path


DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_STOP_LOSSES = (2.0, 3.0, 4.0, 5.0, 8.0)
APPROXIMATION_NOTE = (
    "1m candle path simulation is approximate: if both stop and a favorable move happen inside the same minute, "
    "the script assumes the stop was hit when low <= stop. Intraminute order is unknown."
)


def parse_dt(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


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


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def pct(price: float | None, entry: float | None) -> float | None:
    if price is None or entry is None or entry <= 0:
        return None
    return ((price / entry) - 1.0) * 100.0


def iso_date(value: pd.Timestamp | None) -> str:
    if value is None:
        return ""
    return value.strftime("%F")


def iso_ts(value: pd.Timestamp | None) -> str:
    if value is None:
        return ""
    return value.isoformat()


def sqlite_connect_readonly(sqlite_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(sqlite_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def selected_closed_trades(sqlite_path: str | Path, session_date: str, strategy: str | None = None) -> pd.DataFrame:
    clause = ""
    params: list[Any] = [session_date, session_date]
    if strategy:
        clause = "AND COALESCE(strategy_name, 'unknown') = ?"
        params.append(strategy)
    with sqlite_connect_readonly(sqlite_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT
                trade_id,
                strategy_name,
                session_date,
                symbol,
                status,
                entry_fill_time,
                exit_fill_time,
                closed_at,
                entry_price,
                exit_price,
                quantity,
                gross_pnl,
                commission,
                net_pnl,
                mfe_pct,
                mae_pct,
                peak_price,
                low_price,
                peak_unrealized_pnl,
                max_adverse_unrealized_pnl,
                giveback_from_peak,
                exit_reason,
                raw_json
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
              AND (
                substr(exit_fill_time, 1, 10) = ?
                OR substr(closed_at, 1, 10) = ?
              )
              {clause}
            ORDER BY COALESCE(exit_fill_time, closed_at), symbol, trade_id
            """,
            conn,
            params=params,
        )


def selected_executions(sqlite_path: str | Path, session_date: str, strategy: str | None = None) -> pd.DataFrame:
    clause = ""
    params: list[Any] = [session_date, session_date, session_date]
    if strategy:
        clause = "AND COALESCE(strategy_name, 'unknown') = ?"
        params.append(strategy)
    with sqlite_connect_readonly(sqlite_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT
                execution_id,
                trade_id,
                order_id,
                perm_id,
                strategy_name,
                session_date,
                symbol,
                side,
                quantity,
                price,
                executed_at,
                recorded_at,
                commission,
                commission_source,
                realized_pnl,
                raw_json
            FROM executions
            WHERE (
                session_date = ?
                OR substr(executed_at, 1, 10) = ?
                OR substr(recorded_at, 1, 10) = ?
            )
            {clause}
            ORDER BY COALESCE(executed_at, recorded_at), execution_id
            """,
            conn,
            params=params,
        )


def execution_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"BOT", "BUY", "BOUGHT"}:
        return "BUY"
    if side in {"SLD", "SELL", "SOLD"}:
        return "SELL"
    return side


def match_trade_executions(trade: dict[str, Any], executions: pd.DataFrame) -> pd.DataFrame:
    if executions.empty:
        return executions
    trade_id = str(trade.get("trade_id") or "")
    symbol = str(trade.get("symbol") or "").upper()
    entry_time = parse_dt(trade.get("entry_fill_time"))
    exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
    rows = executions.copy()
    if trade_id and "trade_id" in rows.columns:
        exact = rows[rows["trade_id"].fillna("").astype(str) == trade_id]
        if not exact.empty:
            return exact
    rows["symbol_norm"] = rows.get("symbol", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    rows["event_time"] = pd.to_datetime(rows.get("executed_at").fillna(rows.get("recorded_at")), errors="coerce", utc=True)
    matched = rows[rows["symbol_norm"] == symbol]
    if entry_time is not None and exit_time is not None:
        matched = matched[(matched["event_time"] >= entry_time) & (matched["event_time"] <= exit_time)]
    return matched


def execution_diagnostics(trade: dict[str, Any], executions: pd.DataFrame) -> dict[str, Any]:
    matched = match_trade_executions(trade, executions)
    if matched.empty:
        return {
            "execution_count": 0,
            "buy_execution_count": 0,
            "sell_execution_count": 0,
            "execution_commission": 0.0,
            "sell_realized_pnl": 0.0,
            "execution_ids": "",
        }
    sides = matched.get("side", pd.Series(dtype=str)).map(execution_side)
    commissions = pd.to_numeric(matched.get("commission", pd.Series(dtype=float)), errors="coerce").abs().fillna(0.0)
    realized = pd.to_numeric(matched.get("realized_pnl", pd.Series(dtype=float)), errors="coerce")
    sell_mask = sides == "SELL"
    return {
        "execution_count": int(len(matched)),
        "buy_execution_count": int((sides == "BUY").sum()),
        "sell_execution_count": int(sell_mask.sum()),
        "execution_commission": float(commissions.sum()),
        "sell_realized_pnl": float(realized[sell_mask].dropna().sum()),
        "execution_ids": ",".join(str(x) for x in matched.get("execution_id", pd.Series(dtype=str)).dropna().tolist()),
    }


def load_trade_candles(
    history_dir: Path,
    symbol: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    session_type: str,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    cursor = entry_time.date()
    end_date = exit_time.date()
    while cursor <= end_date:
        path = parquet_path(history_dir, symbol, cursor, session_type)
        if path.exists():
            try:
                frames.append(normalize_history_df(pd.read_parquet(path)))
            except Exception:
                missing.append(str(path))
        else:
            missing.append(str(path))
        cursor += timedelta(days=1)
    if not frames:
        return pd.DataFrame(), missing
    candles = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    candles = candles[(candles["timestamp"] >= entry_time) & (candles["timestamp"] <= exit_time)]
    return candles.reset_index(drop=True), missing


@dataclass
class PathStats:
    peak_price: float | None = None
    low_price: float | None = None
    peak_time: pd.Timestamp | None = None
    low_time: pd.Timestamp | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    giveback_from_peak_pct: float | None = None
    time_to_peak_minutes: float | None = None
    time_to_low_minutes: float | None = None


def calculate_path_stats(candles: pd.DataFrame, entry_price: float, exit_price: float, entry_time: pd.Timestamp) -> PathStats:
    if candles.empty or entry_price <= 0:
        fallback_peak = max(entry_price, exit_price)
        fallback_low = min(entry_price, exit_price)
        return PathStats(
            peak_price=fallback_peak,
            low_price=fallback_low,
            mfe_pct=pct(fallback_peak, entry_price),
            mae_pct=pct(fallback_low, entry_price),
            giveback_from_peak_pct=pct(exit_price, fallback_peak),
        )
    peak_idx = candles["high"].astype(float).idxmax()
    low_idx = candles["low"].astype(float).idxmin()
    peak_price = float(candles.loc[peak_idx, "high"])
    low_price = float(candles.loc[low_idx, "low"])
    peak_time = candles.loc[peak_idx, "timestamp"]
    low_time = candles.loc[low_idx, "timestamp"]
    return PathStats(
        peak_price=peak_price,
        low_price=low_price,
        peak_time=peak_time,
        low_time=low_time,
        mfe_pct=pct(peak_price, entry_price),
        mae_pct=pct(low_price, entry_price),
        giveback_from_peak_pct=pct(exit_price, peak_price),
        time_to_peak_minutes=(peak_time - entry_time).total_seconds() / 60.0,
        time_to_low_minutes=(low_time - entry_time).total_seconds() / 60.0,
    )


def simulate_stop(
    candles: pd.DataFrame,
    *,
    entry_price: float,
    actual_exit_price: float,
    actual_exit_time: pd.Timestamp,
    quantity: float,
    commission: float,
    stop_loss_pct: float,
) -> dict[str, Any]:
    stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    stop_hit_rows = candles[candles["low"].astype(float) <= stop_price] if not candles.empty else pd.DataFrame()
    if not stop_hit_rows.empty:
        first = stop_hit_rows.iloc[0]
        exit_time = first["timestamp"]
        exit_price = stop_price
        stop_hit = True
    else:
        exit_time = actual_exit_time
        exit_price = actual_exit_price
        stop_hit = False
    gross = (float(exit_price) - entry_price) * quantity
    net = gross - abs(commission)
    return {
        "stop_loss_pct": stop_loss_pct,
        "simulated_stop_hit": int(stop_hit),
        "simulated_exit_time": iso_ts(exit_time),
        "simulated_exit_price": float(exit_price),
        "simulated_gross_pnl": gross,
        "simulated_net_pnl": net,
    }


def analyze_trades(
    trades: pd.DataFrame,
    *,
    executions: pd.DataFrame | None = None,
    history_dir: Path,
    session_type: str,
    stop_losses: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for row in trades.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        entry_time = parse_dt(row.get("entry_fill_time"))
        exit_time = parse_dt(row.get("exit_fill_time") or row.get("closed_at"))
        entry_price = fnum(row.get("entry_price"))
        exit_price = fnum(row.get("exit_price"))
        quantity = abs(fnum(row.get("quantity"), 0.0) or 0.0)
        commission = abs(fnum(row.get("commission"), 0.0) or 0.0)
        exec_diag = execution_diagnostics(row, executions if executions is not None else pd.DataFrame())
        if commission <= 0 and exec_diag["execution_commission"] > 0:
            commission = float(exec_diag["execution_commission"])
        actual_net = fnum(row.get("net_pnl"))
        symbol = str(row.get("symbol") or "").upper()
        if entry_time is None or exit_time is None or entry_price is None or exit_price is None or quantity <= 0:
            candles = pd.DataFrame()
            missing_history = ["missing_entry_or_exit"]
            stats = PathStats()
        else:
            candles, missing_history = load_trade_candles(history_dir, symbol, entry_time, exit_time, session_type)
            stats = calculate_path_stats(candles, entry_price, exit_price, entry_time)
        trade_base = {
            "trade_id": row.get("trade_id"),
            "session_date": row.get("session_date"),
            "exit_date": iso_date(exit_time),
            "symbol": symbol,
            "strategy": row.get("strategy_name"),
            "entry_time": iso_ts(entry_time),
            "exit_time": iso_ts(exit_time),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "commission": commission,
            "execution_count": exec_diag["execution_count"],
            "buy_execution_count": exec_diag["buy_execution_count"],
            "sell_execution_count": exec_diag["sell_execution_count"],
            "execution_commission": exec_diag["execution_commission"],
            "sell_realized_pnl": exec_diag["sell_realized_pnl"],
            "execution_ids": exec_diag["execution_ids"],
            "actual_exit_reason": row.get("exit_reason") or raw.get("exit_reason") or "unknown_exit_reason",
            "actual_gross_pnl": fnum(row.get("gross_pnl")),
            "actual_net_pnl": actual_net,
            "sqlite_mfe_pct": fnum(row.get("mfe_pct")),
            "sqlite_mae_pct": fnum(row.get("mae_pct")),
            "peak_price_since_entry": stats.peak_price,
            "low_price_since_entry": stats.low_price,
            "mfe_pct": stats.mfe_pct,
            "mae_pct": stats.mae_pct,
            "giveback_from_peak_pct": stats.giveback_from_peak_pct,
            "time_to_peak_minutes": stats.time_to_peak_minutes,
            "time_to_low_minutes": stats.time_to_low_minutes,
            "peak_time": iso_ts(stats.peak_time),
            "low_time": iso_ts(stats.low_time),
            "candles_used": int(len(candles)),
            "missing_history_files": ";".join(missing_history),
            "simulation_note": APPROXIMATION_NOTE,
        }
        trade_rows.append(trade_base)
        for stop_loss in stop_losses:
            sim = simulate_stop(
                candles,
                entry_price=entry_price or 0.0,
                actual_exit_price=exit_price or 0.0,
                actual_exit_time=exit_time or pd.Timestamp.now(tz=timezone.utc),
                quantity=quantity,
                commission=commission,
                stop_loss_pct=stop_loss,
            )
            variant_rows.append({
                **{k: trade_base[k] for k in ("trade_id", "symbol", "strategy", "entry_time", "exit_time", "entry_price", "exit_price", "quantity", "commission", "execution_count", "execution_commission", "actual_net_pnl")},
                **sim,
                "net_vs_actual": (sim["simulated_net_pnl"] - actual_net) if actual_net is not None else None,
            })
    variants = pd.DataFrame(variant_rows)
    if variants.empty:
        summary = pd.DataFrame(columns=[
            "stop_loss_pct", "trades", "stop_hits", "total_net_pnl", "average_net_pnl_per_trade",
            "win_rate", "max_loss_per_trade", "actual_total_net_pnl", "net_vs_actual",
        ])
    else:
        grouped = variants.groupby("stop_loss_pct", dropna=False)
        summary = grouped.agg(
            trades=("trade_id", "count"),
            stop_hits=("simulated_stop_hit", "sum"),
            total_net_pnl=("simulated_net_pnl", "sum"),
            average_net_pnl_per_trade=("simulated_net_pnl", "mean"),
            wins=("simulated_net_pnl", lambda values: int((values > 0).sum())),
            max_loss_per_trade=("simulated_net_pnl", "min"),
            actual_total_net_pnl=("actual_net_pnl", "sum"),
            net_vs_actual=("net_vs_actual", "sum"),
        ).reset_index()
        summary["win_rate"] = (summary["wins"] / summary["trades"] * 100.0).fillna(0.0)
        summary = summary.drop(columns=["wins"])
    return pd.DataFrame(trade_rows), variants, summary


def parse_stop_losses(value: str) -> tuple[float, ...]:
    out = []
    for part in str(value or "").split(","):
        parsed = fnum(part.strip())
        if parsed is not None and parsed > 0:
            out.append(float(parsed))
    return tuple(out or DEFAULT_STOP_LOSSES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-session stop-loss analysis from SQLite trades/executions and 1m parquet candles.")
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%F"), help="Exit/session date to analyze, YYYY-MM-DD.")
    parser.add_argument("--sqlite-path", default=resolve_sqlite_path(None))
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--stop-losses", default=",".join(str(x).rstrip("0").rstrip(".") for x in DEFAULT_STOP_LOSSES))
    args = parser.parse_args()

    stop_losses = parse_stop_losses(args.stop_losses)
    trades = selected_closed_trades(args.sqlite_path, args.date, args.strategy)
    executions = selected_executions(args.sqlite_path, args.date, args.strategy)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.date.replace("-", "")
    trades_out = args.reports_dir / f"post_session_stop_loss_trades_{stamp}.csv"
    variants_out = args.reports_dir / f"post_session_stop_loss_variants_{stamp}.csv"
    summary_out = args.reports_dir / f"post_session_stop_loss_summary_{stamp}.csv"

    if trades.empty:
        pd.DataFrame().to_csv(trades_out, index=False)
        pd.DataFrame().to_csv(variants_out, index=False)
        pd.DataFrame().to_csv(summary_out, index=False)
        print(f"POST_SESSION_STOP_LOSS_ANALYSIS date={args.date} trades=0 output={trades_out}")
        return 0

    trade_rows, variants, summary = analyze_trades(
        trades,
        executions=executions,
        history_dir=args.history_dir,
        session_type=args.session_type,
        stop_losses=stop_losses,
    )
    trade_rows.to_csv(trades_out, index=False)
    variants.to_csv(variants_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"POST_SESSION_STOP_LOSS_ANALYSIS date={args.date} trades={len(trade_rows)} variants={len(variants)}")
    print(f"approximation_note={APPROXIMATION_NOTE}")
    print(f"trades_csv={trades_out}")
    print(f"variants_csv={variants_out}")
    print(f"summary_csv={summary_out}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
