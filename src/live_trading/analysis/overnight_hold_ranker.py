from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.bad_entries_analyzer import first_green_seconds, load_closed_trades
from src.live_trading.analysis.common import (
    calculate_path_stats,
    first_existing_column,
    fnum,
    load_session_candles,
    load_trade_candles,
    parse_dt,
    parse_raw_json,
    pct,
)
from src.live_trading.market_calendar import is_us_equity_trading_day
from src.live_trading.storage.sqlite_store import OVERNIGHT_HOLD_TRADE_COLUMNS


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")


def next_us_equity_trading_day(value: date | datetime | str) -> date:
    cur = pd.Timestamp(value).date() if not isinstance(value, date) else value
    cur += timedelta(days=1)
    while not is_us_equity_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def connect_sqlite(sqlite_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def ensure_overnight_columns(sqlite_path: str | Path) -> list[str]:
    path = Path(sqlite_path)
    if not path.exists():
        return []
    conn = connect_sqlite(path)
    added: list[str] = []
    try:
        existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        for column, col_type in OVERNIGHT_HOLD_TRADE_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {column} {col_type}")
                added.append(column)
        conn.commit()
    finally:
        conn.close()
    return added


def score_bucket(score: float) -> str:
    if score >= 70:
        return "strong_hold_candidate"
    if score >= 50:
        return "possible_hold_candidate"
    if score >= 30:
        return "weak_hold_candidate"
    return "avoid_overnight"


def overnight_score(features: dict[str, Any]) -> tuple[float, str, str]:
    score = 0.0
    reasons: list[str] = []

    def add(points: float, key: str) -> None:
        nonlocal score
        score += points
        reasons.append(key)

    if (features.get("next_session_high_from_entry_pct") or 0) >= 5:
        add(25, "next_high>=5")
    if (features.get("next_session_close_from_entry_pct") or 0) >= 2:
        add(15, "next_close>=2")
    if (features.get("next_session_open_gap_pct") or 0) >= 1:
        add(10, "gap>=1")
    if (features.get("mfe_pct") or 0) >= 3:
        add(10, "mfe>=3")
    if (features.get("live_entry_score") or 0) >= 30:
        add(10, "live_score>=30")
    rank = fnum(features.get("top100_rank"))
    if rank is not None and rank <= 25:
        add(10, "rank<=25")
    if (features.get("next_session_max_drawdown_from_entry_pct") or 0) <= -5:
        add(-25, "next_dd<=-5")
    if (features.get("next_session_close_from_entry_pct") or 0) <= -2:
        add(-15, "next_close<=-2")
    if (features.get("mae_pct") or 0) <= -5:
        add(-10, "mae<=-5")
    if int(features.get("never_green") or 0) == 1:
        add(-10, "never_green")
    if int(features.get("immediate_drop") or 0) == 1:
        add(-10, "immediate_drop")
    score = max(0.0, min(100.0, score))
    return score, score_bucket(score), ",".join(reasons)


def analyze_trade_overnight(
    trade: dict[str, Any],
    *,
    history_dir: Path,
    recorder_dir: Path,
    session_type: str,
) -> dict[str, Any]:
    raw = parse_raw_json(trade.get("raw_json"))
    symbol = str(trade.get("symbol") or "").upper()
    session_date = str(first_existing_column(trade, ["session_date"]) or "")
    entry_time = parse_dt(first_existing_column(trade, ["entry_fill_time", "entry_time"]) or raw.get("entry_time"))
    exit_time = parse_dt(first_existing_column(trade, ["exit_fill_time", "closed_at", "exit_time"]) or raw.get("exit_time"))
    entry_price = fnum(first_existing_column(trade, ["entry_price"]) or raw.get("entry_price"))
    exit_price = fnum(first_existing_column(trade, ["exit_price"]) or raw.get("exit_price"))
    next_day = next_us_equity_trading_day(session_date or (exit_time.strftime("%F") if exit_time is not None else datetime.now(timezone.utc).strftime("%F")))
    next_candles = load_session_candles(history_dir, symbol, next_day, session_type)
    trade_candles = load_trade_candles(history_dir, recorder_dir, symbol, session_date, entry_time, exit_time, session_type)
    path_stats = calculate_path_stats(trade_candles, entry_price or 0.0, entry_time)
    first_green = first_green_seconds(trade_candles, entry_price or 0.0, entry_time)
    next_open = fnum(next_candles.iloc[0].get("open")) if not next_candles.empty else None
    next_high = fnum(next_candles["high"].max()) if not next_candles.empty else None
    next_low = fnum(next_candles["low"].min()) if not next_candles.empty else None
    next_close = fnum(next_candles.iloc[-1].get("close")) if not next_candles.empty else None
    net_pnl = fnum(first_existing_column(trade, ["net_pnl"]))
    quantity = abs(fnum(trade.get("quantity"), 0.0) or 0.0)
    final_pnl_pct = (net_pnl / (entry_price * quantity) * 100.0) if net_pnl is not None and entry_price and quantity > 0 else None
    features = {
        "trade_id": trade.get("trade_id"),
        "symbol": symbol,
        "session_date": session_date,
        "next_session_date": next_day.strftime("%F"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "mfe_pct": fnum(trade.get("mfe_pct")) if fnum(trade.get("mfe_pct")) is not None else path_stats.mfe_pct,
        "mae_pct": fnum(trade.get("mae_pct")) if fnum(trade.get("mae_pct")) is not None else path_stats.mae_pct,
        "peak_pct": fnum(trade.get("peak_pct")) or path_stats.mfe_pct,
        "final_pnl_pct": final_pnl_pct,
        "top100_rank": first_existing_column(trade, ["top100_rank"]) or raw.get("top100_rank"),
        "top100_score": first_existing_column(trade, ["top100_score"]) or raw.get("top100_score"),
        "live_entry_score": first_existing_column(trade, ["live_entry_score"]) or raw.get("live_entry_score"),
        "live_entry_rank": first_existing_column(trade, ["live_entry_rank"]) or raw.get("live_entry_rank"),
        "first_green_seconds": first_green,
        "never_green": int(first_green is None),
        "immediate_drop": int((path_stats.mae_pct or 0.0) < 0.0 and first_green is None),
        "next_session_open": next_open,
        "next_session_high": next_high,
        "next_session_low": next_low,
        "next_session_close": next_close,
        "next_session_open_gap_pct": pct(next_open, exit_price),
        "next_session_high_from_entry_pct": pct(next_high, entry_price),
        "next_session_close_from_entry_pct": pct(next_close, entry_price),
        "next_session_max_drawdown_from_entry_pct": pct(next_low, entry_price),
    }
    score, bucket, reason = overnight_score(features)
    features["overnight_hold_score"] = score
    features["overnight_hold_bucket"] = bucket
    features["overnight_hold_reason"] = reason
    features["missing_next_session_data"] = int(next_candles.empty)
    return features


def rank_closed_trades(
    *,
    start_date: str,
    end_date: str,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    session_type: str,
) -> pd.DataFrame:
    trades = load_closed_trades(sqlite_path, start_date, end_date)
    rows = [
        analyze_trade_overnight(row, history_dir=history_dir, recorder_dir=recorder_dir, session_type=session_type)
        for row in trades.to_dict("records")
    ]
    return pd.DataFrame(rows)


def update_trades(sqlite_path: Path, rankings: pd.DataFrame) -> int:
    if rankings.empty:
        return 0
    ensure_overnight_columns(sqlite_path)
    conn = connect_sqlite(sqlite_path)
    updated = 0
    try:
        for row in rankings.to_dict("records"):
            trade_id = row.get("trade_id")
            if not trade_id:
                continue
            features_json = json.dumps(row, default=str, sort_keys=True)
            conn.execute(
                """
                UPDATE trades
                SET overnight_hold_score = ?,
                    overnight_hold_bucket = ?,
                    overnight_hold_reason = ?,
                    overnight_hold_features_json = ?,
                    next_session_open = ?,
                    next_session_high = ?,
                    next_session_low = ?,
                    next_session_close = ?,
                    next_session_open_gap_pct = ?,
                    next_session_high_from_entry_pct = ?,
                    next_session_close_from_entry_pct = ?,
                    next_session_max_drawdown_from_entry_pct = ?,
                    overnight_hold_updated_at = ?
                WHERE trade_id = ?
                """,
                [
                    row.get("overnight_hold_score"),
                    row.get("overnight_hold_bucket"),
                    row.get("overnight_hold_reason"),
                    features_json,
                    row.get("next_session_open"),
                    row.get("next_session_high"),
                    row.get("next_session_low"),
                    row.get("next_session_close"),
                    row.get("next_session_open_gap_pct"),
                    row.get("next_session_high_from_entry_pct"),
                    row.get("next_session_close_from_entry_pct"),
                    row.get("next_session_max_drawdown_from_entry_pct"),
                    datetime.now(timezone.utc).isoformat(),
                    trade_id,
                ],
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    return updated


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    scored = int(pd.to_numeric(df.get("overnight_hold_score", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not df.empty else 0
    missing = int(pd.to_numeric(df.get("missing_next_session_data", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not df.empty else 0
    avg_score = pd.to_numeric(df.get("overnight_hold_score", pd.Series(dtype=float)), errors="coerce").mean() if not df.empty else 0.0
    print(f"OVERNIGHT_HOLD_RANKING total_closed_trades={total} scored_trades={scored} missing_next_session_data={missing} avg_overnight_hold_score={avg_score:.2f}")
    if df.empty:
        return
    print("bucket_counts=" + ", ".join(f"{k}:{v}" for k, v in df["overnight_hold_bucket"].fillna("missing").value_counts().to_dict().items()))
    grouped = df.groupby("overnight_hold_bucket", dropna=False).agg(
        avg_next_high_from_entry_pct=("next_session_high_from_entry_pct", "mean"),
        avg_next_close_from_entry_pct=("next_session_close_from_entry_pct", "mean"),
        trades=("symbol", "count"),
    ).reset_index()
    print(grouped.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank closed trades for hypothetical overnight hold quality.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date")
    group.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = args.date or args.start_date
    end = args.date or args.end_date
    if not end:
        parser.error("--end-date is required with --start-date")
    output = args.output or Path(f"data/analysis/overnight_hold_rankings_{start if start == end else start + '_to_' + end}.csv")
    df = rank_closed_trades(
        start_date=start,
        end_date=end,
        sqlite_path=args.sqlite_path,
        history_dir=args.history_dir,
        recorder_dir=args.recorder_dir,
        session_type=args.session_type,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    updated = 0
    if not args.dry_run:
        updated = update_trades(args.sqlite_path, df)
    print_summary(df)
    print(f"output={output}")
    print(f"sqlite_updated_rows={updated} dry_run={int(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
