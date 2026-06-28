from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_path_stats,
    entry_minutes_after_open,
    entry_time_bucket,
    first_existing_column,
    fnum,
    iso_ts,
    load_trade_candles,
    min_after_pct,
    nearest_row,
    parse_dt,
    parse_raw_json,
    pct,
    read_sql_table,
    safe_read_csv,
    simulate_tp_sl,
)


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")

OUTPUT_COLUMNS = [
    "date",
    "trade_id",
    "symbol",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "quantity",
    "net_pnl",
    "net_pnl_pct",
    "exit_reason",
    "mfe_pct",
    "mae_pct",
    "peak_pct",
    "low_after_entry_pct",
    "immediate_drop",
    "never_green",
    "min_after_1m_pct",
    "min_after_3m_pct",
    "min_after_5m_pct",
    "min_after_10m_pct",
    "time_to_peak_seconds",
    "time_to_low_seconds",
    "max_drawdown_from_peak_pct",
    "entry_minutes_after_open",
    "entry_time_bucket",
    "signal_age_seconds",
    "signal_age_warning",
    "spread_bps_at_entry",
    "top100_rank",
    "top100_score",
    "live_entry_score",
    "live_entry_rank",
    "tp2_sl1_5_exit_reason",
    "tp2_sl1_5_pnl_pct",
    "tp3_sl2_exit_reason",
    "tp3_sl2_pnl_pct",
    "tp4_sl2_exit_reason",
    "tp4_sl2_pnl_pct",
]


def load_closed_trades(sqlite_path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    trades = read_sql_table(
        sqlite_path,
        "trades",
        where=(
            "UPPER(COALESCE(status, '')) = 'CLOSED' "
            "AND ("
            "(COALESCE(session_date, '') != '' AND session_date BETWEEN ? AND ?) "
            "OR (COALESCE(session_date, '') = '' AND substr(entry_fill_time, 1, 10) BETWEEN ? AND ?)"
            ")"
        ),
        params=[start_date, end_date, start_date, end_date],
        order_by="COALESCE(exit_fill_time, closed_at), symbol, trade_id",
    )
    return trades


def load_spread_snapshots(recorder_dir: Path, session_date: str) -> pd.DataFrame:
    root = recorder_dir / session_date
    for name in ["spread_snapshots.csv", "market_snapshots.csv"]:
        df = safe_read_csv(root / name)
        if not df.empty:
            return df
    print(f"SPREAD_DATA_MISSING date={session_date} reason=no_spread_or_market_snapshots")
    return pd.DataFrame()


def signal_age(row: dict[str, Any], entry_time: pd.Timestamp | None) -> tuple[float | None, str]:
    if entry_time is None:
        return None, ""
    raw = parse_raw_json(row.get("raw_json"))
    signal_time = first_existing_column(row, ["ready_since"]) or raw.get("ready_since")
    if signal_time in (None, ""):
        signal_time = first_existing_column(row, ["signal_time"]) or raw.get("signal_time")
    signal_ts = parse_dt(signal_time)
    if signal_ts is None:
        return None, ""
    age = (entry_time - signal_ts).total_seconds()
    if age < 0:
        return None, "negative_age"
    return age, ""


def row_value(row: dict[str, Any], raw: dict[str, Any], names: list[str]) -> Any:
    direct = first_existing_column(row, names)
    if direct not in (None, ""):
        return direct
    for name in names:
        value = raw.get(name)
        if value not in (None, ""):
            return value
    return None


def analyze_bad_entries(
    *,
    start_date: str,
    end_date: str,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    session_type: str = "RTH",
) -> pd.DataFrame:
    trades = load_closed_trades(sqlite_path, start_date, end_date)
    rows: list[dict[str, Any]] = []
    spread_cache: dict[str, pd.DataFrame] = {}
    for trade in trades.to_dict("records"):
        raw = parse_raw_json(trade.get("raw_json"))
        symbol = str(trade.get("symbol") or "").upper()
        entry_time = parse_dt(first_existing_column(trade, ["entry_fill_time", "entry_time"]) or raw.get("entry_time"))
        exit_time = parse_dt(first_existing_column(trade, ["exit_fill_time", "closed_at", "exit_time"]) or raw.get("exit_time"))
        session_date = str(first_existing_column(trade, ["session_date"]) or (exit_time.strftime("%F") if exit_time is not None else start_date))
        entry_price = fnum(first_existing_column(trade, ["entry_price", "buy_price"]) or raw.get("entry_price"))
        exit_price = fnum(first_existing_column(trade, ["exit_price", "sell_price"]) or raw.get("exit_price"))
        quantity = abs(fnum(trade.get("quantity"), 0.0) or 0.0)
        net_pnl = fnum(first_existing_column(trade, ["net_pnl", "net_actual"]))
        if net_pnl is None:
            gross = fnum(trade.get("gross_pnl"), 0.0) or 0.0
            commission = abs(fnum(trade.get("commission"), 0.0) or 0.0)
            net_pnl = gross - commission
        candles = load_trade_candles(history_dir, recorder_dir, symbol, session_date, entry_time, exit_time, session_type)
        stats = calculate_path_stats(candles, entry_price or 0.0, entry_time)
        fallback_exit_time = exit_time
        fallback_exit_price = exit_price
        sim_2 = simulate_tp_sl(candles, entry_price=entry_price or 0.0, tp_pct=2.0, sl_pct=-1.5, fallback_exit_time=fallback_exit_time, fallback_exit_price=fallback_exit_price)
        sim_3 = simulate_tp_sl(candles, entry_price=entry_price or 0.0, tp_pct=3.0, sl_pct=-2.0, fallback_exit_time=fallback_exit_time, fallback_exit_price=fallback_exit_price)
        sim_4 = simulate_tp_sl(candles, entry_price=entry_price or 0.0, tp_pct=4.0, sl_pct=-2.0, fallback_exit_time=fallback_exit_time, fallback_exit_price=fallback_exit_price)
        if session_date not in spread_cache:
            spread_cache[session_date] = load_spread_snapshots(recorder_dir, session_date)
        spread = nearest_row(spread_cache[session_date], entry_time, symbol)
        sig_age, sig_age_warning = signal_age(trade, entry_time)
        net_pnl_pct = None
        if entry_price and entry_price > 0 and quantity > 0 and net_pnl is not None:
            net_pnl_pct = net_pnl / (entry_price * quantity) * 100.0
        row = {
            "date": session_date,
            "trade_id": trade.get("trade_id"),
            "symbol": symbol,
            "entry_time": iso_ts(entry_time),
            "entry_price": entry_price,
            "exit_time": iso_ts(exit_time),
            "exit_price": exit_price,
            "quantity": quantity,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl_pct,
            "exit_reason": first_existing_column(trade, ["exit_reason"]) or raw.get("exit_reason") or "unknown_exit_reason",
            "mfe_pct": stats.mfe_pct,
            "mae_pct": stats.mae_pct,
            "peak_pct": stats.mfe_pct,
            "low_after_entry_pct": stats.mae_pct,
            "immediate_drop": int((min_after_pct(candles, entry_price or 0.0, entry_time, 1) or 0.0) < 0.0),
            "never_green": int((stats.mfe_pct or 0.0) <= 0.0),
            "min_after_1m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 1),
            "min_after_3m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 3),
            "min_after_5m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 5),
            "min_after_10m_pct": min_after_pct(candles, entry_price or 0.0, entry_time, 10),
            "time_to_peak_seconds": stats.time_to_peak_seconds,
            "time_to_low_seconds": stats.time_to_low_seconds,
            "max_drawdown_from_peak_pct": stats.max_drawdown_from_peak_pct,
            "entry_minutes_after_open": entry_minutes_after_open(entry_time, session_date),
            "entry_time_bucket": entry_time_bucket(entry_time, session_date),
            "signal_age_seconds": sig_age,
            "signal_age_warning": sig_age_warning,
            "spread_bps_at_entry": first_existing_column(spread, ["spread_bps", "bid_ask_spread_bps"]),
            "top100_rank": row_value(trade, raw, ["top100_rank"]),
            "top100_score": row_value(trade, raw, ["top100_score"]),
            "live_entry_score": row_value(trade, raw, ["live_entry_score", "entry_score", "score"]),
            "live_entry_rank": row_value(trade, raw, ["live_entry_rank", "ranking_position"]),
            "tp2_sl1_5_exit_reason": sim_2.exit_reason,
            "tp2_sl1_5_pnl_pct": sim_2.pnl_pct,
            "tp3_sl2_exit_reason": sim_3.exit_reason,
            "tp3_sl2_pnl_pct": sim_3.pnl_pct,
            "tp4_sl2_exit_reason": sim_4.exit_reason,
            "tp4_sl2_pnl_pct": sim_4.pnl_pct,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def score_bucket(value: Any) -> str:
    score = fnum(value)
    if score is None:
        return "missing"
    if score >= 80:
        return ">=80"
    if score >= 60:
        return "60-80"
    if score >= 40:
        return "40-60"
    if score >= 20:
        return "20-40"
    return "<20"


def rank_bucket(value: Any) -> str:
    rank = fnum(value)
    if rank is None:
        return "missing"
    if rank <= 10:
        return "1-10"
    if rank <= 25:
        return "11-25"
    if rank <= 50:
        return "26-50"
    if rank <= 75:
        return "51-75"
    if rank <= 100:
        return "76-100"
    return "missing"


def signal_age_bucket(value: Any, warning: Any = "") -> str:
    if str(warning or "") == "negative_age":
        return "invalid_negative"
    seconds = fnum(value)
    if seconds is None:
        return "missing"
    if seconds < 0:
        return "invalid_negative"
    if seconds <= 30:
        return "0-30s"
    if seconds <= 60:
        return "30-60s"
    if seconds <= 180:
        return "1-3m"
    if seconds <= 300:
        return "3-5m"
    return "5m+"


def generic_bucket(value: Any) -> str:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    return str(value)


def print_bucket_summary(df: pd.DataFrame, column: str) -> None:
    if df.empty or column not in df.columns:
        return
    tmp = df.copy()
    if "score" in column:
        tmp["_bucket"] = tmp[column].map(score_bucket)
    elif column == "top100_rank":
        tmp["_bucket"] = tmp[column].map(rank_bucket)
    elif column == "signal_age_seconds":
        warnings = tmp["signal_age_warning"] if "signal_age_warning" in tmp.columns else pd.Series([""] * len(tmp))
        tmp["_bucket"] = [signal_age_bucket(value, warning) for value, warning in zip(tmp[column], warnings)]
    else:
        tmp["_bucket"] = tmp[column].map(generic_bucket)
    grouped = tmp.groupby("_bucket", dropna=False).agg(
        trade_count=("symbol", "count"),
        win_rate=("net_pnl", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean() * 100.0) if len(values) else 0.0),
        avg_net_pnl=("net_pnl", "mean"),
        expectancy=("net_pnl", "mean"),
    )
    print(f"{column}_bucket_summary:")
    print(grouped.reset_index().to_string(index=False))


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    net = pd.to_numeric(df.get("net_pnl", pd.Series(dtype=float)), errors="coerce")
    mfe = pd.to_numeric(df.get("mfe_pct", pd.Series(dtype=float)), errors="coerce")
    mae = pd.to_numeric(df.get("mae_pct", pd.Series(dtype=float)), errors="coerce")
    print(
        "BAD_ENTRIES "
        f"total_trades={total} closed_trades={total} net_pnl={net.sum():.2f} "
        f"avg_mfe={mfe.mean() if not mfe.empty else 0:.2f} median_mfe={mfe.median() if not mfe.empty else 0:.2f} "
        f"avg_mae={mae.mean() if not mae.empty else 0:.2f} median_mae={mae.median() if not mae.empty else 0:.2f}"
    )
    if df.empty:
        return
    print(
        f"peak_ge_1={(mfe >= 1).sum()} peak_ge_2={(mfe >= 2).sum()} peak_ge_3={(mfe >= 3).sum()} "
        f"peak_eq_0={(mfe == 0).sum()} immediate_drop={int(df['immediate_drop'].sum())} never_green={int(df['never_green'].sum())}"
    )
    for column in ["live_entry_score", "top100_rank", "entry_time_bucket", "spread_bps_at_entry", "signal_age_seconds"]:
        print_bucket_summary(df, column)
    print("worst20_by_net_pnl_pct:")
    print(df.sort_values("net_pnl_pct", ascending=True)[["symbol", "entry_time", "net_pnl_pct", "mfe_pct", "mae_pct", "live_entry_score"]].head(20).to_string(index=False))
    print("best20_by_peak_pct:")
    print(df.sort_values("peak_pct", ascending=False)[["symbol", "entry_time", "peak_pct", "net_pnl_pct", "live_entry_score"]].head(20).to_string(index=False))
    actual = net.sum()
    for prefix in ["tp2_sl1_5", "tp3_sl2", "tp4_sl2"]:
        pnl = pd.to_numeric(df[f"{prefix}_pnl_pct"], errors="coerce")
        print(f"{prefix}_simulation avg_pnl_pct={pnl.mean():.2f} wins={(pnl > 0).sum()} losses={(pnl <= 0).sum()} actual_net_pnl={actual:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze bot entries that dropped after buy and simulate TP/SL variants.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single session/exit date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for date range, YYYY-MM-DD. Required with --start-date.")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--session-type", default="RTH")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = args.date or args.start_date
    end = args.date or args.end_date
    if not end:
        parser.error("--end-date is required with --start-date")
    output = args.output or Path(f"data/analysis/bad_entries_{start if start == end else start + '_to_' + end}.csv")
    df = analyze_bad_entries(
        start_date=start,
        end_date=end,
        sqlite_path=args.sqlite_path,
        history_dir=args.history_dir,
        recorder_dir=args.recorder_dir,
        session_type=args.session_type,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print_summary(df)
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
