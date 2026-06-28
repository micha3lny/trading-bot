from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.bad_entries_analyzer import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_RECORDER_DIR,
    DEFAULT_SQLITE_PATH,
    augment_trades_from_executions,
    choose_trade_candles,
    load_closed_trades,
    load_executions,
    match_buy_execution,
    resolve_entry_time,
)
from src.live_trading.analysis.common import (
    calculate_path_stats,
    first_existing_column,
    fnum,
    iso_ts,
    load_recorder_candles,
    load_session_candles,
    min_after_pct,
    parse_dt,
    parse_raw_json,
    pct,
)


OUTPUT_COLUMNS = [
    "date",
    "trade_id",
    "analysis_source",
    "logical_trade_group",
    "symbol",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "quantity",
    "net_pnl",
    "net_pnl_pct",
    "mfe_pct",
    "mae_pct",
    "immediate_drop",
    "never_green",
    "min_after_1m_pct",
    "min_after_2m_pct",
    "min_after_5m_pct",
    "min_after_10m_pct",
    "max_after_5m_pct",
    "first_green_seconds",
    "top100_rank",
    "live_entry_score",
    "candle_source",
    "candle_coverage_warning",
    "pre_entry_window_high_1m",
    "pre_entry_window_high_3m",
    "pre_entry_window_high_5m",
    "pre_entry_window_high_10m",
    "pre_entry_pullback_from_1m_high_pct",
    "pre_entry_pullback_from_3m_high_pct",
    "pre_entry_pullback_from_5m_high_pct",
    "pre_entry_pullback_from_10m_high_pct",
    "entry_vs_previous_candle_close_pct",
    "entry_vs_previous_candle_high_pct",
    "entry_after_green_candle_count_3m",
    "entry_after_red_candle_count_3m",
    "entry_local_momentum_3m_pct",
    "entry_local_momentum_5m_pct",
    "entry_chasing_score",
    "pb_0_5_would_enter",
    "pb_0_5_entry_time",
    "pb_0_5_entry_price",
    "pb_0_5_mfe_pct",
    "pb_0_5_mae_pct",
    "pb_0_5_min_after_5m_pct",
    "pb_1_0_would_enter",
    "pb_1_0_entry_time",
    "pb_1_0_entry_price",
    "pb_1_0_mfe_pct",
    "pb_1_0_mae_pct",
    "pb_1_0_min_after_5m_pct",
    "pb_1_5_would_enter",
    "pb_1_5_entry_time",
    "pb_1_5_entry_price",
    "pb_1_5_mfe_pct",
    "pb_1_5_mae_pct",
    "pb_1_5_min_after_5m_pct",
    "pb_2_0_would_enter",
    "pb_2_0_entry_time",
    "pb_2_0_entry_price",
    "pb_2_0_mfe_pct",
    "pb_2_0_mae_pct",
    "pb_2_0_min_after_5m_pct",
]


def prior_window(candles: pd.DataFrame, entry_time: pd.Timestamp | None, minutes: int) -> pd.DataFrame:
    if candles.empty or entry_time is None or "timestamp" not in candles.columns:
        return pd.DataFrame()
    return candles[(candles["timestamp"] < entry_time) & (candles["timestamp"] >= entry_time - pd.Timedelta(minutes=minutes))]


def candles_after(candles: pd.DataFrame, entry_time: pd.Timestamp | None, exit_time: pd.Timestamp | None = None) -> pd.DataFrame:
    if candles.empty or entry_time is None or "timestamp" not in candles.columns:
        return pd.DataFrame()
    out = candles[candles["timestamp"] >= entry_time]
    if exit_time is not None:
        out = out[out["timestamp"] <= exit_time]
    return out.sort_values("timestamp").reset_index(drop=True)


def pre_entry_high(candles: pd.DataFrame, entry_time: pd.Timestamp | None, minutes: int) -> float | None:
    rows = prior_window(candles, entry_time, minutes)
    if rows.empty or "high" not in rows.columns:
        return None
    return fnum(pd.to_numeric(rows["high"], errors="coerce").max())


def pullback_from_recent_high(entry_price: float | None, recent_high: float | None) -> float | None:
    return pct(entry_price, recent_high)


def previous_candle(candles: pd.DataFrame, entry_time: pd.Timestamp | None) -> dict[str, Any]:
    rows = prior_window(candles, entry_time, 60)
    if rows.empty:
        return {}
    return rows.sort_values("timestamp").iloc[-1].to_dict()


def candle_color_counts(candles: pd.DataFrame, entry_time: pd.Timestamp | None, minutes: int = 3) -> tuple[int, int]:
    rows = prior_window(candles, entry_time, minutes)
    if rows.empty:
        return 0, 0
    opens = pd.to_numeric(rows.get("open", pd.Series(dtype=float)), errors="coerce")
    closes = pd.to_numeric(rows.get("close", pd.Series(dtype=float)), errors="coerce")
    green = int((closes > opens).sum())
    red = int((closes < opens).sum())
    return green, red


def local_momentum(candles: pd.DataFrame, entry_price: float | None, entry_time: pd.Timestamp | None, minutes: int) -> float | None:
    if entry_price is None or entry_time is None:
        return None
    rows = candles[candles["timestamp"] <= entry_time - pd.Timedelta(minutes=minutes)] if not candles.empty and "timestamp" in candles.columns else pd.DataFrame()
    if rows.empty:
        return None
    close = fnum(rows.sort_values("timestamp").iloc[-1].get("close"))
    return pct(entry_price, close)


def entry_chasing_score(
    *,
    pullback_5m_pct: float | None,
    momentum_3m_pct: float | None,
    entry_vs_prev_high_pct: float | None,
    green_count_3m: int,
    min_after_5m_pct: float | None,
) -> float:
    score = 0.0
    if pullback_5m_pct is not None and pullback_5m_pct >= -0.25:
        score += 30.0
    if momentum_3m_pct is not None and momentum_3m_pct >= 2.0:
        score += 20.0
    if entry_vs_prev_high_pct is not None and entry_vs_prev_high_pct >= -0.10:
        score += 20.0
    if green_count_3m >= 2:
        score += 15.0
    if min_after_5m_pct is not None and min_after_5m_pct <= -1.0:
        score += 15.0
    return max(0.0, min(100.0, score))


def first_green_seconds(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None) -> float | None:
    if candles.empty or entry_time is None or entry_price <= 0:
        return None
    rows = candles[candles["timestamp"] >= entry_time].sort_values("timestamp")
    green = rows[(pd.to_numeric(rows["close"], errors="coerce") > entry_price) | (pd.to_numeric(rows["high"], errors="coerce") > entry_price)]
    if green.empty:
        return None
    return (green.iloc[0]["timestamp"] - entry_time).total_seconds()


def max_after_pct(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, minutes: int) -> float | None:
    if candles.empty or entry_time is None or entry_price <= 0:
        return None
    rows = candles[(candles["timestamp"] >= entry_time) & (candles["timestamp"] <= entry_time + pd.Timedelta(minutes=minutes))]
    if rows.empty:
        return None
    return pct(fnum(pd.to_numeric(rows["high"], errors="coerce").max()), entry_price)


def simulate_pullback_entry(
    candles: pd.DataFrame,
    *,
    original_entry_time: pd.Timestamp | None,
    original_entry_price: float | None,
    exit_time: pd.Timestamp | None,
    pullback_pct: float,
) -> dict[str, Any]:
    if candles.empty or original_entry_time is None or original_entry_price is None or original_entry_price <= 0:
        return {"would_enter": 0, "entry_time": "", "entry_price": None, "mfe_pct": None, "mae_pct": None, "min_after_5m_pct": None}
    target = original_entry_price * (1.0 - pullback_pct / 100.0)
    window = candles[(candles["timestamp"] >= original_entry_time) & (candles["timestamp"] <= original_entry_time + pd.Timedelta(minutes=10))]
    hit = window[pd.to_numeric(window["low"], errors="coerce") <= target]
    if hit.empty:
        return {"would_enter": 0, "entry_time": "", "entry_price": None, "mfe_pct": None, "mae_pct": None, "min_after_5m_pct": None}
    sim_entry_time = hit.iloc[0]["timestamp"]
    path = candles_after(candles, sim_entry_time, exit_time)
    stats = calculate_path_stats(path, target, sim_entry_time)
    return {
        "would_enter": 1,
        "entry_time": iso_ts(sim_entry_time),
        "entry_price": target,
        "mfe_pct": stats.mfe_pct,
        "mae_pct": stats.mae_pct,
        "min_after_5m_pct": min_after_pct(path, target, sim_entry_time, 5),
    }


def analyze_entry_timing(
    *,
    start_date: str,
    end_date: str,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    session_type: str = "RTH",
) -> pd.DataFrame:
    trades = load_closed_trades(sqlite_path, start_date, end_date)
    executions = load_executions(sqlite_path, start_date, end_date)
    trades = augment_trades_from_executions(trades, executions, start_date, end_date)
    rows: list[dict[str, Any]] = []
    for trade in trades.to_dict("records"):
        raw = parse_raw_json(trade.get("raw_json"))
        symbol = str(trade.get("symbol") or "").upper()
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
        buy_exec = match_buy_execution(trade, executions)
        preliminary = load_recorder_candles(recorder_dir, session_date, symbol)
        if preliminary.empty:
            preliminary = load_session_candles(history_dir, symbol, session_date, session_type)
        entry_time, _, entry_warning, _ = resolve_entry_time(trade, raw, buy_exec, preliminary, session_date)
        candles_full, candle_source, candle_warning, _, _ = choose_trade_candles(
            history_dir=history_dir,
            recorder_dir=recorder_dir,
            symbol=symbol,
            session_date=session_date,
            entry_time=entry_time,
            exit_time=exit_time,
            session_type=session_type,
        )
        trade_candles = candles_after(candles_full, entry_time, exit_time) if entry_warning != "time_mismatch" else pd.DataFrame()
        stats = calculate_path_stats(trade_candles, entry_price or 0.0, entry_time)
        first_green = first_green_seconds(trade_candles, entry_price or 0.0, entry_time)
        min_1 = min_after_pct(trade_candles, entry_price or 0.0, entry_time, 1)
        min_2 = min_after_pct(trade_candles, entry_price or 0.0, entry_time, 2)
        min_5 = min_after_pct(trade_candles, entry_price or 0.0, entry_time, 5)
        min_10 = min_after_pct(trade_candles, entry_price or 0.0, entry_time, 10)
        highs = {minutes: pre_entry_high(candles_full, entry_time, minutes) for minutes in [1, 3, 5, 10]}
        prev = previous_candle(candles_full, entry_time)
        green_count, red_count = candle_color_counts(candles_full, entry_time, 3)
        momentum_3 = local_momentum(candles_full, entry_price, entry_time, 3)
        momentum_5 = local_momentum(candles_full, entry_price, entry_time, 5)
        prev_close_pct = pct(entry_price, fnum(prev.get("close")))
        prev_high_pct = pct(entry_price, fnum(prev.get("high")))
        pullback_5 = pullback_from_recent_high(entry_price, highs[5])
        chasing = entry_chasing_score(
            pullback_5m_pct=pullback_5,
            momentum_3m_pct=momentum_3,
            entry_vs_prev_high_pct=prev_high_pct,
            green_count_3m=green_count,
            min_after_5m_pct=min_5,
        )
        net_pnl_pct = net_pnl / (entry_price * quantity) * 100.0 if entry_price and quantity and net_pnl is not None else None
        row = {
            "date": session_date,
            "trade_id": trade.get("trade_id"),
            "analysis_source": trade.get("analysis_source") or "sqlite_trades",
            "logical_trade_group": trade.get("logical_trade_group"),
            "symbol": symbol,
            "entry_time": iso_ts(entry_time),
            "entry_price": entry_price,
            "exit_time": iso_ts(exit_time),
            "exit_price": exit_price,
            "quantity": quantity,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl_pct,
            "mfe_pct": stats.mfe_pct,
            "mae_pct": stats.mae_pct,
            "immediate_drop": int(min_1 is not None and min_1 < 0.0),
            "never_green": int(first_green is None and not trade_candles.empty),
            "min_after_1m_pct": min_1,
            "min_after_2m_pct": min_2,
            "min_after_5m_pct": min_5,
            "min_after_10m_pct": min_10,
            "max_after_5m_pct": max_after_pct(trade_candles, entry_price or 0.0, entry_time, 5),
            "first_green_seconds": first_green,
            "top100_rank": first_existing_column(trade, ["top100_rank"]) or raw.get("top100_rank"),
            "live_entry_score": first_existing_column(trade, ["live_entry_score", "entry_score", "score"]) or raw.get("live_entry_score") or raw.get("score"),
            "candle_source": candle_source,
            "candle_coverage_warning": candle_warning,
            "pre_entry_window_high_1m": highs[1],
            "pre_entry_window_high_3m": highs[3],
            "pre_entry_window_high_5m": highs[5],
            "pre_entry_window_high_10m": highs[10],
            "pre_entry_pullback_from_1m_high_pct": pullback_from_recent_high(entry_price, highs[1]),
            "pre_entry_pullback_from_3m_high_pct": pullback_from_recent_high(entry_price, highs[3]),
            "pre_entry_pullback_from_5m_high_pct": pullback_5,
            "pre_entry_pullback_from_10m_high_pct": pullback_from_recent_high(entry_price, highs[10]),
            "entry_vs_previous_candle_close_pct": prev_close_pct,
            "entry_vs_previous_candle_high_pct": prev_high_pct,
            "entry_after_green_candle_count_3m": green_count,
            "entry_after_red_candle_count_3m": red_count,
            "entry_local_momentum_3m_pct": momentum_3,
            "entry_local_momentum_5m_pct": momentum_5,
            "entry_chasing_score": chasing,
        }
        for label, pullback in [("0_5", 0.5), ("1_0", 1.0), ("1_5", 1.5), ("2_0", 2.0)]:
            sim = simulate_pullback_entry(candles_full, original_entry_time=entry_time, original_entry_price=entry_price, exit_time=exit_time, pullback_pct=pullback)
            row[f"pb_{label}_would_enter"] = sim["would_enter"]
            row[f"pb_{label}_entry_time"] = sim["entry_time"]
            row[f"pb_{label}_entry_price"] = sim["entry_price"]
            row[f"pb_{label}_mfe_pct"] = sim["mfe_pct"]
            row[f"pb_{label}_mae_pct"] = sim["mae_pct"]
            row[f"pb_{label}_min_after_5m_pct"] = sim["min_after_5m_pct"]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def chasing_bucket(value: Any) -> str:
    score = fnum(value)
    if score is None:
        return "missing"
    if score < 20:
        return "0-20"
    if score < 40:
        return "20-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"ENTRY_TIMING total_logical_trades={total}")
    if df.empty:
        return
    for col in ["pre_entry_pullback_from_5m_high_pct", "entry_local_momentum_3m_pct", "entry_chasing_score"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        print(f"{col} avg={vals.mean():.2f} median={vals.median():.2f}")
    print(f"immediate_drop={int(pd.to_numeric(df['immediate_drop'], errors='coerce').fillna(0).sum())} never_green={int(pd.to_numeric(df['never_green'], errors='coerce').fillna(0).sum())}")
    tmp = df.copy()
    tmp["_chasing_bucket"] = tmp["entry_chasing_score"].map(chasing_bucket)
    grouped = tmp.groupby("_chasing_bucket", dropna=False).agg(
        trade_count=("symbol", "count"),
        avg_min_after_5m_pct=("min_after_5m_pct", "mean"),
        avg_mfe_pct=("mfe_pct", "mean"),
        avg_mae_pct=("mae_pct", "mean"),
        win_rate=("net_pnl", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean() * 100.0) if len(values) else 0.0),
    )
    print("entry_chasing_score_bucket_summary:")
    print(grouped.reset_index().to_string(index=False))
    actual_mfe = pd.to_numeric(df["mfe_pct"], errors="coerce").mean()
    actual_mae = pd.to_numeric(df["mae_pct"], errors="coerce").mean()
    actual_min5 = pd.to_numeric(df["min_after_5m_pct"], errors="coerce").mean()
    for label in ["0_5", "1_0", "1_5", "2_0"]:
        would = pd.to_numeric(df[f"pb_{label}_would_enter"], errors="coerce").fillna(0)
        print(
            f"pullback_{label}_summary "
            f"would_enter_count={int(would.sum())} "
            f"avg_sim_mfe={pd.to_numeric(df[f'pb_{label}_mfe_pct'], errors='coerce').mean():.2f} "
            f"avg_sim_mae={pd.to_numeric(df[f'pb_{label}_mae_pct'], errors='coerce').mean():.2f} "
            f"avg_sim_min_after_5m={pd.to_numeric(df[f'pb_{label}_min_after_5m_pct'], errors='coerce').mean():.2f} "
            f"actual_avg_mfe={actual_mfe:.2f} actual_avg_mae={actual_mae:.2f} actual_avg_min_after_5m={actual_min5:.2f}"
        )
    print("worst_chasing_entries:")
    print(df.sort_values(["entry_chasing_score", "min_after_5m_pct"], ascending=[False, True])[["symbol", "entry_time", "entry_chasing_score", "pre_entry_pullback_from_5m_high_pct", "entry_local_momentum_3m_pct", "min_after_5m_pct", "mfe_pct", "net_pnl_pct"]].head(20).to_string(index=False))
    improved = df[
        (pd.to_numeric(df["pb_1_0_would_enter"], errors="coerce") == 1)
        & (pd.to_numeric(df["pb_1_0_mae_pct"], errors="coerce") > pd.to_numeric(df["mae_pct"], errors="coerce"))
        & (pd.to_numeric(df["pb_1_0_mfe_pct"], errors="coerce") >= pd.to_numeric(df["mfe_pct"], errors="coerce") - 0.5)
    ]
    print("pullback_1pct_improves_mae_without_killing_mfe:")
    print(improved.sort_values("mae_pct")[["symbol", "entry_time", "mae_pct", "pb_1_0_mae_pct", "mfe_pct", "pb_1_0_mfe_pct", "min_after_5m_pct", "pb_1_0_min_after_5m_pct"]].head(20).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether live entries chase spikes and simulate delayed pullback entries.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single session date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for date range. Required with --start-date.")
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
    output = args.output or Path(f"data/analysis/entry_timing_{start if start == end else start + '_to_' + end}.csv")
    df = analyze_entry_timing(
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
