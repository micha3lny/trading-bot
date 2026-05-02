"""Analyze large momentum days to learn entry/exit patterns.

Goal:
- Study days with large intraday moves, e.g. +5% or +10% high-vs-open.
- Understand why the reversal-pullback strategy misses them.
- Test simple diagnostic entry ideas: opening-range breakout and pullback/reclaim.

This is intentionally analysis-only, not a live strategy.

Run examples:
python -m src.analysis.momentum_day_pattern_analysis --min-intraday-high 10 --top 100
python -m src.analysis.momentum_day_pattern_analysis --min-intraday-high 5 --top 200
python -m src.analysis.momentum_day_pattern_analysis --min-close-return 10 --top 100
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    load_all_data,
)


@dataclass(frozen=True)
class SimResult:
    entry_found: bool
    entry_time: str | None
    entry_price: float | None
    max_pnl_pct: float | None
    min_pnl_pct: float | None
    close_pnl_pct: float | None
    trailing_10_07_pnl_pct: float | None
    trailing_15_10_pnl_pct: float | None


def pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def close_strength(row) -> float:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if high <= low:
        return 0.0
    return (close - low) / (high - low)


def daily_opportunities(daily_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, df in daily_data.items():
        required = {"date", "open", "high", "low", "close"}
        if not required.issubset(df.columns):
            continue

        d = df.sort_values("date").copy()
        d["session_date"] = d["date"].dt.date.astype(str)
        d["prev_close"] = d["close"].shift(1)
        d["gap_pct"] = (d["open"] - d["prev_close"]) / d["prev_close"] * 100.0
        d["daily_return_pct"] = (d["close"] - d["open"]) / d["open"] * 100.0
        d["intraday_high_pct"] = (d["high"] - d["open"]) / d["open"] * 100.0
        d["intraday_low_pct"] = (d["low"] - d["open"]) / d["open"] * 100.0

        for _, row in d.dropna(subset=["prev_close"]).iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": str(row["session_date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "prev_close": float(row["prev_close"]),
                    "gap_pct": float(row["gap_pct"]),
                    "daily_return_pct": float(row["daily_return_pct"]),
                    "intraday_high_pct": float(row["intraday_high_pct"]),
                    "intraday_low_pct": float(row["intraday_low_pct"]),
                }
            )
    return pd.DataFrame(rows)


def session_1m(data_1m: dict[str, pd.DataFrame], symbol: str, session_date: str) -> pd.DataFrame:
    df = data_1m.get(symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "session_date" not in work.columns:
        work["session_date"] = work["date"].dt.date
    work = work[work["session_date"].astype(str) == str(session_date)].sort_values("date")
    return work.reset_index(drop=True)


def time_to_high_minutes(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    start = df.iloc[0]["date"]
    idx = df["high"].idxmax()
    high_time = df.loc[idx, "date"]
    return (high_time - start).total_seconds() / 60.0


def first_window_stats(df: pd.DataFrame, bars: int, open_price: float) -> dict[str, float]:
    if df.empty:
        return {f"first_{bars}m_high_pct": 0.0, f"first_{bars}m_close_pct": 0.0, f"first_{bars}m_low_pct": 0.0}
    window = df.head(min(bars, len(df)))
    return {
        f"first_{bars}m_high_pct": pct(float(window["high"].max()), open_price),
        f"first_{bars}m_close_pct": pct(float(window.iloc[-1]["close"]), open_price),
        f"first_{bars}m_low_pct": pct(float(window["low"].min()), open_price),
    }


def simulate_from_index(df: pd.DataFrame, entry_idx: int, entry_price: float) -> SimResult:
    if df.empty or entry_idx >= len(df):
        return SimResult(False, None, None, None, None, None, None, None)

    trade = df.iloc[entry_idx:].reset_index(drop=True)
    max_pnl = pct(float(trade["high"].max()), entry_price)
    min_pnl = pct(float(trade["low"].min()), entry_price)
    close_pnl = pct(float(trade.iloc[-1]["close"]), entry_price)

    def trailing_exit(activation_pct: float, trail_pct: float) -> float:
        peak = entry_price
        active = False
        for _, row in trade.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if high > peak:
                peak = high
            if pct(peak, entry_price) >= activation_pct:
                active = True
            if active:
                stop_price = peak * (1.0 - trail_pct / 100.0)
                if low <= stop_price:
                    return pct(stop_price, entry_price)
        return close_pnl

    return SimResult(
        entry_found=True,
        entry_time=str(trade.iloc[0]["date"]),
        entry_price=entry_price,
        max_pnl_pct=max_pnl,
        min_pnl_pct=min_pnl,
        close_pnl_pct=close_pnl,
        trailing_10_07_pnl_pct=trailing_exit(1.0, 0.7),
        trailing_15_10_pnl_pct=trailing_exit(1.5, 1.0),
    )


def opening_range_breakout(df: pd.DataFrame, bars: int) -> SimResult:
    if len(df) <= bars:
        return SimResult(False, None, None, None, None, None, None, None)
    opening = df.iloc[:bars]
    or_high = float(opening["high"].max())
    after = df.iloc[bars:]
    hits = after[after["high"] >= or_high]
    if hits.empty:
        return SimResult(False, None, None, None, None, None, None, None)
    entry_idx = int(hits.index[0])
    return simulate_from_index(df, entry_idx, or_high)


def pullback_reclaim_entry(
    df: pd.DataFrame,
    min_initial_move_pct: float,
    min_pullback_pct: float,
    reclaim_buffer_pct: float = 0.10,
) -> SimResult:
    if len(df) < 5:
        return SimResult(False, None, None, None, None, None, None, None)

    open_price = float(df.iloc[0]["open"])
    running_high = open_price
    pullback_low = None
    armed = False

    for idx in range(1, len(df)):
        row = df.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])

        if high > running_high:
            running_high = high

        if pct(running_high, open_price) >= min_initial_move_pct:
            armed = True

        if not armed:
            continue

        pullback_now = (running_high - low) / running_high * 100.0 if running_high else 0.0
        if pullback_now >= min_pullback_pct:
            pullback_low = low if pullback_low is None else min(pullback_low, low)

        if pullback_low is None:
            continue

        prev_high = float(df.iloc[idx - 1]["high"])
        reclaim_price = prev_high * (1.0 + reclaim_buffer_pct / 100.0)
        if high >= reclaim_price:
            return simulate_from_index(df, idx, reclaim_price)

    return SimResult(False, None, None, None, None, None, None, None)


def first_pullback_depth_after_move(df: pd.DataFrame, min_initial_move_pct: float) -> float | None:
    if len(df) < 5:
        return None
    open_price = float(df.iloc[0]["open"])
    running_high = open_price
    after_armed_lows: list[float] = []
    armed = False
    for _, row in df.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if high > running_high:
            running_high = high
        if pct(running_high, open_price) >= min_initial_move_pct:
            armed = True
        if armed:
            after_armed_lows.append(low)
            if len(after_armed_lows) >= 20:
                break
    if not after_armed_lows or running_high == 0:
        return None
    return (running_high - min(after_armed_lows)) / running_high * 100.0


def add_sim(prefix: str, result: SimResult, row: dict) -> None:
    row[f"{prefix}_found"] = result.entry_found
    row[f"{prefix}_entry_time"] = result.entry_time
    row[f"{prefix}_entry_price"] = result.entry_price
    row[f"{prefix}_max_pnl_pct"] = result.max_pnl_pct
    row[f"{prefix}_min_pnl_pct"] = result.min_pnl_pct
    row[f"{prefix}_close_pnl_pct"] = result.close_pnl_pct
    row[f"{prefix}_trail_10_07_pnl_pct"] = result.trailing_10_07_pnl_pct
    row[f"{prefix}_trail_15_10_pnl_pct"] = result.trailing_15_10_pnl_pct


def mean_found(df: pd.DataFrame, col: str, found_col: str) -> float | None:
    subset = df[df[found_col] == True]  # noqa: E712
    if subset.empty:
        return None
    return float(subset[col].mean())


def print_entry_summary(df: pd.DataFrame, name: str) -> None:
    found_col = f"{name}_found"
    if found_col not in df:
        return
    found = int((df[found_col] == True).sum())  # noqa: E712
    print(f"\n--- {name} ---")
    print(f"entries found: {found}/{len(df)}")
    if found == 0:
        return
    for col in [
        f"{name}_max_pnl_pct",
        f"{name}_min_pnl_pct",
        f"{name}_close_pnl_pct",
        f"{name}_trail_10_07_pnl_pct",
        f"{name}_trail_15_10_pnl_pct",
    ]:
        value = mean_found(df, col, found_col)
        print(f"avg {col.replace(name + '_', '')}: {value:.2f}%")


def quantile_text(values: pd.Series) -> str:
    clean = values.dropna()
    if clean.empty:
        return "n/a"
    return (
        f"p25={clean.quantile(0.25):.2f}, "
        f"median={clean.quantile(0.50):.2f}, "
        f"p75={clean.quantile(0.75):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-intraday-high", type=float, default=10.0)
    parser.add_argument("--min-close-return", type=float, default=None)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--output-dir", default="data/backtests")
    args = parser.parse_args()

    print("\nMomentum day pattern analysis")
    if args.min_close_return is not None:
        print(f"Opportunity filter: close return >= {args.min_close_return:.2f}%")
    else:
        print(f"Opportunity filter: intraday high >= {args.min_intraday_high:.2f}%")
    print(f"Top opportunities: {args.top}")

    _data_15m, _data_5m, data_1m, daily_data = load_all_data()
    opportunities = daily_opportunities(daily_data)
    if opportunities.empty:
        print("No daily opportunities found.")
        return

    if args.min_close_return is not None:
        opportunities = opportunities[opportunities["daily_return_pct"] >= args.min_close_return]
        opportunities = opportunities.sort_values("daily_return_pct", ascending=False).head(args.top)
        suffix = f"close_ge_{args.min_close_return:g}"
    else:
        opportunities = opportunities[opportunities["intraday_high_pct"] >= args.min_intraday_high]
        opportunities = opportunities.sort_values("intraday_high_pct", ascending=False).head(args.top)
        suffix = f"intraday_ge_{args.min_intraday_high:g}"

    rows: list[dict] = []
    skipped_no_1m = 0

    for _, opp in opportunities.iterrows():
        symbol = str(opp["symbol"])
        session_date = str(opp["session_date"])
        day_1m = session_1m(data_1m, symbol, session_date)
        if day_1m.empty:
            skipped_no_1m += 1
            continue

        open_price = float(day_1m.iloc[0]["open"])
        high_idx = int(day_1m["high"].idxmax())
        high_time = str(day_1m.loc[high_idx, "date"])
        high_price = float(day_1m.loc[high_idx, "high"])
        low_before_high = float(day_1m.iloc[: high_idx + 1]["low"].min())
        close_price = float(day_1m.iloc[-1]["close"])

        row = {
            **opp.to_dict(),
            "bars_1m": len(day_1m),
            "real_open_from_1m": open_price,
            "time_to_high_minutes": time_to_high_minutes(day_1m),
            "high_time": high_time,
            "high_from_1m_pct": pct(high_price, open_price),
            "low_before_high_pct": pct(low_before_high, open_price),
            "close_from_1m_pct": pct(close_price, open_price),
            "pullback_after_3pct_move_pct": first_pullback_depth_after_move(day_1m, 3.0),
            "pullback_after_5pct_move_pct": first_pullback_depth_after_move(day_1m, 5.0),
        }
        for bars in [5, 15, 30, 60]:
            row.update(first_window_stats(day_1m, bars, open_price))

        simulations = {
            "or5_breakout": opening_range_breakout(day_1m, 5),
            "or15_breakout": opening_range_breakout(day_1m, 15),
            "or30_breakout": opening_range_breakout(day_1m, 30),
            "pullback_3up_1pb": pullback_reclaim_entry(day_1m, 3.0, 1.0),
            "pullback_5up_1pb": pullback_reclaim_entry(day_1m, 5.0, 1.0),
            "pullback_5up_2pb": pullback_reclaim_entry(day_1m, 5.0, 2.0),
        }
        for name, sim in simulations.items():
            add_sim(name, sim, row)

        rows.append(row)

    out = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"momentum_day_pattern_analysis_{suffix}_top{args.top}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved analysis CSV: {out_path}")
    print(f"Analyzed days with 1m data: {len(out)}")
    print(f"Skipped no 1m data: {skipped_no_1m}")

    if out.empty:
        print("No rows to summarize.")
        return

    print("\n=== Big momentum day shape ===")
    print(f"avg gap: {out['gap_pct'].mean():.2f}%")
    print(f"avg intraday high: {out['intraday_high_pct'].mean():.2f}%")
    print(f"avg close return: {out['daily_return_pct'].mean():.2f}%")
    print(f"avg time to high: {out['time_to_high_minutes'].mean():.1f} min")
    print(f"first 5m high pct: {quantile_text(out['first_5m_high_pct'])}")
    print(f"first 15m high pct: {quantile_text(out['first_15m_high_pct'])}")
    print(f"first 30m high pct: {quantile_text(out['first_30m_high_pct'])}")
    print(f"pullback after +3% move: {quantile_text(out['pullback_after_3pct_move_pct'])}")
    print(f"pullback after +5% move: {quantile_text(out['pullback_after_5pct_move_pct'])}")

    print("\n=== Diagnostic entry simulations ===")
    for name in [
        "or5_breakout",
        "or15_breakout",
        "or30_breakout",
        "pullback_3up_1pb",
        "pullback_5up_1pb",
        "pullback_5up_2pb",
    ]:
        print_entry_summary(out, name)

    print("\n=== Top analyzed opportunities ===")
    display_cols = [
        "session_date",
        "symbol",
        "gap_pct",
        "intraday_high_pct",
        "daily_return_pct",
        "time_to_high_minutes",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "pullback_after_5pct_move_pct",
        "or5_breakout_found",
        "or5_breakout_max_pnl_pct",
        "pullback_5up_1pb_found",
        "pullback_5up_1pb_max_pnl_pct",
    ]
    print(
        out[display_cols]
        .head(min(len(out), 40))
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nInterpretation:")
    print("- If OR breakout has high coverage and max_pnl, these days are momentum/gap days, not selloff-pullback days.")
    print("- If pullback entries have lower coverage but better drawdown, we can design a separate momentum continuation strategy.")
    print("- If time_to_high is early, many moves are front-loaded and need opening-range logic, not v38 reversal logic.")


if __name__ == "__main__":
    main()
