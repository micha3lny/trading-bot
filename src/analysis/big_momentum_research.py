"""Research big momentum days and missed opportunities.

Purpose:
- Identify historical daily opportunities with large intraday high-vs-open moves.
- Separate opportunities that have 1m intraday data from opportunities missing 1m data.
- Analyze available 1m cases with simple momentum-entry diagnostics.
- Produce a prioritized list of missing 1m symbol/date pairs to backfill.

This is analysis-only. It does not place trades.

Examples:
python -m src.analysis.big_momentum_research --min-intraday-high 10 --top 200
python -m src.analysis.big_momentum_research --min-intraday-high 5 --top 500
python -m src.analysis.big_momentum_research --min-close-return 10 --top 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.momentum_day_pattern_analysis import (
    add_sim,
    daily_opportunities,
    first_pullback_depth_after_move,
    first_window_stats,
    opening_range_breakout,
    pct,
    pullback_reclaim_entry,
    quantile_text,
    session_1m,
    simulate_from_index,
    time_to_high_minutes,
)
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    load_all_data,
)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df:
        return None
    clean = df[col].dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def safe_rate(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df:
        return None
    return float((df[col] == True).mean() * 100.0)  # noqa: E712


def simulate_trailing_variants(df: pd.DataFrame, entry_idx: int, entry_price: float) -> dict[str, float | None]:
    if df.empty or entry_idx >= len(df):
        return {}

    trade = df.iloc[entry_idx:].reset_index(drop=True)
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

    return {
        "trail_08_06_pnl_pct": trailing_exit(0.8, 0.6),
        "trail_10_07_pnl_pct": trailing_exit(1.0, 0.7),
        "trail_15_10_pnl_pct": trailing_exit(1.5, 1.0),
        "trail_25_15_pnl_pct": trailing_exit(2.5, 1.5),
        "close_pnl_pct": close_pnl,
    }


def first_breakout_entry_index(df: pd.DataFrame, bars: int) -> tuple[int | None, float | None]:
    if len(df) <= bars:
        return None, None
    opening = df.iloc[:bars]
    or_high = float(opening["high"].max())
    after = df.iloc[bars:]
    hits = after[after["high"] >= or_high]
    if hits.empty:
        return None, None
    return int(hits.index[0]), or_high


def analyze_available_day(opp: pd.Series, day_1m: pd.DataFrame) -> dict[str, Any]:
    symbol = str(opp["symbol"])
    session_date = str(opp["session_date"])
    open_price = float(day_1m.iloc[0]["open"])
    high_idx = int(day_1m["high"].idxmax())
    high_price = float(day_1m.loc[high_idx, "high"])
    high_time = day_1m.loc[high_idx, "date"]
    close_price = float(day_1m.iloc[-1]["close"])
    low_before_high = float(day_1m.iloc[: high_idx + 1]["low"].min())

    row: dict[str, Any] = {
        **opp.to_dict(),
        "symbol": symbol,
        "session_date": session_date,
        "has_1m_data": True,
        "bars_1m": len(day_1m),
        "real_open_from_1m": open_price,
        "high_from_1m_pct": pct(high_price, open_price),
        "high_time": str(high_time),
        "time_to_high_minutes": time_to_high_minutes(day_1m),
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

    # Extra exit diagnostics from the OR5 entry, because big momentum days often start as opening-range continuation.
    entry_idx, entry_price = first_breakout_entry_index(day_1m, 5)
    if entry_idx is not None and entry_price is not None:
        for key, value in simulate_trailing_variants(day_1m, entry_idx, entry_price).items():
            row[f"or5_{key}"] = value
    else:
        for key in [
            "trail_08_06_pnl_pct",
            "trail_10_07_pnl_pct",
            "trail_15_10_pnl_pct",
            "trail_25_15_pnl_pct",
            "close_pnl_pct",
        ]:
            row[f"or5_{key}"] = None

    return row


def print_sim_summary(df: pd.DataFrame, name: str) -> None:
    found_col = f"{name}_found"
    if found_col not in df:
        return
    found = int((df[found_col] == True).sum())  # noqa: E712
    total = len(df)
    print(f"\n--- {name} ---")
    print(f"coverage: {found}/{total} ({(found / total * 100.0) if total else 0.0:.1f}%)")
    if found == 0:
        return
    subset = df[df[found_col] == True]  # noqa: E712
    for col in [
        f"{name}_max_pnl_pct",
        f"{name}_min_pnl_pct",
        f"{name}_close_pnl_pct",
        f"{name}_trail_10_07_pnl_pct",
        f"{name}_trail_15_10_pnl_pct",
    ]:
        if col in subset:
            print(f"avg {col.replace(name + '_', '')}: {subset[col].mean():.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-intraday-high", type=float, default=10.0)
    parser.add_argument("--min-close-return", type=float, default=None)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--output-dir", default="data/backtests")
    args = parser.parse_args()

    print("\nBig momentum research")
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
        filtered = opportunities[opportunities["daily_return_pct"] >= args.min_close_return].copy()
        filtered = filtered.sort_values("daily_return_pct", ascending=False).head(args.top)
        suffix = f"close_ge_{args.min_close_return:g}_top{args.top}"
    else:
        filtered = opportunities[opportunities["intraday_high_pct"] >= args.min_intraday_high].copy()
        filtered = filtered.sort_values("intraday_high_pct", ascending=False).head(args.top)
        suffix = f"intraday_ge_{args.min_intraday_high:g}_top{args.top}"

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for _, opp in filtered.iterrows():
        symbol = str(opp["symbol"])
        session_date = str(opp["session_date"])
        day_1m = session_1m(data_1m, symbol, session_date)
        if day_1m.empty:
            miss = opp.to_dict()
            miss["has_1m_data"] = False
            miss["missing_reason"] = "no_1m_session_for_symbol_date"
            missing_rows.append(miss)
            continue
        rows.append(analyze_available_day(opp, day_1m))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzed = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows)

    analyzed_path = out_dir / f"big_momentum_research_analyzed_{suffix}.csv"
    missing_path = out_dir / f"big_momentum_research_missing_1m_{suffix}.csv"
    summary_path = out_dir / f"big_momentum_research_symbols_to_backfill_{suffix}.csv"

    analyzed.to_csv(analyzed_path, index=False)
    missing.to_csv(missing_path, index=False)

    print(f"\nSaved analyzed CSV: {analyzed_path}")
    print(f"Saved missing 1m CSV: {missing_path}")
    print(f"Opportunities selected: {len(filtered)}")
    print(f"Analyzed with 1m data: {len(analyzed)}")
    print(f"Missing 1m data: {len(missing)}")

    if not missing.empty:
        backfill_cols = ["symbol", "session_date", "intraday_high_pct", "daily_return_pct", "gap_pct"]
        missing[backfill_cols].to_csv(summary_path, index=False)
        print(f"Saved backfill list: {summary_path}")
        print("\n=== Top missing 1m days to backfill ===")
        print(
            missing[backfill_cols]
            .head(40)
            .to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )

        print("\n=== Symbols most often missing among selected opportunities ===")
        print(missing["symbol"].value_counts().head(30).to_string())

    if analyzed.empty:
        print("\nNo available 1m rows to analyze yet.")
        print("Next step: use the missing/backfill CSV to download 1m data for those symbol/date pairs.")
        return

    print("\n=== Available 1m momentum-day shape ===")
    print(f"avg gap: {fmt(safe_mean(analyzed, 'gap_pct'))}%")
    print(f"avg intraday high: {fmt(safe_mean(analyzed, 'intraday_high_pct'))}%")
    print(f"avg close return: {fmt(safe_mean(analyzed, 'daily_return_pct'))}%")
    print(f"avg time to high: {fmt(safe_mean(analyzed, 'time_to_high_minutes'), 1)} min")
    print(f"first 5m high pct: {quantile_text(analyzed['first_5m_high_pct'])}")
    print(f"first 15m high pct: {quantile_text(analyzed['first_15m_high_pct'])}")
    print(f"first 30m high pct: {quantile_text(analyzed['first_30m_high_pct'])}")
    print(f"pullback after +3% move: {quantile_text(analyzed['pullback_after_3pct_move_pct'])}")
    print(f"pullback after +5% move: {quantile_text(analyzed['pullback_after_5pct_move_pct'])}")

    print("\n=== Diagnostic entry simulations ===")
    for name in [
        "or5_breakout",
        "or15_breakout",
        "or30_breakout",
        "pullback_3up_1pb",
        "pullback_5up_1pb",
        "pullback_5up_2pb",
    ]:
        print_sim_summary(analyzed, name)

    print("\n=== OR5 trailing variants on available 1m rows ===")
    for col in [
        "or5_trail_08_06_pnl_pct",
        "or5_trail_10_07_pnl_pct",
        "or5_trail_15_10_pnl_pct",
        "or5_trail_25_15_pnl_pct",
        "or5_close_pnl_pct",
    ]:
        print(f"avg {col.replace('or5_', '')}: {fmt(safe_mean(analyzed, col))}%")

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
        "or5_breakout_found",
        "or5_breakout_max_pnl_pct",
        "pullback_3up_1pb_found",
        "pullback_3up_1pb_max_pnl_pct",
        "or5_trail_15_10_pnl_pct",
        "or5_close_pnl_pct",
    ]
    existing_cols = [c for c in display_cols if c in analyzed.columns]
    print(
        analyzed[existing_cols]
        .head(50)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nInterpretation:")
    print("- If most rows are missing 1m, we cannot learn entries for those huge daily moves yet.")
    print("- The missing/backfill CSV is the shopping list for historical 1m data.")
    print("- If OR breakout or early pullback has high max_pnl, this should become a separate momentum-continuation strategy, not a reversal-pullback strategy.")


if __name__ == "__main__":
    main()
