"""Raw breakout candidate analysis for Momentum Trailing Intraday.

This script scans all opening-range breakout candidates before strict entry filters.
It is meant to answer whether the raw setup has any statistical edge at all.

No orders are placed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt


OUTPUT_DIR = Path("data/backtests")
OUTPUT_FILE = OUTPUT_DIR / "momentum_raw_candidates.csv"


def calculate_forward_return(session: pd.DataFrame, position: int) -> tuple[float, str]:
    entry_price = float(session.iloc[position]["close"])
    bars_after_entry = session.iloc[position + 1 :]

    if bars_after_entry.empty or entry_price == 0:
        return 0.0, "no bars after entry"

    exit_price = float(bars_after_entry.iloc[-1]["close"])
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return pnl_pct, "end of session"


def scan_candidates() -> pd.DataFrame:
    intraday_data, daily_data = bt.load_all_data()
    market_regimes = bt.build_market_regimes(intraday_data)
    rows: list[dict] = []

    for symbol, intraday in intraday_data.items():
        daily = daily_data[symbol]

        for session_date, session in intraday.groupby("session_date"):
            regime = market_regimes.get(str(session_date))
            if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
                continue

            session = session.sort_values("date").reset_index(drop=True)
            if len(session) <= bt.OPENING_RANGE_BARS:
                continue

            opening_range = session.iloc[: bt.OPENING_RANGE_BARS]
            opening_range_high = float(opening_range["high"].max())
            opening_range_low = float(opening_range["low"].min())
            daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)

            for position in range(bt.OPENING_RANGE_BARS, len(session)):
                row = session.iloc[position]
                close = float(row["close"])
                if close <= opening_range_high:
                    continue

                breakout_pct = bt.calculate_breakout_pct(close, opening_range_high)
                close_strength = bt.calculate_close_strength(row)
                entry_risk_pct = bt.calculate_entry_risk_pct(close, opening_range_low)
                pnl_to_close_pct, exit_reason = calculate_forward_return(session, position)

                next_bar_return_pct = None
                next_bar_close_strength = None
                if position + 1 < len(session):
                    next_bar = session.iloc[position + 1]
                    next_close = float(next_bar["close"])
                    next_bar_return_pct = (next_close - close) / close * 100.0 if close else 0.0
                    next_bar_close_strength = bt.calculate_close_strength(next_bar)

                rows.append(
                    {
                        "symbol": symbol,
                        "session_date": str(session_date),
                        "candidate_time": str(row["date"]),
                        "position": position,
                        "entry_price": close,
                        "opening_range_high": opening_range_high,
                        "opening_range_low": opening_range_low,
                        "breakout_pct": breakout_pct,
                        "close_strength": close_strength,
                        "entry_risk_pct": entry_risk_pct,
                        "daily_trend_pct": daily_trend_pct,
                        "next_bar_return_pct": next_bar_return_pct,
                        "next_bar_close_strength": next_bar_close_strength,
                        "pnl_to_close_pct": pnl_to_close_pct,
                        "exit_reason": exit_reason,
                    }
                )

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    return df


def print_segment_analysis(df: pd.DataFrame) -> None:
    print("\nRaw breakout candidates analysis")
    print(f"Candidates: {len(df)}")
    print(f"Saved: {OUTPUT_FILE}")

    if df.empty:
        return

    wins = df[df["pnl_to_close_pct"] > 0]
    print(f"Win rate to close: {len(wins) / len(df) * 100.0:.2f}%")
    print(f"Average PnL to close: {df['pnl_to_close_pct'].mean():.3f}%")
    print(f"Median PnL to close: {df['pnl_to_close_pct'].median():.3f}%")

    segments = {
        "breakout_pct": [0.25, 0.5, 0.75, 1.0, 1.5],
        "daily_trend_pct": [-5, 0, 2, 5, 10, 20],
        "entry_risk_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
        "close_strength": [0.4, 0.6, 0.8, 0.9, 1.0],
        "next_bar_return_pct": [-1.0, -0.25, 0.0, 0.1, 0.25, 0.5, 1.0],
    }

    for col, bins in segments.items():
        if col not in df or df[col].dropna().empty:
            continue
        bucket = pd.cut(df[col], bins=[-999] + bins + [999])
        grouped = df.groupby(bucket, observed=True)["pnl_to_close_pct"].agg(["count", "mean", "median"])
        print(f"\n{col}:")
        print(grouped)

    print("\nTop 20 candidates by pnl_to_close_pct")
    print(
        df.sort_values("pnl_to_close_pct", ascending=False)
        .head(20)[[
            "session_date",
            "symbol",
            "candidate_time",
            "pnl_to_close_pct",
            "breakout_pct",
            "daily_trend_pct",
            "entry_risk_pct",
            "close_strength",
            "next_bar_return_pct",
        ]]
        .to_string(index=False)
    )

    print("\nWorst 20 candidates by pnl_to_close_pct")
    print(
        df.sort_values("pnl_to_close_pct", ascending=True)
        .head(20)[[
            "session_date",
            "symbol",
            "candidate_time",
            "pnl_to_close_pct",
            "breakout_pct",
            "daily_trend_pct",
            "entry_risk_pct",
            "close_strength",
            "next_bar_return_pct",
        ]]
        .to_string(index=False)
    )


def main() -> None:
    df = scan_candidates()
    print_segment_analysis(df)


if __name__ == "__main__":
    main()
