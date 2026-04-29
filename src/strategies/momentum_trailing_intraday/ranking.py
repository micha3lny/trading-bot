"""Initial ranking for Momentum Trailing Intraday strategy.

The ranking is strategy-specific. It does not decide to buy.
It only selects symbols worth observing for intraday entry logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_market_data_bundle


@dataclass(frozen=True)
class RankingRow:
    symbol: str
    score: float
    intraday_momentum_pct: float
    daily_trend_pct: float
    volume_ratio: float
    last_intraday_date: str


def get_last_intraday_session(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """Return only the latest available intraday trading session."""
    if df_intraday.empty or "date" not in df_intraday.columns:
        return df_intraday.iloc[0:0]

    df = df_intraday.copy()
    df["session_date"] = df["date"].dt.date
    latest_session = df["session_date"].max()

    return df[df["session_date"] == latest_session].reset_index(drop=True)


def compute_intraday_momentum(df_intraday: pd.DataFrame) -> float:
    """Momentum from first bar to last bar of the latest session."""
    latest_session = get_last_intraday_session(df_intraday)
    if latest_session.empty:
        return 0.0

    first_price = latest_session.iloc[0]["close"]
    last_price = latest_session.iloc[-1]["close"]

    if first_price == 0:
        return 0.0

    return (last_price - first_price) / first_price * 100.0


def compute_daily_trend(df_daily: pd.DataFrame) -> float:
    """Trend filter: last close vs 20-day moving average."""
    if len(df_daily) < 20:
        return 0.0

    df = df_daily.copy()
    df["ma20"] = df["close"].rolling(20).mean()

    last_row = df.iloc[-1]
    if last_row["ma20"] == 0:
        return 0.0

    return (last_row["close"] - last_row["ma20"]) / last_row["ma20"] * 100.0


def compute_volume_ratio(df_intraday: pd.DataFrame) -> float:
    """Latest session volume vs average session volume in loaded intraday data."""
    if df_intraday.empty or "volume" not in df_intraday.columns or "date" not in df_intraday.columns:
        return 1.0

    df = df_intraday.copy()
    df["session_date"] = df["date"].dt.date
    session_volumes = df.groupby("session_date")["volume"].sum()

    if len(session_volumes) < 2:
        return 1.0

    latest_volume = session_volumes.iloc[-1]
    average_volume = session_volumes.iloc[:-1].mean()

    if average_volume == 0:
        return 1.0

    return float(latest_volume / average_volume)


def get_last_intraday_date(df_intraday: pd.DataFrame) -> str:
    latest_session = get_last_intraday_session(df_intraday)
    if latest_session.empty:
        return "n/a"

    return str(latest_session.iloc[-1]["date"].date())


def compute_score(
    intraday_momentum_pct: float,
    daily_trend_pct: float,
    volume_ratio: float,
) -> float:
    """Compute strategy-specific ranking score.

    Negative intraday momentum is penalized heavily because this strategy
    is looking for symbols with current-session continuation potential.
    """
    if intraday_momentum_pct <= 0:
        return intraday_momentum_pct * 10.0

    trend_component = max(daily_trend_pct, 0.0) * 0.10
    volume_component = min(max(volume_ratio - 1.0, 0.0), 2.0) * 0.50

    return intraday_momentum_pct + trend_component + volume_component


def rank_symbols() -> list[RankingRow]:
    rows: list[RankingRow] = []

    for spec in UNIVERSE:
        try:
            bundle = load_market_data_bundle(spec.symbol)

            intraday_momentum = compute_intraday_momentum(bundle.intraday)
            daily_trend = compute_daily_trend(bundle.daily)
            volume_ratio = compute_volume_ratio(bundle.intraday)
            score = compute_score(intraday_momentum, daily_trend, volume_ratio)

            rows.append(
                RankingRow(
                    symbol=spec.symbol,
                    score=score,
                    intraday_momentum_pct=intraday_momentum,
                    daily_trend_pct=daily_trend,
                    volume_ratio=volume_ratio,
                    last_intraday_date=get_last_intraday_date(bundle.intraday),
                )
            )

        except Exception as exc:  # noqa: BLE001 - ranking should continue for other symbols
            print(f"Skipping {spec.symbol}: {exc}")

    rows.sort(key=lambda row: row.score, reverse=True)
    return rows


def main() -> None:
    ranking = rank_symbols()

    print("\nTop 10 (Momentum Trailing Intraday)\n")
    print("Rank | Symbol | Score | Intraday % | Daily trend % | Vol ratio | Date")
    print("---------------------------------------------------------------------")

    for i, row in enumerate(ranking[:10], start=1):
        print(
            f"{i:>4} | "
            f"{row.symbol:<6} | "
            f"{row.score:>6.2f} | "
            f"{row.intraday_momentum_pct:>10.2f} | "
            f"{row.daily_trend_pct:>13.2f} | "
            f"{row.volume_ratio:>9.2f} | "
            f"{row.last_intraday_date}"
        )


if __name__ == "__main__":
    main()
