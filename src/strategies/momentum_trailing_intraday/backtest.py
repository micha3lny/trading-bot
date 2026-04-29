"""Simple historical backtest for Momentum Trailing Intraday strategy.

This is the first real walk-forward test:
- each intraday session is evaluated independently
- entry can happen during the session, not only on the last bar
- exit is simulated on bars after entry
- no orders are placed
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_daily, load_intraday
from src.strategies.momentum_trailing_intraday.exit import (
    INITIAL_STOP_LOSS_PCT,
    TRAILING_ACTIVATION_PROFIT_PCT,
    TRAILING_STOP_PCT,
)


OPENING_RANGE_BARS = 4
MIN_BREAKOUT_PCT = 0.25
MIN_CLOSE_STRENGTH = 0.60
MAX_ENTRY_RISK_PCT = 2.0
MAX_POSITIONS_PER_DAY = 3
MIN_DAILY_TREND_PCT = 0.0

# Market regime filter based on the current top30 universe breadth.
ENABLE_MARKET_REGIME_FILTER = True
MIN_POSITIVE_BREADTH_PCT = 50.0
MIN_AVERAGE_OPENING_RANGE_RETURN_PCT = 0.02

# Pullback/retest is kept as an experiment, but disabled by default after weak results.
ENABLE_PULLBACK_RETEST_ENTRY = False
PULLBACK_RETEST_TOLERANCE_PCT = 0.35


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    session_date: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    breakout_pct: float
    close_strength: float
    entry_risk_pct: float
    daily_trend_pct: float
    setup_type: str


@dataclass(frozen=True)
class MarketRegime:
    session_date: str
    positive_breadth_pct: float
    average_opening_range_return_pct: float
    tradable: bool


def calculate_close_strength(row: pd.Series) -> float:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    if high == low:
        return 0.0

    return (close - low) / (high - low)


def calculate_breakout_pct(close: float, opening_range_high: float) -> float:
    if opening_range_high == 0:
        return 0.0

    return (close - opening_range_high) / opening_range_high * 100.0


def calculate_entry_risk_pct(entry_price: float, opening_range_low: float) -> float:
    if entry_price == 0:
        return 0.0

    return (entry_price - opening_range_low) / entry_price * 100.0


def prepare_daily_data(symbol: str) -> pd.DataFrame:
    daily = load_daily(symbol).copy()
    daily["session_date"] = daily["date"].dt.date
    daily["ma20"] = daily["close"].rolling(20).mean()
    daily["daily_trend_pct"] = (daily["close"] - daily["ma20"]) / daily["ma20"] * 100.0
    return daily


def get_daily_trend_before_session(daily: pd.DataFrame, session_date) -> float:
    history = daily[daily["session_date"] < session_date].dropna(subset=["daily_trend_pct"])
    if history.empty:
        return 0.0
    return float(history.iloc[-1]["daily_trend_pct"])


def calculate_session_opening_range_return(session: pd.DataFrame) -> float | None:
    if len(session) <= OPENING_RANGE_BARS:
        return None

    first_close = float(session.iloc[0]["close"])
    post_or_close = float(session.iloc[OPENING_RANGE_BARS]["close"])
    if first_close == 0:
        return None

    return (post_or_close - first_close) / first_close * 100.0


def build_market_regimes(intraday_data: dict[str, pd.DataFrame]) -> dict[str, MarketRegime]:
    values_by_day: dict[str, list[float]] = {}

    for df in intraday_data.values():
        for session_date, session in df.groupby("session_date"):
            session = session.sort_values("date").reset_index(drop=True)
            opening_range_return = calculate_session_opening_range_return(session)
            if opening_range_return is None:
                continue
            values_by_day.setdefault(str(session_date), []).append(opening_range_return)

    regimes: dict[str, MarketRegime] = {}
    for session_date, values in values_by_day.items():
        if not values:
            continue

        positive_breadth_pct = sum(1 for value in values if value > 0) / len(values) * 100.0
        average_opening_range_return_pct = sum(values) / len(values)
        tradable = (
            positive_breadth_pct >= MIN_POSITIVE_BREADTH_PCT
            or average_opening_range_return_pct >= MIN_AVERAGE_OPENING_RANGE_RETURN_PCT
        )

        regimes[session_date] = MarketRegime(
            session_date=session_date,
            positive_breadth_pct=positive_breadth_pct,
            average_opening_range_return_pct=average_opening_range_return_pct,
            tradable=tradable,
        )

    return regimes


def find_entry_bar(session: pd.DataFrame) -> tuple[int, float, float, float, str] | None:
    if len(session) <= OPENING_RANGE_BARS:
        return None

    opening_range = session.iloc[:OPENING_RANGE_BARS]
    opening_range_high = float(opening_range["high"].max())
    opening_range_low = float(opening_range["low"].min())
    retest_low_threshold = opening_range_high * (1.0 - PULLBACK_RETEST_TOLERANCE_PCT / 100.0)

    broke_out = False
    retested = False

    for position in range(OPENING_RANGE_BARS, len(session)):
        row = session.iloc[position]
        close = float(row["close"])
        low = float(row["low"])
        breakout_pct = calculate_breakout_pct(close, opening_range_high)
        close_strength = calculate_close_strength(row)
        entry_risk_pct = calculate_entry_risk_pct(close, opening_range_low)

        if close > opening_range_high:
            if not broke_out:
                broke_out = True

            if (
                breakout_pct >= MIN_BREAKOUT_PCT
                and close_strength >= MIN_CLOSE_STRENGTH
                and entry_risk_pct <= MAX_ENTRY_RISK_PCT
            ):
                return position, breakout_pct, close_strength, entry_risk_pct, "breakout"

            if ENABLE_PULLBACK_RETEST_ENTRY and retested and entry_risk_pct <= MAX_ENTRY_RISK_PCT:
                return position, breakout_pct, close_strength, entry_risk_pct, "pullback_retest"

        if broke_out and low <= opening_range_high and low >= retest_low_threshold:
            retested = True

    return None


def simulate_exit(
    symbol: str,
    session: pd.DataFrame,
    entry_position: int,
    breakout_pct: float,
    close_strength: float,
    entry_risk_pct: float,
    daily_trend_pct: float,
    setup_type: str,
) -> BacktestTrade:
    entry_bar = session.iloc[entry_position]
    entry_price = float(entry_bar["close"])
    entry_time = str(entry_bar["date"])
    session_date = str(entry_bar["date"].date())

    initial_stop_price = entry_price * (1.0 - INITIAL_STOP_LOSS_PCT / 100.0)
    activation_price = entry_price * (1.0 + TRAILING_ACTIVATION_PROFIT_PCT / 100.0)

    highest_price = entry_price
    trailing_activated = False
    bars_after_entry = session.iloc[entry_position + 1 :]

    if bars_after_entry.empty:
        return BacktestTrade(symbol, session_date, entry_time, entry_time, entry_price, entry_price, 0.0, "no bars after entry", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

    last_bar = bars_after_entry.iloc[-1]

    for _, bar in bars_after_entry.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_time = str(bar["date"])
        highest_price = max(highest_price, bar_high)

        if bar_low <= initial_stop_price:
            pnl_pct = (initial_stop_price - entry_price) / entry_price * 100.0
            return BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, initial_stop_price, pnl_pct, "initial stop-loss", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

        if highest_price >= activation_price:
            trailing_activated = True

        if trailing_activated:
            trailing_stop_price = highest_price * (1.0 - TRAILING_STOP_PCT / 100.0)
            if bar_low <= trailing_stop_price:
                pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                return BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, trailing_stop_price, pnl_pct, "trailing stop", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

    exit_price = float(last_bar["close"])
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return BacktestTrade(symbol, session_date, entry_time, str(last_bar["date"]), entry_price, exit_price, pnl_pct, "end of session", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)


def backtest_symbol(symbol: str, intraday: pd.DataFrame, daily: pd.DataFrame, market_regimes: dict[str, MarketRegime]) -> list[BacktestTrade]:
    trades: list[BacktestTrade] = []

    for session_date, session in intraday.groupby("session_date"):
        session_date_str = str(session_date)
        regime = market_regimes.get(session_date_str)
        if ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
            continue

        session = session.sort_values("date").reset_index(drop=True)
        entry = find_entry_bar(session)
        if entry is None:
            continue

        entry_position, breakout_pct, close_strength, entry_risk_pct, setup_type = entry
        daily_trend_pct = get_daily_trend_before_session(daily, session_date)
        if daily_trend_pct < MIN_DAILY_TREND_PCT:
            continue

        trades.append(simulate_exit(symbol, session, entry_position, breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type))

    return trades


def load_all_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    intraday_data: dict[str, pd.DataFrame] = {}
    daily_data: dict[str, pd.DataFrame] = {}

    for spec in UNIVERSE:
        try:
            intraday = load_intraday(spec.symbol).copy()
            intraday["session_date"] = intraday["date"].dt.date
            intraday_data[spec.symbol] = intraday
            daily_data[spec.symbol] = prepare_daily_data(spec.symbol)
        except Exception as exc:  # noqa: BLE001 - backtest should continue for other symbols
            print(f"Skipping {spec.symbol}: {exc}")

    return intraday_data, daily_data


def limit_positions_per_day(trades: list[BacktestTrade]) -> list[BacktestTrade]:
    by_day: dict[str, list[BacktestTrade]] = {}
    for trade in trades:
        by_day.setdefault(trade.session_date, []).append(trade)

    selected: list[BacktestTrade] = []
    for day_trades in by_day.values():
        selected.extend(
            sorted(
                day_trades,
                key=lambda trade: (
                    trade.daily_trend_pct,
                    trade.breakout_pct,
                    trade.close_strength,
                    -trade.entry_risk_pct,
                ),
                reverse=True,
            )[:MAX_POSITIONS_PER_DAY]
        )

    return sorted(selected, key=lambda trade: (trade.session_date, trade.entry_time))


def summarize(trades: list[BacktestTrade], market_regimes: dict[str, MarketRegime]) -> None:
    print("\nBacktest: Momentum Trailing Intraday\n")
    print(
        f"Params: opening_range_bars={OPENING_RANGE_BARS}, "
        f"min_breakout={MIN_BREAKOUT_PCT:.2f}%, "
        f"min_close_strength={MIN_CLOSE_STRENGTH:.2f}, "
        f"max_entry_risk={MAX_ENTRY_RISK_PCT:.2f}%, "
        f"min_daily_trend={MIN_DAILY_TREND_PCT:.2f}%, "
        f"max_positions_per_day={MAX_POSITIONS_PER_DAY}, "
        f"pullback_retest={ENABLE_PULLBACK_RETEST_ENTRY}, "
        f"market_regime={ENABLE_MARKET_REGIME_FILTER}, "
        "selection=daily_trend+breakout"
    )
    print(
        f"Regime: min_positive_breadth={MIN_POSITIVE_BREADTH_PCT:.1f}%, "
        f"min_avg_or_return={MIN_AVERAGE_OPENING_RANGE_RETURN_PCT:.2f}%"
    )
    print(
        f"Exit: stop={INITIAL_STOP_LOSS_PCT:.2f}%, "
        f"trail_activation={TRAILING_ACTIVATION_PROFIT_PCT:.2f}%, "
        f"trail_stop={TRAILING_STOP_PCT:.2f}%"
    )

    tradable_days = sum(1 for regime in market_regimes.values() if regime.tradable)
    print(f"Tradable regime days: {tradable_days}/{len(market_regimes)}")
    print()

    if not trades:
        print("No trades found.")
        return

    pnl_values = [trade.pnl_pct for trade in trades]
    wins = [pnl for pnl in pnl_values if pnl > 0]
    losses = [pnl for pnl in pnl_values if pnl <= 0]

    print(f"Trades: {len(trades)}")
    print(f"Win rate: {len(wins) / len(trades) * 100.0:.2f}%")
    print(f"Average PnL: {sum(pnl_values) / len(pnl_values):.2f}%")
    print(f"Total PnL sum: {sum(pnl_values):.2f}%")
    print(f"Best trade: {max(pnl_values):.2f}%")
    print(f"Worst trade: {min(pnl_values):.2f}%")
    if losses:
        print(f"Average loss: {sum(losses) / len(losses):.2f}%")
    if wins:
        print(f"Average win: {sum(wins) / len(wins):.2f}%")

    setup_counts: dict[str, int] = {}
    for trade in trades:
        setup_counts[trade.setup_type] = setup_counts.get(trade.setup_type, 0) + 1
    print("Setup counts:", setup_counts)

    print("\nRecent trades\n")
    print("Date | Symbol | Setup | Entry | Exit | PnL % | Trend % | Break % | CloseStr | Risk % | Reason")
    print("----------------------------------------------------------------------------------------------")

    for trade in trades[-20:]:
        print(
            f"{trade.session_date} | "
            f"{trade.symbol:<6} | "
            f"{trade.setup_type:<8} | "
            f"{trade.entry_price:>7.2f} | "
            f"{trade.exit_price:>7.2f} | "
            f"{trade.pnl_pct:>5.2f} | "
            f"{trade.daily_trend_pct:>7.2f} | "
            f"{trade.breakout_pct:>7.2f} | "
            f"{trade.close_strength:>8.2f} | "
            f"{trade.entry_risk_pct:>6.2f} | "
            f"{trade.exit_reason}"
        )


def main() -> None:
    intraday_data, daily_data = load_all_data()
    market_regimes = build_market_regimes(intraday_data)

    all_trades: list[BacktestTrade] = []
    for symbol, intraday in intraday_data.items():
        all_trades.extend(backtest_symbol(symbol, intraday, daily_data[symbol], market_regimes))

    selected_trades = limit_positions_per_day(all_trades)
    summarize(selected_trades, market_regimes)


if __name__ == "__main__":
    main()
