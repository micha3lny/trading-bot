"""Fast parameter optimizer for Momentum Trailing Intraday backtest.

Default mode is intentionally small and quick.
It does not place orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_intraday


OPENING_RANGE_BARS = 4
MAX_POSITIONS_PER_DAY = 3
MIN_TRADES_FOR_RESULT = 15
MAX_PARAM_SETS = 72


@dataclass(frozen=True)
class Params:
    min_breakout_pct: float
    min_close_strength: float
    max_entry_risk_pct: float
    initial_stop_loss_pct: float
    trailing_activation_profit_pct: float
    trailing_stop_pct: float


@dataclass(frozen=True)
class Trade:
    symbol: str
    session_date: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    breakout_pct: float
    close_strength: float
    entry_risk_pct: float
    exit_reason: str


@dataclass(frozen=True)
class OptimizationResult:
    params: Params
    trades: int
    win_rate_pct: float
    average_pnl_pct: float
    total_pnl_pct: float
    average_win_pct: float
    average_loss_pct: float
    best_trade_pct: float
    worst_trade_pct: float


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


def find_entry_bar(session: pd.DataFrame, params: Params) -> tuple[int, float, float, float] | None:
    if len(session) <= OPENING_RANGE_BARS:
        return None

    opening_range = session.iloc[:OPENING_RANGE_BARS]
    opening_range_high = float(opening_range["high"].max())
    opening_range_low = float(opening_range["low"].min())

    for position in range(OPENING_RANGE_BARS, len(session)):
        row = session.iloc[position]
        close = float(row["close"])
        breakout_pct = calculate_breakout_pct(close, opening_range_high)
        close_strength = calculate_close_strength(row)
        entry_risk_pct = calculate_entry_risk_pct(close, opening_range_low)

        if breakout_pct >= params.min_breakout_pct and close_strength >= params.min_close_strength and entry_risk_pct <= params.max_entry_risk_pct:
            return position, breakout_pct, close_strength, entry_risk_pct

    return None


def simulate_exit(symbol: str, session: pd.DataFrame, entry_position: int, breakout_pct: float, close_strength: float, entry_risk_pct: float, params: Params) -> Trade:
    entry_bar = session.iloc[entry_position]
    entry_price = float(entry_bar["close"])
    entry_time = str(entry_bar["date"])
    session_date = str(entry_bar["date"].date())

    initial_stop_price = entry_price * (1.0 - params.initial_stop_loss_pct / 100.0)
    activation_price = entry_price * (1.0 + params.trailing_activation_profit_pct / 100.0)
    highest_price = entry_price
    trailing_activated = False
    bars_after_entry = session.iloc[entry_position + 1 :]

    if bars_after_entry.empty:
        return Trade(symbol, session_date, entry_time, entry_time, entry_price, entry_price, 0.0, breakout_pct, close_strength, entry_risk_pct, "no bars after entry")

    last_bar = bars_after_entry.iloc[-1]

    for _, bar in bars_after_entry.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_time = str(bar["date"])
        highest_price = max(highest_price, bar_high)

        if bar_low <= initial_stop_price:
            pnl_pct = (initial_stop_price - entry_price) / entry_price * 100.0
            return Trade(symbol, session_date, entry_time, bar_time, entry_price, initial_stop_price, pnl_pct, breakout_pct, close_strength, entry_risk_pct, "initial stop-loss")

        if highest_price >= activation_price:
            trailing_activated = True

        if trailing_activated:
            trailing_stop_price = highest_price * (1.0 - params.trailing_stop_pct / 100.0)
            if bar_low <= trailing_stop_price:
                pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                return Trade(symbol, session_date, entry_time, bar_time, entry_price, trailing_stop_price, pnl_pct, breakout_pct, close_strength, entry_risk_pct, "trailing stop")

    exit_price = float(last_bar["close"])
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return Trade(symbol, session_date, entry_time, str(last_bar["date"]), entry_price, exit_price, pnl_pct, breakout_pct, close_strength, entry_risk_pct, "end of session")


def backtest_symbol(symbol: str, df: pd.DataFrame, params: Params) -> list[Trade]:
    trades: list[Trade] = []
    for _, session in df.groupby("session_date"):
        session = session.sort_values("date").reset_index(drop=True)
        entry = find_entry_bar(session, params)
        if entry is None:
            continue
        entry_position, breakout_pct, close_strength, entry_risk_pct = entry
        trades.append(simulate_exit(symbol, session, entry_position, breakout_pct, close_strength, entry_risk_pct, params))
    return trades


def limit_positions_per_day(trades: list[Trade]) -> list[Trade]:
    by_day: dict[str, list[Trade]] = {}
    for trade in trades:
        by_day.setdefault(trade.session_date, []).append(trade)

    selected: list[Trade] = []
    for day_trades in by_day.values():
        selected.extend(sorted(day_trades, key=lambda trade: (trade.breakout_pct, trade.close_strength, -trade.entry_risk_pct), reverse=True)[:MAX_POSITIONS_PER_DAY])
    return selected


def summarize(params: Params, trades: list[Trade]) -> OptimizationResult | None:
    if not trades:
        return None
    pnl_values = [trade.pnl_pct for trade in trades]
    wins = [pnl for pnl in pnl_values if pnl > 0]
    losses = [pnl for pnl in pnl_values if pnl <= 0]
    return OptimizationResult(params, len(trades), len(wins) / len(trades) * 100.0, sum(pnl_values) / len(pnl_values), sum(pnl_values), sum(wins) / len(wins) if wins else 0.0, sum(losses) / len(losses) if losses else 0.0, max(pnl_values), min(pnl_values))


def load_all_intraday_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for spec in UNIVERSE:
        try:
            df = load_intraday(spec.symbol).copy()
            df["session_date"] = df["date"].dt.date
            data[spec.symbol] = df
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {spec.symbol}: {exc}")
    return data


def build_param_grid() -> list[Params]:
    grid = [
        Params(*values)
        for values in product(
            [0.25, 0.50, 0.75],
            [0.60, 0.75],
            [2.0, 3.0],
            [0.8, 1.0],
            [0.6, 0.8, 1.0],
            [0.8, 1.0],
        )
    ]
    return grid[:MAX_PARAM_SETS]


def run_grid_search() -> list[OptimizationResult]:
    all_data = load_all_intraday_data()
    param_grid = build_param_grid()
    results: list[OptimizationResult] = []

    print(f"Testing {len(param_grid)} parameter sets on {len(all_data)} symbols...")

    for i, params in enumerate(param_grid, start=1):
        if i == 1 or i % 10 == 0 or i == len(param_grid):
            print(f"Progress: {i}/{len(param_grid)}")

        all_trades: list[Trade] = []
        for symbol, df in all_data.items():
            all_trades.extend(backtest_symbol(symbol, df, params))

        result = summarize(params, limit_positions_per_day(all_trades))
        if result is not None and result.trades >= MIN_TRADES_FOR_RESULT:
            results.append(result)

    return sorted(results, key=lambda result: (result.average_pnl_pct, result.total_pnl_pct, result.win_rate_pct), reverse=True)


def main() -> None:
    results = run_grid_search()
    print("\nOptimization: Momentum Trailing Intraday\n")
    print(f"Showing top 15 parameter sets with at least {MIN_TRADES_FOR_RESULT} trades.\n")

    if not results:
        print("No optimization results found.")
        return

    print("Rank | Trades | Win % | Avg PnL | Total PnL | Avg Win | Avg Loss | Break | Close | Risk | Stop | Act | Trail")
    print("------------------------------------------------------------------------------------------------")
    for i, result in enumerate(results[:15], start=1):
        p = result.params
        print(f"{i:>4} | {result.trades:>6} | {result.win_rate_pct:>5.1f} | {result.average_pnl_pct:>7.2f} | {result.total_pnl_pct:>9.2f} | {result.average_win_pct:>7.2f} | {result.average_loss_pct:>8.2f} | {p.min_breakout_pct:>5.2f} | {p.min_close_strength:>5.2f} | {p.max_entry_risk_pct:>4.1f} | {p.initial_stop_loss_pct:>4.1f} | {p.trailing_activation_profit_pct:>3.1f} | {p.trailing_stop_pct:>5.1f}")


if __name__ == "__main__":
    main()
