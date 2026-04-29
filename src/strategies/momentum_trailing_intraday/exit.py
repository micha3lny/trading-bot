"""Exit logic for Momentum Trailing Intraday strategy.

This module does not place orders.
It simulates exit decisions for confirmed entry candidates using historical intraday bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.load_market_data import load_market_data_bundle
from src.strategies.momentum_trailing_intraday.entry import EntrySignal, find_entry_signals
from src.strategies.momentum_trailing_intraday.ranking import get_last_intraday_session


# Strategy-specific exit parameters.
# These are initial intraday 15m defaults. Backtesting will tune them later.
INITIAL_STOP_LOSS_PCT = 1.0
TRAILING_ACTIVATION_PROFIT_PCT = 0.8
TRAILING_STOP_PCT = 1.2
FORCE_EXIT_BEFORE_MARKET_CLOSE = True
FORCE_EXIT_BARS_BEFORE_CLOSE = 1


@dataclass(frozen=True)
class ExitSimulation:
    symbol: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    highest_price_after_entry: float
    trailing_activated: bool
    bars_held: int


def simulate_exit_for_signal(signal: EntrySignal, df_intraday: pd.DataFrame) -> ExitSimulation:
    """Simulate exit after entry using latest intraday session.

    Entry price is approximated as the signal last price for now.
    Later, this will be replaced by real-time/paper trade fill price.
    """
    session = get_last_intraday_session(df_intraday)

    if session.empty:
        return ExitSimulation(
            symbol=signal.symbol,
            entry_price=signal.last_price,
            exit_price=signal.last_price,
            pnl_pct=0.0,
            exit_reason="no intraday data",
            highest_price_after_entry=signal.last_price,
            trailing_activated=False,
            bars_held=0,
        )

    entry_price = signal.last_price
    initial_stop_price = entry_price * (1.0 - INITIAL_STOP_LOSS_PCT / 100.0)
    activation_price = entry_price * (1.0 + TRAILING_ACTIVATION_PROFIT_PCT / 100.0)

    highest_price = entry_price
    trailing_activated = False
    bars_held = 0

    # For the first implementation we simulate from the last available bar onward.
    # With historical data this usually means there are no future bars yet, so force-exit
    # may be the only available result. Backtesting will later walk forward from the true entry bar.
    bars_after_entry = session.iloc[-1:]

    for _, bar in bars_after_entry.iterrows():
        bars_held += 1
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        highest_price = max(highest_price, bar_high)

        if bar_low <= initial_stop_price:
            pnl_pct = (initial_stop_price - entry_price) / entry_price * 100.0
            return ExitSimulation(
                symbol=signal.symbol,
                entry_price=entry_price,
                exit_price=initial_stop_price,
                pnl_pct=pnl_pct,
                exit_reason="initial stop-loss",
                highest_price_after_entry=highest_price,
                trailing_activated=trailing_activated,
                bars_held=bars_held,
            )

        if highest_price >= activation_price:
            trailing_activated = True

        if trailing_activated:
            trailing_stop_price = highest_price * (1.0 - TRAILING_STOP_PCT / 100.0)
            if bar_low <= trailing_stop_price:
                pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                return ExitSimulation(
                    symbol=signal.symbol,
                    entry_price=entry_price,
                    exit_price=trailing_stop_price,
                    pnl_pct=pnl_pct,
                    exit_reason="trailing stop",
                    highest_price_after_entry=highest_price,
                    trailing_activated=trailing_activated,
                    bars_held=bars_held,
                )

        if FORCE_EXIT_BEFORE_MARKET_CLOSE:
            pnl_pct = (bar_close - entry_price) / entry_price * 100.0
            return ExitSimulation(
                symbol=signal.symbol,
                entry_price=entry_price,
                exit_price=bar_close,
                pnl_pct=pnl_pct,
                exit_reason="force exit before market close",
                highest_price_after_entry=highest_price,
                trailing_activated=trailing_activated,
                bars_held=bars_held,
            )

    return ExitSimulation(
        symbol=signal.symbol,
        entry_price=entry_price,
        exit_price=entry_price,
        pnl_pct=0.0,
        exit_reason="position still open",
        highest_price_after_entry=highest_price,
        trailing_activated=trailing_activated,
        bars_held=bars_held,
    )


def simulate_exits_for_final_picks() -> list[ExitSimulation]:
    signals = [signal for signal in find_entry_signals() if signal.is_final_pick]
    simulations: list[ExitSimulation] = []

    for signal in signals:
        bundle = load_market_data_bundle(signal.symbol)
        simulations.append(simulate_exit_for_signal(signal, bundle.intraday))

    return simulations


def main() -> None:
    simulations = simulate_exits_for_final_picks()

    print("\nExit simulation: Momentum Trailing Intraday\n")
    print(
        f"Params: initial_stop={INITIAL_STOP_LOSS_PCT:.2f}%, "
        f"trailing_activation={TRAILING_ACTIVATION_PROFIT_PCT:.2f}%, "
        f"trailing_stop={TRAILING_STOP_PCT:.2f}%"
    )
    print()

    if not simulations:
        print("No final picks to simulate.")
        return

    print("Symbol | Entry | Exit | PnL % | High | Trail | Bars | Reason")
    print("--------------------------------------------------------------")

    for result in simulations:
        trail_text = "YES" if result.trailing_activated else "NO"
        print(
            f"{result.symbol:<6} | "
            f"{result.entry_price:>7.2f} | "
            f"{result.exit_price:>7.2f} | "
            f"{result.pnl_pct:>5.2f} | "
            f"{result.highest_price_after_entry:>7.2f} | "
            f"{trail_text:<5} | "
            f"{result.bars_held:>4} | "
            f"{result.exit_reason}"
        )


if __name__ == "__main__":
    main()
