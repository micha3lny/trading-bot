"""Unified experiment runner for Reversal Pullback MTF strategy.

This file replaces the need to create a new backtest file for every vXX tweak.
It centralizes:
- 15m setup filters
- 5m pullback filters
- 1m trigger filters
- market/ADR filters
- optional symbol and trend-regime filters
- exit model

Run examples:
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_experiments --config v22
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_experiments --config v26
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_experiments --list

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v17_1m_entry as base


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str

    enable_market_regime_filter: bool = True
    min_avg_daily_range_pct: float = 5.0
    excluded_symbols: frozenset[str] = field(default_factory=frozenset)

    min_daily_trend_pct: float = -999.0
    max_daily_trend_pct: float = -5.0
    allowed_daily_trend_ranges: tuple[tuple[float, float], ...] | None = None

    min_breakout_pct: float = 1.0
    max_breakout_pct: float = 1.8
    min_next_bar_return_pct: float = 0.0
    min_confirmation_close_strength: float = 0.40
    max_confirmation_close_strength: float = 0.95
    max_setup_entry_risk_pct: float = 12.0

    pullback_lookahead_5m_bars: int = 12
    min_pullback_from_confirmation_pct: float = 0.30
    max_pullback_from_confirmation_pct: float = 2.50
    min_5m_close_strength: float = 0.00
    max_5m_close_strength: float = 0.60
    max_5m_entry_risk_pct: float = 10.0
    max_close_below_or_high_pct: float = 0.75

    trigger_lookahead_1m_bars: int = 8
    min_1m_close_strength: float = 0.80
    max_1m_close_strength: float | None = 0.90
    max_1m_entry_risk_pct: float = 10.0
    require_1m_close_above_prev_close: bool = True
    require_1m_close_above_prev_high: bool = False
    max_1m_close_below_or_high_pct: float = 0.75

    take_profit_pct: float | None = 3.0
    stop_loss_pct: float = 1.0
    time_exit_bars: int | None = 60
    time_exit_min_pnl_pct: float = 0.50
    trailing_activation_profit_pct: float = 1.50
    trailing_stop_pct: float = 1.00


CONFIGS: dict[str, ExperimentConfig] = {
    "v20": ExperimentConfig(
        name="v20",
        description="quality sweet spot entry, original shared exit",
        take_profit_pct=None,
        trailing_activation_profit_pct=1.2,
        trailing_stop_pct=1.8,
    ),
    "v22": ExperimentConfig(
        name="v22",
        description="v20 quality sweet spot + smart exit",
    ),
    "scaled": ExperimentConfig(
        name="scaled",
        description="wider entry based on diagnostics + smart exit",
        min_breakout_pct=0.80,
        max_breakout_pct=2.20,
        max_5m_close_strength=0.70,
        max_1m_close_strength=0.92,
    ),
    "expanded": ExperimentConfig(
        name="expanded",
        description="scaled entry + lower ADR + market regime disabled",
        enable_market_regime_filter=False,
        min_avg_daily_range_pct=4.0,
        min_breakout_pct=0.80,
        max_breakout_pct=2.20,
        max_5m_close_strength=0.70,
        max_1m_close_strength=0.92,
    ),
    "more_entries": ExperimentConfig(
        name="more_entries",
        description="diagnostic wider entry to intentionally increase sample size",
        enable_market_regime_filter=False,
        min_avg_daily_range_pct=4.0,
        max_daily_trend_pct=-3.0,
        min_breakout_pct=0.60,
        max_breakout_pct=2.50,
        min_confirmation_close_strength=0.30,
        max_confirmation_close_strength=1.00,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.80,
        min_1m_close_strength=0.70,
        max_1m_close_strength=0.95,
    ),
    "v25": ExperimentConfig(
        name="v25",
        description="candidate compromise: enough entries without accepting the worst more_entries noise",
        enable_market_regime_filter=False,
        min_avg_daily_range_pct=4.5,
        max_daily_trend_pct=-3.0,
        min_breakout_pct=1.00,
        max_breakout_pct=2.20,
        min_confirmation_close_strength=0.30,
        max_confirmation_close_strength=1.00,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.75,
        min_1m_close_strength=0.80,
        max_1m_close_strength=0.90,
    ),
    "v25_quality": ExperimentConfig(
        name="v25_quality",
        description="quality-biased v25: stricter trend/ADR while keeping market regime disabled",
        enable_market_regime_filter=False,
        min_avg_daily_range_pct=5.0,
        max_daily_trend_pct=-5.0,
        min_breakout_pct=1.00,
        max_breakout_pct=2.20,
        min_confirmation_close_strength=0.35,
        max_confirmation_close_strength=0.95,
        min_pullback_from_confirmation_pct=0.25,
        max_pullback_from_confirmation_pct=2.75,
        max_5m_close_strength=0.70,
        min_1m_close_strength=0.80,
        max_1m_close_strength=0.90,
    ),
    "v26": ExperimentConfig(
        name="v26",
        description="data-driven filters from segmentation: tight breakout/CS, trend pockets, symbol exclusions",
        enable_market_regime_filter=False,
        min_avg_daily_range_pct=4.5,
        min_daily_trend_pct=-30.0,
        max_daily_trend_pct=-3.0,
        allowed_daily_trend_ranges=((-30.0, -20.0), (-5.0, -3.0)),
        min_breakout_pct=1.00,
        max_breakout_pct=1.50,
        min_confirmation_close_strength=0.30,
        max_confirmation_close_strength=1.00,
        max_setup_entry_risk_pct=10.0,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.75,
        max_5m_entry_risk_pct=7.0,
        min_1m_close_strength=0.80,
        max_1m_close_strength=0.85,
        max_1m_entry_risk_pct=7.0,
        excluded_symbols=frozenset({"SMCI", "UUUU", "PINS", "SOUN", "RKLB", "TEAM", "SOXS"}),
    ),
}


def apply_config(config: ExperimentConfig) -> None:
    base.bt.ENABLE_MARKET_REGIME_FILTER = config.enable_market_regime_filter

    base.MIN_AVG_DAILY_RANGE_PCT = config.min_avg_daily_range_pct

    base.MIN_DAILY_TREND_PCT = config.min_daily_trend_pct
    base.MAX_DAILY_TREND_PCT = config.max_daily_trend_pct

    base.MIN_BREAKOUT_PCT = config.min_breakout_pct
    base.MAX_BREAKOUT_PCT = config.max_breakout_pct
    base.MIN_NEXT_BAR_RETURN_PCT = config.min_next_bar_return_pct
    base.MIN_CONFIRMATION_CLOSE_STRENGTH = config.min_confirmation_close_strength
    base.MAX_CONFIRMATION_CLOSE_STRENGTH = config.max_confirmation_close_strength
    base.MAX_SETUP_ENTRY_RISK_PCT = config.max_setup_entry_risk_pct

    base.PULLBACK_LOOKAHEAD_5M_BARS = config.pullback_lookahead_5m_bars
    base.MIN_PULLBACK_FROM_CONFIRMATION_PCT = config.min_pullback_from_confirmation_pct
    base.MAX_PULLBACK_FROM_CONFIRMATION_PCT = config.max_pullback_from_confirmation_pct
    base.MIN_5M_CLOSE_STRENGTH = config.min_5m_close_strength
    base.MAX_5M_CLOSE_STRENGTH = config.max_5m_close_strength
    base.MAX_5M_ENTRY_RISK_PCT = config.max_5m_entry_risk_pct
    base.MAX_CLOSE_BELOW_OR_HIGH_PCT = config.max_close_below_or_high_pct

    base.TRIGGER_LOOKAHEAD_1M_BARS = config.trigger_lookahead_1m_bars
    base.MIN_1M_CLOSE_STRENGTH = config.min_1m_close_strength
    base.MAX_1M_ENTRY_RISK_PCT = config.max_1m_entry_risk_pct
    base.REQUIRE_1M_CLOSE_ABOVE_PREV_CLOSE = config.require_1m_close_above_prev_close
    base.REQUIRE_1M_CLOSE_ABOVE_PREV_HIGH = config.require_1m_close_above_prev_high
    base.MAX_1M_CLOSE_BELOW_OR_HIGH_PCT = config.max_1m_close_below_or_high_pct

    patch_1m_upper_bound(config)
    patch_trend_ranges(config)
    patch_symbol_exclusions(config)
    patch_exit(config)


def patch_trend_ranges(config: ExperimentConfig) -> None:
    if config.allowed_daily_trend_ranges is None:
        return

    original_find_15m_setup: Callable = base.find_15m_setup

    def find_15m_setup(session_15m, daily_trend_pct: float):
        in_allowed_range = any(low <= daily_trend_pct <= high for low, high in config.allowed_daily_trend_ranges or ())
        if not in_allowed_range:
            return None
        return original_find_15m_setup(session_15m, daily_trend_pct)

    base.find_15m_setup = find_15m_setup


def patch_symbol_exclusions(config: ExperimentConfig) -> None:
    if not config.excluded_symbols:
        return

    original_backtest_symbol: Callable = base.backtest_symbol

    def backtest_symbol(symbol, data_15m, data_5m, data_1m, daily, regimes):
        if symbol in config.excluded_symbols:
            return []
        return original_backtest_symbol(symbol, data_15m, data_5m, data_1m, daily, regimes)

    base.backtest_symbol = backtest_symbol


def patch_1m_upper_bound(config: ExperimentConfig) -> None:
    original_find_1m_entry_trigger: Callable = base.find_1m_entry_trigger

    def find_1m_entry_trigger(session_1m, pullback):
        pullback_time = base.pd.Timestamp(pullback["pullback_time"])
        after_pullback = session_1m[session_1m["date"] >= pullback_time].copy()
        if after_pullback.empty:
            return None

        after_pullback = after_pullback.sort_values("date").reset_index(drop=True)
        window = after_pullback.iloc[: base.TRIGGER_LOOKAHEAD_1M_BARS].reset_index(drop=True)

        if config.max_1m_close_strength is None:
            return original_find_1m_entry_trigger(window, pullback)

        for idx in range(1, len(window)):
            row = window.iloc[idx]
            close_strength = base.bt.calculate_close_strength(row)
            if close_strength > config.max_1m_close_strength:
                continue

            candidate_window = window.iloc[idx - 1 : idx + 1].copy().reset_index(drop=True)
            result = original_find_1m_entry_trigger(candidate_window, pullback)
            if result is not None:
                return result

        return None

    base.find_1m_entry_trigger = find_1m_entry_trigger


def patch_exit(config: ExperimentConfig) -> None:
    def simulate_exit(
        symbol: str,
        session,
        entry_position: int,
        breakout_pct: float,
        close_strength: float,
        entry_risk_pct: float,
        daily_trend_pct: float,
        setup_type: str,
    ):
        entry_bar = session.iloc[entry_position]
        entry_price = float(entry_bar["close"])
        entry_time = str(entry_bar["date"])
        session_date = str(entry_bar["date"].date())

        stop_price = entry_price * (1.0 - config.stop_loss_pct / 100.0)
        take_profit_price = None
        if config.take_profit_pct is not None:
            take_profit_price = entry_price * (1.0 + config.take_profit_pct / 100.0)
        trailing_activation_price = entry_price * (1.0 + config.trailing_activation_profit_pct / 100.0)

        highest_price = entry_price
        trailing_activated = False
        bars_after_entry = session.iloc[entry_position + 1 :]

        if bars_after_entry.empty:
            return base.bt.BacktestTrade(
                symbol,
                session_date,
                entry_time,
                entry_time,
                entry_price,
                entry_price,
                0.0,
                "no bars after entry",
                breakout_pct,
                close_strength,
                entry_risk_pct,
                daily_trend_pct,
                setup_type,
            )

        last_bar = bars_after_entry.iloc[-1]
        for bars_held, (_, bar) in enumerate(bars_after_entry.iterrows(), start=1):
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            bar_time = str(bar["date"])
            highest_price = max(highest_price, bar_high)

            if bar_low <= stop_price:
                pnl_pct = (stop_price - entry_price) / entry_price * 100.0
                return base.bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, stop_price, pnl_pct, "stop-loss", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

            if take_profit_price is not None and bar_high >= take_profit_price:
                pnl_pct = (take_profit_price - entry_price) / entry_price * 100.0
                return base.bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, take_profit_price, pnl_pct, "take-profit", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

            if highest_price >= trailing_activation_price:
                trailing_activated = True

            if trailing_activated:
                trailing_stop_price = highest_price * (1.0 - config.trailing_stop_pct / 100.0)
                if bar_low <= trailing_stop_price:
                    pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                    return base.bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, trailing_stop_price, pnl_pct, "trailing stop", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

            if config.time_exit_bars is not None and bars_held >= config.time_exit_bars:
                pnl_pct = (bar_close - entry_price) / entry_price * 100.0
                if pnl_pct < config.time_exit_min_pnl_pct:
                    return base.bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, bar_close, pnl_pct, "time exit", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

        exit_price = float(last_bar["close"])
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        return base.bt.BacktestTrade(symbol, session_date, entry_time, str(last_bar["date"]), entry_price, exit_price, pnl_pct, "end of session", breakout_pct, close_strength, entry_risk_pct, daily_trend_pct, setup_type)

    base.bt.simulate_exit = simulate_exit


def print_config(config: ExperimentConfig) -> None:
    print(f"\nExperiment: {config.name}")
    print(config.description)
    print("\nFilters:")
    print(f"- market_regime_filter={config.enable_market_regime_filter}")
    print(f"- ADR >= {config.min_avg_daily_range_pct:.2f}%")
    print(f"- excluded_symbols={sorted(config.excluded_symbols)}")
    print(f"- daily_trend: {config.min_daily_trend_pct:.2f}% to {config.max_daily_trend_pct:.2f}%")
    if config.allowed_daily_trend_ranges is not None:
        print(f"- allowed_daily_trend_ranges={config.allowed_daily_trend_ranges}")
    print(f"- 15m breakout: {config.min_breakout_pct:.2f}% to {config.max_breakout_pct:.2f}%")
    print(f"- 15m confirmation CS: {config.min_confirmation_close_strength:.2f} to {config.max_confirmation_close_strength:.2f}")
    print(f"- 5m pullback: {config.min_pullback_from_confirmation_pct:.2f}% to {config.max_pullback_from_confirmation_pct:.2f}%")
    print(f"- 5m close_strength <= {config.max_5m_close_strength:.2f}")
    print(f"- 5m entry_risk <= {config.max_5m_entry_risk_pct:.2f}%")
    upper_1m = "none" if config.max_1m_close_strength is None else f"{config.max_1m_close_strength:.2f}"
    print(f"- 1m close_strength: {config.min_1m_close_strength:.2f} to {upper_1m}")
    print(f"- 1m entry_risk <= {config.max_1m_entry_risk_pct:.2f}%")
    print("\nExit:")
    print(f"- take_profit={config.take_profit_pct}")
    print(f"- stop_loss={config.stop_loss_pct:.2f}%")
    print(f"- trailing_activation={config.trailing_activation_profit_pct:.2f}%, trailing_stop={config.trailing_stop_pct:.2f}%")
    print(f"- time_exit_bars={config.time_exit_bars}, min_pnl={config.time_exit_min_pnl_pct:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v22", choices=sorted(CONFIGS), help="Experiment config to run")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for config in CONFIGS.values():
            print(f"- {config.name}: {config.description}")
        return

    config = CONFIGS[args.config]
    apply_config(config)
    print_config(config)
    base.main()


if __name__ == "__main__":
    main()
