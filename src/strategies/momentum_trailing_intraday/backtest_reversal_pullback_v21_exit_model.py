"""Reversal pullback v21: v20 entry with reversal-specific exit model.

Builds on v20 quality sweet spot.

Entry stays unchanged:
- 15m breakout attempt: 1.0% - 1.8%
- 1m close strength: 0.80 - 0.90
- ADR >= 5.0%

Exit experiment:
- fixed take-profit for quick reversal capture
- fixed stop-loss
- time exit if reversal does not follow through quickly
- optional late trailing only after a larger move

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v20_quality_sweet_spot as v20

TAKE_PROFIT_PCT = 2.0
STOP_LOSS_PCT = 1.0
TIME_EXIT_BARS = 60  # 60 x 1m bars after entry.
TIME_EXIT_MIN_PNL_PCT = 0.50
TRAILING_ACTIVATION_PROFIT_PCT = 2.50
TRAILING_STOP_PCT = 1.20


def simulate_exit_v21(
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

    stop_price = entry_price * (1.0 - STOP_LOSS_PCT / 100.0)
    take_profit_price = entry_price * (1.0 + TAKE_PROFIT_PCT / 100.0)
    trailing_activation_price = entry_price * (1.0 + TRAILING_ACTIVATION_PROFIT_PCT / 100.0)

    highest_price = entry_price
    trailing_activated = False
    bars_after_entry = session.iloc[entry_position + 1 :]

    if bars_after_entry.empty:
        return v20.v17.bt.BacktestTrade(
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

        # Conservative order inside the same 1m bar: stop first, then TP.
        if bar_low <= stop_price:
            pnl_pct = (stop_price - entry_price) / entry_price * 100.0
            return v20.v17.bt.BacktestTrade(
                symbol,
                session_date,
                entry_time,
                bar_time,
                entry_price,
                stop_price,
                pnl_pct,
                "v21 stop-loss",
                breakout_pct,
                close_strength,
                entry_risk_pct,
                daily_trend_pct,
                setup_type,
            )

        if bar_high >= take_profit_price:
            pnl_pct = (take_profit_price - entry_price) / entry_price * 100.0
            return v20.v17.bt.BacktestTrade(
                symbol,
                session_date,
                entry_time,
                bar_time,
                entry_price,
                take_profit_price,
                pnl_pct,
                "v21 take-profit",
                breakout_pct,
                close_strength,
                entry_risk_pct,
                daily_trend_pct,
                setup_type,
            )

        if highest_price >= trailing_activation_price:
            trailing_activated = True

        if trailing_activated:
            trailing_stop_price = highest_price * (1.0 - TRAILING_STOP_PCT / 100.0)
            if bar_low <= trailing_stop_price:
                pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                return v20.v17.bt.BacktestTrade(
                    symbol,
                    session_date,
                    entry_time,
                    bar_time,
                    entry_price,
                    trailing_stop_price,
                    pnl_pct,
                    "v21 trailing stop",
                    breakout_pct,
                    close_strength,
                    entry_risk_pct,
                    daily_trend_pct,
                    setup_type,
                )

        if bars_held >= TIME_EXIT_BARS:
            pnl_pct = (bar_close - entry_price) / entry_price * 100.0
            if pnl_pct < TIME_EXIT_MIN_PNL_PCT:
                return v20.v17.bt.BacktestTrade(
                    symbol,
                    session_date,
                    entry_time,
                    bar_time,
                    entry_price,
                    bar_close,
                    pnl_pct,
                    "v21 time exit",
                    breakout_pct,
                    close_strength,
                    entry_risk_pct,
                    daily_trend_pct,
                    setup_type,
                )

    exit_price = float(last_bar["close"])
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return v20.v17.bt.BacktestTrade(
        symbol,
        session_date,
        entry_time,
        str(last_bar["date"]),
        entry_price,
        exit_price,
        pnl_pct,
        "v21 end of session",
        breakout_pct,
        close_strength,
        entry_risk_pct,
        daily_trend_pct,
        setup_type,
    )


v20.v17.bt.simulate_exit = simulate_exit_v21


def main():
    print("\nExperiment: reversal pullback v21 exit model")
    print("Entry: v20 quality sweet spot")
    print("Exit model:")
    print(f"- take_profit={TAKE_PROFIT_PCT:.2f}%")
    print(f"- stop_loss={STOP_LOSS_PCT:.2f}%")
    print(f"- time_exit_bars={TIME_EXIT_BARS}, min_pnl={TIME_EXIT_MIN_PNL_PCT:.2f}%")
    print(f"- trailing_activation={TRAILING_ACTIVATION_PROFIT_PCT:.2f}%, trailing_stop={TRAILING_STOP_PCT:.2f}%")
    v20.main()


if __name__ == "__main__":
    main()
