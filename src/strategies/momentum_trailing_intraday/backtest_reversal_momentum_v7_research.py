"""Reversal-momentum v7 research mode.

Purpose:
- inspect all valid signals without portfolio ranking
- no max 3 positions/day selection
- equal notional per signal for diagnostics
- costs included

This is not a production portfolio simulation. It is a signal-quality research run.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades

MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -10.0
MIN_BREAKOUT_PCT = 0.75
MAX_BREAKOUT_PCT = 1.75
MIN_NEXT_BAR_RETURN_PCT = 0.10
MIN_CONFIRMATION_CLOSE_STRENGTH = 0.60
MAX_CONFIRMATION_CLOSE_STRENGTH = 0.80
MAX_ENTRY_RISK_PCT = 10.0

PULLBACK_LOOKAHEAD_BARS = 3
MIN_PULLBACK_FROM_CONFIRMATION_PCT = 0.15
MAX_PULLBACK_FROM_CONFIRMATION_PCT = 1.20
MIN_PULLBACK_CLOSE_STRENGTH = 0.35
REQUIRE_PULLBACK_CLOSE_ABOVE_OR_HIGH = True


def calculate_next_bar_return(session: pd.DataFrame, position: int) -> float | None:
    if position + 1 >= len(session):
        return None
    current_close = float(session.iloc[position]["close"])
    next_close = float(session.iloc[position + 1]["close"])
    if current_close == 0:
        return None
    return (next_close - current_close) / current_close * 100.0


def find_pullback_entry(session, confirmation_position, opening_range_high, opening_range_low):
    confirmation_close = float(session.iloc[confirmation_position]["close"])
    last_position = min(len(session), confirmation_position + 1 + PULLBACK_LOOKAHEAD_BARS)

    for entry_position in range(confirmation_position + 1, last_position):
        row = session.iloc[entry_position]
        close = float(row["close"])
        low = float(row["low"])

        if REQUIRE_PULLBACK_CLOSE_ABOVE_OR_HIGH and close <= opening_range_high:
            continue
        if confirmation_close == 0:
            continue

        pullback_from_confirmation_pct = (confirmation_close - low) / confirmation_close * 100.0
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, opening_range_low)
        breakout_pct = bt.calculate_breakout_pct(close, opening_range_high)

        if (
            MIN_PULLBACK_FROM_CONFIRMATION_PCT <= pullback_from_confirmation_pct <= MAX_PULLBACK_FROM_CONFIRMATION_PCT
            and close_strength >= MIN_PULLBACK_CLOSE_STRENGTH
            and entry_risk_pct <= MAX_ENTRY_RISK_PCT
        ):
            return entry_position, breakout_pct, close_strength, entry_risk_pct

    return None


def find_candidate(session: pd.DataFrame, daily_trend_pct: float):
    if len(session) <= bt.OPENING_RANGE_BARS + 2:
        return None
    if not (MIN_DAILY_TREND_PCT <= daily_trend_pct <= MAX_DAILY_TREND_PCT):
        return None

    opening = session.iloc[: bt.OPENING_RANGE_BARS]
    opening_range_high = float(opening["high"].max())
    opening_range_low = float(opening["low"].min())

    for breakout_position in range(bt.OPENING_RANGE_BARS, len(session) - 2):
        breakout_row = session.iloc[breakout_position]
        breakout_close = float(breakout_row["close"])
        if breakout_close <= opening_range_high:
            continue

        breakout_pct = bt.calculate_breakout_pct(breakout_close, opening_range_high)
        next_bar_return_pct = calculate_next_bar_return(session, breakout_position)
        if next_bar_return_pct is None:
            return None

        confirmation_position = breakout_position + 1
        confirmation_row = session.iloc[confirmation_position]
        confirmation_close_strength = bt.calculate_close_strength(confirmation_row)

        if (
            MIN_BREAKOUT_PCT <= breakout_pct <= MAX_BREAKOUT_PCT
            and next_bar_return_pct >= MIN_NEXT_BAR_RETURN_PCT
            and MIN_CONFIRMATION_CLOSE_STRENGTH <= confirmation_close_strength <= MAX_CONFIRMATION_CLOSE_STRENGTH
        ):
            return find_pullback_entry(
                session,
                confirmation_position,
                opening_range_high,
                opening_range_low,
            )

        # First breakout only for this symbol/session.
        return None

    return None


def backtest_symbol(symbol, intraday, daily, regimes):
    trades = []
    for session_date, session in intraday.groupby("session_date"):
        regime = regimes.get(str(session_date))
        if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
            continue

        session = session.sort_values("date").reset_index(drop=True)
        daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)
        candidate = find_candidate(session, daily_trend_pct)
        if candidate is None:
            continue

        entry_position, breakout_pct, close_strength, entry_risk_pct = candidate
        trades.append(
            bt.simulate_exit(
                symbol,
                session,
                entry_position,
                breakout_pct,
                close_strength,
                entry_risk_pct,
                daily_trend_pct,
                "reversal_momentum_v7_research",
            )
        )

    return trades


def summarize_research(trades):
    print("\nResearch-mode signal summary")
    print("No portfolio ranking. No max positions/day. Each row is one valid signal.")
    print(f"Signals: {len(trades)}")
    if not trades:
        return

    pnl_values = [trade.pnl_pct for trade in trades]
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p <= 0]
    print(f"Win rate: {len(wins) / len(trades) * 100.0:.2f}%")
    print(f"Average PnL after costs: {sum(pnl_values) / len(pnl_values):.2f}%")
    print(f"Total PnL sum after costs: {sum(pnl_values):.2f}%")
    print(f"Best signal: {max(pnl_values):.2f}%")
    print(f"Worst signal: {min(pnl_values):.2f}%")
    if wins:
        print(f"Average win: {sum(wins) / len(wins):.2f}%")
    if losses:
        print(f"Average loss: {sum(losses) / len(losses):.2f}%")

    by_day = {}
    for trade in trades:
        by_day.setdefault(trade.session_date, []).append(trade)
    print(f"Signal days: {len(by_day)}")
    print(f"Max signals on one day: {max(len(day_trades) for day_trades in by_day.values())}")

    print("\nRecent signals")
    print("Date | Symbol | Entry | Exit | PnL % | Trend % | Break % | CloseStr | Risk % | Reason")
    print("-----------------------------------------------------------------------------------------")
    for trade in trades[-30:]:
        print(
            f"{trade.session_date} | {trade.symbol:<6} | {trade.entry_price:>7.2f} | "
            f"{trade.exit_price:>7.2f} | {trade.pnl_pct:>6.2f} | "
            f"{trade.daily_trend_pct:>7.2f} | {trade.breakout_pct:>7.2f} | "
            f"{trade.close_strength:>8.2f} | {trade.entry_risk_pct:>6.2f} | {trade.exit_reason}"
        )


def main():
    intraday_data, daily_data = bt.load_all_data()
    regimes = bt.build_market_regimes(intraday_data)

    all_trades = []
    for symbol, intraday in intraday_data.items():
        all_trades.extend(backtest_symbol(symbol, intraday, daily_data[symbol], regimes))

    net_trades = apply_costs_to_trades(all_trades)

    print("\nExperiment: reversal momentum v7 research mode (with costs)")
    print(
        f"Setup filters: trend<= {MAX_DAILY_TREND_PCT:.2f}%, "
        f"breakout=[{MIN_BREAKOUT_PCT:.2f}%, {MAX_BREAKOUT_PCT:.2f}%], "
        f"confirmation_cs=[{MIN_CONFIRMATION_CLOSE_STRENGTH:.2f}, {MAX_CONFIRMATION_CLOSE_STRENGTH:.2f}], "
        f"next_bar_return>={MIN_NEXT_BAR_RETURN_PCT:.2f}%"
    )
    print(
        f"Pullback filters: lookahead={PULLBACK_LOOKAHEAD_BARS}, "
        f"pullback=[{MIN_PULLBACK_FROM_CONFIRMATION_PCT:.2f}%, {MAX_PULLBACK_FROM_CONFIRMATION_PCT:.2f}%], "
        f"pullback_close_strength>={MIN_PULLBACK_CLOSE_STRENGTH:.2f}, "
        f"entry_risk<={MAX_ENTRY_RISK_PCT:.2f}%"
    )
    print("Costs: default round-trip cost model applied")

    summarize_research(net_trades)
    df = export_trades(net_trades)
    analyze(df)


if __name__ == "__main__":
    main()
