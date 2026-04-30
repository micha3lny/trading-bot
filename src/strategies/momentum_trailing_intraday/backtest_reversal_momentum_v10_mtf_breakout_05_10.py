"""Reversal-momentum v10 MTF.

Builds on v9 and changes only the 15m breakout range:
- breakout 0.50% to 1.00%

Reason: v9 analysis showed the 0.5-1.0 bucket had positive expectancy after costs,
while stronger breakouts were negative.

No orders are placed.
"""

from __future__ import annotations

import pandas as pd

from src.data.load_market_data import load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades

MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -5.0
MIN_BREAKOUT_PCT = 0.50
MAX_BREAKOUT_PCT = 1.00
MIN_NEXT_BAR_RETURN_PCT = 0.10
MIN_CONFIRMATION_CLOSE_STRENGTH = 0.50
MAX_CONFIRMATION_CLOSE_STRENGTH = 0.90
MAX_SETUP_ENTRY_RISK_PCT = 10.0

PULLBACK_LOOKAHEAD_5M_BARS = 9
MIN_PULLBACK_FROM_CONFIRMATION_PCT = 0.05
MAX_PULLBACK_FROM_CONFIRMATION_PCT = 1.50
MIN_5M_CLOSE_STRENGTH = 0.35
REQUIRE_5M_CLOSE_ABOVE_OR_HIGH = True
MAX_5M_ENTRY_RISK_PCT = 10.0


def calculate_return_pct(current_price: float, next_price: float) -> float | None:
    if current_price == 0:
        return None
    return (next_price - current_price) / current_price * 100.0


def calculate_next_bar_return(session: pd.DataFrame, position: int) -> float | None:
    if position + 1 >= len(session):
        return None
    return calculate_return_pct(float(session.iloc[position]["close"]), float(session.iloc[position + 1]["close"]))


def find_15m_setup(session_15m: pd.DataFrame, daily_trend_pct: float):
    if len(session_15m) <= bt.OPENING_RANGE_BARS + 1:
        return None
    if not (MIN_DAILY_TREND_PCT <= daily_trend_pct <= MAX_DAILY_TREND_PCT):
        return None

    opening = session_15m.iloc[: bt.OPENING_RANGE_BARS]
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())

    for breakout_position in range(bt.OPENING_RANGE_BARS, len(session_15m) - 1):
        breakout_row = session_15m.iloc[breakout_position]
        breakout_close = float(breakout_row["close"])
        if breakout_close <= or_high:
            continue

        breakout_pct = bt.calculate_breakout_pct(breakout_close, or_high)
        setup_entry_risk_pct = bt.calculate_entry_risk_pct(breakout_close, or_low)
        next_bar_return_pct = calculate_next_bar_return(session_15m, breakout_position)
        if next_bar_return_pct is None:
            return None

        confirmation_position = breakout_position + 1
        confirmation_row = session_15m.iloc[confirmation_position]
        confirmation_close_strength = bt.calculate_close_strength(confirmation_row)

        if (
            MIN_BREAKOUT_PCT <= breakout_pct <= MAX_BREAKOUT_PCT
            and setup_entry_risk_pct <= MAX_SETUP_ENTRY_RISK_PCT
            and next_bar_return_pct >= MIN_NEXT_BAR_RETURN_PCT
            and MIN_CONFIRMATION_CLOSE_STRENGTH <= confirmation_close_strength <= MAX_CONFIRMATION_CLOSE_STRENGTH
        ):
            return {
                "or_high": or_high,
                "or_low": or_low,
                "confirmation_time": confirmation_row["date"],
                "confirmation_close": float(confirmation_row["close"]),
                "daily_trend_pct": daily_trend_pct,
            }

        return None

    return None


def find_5m_pullback_entry(session_5m: pd.DataFrame, setup: dict):
    confirmation_time = pd.Timestamp(setup["confirmation_time"])
    confirmation_close = float(setup["confirmation_close"])
    or_high = float(setup["or_high"])
    or_low = float(setup["or_low"])

    after_confirmation = session_5m[session_5m["date"] > confirmation_time].copy()
    if after_confirmation.empty:
        return None

    after_confirmation = after_confirmation.sort_values("date").reset_index(drop=True)
    window = after_confirmation.iloc[:PULLBACK_LOOKAHEAD_5M_BARS]

    for _, row in window.iterrows():
        close = float(row["close"])
        low = float(row["low"])

        if REQUIRE_5M_CLOSE_ABOVE_OR_HIGH and close <= or_high:
            continue
        if confirmation_close == 0:
            continue

        pullback_pct = (confirmation_close - low) / confirmation_close * 100.0
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        breakout_pct = bt.calculate_breakout_pct(close, or_high)

        if (
            MIN_PULLBACK_FROM_CONFIRMATION_PCT <= pullback_pct <= MAX_PULLBACK_FROM_CONFIRMATION_PCT
            and close_strength >= MIN_5M_CLOSE_STRENGTH
            and entry_risk_pct <= MAX_5M_ENTRY_RISK_PCT
        ):
            return {
                "entry_time": row["date"],
                "breakout_pct": breakout_pct,
                "close_strength": close_strength,
                "entry_risk_pct": entry_risk_pct,
            }

    return None


def simulate_exit_on_5m(symbol: str, session_5m: pd.DataFrame, entry_time, breakout_pct, close_strength, entry_risk_pct, daily_trend_pct):
    session_5m = session_5m.reset_index(drop=True)
    matches = session_5m.index[session_5m["date"] == entry_time].tolist()
    if not matches:
        return None
    return bt.simulate_exit(
        symbol=symbol,
        session=session_5m,
        entry_position=matches[0],
        breakout_pct=breakout_pct,
        close_strength=close_strength,
        entry_risk_pct=entry_risk_pct,
        daily_trend_pct=daily_trend_pct,
        setup_type="reversal_momentum_v10_mtf_breakout_05_10",
    )


def backtest_symbol(symbol, intraday_15m, intraday_5m, daily, regimes):
    trades = []
    intraday_5m = intraday_5m.copy()
    intraday_5m["session_date"] = intraday_5m["date"].dt.date

    for session_date, session_15m in intraday_15m.groupby("session_date"):
        regime = regimes.get(str(session_date))
        if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
            continue

        session_15m = session_15m.sort_values("date").reset_index(drop=True)
        daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)
        setup = find_15m_setup(session_15m, daily_trend_pct)
        if setup is None:
            continue

        session_5m = intraday_5m[intraday_5m["session_date"] == session_date].sort_values("date").reset_index(drop=True)
        if session_5m.empty:
            continue

        entry = find_5m_pullback_entry(session_5m, setup)
        if entry is None:
            continue

        trade = simulate_exit_on_5m(
            symbol,
            session_5m,
            entry["entry_time"],
            entry["breakout_pct"],
            entry["close_strength"],
            entry["entry_risk_pct"],
            daily_trend_pct,
        )
        if trade is not None:
            trades.append(trade)

    return trades


def summarize_research(trades):
    print("\nMTF v10 research-mode signal summary")
    print("15m breakout 0.5-1.0 setup, 5m pullback entry, costs included.")
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
    intraday_15m, daily_data = bt.load_all_data()
    regimes = bt.build_market_regimes(intraday_15m)

    all_trades = []
    loaded_5m = 0
    for symbol, data_15m in intraday_15m.items():
        try:
            data_5m = load_intraday(symbol, interval="5m")
            loaded_5m += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {symbol}: no 5m data ({exc})")
            continue
        all_trades.extend(backtest_symbol(symbol, data_15m, data_5m, daily_data[symbol], regimes))

    net_trades = apply_costs_to_trades(all_trades)

    print("\nExperiment: reversal momentum v10 MTF breakout 0.5-1.0 (with costs)")
    print(f"Loaded 5m data for {loaded_5m} symbols")
    print(
        f"15m setup: trend<= {MAX_DAILY_TREND_PCT:.2f}%, "
        f"breakout=[{MIN_BREAKOUT_PCT:.2f}%, {MAX_BREAKOUT_PCT:.2f}%], "
        f"confirmation_cs=[{MIN_CONFIRMATION_CLOSE_STRENGTH:.2f}, {MAX_CONFIRMATION_CLOSE_STRENGTH:.2f}], "
        f"next_bar_return>={MIN_NEXT_BAR_RETURN_PCT:.2f}%"
    )
    print(
        f"5m entry: lookahead={PULLBACK_LOOKAHEAD_5M_BARS}, "
        f"pullback=[{MIN_PULLBACK_FROM_CONFIRMATION_PCT:.2f}%, {MAX_PULLBACK_FROM_CONFIRMATION_PCT:.2f}%], "
        f"close_strength>={MIN_5M_CLOSE_STRENGTH:.2f}, entry_risk<={MAX_5M_ENTRY_RISK_PCT:.2f}%"
    )
    print("Costs: default round-trip cost model applied")

    summarize_research(net_trades)
    df = export_trades(net_trades)
    analyze(df)


if __name__ == "__main__":
    main()
