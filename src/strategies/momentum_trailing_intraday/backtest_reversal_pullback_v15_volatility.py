"""Reversal pullback v15: full universe + volatility filter.

Builds on v12 best-known config, not v14.

Goal:
- stop limiting research to the 51-symbol day-trading subset
- use every symbol that has local 1D, 15m and 5m data
- filter for high volatility names using daily average range percent

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_daily, load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades

# v12-style setup/entry.
MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -5.0
MIN_BREAKOUT_PCT = 0.50
MAX_BREAKOUT_PCT = 2.50
MIN_NEXT_BAR_RETURN_PCT = 0.00
MIN_CONFIRMATION_CLOSE_STRENGTH = 0.40
MAX_CONFIRMATION_CLOSE_STRENGTH = 0.95
MAX_SETUP_ENTRY_RISK_PCT = 12.0

PULLBACK_LOOKAHEAD_5M_BARS = 12
MIN_PULLBACK_FROM_CONFIRMATION_PCT = 0.30
MAX_PULLBACK_FROM_CONFIRMATION_PCT = 2.50
MIN_5M_CLOSE_STRENGTH = 0.00
MAX_5M_CLOSE_STRENGTH = 0.60
MAX_5M_ENTRY_RISK_PCT = 10.0
MAX_CLOSE_BELOW_OR_HIGH_PCT = 0.75

# Volatility filter: average daily high-low range percent over previous 20 sessions.
VOLATILITY_LOOKBACK_DAYS = 20
MIN_AVG_DAILY_RANGE_PCT = 4.0
INITIAL_CAPITAL = 10_000.0


def prepare_daily_data(symbol: str) -> pd.DataFrame:
    daily = load_daily(symbol).copy()
    daily["session_date"] = daily["date"].dt.date
    daily["ma20"] = daily["close"].rolling(20).mean()
    daily["daily_trend_pct"] = (daily["close"] - daily["ma20"]) / daily["ma20"] * 100.0
    daily["daily_range_pct"] = (daily["high"] - daily["low"]) / daily["close"] * 100.0
    daily["avg_daily_range_pct"] = daily["daily_range_pct"].rolling(VOLATILITY_LOOKBACK_DAYS).mean()
    return daily


def get_daily_trend_before_session(daily: pd.DataFrame, session_date) -> float:
    history = daily[daily["session_date"] < session_date].dropna(subset=["daily_trend_pct"])
    if history.empty:
        return 0.0
    return float(history.iloc[-1]["daily_trend_pct"])


def get_avg_daily_range_before_session(daily: pd.DataFrame, session_date) -> float:
    history = daily[daily["session_date"] < session_date].dropna(subset=["avg_daily_range_pct"])
    if history.empty:
        return 0.0
    return float(history.iloc[-1]["avg_daily_range_pct"])


def load_full_available_data():
    intraday_15m = {}
    intraday_5m = {}
    daily_data = {}
    for spec in UNIVERSE:
        symbol = spec.symbol
        try:
            data_15m = load_intraday(symbol, interval="15m").copy()
            data_5m = load_intraday(symbol, interval="5m").copy()
            daily = prepare_daily_data(symbol)
        except Exception:
            continue

        data_15m["session_date"] = data_15m["date"].dt.date
        data_5m["session_date"] = data_5m["date"].dt.date
        intraday_15m[symbol] = data_15m
        intraday_5m[symbol] = data_5m
        daily_data[symbol] = daily

    print(f"Loaded full available universe: {len(intraday_15m)} symbols with 1D + 15m + 5m data")
    return intraday_15m, intraday_5m, daily_data


def calculate_return_pct(current_price: float, next_price: float) -> float | None:
    if current_price == 0:
        return None
    return (next_price - current_price) / current_price * 100.0


def calculate_next_bar_return(session: pd.DataFrame, position: int) -> float | None:
    if position + 1 >= len(session):
        return None
    return calculate_return_pct(float(session.iloc[position]["close"]), float(session.iloc[position + 1]["close"]))


def distance_below_or_high_pct(close: float, opening_range_high: float) -> float:
    if opening_range_high == 0 or close >= opening_range_high:
        return 0.0
    return (opening_range_high - close) / opening_range_high * 100.0


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


def find_5m_reversal_pullback_entry(session_5m: pd.DataFrame, setup: dict):
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
        if confirmation_close == 0:
            continue

        pullback_pct = (confirmation_close - low) / confirmation_close * 100.0
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        below_or_pct = distance_below_or_high_pct(close, or_high)
        breakout_pct = bt.calculate_breakout_pct(close, or_high)

        if (
            MIN_PULLBACK_FROM_CONFIRMATION_PCT <= pullback_pct <= MAX_PULLBACK_FROM_CONFIRMATION_PCT
            and MIN_5M_CLOSE_STRENGTH <= close_strength <= MAX_5M_CLOSE_STRENGTH
            and entry_risk_pct <= MAX_5M_ENTRY_RISK_PCT
            and below_or_pct <= MAX_CLOSE_BELOW_OR_HIGH_PCT
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
        setup_type="reversal_pullback_v15_volatility",
    )


def backtest_symbol(symbol, data_15m, data_5m, daily, regimes):
    trades = []
    for session_date, session_15m in data_15m.groupby("session_date"):
        regime = regimes.get(str(session_date))
        if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
            continue

        daily_trend_pct = get_daily_trend_before_session(daily, session_date)
        avg_daily_range_pct = get_avg_daily_range_before_session(daily, session_date)
        if avg_daily_range_pct < MIN_AVG_DAILY_RANGE_PCT:
            continue

        session_15m = session_15m.sort_values("date").reset_index(drop=True)
        setup = find_15m_setup(session_15m, daily_trend_pct)
        if setup is None:
            continue

        session_5m = data_5m[data_5m["session_date"] == session_date].sort_values("date").reset_index(drop=True)
        if session_5m.empty:
            continue

        entry = find_5m_reversal_pullback_entry(session_5m, setup)
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


def pseudo_equity(trades):
    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_drawdown_pct = 0.0
    for trade in sorted(trades, key=lambda t: (t.session_date, t.entry_time, t.symbol)):
        capital *= 1.0 + trade.pnl_pct / 100.0
        peak = max(peak, capital)
        max_drawdown_pct = min(max_drawdown_pct, (capital - peak) / peak * 100.0)
    return capital, (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, max_drawdown_pct


def summarize_research(trades):
    print("\nMTF v15 volatility-filtered research summary")
    print("v12-style reversal pullback, full available universe, volatility filter, costs included.")
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

    final_capital, return_pct, max_drawdown_pct = pseudo_equity(trades)
    print("\nPseudo equity simulation")
    print("Assumption: every signal compounds sequentially at 100% notional. Research only, not portfolio sizing.")
    print(f"Initial capital: {INITIAL_CAPITAL:.2f}")
    print(f"Final capital:   {final_capital:.2f}")
    print(f"Return:          {return_pct:.2f}%")
    print(f"Max drawdown:    {max_drawdown_pct:.2f}%")

    by_day = {}
    by_symbol = {}
    for trade in trades:
        by_day.setdefault(trade.session_date, []).append(trade)
        by_symbol.setdefault(trade.symbol, []).append(trade.pnl_pct)
    print(f"Signal days: {len(by_day)}")
    print(f"Max signals on one day: {max(len(day_trades) for day_trades in by_day.values())}")

    print("\nTop symbols by signal count")
    symbol_rows = sorted(
        ((symbol, len(values), sum(values) / len(values)) for symbol, values in by_symbol.items()),
        key=lambda row: row[1],
        reverse=True,
    )[:20]
    for symbol, count, avg in symbol_rows:
        print(f"{symbol:<6} count={count:>2} avg={avg:>6.2f}%")

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
    data_15m, data_5m, daily_data = load_full_available_data()
    regimes = bt.build_market_regimes(data_15m)

    all_trades = []
    for symbol, intraday_15m in data_15m.items():
        all_trades.extend(backtest_symbol(symbol, intraday_15m, data_5m[symbol], daily_data[symbol], regimes))

    net_trades = apply_costs_to_trades(all_trades)

    print("\nExperiment: reversal pullback v15 volatility-filtered full universe (with costs)")
    print(
        f"Universe: all local symbols with 1D + 15m + 5m data; volatility filter avg_daily_range >= {MIN_AVG_DAILY_RANGE_PCT:.2f}%"
    )
    print(
        f"15m setup: trend<= {MAX_DAILY_TREND_PCT:.2f}%, breakout_attempt=[{MIN_BREAKOUT_PCT:.2f}%, {MAX_BREAKOUT_PCT:.2f}%]"
    )
    print(
        f"5m pullback entry: pullback=[{MIN_PULLBACK_FROM_CONFIRMATION_PCT:.2f}%, {MAX_PULLBACK_FROM_CONFIRMATION_PCT:.2f}%], "
        f"close_strength<={MAX_5M_CLOSE_STRENGTH:.2f}, entry_risk<={MAX_5M_ENTRY_RISK_PCT:.2f}%"
    )
    print("Exit: shared trailing/session exit from backtest.py")
    print("Costs: default round-trip cost model applied")

    summarize_research(net_trades)
    df = export_trades(net_trades)
    analyze(df)


if __name__ == "__main__":
    main()
