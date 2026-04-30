# v12: adds entry filter (close_strength <= 0.60) + pseudo equity simulation
from __future__ import annotations

import pandas as pd

from src.data.load_market_data import load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades

# same setup as v11
MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -5.0
MIN_BREAKOUT_PCT = 0.50
MAX_BREAKOUT_PCT = 2.50

# ENTRY FILTER (key change)
MAX_5M_CLOSE_STRENGTH = 0.60

PULLBACK_LOOKAHEAD_5M_BARS = 12
MIN_PULLBACK_FROM_CONFIRMATION_PCT = 0.30
MAX_PULLBACK_FROM_CONFIRMATION_PCT = 2.50
MIN_5M_CLOSE_STRENGTH = 0.0
MAX_5M_ENTRY_RISK_PCT = 10.0
MAX_CLOSE_BELOW_OR_HIGH_PCT = 0.75


def find_entry(session_5m, setup):
    confirmation_time = pd.Timestamp(setup["confirmation_time"])
    after = session_5m[session_5m["date"] > confirmation_time].copy()
    after = after.sort_values("date").reset_index(drop=True)
    window = after.iloc[:PULLBACK_LOOKAHEAD_5M_BARS]

    for _, row in window.iterrows():
        close = float(row["close"])
        cs = bt.calculate_close_strength(row)
        if cs > MAX_5M_CLOSE_STRENGTH:
            continue

        return row["date"], cs

    return None, None


def pseudo_equity(trades):
    capital = 10000.0
    curve = []
    for t in trades:
        capital *= (1 + t.pnl_pct / 100.0)
        curve.append(capital)
    return capital, curve


def main():
    intraday_15m, daily_data = bt.load_all_data()
    regimes = bt.build_market_regimes(intraday_15m)

    trades = []
    for symbol, data_15m in intraday_15m.items():
        try:
            data_5m = load_intraday(symbol, interval="5m")
        except:
            continue

        for session_date, session_15m in data_15m.groupby("session_date"):
            session_15m = session_15m.sort_values("date").reset_index(drop=True)
            setup = bt.find_first_breakout(session_15m)
            if setup is None:
                continue

            session_5m = data_5m[data_5m["date"].dt.date == session_date]
            entry_time, cs = find_entry(session_5m, setup)
            if entry_time is None:
                continue

            trade = bt.simulate_exit(symbol, session_5m.reset_index(drop=True), 0, 0, cs, 0, 0, setup_type="v12")
            if trade:
                trades.append(trade)

    trades = apply_costs_to_trades(trades)

    print("\n=== V12 RESULTS ===")
    print(f"Trades: {len(trades)}")

    if trades:
        avg = sum(t.pnl_pct for t in trades) / len(trades)
        print(f"Avg PnL: {avg:.2f}%")

        final_cap, _ = pseudo_equity(trades)
        print(f"Initial: 10000")
        print(f"Final: {final_cap:.2f}")
        print(f"Return: {(final_cap/10000-1)*100:.2f}%")

    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
