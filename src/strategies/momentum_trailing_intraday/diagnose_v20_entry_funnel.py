"""Diagnose why v20/v22 produce so few entries.

This script does not simulate exits and does not place orders.
It counts how many symbol-days survive each stage of the MTF entry funnel:

1. data/session availability
2. market regime filter
3. ADR volatility filter
4. daily trend filter
5. 15m breakout attempt
6. 15m confirmation filters
7. 5m pullback candidate
8. 1m trigger candidate
9. v20 close-strength upper bound

Run:
python -m src.strategies.momentum_trailing_intraday.diagnose_v20_entry_funnel
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_daily, load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v17_1m_entry as v17

# v20/v22 entry settings.
MIN_BREAKOUT_PCT = 1.00
MAX_BREAKOUT_PCT = 1.80
MIN_1M_CLOSE_STRENGTH = 0.80
MAX_1M_CLOSE_STRENGTH = 0.90
MIN_AVG_DAILY_RANGE_PCT = 5.0


def prepare_daily_data(symbol: str) -> pd.DataFrame:
    daily = load_daily(symbol).copy()
    daily["session_date"] = daily["date"].dt.date
    daily["ma20"] = daily["close"].rolling(20).mean()
    daily["daily_trend_pct"] = (daily["close"] - daily["ma20"]) / daily["ma20"] * 100.0
    daily["daily_range_pct"] = (daily["high"] - daily["low"]) / daily["close"] * 100.0
    daily["avg_daily_range_pct"] = daily["daily_range_pct"].rolling(v17.VOLATILITY_LOOKBACK_DAYS).mean()
    return daily


def get_daily_value_before_session(daily: pd.DataFrame, session_date, column: str) -> float:
    history = daily[daily["session_date"] < session_date].dropna(subset=[column])
    if history.empty:
        return 0.0
    return float(history.iloc[-1][column])


def distance_below_or_high_pct(close: float, opening_range_high: float) -> float:
    if opening_range_high == 0 or close >= opening_range_high:
        return 0.0
    return (opening_range_high - close) / opening_range_high * 100.0


def load_data():
    data_15m = {}
    data_5m = {}
    data_1m = {}
    daily_data = {}

    for spec in UNIVERSE:
        symbol = spec.symbol
        try:
            d15 = load_intraday(symbol, interval="15m").copy()
            d5 = load_intraday(symbol, interval="5m").copy()
            d1 = load_intraday(symbol, interval="1m").copy()
            daily = prepare_daily_data(symbol)
        except Exception:
            continue

        d15["session_date"] = d15["date"].dt.date
        d5["session_date"] = d5["date"].dt.date
        d1["session_date"] = d1["date"].dt.date

        data_15m[symbol] = d15
        data_5m[symbol] = d5
        data_1m[symbol] = d1
        daily_data[symbol] = daily

    return data_15m, data_5m, data_1m, daily_data


def classify_15m_setup(session_15m: pd.DataFrame, daily_trend_pct: float):
    if len(session_15m) <= bt.OPENING_RANGE_BARS + 1:
        return "too_few_15m_bars", None
    if not (v17.MIN_DAILY_TREND_PCT <= daily_trend_pct <= v17.MAX_DAILY_TREND_PCT):
        return "daily_trend_rejected", None

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
        next_bar_return_pct = v17.calculate_next_bar_return(session_15m, breakout_position)
        if next_bar_return_pct is None:
            return "no_next_15m_bar", None

        confirmation_position = breakout_position + 1
        confirmation_row = session_15m.iloc[confirmation_position]
        confirmation_close_strength = bt.calculate_close_strength(confirmation_row)

        if breakout_pct < MIN_BREAKOUT_PCT:
            return "breakout_too_small", None
        if breakout_pct > MAX_BREAKOUT_PCT:
            return "breakout_too_large", None
        if setup_entry_risk_pct > v17.MAX_SETUP_ENTRY_RISK_PCT:
            return "setup_entry_risk_too_high", None
        if next_bar_return_pct < v17.MIN_NEXT_BAR_RETURN_PCT:
            return "next_15m_bar_negative", None
        if confirmation_close_strength < v17.MIN_CONFIRMATION_CLOSE_STRENGTH:
            return "confirmation_too_weak", None
        if confirmation_close_strength > v17.MAX_CONFIRMATION_CLOSE_STRENGTH:
            return "confirmation_too_strong", None

        return "passed_15m", {
            "or_high": or_high,
            "or_low": or_low,
            "confirmation_time": confirmation_row["date"],
            "confirmation_close": float(confirmation_row["close"]),
            "daily_trend_pct": daily_trend_pct,
        }

    return "no_15m_breakout", None


def classify_5m_pullback(session_5m: pd.DataFrame, setup: dict):
    confirmation_time = pd.Timestamp(setup["confirmation_time"])
    confirmation_close = float(setup["confirmation_close"])
    or_high = float(setup["or_high"])
    or_low = float(setup["or_low"])

    after_confirmation = session_5m[session_5m["date"] > confirmation_time].copy()
    if after_confirmation.empty:
        return "no_5m_after_confirmation", None

    after_confirmation = after_confirmation.sort_values("date").reset_index(drop=True)
    window = after_confirmation.iloc[: v17.PULLBACK_LOOKAHEAD_5M_BARS]

    saw_pullback = False
    for _, row in window.iterrows():
        close = float(row["close"])
        low = float(row["low"])
        if confirmation_close == 0:
            continue

        pullback_pct = (confirmation_close - low) / confirmation_close * 100.0
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        below_or_pct = distance_below_or_high_pct(close, or_high)

        if pullback_pct >= v17.MIN_PULLBACK_FROM_CONFIRMATION_PCT:
            saw_pullback = True
        if pullback_pct < v17.MIN_PULLBACK_FROM_CONFIRMATION_PCT:
            continue
        if pullback_pct > v17.MAX_PULLBACK_FROM_CONFIRMATION_PCT:
            return "pullback_too_deep", None
        if close_strength < v17.MIN_5M_CLOSE_STRENGTH:
            return "pullback_close_strength_too_low", None
        if close_strength > v17.MAX_5M_CLOSE_STRENGTH:
            return "pullback_close_strength_too_high", None
        if entry_risk_pct > v17.MAX_5M_ENTRY_RISK_PCT:
            return "pullback_entry_risk_too_high", None
        if below_or_pct > v17.MAX_CLOSE_BELOW_OR_HIGH_PCT:
            return "pullback_too_far_below_or", None

        return "passed_5m", {"pullback_time": row["date"], "or_high": or_high, "or_low": or_low}

    return "no_min_pullback" if not saw_pullback else "no_valid_5m_pullback", None


def classify_1m_trigger(session_1m: pd.DataFrame, pullback: dict):
    pullback_time = pd.Timestamp(pullback["pullback_time"])
    or_high = float(pullback["or_high"])
    or_low = float(pullback["or_low"])

    after_pullback = session_1m[session_1m["date"] >= pullback_time].copy()
    if after_pullback.empty:
        return "no_1m_after_pullback"

    after_pullback = after_pullback.sort_values("date").reset_index(drop=True)
    window = after_pullback.iloc[: v17.TRIGGER_LOOKAHEAD_1M_BARS].reset_index(drop=True)

    for idx in range(1, len(window)):
        prev = window.iloc[idx - 1]
        row = window.iloc[idx]

        close = float(row["close"])
        prev_close = float(prev["close"])
        prev_high = float(prev["high"])
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        below_or_pct = distance_below_or_high_pct(close, or_high)

        if close_strength < MIN_1M_CLOSE_STRENGTH:
            continue
        if close_strength > MAX_1M_CLOSE_STRENGTH:
            return "trigger_close_strength_too_high"
        if v17.REQUIRE_1M_CLOSE_ABOVE_PREV_CLOSE and close <= prev_close:
            return "trigger_not_above_prev_close"
        if v17.REQUIRE_1M_CLOSE_ABOVE_PREV_HIGH and close <= prev_high:
            return "trigger_not_above_prev_high"
        if entry_risk_pct > v17.MAX_1M_ENTRY_RISK_PCT:
            return "trigger_entry_risk_too_high"
        if below_or_pct > v17.MAX_1M_CLOSE_BELOW_OR_HIGH_PCT:
            return "trigger_too_far_below_or"

        return "passed_1m"

    return "no_1m_trigger"


def main():
    data_15m, data_5m, data_1m, daily_data = load_data()
    regimes = bt.build_market_regimes(data_15m)

    counters = Counter()
    examples = {}

    for symbol, d15 in data_15m.items():
        for session_date, session_15m in d15.groupby("session_date"):
            key = f"{session_date} {symbol}"
            counters["symbol_days"] += 1

            regime = regimes.get(str(session_date))
            if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
                counters["rejected_market_regime"] += 1
                examples.setdefault("rejected_market_regime", key)
                continue
            counters["passed_market_regime"] += 1

            daily = daily_data[symbol]
            adr = get_daily_value_before_session(daily, session_date, "avg_daily_range_pct")
            if adr < MIN_AVG_DAILY_RANGE_PCT:
                counters["rejected_adr"] += 1
                examples.setdefault("rejected_adr", key)
                continue
            counters["passed_adr"] += 1

            daily_trend = get_daily_value_before_session(daily, session_date, "daily_trend_pct")
            session_15m = session_15m.sort_values("date").reset_index(drop=True)
            reason_15m, setup = classify_15m_setup(session_15m, daily_trend)
            counters[reason_15m] += 1
            examples.setdefault(reason_15m, key)
            if setup is None:
                continue

            session_5m = data_5m[symbol][data_5m[symbol]["session_date"] == session_date].sort_values("date").reset_index(drop=True)
            reason_5m, pullback = classify_5m_pullback(session_5m, setup)
            counters[reason_5m] += 1
            examples.setdefault(reason_5m, key)
            if pullback is None:
                continue

            session_1m = data_1m[symbol][data_1m[symbol]["session_date"] == session_date].sort_values("date").reset_index(drop=True)
            reason_1m = classify_1m_trigger(session_1m, pullback)
            counters[reason_1m] += 1
            examples.setdefault(reason_1m, key)

    print("\nV20/V22 entry funnel diagnostics")
    print(f"Universe loaded: {len(data_15m)} symbols")
    print("Settings:")
    print(f"- breakout: {MIN_BREAKOUT_PCT:.2f}% - {MAX_BREAKOUT_PCT:.2f}%")
    print(f"- 1m close_strength: {MIN_1M_CLOSE_STRENGTH:.2f} - {MAX_1M_CLOSE_STRENGTH:.2f}")
    print(f"- ADR >= {MIN_AVG_DAILY_RANGE_PCT:.2f}%")
    print("\nCounts:")
    for name, count in counters.most_common():
        print(f"{name:<36} {count:>6}  example={examples.get(name, '-')}")


if __name__ == "__main__":
    main()
