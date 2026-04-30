"""Entry-only scanner for Reversal Pullback MTF strategy.

Purpose:
- separate ENTRY research from EXIT research
- scan all 1D + 15m + 5m + 1m data
- report how many raw entry opportunities exist before any exit/PnL logic
- make it easy to loosen/tighten entry assumptions without mixing in stop-loss,
  take-profit, trailing stop, or time exit effects

No orders are placed. No exits are simulated.

Run examples:
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan --preset loose
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan --preset balanced
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan --preset quality
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_daily, load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt


@dataclass(frozen=True)
class EntryScanPreset:
    name: str
    description: str

    enable_market_regime_filter: bool = False
    min_avg_daily_range_pct: float = 3.0
    min_daily_trend_pct: float = -999.0
    max_daily_trend_pct: float = 0.0

    opening_range_bars: int = 4
    min_breakout_pct: float = 0.30
    max_breakout_pct: float = 5.00
    require_next_15m_bar_positive: bool = False
    min_confirmation_close_strength: float = 0.00
    max_confirmation_close_strength: float = 1.00
    max_setup_entry_risk_pct: float = 15.0

    pullback_lookahead_5m_bars: int = 18
    min_pullback_from_confirmation_pct: float = 0.10
    max_pullback_from_confirmation_pct: float = 4.00
    max_5m_close_strength: float = 1.00
    max_5m_entry_risk_pct: float = 15.0
    max_5m_close_below_or_high_pct: float = 2.00

    trigger_lookahead_1m_bars: int = 12
    min_1m_close_strength: float = 0.50
    max_1m_close_strength: float = 1.00
    require_1m_close_above_prev_close: bool = True
    require_1m_close_above_prev_high: bool = False
    max_1m_entry_risk_pct: float = 15.0
    max_1m_close_below_or_high_pct: float = 2.00


PRESETS: dict[str, EntryScanPreset] = {
    "loose": EntryScanPreset(
        name="loose",
        description="very permissive entry scan to estimate maximum opportunity count",
    ),
    "balanced": EntryScanPreset(
        name="balanced",
        description="moderate entry scan; target roughly one opportunity per active session",
        min_avg_daily_range_pct=4.0,
        max_daily_trend_pct=-3.0,
        min_breakout_pct=0.60,
        max_breakout_pct=3.00,
        min_confirmation_close_strength=0.20,
        max_confirmation_close_strength=1.00,
        max_setup_entry_risk_pct=12.0,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.85,
        max_5m_entry_risk_pct=10.0,
        max_5m_close_below_or_high_pct=1.00,
        min_1m_close_strength=0.65,
        max_1m_close_strength=0.95,
        max_1m_entry_risk_pct=10.0,
        max_1m_close_below_or_high_pct=1.00,
    ),
    "quality": EntryScanPreset(
        name="quality",
        description="stricter entry scan matching the current high-quality research direction",
        min_avg_daily_range_pct=4.5,
        max_daily_trend_pct=-3.0,
        min_breakout_pct=1.00,
        max_breakout_pct=2.20,
        min_confirmation_close_strength=0.30,
        max_confirmation_close_strength=1.00,
        max_setup_entry_risk_pct=10.0,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.75,
        max_5m_entry_risk_pct=8.0,
        max_5m_close_below_or_high_pct=0.75,
        min_1m_close_strength=0.75,
        max_1m_close_strength=0.90,
        max_1m_entry_risk_pct=8.0,
        max_1m_close_below_or_high_pct=0.75,
    ),
}


@dataclass(frozen=True)
class EntryCandidate:
    symbol: str
    session_date: str
    entry_time: str
    entry_price: float
    daily_trend_pct: float
    avg_daily_range_pct: float
    breakout_pct_15m: float
    confirmation_close_strength_15m: float
    setup_entry_risk_pct_15m: float
    pullback_pct_5m: float
    pullback_close_strength_5m: float
    pullback_entry_risk_pct_5m: float
    entry_close_strength_1m: float
    entry_risk_pct_1m: float
    close_below_or_high_pct_1m: float


def prepare_daily_data(symbol: str) -> pd.DataFrame:
    daily = load_daily(symbol).copy()
    daily["session_date"] = daily["date"].dt.date
    daily["ma20"] = daily["close"].rolling(20).mean()
    daily["daily_trend_pct"] = (daily["close"] - daily["ma20"]) / daily["ma20"] * 100.0
    daily["daily_range_pct"] = (daily["high"] - daily["low"]) / daily["close"] * 100.0
    daily["avg_daily_range_pct"] = daily["daily_range_pct"].rolling(20).mean()
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


def calculate_return_pct(current_price: float, next_price: float) -> float | None:
    if current_price == 0:
        return None
    return (next_price - current_price) / current_price * 100.0


def calculate_next_bar_return(session: pd.DataFrame, position: int) -> float | None:
    if position + 1 >= len(session):
        return None
    return calculate_return_pct(float(session.iloc[position]["close"]), float(session.iloc[position + 1]["close"]))


def load_all_data():
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


def find_15m_setup(session_15m: pd.DataFrame, daily_trend_pct: float, preset: EntryScanPreset):
    if len(session_15m) <= preset.opening_range_bars + 1:
        return None, "too_few_15m_bars"
    if not (preset.min_daily_trend_pct <= daily_trend_pct <= preset.max_daily_trend_pct):
        return None, "daily_trend_rejected"

    opening = session_15m.iloc[: preset.opening_range_bars]
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())

    for breakout_position in range(preset.opening_range_bars, len(session_15m) - 1):
        breakout_row = session_15m.iloc[breakout_position]
        breakout_close = float(breakout_row["close"])
        if breakout_close <= or_high:
            continue

        breakout_pct = bt.calculate_breakout_pct(breakout_close, or_high)
        setup_entry_risk_pct = bt.calculate_entry_risk_pct(breakout_close, or_low)
        next_bar_return_pct = calculate_next_bar_return(session_15m, breakout_position)
        if next_bar_return_pct is None:
            return None, "no_next_15m_bar"

        confirmation_row = session_15m.iloc[breakout_position + 1]
        confirmation_close_strength = bt.calculate_close_strength(confirmation_row)

        if breakout_pct < preset.min_breakout_pct:
            return None, "breakout_too_small"
        if breakout_pct > preset.max_breakout_pct:
            return None, "breakout_too_large"
        if setup_entry_risk_pct > preset.max_setup_entry_risk_pct:
            return None, "setup_entry_risk_too_high"
        if preset.require_next_15m_bar_positive and next_bar_return_pct < 0:
            return None, "next_15m_bar_negative"
        if confirmation_close_strength < preset.min_confirmation_close_strength:
            return None, "confirmation_too_weak"
        if confirmation_close_strength > preset.max_confirmation_close_strength:
            return None, "confirmation_too_strong"

        return {
            "or_high": or_high,
            "or_low": or_low,
            "confirmation_time": confirmation_row["date"],
            "confirmation_close": float(confirmation_row["close"]),
            "breakout_pct_15m": breakout_pct,
            "confirmation_close_strength_15m": confirmation_close_strength,
            "setup_entry_risk_pct_15m": setup_entry_risk_pct,
        }, "passed_15m"

    return None, "no_15m_breakout"


def find_5m_pullback(session_5m: pd.DataFrame, setup: dict, preset: EntryScanPreset):
    confirmation_time = pd.Timestamp(setup["confirmation_time"])
    confirmation_close = float(setup["confirmation_close"])
    or_high = float(setup["or_high"])
    or_low = float(setup["or_low"])

    after_confirmation = session_5m[session_5m["date"] > confirmation_time].copy()
    if after_confirmation.empty:
        return None, "no_5m_after_confirmation"

    after_confirmation = after_confirmation.sort_values("date").reset_index(drop=True)
    window = after_confirmation.iloc[: preset.pullback_lookahead_5m_bars]

    for _, row in window.iterrows():
        close = float(row["close"])
        low = float(row["low"])
        if confirmation_close == 0:
            continue

        pullback_pct = (confirmation_close - low) / confirmation_close * 100.0
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        below_or_pct = distance_below_or_high_pct(close, or_high)

        if pullback_pct < preset.min_pullback_from_confirmation_pct:
            continue
        if pullback_pct > preset.max_pullback_from_confirmation_pct:
            return None, "pullback_too_deep"
        if close_strength > preset.max_5m_close_strength:
            return None, "pullback_close_strength_too_high"
        if entry_risk_pct > preset.max_5m_entry_risk_pct:
            return None, "pullback_entry_risk_too_high"
        if below_or_pct > preset.max_5m_close_below_or_high_pct:
            return None, "pullback_too_far_below_or"

        return {
            "pullback_time": row["date"],
            "pullback_pct_5m": pullback_pct,
            "pullback_close_strength_5m": close_strength,
            "pullback_entry_risk_pct_5m": entry_risk_pct,
            "or_high": or_high,
            "or_low": or_low,
        }, "passed_5m"

    return None, "no_valid_5m_pullback"


def find_1m_trigger(session_1m: pd.DataFrame, pullback: dict, preset: EntryScanPreset):
    pullback_time = pd.Timestamp(pullback["pullback_time"])
    or_high = float(pullback["or_high"])
    or_low = float(pullback["or_low"])

    after_pullback = session_1m[session_1m["date"] >= pullback_time].copy()
    if after_pullback.empty:
        return None, "no_1m_after_pullback"

    after_pullback = after_pullback.sort_values("date").reset_index(drop=True)
    window = after_pullback.iloc[: preset.trigger_lookahead_1m_bars].reset_index(drop=True)

    for idx in range(1, len(window)):
        prev = window.iloc[idx - 1]
        row = window.iloc[idx]
        close = float(row["close"])
        prev_close = float(prev["close"])
        prev_high = float(prev["high"])
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        below_or_pct = distance_below_or_high_pct(close, or_high)

        if close_strength < preset.min_1m_close_strength:
            continue
        if close_strength > preset.max_1m_close_strength:
            return None, "trigger_close_strength_too_high"
        if preset.require_1m_close_above_prev_close and close <= prev_close:
            return None, "trigger_not_above_prev_close"
        if preset.require_1m_close_above_prev_high and close <= prev_high:
            return None, "trigger_not_above_prev_high"
        if entry_risk_pct > preset.max_1m_entry_risk_pct:
            return None, "trigger_entry_risk_too_high"
        if below_or_pct > preset.max_1m_close_below_or_high_pct:
            return None, "trigger_too_far_below_or"

        return {
            "entry_time": row["date"],
            "entry_price": close,
            "entry_close_strength_1m": close_strength,
            "entry_risk_pct_1m": entry_risk_pct,
            "close_below_or_high_pct_1m": below_or_pct,
        }, "passed_1m"

    return None, "no_1m_trigger"


def scan_entries(preset: EntryScanPreset):
    data_15m, data_5m, data_1m, daily_data = load_all_data()
    regimes = bt.build_market_regimes(data_15m)

    counters = Counter()
    candidates: list[EntryCandidate] = []
    candidates_by_day = defaultdict(int)

    for symbol, d15 in data_15m.items():
        for session_date, session_15m in d15.groupby("session_date"):
            counters["symbol_days"] += 1

            regime = regimes.get(str(session_date))
            if preset.enable_market_regime_filter and (regime is None or not regime.tradable):
                counters["rejected_market_regime"] += 1
                continue
            counters["passed_market_regime"] += 1

            daily = daily_data[symbol]
            avg_daily_range_pct = get_daily_value_before_session(daily, session_date, "avg_daily_range_pct")
            if avg_daily_range_pct < preset.min_avg_daily_range_pct:
                counters["rejected_adr"] += 1
                continue
            counters["passed_adr"] += 1

            daily_trend_pct = get_daily_value_before_session(daily, session_date, "daily_trend_pct")
            session_15m = session_15m.sort_values("date").reset_index(drop=True)
            setup, reason_15m = find_15m_setup(session_15m, daily_trend_pct, preset)
            counters[reason_15m] += 1
            if setup is None:
                continue

            session_5m = data_5m[symbol][data_5m[symbol]["session_date"] == session_date].sort_values("date").reset_index(drop=True)
            pullback, reason_5m = find_5m_pullback(session_5m, setup, preset)
            counters[reason_5m] += 1
            if pullback is None:
                continue

            session_1m = data_1m[symbol][data_1m[symbol]["session_date"] == session_date].sort_values("date").reset_index(drop=True)
            trigger, reason_1m = find_1m_trigger(session_1m, pullback, preset)
            counters[reason_1m] += 1
            if trigger is None:
                continue

            candidate = EntryCandidate(
                symbol=symbol,
                session_date=str(session_date),
                entry_time=str(trigger["entry_time"]),
                entry_price=float(trigger["entry_price"]),
                daily_trend_pct=daily_trend_pct,
                avg_daily_range_pct=avg_daily_range_pct,
                breakout_pct_15m=float(setup["breakout_pct_15m"]),
                confirmation_close_strength_15m=float(setup["confirmation_close_strength_15m"]),
                setup_entry_risk_pct_15m=float(setup["setup_entry_risk_pct_15m"]),
                pullback_pct_5m=float(pullback["pullback_pct_5m"]),
                pullback_close_strength_5m=float(pullback["pullback_close_strength_5m"]),
                pullback_entry_risk_pct_5m=float(pullback["pullback_entry_risk_pct_5m"]),
                entry_close_strength_1m=float(trigger["entry_close_strength_1m"]),
                entry_risk_pct_1m=float(trigger["entry_risk_pct_1m"]),
                close_below_or_high_pct_1m=float(trigger["close_below_or_high_pct_1m"]),
            )
            candidates.append(candidate)
            candidates_by_day[str(session_date)] += 1

    return counters, candidates, candidates_by_day


def export_candidates(candidates: list[EntryCandidate]) -> pd.DataFrame:
    path = Path("data/backtests")
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(candidate) for candidate in candidates])
    df.to_csv(path / "reversal_pullback_entry_candidates.csv", index=False)
    return df


def print_summary(preset: EntryScanPreset, counters: Counter, candidates: list[EntryCandidate], candidates_by_day: dict[str, int]) -> None:
    print(f"\nEntry scan preset: {preset.name}")
    print(preset.description)
    print("\nEntry filters:")
    print(f"- market_regime_filter={preset.enable_market_regime_filter}")
    print(f"- ADR >= {preset.min_avg_daily_range_pct:.2f}%")
    print(f"- daily_trend: {preset.min_daily_trend_pct:.2f}% to {preset.max_daily_trend_pct:.2f}%")
    print(f"- 15m breakout: {preset.min_breakout_pct:.2f}% to {preset.max_breakout_pct:.2f}%")
    print(f"- 5m pullback: {preset.min_pullback_from_confirmation_pct:.2f}% to {preset.max_pullback_from_confirmation_pct:.2f}%")
    print(f"- 1m close_strength: {preset.min_1m_close_strength:.2f} to {preset.max_1m_close_strength:.2f}")

    print("\nFunnel counts:")
    for name, count in counters.most_common():
        print(f"{name:<36} {count:>6}")

    active_days = len(candidates_by_day)
    max_candidates_one_day = max(candidates_by_day.values()) if candidates_by_day else 0
    print("\nEntry candidates:")
    print(f"Candidates: {len(candidates)}")
    print(f"Active days: {active_days}")
    print(f"Max candidates on one day: {max_candidates_one_day}")
    if candidates:
        print(f"Avg candidates per active day: {len(candidates) / active_days:.2f}")

    by_symbol = Counter(candidate.symbol for candidate in candidates)
    print("\nTop symbols by entry count:")
    for symbol, count in by_symbol.most_common(30):
        print(f"{symbol:<6} {count:>3}")

    print("\nRecent entry candidates:")
    print("Date | Symbol | Entry | Trend % | ADR % | Break % | 5m PB % | 1m CS | Risk %")
    print("--------------------------------------------------------------------------------")
    for candidate in candidates[-40:]:
        print(
            f"{candidate.session_date} | {candidate.symbol:<6} | {candidate.entry_price:>7.2f} | "
            f"{candidate.daily_trend_pct:>7.2f} | {candidate.avg_daily_range_pct:>5.2f} | "
            f"{candidate.breakout_pct_15m:>7.2f} | {candidate.pullback_pct_5m:>7.2f} | "
            f"{candidate.entry_close_strength_1m:>5.2f} | {candidate.entry_risk_pct_1m:>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    counters, candidates, candidates_by_day = scan_entries(preset)
    export_candidates(candidates)
    print_summary(preset, counters, candidates, candidates_by_day)


if __name__ == "__main__":
    main()
