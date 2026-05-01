"""v29 simple entry scanner.

Goal:
- debug why the full MTF 15m -> 5m -> 1m funnel produces too few entries
- scan for simpler reversal entries using daily context + 5m weakness + 1m reversal
- no exit simulation, no PnL, no orders

This scanner intentionally removes the hard dependency on a prior 15m breakout
confirmation. It answers only one question:

    How many plausible entry candidates exist if entry is driven mainly by
    5m pullback/weakness and 1m reversal trigger?

Run:
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple --preset balanced
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple --preset loose
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple --preset quality
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_daily, load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt


@dataclass(frozen=True)
class SimpleEntryPreset:
    name: str
    description: str

    min_avg_daily_range_pct: float = 4.0
    min_daily_trend_pct: float = -35.0
    max_daily_trend_pct: float = -2.0

    opening_range_15m_bars: int = 4
    min_distance_below_opening_high_pct: float = 0.10
    max_distance_below_opening_high_pct: float = 3.00

    min_5m_pullback_from_recent_high_pct: float = 0.30
    max_5m_pullback_from_recent_high_pct: float = 4.00
    max_5m_close_strength: float = 0.80
    max_5m_entry_risk_pct: float = 12.0
    lookback_5m_bars: int = 6

    min_1m_close_strength: float = 0.65
    max_1m_close_strength: float = 0.95
    require_1m_close_above_prev_close: bool = True
    require_1m_close_above_prev_high: bool = False
    max_1m_entry_risk_pct: float = 12.0

    max_entries_per_symbol_day: int = 1


PRESETS: dict[str, SimpleEntryPreset] = {
    "loose": SimpleEntryPreset(
        name="loose",
        description="very permissive 5m/1m reversal scanner",
        min_avg_daily_range_pct=3.0,
        min_daily_trend_pct=-999.0,
        max_daily_trend_pct=0.0,
        min_distance_below_opening_high_pct=0.00,
        max_distance_below_opening_high_pct=5.00,
        min_5m_pullback_from_recent_high_pct=0.10,
        max_5m_pullback_from_recent_high_pct=5.00,
        max_5m_close_strength=1.00,
        max_5m_entry_risk_pct=15.0,
        min_1m_close_strength=0.50,
        max_1m_close_strength=1.00,
        max_1m_entry_risk_pct=15.0,
    ),
    "balanced": SimpleEntryPreset(
        name="balanced",
        description="balanced 5m/1m reversal scanner, target approx 1+ entry/day",
    ),
    "quality": SimpleEntryPreset(
        name="quality",
        description="stricter simple scanner for cleaner candidates",
        min_avg_daily_range_pct=4.5,
        min_daily_trend_pct=-30.0,
        max_daily_trend_pct=-3.0,
        min_distance_below_opening_high_pct=0.25,
        max_distance_below_opening_high_pct=2.50,
        min_5m_pullback_from_recent_high_pct=0.40,
        max_5m_pullback_from_recent_high_pct=3.00,
        max_5m_close_strength=0.75,
        max_5m_entry_risk_pct=10.0,
        min_1m_close_strength=0.75,
        max_1m_close_strength=0.90,
        max_1m_entry_risk_pct=10.0,
    ),
}


@dataclass(frozen=True)
class SimpleEntryCandidate:
    symbol: str
    session_date: str
    entry_time: str
    entry_price: float
    daily_trend_pct: float
    avg_daily_range_pct: float
    distance_below_or_high_pct: float
    pullback_from_recent_5m_high_pct: float
    close_strength_5m: float
    entry_close_strength_1m: float
    entry_risk_pct_1m: float


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


def close_strength(row) -> float:
    return bt.calculate_close_strength(row)


def entry_risk_pct(entry_price: float, reference_low: float) -> float:
    return bt.calculate_entry_risk_pct(entry_price, reference_low)


def distance_below_or_high_pct(close: float, opening_high: float) -> float:
    if opening_high == 0 or close >= opening_high:
        return 0.0
    return (opening_high - close) / opening_high * 100.0


def load_all_data():
    data_15m, data_5m, data_1m, daily_data = {}, {}, {}, {}
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


def find_opening_context(session_15m: pd.DataFrame, preset: SimpleEntryPreset):
    if len(session_15m) <= preset.opening_range_15m_bars:
        return None
    opening = session_15m.iloc[: preset.opening_range_15m_bars]
    return {
        "or_high": float(opening["high"].max()),
        "or_low": float(opening["low"].min()),
    }


def find_simple_entry_for_session(
    symbol: str,
    session_date,
    session_15m: pd.DataFrame,
    session_5m: pd.DataFrame,
    session_1m: pd.DataFrame,
    daily_trend_pct: float,
    avg_daily_range_pct: float,
    preset: SimpleEntryPreset,
    counters: Counter,
):
    context = find_opening_context(session_15m, preset)
    if context is None:
        counters["too_few_15m_bars"] += 1
        return []

    or_high = context["or_high"]
    or_low = context["or_low"]

    if session_5m.empty or session_1m.empty:
        counters["missing_intraday_session"] += 1
        return []

    candidates: list[SimpleEntryCandidate] = []
    session_5m = session_5m.sort_values("date").reset_index(drop=True)
    session_1m = session_1m.sort_values("date").reset_index(drop=True)

    for idx in range(preset.lookback_5m_bars, len(session_5m)):
        row_5m = session_5m.iloc[idx]
        window_5m = session_5m.iloc[idx - preset.lookback_5m_bars : idx + 1]
        recent_high = float(window_5m["high"].max())
        recent_low = float(window_5m["low"].min())
        close_5m = float(row_5m["close"])
        low_5m = float(row_5m["low"])

        if recent_high == 0:
            continue

        pullback_pct = (recent_high - low_5m) / recent_high * 100.0
        cs_5m = close_strength(row_5m)
        below_or_pct = distance_below_or_high_pct(close_5m, or_high)
        risk_5m = entry_risk_pct(close_5m, min(or_low, recent_low))

        if pullback_pct < preset.min_5m_pullback_from_recent_high_pct:
            continue
        if pullback_pct > preset.max_5m_pullback_from_recent_high_pct:
            counters["pullback_too_deep"] += 1
            continue
        if cs_5m > preset.max_5m_close_strength:
            counters["5m_close_strength_too_high"] += 1
            continue
        if below_or_pct < preset.min_distance_below_opening_high_pct:
            counters["too_close_or_above_opening_high"] += 1
            continue
        if below_or_pct > preset.max_distance_below_opening_high_pct:
            counters["too_far_below_opening_high"] += 1
            continue
        if risk_5m > preset.max_5m_entry_risk_pct:
            counters["5m_entry_risk_too_high"] += 1
            continue

        trigger_window = session_1m[session_1m["date"] >= row_5m["date"]].head(12).reset_index(drop=True)
        if len(trigger_window) < 2:
            counters["no_1m_after_5m"] += 1
            continue

        for j in range(1, len(trigger_window)):
            prev = trigger_window.iloc[j - 1]
            row_1m = trigger_window.iloc[j]
            close_1m = float(row_1m["close"])
            prev_close = float(prev["close"])
            prev_high = float(prev["high"])
            cs_1m = close_strength(row_1m)
            risk_1m = entry_risk_pct(close_1m, min(or_low, recent_low))

            if cs_1m < preset.min_1m_close_strength:
                continue
            if cs_1m > preset.max_1m_close_strength:
                counters["1m_close_strength_too_high"] += 1
                continue
            if preset.require_1m_close_above_prev_close and close_1m <= prev_close:
                counters["1m_not_above_prev_close"] += 1
                continue
            if preset.require_1m_close_above_prev_high and close_1m <= prev_high:
                counters["1m_not_above_prev_high"] += 1
                continue
            if risk_1m > preset.max_1m_entry_risk_pct:
                counters["1m_entry_risk_too_high"] += 1
                continue

            candidates.append(
                SimpleEntryCandidate(
                    symbol=symbol,
                    session_date=str(session_date),
                    entry_time=str(row_1m["date"]),
                    entry_price=close_1m,
                    daily_trend_pct=daily_trend_pct,
                    avg_daily_range_pct=avg_daily_range_pct,
                    distance_below_or_high_pct=below_or_pct,
                    pullback_from_recent_5m_high_pct=pullback_pct,
                    close_strength_5m=cs_5m,
                    entry_close_strength_1m=cs_1m,
                    entry_risk_pct_1m=risk_1m,
                )
            )
            counters["passed_1m"] += 1
            if len(candidates) >= preset.max_entries_per_symbol_day:
                return candidates
            break

    if not candidates:
        counters["no_simple_entry"] += 1
    return candidates


def scan_entries(preset: SimpleEntryPreset):
    data_15m, data_5m, data_1m, daily_data = load_all_data()
    counters = Counter()
    candidates: list[SimpleEntryCandidate] = []
    candidates_by_day = defaultdict(int)

    for symbol, d15 in data_15m.items():
        for session_date, session_15m in d15.groupby("session_date"):
            counters["symbol_days"] += 1
            daily = daily_data[symbol]
            avg_daily_range_pct = get_daily_value_before_session(daily, session_date, "avg_daily_range_pct")
            if avg_daily_range_pct < preset.min_avg_daily_range_pct:
                counters["rejected_adr"] += 1
                continue
            counters["passed_adr"] += 1

            daily_trend_pct = get_daily_value_before_session(daily, session_date, "daily_trend_pct")
            if not (preset.min_daily_trend_pct <= daily_trend_pct <= preset.max_daily_trend_pct):
                counters["daily_trend_rejected"] += 1
                continue
            counters["passed_daily_trend"] += 1

            session_15m = session_15m.sort_values("date").reset_index(drop=True)
            session_5m = data_5m[symbol][data_5m[symbol]["session_date"] == session_date]
            session_1m = data_1m[symbol][data_1m[symbol]["session_date"] == session_date]

            found = find_simple_entry_for_session(
                symbol,
                session_date,
                session_15m,
                session_5m,
                session_1m,
                daily_trend_pct,
                avg_daily_range_pct,
                preset,
                counters,
            )
            for candidate in found:
                candidates.append(candidate)
                candidates_by_day[candidate.session_date] += 1

    return counters, candidates, candidates_by_day


def export_candidates(candidates: list[SimpleEntryCandidate]) -> pd.DataFrame:
    path = Path("data/backtests")
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(candidate) for candidate in candidates])
    df.to_csv(path / "reversal_pullback_v29_simple_entry_candidates.csv", index=False)
    return df


def print_summary(preset: SimpleEntryPreset, counters: Counter, candidates: list[SimpleEntryCandidate], candidates_by_day):
    print(f"\nSimple entry scan preset: {preset.name}")
    print(preset.description)
    print("\nEntry filters:")
    print(f"- ADR >= {preset.min_avg_daily_range_pct:.2f}%")
    print(f"- daily_trend: {preset.min_daily_trend_pct:.2f}% to {preset.max_daily_trend_pct:.2f}%")
    print(f"- distance below OR high: {preset.min_distance_below_opening_high_pct:.2f}% to {preset.max_distance_below_opening_high_pct:.2f}%")
    print(f"- 5m pullback from recent high: {preset.min_5m_pullback_from_recent_high_pct:.2f}% to {preset.max_5m_pullback_from_recent_high_pct:.2f}%")
    print(f"- 5m close_strength <= {preset.max_5m_close_strength:.2f}")
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

    print("\nRecent simple entry candidates:")
    print("Date | Symbol | Entry | Trend % | ADR % | BelowOR % | 5mPB % | 5mCS | 1mCS | Risk %")
    print("------------------------------------------------------------------------------------------")
    for candidate in candidates[-40:]:
        print(
            f"{candidate.session_date} | {candidate.symbol:<6} | {candidate.entry_price:>7.2f} | "
            f"{candidate.daily_trend_pct:>7.2f} | {candidate.avg_daily_range_pct:>5.2f} | "
            f"{candidate.distance_below_or_high_pct:>8.2f} | {candidate.pullback_from_recent_5m_high_pct:>6.2f} | "
            f"{candidate.close_strength_5m:>5.2f} | {candidate.entry_close_strength_1m:>5.2f} | {candidate.entry_risk_pct_1m:>6.2f}"
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
