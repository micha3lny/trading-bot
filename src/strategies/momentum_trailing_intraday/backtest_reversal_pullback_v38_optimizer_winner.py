"""v38: data-driven optimizer winner.

Built from reversal_pullback_entry_optimizer top result on quality preset.

Strict profile = exact top optimizer segment:
- daily_trend: -7% to -3%
- pullback proxy: 1.2% to 3.0%
- 1m close_strength: 0.80 to 0.90
- entry_risk: 4% to 8%
- pre_15m_low_to_entry <= 2.0%
- 5m close_strength <= 0.75
- distance below OR high: 0% to 5%

Scaled profile = looser validation sample:
- daily_trend: -10% to -3%
- pullback proxy: 1.0% to 3.0%
- 1m close_strength: 0.75 to 0.90
- entry_risk: 4% to 8%
- pre_15m_low_to_entry <= 2.0%
- 5m close_strength <= 0.80

Run:
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality --profile scaled
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v17_1m_entry import summarize_research
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    NOISY_SYMBOLS,
    build_candidate_session_map,
    find_entry_position,
    simulate_v22_exit,
)
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_audit import pre_entry_stats
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)


@dataclass(frozen=True)
class V38Profile:
    name: str
    description: str
    trend_min: float
    trend_max: float
    pullback_min: float
    pullback_max: float
    cs_min: float
    cs_max: float
    risk_min: float
    risk_max: float
    pre15_bounce_max: float
    cs5_max: float
    below_or_min: float
    below_or_max: float
    excluded_symbols: frozenset[str]


PROFILES = {
    "strict": V38Profile(
        name="strict",
        description="exact top optimizer result, high edge but small sample",
        trend_min=-7.0,
        trend_max=-3.0,
        pullback_min=1.2,
        pullback_max=3.0,
        cs_min=0.80,
        cs_max=0.90,
        risk_min=4.0,
        risk_max=8.0,
        pre15_bounce_max=2.0,
        cs5_max=0.75,
        below_or_min=0.0,
        below_or_max=5.0,
        excluded_symbols=frozenset(NOISY_SYMBOLS),
    ),
    "scaled": V38Profile(
        name="scaled",
        description="looser validation profile for bigger sample",
        trend_min=-10.0,
        trend_max=-3.0,
        pullback_min=1.0,
        pullback_max=3.0,
        cs_min=0.75,
        cs_max=0.90,
        risk_min=4.0,
        risk_max=8.0,
        pre15_bounce_max=2.0,
        cs5_max=0.80,
        below_or_min=0.0,
        below_or_max=5.0,
        excluded_symbols=frozenset(NOISY_SYMBOLS),
    ),
    "sample20": V38Profile(
        name="sample20",
        description="middle profile targeting more than strict without opening full noise",
        trend_min=-10.0,
        trend_max=-3.0,
        pullback_min=1.2,
        pullback_max=3.0,
        cs_min=0.75,
        cs_max=0.90,
        risk_min=4.0,
        risk_max=8.0,
        pre15_bounce_max=2.0,
        cs5_max=0.75,
        below_or_min=0.0,
        below_or_max=5.0,
        excluded_symbols=frozenset(NOISY_SYMBOLS),
    ),
}


def candidate_pre15_bounce(candidate: SimpleEntryCandidate, data_1m: dict[str, pd.DataFrame]) -> float | None:
    if candidate.symbol not in data_1m:
        return None
    session = (
        data_1m[candidate.symbol][data_1m[candidate.symbol]["session_date"].astype(str) == candidate.session_date]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if session.empty:
        return None
    entry_idx = find_entry_position(session, candidate.entry_time)
    if entry_idx is None:
        return None
    return float(pre_entry_stats(session, entry_idx, 15)["pre_15m_low_to_entry_pct"])


def passes_profile(candidate: SimpleEntryCandidate, profile: V38Profile, data_1m: dict[str, pd.DataFrame]) -> bool:
    if candidate.symbol in profile.excluded_symbols:
        return False
    if not (profile.trend_min <= candidate.daily_trend_pct <= profile.trend_max):
        return False
    if not (profile.pullback_min <= candidate.pullback_from_recent_5m_high_pct <= profile.pullback_max):
        return False
    if not (profile.cs_min <= candidate.entry_close_strength_1m <= profile.cs_max):
        return False
    if not (profile.risk_min <= candidate.entry_risk_pct_1m <= profile.risk_max):
        return False
    if candidate.close_strength_5m > profile.cs5_max:
        return False
    if not (profile.below_or_min <= candidate.distance_below_or_high_pct <= profile.below_or_max):
        return False
    pre15_bounce = candidate_pre15_bounce(candidate, data_1m)
    if pre15_bounce is None:
        return False
    if pre15_bounce > profile.pre15_bounce_max:
        return False
    return True


def run_backtest(preset_name: str, profile: V38Profile):
    _counters, raw_candidates, _candidates_by_day = scan_entries(PRESETS[preset_name])
    _data_15m, _data_5m, data_1m, _daily_data = load_all_data()

    counters = Counter()
    candidates: list[SimpleEntryCandidate] = []
    for candidate in raw_candidates:
        counters["raw_candidates"] += 1
        if passes_profile(candidate, profile, data_1m):
            counters["passed_v38_profile"] += 1
            candidates.append(candidate)
        else:
            counters["rejected_v38_profile"] += 1

    candidates_by_symbol_day = build_candidate_session_map(candidates)
    trades = []
    for (symbol, session_date), day_candidates in candidates_by_symbol_day.items():
        if symbol not in data_1m:
            continue
        session_1m = (
            data_1m[symbol][data_1m[symbol]["session_date"].astype(str) == session_date]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if session_1m.empty:
            continue
        for candidate in day_candidates:
            trade = simulate_v22_exit(symbol, session_1m, candidate, False, False, False)
            if trade is not None:
                trades.append(trade)

    return counters, candidates, apply_costs_to_trades(trades)


def summarize_inputs(counters: Counter, candidates: list[SimpleEntryCandidate], profile: V38Profile) -> None:
    print("\nFunnel counts:")
    for name, count in counters.most_common():
        print(f"{name:<28} {count:>6}")

    print("\nEntry candidate pool")
    print(f"Candidates after filters: {len(candidates)}")
    active_days = len(set(candidate.session_date for candidate in candidates))
    print(f"Active days: {active_days}")
    if candidates and active_days:
        by_day = Counter(candidate.session_date for candidate in candidates)
        print(f"Max candidates on one day: {max(by_day.values())}")
        print(f"Avg candidates per active day: {len(candidates) / active_days:.2f}")

    print("v38 profile:")
    print(f"- profile: {profile.name} - {profile.description}")
    print(f"- daily_trend: {profile.trend_min:.2f}% to {profile.trend_max:.2f}%")
    print(f"- pullback proxy: {profile.pullback_min:.2f}% to {profile.pullback_max:.2f}%")
    print(f"- 1m close_strength: {profile.cs_min:.2f} to {profile.cs_max:.2f}")
    print(f"- entry_risk: {profile.risk_min:.2f}% to {profile.risk_max:.2f}%")
    print(f"- pre_15m_low_to_entry <= {profile.pre15_bounce_max:.2f}%")
    print(f"- 5m close_strength <= {profile.cs5_max:.2f}")
    print(f"- distance below OR high: {profile.below_or_min:.2f}% to {profile.below_or_max:.2f}%")
    print(f"- excluded symbols: {sorted(profile.excluded_symbols)}")

    print("\nTop symbols by candidate count")
    for symbol, count in Counter(candidate.symbol for candidate in candidates).most_common(30):
        print(f"{symbol:<6} {count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--profile", default="strict", choices=sorted(PROFILES))
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    print("\nExperiment: v38 optimizer winner + v22 smart exit")
    print(f"Entry preset: {args.preset} - {PRESETS[args.preset].description}")
    print(f"Profile: {profile.name} - {profile.description}")
    print("Exit model:")
    print("- take_profit=3.00%")
    print("- stop_loss=1.00%")
    print("- time_exit_bars=60, min_pnl=0.50%")
    print("- trailing_activation=1.50%, trailing_stop=1.00%")

    counters, candidates, trades = run_backtest(args.preset, profile)
    summarize_inputs(counters, candidates, profile)
    summarize_research(trades)
    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
