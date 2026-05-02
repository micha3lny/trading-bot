"""Audit why the reversal pullback strategy misses the biggest daily movers.

This is intentionally diagnostic, not a trading strategy.

It compares large daily opportunities against the existing simple-entry candidate pool
and v38/v39 profiles. The goal is to answer:
- Did the broad entry scanner create any candidate on a big-move day?
- If yes, why did v38/v39 reject it?
- If no, was the day outside the pullback/reversal assumptions entirely?

Run examples:
python -m src.analysis.big_move_missed_entry_audit --preset quality --min-intraday-high 10 --top 80
python -m src.analysis.big_move_missed_entry_audit --preset loose --min-intraday-high 10 --top 80
python -m src.analysis.big_move_missed_entry_audit --preset quality --min-close-return 10 --top 80
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner import (
    PROFILES,
    V38Profile,
    candidate_pre15_bounce,
)
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)


NO_VALUE = 999999.0


def daily_opportunities(daily_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, df in daily_data.items():
        if not {"date", "open", "high", "low", "close"}.issubset(df.columns):
            continue
        d = df.sort_values("date").copy()
        d["session_date"] = d["date"].dt.date.astype(str)
        d["daily_return_pct"] = (d["close"] - d["open"]) / d["open"] * 100.0
        d["intraday_high_pct"] = (d["high"] - d["open"]) / d["open"] * 100.0
        d["intraday_low_pct"] = (d["low"] - d["open"]) / d["open"] * 100.0
        for _, row in d.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": str(row["session_date"]),
                    "daily_return_pct": float(row["daily_return_pct"]),
                    "intraday_high_pct": float(row["intraday_high_pct"]),
                    "intraday_low_pct": float(row["intraday_low_pct"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
    return pd.DataFrame(rows)


def profile_rejection_reasons(
    candidate: SimpleEntryCandidate,
    profile: V38Profile,
    data_1m: dict[str, pd.DataFrame],
) -> list[str]:
    reasons: list[str] = []
    if candidate.symbol in profile.excluded_symbols:
        reasons.append("excluded_symbol")
    if not (profile.trend_min <= candidate.daily_trend_pct <= profile.trend_max):
        reasons.append("daily_trend")
    if not (profile.pullback_min <= candidate.pullback_from_recent_5m_high_pct <= profile.pullback_max):
        reasons.append("pullback_proxy")
    if not (profile.cs_min <= candidate.entry_close_strength_1m <= profile.cs_max):
        reasons.append("cs_1m")
    if not (profile.risk_min <= candidate.entry_risk_pct_1m <= profile.risk_max):
        reasons.append("entry_risk")
    if candidate.close_strength_5m > profile.cs5_max:
        reasons.append("cs5_too_high")
    if not (profile.below_or_min <= candidate.distance_below_or_high_pct <= profile.below_or_max):
        reasons.append("below_or_distance")
    pre15_bounce = candidate_pre15_bounce(candidate, data_1m)
    if pre15_bounce is None:
        reasons.append("pre15_missing")
    elif pre15_bounce > profile.pre15_bounce_max:
        reasons.append("pre15_bounce_too_late")
    return reasons


def first_or_best_candidate(candidates: list[SimpleEntryCandidate]) -> SimpleEntryCandidate | None:
    if not candidates:
        return None
    # Prefer a candidate with stronger 1m confirmation and lower bounce distance.
    return sorted(candidates, key=lambda c: (-c.entry_close_strength_1m, c.entry_risk_pct_1m))[0]


def summarize_reasons(reason_lists: Iterable[list[str]]) -> Counter:
    counter = Counter()
    for reasons in reason_lists:
        if not reasons:
            counter["passed_profile"] += 1
        for reason in reasons:
            counter[reason] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--profile", default="sample20", choices=sorted(PROFILES))
    parser.add_argument("--min-intraday-high", type=float, default=10.0)
    parser.add_argument("--min-close-return", type=float, default=None)
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    print("\nBig move missed-entry audit")
    print(f"Preset: {args.preset}")
    print(f"Profile: {args.profile}")
    if args.min_close_return is not None:
        print(f"Opportunity filter: daily close return >= {args.min_close_return:.2f}%")
    else:
        print(f"Opportunity filter: intraday high >= {args.min_intraday_high:.2f}%")

    print("\nLoading data and broad candidate pool...")
    _data_15m, _data_5m, data_1m, daily_data = load_all_data()
    _counters, candidates, _by_day = scan_entries(PRESETS[args.preset])
    profile = PROFILES[args.profile]

    opps = daily_opportunities(daily_data)
    if args.min_close_return is not None:
        opps = opps[opps["daily_return_pct"] >= args.min_close_return]
        opps = opps.sort_values("daily_return_pct", ascending=False).head(args.top)
    else:
        opps = opps[opps["intraday_high_pct"] >= args.min_intraday_high]
        opps = opps.sort_values("intraday_high_pct", ascending=False).head(args.top)

    candidates_by_key: dict[tuple[str, str], list[SimpleEntryCandidate]] = {}
    for c in candidates:
        candidates_by_key.setdefault((c.symbol, c.session_date), []).append(c)

    rows: list[dict] = []
    all_profile_reasons: list[list[str]] = []
    for _, opp in opps.iterrows():
        key = (opp["symbol"], opp["session_date"])
        day_candidates = candidates_by_key.get(key, [])
        chosen = first_or_best_candidate(day_candidates)

        if chosen is None:
            row = {
                **opp.to_dict(),
                "broad_candidate_count": 0,
                "candidate_entry_time": "",
                "candidate_entry_price": None,
                "daily_trend_pct": None,
                "avg_daily_range_pct": None,
                "below_or_pct": None,
                "pullback_proxy_pct": None,
                "cs5": None,
                "cs1": None,
                "entry_risk_pct": None,
                "pre15_bounce_pct": None,
                "profile_status": "no_broad_candidate",
                "profile_reject_reasons": "no_broad_candidate",
            }
        else:
            pre15_bounce = candidate_pre15_bounce(chosen, data_1m)
            reasons = profile_rejection_reasons(chosen, profile, data_1m)
            all_profile_reasons.append(reasons)
            row = {
                **opp.to_dict(),
                "broad_candidate_count": len(day_candidates),
                "candidate_entry_time": chosen.entry_time,
                "candidate_entry_price": chosen.entry_price,
                "daily_trend_pct": chosen.daily_trend_pct,
                "avg_daily_range_pct": chosen.avg_daily_range_pct,
                "below_or_pct": chosen.distance_below_or_high_pct,
                "pullback_proxy_pct": chosen.pullback_from_recent_5m_high_pct,
                "cs5": chosen.close_strength_5m,
                "cs1": chosen.entry_close_strength_1m,
                "entry_risk_pct": chosen.entry_risk_pct_1m,
                "pre15_bounce_pct": pre15_bounce if pre15_bounce is not None else NO_VALUE,
                "profile_status": "passed_profile" if not reasons else "rejected_profile",
                "profile_reject_reasons": ",".join(reasons) if reasons else "",
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    output_dir = Path("data/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.preset}_{args.profile}"
    if args.min_close_return is not None:
        suffix += f"_close_ge_{args.min_close_return:g}"
    else:
        suffix += f"_intraday_ge_{args.min_intraday_high:g}"
    path = output_dir / f"big_move_missed_entry_audit_{suffix}.csv"
    out.to_csv(path, index=False)

    print(f"\nSaved audit CSV: {path}")
    print("\n=== Big move coverage ===")
    print(f"Audited opportunities: {len(out)}")
    print(f"With broad candidate same day: {(out['broad_candidate_count'] > 0).sum()}")
    print(f"No broad candidate same day: {(out['broad_candidate_count'] == 0).sum()}")
    print(f"Passed selected profile: {(out['profile_status'] == 'passed_profile').sum()}")
    print(f"Rejected by selected profile: {(out['profile_status'] == 'rejected_profile').sum()}")

    print("\n=== Profile rejection reasons for big-move days with candidates ===")
    reason_counts = summarize_reasons(all_profile_reasons)
    for reason, count in reason_counts.most_common():
        print(f"{reason:<28} {count}")

    print("\n=== Top missed/covered opportunities ===")
    display_cols = [
        "session_date",
        "symbol",
        "intraday_high_pct",
        "daily_return_pct",
        "broad_candidate_count",
        "profile_status",
        "profile_reject_reasons",
        "daily_trend_pct",
        "pullback_proxy_pct",
        "cs5",
        "cs1",
        "entry_risk_pct",
        "pre15_bounce_pct",
    ]
    print(
        out[display_cols]
        .head(min(args.top, 40))
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nInterpretation hints:")
    print("- no_broad_candidate = v29 scanner never saw a pullback/reversal setup on that day.")
    print("- daily_trend = day was usually not in the selloff-bounce context.")
    print("- pullback_proxy / cs5 / cs1 = move shape was momentum/gap, not clean pullback entry.")
    print("- pre15_bounce_too_late = entry appeared only after a big part of the move already happened.")


if __name__ == "__main__":
    main()
