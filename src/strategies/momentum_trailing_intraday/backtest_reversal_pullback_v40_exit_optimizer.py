"""v40: exit optimizer for the v39 overnight reversal pullback strategy.

Purpose:
- Keep the v38/v39 entry profile fixed.
- Grid-search only the overnight exit parameters.
- Find whether holding overnight / changing trailing parameters improves edge.

Run examples:
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v40_exit_optimizer --preset quality --profile strict
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v40_exit_optimizer --preset quality --profile sample20
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v40_exit_optimizer --preset quality --profile sample20 --top 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import pandas as pd

from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    build_candidate_session_map,
)
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner import (
    PROFILES,
    passes_profile,
)
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v39_overnight import (
    simulate_overnight_exit,
)
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)

OUTPUT_DIR = Path("data/backtests")

STOP_LOSSES = [0.8, 1.0, 1.2, 1.5, 2.0]
TAKE_PROFITS = [2.0, 2.5, 3.0]
TRAILING_ACTIVATIONS = [0.8, 1.0, 1.2, 1.5]
TRAILING_STOPS = [0.6, 0.7, 0.8, 1.0]
MAX_HOLD_DAYS = [1, 2, 3, 5]


def build_profile_candidates(preset_name: str, profile_name: str):
    profile = PROFILES[profile_name]
    _counters, raw_candidates, _candidates_by_day = scan_entries(PRESETS[preset_name])
    _data_15m, _data_5m, data_1m, _daily_data = load_all_data()

    counters = Counter()
    candidates: list[SimpleEntryCandidate] = []
    for candidate in raw_candidates:
        counters["raw_candidates"] += 1
        if passes_profile(candidate, profile, data_1m):
            counters["passed_profile"] += 1
            candidates.append(candidate)
        else:
            counters["rejected_profile"] += 1

    return profile, counters, candidates, data_1m


def simulate_exit_grid(
    candidates: list[SimpleEntryCandidate],
    data_1m: dict[str, pd.DataFrame],
    *,
    stop_loss: float,
    take_profit: float,
    trailing_activation: float,
    trailing_stop: float,
    max_hold_days: int,
):
    candidates_by_symbol_day = build_candidate_session_map(candidates)
    trades = []

    for (symbol, _session_date), day_candidates in candidates_by_symbol_day.items():
        if symbol not in data_1m:
            continue
        symbol_1m = data_1m[symbol].sort_values("date").reset_index(drop=True)
        for candidate in day_candidates:
            trade = simulate_overnight_exit(
                symbol,
                symbol_1m,
                candidate,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                trailing_activation_pct=trailing_activation,
                trailing_stop_pct=trailing_stop,
                max_hold_days=max_hold_days,
            )
            if trade is not None:
                trades.append(trade)

    return apply_costs_to_trades(trades)


def summarize_trades(trades) -> dict[str, float | int | str]:
    if not trades:
        return {
            "count": 0,
            "active_days": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "stop_rate": 0.0,
            "take_profit_rate": 0.0,
            "trailing_rate": 0.0,
            "max_loss": 0.0,
            "max_win": 0.0,
            "profit_factor": 0.0,
            "score": 0.0,
            "exit_reasons": "",
        }

    pnls = [float(trade.pnl_pct) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl <= 0]
    exit_reasons = Counter(str(trade.exit_reason) for trade in trades)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss else 999.0

    count = len(trades)
    active_days = len(set(str(trade.session_date) for trade in trades))
    win_rate = len(wins) / count * 100.0
    avg_pnl = mean(pnls)
    total_pnl = sum(pnls)
    stop_count = sum(count for reason, count in exit_reasons.items() if "stop-loss" in reason)
    take_profit_count = sum(count for reason, count in exit_reasons.items() if "take-profit" in reason)
    trailing_count = sum(count for reason, count in exit_reasons.items() if "trailing" in reason)

    # Prefer stable edge over raw total PnL. Penalize high stop rate and tiny samples.
    score = (
        avg_pnl * max(win_rate, 1.0)
        + total_pnl * 0.20
        + profit_factor * 0.50
        - (stop_count / count * 100.0) * 0.03
    )

    return {
        "count": count,
        "active_days": active_days,
        "avg_per_active_day": count / active_days if active_days else 0.0,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": float(pd.Series(pnls).median()),
        "total_pnl": total_pnl,
        "avg_win": mean(wins) if wins else 0.0,
        "avg_loss": mean(losses) if losses else 0.0,
        "stop_rate": stop_count / count * 100.0,
        "take_profit_rate": take_profit_count / count * 100.0,
        "trailing_rate": trailing_count / count * 100.0,
        "max_loss": min(pnls),
        "max_win": max(pnls),
        "profit_factor": profit_factor,
        "score": score,
        "exit_reasons": "; ".join(f"{reason}={cnt}" for reason, cnt in exit_reasons.most_common()),
    }


def run_grid(preset_name: str, profile_name: str) -> pd.DataFrame:
    profile, counters, candidates, data_1m = build_profile_candidates(preset_name, profile_name)

    print(f"\nEntry preset: {preset_name} - {PRESETS[preset_name].description}")
    print(f"Profile: {profile_name} - {profile.description}")
    print("Funnel counts:")
    for name, count in counters.most_common():
        print(f"{name:<24} {count:>6}")
    print(f"Candidates: {len(candidates)}")

    rows = []
    total_combos = len(STOP_LOSSES) * len(TAKE_PROFITS) * len(TRAILING_ACTIVATIONS) * len(TRAILING_STOPS) * len(MAX_HOLD_DAYS)
    done = 0

    for stop_loss in STOP_LOSSES:
        for take_profit in TAKE_PROFITS:
            for trailing_activation in TRAILING_ACTIVATIONS:
                for trailing_stop in TRAILING_STOPS:
                    for max_hold_days in MAX_HOLD_DAYS:
                        done += 1
                        trades = simulate_exit_grid(
                            candidates,
                            data_1m,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            trailing_activation=trailing_activation,
                            trailing_stop=trailing_stop,
                            max_hold_days=max_hold_days,
                        )
                        row = summarize_trades(trades)
                        row.update(
                            {
                                "preset": preset_name,
                                "profile": profile_name,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "trailing_activation": trailing_activation,
                                "trailing_stop": trailing_stop,
                                "max_hold_days": max_hold_days,
                            }
                        )
                        rows.append(row)

                        if done % 50 == 0 or done == total_combos:
                            print(f"Grid progress: {done}/{total_combos}")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["score", "avg_pnl", "total_pnl", "win_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--profile", default="sample20", choices=sorted(PROFILES))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    print("\nExperiment: v40 overnight exit optimizer")
    print("Grid:")
    print(f"- stop_losses={STOP_LOSSES}")
    print(f"- take_profits={TAKE_PROFITS}")
    print(f"- trailing_activations={TRAILING_ACTIVATIONS}")
    print(f"- trailing_stops={TRAILING_STOPS}")
    print(f"- max_hold_days={MAX_HOLD_DAYS}")

    df = run_grid(args.preset, args.profile)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"v40_exit_optimizer_{args.preset}_{args.profile}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved optimizer results CSV: {output_path}")
    print(f"\n=== Top {args.top} exit configs ===")
    display_columns = [
        "count",
        "active_days",
        "win_rate",
        "avg_pnl",
        "median_pnl",
        "total_pnl",
        "avg_win",
        "avg_loss",
        "stop_rate",
        "take_profit_rate",
        "trailing_rate",
        "profit_factor",
        "stop_loss",
        "take_profit",
        "trailing_activation",
        "trailing_stop",
        "max_hold_days",
        "score",
        "exit_reasons",
    ]
    print(df[display_columns].head(args.top).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not df.empty:
        best = df.iloc[0]
        print("\nBest command idea:")
        print(
            "python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v39_overnight "
            f"--preset {args.preset} --profile {args.profile} "
            f"--stop-loss {best['stop_loss']} "
            f"--take-profit {best['take_profit']} "
            f"--trailing-activation {best['trailing_activation']} "
            f"--trailing-stop {best['trailing_stop']} "
            f"--max-hold-days {int(best['max_hold_days'])}"
        )


if __name__ == "__main__":
    main()
