"""Data-driven entry optimizer for reversal pullback research.

Fast mode is now the default. It avoids huge brute-force grids and finishes quickly.

Run:
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_optimizer --preset quality
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_optimizer --preset quality --min-sample 10
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    NOISY_SYMBOLS,
    build_candidate_session_map,
    simulate_v22_exit,
)
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_audit import (
    find_entry_position,
    pre_entry_stats,
    window_stats,
)
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)

MIN_SAMPLE_DEFAULT = 20


def get_exit_reason(trade) -> str:
    return str(getattr(trade, "exit_reason", getattr(trade, "reason", "unknown")))


def simulate_all_candidates(candidates: list[SimpleEntryCandidate], data_1m: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates_by_symbol_day = build_candidate_session_map(candidates)
    rows = []

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
            if trade is None:
                continue

            entry_idx = find_entry_position(session_1m, candidate.entry_time)
            if entry_idx is None:
                continue

            row = {
                "date": candidate.session_date,
                "symbol": candidate.symbol,
                "entry_time": candidate.entry_time,
                "entry_price": candidate.entry_price,
                "daily_trend_pct": candidate.daily_trend_pct,
                "adr_pct": candidate.avg_daily_range_pct,
                "distance_below_or_high_pct": candidate.distance_below_or_high_pct,
                "pullback_proxy_pct": candidate.pullback_from_recent_5m_high_pct,
                "cs_5m": candidate.close_strength_5m,
                "cs_1m": candidate.entry_close_strength_1m,
                "entry_risk_pct": candidate.entry_risk_pct_1m,
                "pnl_pct": trade.pnl_pct,
                "exit_reason": get_exit_reason(trade),
            }
            for minutes in (5, 15, 30):
                row.update(pre_entry_stats(session_1m, entry_idx, minutes))
            for minutes in (5, 15, 30, 60):
                row.update(window_stats(session_1m, entry_idx, minutes))
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_subset(df: pd.DataFrame, name: str) -> dict:
    if df.empty:
        return {
            "name": name,
            "count": 0,
            "active_days": 0,
            "avg_per_active_day": 0.0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "total_pnl": 0.0,
            "max_loss": 0.0,
            "max_win": 0.0,
            "stop_rate": 0.0,
        }

    active_days = df["date"].nunique()
    return {
        "name": name,
        "count": int(len(df)),
        "active_days": int(active_days),
        "avg_per_active_day": float(len(df) / active_days) if active_days else 0.0,
        "win_rate": float((df["pnl_pct"] > 0).mean() * 100.0),
        "avg_pnl": float(df["pnl_pct"].mean()),
        "median_pnl": float(df["pnl_pct"].median()),
        "total_pnl": float(df["pnl_pct"].sum()),
        "max_loss": float(df["pnl_pct"].min()),
        "max_win": float(df["pnl_pct"].max()),
        "stop_rate": float(df["exit_reason"].astype(str).str.contains("stop-loss", case=False, na=False).mean() * 100.0),
    }


def apply_filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df
    if params.get("exclude_noisy", True):
        out = out[~out["symbol"].isin(NOISY_SYMBOLS)]
    excluded_symbols = params.get("excluded_symbols") or set()
    if excluded_symbols:
        out = out[~out["symbol"].isin(excluded_symbols)]

    return out[
        (out["daily_trend_pct"] >= params["trend_min"])
        & (out["daily_trend_pct"] <= params["trend_max"])
        & (out["pullback_proxy_pct"] >= params["pullback_min"])
        & (out["pullback_proxy_pct"] <= params["pullback_max"])
        & (out["cs_1m"] >= params["cs_min"])
        & (out["cs_1m"] <= params["cs_max"])
        & (out["cs_5m"] <= params["cs5_max"])
        & (out["entry_risk_pct"] >= params["risk_min"])
        & (out["entry_risk_pct"] <= params["risk_max"])
        & (out["distance_below_or_high_pct"] >= params["below_or_min"])
        & (out["distance_below_or_high_pct"] <= params["below_or_max"])
        & (out["pre_15m_low_to_entry_pct"] <= params["pre15_bounce_max"])
    ]


def build_fast_grid():
    return product(
        [(-30, -3), (-15, -3), (-10, -3), (-7, -3), (-5, -3)],
        [(0.8, 3.0), (1.0, 3.0), (1.2, 3.0), (1.5, 3.0), (2.0, 2.5)],
        [(0.70, 0.90), (0.75, 0.90), (0.80, 0.90), (0.70, 0.85), (0.75, 0.85)],
        [(0.0, 10.0), (2.0, 10.0), (4.0, 8.0), (5.0, 8.0), (6.0, 7.0)],
        [0.8, 1.0, 1.2, 1.5, 2.0, 999.0],
        [0.70, 0.75, 0.80, 1.00],
        [(0.0, 5.0), (0.25, 3.5), (0.25, 2.5)],
        [set(), {"UUUU"}, {"UUUU", "RIOT", "RXRX", "JOBY", "DNA"}],
    )


def grid_search(df: pd.DataFrame, min_sample: int) -> pd.DataFrame:
    results = []
    total = 5 * 5 * 5 * 5 * 6 * 4 * 3 * 3
    checked = 0
    print(f"\nRunning fast grid search: {total} combinations")

    for (
        (trend_min, trend_max),
        (pullback_min, pullback_max),
        (cs_min, cs_max),
        (risk_min, risk_max),
        pre15_bounce_max,
        cs5_max,
        (below_or_min, below_or_max),
        excluded_symbols,
    ) in build_fast_grid():
        checked += 1
        if checked % 5000 == 0:
            print(f"Grid progress: {checked}/{total}, results={len(results)}")

        params = {
            "trend_min": trend_min,
            "trend_max": trend_max,
            "pullback_min": pullback_min,
            "pullback_max": pullback_max,
            "cs_min": cs_min,
            "cs_max": cs_max,
            "risk_min": risk_min,
            "risk_max": risk_max,
            "pre15_bounce_max": pre15_bounce_max,
            "cs5_max": cs5_max,
            "below_or_min": below_or_min,
            "below_or_max": below_or_max,
            "exclude_noisy": True,
            "excluded_symbols": excluded_symbols,
        }
        subset = apply_filter(df, params)
        if len(subset) < min_sample:
            continue
        summary = summarize_subset(subset, "grid")
        summary.update({k: v for k, v in params.items() if k != "excluded_symbols"})
        summary["excluded_symbols"] = ",".join(sorted(excluded_symbols))
        results.append(summary)

    if not results:
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    results_df["score"] = (
        results_df["avg_pnl"] * 10.0
        + results_df["win_rate"] * 0.05
        + results_df["total_pnl"] * 0.03
        - results_df["stop_rate"] * 0.03
        + results_df["active_days"] * 0.02
    )
    return results_df.sort_values(["score", "avg_pnl", "total_pnl"], ascending=False)


def print_top_results(results_df: pd.DataFrame, title: str, n: int = 20) -> None:
    print(f"\n=== {title} ===")
    if results_df.empty:
        print("No results with requested minimum sample.")
        return

    cols = [
        "count",
        "active_days",
        "avg_per_active_day",
        "win_rate",
        "avg_pnl",
        "median_pnl",
        "total_pnl",
        "stop_rate",
        "trend_min",
        "trend_max",
        "pullback_min",
        "pullback_max",
        "cs_min",
        "cs_max",
        "risk_min",
        "risk_max",
        "pre15_bounce_max",
        "cs5_max",
        "below_or_min",
        "below_or_max",
        "excluded_symbols",
        "score",
    ]
    print(results_df[cols].head(n).round(3).to_string(index=False))


def print_feature_bins(df: pd.DataFrame) -> None:
    print("\n=== Feature bins on broad pool ===")
    bins = {
        "cs_1m": [0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 999],
        "pullback_proxy_pct": [0, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 999],
        "pre_15m_low_to_entry_pct": [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 999],
        "entry_risk_pct": [0, 2, 4, 5, 6, 7, 8, 10, 999],
        "daily_trend_pct": [-999, -30, -20, -10, -7, -5, -3, 0, 999],
    }
    for col, bin_edges in bins.items():
        tmp = df.copy()
        tmp["bin"] = pd.cut(tmp[col], bin_edges)
        grouped = tmp.groupby("bin", observed=False)["pnl_pct"].agg(["count", "mean", "median", "sum"]).round(3)
        print(f"\n--- {col} ---")
        print(grouped.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--min-sample", type=int, default=MIN_SAMPLE_DEFAULT)
    args = parser.parse_args()

    print(f"\nEntry optimizer preset={args.preset}")
    print("Building broad candidate pool and simulating exits...")

    _counters, candidates, _by_day = scan_entries(PRESETS[args.preset])
    _data_15m, _data_5m, data_1m, _daily_data = load_all_data()
    df = simulate_all_candidates(candidates, data_1m)

    output_dir = Path("data/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / f"entry_optimizer_pool_{args.preset}.csv"
    df.to_csv(pool_path, index=False)
    print(f"Saved broad pool CSV: {pool_path}")

    broad = summarize_subset(df, "broad_pool")
    print("\nBroad pool summary:")
    print(pd.DataFrame([broad]).round(3).to_string(index=False))

    print_feature_bins(df)

    results_df = grid_search(df, args.min_sample)
    results_path = output_dir / f"entry_optimizer_results_{args.preset}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved optimizer results CSV: {results_path}")

    print_top_results(results_df, f"Top data-driven filters, min_sample={args.min_sample}")

    if not results_df.empty:
        best = results_df.iloc[0]
        print("\nBest candidate command idea:")
        for key in [
            "trend_min",
            "trend_max",
            "pullback_min",
            "pullback_max",
            "cs_min",
            "cs_max",
            "risk_min",
            "risk_max",
            "pre15_bounce_max",
            "cs5_max",
            "below_or_min",
            "below_or_max",
            "excluded_symbols",
        ]:
            print(f"- {key}: {best[key]}")


if __name__ == "__main__":
    main()
