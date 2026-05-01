"""Entry timing audit for reversal pullback research.

This is NOT a strategy and does not optimize filters.
It answers whether current entry candidates are early, late, or chasing the bounce.

For each v29 simple entry candidate it reports:
- entry timestamp and price
- 1m move before entry over 5/15/30 minutes
- MFE/MAE after entry over 5/15/30/60 minutes
- close-to-high-after-entry ratio: how much of the future move was already gone
- whether the first 15/30 minutes after entry went favorable or adverse

Run:
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_audit --preset quality
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_audit --preset balanced
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_audit --preset quality --v33-context
python -m src.strategies.momentum_trailing_intraday.reversal_pullback_entry_audit --preset quality --v35-context
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    NOISY_SYMBOLS,
    passes_v33_context,
)
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v35_context import passes_v35_context
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)


def find_entry_position(session_1m: pd.DataFrame, entry_time: str) -> int | None:
    ts = pd.Timestamp(entry_time)
    matches = session_1m.index[session_1m["date"] == ts].tolist()
    if matches:
        return int(matches[0])
    after = session_1m.index[session_1m["date"] >= ts].tolist()
    if after:
        return int(after[0])
    return None


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def window_stats(session: pd.DataFrame, entry_idx: int, minutes: int) -> dict[str, float | str]:
    entry_bar = session.iloc[entry_idx]
    entry_price = float(entry_bar["close"])
    future = session.iloc[entry_idx + 1 : entry_idx + 1 + minutes]

    if future.empty:
        return {
            f"mfe_{minutes}m_pct": 0.0,
            f"mae_{minutes}m_pct": 0.0,
            f"close_{minutes}m_pct": 0.0,
            f"first_direction_{minutes}m": "no_data",
        }

    high = float(future["high"].max())
    low = float(future["low"].min())
    close = float(future.iloc[-1]["close"])
    mfe = pct_change(high, entry_price)
    mae = pct_change(low, entry_price)
    close_return = pct_change(close, entry_price)

    if abs(mfe) >= abs(mae) and mfe > 0:
        direction = "favorable"
    elif mae < 0:
        direction = "adverse"
    else:
        direction = "flat"

    return {
        f"mfe_{minutes}m_pct": mfe,
        f"mae_{minutes}m_pct": mae,
        f"close_{minutes}m_pct": close_return,
        f"first_direction_{minutes}m": direction,
    }


def pre_entry_stats(session: pd.DataFrame, entry_idx: int, minutes: int) -> dict[str, float]:
    entry_price = float(session.iloc[entry_idx]["close"])
    start_idx = max(0, entry_idx - minutes)
    before = session.iloc[start_idx:entry_idx]
    if before.empty:
        return {
            f"pre_{minutes}m_return_pct": 0.0,
            f"pre_{minutes}m_low_to_entry_pct": 0.0,
            f"pre_{minutes}m_high_to_entry_pct": 0.0,
        }

    first_close = float(before.iloc[0]["close"])
    low = float(before["low"].min())
    high = float(before["high"].max())
    return {
        f"pre_{minutes}m_return_pct": pct_change(entry_price, first_close),
        f"pre_{minutes}m_low_to_entry_pct": pct_change(entry_price, low),
        f"pre_{minutes}m_high_to_entry_pct": pct_change(entry_price, high),
    }


def classify_timing(row: dict) -> str:
    pre_15 = row["pre_15m_low_to_entry_pct"]
    mfe_15 = row["mfe_15m_pct"]
    mae_15 = row["mae_15m_pct"]
    mfe_60 = row["mfe_60m_pct"]

    # Entry is likely late if price already bounced materially from recent low
    # and future upside is smaller than the bounce already captured before entry.
    if pre_15 >= 1.0 and mfe_15 < pre_15 * 0.75:
        return "late_after_bounce"
    if mae_15 <= -1.0 and mfe_15 < 0.75:
        return "too_early_or_wrong"
    if mfe_15 >= 1.0 or mfe_60 >= 1.5:
        return "good_timing"
    return "weak_followthrough"


def filter_candidates(candidates: list[SimpleEntryCandidate], mode: str) -> list[SimpleEntryCandidate]:
    if mode == "all":
        return candidates
    if mode == "exclude_noisy":
        return [candidate for candidate in candidates if candidate.symbol not in NOISY_SYMBOLS]
    if mode == "v33":
        return [candidate for candidate in candidates if passes_v33_context(candidate)]
    if mode == "v35":
        return [candidate for candidate in candidates if passes_v35_context(candidate)]
    raise ValueError(f"Unknown mode: {mode}")


def audit_candidates(preset_name: str, mode: str) -> pd.DataFrame:
    _counters, candidates, _by_day = scan_entries(PRESETS[preset_name])
    candidates = filter_candidates(candidates, mode)
    _data_15m, _data_5m, data_1m, _daily_data = load_all_data()

    rows = []
    for candidate in candidates:
        if candidate.symbol not in data_1m:
            continue
        session = data_1m[candidate.symbol][data_1m[candidate.symbol]["session_date"].astype(str) == candidate.session_date].sort_values("date").reset_index(drop=True)
        if session.empty:
            continue
        entry_idx = find_entry_position(session, candidate.entry_time)
        if entry_idx is None:
            continue

        entry_price = float(session.iloc[entry_idx]["close"])
        row = {
            "date": candidate.session_date,
            "symbol": candidate.symbol,
            "entry_time": str(session.iloc[entry_idx]["date"]),
            "entry_price": entry_price,
            "daily_trend_pct": candidate.daily_trend_pct,
            "adr_pct": candidate.avg_daily_range_pct,
            "pullback_proxy_pct": candidate.pullback_from_recent_5m_high_pct,
            "distance_below_or_high_pct": candidate.distance_below_or_high_pct,
            "cs_5m": candidate.close_strength_5m,
            "cs_1m": candidate.entry_close_strength_1m,
            "entry_risk_pct": candidate.entry_risk_pct_1m,
        }

        for minutes in (5, 15, 30):
            row.update(pre_entry_stats(session, entry_idx, minutes))
        for minutes in (5, 15, 30, 60):
            row.update(window_stats(session, entry_idx, minutes))

        row["timing_class"] = classify_timing(row)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_audit(df: pd.DataFrame) -> None:
    print("\nEntry timing audit")
    print(f"Audited entries: {len(df)}")
    if df.empty:
        return

    print("\nTiming class counts:")
    for label, count in Counter(df["timing_class"]).most_common():
        pct = count / len(df) * 100.0
        print(f"{label:<22} {count:>5} ({pct:5.1f}%)")

    print("\nCore timing stats:")
    metrics = [
        "pre_15m_low_to_entry_pct",
        "pre_15m_return_pct",
        "mfe_15m_pct",
        "mae_15m_pct",
        "mfe_30m_pct",
        "mae_30m_pct",
        "mfe_60m_pct",
        "mae_60m_pct",
    ]
    print(df[metrics].describe().round(3).to_string())

    print("\nTiming class means:")
    print(
        df.groupby("timing_class")[[
            "pre_15m_low_to_entry_pct",
            "mfe_15m_pct",
            "mae_15m_pct",
            "mfe_60m_pct",
            "mae_60m_pct",
            "pullback_proxy_pct",
            "cs_1m",
            "entry_risk_pct",
        ]]
        .mean()
        .round(3)
        .sort_values("mfe_60m_pct", ascending=False)
        .to_string()
    )

    print("\nRecent audited entries:")
    cols = [
        "date",
        "symbol",
        "entry_price",
        "daily_trend_pct",
        "pullback_proxy_pct",
        "cs_1m",
        "pre_15m_low_to_entry_pct",
        "mfe_15m_pct",
        "mae_15m_pct",
        "mfe_60m_pct",
        "mae_60m_pct",
        "timing_class",
    ]
    print(df[cols].tail(40).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--mode", default="all", choices=["all", "exclude_noisy", "v33", "v35"])
    # Backward-compatible convenience flags.
    parser.add_argument("--v33-context", action="store_true")
    parser.add_argument("--v35-context", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    if args.v33_context:
        mode = "v33"
    if args.v35_context:
        mode = "v35"

    print(f"\nEntry audit preset={args.preset} mode={mode}")
    df = audit_candidates(args.preset, mode)

    output_dir = Path("data/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"entry_audit_{args.preset}_{mode}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved audit CSV: {output_path}")

    summarize_audit(df)


if __name__ == "__main__":
    main()
