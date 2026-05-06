from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    _print_metric_block,
    build_available_1m_opportunities,
)


WEIRD_SUFFIXES = (
    "W",
    "WS",
    "WT",
    "WTS",
    "U",
    "R",
    "RT",
    "RIGHT",
    "UNIT",
)


DEFAULT_EXCLUDED_SYMBOLS = {
    # Leveraged / inverse ETFs are useful for separate experiments, but they distort single-stock momentum research.
    "SOXL",
    "SOXS",
    "TQQQ",
    "SQQQ",
    "SPXL",
    "SPXS",
    "UVXY",
}


def is_weird_symbol(symbol: str) -> bool:
    s = symbol.upper().strip()
    if "." in s or "-" in s or "/" in s:
        return True
    if len(s) > 5:
        # Most common noisy cases in raw NASDAQ lists are warrants/units/rights or test-ish symbols.
        return True
    return any(s.endswith(suffix) for suffix in WEIRD_SUFFIXES)


def apply_quality_filter(
    df: pd.DataFrame,
    *,
    min_open_price: float,
    max_open_price: float | None,
    max_abs_gap: float,
    max_intraday_high: float,
    max_first_5m_high: float,
    min_dollar_volume: float,
    exclude_weird_symbols: bool,
    excluded_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df.copy()
    reasons = []

    # Approximate open price can be derived from low/high only poorly, so use available percent features for anomaly filtering.
    # We keep open price filtering optional for future extension if the base research exports open_price.
    if max_open_price is not None and "open_price" in filtered.columns:
        reasons.append((filtered["open_price"] > max_open_price, "open_price_too_high"))
    if "open_price" in filtered.columns:
        reasons.append((filtered["open_price"] < min_open_price, "open_price_too_low"))

    if "gap_pct" in filtered.columns:
        reasons.append((filtered["gap_pct"].abs() > max_abs_gap, "gap_abs_too_large"))
    if "intraday_high_pct" in filtered.columns:
        reasons.append((filtered["intraday_high_pct"] > max_intraday_high, "intraday_high_too_large"))
    if "first_5m_high_pct" in filtered.columns:
        reasons.append((filtered["first_5m_high_pct"] > max_first_5m_high, "first_5m_spike_too_large"))
    if "dollar_volume" in filtered.columns:
        reasons.append((filtered["dollar_volume"] < min_dollar_volume, "dollar_volume_too_low"))

    if exclude_weird_symbols:
        reasons.append((filtered["symbol"].map(is_weird_symbol), "weird_symbol"))
    if excluded_symbols:
        reasons.append((filtered["symbol"].str.upper().isin(excluded_symbols), "excluded_symbol"))

    reject_reason = pd.Series("", index=filtered.index, dtype="object")
    keep_mask = pd.Series(True, index=filtered.index)
    for mask, reason in reasons:
        mask = mask.fillna(False)
        reject_reason.loc[mask & (reject_reason == "")] = reason
        reject_reason.loc[mask & (reject_reason != reason) & (reject_reason != "")] += ";" + reason
        keep_mask &= ~mask

    rejected = filtered.loc[~keep_mask].copy()
    rejected["quality_reject_reason"] = reject_reason.loc[~keep_mask]
    passed = filtered.loc[keep_mask].copy()
    return passed, rejected


def print_summary(opps: pd.DataFrame) -> None:
    print("\n=== Clean momentum day shape ===")
    for col in [
        "gap_pct",
        "intraday_high_pct",
        "open_to_close_pct",
        "daily_return_pct",
        "time_to_high_minutes",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "first_30m_high_pct",
    ]:
        if col in opps.columns:
            print(f"{col}: avg={opps[col].mean():.2f}, median={opps[col].median():.2f}")

    print("\n=== Diagnostic entry simulations on clean opportunities ===")
    _print_metric_block(
        "or5_breakout",
        opps[opps["or5_found"] == True],
        [
            "or5_max_pnl_pct",
            "or5_min_pnl_pct",
            "or5_close_pnl_pct",
            "or5_trail_10_07_pnl_pct",
            "or5_trail_15_10_pnl_pct",
            "or5_trail_25_15_pnl_pct",
        ],
    )
    _print_metric_block(
        "or15_breakout",
        opps[opps["or15_found"] == True],
        ["or15_max_pnl_pct", "or15_min_pnl_pct", "or15_close_pnl_pct"],
    )
    _print_metric_block(
        "or30_breakout",
        opps[opps["or30_found"] == True],
        ["or30_max_pnl_pct", "or30_min_pnl_pct", "or30_close_pnl_pct"],
    )

    print("\n=== Symbols with most clean >= threshold moves ===")
    print(opps["symbol"].value_counts().head(40).to_string())

    print("\n=== Top clean opportunities ===")
    cols = [
        "session_date",
        "symbol",
        "rows_1m",
        "gap_pct",
        "intraday_high_pct",
        "open_to_close_pct",
        "time_to_high_minutes",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "first_30m_high_pct",
        "or5_max_pnl_pct",
        "or5_min_pnl_pct",
        "or5_close_pnl_pct",
        "or5_trail_25_15_pnl_pct",
    ]
    existing_cols = [c for c in cols if c in opps.columns]
    print(opps[existing_cols].head(50).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze big momentum opportunities from available local 1m candles with tradable-quality filters."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--min-intraday-high", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--min-open-price", type=float, default=1.0)
    parser.add_argument("--max-open-price", type=float, default=None)
    parser.add_argument("--max-abs-gap", type=float, default=50.0)
    parser.add_argument("--max-intraday-high", type=float, default=100.0)
    parser.add_argument("--max-first-5m-high", type=float, default=50.0)
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--include-weird-symbols", action="store_true")
    parser.add_argument("--include-leveraged-etfs", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Clean available 1m big momentum research")
    print(f"Data dir: {data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Opportunity filter: intraday high >= {args.min_intraday_high:.2f}%")
    print("Quality filters:")
    print(f"- rows_1m >= {args.min_rows}")
    print(f"- abs(gap_pct) <= {args.max_abs_gap:.2f}%")
    print(f"- intraday_high_pct <= {args.max_intraday_high:.2f}%")
    print(f"- first_5m_high_pct <= {args.max_first_5m_high:.2f}%")
    print(f"- exclude weird symbols: {not args.include_weird_symbols}")
    print(f"- exclude leveraged ETFs: {not args.include_leveraged_etfs}")

    all_days = build_available_1m_opportunities(data_dir, args.recent_days)
    if all_days.empty:
        print("No local 1m sessions found.")
        return 1

    full_days = all_days[all_days["rows_1m"] >= args.min_rows].copy()
    raw_opps = full_days[full_days["intraday_high_pct"] >= args.min_intraday_high].copy()

    excluded_symbols = set() if args.include_leveraged_etfs else DEFAULT_EXCLUDED_SYMBOLS
    clean_opps, rejected = apply_quality_filter(
        raw_opps,
        min_open_price=args.min_open_price,
        max_open_price=args.max_open_price,
        max_abs_gap=args.max_abs_gap,
        max_intraday_high=args.max_intraday_high,
        max_first_5m_high=args.max_first_5m_high,
        min_dollar_volume=args.min_dollar_volume,
        exclude_weird_symbols=not args.include_weird_symbols,
        excluded_symbols=excluded_symbols,
    )
    clean_opps = clean_opps.sort_values("intraday_high_pct", ascending=False).head(args.top)

    suffix = (
        f"clean_recent{args.recent_days}_intraday_ge_{int(args.min_intraday_high)}_"
        f"maxgap{int(args.max_abs_gap)}_maxhi{int(args.max_intraday_high)}_top{args.top}"
    )
    out_all = output_dir / f"big_momentum_available_1m_clean_all_days_{suffix}.csv"
    out_opps = output_dir / f"big_momentum_available_1m_clean_opportunities_{suffix}.csv"
    out_rejected = output_dir / f"big_momentum_available_1m_clean_rejected_{suffix}.csv"
    all_days.to_csv(out_all, index=False)
    clean_opps.to_csv(out_opps, index=False)
    rejected.to_csv(out_rejected, index=False)

    print(f"\nSaved all 1m day stats CSV: {out_all}")
    print(f"Saved clean opportunity CSV: {out_opps}")
    print(f"Saved rejected opportunity CSV: {out_rejected}")

    print("\n=== Coverage from local 1m files only ===")
    print(f"1m files found: {len(list(data_dir.glob('*.csv')))}")
    print(f"Analyzed sessions: {len(all_days)}")
    print(f"Full-ish sessions rows>={args.min_rows}: {len(full_days)}")
    print(f"Raw opportunities >= threshold: {len(raw_opps)}")
    print(f"Rejected by quality filter: {len(rejected)}")
    print(f"Clean opportunities selected: {len(clean_opps)}")

    if not rejected.empty:
        print("\n=== Quality rejection reasons ===")
        reason_counts = rejected["quality_reject_reason"].value_counts().head(30)
        print(reason_counts.to_string())

    if clean_opps.empty:
        print("\nNo clean opportunities matched the filter.")
        return 0

    print_summary(clean_opps)

    print("\nInterpretation:")
    print("- This is the cleaner research set for a tradable momentum-continuation strategy.")
    print("- Raw absurd moves are saved in the rejected CSV so we can inspect them instead of silently deleting them.")
    print("- If the clean set is too small, loosen max-intraday-high or max-first-5m-high; if it is still noisy, tighten them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
