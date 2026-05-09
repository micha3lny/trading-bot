from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = "data/backtests/v60_baseline_costed.csv"
DEFAULT_FALLBACK_INPUT = "data/backtests/v58_wide_universe_costed.csv"
DEFAULT_OUTPUT_DIR = "data/analysis"

FEATURE_CANDIDATES = [
    # Live/early-ish features available in the candidate table
    "entry_price",
    "gap_pct",
    "first_5m_high_pct",
    "first_15m_high_pct",
    "or_range_pct",
    "or_close_strength",
    "entry_minutes_from_open",
    "entry_candle_strength",
    "pre_entry_break_below_or_pct",
    "entry_from_open_pct",
    "market_return_from_open_pct",
    "market_above_vwap",
    "market_first5_high_pct",
    "market_first5_low_pct",
    "market_first15_high_pct",
    "market_first15_low_pct",
    "position_mult",
    "candidate_rank",
    "open_exposure_before",
    "open_positions_before",
    "or_volume",
    "or_dollar_volume",
    "session_volume",
    "session_dollar_volume",
    "median_1m_volume",
    "first_15m_volume",
    "first_15m_dollar_volume",
    "v45_score",
]

GROUP_FEATURES = ["market_regime", "setup_quality", "reason", "variant"]


def pick_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    net_col = None
    for c in ["net_profit_usd", "profit_after_costs_usd", "net_pnl_usd"]:
        if c in out.columns:
            net_col = c
            break
    gross_col = None
    for c in ["gross_profit_usd", "profit_usd", "gross_pnl_usd"]:
        if c in out.columns:
            gross_col = c
            break
    if net_col is None:
        raise ValueError("Input must contain net result column: net_profit_usd/profit_after_costs_usd/net_pnl_usd")
    if gross_col is None:
        raise ValueError("Input must contain gross result column: gross_profit_usd/profit_usd/gross_pnl_usd")

    out["net_profit_usd"] = pd.to_numeric(out[net_col], errors="coerce")
    out["gross_profit_usd"] = pd.to_numeric(out[gross_col], errors="coerce")
    if "execution_cost_usd" in out.columns:
        out["execution_cost_usd"] = pd.to_numeric(out["execution_cost_usd"], errors="coerce")
    else:
        out["execution_cost_usd"] = out["gross_profit_usd"] - out["net_profit_usd"]

    if "entry_dt" not in out.columns and "entry_time" in out.columns:
        out["entry_dt"] = pd.to_datetime(out["entry_time"], errors="coerce")
    elif "entry_dt" in out.columns:
        out["entry_dt"] = pd.to_datetime(out["entry_dt"], errors="coerce")

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper()

    for c in FEATURE_CANDIDATES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "market_above_vwap" in out.columns:
        out["market_above_vwap"] = out["market_above_vwap"].astype(str).str.lower().isin(["true", "1", "yes"])
        out["market_above_vwap"] = out["market_above_vwap"].astype(float)

    return out.dropna(subset=["net_profit_usd"]).copy()


def label_continuation_archetype(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()

    # This label intentionally includes hindsight to identify the archetype.
    # We do NOT use this directly in live trading.
    out["is_continuation_winner_hindsight"] = (
        (out["net_profit_usd"] > 0)
        & (out.get("first_15m_high_pct", pd.Series(index=out.index, dtype=float)) >= args.min_first_15m_high_pct)
        & (out.get("time_to_high_minutes", pd.Series(index=out.index, dtype=float)) >= args.min_time_to_high_minutes)
    )

    out["is_bad_spike_fade_hindsight"] = (
        (out["net_profit_usd"] < 0)
        & (out.get("time_to_high_minutes", pd.Series(index=out.index, dtype=float)) < args.max_spike_fade_time_to_high)
    )
    return out


def feature_diagnostics(df: pd.DataFrame, target_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    pos = df[df[target_col] == True]
    neg = df[df[target_col] == False]

    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature]
        if series.notna().sum() < 10:
            continue

        pos_s = pos[feature].dropna()
        neg_s = neg[feature].dropna()
        if len(pos_s) < 3 or len(neg_s) < 3:
            continue

        pos_mean = float(pos_s.mean())
        neg_mean = float(neg_s.mean())
        pos_median = float(pos_s.median())
        neg_median = float(neg_s.median())
        pooled = float(series.std()) if series.std() and not pd.isna(series.std()) else 0.0
        effect = (pos_mean - neg_mean) / pooled if pooled else 0.0

        # Simple one-way threshold suggestion: try quantiles and pick highest expectancy lift.
        best = None
        for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            threshold = float(series.quantile(q))
            for direction in [">=", "<="]:
                mask = series >= threshold if direction == ">=" else series <= threshold
                subset = df[mask]
                if len(subset) < 5:
                    continue
                row = {
                    "threshold": threshold,
                    "direction": direction,
                    "trades": len(subset),
                    "net_profit_usd": float(subset["net_profit_usd"].sum()),
                    "avg_net_trade_usd": float(subset["net_profit_usd"].mean()),
                    "win_rate_pct": float((subset["net_profit_usd"] > 0).mean() * 100),
                }
                if best is None or row["avg_net_trade_usd"] > best["avg_net_trade_usd"]:
                    best = row

        rows.append({
            "feature": feature,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "pos_median": pos_median,
            "neg_median": neg_median,
            "effect_size_mean_diff_std": effect,
            "best_direction": best["direction"] if best else None,
            "best_threshold": best["threshold"] if best else np.nan,
            "best_threshold_trades": best["trades"] if best else 0,
            "best_threshold_net_profit_usd": best["net_profit_usd"] if best else np.nan,
            "best_threshold_avg_net_trade_usd": best["avg_net_trade_usd"] if best else np.nan,
            "best_threshold_win_rate_pct": best["win_rate_pct"] if best else np.nan,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("best_threshold_avg_net_trade_usd", ascending=False)
    return out


def group_diagnostics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for value, g in df.groupby(group_col, dropna=False):
        if len(g) < 3:
            continue
        rows.append({
            group_col: str(value),
            "trades": len(g),
            "symbols": g["symbol"].nunique() if "symbol" in g.columns else 0,
            "net_profit_usd": float(g["net_profit_usd"].sum()),
            "avg_net_trade_usd": float(g["net_profit_usd"].mean()),
            "gross_profit_usd": float(g["gross_profit_usd"].sum()),
            "execution_cost_usd": float(g["execution_cost_usd"].sum()),
            "win_rate_pct": float((g["net_profit_usd"] > 0).mean() * 100),
            "continuation_winner_rate_pct": float(g["is_continuation_winner_hindsight"].mean() * 100),
            "spike_fade_rate_pct": float(g["is_bad_spike_fade_hindsight"].mean() * 100),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("avg_net_trade_usd", ascending=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v61 continuation winner feature diagnostics")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=5.0)
    parser.add_argument("--min-time-to-high-minutes", type=float, default=30.0)
    parser.add_argument("--max-spike-fade-time-to-high", type=float, default=20.0)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        fallback = Path(DEFAULT_FALLBACK_INPUT)
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(path)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = normalize_schema(pd.read_csv(path))
    df = label_continuation_archetype(df, args)

    features = pick_existing(df, FEATURE_CANDIDATES)
    diagnostics = feature_diagnostics(df, "is_continuation_winner_hindsight", features)

    group_frames = []
    for gcol in GROUP_FEATURES:
        gdf = group_diagnostics(df, gcol)
        if not gdf.empty:
            gdf.insert(0, "group_column", gcol)
            group_frames.append(gdf)
    groups = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()

    archetype_summary = pd.DataFrame([
        {
            "segment": "continuation_winner_hindsight",
            "trades": int(df["is_continuation_winner_hindsight"].sum()),
            "net_profit_usd": float(df.loc[df["is_continuation_winner_hindsight"], "net_profit_usd"].sum()),
            "avg_net_trade_usd": float(df.loc[df["is_continuation_winner_hindsight"], "net_profit_usd"].mean()),
            "win_rate_pct": float((df.loc[df["is_continuation_winner_hindsight"], "net_profit_usd"] > 0).mean() * 100),
        },
        {
            "segment": "bad_spike_fade_hindsight",
            "trades": int(df["is_bad_spike_fade_hindsight"].sum()),
            "net_profit_usd": float(df.loc[df["is_bad_spike_fade_hindsight"], "net_profit_usd"].sum()),
            "avg_net_trade_usd": float(df.loc[df["is_bad_spike_fade_hindsight"], "net_profit_usd"].mean()),
            "win_rate_pct": float((df.loc[df["is_bad_spike_fade_hindsight"], "net_profit_usd"] > 0).mean() * 100),
        },
        {
            "segment": "all_trades",
            "trades": len(df),
            "net_profit_usd": float(df["net_profit_usd"].sum()),
            "avg_net_trade_usd": float(df["net_profit_usd"].mean()),
            "win_rate_pct": float((df["net_profit_usd"] > 0).mean() * 100),
        },
    ])

    diagnostics.to_csv(outdir / "v61_continuation_feature_diagnostics.csv", index=False)
    groups.to_csv(outdir / "v61_group_diagnostics.csv", index=False)
    archetype_summary.to_csv(outdir / "v61_continuation_archetype_summary.csv", index=False)
    df.to_csv(outdir / "v61_labeled_trades.csv", index=False)

    print("=== v61 continuation feature analysis ===")
    print(f"Input: {path}")
    print(f"Trades: {len(df)}")

    print("\n=== Archetype summary ===")
    print(archetype_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Top candidate live/proxy features ===")
    if diagnostics.empty:
        print("No diagnostics generated")
    else:
        show_cols = [
            "feature",
            "pos_mean",
            "neg_mean",
            "effect_size_mean_diff_std",
            "best_direction",
            "best_threshold",
            "best_threshold_trades",
            "best_threshold_net_profit_usd",
            "best_threshold_avg_net_trade_usd",
            "best_threshold_win_rate_pct",
        ]
        print(diagnostics[show_cols].head(25).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Best group diagnostics ===")
    if groups.empty:
        print("No group diagnostics generated")
    else:
        print(groups.head(30).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nSaved:")
    print(outdir / "v61_continuation_feature_diagnostics.csv")
    print(outdir / "v61_group_diagnostics.csv")
    print(outdir / "v61_continuation_archetype_summary.csv")
    print(outdir / "v61_labeled_trades.csv")

    print("\nInterpretation hints:")
    print("- continuation_winner_hindsight is a research label, not a live rule.")
    print("- Prefer features available at or before entry: first_15m_high_pct, OR metrics, entry candle, market regime, gap, price.")
    print("- time_to_high_minutes identifies the archetype but must be replaced by live-safe proxies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
