from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR


def load_trades(path: str, variant: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "variant" not in df.columns:
        raise ValueError("trades CSV must contain a 'variant' column")
    out = df[df["variant"] == variant].copy()
    if out.empty:
        raise ValueError(f"No trades found for variant={variant!r}")
    out["entry_time_dt"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date.astype(str)
    return out.sort_values(["entry_time_dt", "symbol"]).reset_index(drop=True)


def find_market_file(data_dir: str, preferred_symbols: list[str]) -> tuple[str, Path]:
    base = Path(data_dir)
    for symbol in preferred_symbols:
        p = base / f"{symbol.upper()}.csv"
        if p.exists():
            return symbol.upper(), p
    raise FileNotFoundError(f"No market regime file found in {base}; tried: {preferred_symbols}")


def load_market_data(data_dir: str, preferred_symbols: list[str]) -> tuple[str, pd.DataFrame]:
    symbol, path = find_market_file(data_dir, preferred_symbols)
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError(f"Market file {path} must contain datetime column")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["session_date"] = df["datetime"].dt.date.astype(str)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Market file {path} missing columns: {sorted(missing)}")
    return symbol, df


def add_market_features(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    first_by_day = market.groupby("session_date").first()[["open"]].rename(columns={"open": "market_day_open"})
    market = market.merge(first_by_day, left_on="session_date", right_index=True, how="left")
    market["market_return_from_open_pct"] = (market["close"] / market["market_day_open"] - 1.0) * 100.0

    # VWAP proxy using OHLC typical price if volume exists.
    if "volume" in market.columns:
        market["typical_price"] = (market["high"] + market["low"] + market["close"]) / 3.0
        market["pv"] = market["typical_price"] * market["volume"]
        market["cum_pv"] = market.groupby("session_date")["pv"].cumsum()
        market["cum_vol"] = market.groupby("session_date")["volume"].cumsum()
        market["market_vwap"] = market["cum_pv"] / market["cum_vol"].replace(0, pd.NA)
        market["market_above_vwap"] = market["close"] >= market["market_vwap"]
    else:
        market["market_vwap"] = pd.NA
        market["market_above_vwap"] = True

    # Opening range for market ETF.
    rows = []
    for session_date, day in market.groupby("session_date", sort=False):
        day = day.sort_values("datetime")
        first5 = day.iloc[:5]
        first15 = day.iloc[:15]
        if first5.empty:
            continue
        day_open = float(day.iloc[0]["open"])
        rows.append({
            "session_date": session_date,
            "market_first5_high_pct": (float(first5["high"].max()) / day_open - 1.0) * 100.0,
            "market_first5_low_pct": (float(first5["low"].min()) / day_open - 1.0) * 100.0,
            "market_first15_high_pct": (float(first15["high"].max()) / day_open - 1.0) * 100.0,
            "market_first15_low_pct": (float(first15["low"].min()) / day_open - 1.0) * 100.0,
        })
    day_features = pd.DataFrame(rows)

    keep_cols = ["datetime", "session_date", "close", "market_day_open", "market_return_from_open_pct", "market_above_vwap", "market_vwap"]
    m = market[keep_cols].rename(columns={"datetime": "market_time", "close": "market_close"}).sort_values("market_time")

    trades = trades.copy().sort_values("entry_time_dt")
    joined = pd.merge_asof(
        trades,
        m,
        left_on="entry_time_dt",
        right_on="market_time",
        by="session_date",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
    )
    joined = joined.merge(day_features, on="session_date", how="left")
    return joined


def apply_market_regime(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    reasons = []
    for _, row in df.iterrows():
        r = []
        if pd.isna(row.get("market_return_from_open_pct")):
            r.append("missing_market_bar")
        else:
            if args.require_market_green and float(row["market_return_from_open_pct"]) < args.min_market_return_from_open_pct:
                r.append("market_return_too_weak")
            if args.require_market_above_vwap and not bool(row.get("market_above_vwap", False)):
                r.append("market_below_vwap")
            if pd.notna(row.get("market_first15_low_pct")) and float(row["market_first15_low_pct"]) < -args.max_market_first15_drawdown_pct:
                r.append("market_first15_drawdown_too_large")
        if args.avoid_entry_10_15 and 10 < float(row.get("entry_minutes_from_open", 999)) < 15:
            r.append("bad_entry_timing_10_15")
        reasons.append(";".join(r))

    out = df.copy()
    out["v49_reject_reason"] = reasons
    accepted = out[out["v49_reject_reason"] == ""].copy()
    rejected = out[out["v49_reject_reason"] != ""].copy()

    if not accepted.empty:
        accepted["position_usd"] = args.position_size
        accepted["profit_usd"] = accepted["position_usd"] * accepted["pnl_pct"] / 100.0
    return accepted, rejected


def summarize(label: str, df: pd.DataFrame, position_size: float) -> dict[str, object]:
    if df.empty:
        return {"strategy": label, "count": 0, "active_days": 0, "symbols": 0, "win_rate": 0.0, "avg_pnl": 0.0, "median_pnl": 0.0, "total_pnl": 0.0, "profit_usd": 0.0, "max_drawdown_usd": 0.0}
    pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    profit = pnl / 100.0 * position_size
    equity = profit.cumsum()
    dd = equity - equity.cummax()
    return {
        "strategy": label,
        "count": int(len(df)),
        "active_days": int(df["session_date"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "win_rate": float((pnl > 0).mean() * 100.0),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "total_pnl": float(pnl.sum()),
        "profit_usd": float(profit.sum()),
        "avg_profit_per_trade_usd": float(profit.mean()),
        "max_drawdown_usd": float(dd.min()),
        "max_loss": float(pnl.min()),
        "max_win": float(pnl.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v49 real market ETF regime filter on v46 wide_trail trades.")
    parser.add_argument("--trades-csv", default="data/backtests/v46_trend_exit_momentum_or_breakout_trades_recent90_or5_score7_trendexit.csv")
    parser.add_argument("--source-variant", default="wide_trail")
    parser.add_argument("--data-dir", default="data/1m")
    parser.add_argument("--market-symbols", nargs="*", default=["QQQ", "SPY", "IWM"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--opening-range-minutes", type=int, default=5)

    parser.add_argument("--require-market-green", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-market-return-from-open-pct", type=float, default=0.0)
    parser.add_argument("--require-market-above-vwap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-market-first15-drawdown-pct", type=float, default=0.75)
    parser.add_argument("--avoid-entry-10-15", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    print("Experiment: v49 real market regime filter")
    print("Base: v46 wide_trail trades + market ETF 1m candles")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Source variant: {args.source_variant}")
    print(f"Market symbols preference: {args.market_symbols}")

    trades = load_trades(args.trades_csv, args.source_variant)
    market_symbol, market = load_market_data(args.data_dir, args.market_symbols)
    joined = add_market_features(trades, market)
    accepted, rejected = apply_market_regime(joined, args)

    baseline = joined.copy()
    baseline["position_usd"] = args.position_size
    baseline["profit_usd"] = baseline["position_usd"] * baseline["pnl_pct"] / 100.0

    summary = pd.DataFrame([
        summarize(f"baseline_{args.source_variant}", baseline, args.position_size),
        summarize(f"v49_{market_symbol}_regime", accepted, args.position_size),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_{args.source_variant}_{market_symbol}"
    out_trades = output_dir / f"v49_market_regime_trades_{suffix}.csv"
    out_rejected = output_dir / f"v49_market_regime_rejected_{suffix}.csv"
    out_summary = output_dir / f"v49_market_regime_summary_{suffix}.csv"
    accepted.to_csv(out_trades, index=False)
    rejected.to_csv(out_rejected, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nMarket regime symbol used: {market_symbol}")
    print(f"Saved accepted trades CSV: {out_trades}")
    print(f"Saved rejected trades CSV: {out_rejected}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Strategy comparison ===")
    cols = ["strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl", "profit_usd", "avg_profit_per_trade_usd", "max_drawdown_usd", "max_loss", "max_win"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== v49 stats ===")
    print(f"baseline trades: {len(baseline)}")
    print(f"accepted trades: {len(accepted)}")
    print(f"rejected trades: {len(rejected)}")
    if not rejected.empty:
        print("\n=== Rejection reasons ===")
        print(rejected["v49_reject_reason"].value_counts().to_string())

    if not accepted.empty:
        print("\n=== Market return buckets ===")
        tmp = accepted.copy()
        tmp["market_return_bin"] = pd.cut(tmp["market_return_from_open_pct"], bins=[-999, -1, -0.5, 0, 0.25, 0.5, 1, 999])
        print(tmp.groupby("market_return_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Top days after v49 ===")
        print(tmp.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Worst days after v49 ===")
        print(tmp.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=True).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- If v49 improves avg/median/profit while reducing trades, market regime is useful.")
    print("- If it hurts, long momentum may be more stock-specific than index-dependent in this dataset.")
    print("- Try loosening: --no-require-market-above-vwap or --min-market-return-from-open-pct -0.2")
    print("- Try stricter: --min-market-return-from-open-pct 0.2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
