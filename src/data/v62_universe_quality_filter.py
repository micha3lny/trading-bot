from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = "data/1m"
DEFAULT_OUTPUT_DIR = "data/universe"

BAD_SUFFIXES = (
    "W", "WS", "WT", "WTS", "U", "UN", "UNIT", "R", "RT", "RIGHT", "P", "PR",
)
BAD_SUBSTRINGS = (
    "WARRANT", "RIGHT", "UNIT",
)


@dataclass
class SymbolQuality:
    symbol: str
    rows: int
    first_ts: str | None
    last_ts: str | None
    last_close: float | None
    avg_dollar_volume: float | None
    median_dollar_volume: float | None
    keep: bool
    reject_reason: str


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    rename = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "bar_time": "timestamp",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def infer_timestamp_col(df: pd.DataFrame) -> str | None:
    for col in ["timestamp", "date", "datetime", "time"]:
        if col in df.columns:
            return col
    return None


def is_obviously_junk_symbol(symbol: str) -> tuple[bool, str]:
    s = symbol.upper().strip()

    # Common NASDAQ special issue patterns: warrants, units, rights.
    if len(s) >= 5 and s.endswith("W"):
        return True, "warrant_suffix"
    if len(s) >= 5 and s.endswith("U"):
        return True, "unit_suffix"
    if len(s) >= 5 and s.endswith("R"):
        return True, "rights_suffix"

    for suffix in ["WW", "WS", "WQ", "WZ", "WT", "WTS", "U", "R"]:
        if len(s) > 4 and s.endswith(suffix):
            return True, f"special_suffix_{suffix}"

    # Preferred shares / notes are not useful for this momentum bot.
    if s.endswith("P") and len(s) >= 5:
        return True, "preferred_or_special_suffix"

    return False, ""


def read_symbol_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return normalize_columns(df)


def quality_check(path: Path, args: argparse.Namespace) -> tuple[SymbolQuality, pd.DataFrame]:
    symbol = path.stem.upper()
    junk, junk_reason = is_obviously_junk_symbol(symbol)

    df = read_symbol_csv(path)
    if df.empty:
        return SymbolQuality(symbol, 0, None, None, None, None, None, False, "empty_csv"), df

    ts_col = infer_timestamp_col(df)
    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = len(df)
    last_close = float(df["close"].dropna().iloc[-1]) if "close" in df.columns and not df["close"].dropna().empty else None

    avg_dollar_volume = None
    median_dollar_volume = None
    if "close" in df.columns and "volume" in df.columns:
        dv = (df["close"] * df["volume"]).replace([float("inf"), -float("inf")], pd.NA).dropna()
        if not dv.empty:
            avg_dollar_volume = float(dv.tail(args.dollar_volume_lookback_rows).mean())
            median_dollar_volume = float(dv.tail(args.dollar_volume_lookback_rows).median())

    first_ts = None
    last_ts = None
    if ts_col is not None and not df.empty:
        first_ts = df[ts_col].iloc[0].isoformat()
        last_ts = df[ts_col].iloc[-1].isoformat()

    reasons: list[str] = []
    if junk:
        reasons.append(junk_reason)
    if rows < args.min_rows:
        reasons.append("too_few_rows")
    if last_close is None or last_close < args.min_last_price:
        reasons.append("price_too_low")
    if avg_dollar_volume is None or avg_dollar_volume < args.min_avg_dollar_volume:
        reasons.append("avg_dollar_volume_too_low")
    if median_dollar_volume is None or median_dollar_volume < args.min_median_dollar_volume:
        reasons.append("median_dollar_volume_too_low")

    keep = len(reasons) == 0
    return SymbolQuality(
        symbol=symbol,
        rows=rows,
        first_ts=first_ts,
        last_ts=last_ts,
        last_close=last_close,
        avg_dollar_volume=avg_dollar_volume,
        median_dollar_volume=median_dollar_volume,
        keep=keep,
        reject_reason=";".join(reasons),
    ), df


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ts_col = infer_timestamp_col(df)
    required = {"open", "high", "low", "close", "volume"}
    if ts_col is None or not required.issubset(df.columns):
        return pd.DataFrame()

    tmp = df.copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    agg = tmp.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    agg = agg.rename(columns={ts_col: "timestamp"})
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="v62 universe quality filter + 5m/15m builder from 1m candles")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-rows", type=int, default=20_000)
    parser.add_argument("--min-last-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume", type=float, default=50_000.0)
    parser.add_argument("--min-median-dollar-volume", type=float, default=10_000.0)
    parser.add_argument("--dollar-volume-lookback-rows", type=int, default=2_000)
    parser.add_argument("--build-timeframes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeframes", nargs="*", default=["5min", "15min"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== v62 universe quality filter ===")
    print(f"Input: {data_dir}")
    print(f"Output: {out_dir}")
    print(f"Build timeframes: {args.build_timeframes} {args.timeframes}")

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        print("No 1m csv files found")
        return 1

    qualities: list[SymbolQuality] = []
    kept_symbols: list[str] = []

    timeframe_dirs: dict[str, Path] = {}
    if args.build_timeframes:
        for tf in args.timeframes:
            tf_dir = Path(f"data/{tf}")
            tf_dir.mkdir(parents=True, exist_ok=True)
            timeframe_dirs[tf] = tf_dir

    for i, path in enumerate(files, start=1):
        quality, df = quality_check(path, args)
        qualities.append(quality)
        if quality.keep:
            kept_symbols.append(quality.symbol)
            if args.build_timeframes and not df.empty:
                for tf, tf_dir in timeframe_dirs.items():
                    out = tf_dir / f"{quality.symbol}.csv"
                    resampled = resample_ohlcv(df, tf)
                    if not resampled.empty:
                        resampled.to_csv(out, index=False)
        if i % 250 == 0:
            print(f"processed {i}/{len(files)} kept={len(kept_symbols)}")

    rows = [q.__dict__ for q in qualities]
    quality_df = pd.DataFrame(rows).sort_values(["keep", "avg_dollar_volume"], ascending=[False, False])
    accepted = quality_df[quality_df["keep"]].copy()
    rejected = quality_df[~quality_df["keep"]].copy()

    quality_path = out_dir / "v62_universe_quality.csv"
    accepted_path = out_dir / "v62_universe_accepted.csv"
    rejected_path = out_dir / "v62_universe_rejected.csv"
    symbols_path = out_dir / "v62_symbols_clean.txt"

    quality_df.to_csv(quality_path, index=False)
    accepted.to_csv(accepted_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    symbols_path.write_text("\n".join(accepted["symbol"].astype(str).tolist()) + "\n")

    print("\n=== Universe quality summary ===")
    print(f"total_symbols: {len(quality_df)}")
    print(f"accepted: {len(accepted)}")
    print(f"rejected: {len(rejected)}")
    if not rejected.empty:
        print("\n=== Top rejection reasons ===")
        print(rejected["reject_reason"].value_counts().head(20).to_string())
    if not accepted.empty:
        print("\n=== Top accepted by avg dollar volume ===")
        cols = ["symbol", "rows", "last_close", "avg_dollar_volume", "median_dollar_volume"]
        print(accepted[cols].head(25).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved quality: {quality_path}")
    print(f"Saved accepted: {accepted_path}")
    print(f"Saved rejected: {rejected_path}")
    print(f"Saved clean symbols: {symbols_path}")
    if args.build_timeframes:
        for tf, tf_dir in timeframe_dirs.items():
            print(f"Built {tf}: {tf_dir}")

    print("\nInterpretation hints:")
    print("- Use v62_symbols_clean.txt as the live watchlist/backtest universe.")
    print("- 5m/15m candles are derived from 1m candles, so we avoid duplicate IBKR downloads.")
    print("- Rejected symbols are not necessarily bad companies; they are bad inputs for this intraday momentum bot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
