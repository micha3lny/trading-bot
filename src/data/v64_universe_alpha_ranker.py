from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = "data/1m"
DEFAULT_OUTPUT_DIR = "data/universe"


@dataclass
class RankedSymbol:
    symbol: str
    rows: int
    first_ts: str | None
    last_ts: str | None
    last_close: float | None
    median_dollar_volume: float | None
    avg_dollar_volume: float | None
    median_1m_range_bps: float | None
    p90_1m_range_bps: float | None
    avg_abs_1m_return_bps: float | None
    momentum_day_frequency: float | None
    expansion_bar_frequency: float | None
    positive_followthrough_frequency: float | None
    data_completeness_score: float
    liquidity_score: float
    volatility_score: float
    momentum_score: float
    followthrough_score: float
    junk_penalty: float
    alpha_score: float
    rank_bucket: str
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


def symbol_junk_penalty(symbol: str) -> tuple[float, str]:
    s = symbol.upper().strip()
    reasons: list[str] = []
    penalty = 0.0

    if len(s) >= 5 and s.endswith("W"):
        penalty += 100.0
        reasons.append("warrant_suffix")
    if len(s) >= 5 and s.endswith("U"):
        penalty += 100.0
        reasons.append("unit_suffix")
    if len(s) >= 5 and s.endswith("R"):
        penalty += 100.0
        reasons.append("rights_suffix")
    if len(s) >= 5 and s.endswith("P"):
        penalty += 50.0
        reasons.append("preferred_or_special_suffix")
    if len(s) >= 5 and s[-1] in {"L", "Z", "O", "N"}:
        penalty += 20.0
        reasons.append("possible_note_or_special_class")

    return penalty, ";".join(reasons)


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def scaled(value: float | None, low: float, high: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if high <= low:
        return 0.0
    return clamp((float(value) - low) / (high - low) * 100.0)


def read_symbol(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df = normalize_columns(df)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    return df.sort_values("timestamp").reset_index(drop=True)


def analyze_symbol(path: Path, args: argparse.Namespace) -> RankedSymbol:
    symbol = path.stem.upper()
    df = read_symbol(path)
    junk_penalty, junk_reason = symbol_junk_penalty(symbol)

    if df.empty:
        return RankedSymbol(
            symbol=symbol,
            rows=0,
            first_ts=None,
            last_ts=None,
            last_close=None,
            median_dollar_volume=None,
            avg_dollar_volume=None,
            median_1m_range_bps=None,
            p90_1m_range_bps=None,
            avg_abs_1m_return_bps=None,
            momentum_day_frequency=None,
            expansion_bar_frequency=None,
            positive_followthrough_frequency=None,
            data_completeness_score=0.0,
            liquidity_score=0.0,
            volatility_score=0.0,
            momentum_score=0.0,
            followthrough_score=0.0,
            junk_penalty=max(100.0, junk_penalty),
            alpha_score=0.0,
            rank_bucket="reject",
            reject_reason="empty_or_invalid_csv" + (f";{junk_reason}" if junk_reason else ""),
        )

    recent = df.tail(args.lookback_rows).copy()
    rows = len(df)
    last_close = float(df["close"].iloc[-1])
    first_ts = df["timestamp"].iloc[0].isoformat()
    last_ts = df["timestamp"].iloc[-1].isoformat()

    dollar_volume = recent["close"] * recent["volume"]
    median_dv = float(dollar_volume.median()) if not dollar_volume.empty else None
    avg_dv = float(dollar_volume.mean()) if not dollar_volume.empty else None

    range_bps = ((recent["high"] - recent["low"]) / recent["close"].replace(0, pd.NA) * 10_000).dropna()
    ret_bps = (recent["close"].pct_change() * 10_000).dropna()

    median_range = float(range_bps.median()) if not range_bps.empty else None
    p90_range = float(range_bps.quantile(0.90)) if not range_bps.empty else None
    avg_abs_ret = float(ret_bps.abs().mean()) if not ret_bps.empty else None

    daily = df.set_index("timestamp").resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    daily = daily.tail(args.daily_lookback_days)

    if daily.empty:
        momentum_day_frequency = None
    else:
        daily_range_pct = (daily["high"] / daily["low"].replace(0, pd.NA) - 1.0) * 100.0
        daily_close_from_open = (daily["close"] / daily["open"].replace(0, pd.NA) - 1.0) * 100.0
        momentum_days = (daily_range_pct >= args.momentum_day_range_pct) & (daily_close_from_open >= args.momentum_day_close_pct)
        momentum_day_frequency = float(momentum_days.mean() * 100.0)

    expansion_bar_frequency = None
    if not range_bps.empty:
        expansion_bar_frequency = float((range_bps >= args.expansion_bar_range_bps).mean() * 100.0)

    positive_followthrough_frequency = None
    if len(recent) >= 10:
        tmp = recent.copy()
        tmp["ret_1"] = tmp["close"].pct_change() * 10_000
        tmp["next_3_ret"] = tmp["close"].shift(-3) / tmp["close"] - 1.0
        expansion = tmp["ret_1"] >= args.followthrough_trigger_bps
        sample = tmp[expansion & tmp["next_3_ret"].notna()]
        if len(sample) >= args.min_followthrough_samples:
            positive_followthrough_frequency = float((sample["next_3_ret"] > 0).mean() * 100.0)

    data_completeness_score = scaled(rows, args.min_rows_floor, args.target_rows)
    liquidity_score = 0.65 * scaled(median_dv, args.median_dv_floor, args.median_dv_target) + 0.35 * scaled(avg_dv, args.avg_dv_floor, args.avg_dv_target)
    volatility_score = 0.55 * scaled(median_range, args.median_range_floor_bps, args.median_range_target_bps) + 0.45 * scaled(p90_range, args.p90_range_floor_bps, args.p90_range_target_bps)
    momentum_score = 0.55 * scaled(momentum_day_frequency, 0.0, args.momentum_day_frequency_target) + 0.45 * scaled(expansion_bar_frequency, 0.0, args.expansion_bar_frequency_target)
    followthrough_score = scaled(positive_followthrough_frequency, 40.0, 65.0) if positive_followthrough_frequency is not None else 35.0

    price_penalty = 0.0
    reject_reasons: list[str] = []
    if last_close < args.min_price:
        price_penalty += 100.0
        reject_reasons.append("price_too_low")
    if rows < args.min_rows_floor:
        reject_reasons.append("too_few_rows")
    if median_dv is None or median_dv < args.median_dv_floor:
        reject_reasons.append("median_dollar_volume_too_low")
    if avg_dv is None or avg_dv < args.avg_dv_floor:
        reject_reasons.append("avg_dollar_volume_too_low")
    if median_range is None or median_range < args.median_range_floor_bps:
        reject_reasons.append("range_too_dead")
    if junk_reason:
        reject_reasons.append(junk_reason)

    raw_alpha = (
        args.weight_completeness * data_completeness_score
        + args.weight_liquidity * liquidity_score
        + args.weight_volatility * volatility_score
        + args.weight_momentum * momentum_score
        + args.weight_followthrough * followthrough_score
    )
    alpha = clamp(raw_alpha - junk_penalty - price_penalty)

    if alpha >= args.focus_threshold:
        bucket = "focus"
    elif alpha >= args.tradeable_threshold:
        bucket = "tradeable"
    elif alpha >= args.research_threshold:
        bucket = "research"
    else:
        bucket = "reject"

    return RankedSymbol(
        symbol=symbol,
        rows=rows,
        first_ts=first_ts,
        last_ts=last_ts,
        last_close=last_close,
        median_dollar_volume=median_dv,
        avg_dollar_volume=avg_dv,
        median_1m_range_bps=median_range,
        p90_1m_range_bps=p90_range,
        avg_abs_1m_return_bps=avg_abs_ret,
        momentum_day_frequency=momentum_day_frequency,
        expansion_bar_frequency=expansion_bar_frequency,
        positive_followthrough_frequency=positive_followthrough_frequency,
        data_completeness_score=data_completeness_score,
        liquidity_score=liquidity_score,
        volatility_score=volatility_score,
        momentum_score=momentum_score,
        followthrough_score=followthrough_score,
        junk_penalty=junk_penalty + price_penalty,
        alpha_score=alpha,
        rank_bucket=bucket,
        reject_reason=";".join(reject_reasons),
    )


def write_symbol_file(path: Path, symbols: list[str]) -> None:
    path.write_text("\n".join(symbols) + ("\n" if symbols else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="v64 alpha universe ranker for intraday momentum")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback-rows", type=int, default=2_000)
    parser.add_argument("--daily-lookback-days", type=int, default=90)

    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-rows-floor", type=int, default=12_000)
    parser.add_argument("--target-rows", type=int, default=25_000)
    parser.add_argument("--median-dv-floor", type=float, default=2_500.0)
    parser.add_argument("--median-dv-target", type=float, default=150_000.0)
    parser.add_argument("--avg-dv-floor", type=float, default=10_000.0)
    parser.add_argument("--avg-dv-target", type=float, default=400_000.0)
    parser.add_argument("--median-range-floor-bps", type=float, default=2.0)
    parser.add_argument("--median-range-target-bps", type=float, default=35.0)
    parser.add_argument("--p90-range-floor-bps", type=float, default=8.0)
    parser.add_argument("--p90-range-target-bps", type=float, default=120.0)
    parser.add_argument("--expansion-bar-range-bps", type=float, default=50.0)
    parser.add_argument("--momentum-day-range-pct", type=float, default=4.0)
    parser.add_argument("--momentum-day-close-pct", type=float, default=1.0)
    parser.add_argument("--momentum-day-frequency-target", type=float, default=12.0)
    parser.add_argument("--expansion-bar-frequency-target", type=float, default=5.0)
    parser.add_argument("--followthrough-trigger-bps", type=float, default=20.0)
    parser.add_argument("--min-followthrough-samples", type=int, default=20)

    parser.add_argument("--weight-completeness", type=float, default=0.10)
    parser.add_argument("--weight-liquidity", type=float, default=0.20)
    parser.add_argument("--weight-volatility", type=float, default=0.30)
    parser.add_argument("--weight-momentum", type=float, default=0.30)
    parser.add_argument("--weight-followthrough", type=float, default=0.10)

    parser.add_argument("--focus-threshold", type=float, default=70.0)
    parser.add_argument("--tradeable-threshold", type=float, default=50.0)
    parser.add_argument("--research-threshold", type=float, default=30.0)
    parser.add_argument("--top-focus", type=int, default=300)
    parser.add_argument("--top-tradeable", type=int, default=1200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.csv"))
    print("=== v64 alpha universe ranker ===")
    print(f"Input: {data_dir}")
    print(f"Symbols/files: {len(files)}")
    print(f"Output: {output_dir}")

    if not files:
        print("No symbol CSV files found")
        return 1

    ranked: list[RankedSymbol] = []
    for i, path in enumerate(files, start=1):
        ranked.append(analyze_symbol(path, args))
        if i % 250 == 0:
            print(f"processed {i}/{len(files)}")

    df = pd.DataFrame([r.__dict__ for r in ranked])
    df = df.sort_values(["alpha_score", "momentum_score", "volatility_score", "liquidity_score"], ascending=False)

    rank_path = output_dir / "v64_universe_alpha_ranked.csv"
    focus_path = output_dir / "v64_symbols_focus.txt"
    tradeable_path = output_dir / "v64_symbols_tradeable.txt"
    research_path = output_dir / "v64_symbols_research.txt"

    focus = df[df["rank_bucket"].isin(["focus", "tradeable", "research"])].head(args.top_focus)
    tradeable = df[df["rank_bucket"].isin(["focus", "tradeable", "research"])].head(args.top_tradeable)
    research = df[df["rank_bucket"] != "reject"]

    df.to_csv(rank_path, index=False)
    write_symbol_file(focus_path, focus["symbol"].tolist())
    write_symbol_file(tradeable_path, tradeable["symbol"].tolist())
    write_symbol_file(research_path, research["symbol"].tolist())

    print("\n=== v64 alpha universe summary ===")
    print(f"total_symbols: {len(df)}")
    print(f"focus_top: {len(focus)} -> {focus_path}")
    print(f"tradeable_top: {len(tradeable)} -> {tradeable_path}")
    print(f"research_non_reject: {len(research)} -> {research_path}")
    print("\n=== Buckets ===")
    print(df["rank_bucket"].value_counts().to_string())

    print("\n=== Top 40 alpha ranked ===")
    cols = [
        "symbol", "alpha_score", "rank_bucket", "rows", "last_close", "median_dollar_volume",
        "median_1m_range_bps", "p90_1m_range_bps", "momentum_day_frequency",
        "expansion_bar_frequency", "positive_followthrough_frequency", "liquidity_score",
        "volatility_score", "momentum_score", "followthrough_score",
    ]
    print(df[cols].head(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved ranked universe: {rank_path}")
    print("\nInterpretation hints:")
    print("- focus = small realtime candidate pool; tradeable = broad live scan pool; research = wider non-junk pool.")
    print("- This ranker favors intraday expansion/momentum behavior, not just megacap liquidity.")
    print("- Next step: backtest v64_symbols_focus/tradeable against the old wide/liquid lists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
