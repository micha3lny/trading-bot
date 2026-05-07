from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SNAPSHOTS = "data/live/live_signal_snapshots.csv"
DEFAULT_OUTPUT = "data/live/v61_flow_signals.csv"


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def prepare_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    for col in ["reference_price", "last", "mid", "bid", "ask", "volume", "spread_bps"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["price"] = out["reference_price"]
    if "price" not in out or out["price"].isna().all():
        out["price"] = out.get("last")
    out = out.dropna(subset=["timestamp_utc", "symbol", "price"])
    out = out[out["price"] > 0]
    return out.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def add_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("timestamp_utc").copy()
        g["price_change_1"] = g["price"].pct_change() * 100.0
        g["momentum_3"] = g["price"].pct_change(3) * 100.0
        g["momentum_6"] = g["price"].pct_change(6) * 100.0
        g["momentum_acceleration"] = g["momentum_3"] - g["momentum_6"].fillna(g["momentum_3"])

        if "volume" in g.columns:
            g["volume_delta"] = g["volume"].diff().clip(lower=0)
            rolling_volume = g["volume_delta"].rolling(12, min_periods=3).mean()
            g["relative_volume"] = g["volume_delta"] / rolling_volume.replace(0, pd.NA)
        else:
            g["volume_delta"] = pd.NA
            g["relative_volume"] = pd.NA

        if "spread_bps" in g.columns:
            rolling_spread = g["spread_bps"].rolling(12, min_periods=3).mean()
            g["spread_compression"] = rolling_spread - g["spread_bps"]
        else:
            g["spread_compression"] = pd.NA

        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else df


def add_relative_strength(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if df.empty:
        return df
    bench = df[df["symbol"] == benchmark].copy()
    if bench.empty:
        df["benchmark_return_6"] = pd.NA
        df["relative_strength_6"] = pd.NA
        return df

    bench = bench[["timestamp_utc", "momentum_6"]].rename(columns={"momentum_6": "benchmark_return_6"})
    out = df.merge(bench, on="timestamp_utc", how="left")
    out["relative_strength_6"] = out["momentum_6"] - out["benchmark_return_6"]
    return out


def score_flow(row: pd.Series) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    rv = row.get("relative_volume")
    rs = row.get("relative_strength_6")
    accel = row.get("momentum_acceleration")
    mom3 = row.get("momentum_3")
    spread_comp = row.get("spread_compression")
    spread_bps = row.get("spread_bps")

    if pd.notna(rv):
        if rv >= 4.0:
            score += 3.0
            reasons.append("relative_volume_extreme")
        elif rv >= 2.0:
            score += 2.0
            reasons.append("relative_volume_high")
        elif rv >= 1.3:
            score += 1.0
            reasons.append("relative_volume_above_normal")

    if pd.notna(rs):
        if rs >= 1.0:
            score += 2.0
            reasons.append("strong_relative_strength")
        elif rs >= 0.3:
            score += 1.0
            reasons.append("positive_relative_strength")

    if pd.notna(accel):
        if accel >= 0.5:
            score += 2.0
            reasons.append("momentum_acceleration_strong")
        elif accel >= 0.15:
            score += 1.0
            reasons.append("momentum_acceleration_positive")

    if pd.notna(mom3):
        if mom3 >= 1.0:
            score += 2.0
            reasons.append("short_momentum_strong")
        elif mom3 >= 0.3:
            score += 1.0
            reasons.append("short_momentum_positive")

    if pd.notna(spread_comp) and spread_comp > 0:
        score += 0.5
        reasons.append("spread_compressing")

    if pd.notna(spread_bps):
        if spread_bps <= 5.0:
            score += 1.0
            reasons.append("liquid_spread")
        elif spread_bps > 20.0:
            score -= 1.0
            reasons.append("wide_spread_penalty")

    return max(0.0, score), reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="v61 flow signal layer")
    parser.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", default="QQQ")
    args = parser.parse_args()

    print("=== v61 flow signal layer ===")
    print(f"Snapshots: {args.snapshots}")
    print(f"Benchmark: {args.benchmark}")

    snapshots = prepare_snapshots(read_csv(args.snapshots))
    if snapshots.empty:
        print("No usable snapshots found")
        return 1

    features = add_symbol_features(snapshots)
    features = add_relative_strength(features, args.benchmark)

    scored = []
    for _, row in features.iterrows():
        score, reasons = score_flow(row)
        out = row.to_dict()
        out["flow_score"] = score
        out["flow_reasons"] = ";".join(reasons)
        scored.append(out)

    result = pd.DataFrame(scored)

    output_cols = [
        "timestamp_utc",
        "symbol",
        "price",
        "volume",
        "volume_delta",
        "relative_volume",
        "spread_bps",
        "spread_compression",
        "momentum_3",
        "momentum_6",
        "momentum_acceleration",
        "benchmark_return_6",
        "relative_strength_6",
        "flow_score",
        "flow_reasons",
    ]
    output_cols = [c for c in output_cols if c in result.columns]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result[output_cols].to_csv(out, index=False)

    latest = result.sort_values("timestamp_utc").groupby("symbol").tail(1).copy()
    latest = latest.sort_values("flow_score", ascending=False)

    print("\n=== Latest flow signals ===")
    display_cols = [
        "symbol",
        "price",
        "relative_volume",
        "spread_bps",
        "momentum_3",
        "momentum_6",
        "relative_strength_6",
        "flow_score",
        "flow_reasons",
    ]
    display_cols = [c for c in display_cols if c in latest.columns]
    print(latest[display_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved: {out}")
    print("\nInterpretation hints:")
    print("- flow_score is an additive signal, not a standalone buy trigger.")
    print("- Best use: boost existing momentum/breakout score only when liquidity is acceptable.")
    print("- Next step: merge flow_score into v58 order-intent scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
