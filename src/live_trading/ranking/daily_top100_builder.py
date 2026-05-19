from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.ranking.ranking_store import RankingStore


DEFAULT_UNIVERSE = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_HISTORY_DIR = "data/history/universe_1m"
DEFAULT_SQLITE_PATH = "data/runtime/rankings.sqlite"
MIN_LATEST_ROWS = 100


@dataclass(frozen=True)
class SymbolRanking:
    symbol: str
    score: float
    last_close: float
    dollar_volume: float
    day_return_pct: float
    intraday_high_pct: float
    range_pct: float
    volume: float
    gap_pct: float | None
    median_1m_range_bps: float
    avg_abs_1m_return_bps: float
    multi_day_return_pct: float | None
    components: dict[str, Any]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def scaled(value: float | None, low: float, high: float) -> float:
    if value is None or pd.isna(value) or high <= low:
        return 0.0
    return clamp((float(value) - low) / (high - low) * 100.0)


def scaled_log(value: float | None, low: float, high: float) -> float:
    if value is None or value <= 0 or low <= 0 or high <= low:
        return 0.0
    return scaled(math.log10(float(value)), math.log10(low), math.log10(high))


def load_universe(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing universe file: {p}")
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("Universe CSV must contain symbol column")
    if "alpha_score" in df.columns:
        df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("alpha_score", ascending=False)
    symbols = df["symbol"].astype(str).str.upper().str.strip().dropna().drop_duplicates().tolist()
    return [s for s in symbols if s and s != "NAN"]


def parquet_path(history_dir: str | Path, symbol: str, session_date: date, session_type: str = "RTH") -> Path:
    root = Path(history_dir)
    return (
        root
        / f"session_type={session_type.upper()}"
        / f"symbol={symbol.upper()}"
        / f"year={session_date.year:04d}"
        / f"month={session_date.month:02d}"
        / f"day={session_date.day:02d}.parquet"
    )


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "timestamp" not in out.columns and "bar_time_utc" in out.columns:
        out = out.rename(columns={"bar_time_utc": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(out.columns):
        missing = ",".join(sorted(required - set(out.columns)))
        raise ValueError(f"history parquet missing required columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume", "wap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out[(out["open"] > 0) & (out["high"] >= out["low"]) & (out["close"] > 0) & (out["volume"] >= 0)]
    return out.sort_values("timestamp").reset_index(drop=True)


def read_session(history_dir: str | Path, symbol: str, session_date: date, session_type: str = "RTH") -> pd.DataFrame:
    path = parquet_path(history_dir, symbol, session_date, session_type)
    if not path.exists():
        return pd.DataFrame()
    return normalize_history_df(pd.read_parquet(path))


def parse_partition_date(path: Path) -> date | None:
    try:
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
        day_value = str(parts.get("day") or path.stem).replace(".parquet", "")
        return date(int(parts["year"]), int(parts["month"]), int(day_value))
    except Exception:
        return None


def prior_session_paths(history_dir: str | Path, symbol: str, session_date: date, session_type: str = "RTH") -> list[Path]:
    symbol_dir = Path(history_dir) / f"session_type={session_type.upper()}" / f"symbol={symbol.upper()}"
    dated: list[tuple[date, Path]] = []
    for path in symbol_dir.glob("year=*/month=*/day=*.parquet"):
        parsed = parse_partition_date(path)
        if parsed is not None and parsed < session_date:
            dated.append((parsed, path))
    return [p for _, p in sorted(dated, reverse=True)]


def recent_prior_closes(history_dir: str | Path, symbol: str, session_date: date, limit: int, session_type: str = "RTH") -> list[float]:
    closes: list[float] = []
    for path in prior_session_paths(history_dir, symbol, session_date, session_type)[: max(0, limit)]:
        try:
            df = normalize_history_df(pd.read_parquet(path))
        except Exception:
            continue
        if not df.empty:
            closes.append(float(df["close"].iloc[-1]))
    return closes


def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    prior_closes: list[float],
    *,
    min_price: float,
    min_bars: int,
    min_volume: float,
    min_dollar_volume: float,
) -> tuple[SymbolRanking | None, str | None]:
    if df.empty:
        return None, "missing_history"
    if len(df) < min_bars:
        return None, f"too_few_bars:{len(df)}"

    open_price = float(df["open"].iloc[0])
    high = float(df["high"].max())
    low = float(df["low"].min())
    last_close = float(df["close"].iloc[-1])
    volume = float(df["volume"].sum())
    dollar_price = df["wap"] if "wap" in df.columns and df["wap"].notna().any() else df["close"]
    dollar_volume = float((dollar_price.fillna(df["close"]) * df["volume"]).sum())

    if last_close < min_price:
        return None, f"price_too_low:{last_close:.2f}"
    if volume < min_volume:
        return None, f"volume_too_low:{volume:.0f}"
    if dollar_volume < min_dollar_volume:
        return None, f"dollar_volume_too_low:{dollar_volume:.0f}"

    day_return_pct = (last_close / open_price - 1.0) * 100.0
    intraday_high_pct = (high / open_price - 1.0) * 100.0
    range_pct = (high / low - 1.0) * 100.0 if low > 0 else 0.0
    range_bps = ((df["high"] - df["low"]) / df["close"].replace(0, pd.NA) * 10_000).dropna()
    ret_bps = (df["close"].pct_change() * 10_000).dropna()
    median_1m_range_bps = float(range_bps.median()) if not range_bps.empty else 0.0
    avg_abs_1m_return_bps = float(ret_bps.abs().mean()) if not ret_bps.empty else 0.0

    prior_close = prior_closes[0] if prior_closes else None
    gap_pct = (open_price / prior_close - 1.0) * 100.0 if prior_close and prior_close > 0 else None
    multi_day_return_pct = None
    if len(prior_closes) >= 2 and prior_closes[-1] > 0:
        multi_day_return_pct = (last_close / prior_closes[-1] - 1.0) * 100.0

    components = {
        "intraday_high": scaled(intraday_high_pct, 0.5, 15.0),
        "day_return": scaled(day_return_pct, -2.0, 10.0),
        "liquidity": scaled_log(dollar_volume, min_dollar_volume, 75_000_000.0),
        "range": scaled(range_pct, 1.0, 14.0),
        "median_1m_range": scaled(median_1m_range_bps, 2.0, 80.0),
        "gap": scaled(gap_pct, -5.0, 10.0) if gap_pct is not None else 35.0,
        "multi_day": scaled(multi_day_return_pct, -10.0, 25.0) if multi_day_return_pct is not None else 35.0,
    }
    score = (
        0.30 * components["intraday_high"]
        + 0.20 * components["day_return"]
        + 0.20 * components["liquidity"]
        + 0.10 * components["range"]
        + 0.10 * components["median_1m_range"]
        + 0.05 * components["gap"]
        + 0.05 * components["multi_day"]
    )
    components.update(
        {
            "bars": len(df),
            "prior_close": prior_close,
            "weights": {
                "intraday_high": 0.30,
                "day_return": 0.20,
                "liquidity": 0.20,
                "range": 0.10,
                "median_1m_range": 0.10,
                "gap": 0.05,
                "multi_day": 0.05,
            },
        }
    )

    return (
        SymbolRanking(
            symbol=symbol,
            score=round(float(score), 4),
            last_close=last_close,
            dollar_volume=dollar_volume,
            day_return_pct=day_return_pct,
            intraday_high_pct=intraday_high_pct,
            range_pct=range_pct,
            volume=volume,
            gap_pct=gap_pct,
            median_1m_range_bps=median_1m_range_bps,
            avg_abs_1m_return_bps=avg_abs_1m_return_bps,
            multi_day_return_pct=multi_day_return_pct,
            components=components,
        ),
        None,
    )


def ranking_to_row(rank: int, item: SymbolRanking) -> dict[str, Any]:
    components_json = json.dumps(item.components, sort_keys=True)
    return {
        "rank": rank,
        "symbol": item.symbol,
        "score": item.score,
        "alpha_score": item.score,
        "last_close": round(item.last_close, 4),
        "dollar_volume": round(item.dollar_volume, 2),
        "day_return_pct": round(item.day_return_pct, 4),
        "intraday_high_pct": round(item.intraday_high_pct, 4),
        "range_pct": round(item.range_pct, 4),
        "volume": round(item.volume, 2),
        "gap_pct": round(item.gap_pct, 4) if item.gap_pct is not None else "",
        "median_1m_range_bps": round(item.median_1m_range_bps, 4),
        "avg_abs_1m_return_bps": round(item.avg_abs_1m_return_bps, 4),
        "multi_day_return_pct": round(item.multi_day_return_pct, 4) if item.multi_day_return_pct is not None else "",
        "reason": "ranked",
        "components_json": components_json,
    }


def build_daily_top(
    *,
    ranking_date: date,
    universe_path: str | Path,
    history_dir: str | Path,
    top_n: int,
    session_type: str,
    min_price: float,
    min_bars: int,
    min_volume: float,
    min_dollar_volume: float,
    prior_sessions: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    symbols = load_universe(universe_path)
    print(
        f"DAILY_TOP100_START date={ranking_date.isoformat()} symbols={len(symbols)} "
        f"history_dir={history_dir} session_type={session_type}",
        flush=True,
    )
    ranked: list[SymbolRanking] = []
    stats: dict[str, int] = {"symbols": len(symbols), "valid": 0, "missing": 0, "rejected": 0, "errors": 0}
    for idx, symbol in enumerate(symbols, 1):
        try:
            df = read_session(history_dir, symbol, ranking_date, session_type)
            if df.empty:
                stats["missing"] += 1
                print(f"DAILY_TOP100_MISSING_DATA symbol={symbol} date={ranking_date.isoformat()}", flush=True)
                continue
            prior_closes = recent_prior_closes(history_dir, symbol, ranking_date, prior_sessions, session_type)
            item, reject_reason = analyze_symbol(
                symbol,
                df,
                prior_closes,
                min_price=min_price,
                min_bars=min_bars,
                min_volume=min_volume,
                min_dollar_volume=min_dollar_volume,
            )
            if item is None:
                stats["rejected"] += 1
                print(f"DAILY_TOP100_REJECTED symbol={symbol} reason={reject_reason}", flush=True)
                continue
            ranked.append(item)
            stats["valid"] += 1
        except Exception as exc:
            stats["errors"] += 1
            print(f"DAILY_TOP100_SYMBOL_ERROR symbol={symbol} error={exc!r}", flush=True)
        if idx % 250 == 0:
            print(f"DAILY_TOP100_PROGRESS processed={idx} valid={stats['valid']}", flush=True)

    ranked.sort(key=lambda item: (item.score, item.dollar_volume, item.intraday_high_pct), reverse=True)
    rows = [ranking_to_row(rank, item) for rank, item in enumerate(ranked[: max(0, top_n)], 1)]
    return rows, stats


CSV_COLUMNS = [
    "rank",
    "symbol",
    "score",
    "alpha_score",
    "last_close",
    "dollar_volume",
    "day_return_pct",
    "intraday_high_pct",
    "range_pct",
    "volume",
    "gap_pct",
    "median_1m_range_bps",
    "avg_abs_1m_return_bps",
    "multi_day_return_pct",
    "reason",
    "components_json",
]


def render_output_csv(rows: list[dict[str, Any]]) -> str:
    return pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(index=False)


def write_text_atomic(path: str | Path, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(output)


def write_output_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_text_atomic(path, render_output_csv(rows))


def update_latest_output(dated_output: str | Path, latest_output: str | Path, rows: list[dict[str, Any]]) -> bool:
    if len(rows) < MIN_LATEST_ROWS:
        print(
            f"DAILY_TOP100_LATEST_SKIPPED reason=too_few_rows rows={len(rows)} "
            f"required={MIN_LATEST_ROWS} latest_output={latest_output}",
            flush=True,
        )
        return False

    dated = Path(dated_output)
    latest = Path(latest_output)
    content = dated.read_text(encoding="utf-8") if dated.exists() else render_output_csv(rows)
    write_text_atomic(latest, content)
    print(
        f"DAILY_TOP100_LATEST_UPDATED latest_output={latest} rows={len(rows)}",
        flush=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v67/v68 daily Top100 CSV from collected 1m parquet history")
    parser.add_argument("--date", required=True, help="RTH session date to rank, e.g. 2026-05-15")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latest-output", default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-bars", type=int, default=180)
    parser.add_argument("--min-volume", type=float, default=100_000.0)
    parser.add_argument("--min-dollar-volume", type=float, default=500_000.0)
    parser.add_argument("--prior-sessions", type=int, default=5)
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--no-sqlite", action="store_true")
    args = parser.parse_args()

    ranking_date = parse_date(args.date)
    rows, stats = build_daily_top(
        ranking_date=ranking_date,
        universe_path=args.universe,
        history_dir=args.history_dir,
        top_n=int(args.top_n),
        session_type=str(args.session_type).upper(),
        min_price=float(args.min_price),
        min_bars=int(args.min_bars),
        min_volume=float(args.min_volume),
        min_dollar_volume=float(args.min_dollar_volume),
        prior_sessions=int(args.prior_sessions),
    )
    write_output_csv(args.output, rows)
    latest_ok = None
    if args.latest_output:
        latest_ok = update_latest_output(args.output, args.latest_output, rows)
    stored = 0
    if not args.no_sqlite:
        stored = RankingStore(args.sqlite_path).replace_daily_rankings(ranking_date.isoformat(), rows)
    print(
        f"DAILY_TOP100_DONE date={ranking_date.isoformat()} output={args.output} rows={len(rows)} "
        f"valid={stats['valid']} missing={stats['missing']} rejected={stats['rejected']} "
        f"errors={stats['errors']} sqlite_rows={stored}",
        flush=True,
    )
    if len(rows) < int(args.top_n):
        print(f"DAILY_TOP100_WARNING requested_top_n={args.top_n} produced={len(rows)}", flush=True)
    if latest_ok is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
