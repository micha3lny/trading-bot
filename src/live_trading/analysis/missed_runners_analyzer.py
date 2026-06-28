from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_runner_stats,
    first_existing_column,
    fnum,
    iso_ts,
    load_session_candles,
    load_top100,
    load_universe_symbols,
    nearest_row,
    normalize_symbol,
    pct,
    read_sql_table,
    safe_read_csv,
)


DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_UNIVERSE = Path("data/universe/v68_final_daytrading_universe.csv")
DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")

OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "source_bucket",
    "top100_rank",
    "top100_score",
    "open",
    "high",
    "high_time",
    "open_to_high_pct",
    "was_bought",
    "entry_time",
    "entry_price",
    "entry_vs_open_pct",
    "entry_vs_high_pct",
    "missed_reason_group",
    "rejection_reason",
    "blocked_reason",
    "signal_time",
    "ready_since",
    "candidate_age_seconds",
    "live_entry_score",
    "live_entry_rank",
    "spread_bps_near_entry",
    "first_5m_high_pct",
    "first_15m_high_pct",
    "or_range_pct",
]


def load_recorder_table(recorder_dir: Path, session_date: str, names: list[str]) -> pd.DataFrame:
    root = recorder_dir / session_date
    for name in names:
        for path in [root / name, root / f"{name}.csv", root / f"{name}.jsonl"]:
            if path.suffix == ".jsonl" and path.exists():
                try:
                    return pd.read_json(path, lines=True)
                except Exception:
                    return pd.DataFrame()
            df = safe_read_csv(path)
            if not df.empty:
                return df
    return pd.DataFrame()


def load_entries(sqlite_path: str | Path, session_date: str) -> pd.DataFrame:
    trades = read_sql_table(
        sqlite_path,
        "trades",
        where=(
            "session_date = ? OR substr(entry_fill_time, 1, 10) = ? "
            "OR substr(exit_fill_time, 1, 10) = ? OR substr(closed_at, 1, 10) = ?"
        ),
        params=[session_date, session_date, session_date, session_date],
    )
    executions = read_sql_table(
        sqlite_path,
        "executions",
        where="session_date = ? OR substr(executed_at, 1, 10) = ? OR substr(recorded_at, 1, 10) = ?",
        params=[session_date, session_date, session_date],
    )
    rows: list[dict[str, Any]] = []
    if not trades.empty:
        for row in trades.to_dict("records"):
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "entry_time": first_existing_column(row, ["entry_fill_time", "entry_time", "opened_at"]),
                "entry_price": first_existing_column(row, ["entry_price", "avg_price"]),
                "live_entry_score": first_existing_column(row, ["live_entry_score", "entry_score", "score"]),
                "live_entry_rank": first_existing_column(row, ["live_entry_rank", "ranking_position"]),
            })
    if not executions.empty:
        side = executions.get("side", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
        buys = executions[side.isin(["BOT", "BUY", "BOUGHT"])].copy()
        for row in buys.to_dict("records"):
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "entry_time": first_existing_column(row, ["executed_at", "recorded_at"]),
                "entry_price": row.get("price"),
                "live_entry_score": None,
                "live_entry_rank": None,
            })
    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_time", "entry_price", "live_entry_score", "live_entry_rank"])
    out = pd.DataFrame(rows)
    out["entry_ts"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["entry_price_num"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out = out.sort_values(["symbol", "entry_ts"], na_position="last").drop_duplicates("symbol", keep="first")
    return out


def symbol_event_row(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    if df.empty or "symbol" not in df.columns:
        return {}
    rows = df[df["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()


def text_contains(value: Any, needle: str) -> bool:
    return needle.lower() in str(value or "").lower()


def classify_missed_reason(
    *,
    source_bucket: str,
    was_bought: bool,
    entry_time: pd.Timestamp | None,
    high_time: pd.Timestamp | None,
    signal_row: dict[str, Any],
    order_row: dict[str, Any],
) -> str:
    combined = " ".join(str(v) for v in [*signal_row.values(), *order_row.values()] if v not in (None, ""))
    if was_bought:
        if entry_time is not None and high_time is not None and entry_time > high_time:
            return "bought_late"
        return "bought"
    if source_bucket == "outside_top100":
        return "not_in_top100"
    if text_contains(combined, "spread"):
        return "spread_too_wide"
    if text_contains(combined, "risk_guard"):
        return "risk_guard_blocked"
    if text_contains(combined, "max_position") or text_contains(combined, "max positions"):
        return "max_positions_blocked"
    if text_contains(combined, "candidate_age") or text_contains(combined, "stale"):
        return "stale_candidate"
    if text_contains(combined, "failed") or text_contains(combined, "rejected") or text_contains(combined, "error"):
        return "order_failed"
    signal_time = first_existing_column(signal_row, ["signal_time", "ready_since", "timestamp", "event_time"])
    signal_ts = pd.to_datetime(signal_time, errors="coerce", utc=True)
    if high_time is not None and not pd.isna(signal_ts) and signal_ts > high_time:
        return "signal_too_late"
    if not signal_row:
        return "no_signal"
    return "unknown"


def analyze_missed_runners(
    *,
    session_date: str,
    history_dir: Path,
    universe_path: Path,
    top100_path: Path,
    sqlite_path: Path,
    recorder_dir: Path,
    threshold_pct: float,
) -> pd.DataFrame:
    symbols = load_universe_symbols(universe_path)
    top100 = load_top100(top100_path)
    top100_by_symbol = top100.set_index("symbol").to_dict("index") if not top100.empty else {}
    entries = load_entries(sqlite_path, session_date)
    entries_by_symbol = entries.set_index("symbol").to_dict("index") if not entries.empty else {}
    signal_rows = load_recorder_table(recorder_dir, session_date, ["signal_snapshots", "selection_events", "signals"])
    order_rows = load_recorder_table(recorder_dir, session_date, ["order_intents", "orders", "entry_orders"])
    spread_rows = load_recorder_table(recorder_dir, session_date, ["spread_snapshots", "market_snapshots"])

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        candles = load_session_candles(history_dir, symbol, session_date)
        stats = calculate_runner_stats(candles)
        if stats is None or stats.open_to_high_pct < threshold_pct:
            continue
        top_row = top100_by_symbol.get(symbol, {})
        source_bucket = "top100" if top_row else "outside_top100"
        entry = entries_by_symbol.get(symbol, {})
        entry_ts = pd.to_datetime(entry.get("entry_ts"), errors="coerce", utc=True) if entry else pd.NaT
        entry_time = None if pd.isna(entry_ts) else entry_ts
        entry_price = fnum(entry.get("entry_price_num")) if entry else None
        sig = symbol_event_row(signal_rows, symbol)
        order = symbol_event_row(order_rows, symbol)
        spread = nearest_row(spread_rows, entry_time or stats.high_time, symbol)
        was_bought = bool(entry)
        rows.append({
            "date": session_date,
            "symbol": symbol,
            "source_bucket": source_bucket,
            "top100_rank": top_row.get("top100_rank"),
            "top100_score": top_row.get("top100_score"),
            "open": stats.open_price,
            "high": stats.high_price,
            "high_time": iso_ts(stats.high_time),
            "open_to_high_pct": stats.open_to_high_pct,
            "was_bought": int(was_bought),
            "entry_time": iso_ts(entry_time),
            "entry_price": entry_price,
            "entry_vs_open_pct": pct(entry_price, stats.open_price),
            "entry_vs_high_pct": pct(entry_price, stats.high_price),
            "missed_reason_group": classify_missed_reason(
                source_bucket=source_bucket,
                was_bought=was_bought,
                entry_time=entry_time,
                high_time=stats.high_time,
                signal_row=sig,
                order_row=order,
            ),
            "rejection_reason": first_existing_column(order, ["reject_reason", "rejection_reason", "reason", "error"]),
            "blocked_reason": first_existing_column(sig, ["blocked_reason", "entries_blocked_reason", "risk_guard_reason", "reject_reason", "reason"]),
            "signal_time": first_existing_column(sig, ["signal_time", "timestamp", "event_time"]),
            "ready_since": first_existing_column(sig, ["ready_since"]),
            "candidate_age_seconds": first_existing_column(sig, ["candidate_age_seconds"]),
            "live_entry_score": entry.get("live_entry_score") if entry else first_existing_column(sig, ["live_entry_score", "score"]),
            "live_entry_rank": entry.get("live_entry_rank") if entry else first_existing_column(sig, ["live_entry_rank", "ranking_position"]),
            "spread_bps_near_entry": first_existing_column(spread, ["spread_bps", "bid_ask_spread_bps"]),
            "first_5m_high_pct": first_existing_column(sig, ["first_5m_high_pct"]) or stats.first_5m_high_pct,
            "first_15m_high_pct": first_existing_column(sig, ["first_15m_high_pct"]) or stats.first_15m_high_pct,
            "or_range_pct": first_existing_column(sig, ["or_range_pct"]) or stats.or_range_pct,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return out.sort_values("open_to_high_pct", ascending=False)[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    bought = int(pd.to_numeric(df.get("was_bought", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not df.empty else 0
    in_top100 = int((df.get("source_bucket", pd.Series(dtype=str)) == "top100").sum()) if not df.empty else 0
    print(f"MISSED_RUNNERS total_runners={total} in_top100={in_top100} outside_top100={total - in_top100} bought={bought} missed={total - bought}")
    if df.empty:
        return
    reasons = Counter(df["missed_reason_group"].fillna("unknown").astype(str))
    print("reason_counts=" + ", ".join(f"{k}:{v}" for k, v in reasons.most_common()))
    cols = ["symbol", "source_bucket", "open_to_high_pct", "was_bought", "missed_reason_group", "top100_rank", "top100_score"]
    print("top20_by_open_to_high_pct:")
    print(df[cols].head(20).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find >= threshold intraday runners and explain missed or late entries.")
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--threshold-pct", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    top100 = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    output = args.output or Path(f"data/analysis/missed_runners_{args.date}.csv")
    df = analyze_missed_runners(
        session_date=args.date,
        history_dir=args.history_dir,
        universe_path=args.universe,
        top100_path=top100,
        sqlite_path=args.sqlite_path,
        recorder_dir=args.recorder_dir,
        threshold_pct=args.threshold_pct,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print_summary(df)
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

