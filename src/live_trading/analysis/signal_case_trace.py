from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    LIVE_SIGNAL_MIN_FIRST_15M_HIGH_PCT,
    LIVE_SIGNAL_MIN_FIRST_5M_HIGH_PCT,
    LIVE_SIGNAL_MIN_OR_RANGE_PCT,
    calculate_runner_stats,
    fnum,
    iso_ts,
    load_session_candles,
    load_top100,
    load_universe_symbols,
    normalize_symbol,
    parse_dt,
    read_sql_table,
)
from src.live_trading.analysis.missed_runners_analyzer import first_time_above, no_signal_diagnostics
from src.live_trading.analysis.signal_replay_analyzer import (
    build_symbol_timeline,
    classify_replay_reason,
    first_matching_reason,
    has_event,
    merge_timeline_events,
    read_sqlite_sources,
    recorder_events,
    row_event_type,
    row_reason,
    row_text,
    row_time,
    symbol_rows,
)
from src.live_trading.ranking.daily_top100_builder import parquet_path
from src.live_trading.analysis.strategy_config_parity import add_threshold_cli, resolve_threshold_args


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")
DEFAULT_UNIVERSE = Path("data/universe/v68_final_daytrading_universe.csv")
DEFAULT_TRADER_SOURCE = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")

TRACE_TERMS = [
    "SIGNAL_READY",
    "BUY_BLOCKED",
    "RISK_GUARD_BLOCK_ENTRY",
    "ready_since",
    "signal_sent",
    "first_5m",
    "first_15m",
    "or_range",
    "top100",
    "restart_block",
    "max_positions",
    "cooldown",
    "placeOrder",
    "MarketOrder",
]


def pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def source_time_bounds(df: pd.DataFrame) -> tuple[str, str]:
    times = [row_time(row) for row in df.to_dict("records")] if not df.empty else []
    times = [ts for ts in times if ts is not None]
    if not times:
        return "", ""
    return iso_ts(min(times)), iso_ts(max(times))


def event_lines(df: pd.DataFrame, *, symbol: str, center: pd.Timestamp | None, window_minutes: int = 30, limit: int = 25) -> list[str]:
    rows = symbol_rows(df, symbol, center, window_minutes=window_minutes)
    lines: list[str] = []
    for row in rows.to_dict("records")[:limit]:
        ts = iso_ts(row_time(row))
        event = row_event_type(row)
        reason = row_reason(row)
        snippet = row_text(row).replace("\n", " ")[:220]
        lines.append(f"  - {ts} event={event} reason={reason} raw={snippet}")
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more rows omitted")
    return lines


def static_line_refs(path: Path = DEFAULT_TRADER_SOURCE) -> list[str]:
    if not path.exists():
        return [f"{path}: MISSING"]
    refs: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [f"{path}: read_failed={exc!r}"]
    for idx, line in enumerate(lines, start=1):
        lower = line.lower()
        for term in TRACE_TERMS:
            if term.lower() in lower:
                refs.append(f"{path}:{idx}: {line.strip()[:180]}")
                break
    return refs


def decision_classification(offline_ready: bool, timeline: list[dict[str, Any]], counts: dict[str, int]) -> str:
    if not offline_ready:
        return "offline_signal_not_ready"
    replay = classify_replay_reason(timeline, counts)
    mapping = {
        "bought_but_missing_in_missed_runner": "bought",
        "buy_attempt_no_fill": "buy_attempt_no_fill",
        "risk_guard_blocked": "buy_blocked_risk_guard",
        "max_positions_blocked": "buy_blocked_max_positions",
        "restart_blocked": "buy_blocked_restart",
        "top100_blocked": "buy_blocked_top100",
        "cooldown_blocked": "buy_blocked_cooldown",
        "signal_ready_but_no_buy_attempt": "runtime_signal_ready_but_no_buy",
        "no_runtime_evidence": "runtime_no_evidence_for_symbol",
        "runtime_saw_symbol_no_signal": "runtime_saw_symbol_but_no_signal",
    }
    if replay in mapping:
        return mapping[replay]
    if replay in {"contract_or_order_error", "symbol_ineligible", "stale_candidate", "duplicate_position_blocked", "no_market_data"}:
        return replay
    return "unknown"


def load_symbol_traces(sqlite_path: Path, recorder_dir: Path, session_date: str, symbol: str, center: pd.Timestamp | None) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, int]]:
    sqlite_sources = read_sqlite_sources(sqlite_path, session_date)
    recorder_sources = recorder_events(recorder_dir, session_date, symbol, center)
    timeline, raw_counts = build_symbol_timeline(
        row={"symbol": symbol, "open_to_high_pct": ""},
        sqlite_sources=sqlite_sources,
        recorder_sources=recorder_sources,
        center=center,
    )
    counts = {
        "runtime_events_count": raw_counts.get("runtime_events_count", 0),
        "risk_events_count": raw_counts.get("risk_events_count", 0),
        "orders_count": raw_counts.get("orders_count", 0) + raw_counts.get("order_lifecycle_count", 0),
        "trades_count": raw_counts.get("trades_count", 0) + raw_counts.get("trade_lifecycle_count", 0),
        "executions_count": raw_counts.get("executions_count", 0) + raw_counts.get("fills_count", 0),
    }
    return sqlite_sources, recorder_sources, timeline, counts


def trace_signal_case(
    *,
    session_date: str,
    symbol: str,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    top100_path: Path,
    universe_path: Path = DEFAULT_UNIVERSE,
    min_first_5m_high_pct: float = LIVE_SIGNAL_MIN_FIRST_5M_HIGH_PCT,
    min_first_15m_high_pct: float = LIVE_SIGNAL_MIN_FIRST_15M_HIGH_PCT,
    min_or_range_pct: float = LIVE_SIGNAL_MIN_OR_RANGE_PCT,
    config_source: str = "programmatic_explicit_or_historical_default",
) -> str:
    symbol = normalize_symbol(symbol)
    top100 = load_top100(top100_path)
    top_row = {}
    if not top100.empty:
        match = top100[top100["symbol"].map(normalize_symbol) == symbol]
        if not match.empty:
            top_row = match.iloc[0].to_dict()
    universe_symbols = set(load_universe_symbols(universe_path))
    candles = load_session_candles(history_dir, symbol, session_date, "RTH")
    ppath = parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), "RTH")
    stats = calculate_runner_stats(candles)
    diag = no_signal_diagnostics(
        candles,
        min_first_5m_high_pct=min_first_5m_high_pct,
        min_first_15m_high_pct=min_first_15m_high_pct,
        min_or_range_pct=min_or_range_pct,
    )
    center = parse_dt(diag.get("possible_signal_time") or diag.get("opening_range_break_time") or diag.get("first_time_above_8pct"))
    sqlite_sources, recorder_sources, timeline, counts = load_symbol_traces(sqlite_path, recorder_dir, session_date, symbol, center)
    top100_ok = bool(top_row)
    first5_ok = bool(diag.get("had_required_first5"))
    first15_ok = bool(diag.get("had_required_first15"))
    or_ok = bool(diag.get("had_required_or_range"))
    breakout_ok = bool(diag.get("did_break_or_high"))
    breakout_gate_used = bool(diag.get("breakout_gate_used"))
    offline_ready = top100_ok and first5_ok and first15_ok and or_ok and bool(diag.get("possible_signal_time"))
    decision = decision_classification(offline_ready, timeline, counts)
    block_reason = first_matching_reason(timeline, ["BUY_BLOCKED", "RISK_GUARD", "MAX_POSITION", "RESTART_BLOCK", "TOP100_BLOCK", "COOLDOWN", "STALE", "NO_MARKET_DATA", "INELIGIBLE"])

    lines: list[str] = []
    lines.append(f"SIGNAL CASE TRACE date={session_date} symbol={symbol}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("1. Universe / Top100")
    lines.append(f"- in_universe={pass_fail(symbol in universe_symbols) if universe_symbols else 'UNKNOWN'} universe_path={universe_path}")
    lines.append(f"- in_top100={pass_fail(top100_ok)} top100_path={top100_path}")
    lines.append(f"- top100_rank={fmt(top_row.get('top100_rank'))} top100_score={fmt(top_row.get('top100_score'))}")
    lines.append(f"- top100_row={json.dumps(top_row, default=str, ensure_ascii=True)[:1000] if top_row else '{}'}")
    lines.append("")
    lines.append("2. Candle / History")
    lines.append(f"- parquet_path={ppath}")
    lines.append(f"- parquet_exists={int(ppath.exists())} row_count={len(candles)}")
    if not candles.empty:
        lines.append(f"- candles_min_time_utc={iso_ts(candles['timestamp'].min())} candles_max_time_utc={iso_ts(candles['timestamp'].max())}")
        lines.append(f"- first_open={fmt(candles.iloc[0].get('open'))} last_close={fmt(candles.iloc[-1].get('close'))} day_high={fmt(pd.to_numeric(candles['high'], errors='coerce').max())} day_low={fmt(pd.to_numeric(candles['low'], errors='coerce').min())}")
    if stats is not None:
        lines.append(f"- first_5m_high_pct={fmt(stats.first_5m_high_pct)}")
        lines.append(f"- first_15m_high_pct={fmt(stats.first_15m_high_pct)}")
        lines.append(f"- or_range_pct={fmt(stats.or_range_pct)}")
        lines.append(f"- open_to_high_pct={fmt(stats.open_to_high_pct)} high_time={iso_ts(stats.high_time)}")
    lines.append(f"- first_time_above_5pct={diag.get('first_time_above_5pct') or iso_ts(first_time_above(candles, stats.open_price if stats else None, 5.0))}")
    lines.append(f"- first_time_above_8pct={diag.get('first_time_above_8pct') or iso_ts(first_time_above(candles, stats.open_price if stats else None, 8.0))}")
    lines.append(f"- opening_range_high_pct={fmt(diag.get('opening_range_high_pct'))}")
    lines.append(f"- opening_range_low_pct={fmt(diag.get('opening_range_low_pct'))}")
    lines.append(f"- opening_range_break_time={diag.get('opening_range_break_time')}")
    lines.append(f"- possible_signal_time={diag.get('possible_signal_time')}")
    lines.append("")
    lines.append("3. Offline Rule Check")
    lines.append(f"- effective_min_first5={min_first_5m_high_pct} effective_min_first15={min_first_15m_high_pct} effective_min_or_range={min_or_range_pct} config_source={config_source}")
    lines.append(f"- top100 {pass_fail(top100_ok)}")
    lines.append(f"- first5 >= {min_first_5m_high_pct}% {pass_fail(first5_ok)}")
    lines.append(f"- first15 >= {min_first_15m_high_pct}% {pass_fail(first15_ok)}")
    lines.append(f"- OR range >= {min_or_range_pct}% {pass_fail(or_ok)}")
    lines.append(f"- breakout diagnostic {pass_fail(breakout_ok)} gate_used={int(breakout_gate_used)}")
    lines.append(f"- signal_price_source={diag.get('signal_price_source')} earliest_legal_signal_time={diag.get('earliest_legal_signal_time')}")
    lines.append(f"- final offline_signal_ready={'YES' if offline_ready else 'NO'}")
    lines.append("")
    lines.append("4. Runtime Evidence")
    for source, df in {**sqlite_sources, **recorder_sources}.items():
        rows = symbol_rows(df, symbol, center, window_minutes=30)
        min_ts, max_ts = source_time_bounds(rows)
        lines.append(f"- {source}: found={len(rows)} min_time={min_ts} max_time={max_ts}")
        for line in event_lines(df, symbol=symbol, center=center, window_minutes=30, limit=12):
            lines.append(line)
    lines.append("")
    lines.append("5. Decision Trace")
    lines.append(f"- runtime_signal_seen={int(has_event(timeline, ['SIGNAL_READY']))}")
    lines.append(f"- runtime_buy_attempt_seen={int(has_event(timeline, ['PAPER_BUY', 'PAPER BUY', 'BUY_ORDER_SENT', 'ORDER_SUBMITTED']))}")
    lines.append(f"- runtime_buy_blocked_seen={int(has_event(timeline, ['BUY_BLOCKED', 'RISK_GUARD_BLOCK_ENTRY']))}")
    lines.append(f"- runtime_trade_or_fill_seen={int(counts.get('trades_count', 0) > 0 or counts.get('executions_count', 0) > 0)}")
    lines.append(f"- block_reason={block_reason}")
    lines.append(f"- final_decision={decision}")
    lines.append("")
    lines.append("6. Compact Timeline")
    for event in merge_timeline_events(timeline):
        lines.append(f"- {event.get('time')} [{event.get('source')}] {event.get('event')} reason={event.get('reason')} details={str(event.get('details') or '')[:180]}")
    lines.append("")
    lines.append("7. Live Trader Source References")
    for ref in static_line_refs()[:220]:
        lines.append(f"- {ref}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace one should-have-signaled symbol through candles, runtime artifacts, SQLite, and live trader source references.")
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=None)
    add_threshold_cli(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    effective = resolve_threshold_args(args, args.date)
    top100 = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    output = args.output or Path(f"data/analysis/signal_trace_{normalize_symbol(args.symbol)}_{args.date}.txt")
    text = trace_signal_case(
        session_date=args.date,
        symbol=args.symbol,
        sqlite_path=args.sqlite_path,
        history_dir=args.history_dir,
        recorder_dir=args.recorder_dir,
        top100_path=top100,
        universe_path=args.universe,
        min_first_5m_high_pct=effective.min_first5,
        min_first_15m_high_pct=effective.min_first15,
        min_or_range_pct=effective.min_or_range,
        config_source=effective.config_source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
