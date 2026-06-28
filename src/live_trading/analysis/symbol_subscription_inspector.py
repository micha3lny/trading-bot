from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    load_top100,
    normalize_symbol,
    parse_dt,
    read_sql_table,
    safe_read_csv,
)
from src.live_trading.analysis.signal_replay_analyzer import (
    load_jsonl,
    load_recorder_source,
    read_sqlite_sources,
    row_text,
    row_time,
    symbol_rows,
)


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_RECORDER_DIR = Path("data/live/recorder")

JOURNAL_TERMS = [
    "TOP100_RELOAD_START",
    "TOP100_RELOAD_DONE",
    "TOP100_RELOAD_FAILED",
    "TOP100_RELOAD_REQUESTED",
    "TOP100_RELOAD_SUBSCRIBED",
    "TOP100_RELOAD_CONTRACT_FAILED",
    "TOP100_RELOAD_SUBSCRIBE_ERROR",
    "ENTRY_SYMBOL_INELIGIBLE_SKIPPED",
    "DAILY_TOP100_USING_STALE_BLOCKED",
    "DAILY_TOP100_USING_STALE_ALLOWED",
    "entries_blocked_reason",
    "restart_cooldown",
    "RESTART",
    "SIGNAL_READY",
    "BUY_ORDER_SENT",
    "BUY_BLOCKED",
    "heartbeat",
    "subscribed_top100",
    "top100_requested",
    "active_position_symbols_count",
    "max_subscriptions",
    "ticker",
    "market data",
    "delayed",
    "permission",
]


SUMMARY_COLUMNS = [
    "date",
    "symbol",
    "in_top100",
    "top100_rank",
    "possible_signal_time",
    "journal_symbol_lines",
    "contract_metadata_rows",
    "candles_rows",
    "reload_requested_seen",
    "reload_subscribed_seen",
    "contract_failed_seen",
    "subscribe_error_seen",
    "market_data_seen",
    "entries_blocked_at_signal_time",
    "likely_root_cause",
]


def read_text_lines(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


def line_time(line: str) -> pd.Timestamp | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)?)", line)
    if match:
        return parse_dt(match.group(1).replace("Z", "+00:00"))
    match = re.search(r"([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})", line)
    if match:
        return None
    return None


def line_window(lines: list[str], center: pd.Timestamp | None, minutes: int = 30) -> list[str]:
    if center is None:
        return lines
    out = []
    for line in lines:
        ts = line_time(line)
        if ts is None or center - pd.Timedelta(minutes=minutes) <= ts <= center + pd.Timedelta(minutes=minutes):
            out.append(line)
    return out


def symbol_journal_lines(lines: list[str], symbol: str) -> list[str]:
    needle = normalize_symbol(symbol)
    return [line for line in lines if re.search(rf"(?<![A-Z0-9]){re.escape(needle)}(?![A-Z0-9])", line.upper())]


def term_journal_lines(lines: list[str]) -> list[str]:
    lowered_terms = [term.lower() for term in JOURNAL_TERMS]
    return [line for line in lines if any(term in line.lower() for term in lowered_terms)]


def parse_key_values(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)", line):
        out[key] = value.strip(",")
    return out


def top100_row(top100_path: Path, symbol: str) -> dict[str, Any]:
    top100 = load_top100(top100_path)
    if top100.empty:
        return {}
    rows = top100[top100["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def recorder_table(recorder_dir: Path, session_date: str, name: str) -> pd.DataFrame:
    return load_recorder_source(recorder_dir, session_date, name)


def candles_for_symbol(recorder_dir: Path, session_date: str, symbol: str) -> pd.DataFrame:
    df = safe_read_csv(recorder_dir / session_date / "candles_1m.csv")
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    return df[df["symbol"].map(normalize_symbol) == normalize_symbol(symbol)].copy()


def possible_signal_time(sqlite_path: Path, session_date: str, symbol: str) -> str:
    # Prefer the newest analysis CSV if it exists; otherwise fall back to empty.
    for path in [
        Path(f"data/analysis/missed_runners_{session_date}.csv"),
        Path(f"data/analysis/signal_replay_{session_date}.csv"),
    ]:
        df = safe_read_csv(path)
        if df.empty or "symbol" not in df.columns:
            continue
        rows = df[df["symbol"].map(normalize_symbol) == normalize_symbol(symbol)]
        if rows.empty:
            continue
        for col in ["possible_signal_time", "opening_range_break_time", "first_time_above_8pct"]:
            if col in rows.columns:
                value = rows.iloc[0].get(col)
                if value not in (None, "") and not pd.isna(value):
                    return str(value)
    return ""


def sqlite_counts(sqlite_path: Path, session_date: str, symbol: str, center: pd.Timestamp | None) -> dict[str, int]:
    sources = read_sqlite_sources(sqlite_path, session_date)
    return {name: len(symbol_rows(df, symbol, center, window_minutes=30)) for name, df in sources.items()}


def load_contract_metadata(recorder_dir: Path, session_date: str, symbol: str) -> pd.DataFrame:
    df = recorder_table(recorder_dir, session_date, "contract_metadata")
    if df.empty:
        return df
    records = []
    for row in df.to_dict("records"):
        if normalize_symbol(row.get("symbol")) == normalize_symbol(symbol) or normalize_symbol(symbol) in row_text(row).upper():
            records.append(row)
    return pd.DataFrame(records)


def infer_verdict(
    *,
    in_top100: bool,
    journal_symbol: list[str],
    journal_terms: list[str],
    contract_rows: int,
    candles_rows: int,
    sqlite_count_total: int,
    center_lines: list[str],
) -> dict[str, Any]:
    joined_symbol = "\n".join(journal_symbol).upper()
    joined_terms = "\n".join(journal_terms).upper()
    joined_center = "\n".join(center_lines).upper()
    reload_requested = "TOP100_RELOAD_REQUESTED" in joined_symbol
    reload_subscribed = "TOP100_RELOAD_SUBSCRIBED" in joined_symbol
    contract_failed = "TOP100_RELOAD_CONTRACT_FAILED" in joined_symbol
    subscribe_error = "TOP100_RELOAD_SUBSCRIBE_ERROR" in joined_symbol or "MAX NUMBER OF TICKERS" in joined_symbol
    market_data_seen = candles_rows > 0 or "SIGNAL_READY" in joined_symbol or "LIVE_FEATURE_DEBUG" in joined_symbol or sqlite_count_total > 0
    blocked = "ENTRIES_BLOCKED_REASON" in joined_center or "RESTART_COOLDOWN" in joined_center or "TOP100_BLOCK=1" in joined_center
    if not in_top100:
        cause = "not_in_top100"
    elif "ENTRY_SYMBOL_INELIGIBLE_SKIPPED" in joined_symbol or "NO_TRADING_PERMISSION" in joined_symbol:
        cause = "ineligible"
    elif contract_failed:
        cause = "contract_failure"
    elif subscribe_error:
        cause = "subscription_error_or_cap"
    elif not reload_requested and "TOP100_RELOAD_DONE" in joined_terms:
        cause = "not_in_reload_requested_symbols_or_subscription_cap"
    elif reload_requested and not reload_subscribed:
        cause = "requested_but_not_subscribed"
    elif reload_subscribed and not market_data_seen:
        cause = "subscribed_but_no_market_data_seen"
    elif blocked:
        cause = "entries_blocked_at_signal_time"
    elif market_data_seen:
        cause = "market_data_seen_check_signal_pipeline"
    else:
        cause = "no_runtime_subscription_evidence"
    return {
        "reload_requested_seen": int(reload_requested),
        "reload_subscribed_seen": int(reload_subscribed),
        "contract_failed_seen": int(contract_failed),
        "subscribe_error_seen": int(subscribe_error),
        "market_data_seen": int(market_data_seen),
        "entries_blocked_at_signal_time": int(blocked),
        "subscribed_yes_no_unknown": "yes" if reload_subscribed else ("no" if reload_requested or contract_failed or subscribe_error else "unknown"),
        "market_data_seen_yes_no_unknown": "yes" if market_data_seen else ("no" if reload_subscribed or reload_requested else "unknown"),
        "entries_blocked_at_signal_time_yes_no_unknown": "yes" if blocked else "unknown",
        "likely_root_cause": cause,
    }


def top100_reload_diagnostics(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if any(term in line for term in ["TOP100_RELOAD_START", "TOP100_RELOAD_DONE", "TOP100_RELOAD_FAILED"]):
            kv = parse_key_values(line)
            if kv:
                out.append(f"- {line}\n  parsed={kv}")
            else:
                out.append(f"- {line}")
    return out


def inspect_symbol_subscription(
    *,
    session_date: str,
    symbol: str,
    journal_log: Path | None,
    sqlite_path: Path,
    recorder_dir: Path,
    top100_path: Path,
) -> tuple[str, dict[str, Any]]:
    symbol = normalize_symbol(symbol)
    center_text = possible_signal_time(sqlite_path, session_date, symbol)
    center = parse_dt(center_text)
    top = top100_row(top100_path, symbol)
    in_top100 = bool(top)
    journal = read_text_lines(journal_log)
    journal_symbol = symbol_journal_lines(journal, symbol)
    journal_terms = term_journal_lines(journal)
    center_terms = line_window(journal_terms, center, minutes=30)
    contract_df = load_contract_metadata(recorder_dir, session_date, symbol)
    candles_df = candles_for_symbol(recorder_dir, session_date, symbol)
    sqlite_ct = sqlite_counts(sqlite_path, session_date, symbol, center)
    rec_sources = {
        "run_metadata": recorder_table(recorder_dir, session_date, "run_metadata"),
        "trade_lifecycle": recorder_table(recorder_dir, session_date, "trade_lifecycle"),
        "order_lifecycle": recorder_table(recorder_dir, session_date, "order_lifecycle"),
        "fills": recorder_table(recorder_dir, session_date, "fills"),
        "strategy_equity": recorder_table(recorder_dir, session_date, "strategy_equity"),
    }
    rec_counts = {name: len(symbol_rows(df, symbol, center, window_minutes=30)) for name, df in rec_sources.items()}
    verdict = infer_verdict(
        in_top100=in_top100,
        journal_symbol=journal_symbol,
        journal_terms=journal_terms,
        contract_rows=len(contract_df),
        candles_rows=len(candles_df),
        sqlite_count_total=sum(sqlite_ct.values()),
        center_lines=center_terms,
    )
    summary = {
        "date": session_date,
        "symbol": symbol,
        "in_top100": int(in_top100),
        "top100_rank": top.get("top100_rank", ""),
        "possible_signal_time": center_text,
        "journal_symbol_lines": len(journal_symbol),
        "contract_metadata_rows": len(contract_df),
        "candles_rows": len(candles_df),
        **{key: verdict[key] for key in [
            "reload_requested_seen",
            "reload_subscribed_seen",
            "contract_failed_seen",
            "subscribe_error_seen",
            "market_data_seen",
            "entries_blocked_at_signal_time",
            "likely_root_cause",
        ]},
    }
    lines = [
        f"SUBSCRIPTION TRACE date={session_date} symbol={symbol}",
        "=" * 88,
        "",
        "Summary verdict",
        f"- subscribed_yes_no_unknown={verdict['subscribed_yes_no_unknown']}",
        f"- market_data_seen_yes_no_unknown={verdict['market_data_seen_yes_no_unknown']}",
        f"- entries_blocked_at_signal_time_yes_no_unknown={verdict['entries_blocked_at_signal_time_yes_no_unknown']}",
        f"- likely_root_cause={verdict['likely_root_cause']}",
        "",
        "Top100",
        f"- top100_path={top100_path}",
        f"- in_top100={int(in_top100)} top100_rank={top.get('top100_rank', '')} top100_score={top.get('top100_score', '')}",
        f"- top100_row={top}",
        "",
        "Top100 reload diagnostics",
        *top100_reload_diagnostics(journal_terms)[:60],
        "",
        "Symbol-specific journal lines",
    ]
    lines.extend(f"- {line}" for line in journal_symbol[:120])
    if len(journal_symbol) > 120:
        lines.append(f"... {len(journal_symbol) - 120} more symbol lines omitted")
    lines.extend([
        "",
        "Timeline around possible signal / reload / restart",
        f"- possible_signal_time={center_text}",
    ])
    lines.extend(f"- {line}" for line in center_terms[:160])
    if len(center_terms) > 160:
        lines.append(f"... {len(center_terms) - 160} more nearby lines omitted")
    lines.extend([
        "",
        "Recorder evidence",
        f"- contract_metadata_rows={len(contract_df)}",
    ])
    for row in contract_df.head(20).to_dict("records"):
        lines.append(f"  - {row}")
    lines.append(f"- candles_rows={len(candles_df)}")
    if not candles_df.empty:
        lines.append(f"  first={candles_df.head(1).to_dict('records')[0]}")
        lines.append(f"  last={candles_df.tail(1).to_dict('records')[0]}")
    for name, count in rec_counts.items():
        lines.append(f"- {name}_rows={count}")
    lines.extend([
        "",
        "SQLite evidence counts",
    ])
    for name, count in sqlite_ct.items():
        lines.append(f"- {name}={count}")
    lines.extend([
        "",
        "Journal extraction helper",
        'journalctl -u v67-trader --since "2026-06-26 13:20:00 UTC" --until "2026-06-26 15:30:00 UTC" -o cat --no-pager > data/analysis/journal_v67_2026-06-26_1320_1530_utc.log',
    ])
    return "\n".join(lines) + "\n", summary


def parse_symbols(args: argparse.Namespace) -> list[str]:
    raw = args.symbols or args.symbol
    if not raw:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        values = str(raw).split(",")
    return [normalize_symbol(value) for value in values if normalize_symbol(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect whether one or more Top100 symbols were reloaded, subscribed, and receiving runtime market data.")
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD.")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols for batch mode.")
    parser.add_argument("--journal-log", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = parse_symbols(args)
    if not symbols:
        parser.error("--symbol or --symbols is required")
    top100 = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    summaries: list[dict[str, Any]] = []
    for symbol in symbols:
        default_output = Path(f"data/analysis/subscription_trace_{symbol}_{args.date}.txt")
        output = args.output if args.output and len(symbols) == 1 else default_output
        text, summary = inspect_symbol_subscription(
            session_date=args.date,
            symbol=symbol,
            journal_log=args.journal_log,
            sqlite_path=args.sqlite_path,
            recorder_dir=args.recorder_dir,
            top100_path=top100,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        summaries.append(summary)
        if len(symbols) == 1:
            print(text)
            print(f"output={output}")
        else:
            print(f"wrote {output}")
    if len(symbols) > 1:
        summary_path = Path(f"data/analysis/subscription_trace_summary_{args.date}.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries)[SUMMARY_COLUMNS].to_csv(summary_path, index=False)
        print(f"summary_output={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
