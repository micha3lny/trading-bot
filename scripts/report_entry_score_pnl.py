#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_trading.storage.sqlite_store import connect_sqlite, resolve_sqlite_path


BUCKETS = [
    ("score >= 80", 80.0, math.inf),
    ("score 60-80", 60.0, 80.0),
    ("score 40-60", 40.0, 60.0),
    ("score 20-40", 20.0, 40.0),
    ("score <20", -math.inf, 20.0),
    ("score missing", math.nan, math.nan),
]


@dataclass
class RuntimeEventScore:
    event_time: str
    session_date: str
    symbol: str
    trade_id: str
    order_id: str
    score: float | None
    ranking_position: int | None


def parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def iso_date(value: Any) -> str:
    parsed = parse_dt(value)
    if parsed:
        return parsed.date().isoformat()
    text = str(value or "")
    return text[:10] if len(text) >= 10 else ""


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except Exception:
        return None


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def first_float(raw: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = raw.get(key)
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    nested = raw.get("raw_json")
    if isinstance(nested, dict):
        return first_float(nested, keys)
    return None


def first_int(raw: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        value = raw.get(key)
        parsed = safe_int(value)
        if parsed is not None:
            return parsed
    nested = raw.get("raw_json")
    if isinstance(nested, dict):
        return first_int(nested, keys)
    return None


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def fetch_recent_closed_sessions(conn, session_count: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT close_date
        FROM (
            SELECT DISTINCT
                COALESCE(
                    NULLIF(substr(exit_fill_time, 1, 10), ''),
                    NULLIF(substr(closed_at, 1, 10), ''),
                    NULLIF(session_date, '')
                ) AS close_date
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
        )
        WHERE close_date IS NOT NULL AND close_date != ''
        ORDER BY close_date DESC
        LIMIT ?
        """,
        [int(session_count)],
    ).fetchall()
    return [str(row["close_date"]) for row in rows]


def fetch_closed_trades(conn, sessions: list[str]) -> list[dict[str, Any]]:
    if not sessions:
        return []
    session_sql = ",".join(quote(date) for date in sessions)
    return [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT
                trade_id,
                strategy_name,
                session_date,
                symbol,
                entry_fill_time,
                entry_order_time,
                entry_signal_time,
                exit_fill_time,
                closed_at,
                entry_price,
                exit_price,
                quantity,
                gross_pnl,
                commission,
                net_pnl,
                raw_json,
                COALESCE(
                    NULLIF(substr(exit_fill_time, 1, 10), ''),
                    NULLIF(substr(closed_at, 1, 10), ''),
                    NULLIF(session_date, '')
                ) AS close_date
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
              AND close_date IN ({session_sql})
            ORDER BY close_date, COALESCE(exit_fill_time, closed_at, entry_fill_time, session_date), symbol, trade_id
            """
        ).fetchall()
    ]


def fetch_runtime_event_scores(conn, sessions: list[str]) -> list[RuntimeEventScore]:
    if not sessions:
        return []
    session_sql = ",".join(quote(date) for date in sessions)
    rows = conn.execute(
        f"""
        SELECT event_time, session_date, symbol, trade_id, order_id, event_type, raw_json
        FROM runtime_events
        WHERE COALESCE(NULLIF(session_date, ''), substr(event_time, 1, 10)) IN ({session_sql})
          AND UPPER(COALESCE(event_type, '')) IN (
              'BUY_ORDER_SENT',
              'ENTRY_ORDER_SUBMITTED',
              'PAPER_BUY_SENT'
          )
        ORDER BY event_time
        """
    ).fetchall()
    events: list[RuntimeEventScore] = []
    for row in rows:
        raw = parse_jsonish(row["raw_json"])
        score = first_float(raw, ["score", "entry_score", "alpha_score", "final_score"])
        if score is None:
            continue
        events.append(
            RuntimeEventScore(
                event_time=str(row["event_time"] or ""),
                session_date=str(row["session_date"] or iso_date(row["event_time"])),
                symbol=str(row["symbol"] or "").upper(),
                trade_id=str(row["trade_id"] or ""),
                order_id=str(row["order_id"] or ""),
                score=score,
                ranking_position=first_int(raw, ["ranking_position", "rank"]),
            )
        )
    return events


def match_entry_score(trade: dict[str, Any], events: list[RuntimeEventScore]) -> tuple[float | None, str, int | None]:
    raw = parse_jsonish(trade.get("raw_json"))
    score = first_float(raw, ["entry_score", "score", "alpha_score", "final_score"])
    rank = first_int(raw, ["ranking_position", "rank"])
    if score is not None:
        return score, "trades.raw_json", rank

    trade_id = str(trade.get("trade_id") or "")
    symbol = str(trade.get("symbol") or "").upper()
    entry_time = trade.get("entry_fill_time") or trade.get("entry_order_time") or trade.get("entry_signal_time")
    entry_dt = parse_dt(entry_time)
    session_date = str(trade.get("session_date") or iso_date(entry_time) or "")

    if trade_id:
        for event in events:
            if event.trade_id and event.trade_id == trade_id:
                return event.score, "runtime_events.trade_id", event.ranking_position

    candidates = [
        event
        for event in events
        if event.symbol == symbol and (not session_date or event.session_date == session_date or iso_date(event.event_time) == session_date)
    ]
    if not candidates:
        return None, "missing", None
    if entry_dt is None:
        event = candidates[0]
        return event.score, "runtime_events.symbol_session", event.ranking_position

    best: tuple[float, RuntimeEventScore] | None = None
    for event in candidates:
        event_dt = parse_dt(event.event_time)
        if event_dt is None:
            continue
        delta = abs((entry_dt - event_dt).total_seconds())
        if delta <= 30 * 60 and (best is None or delta < best[0]):
            best = (delta, event)
    if best is None:
        return None, "missing", None
    return best[1].score, "runtime_events.nearest_entry", best[1].ranking_position


def bucket_for_score(score: float | None) -> str:
    if score is None:
        return "score missing"
    if score >= 80:
        return "score >= 80"
    if score >= 60:
        return "score 60-80"
    if score >= 40:
        return "score 40-60"
    if score >= 20:
        return "score 20-40"
    return "score <20"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def money(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def render_table(rows: list[dict[str, Any]], columns: list[str], *, limit: int | None = None) -> None:
    shown = rows if limit is None else rows[:limit]
    if not shown:
        print("(brak danych)")
        return
    widths = {col: len(col) for col in columns}
    for row in shown:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in shown:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    if limit is not None and len(rows) > limit:
        print(f"... pokazano {limit}/{len(rows)} rows")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report closed trade PnL grouped by entry score buckets.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--sessions", type=int, default=20, help="Number of most recent closed sessions to include.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--print-limit", type=int, default=200, help="Print at most N trade rows; CSV always contains all rows.")
    args = parser.parse_args()

    sqlite_path = resolve_sqlite_path(args.sqlite_path)
    conn = connect_sqlite(sqlite_path, read_only=True)
    try:
        sessions = fetch_recent_closed_sessions(conn, args.sessions)
        trades = fetch_closed_trades(conn, sessions)
        events = fetch_runtime_event_scores(conn, sessions)
    finally:
        conn.close()

    trade_rows: list[dict[str, Any]] = []
    for trade in trades:
        score, score_source, rank = match_entry_score(trade, events)
        gross = safe_float(trade.get("gross_pnl"))
        commission = safe_float(trade.get("commission")) or 0.0
        net = safe_float(trade.get("net_pnl"))
        if net is None and gross is not None:
            net = gross - commission
        trade_rows.append(
            {
                "close_date": trade.get("close_date") or "",
                "symbol": str(trade.get("symbol") or "").upper(),
                "entry_time": trade.get("entry_fill_time") or trade.get("entry_order_time") or trade.get("entry_signal_time") or "",
                "entry_score": "" if score is None else round(score, 6),
                "score_bucket": bucket_for_score(score),
                "ranking_position": "" if rank is None else rank,
                "gross_pnl": "" if gross is None else round(gross, 6),
                "net_pnl": "" if net is None else round(net, 6),
                "score_source": score_source,
                "trade_id": trade.get("trade_id") or "",
            }
        )

    bucket_rows: list[dict[str, Any]] = []
    for bucket, _, _ in BUCKETS:
        bucket_trades = [row for row in trade_rows if row["score_bucket"] == bucket]
        count = len(bucket_trades)
        net_values = [float(row["net_pnl"]) for row in bucket_trades if row["net_pnl"] != ""]
        wins = sum(1 for value in net_values if value > 0)
        avg_net = (sum(net_values) / len(net_values)) if net_values else 0.0
        bucket_rows.append(
            {
                "score_bucket": bucket,
                "trade_count": count,
                "win_rate": pct((wins / len(net_values) * 100.0) if net_values else 0.0),
                "avg_net_pnl": round(avg_net, 6),
                "expectancy": round(avg_net, 6),
            }
        )

    output_dir = Path(args.output_dir)
    trades_path = output_dir / "entry_score_pnl_last20_trades.csv"
    summary_path = output_dir / "entry_score_pnl_last20_summary.csv"
    trade_fields = [
        "close_date",
        "symbol",
        "entry_time",
        "entry_score",
        "score_bucket",
        "ranking_position",
        "gross_pnl",
        "net_pnl",
        "score_source",
        "trade_id",
    ]
    summary_fields = ["score_bucket", "trade_count", "win_rate", "avg_net_pnl", "expectancy"]
    write_csv(trades_path, trade_rows, trade_fields)
    write_csv(summary_path, bucket_rows, summary_fields)

    print(f"SQLite: {sqlite_path}")
    print(f"Sessions: {', '.join(sessions) if sessions else 'none'}")
    print(f"Trades: {len(trade_rows)}")
    print(f"CSV trades: {trades_path}")
    print(f"CSV summary: {summary_path}")
    print()
    print("TRANSAKCJE")
    render_table(
        trade_rows,
        ["symbol", "entry_time", "entry_score", "gross_pnl", "net_pnl"],
        limit=max(0, int(args.print_limit)),
    )
    print()
    print("AGREGACJA")
    render_table(bucket_rows, summary_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
