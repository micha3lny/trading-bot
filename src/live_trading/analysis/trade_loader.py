from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live_trading.analysis.common import read_sql_table


def sqlite_table_columns(sqlite_path: str | Path, table: str) -> set[str]:
    import sqlite3
    try:
        with sqlite3.connect(sqlite_path) as conn:
            return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()

FINALIZED_TRADE_STATUSES = ("CLOSED", "COMMISSION_PENDING", "PNL_PENDING")
FINALIZED_TRADE_WHERE = """
UPPER(COALESCE(status, '')) IN ('CLOSED', 'COMMISSION_PENDING', 'PNL_PENDING')
AND NULLIF(entry_fill_time, '') IS NOT NULL
AND NULLIF(exit_fill_time, '') IS NOT NULL
AND entry_price IS NOT NULL
AND exit_price IS NOT NULL
""".strip()


def finalized_trade_date_filter(start_date: str, end_date: str) -> tuple[str, list[str]]:
    return (
        f"{FINALIZED_TRADE_WHERE} AND ("
        "(COALESCE(session_date, '') != '' AND session_date BETWEEN ? AND ?) "
        "OR (COALESCE(session_date, '') = '' AND substr(exit_fill_time, 1, 10) BETWEEN ? AND ?)"
        ")",
        [start_date, end_date, start_date, end_date],
    )


def load_finalized_canonical_trades(sqlite_path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    required = {"status", "entry_fill_time", "exit_fill_time", "entry_price", "exit_price"}
    columns = sqlite_table_columns(sqlite_path, "trades")
    if not required.issubset(columns):
        out = pd.DataFrame()
        out.attrs["finalized_trade_filter"] = FINALIZED_TRADE_WHERE
        out.attrs["missing_required_columns"] = sorted(required - columns)
        return out
    where, params = finalized_trade_date_filter(start_date, end_date)
    trades = read_sql_table(
        sqlite_path,
        "trades",
        where=where,
        params=params,
        order_by="COALESCE(exit_fill_time, closed_at), symbol, trade_id",
    )
    if trades.empty:
        return trades
    trades = trades.copy()
    trades.attrs["finalized_trade_filter"] = FINALIZED_TRADE_WHERE
    return trades
