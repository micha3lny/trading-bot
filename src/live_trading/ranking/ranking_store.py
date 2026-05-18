from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class RankingStore:
    """Tiny SQLite store for daily ranking snapshots.

    The CSV remains the runtime handoff artifact for v67. SQLite is used as a
    durable audit trail so daily outputs can be inspected without parsing many
    dated files.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_rankings (
                    date TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    score REAL NOT NULL,
                    components_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (date, rank)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_rankings_date_symbol "
                "ON daily_rankings(date, symbol)"
            )

    def replace_daily_rankings(self, ranking_date: str, rows: Iterable[dict[str, Any]]) -> int:
        self.init_schema()
        created_at = datetime.now(timezone.utc).isoformat()
        payload: list[tuple[str, int, str, float, str, str]] = []
        for row in rows:
            components = row.get("components_json")
            if not isinstance(components, str):
                components = json.dumps(row.get("components") or {}, sort_keys=True)
            payload.append(
                (
                    ranking_date,
                    int(row["rank"]),
                    str(row["symbol"]).upper(),
                    float(row["score"]),
                    components,
                    created_at,
                )
            )
        with self.connect() as conn:
            conn.execute("DELETE FROM daily_rankings WHERE date = ?", (ranking_date,))
            conn.executemany(
                """
                INSERT INTO daily_rankings(date, rank, symbol, score, components_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_daily_rankings(self, ranking_date: str, limit: int | None = None) -> list[dict[str, Any]]:
        self.init_schema()
        sql = (
            "SELECT date, rank, symbol, score, components_json, created_at "
            "FROM daily_rankings WHERE date = ? ORDER BY rank"
        )
        params: list[Any] = [ranking_date]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

