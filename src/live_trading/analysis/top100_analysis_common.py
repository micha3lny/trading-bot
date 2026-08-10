from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.live_trading.analysis.common import load_top100, normalize_symbol
from src.live_trading.candidate_snapshot_telemetry import snapshot_chunk_paths
from src.live_trading.market_calendar import is_us_equity_trading_day, previous_us_equity_trading_day


def session_dates(date_value: str | None, start_date: str | None, end_date: str | None) -> list[str]:
    if date_value:
        requested = date.fromisoformat(date_value)
        if not is_us_equity_trading_day(requested):
            raise ValueError(f"not a US equity trading session: {requested.isoformat()}")
        return [requested.isoformat()]
    if not start_date or not end_date:
        raise ValueError("provide --date or both --start-date and --end-date")
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end date precedes start date")
    return [
        item.date().isoformat()
        for item in pd.date_range(start, end, freq="D")
        if is_us_equity_trading_day(item.date())
    ]


def find_dated_top100(top100_dir: str | Path, session_date: str, ranking_source_date: str | None = None) -> tuple[Path | None, str | None]:
    root = Path(top100_dir)
    source = date.fromisoformat(ranking_source_date) if ranking_source_date else previous_us_equity_trading_day(date.fromisoformat(session_date))
    exact = root / f"daily_top100_{source.isoformat()}.csv"
    if exact.exists():
        return exact, source.isoformat()
    same_day = root / f"daily_top100_{session_date}.csv"
    if ranking_source_date is None and same_day.exists():
        return same_day, session_date
    candidates: list[tuple[date, Path]] = []
    for path in root.glob("daily_top100_*.csv"):
        raw = path.stem.removeprefix("daily_top100_")
        if raw.endswith("_diagnostics") or raw == "latest":
            continue
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        if parsed <= source:
            candidates.append((parsed, path))
    if not candidates:
        return None, None
    selected_date, selected = max(candidates, key=lambda item: item[0])
    return selected, selected_date.isoformat()


def load_top100_source(top100_dir: str | Path, session_date: str, ranking_source_date: str | None = None) -> tuple[pd.DataFrame, Path | None, str | None]:
    path, source_date = find_dated_top100(top100_dir, session_date, ranking_source_date)
    if path is None:
        return pd.DataFrame(), None, source_date
    raw = pd.read_csv(path)
    normalized = load_top100(path)
    if raw.empty:
        return normalized, path, source_date
    raw = raw.copy()
    raw["symbol"] = raw["symbol"].map(normalize_symbol)
    raw = raw.drop_duplicates("symbol")
    for column in normalized.columns:
        if column not in raw.columns:
            raw[column] = normalized.set_index("symbol")[column].reindex(raw["symbol"]).to_numpy()
    if "top100_rank" not in raw.columns:
        raw["top100_rank"] = range(1, len(raw) + 1)
    return raw, path, source_date


def read_snapshot_chunks(recorder_dir: str | Path, session_date: str, kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in snapshot_chunk_paths(recorder_dir, session_date, kind):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    keys = ["session_date", "process_start_id", "scan_id", "symbol"] if kind == "light" else ["session_date", "process_start_id", "symbol", "candle_timestamp", "feature_state_revision"]
    available = [key for key in keys if key in out.columns]
    if available:
        out = out.drop_duplicates(available, keep="last")
    return out.sort_values([column for column in ["timestamp", "process_start_id", "scan_id", "symbol"] if column in out.columns]).reset_index(drop=True)


def read_snapshot_manifest(recorder_dir: str | Path, session_date: str) -> dict[str, Any]:
    path = Path(recorder_dir) / session_date / "top100_candidate_snapshots" / "candidate_snapshot_manifest.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def numeric(series: pd.Series | Iterable[Any]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce")
