from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.history_readiness import canonical_history_readiness


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parquet_path(history_dir: Path, symbol: str, session_date: str, session_type: str = "RTH") -> Path:
    return (
        history_dir
        / f"session_type={session_type.upper()}"
        / f"symbol={symbol.upper()}"
        / f"year={session_date[:4]}"
        / f"month={session_date[5:7]}"
        / f"day={session_date[8:10]}.parquet"
    )


def status_key(symbol: str, session_date: str, session_type: str = "RTH") -> str:
    return f"{symbol.upper()}_{session_date}_{session_type.upper()}"


def load_symbols(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe CSV missing symbol column: {path}")
    return [str(value).upper().strip() for value in df["symbol"].tolist() if str(value).strip()]


def load_status(history_dir: Path) -> dict[str, Any]:
    path = history_dir.parent / "collector_status.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def assess(*, history_dir: Path, universe: Path, session_date: str, session_type: str) -> dict[str, int | str | float]:
    symbols = load_symbols(universe)
    status = load_status(history_dir)
    counts = {
        "ranking_date": session_date,
        "expected_symbols": len(symbols),
        "complete_symbols": 0,
        "partial_symbols": 0,
        "no_data_symbols": 0,
        "missing_symbols": 0,
        "failed_symbols": 0,
        "parquet_files": 0,
    }
    for symbol in symbols:
        path = parquet_path(history_dir, symbol, session_date, session_type)
        has_parquet = path.exists() and path.stat().st_size > 0
        if has_parquet:
            counts["parquet_files"] += 1
        row = status.get(status_key(symbol, session_date, session_type))
        row_status = str(row.get("status") or "").lower() if isinstance(row, dict) else ""
        if has_parquet or row_status == "complete":
            counts["complete_symbols"] += 1
        elif row_status == "partial":
            counts["partial_symbols"] += 1
        elif row_status in {"no_data", "no_data_permanent"}:
            counts["no_data_symbols"] += 1
        elif row_status in {"failed", "failed_permanent"}:
            counts["failed_symbols"] += 1
        else:
            counts["missing_symbols"] += 1
    readiness = canonical_history_readiness(
        session_date=session_date,
        session_type=session_type,
        expected_symbols=int(counts["expected_symbols"]),
        complete_symbols=int(counts["complete_symbols"]),
        partial_symbols=int(counts["partial_symbols"]),
        no_data_symbols=int(counts["no_data_symbols"]),
        missing_symbols=int(counts["missing_symbols"]),
        failed_symbols=int(counts["failed_symbols"]),
        parquet_files=int(counts["parquet_files"]),
    )
    return {**counts, **readiness, "ranking_date": session_date}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast readiness gate for daily Top100 history input")
    parser.add_argument("--date", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--max-missing", type=int, default=0)
    parser.add_argument("--max-partial", type=int, default=0)
    parser.add_argument("--max-failed", type=int, default=0)
    args = parser.parse_args()

    counts = assess(
        history_dir=Path(args.history_dir),
        universe=Path(args.universe),
        session_date=str(args.date),
        session_type=str(args.session_type),
    )
    ready = (
        int(counts["expected_symbols"]) > 0
        and int(counts["missing_symbols"]) <= int(args.max_missing)
        and int(counts["partial_symbols"]) <= int(args.max_partial)
        and int(counts["failed_symbols"]) <= int(args.max_failed)
    )
    status = "OK" if ready else str(counts.get("readiness_status") or "NOT_READY")
    print(
        f"{now_iso()} HISTORY_READINESS_CHECK ranking_date={counts['ranking_date']} "
        f"expected_symbols={counts['expected_symbols']} complete_symbols={counts['complete_symbols']} "
        f"partial_symbols={counts['partial_symbols']} missing_symbols={counts['missing_symbols']} "
        f"no_data_symbols={counts['no_data_symbols']} failed_symbols={counts['failed_symbols']} "
        f"parquet_files={counts['parquet_files']} "
        f"effective_completion_pct={counts['effective_completion_pct']} "
        f"parquet_completion_pct={counts['parquet_completion_pct']} "
        f"completion_pct={counts['completion_pct']} readiness_status={status}",
        flush=True,
    )
    if not ready:
        print(
            f"{now_iso()} DAILY_TOP100_BLOCKED_HISTORY_NOT_READY ranking_date={counts['ranking_date']} "
            f"expected_symbols={counts['expected_symbols']} complete_symbols={counts['complete_symbols']} "
            f"partial_symbols={counts['partial_symbols']} missing_symbols={counts['missing_symbols']} "
            f"no_data_symbols={counts['no_data_symbols']} failed_symbols={counts['failed_symbols']} "
            f"max_missing={int(args.max_missing)} max_partial={int(args.max_partial)} max_failed={int(args.max_failed)}",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
