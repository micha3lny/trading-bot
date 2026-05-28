from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

from src.live_trading.unified_logger import daily_log_path, resolve_log_dir


PATTERNS = {
    "heartbeat": re.compile(r"\bheartbeat\b"),
    "paper_buy_sent": re.compile(r"\bPAPER BUY SENT\b"),
    "paper_sell_sent": re.compile(r"\bPAPER SELL SENT\b"),
    "reconciliation": re.compile(r"\b(?:RECONCILIATION|STARTUP_RECONCILIATION)"),
    "control_api": re.compile(r"\bCONTROL_API_"),
}


def count_patterns(lines: Iterable[str]) -> dict[str, int]:
    counts = {key: 0 for key in PATTERNS}
    for line in lines:
        for key, pattern in PATTERNS.items():
            if pattern.search(line):
                counts[key] += 1
    return counts


def read_unified_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def read_journal_lines(unit: str, since: str) -> list[str]:
    completed = subprocess.run(
        ["journalctl", "-u", unit, "--since", since, "--no-pager"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or f"journalctl failed with returncode={completed.returncode}")
    return completed.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare systemd journal event counts with unified trading bot log.")
    parser.add_argument("--unit", default="trading-bot")
    parser.add_argument("--since", default="today")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--date", default=None, help="UTC date for unified log, defaults to today")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.date:
        from datetime import datetime

        day = datetime.strptime(args.date, "%Y-%m-%d").date()
        unified_path = daily_log_path(resolve_log_dir(args.log_dir), day=day)
    else:
        unified_path = daily_log_path(resolve_log_dir(args.log_dir))

    journal_counts = count_patterns(read_journal_lines(args.unit, args.since))
    unified_counts = count_patterns(read_unified_lines(unified_path))
    rows = []
    for key in PATTERNS:
        journal = journal_counts[key]
        unified = unified_counts[key]
        rows.append({
            "event": key,
            "journal": journal,
            "unified": unified,
            "delta": journal - unified,
            "coverage_pct": round((unified / journal * 100.0), 2) if journal else 100.0,
        })
    payload = {"unified_log": str(unified_path), "rows": rows}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"unified_log={unified_path}")
        print("event journal unified delta coverage_pct")
        for row in rows:
            print(f"{row['event']} {row['journal']} {row['unified']} {row['delta']} {row['coverage_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
