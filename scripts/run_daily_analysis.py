#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "data" / "analysis"


@dataclass
class StepResult:
    name: str
    command: list[str]
    skipped: bool
    status: str
    exit_code: int | None
    elapsed_seconds: float


def parse_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size <= 0:
        return [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def numeric_sum(rows: list[dict[str, str]], column: str) -> float | None:
    total = 0.0
    seen = False
    for row in rows:
        raw = row.get(column)
        if raw in (None, ""):
            continue
        try:
            total += float(raw)
            seen = True
        except ValueError:
            continue
    return total if seen else None


def value_counts(rows: list[dict[str, str]], column: str, *, limit: int = 8) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column) or "missing")
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def summarize_csv(path: Path) -> list[str]:
    fields, rows = csv_rows(path)
    if not path.exists():
        return [f"- `{rel(path)}`: missing"]
    lines = [f"- `{rel(path)}`: rows={len(rows)}"]
    if not rows:
        return lines

    if "net_pnl" in fields:
        lines.append(f"  - net_pnl_sum={format_float(numeric_sum(rows, 'net_pnl'))}")
    if "actual_net_pnl" in fields:
        lines.append(f"  - actual_net_pnl_sum={format_float(numeric_sum(rows, 'actual_net_pnl'))}")
    if "gross_pnl" in fields:
        lines.append(f"  - gross_pnl_sum={format_float(numeric_sum(rows, 'gross_pnl'))}")
    if "open_to_high_pct" in fields:
        values = []
        for row in rows:
            try:
                values.append(float(row.get("open_to_high_pct") or ""))
            except ValueError:
                pass
        if values:
            lines.append(f"  - max_open_to_high_pct={format_float(max(values))}")

    for column in (
        "final_classification",
        "final_no_buy_reason",
        "missed_reason_group",
        "top100_no_signal_reason",
        "data_quality",
        "source_bucket",
    ):
        if column in fields:
            counts = ", ".join(f"{key}={count}" for key, count in value_counts(rows, column))
            lines.append(f"  - {column}: {counts}")

    if len(rows) == 1:
        interesting = [
            "date",
            "session_date",
            "total_should_have_signaled",
            "runtime_signal_ready_but_no_buy",
            "bought_late",
            "restart_blocked",
            "total_cases",
            "post_signal_stale_or_backfill_skip",
            "post_signal_already_open_skip",
            "unexplained_after_signal_before_dispatch",
            "ambiguous_event_correlation",
            "offline_signal_expected_runtime_signal_not_observed",
            "unknown_no_buy_after_signal",
            "coverage_pct",
            "capture_pct",
        ]
        pairs = [f"{key}={rows[0].get(key)}" for key in interesting if key in fields and rows[0].get(key) not in (None, "")]
        if pairs:
            lines.append(f"  - summary: {', '.join(pairs)}")
    return lines


def build_steps(session_date: str, args: argparse.Namespace) -> list[tuple[str, list[str], bool]]:
    force_args = [] if args.no_force else ["--force"]
    py = sys.executable
    return [
        (
            "coverage",
            [py, "scripts/build_strategy_coverage_report.py", "--date", session_date, *force_args],
            args.skip_coverage,
        ),
        (
            "missed",
            [py, "scripts/analyze_missed_runners.py", "--date", session_date, *force_args],
            args.skip_missed,
        ),
        (
            "bad_entries",
            [py, "scripts/analyze_bad_entries.py", "--date", session_date],
            args.skip_bad_entries,
        ),
        (
            "shs",
            [py, "scripts/investigate_should_have_signaled.py", "--date", session_date, *force_args],
            args.skip_shs,
        ),
        (
            "nbas",
            [py, "scripts/investigate_no_buy_after_signal.py", "--date", session_date, *force_args],
            args.skip_nbas,
        ),
    ]


def expected_outputs(session_date: str) -> list[Path]:
    return [
        DEFAULT_ANALYSIS_DIR / f"coverage_report_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / "coverage_history.csv",
        DEFAULT_ANALYSIS_DIR / f"missed_runners_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"bad_entries_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"should_have_signaled_cases_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"should_have_signaled_summary_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / "should_have_signaled_summary_ALL.csv",
        DEFAULT_ANALYSIS_DIR / f"no_buy_after_signal_cases_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"no_buy_after_signal_summary_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / "no_buy_after_signal_summary_ALL.csv",
    ]


def write_summary(session_date: str, results: list[StepResult], *, total_elapsed: float, final_failed: bool) -> Path:
    DEFAULT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_ANALYSIS_DIR / f"daily_analysis_summary_{session_date}.md"
    outputs = expected_outputs(session_date)
    generated = [p for p in outputs if p.exists()]

    lines: list[str] = [
        f"# Daily Analysis Summary {session_date}",
        "",
        f"- final_status: {'FAILED' if final_failed else 'OK'}",
        f"- elapsed_seconds: {total_elapsed:.2f}",
        "",
        "## Steps",
        "",
        "| step | status | exit_code | elapsed_seconds | command |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        command = " ".join(result.command)
        exit_code = "" if result.exit_code is None else str(result.exit_code)
        lines.append(
            f"| {result.name} | {result.status} | {exit_code} | {result.elapsed_seconds:.2f} | `{command}` |"
        )

    lines.extend(["", "## Generated Output Files", ""])
    if generated:
        for output in generated:
            lines.append(f"- `{rel(output)}`")
    else:
        lines.append("- none found")

    lines.extend(["", "## CSV Summaries", ""])
    summary_targets = [
        DEFAULT_ANALYSIS_DIR / f"coverage_report_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"missed_runners_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"bad_entries_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"should_have_signaled_summary_{session_date}.csv",
        DEFAULT_ANALYSIS_DIR / f"no_buy_after_signal_summary_{session_date}.csv",
    ]
    for target in summary_targets:
        lines.extend(summarize_csv(target))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_step(name: str, command: list[str], *, skipped: bool) -> StepResult:
    if skipped:
        print(f"DAILY_ANALYSIS_STEP_SKIPPED name={name}", flush=True)
        return StepResult(name=name, command=command, skipped=True, status="skipped", exit_code=None, elapsed_seconds=0.0)

    print(f"DAILY_ANALYSIS_STEP_START name={name} command={' '.join(command)}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT)
    elapsed = time.monotonic() - started
    if completed.returncode == 0:
        print(f"DAILY_ANALYSIS_STEP_DONE name={name} elapsed={elapsed:.2f}", flush=True)
        status = "ok"
    else:
        print(f"DAILY_ANALYSIS_STEP_FAILED name={name} exit_code={completed.returncode} elapsed={elapsed:.2f}", flush=True)
        status = "failed"
    return StepResult(
        name=name,
        command=command,
        skipped=False,
        status=status,
        exit_code=completed.returncode,
        elapsed_seconds=elapsed,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full read-only daily analysis suite for one completed session.")
    parser.add_argument("--date", required=True, type=parse_date, help="Completed session date, YYYY-MM-DD.")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-missed", action="store_true")
    parser.add_argument("--skip-bad-entries", action="store_true")
    parser.add_argument("--skip-shs", action="store_true")
    parser.add_argument("--skip-nbas", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--no-force", action="store_true", help="Do not pass --force to analyzers that support it.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    session_date = args.date
    print(f"DAILY_ANALYSIS_START date={session_date}", flush=True)
    total_started = time.monotonic()
    results: list[StepResult] = []
    failed = False

    for name, command, skipped in build_steps(session_date, args):
        result = run_step(name, command, skipped=skipped)
        results.append(result)
        if result.status == "failed":
            failed = True
            if args.stop_on_failure:
                break

    total_elapsed = time.monotonic() - total_started
    summary_path = write_summary(session_date, results, total_elapsed=total_elapsed, final_failed=failed)
    print(f"DAILY_ANALYSIS_SUMMARY_WRITTEN path={rel(summary_path)}", flush=True)
    print(f"DAILY_ANALYSIS_DONE date={session_date} elapsed={total_elapsed:.2f} status={'FAILED' if failed else 'OK'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
