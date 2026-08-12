#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "data" / "analysis"


@dataclass(frozen=True)
class AnalyzerSpec:
    name: str
    script: str
    outputs: tuple[str, ...]
    summary_targets: tuple[str, ...] = ()
    supports_force: bool = False
    pass_sqlite_path: bool = False
    pass_history_dir: bool = False
    pass_signal_thresholds: bool = False
    output_dir_arg: str | None = None
    output_file_pattern: str | None = None
    category: str = "strategy"
    required: bool = True
    extra_args: tuple[str, ...] = ()


ANALYZER_REGISTRY: tuple[AnalyzerSpec, ...] = (
    AnalyzerSpec(
        name="coverage",
        script="scripts/build_strategy_coverage_report.py",
        outputs=("coverage_report_{date}.csv", "coverage_history.csv"),
        summary_targets=("coverage_report_{date}.csv",),
        supports_force=True,
        pass_sqlite_path=True,
        pass_history_dir=True,
        pass_signal_thresholds=True,
        output_dir_arg="--output-dir",
    ),
    AnalyzerSpec(
        name="missed",
        script="scripts/analyze_missed_runners.py",
        outputs=("missed_runners_{date}.csv",),
        summary_targets=("missed_runners_{date}.csv",),
        supports_force=True,
        pass_sqlite_path=True,
        pass_history_dir=True,
        pass_signal_thresholds=True,
        output_file_pattern="missed_runners_{date}.csv",
    ),
    AnalyzerSpec(
        name="bad_entries",
        script="scripts/analyze_bad_entries.py",
        outputs=(
            "bad_entries_{date}.csv",
            "bad_entries_trades_{date}.csv",
            "bad_entries_time_buckets_{date}.csv",
            "bad_entries_feature_buckets_{date}.csv",
            "bad_entries_filter_simulation_{date}.csv",
            "bad_entries_recommendations_{date}.md",
            "bad_entries_data_quality_{date}.json",
        ),
        summary_targets=("bad_entries_{date}.csv", "bad_entries_filter_simulation_{date}.csv", "bad_entries_recommendations_{date}.md"),
        pass_sqlite_path=True,
        pass_history_dir=True,
        output_dir_arg="--output-dir",
        output_file_pattern="bad_entries_{date}.csv",
    ),
    AnalyzerSpec(
        name="early_loser",
        script="scripts/early_loser_exit_analyzer.py",
        outputs=("early_loser_trade_paths_{date}.csv", "early_loser_rules_{date}.csv", "early_loser_summary_{date}.md"),
        summary_targets=("early_loser_rules_{date}.csv", "early_loser_summary_{date}.md"),
        pass_sqlite_path=True,
        pass_history_dir=True,
        output_dir_arg="--output-dir",
    ),
    AnalyzerSpec(
        name="stop_loss",
        script="scripts/stop_loss_strategy_analyzer.py",
        outputs=(
            "stop_loss_trade_paths_{date}.csv",
            "stop_loss_fixed_grid_{date}.csv",
            "stop_loss_activation_delay_{date}.csv",
            "stop_loss_slippage_sensitivity_{date}.csv",
            "stop_loss_segment_analysis_{date}.csv",
            "stop_loss_dynamic_rules_{date}.csv",
            "stop_loss_hybrid_rules_{date}.csv",
            "stop_loss_data_quality_{date}.json",
            "stop_loss_recommendations_{date}.md",
        ),
        summary_targets=(
            "stop_loss_fixed_grid_{date}.csv",
            "stop_loss_activation_delay_{date}.csv",
            "stop_loss_slippage_sensitivity_{date}.csv",
            "stop_loss_segment_analysis_{date}.csv",
            "stop_loss_dynamic_rules_{date}.csv",
            "stop_loss_hybrid_rules_{date}.csv",
            "stop_loss_data_quality_{date}.json",
            "stop_loss_recommendations_{date}.md",
        ),
        pass_sqlite_path=True,
        pass_history_dir=True,
        output_dir_arg="--output-dir",
    ),
    AnalyzerSpec(
        name="shs",
        script="scripts/investigate_should_have_signaled.py",
        outputs=("should_have_signaled_cases_{date}.csv", "should_have_signaled_summary_{date}.csv", "should_have_signaled_summary_ALL.csv"),
        summary_targets=("should_have_signaled_summary_{date}.csv",),
        supports_force=True,
        pass_sqlite_path=True,
        output_dir_arg="--output-dir",
    ),
    AnalyzerSpec(
        name="nbas",
        script="scripts/investigate_no_buy_after_signal.py",
        outputs=("no_buy_after_signal_cases_{date}.csv", "no_buy_after_signal_summary_{date}.csv", "no_buy_after_signal_summary_ALL.csv"),
        summary_targets=("no_buy_after_signal_summary_{date}.csv",),
        supports_force=True,
        pass_sqlite_path=True,
        output_dir_arg="--output-dir",
        category="forensic",
    ),
    AnalyzerSpec(
        name="offline_runtime_pre_signal",
        script="scripts/investigate_offline_runtime_pre_signal.py",
        outputs=("offline_runtime_pre_signal_cases_{date}.csv", "offline_runtime_pre_signal_summary_{date}.csv", "offline_runtime_pre_signal_summary_ALL.csv"),
        summary_targets=("offline_runtime_pre_signal_summary_{date}.csv",),
        supports_force=True,
        pass_sqlite_path=True,
        pass_history_dir=True,
        pass_signal_thresholds=True,
        output_dir_arg="--output-dir",
        category="forensic",
    ),
    AnalyzerSpec(
        name="top100_buy",
        script="scripts/analyze_top100_buy.py",
        outputs=(
            "top100_buy_symbol_day_{date}.csv",
            "top100_buy_snapshots_{date}.parquet",
            "top100_buy_feature_analysis_{date}.csv",
            "top100_buy_filter_simulation_{date}.csv",
            "top100_buy_portfolio_replay_{date}.csv",
            "top100_buy_summary_{date}.md",
            "top100_buy_data_quality_{date}.json",
        ),
        summary_targets=(
            "top100_buy_summary_{date}.md",
            "top100_buy_filter_simulation_{date}.csv",
            "top100_buy_data_quality_{date}.json",
        ),
        supports_force=True,
        pass_sqlite_path=True,
        pass_history_dir=True,
        pass_signal_thresholds=True,
        output_dir_arg="--output-dir",
        category="strategy",
    ),
)


@dataclass
class StepResult:
    name: str
    command: list[str]
    skipped: bool
    status: str
    exit_code: int | None
    elapsed_seconds: float
    output_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expected_output_files: list[str] = field(default_factory=list)
    existing_output_files: list[str] = field(default_factory=list)
    row_counts: dict[str, int | None] = field(default_factory=dict)
    failure_traceback_summary: str = ""
    output_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expected_output_files: list[str] = field(default_factory=list)
    existing_output_files: list[str] = field(default_factory=list)
    row_counts: dict[str, int | None] = field(default_factory=dict)
    failure_traceback_summary: str = ""


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
    if path.suffix.lower() == ".md":
        if not path.exists():
            return [f"- `{rel(path)}`: missing"]
        text = path.read_text(encoding="utf-8", errors="replace")
        details = []
        for prefix in ("FACT:", "HYPOTHESIS:", "NOT AVAILABLE:", "BASELINE ONLY:"):
            line = next((line.strip() for line in text.splitlines() if line.startswith(prefix)), "")
            if line:
                details.append(line)
        return [f"- `{rel(path)}`: {' | '.join(details) if details else 'written'}"]
    if path.suffix.lower() == ".json":
        if not path.exists():
            return [f"- `{rel(path)}`: missing"]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return [f"- `{rel(path)}`: written"]
        bits = []
        if data.get("premarket_feature_coverage"):
            bits.append(f"premarket_feature_coverage={data.get('premarket_feature_coverage')}")
        return [f"- `{rel(path)}`: {', '.join(bits) if bits else 'written'}"]
    fields, rows = csv_rows(path)
    if not path.exists():
        return [f"- `{rel(path)}`: missing"]
    lines = [f"- `{rel(path)}`: rows={len(rows)}"]
    if not rows:
        return lines
    for col in ("net_pnl", "actual_net_pnl", "gross_pnl", "net_improvement"):
        if col in fields:
            lines.append(f"  - {col}_sum={format_float(numeric_sum(rows, col))}")
    if "open_to_high_pct" in fields:
        vals = []
        for row in rows:
            try: vals.append(float(row.get("open_to_high_pct") or ""))
            except ValueError: pass
        if vals: lines.append(f"  - max_open_to_high_pct={format_float(max(vals))}")
    for column in ("final_classification", "final_no_buy_reason", "missed_reason_group", "top100_no_signal_reason", "data_quality", "source_bucket", "feature", "coverage", "bad_entry_label"):
        if column in fields:
            lines.append(f"  - {column}: " + ", ".join(f"{k}={v}" for k, v in value_counts(rows, column)))
    if len(rows) == 1:
        interesting = ["date", "session_date", "total_should_have_signaled", "runtime_signal_ready_but_no_buy", "bought_late", "restart_blocked", "total_cases", "offline_should_have_signaled_runtime_signal_not_observed", "coverage_pct", "capture_pct"]
        pairs = [f"{k}={rows[0].get(k)}" for k in interesting if k in fields and rows[0].get(k) not in (None, "")]
        if pairs: lines.append(f"  - summary: {', '.join(pairs)}")
    return lines


def output_paths(spec: AnalyzerSpec, session_date: str, output_dir: Path) -> list[Path]:
    return [output_dir / pattern.format(date=session_date) for pattern in spec.outputs]


def summary_paths(spec: AnalyzerSpec, session_date: str, output_dir: Path) -> list[Path]:
    return [output_dir / pattern.format(date=session_date) for pattern in spec.summary_targets]


def command_for(spec: AnalyzerSpec, session_date: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, spec.script]
    if spec.extra_args:
        command.extend(part.format(date=session_date) for part in spec.extra_args)
    else:
        command.extend(["--date", session_date])
    if spec.supports_force and not args.no_force:
        command.append("--force")
    if spec.pass_sqlite_path:
        command.extend(["--sqlite-path", str(args.sqlite_path)])
    if spec.pass_history_dir:
        command.extend(["--history-dir", str(args.history_dir)])
    thresholds = (
        getattr(args, "min_first_5m_high_pct", None),
        getattr(args, "min_first_15m_high_pct", None),
        getattr(args, "min_or_range_pct", None),
    )
    if spec.pass_signal_thresholds and all(value is not None for value in thresholds):
        command.extend(
            [
                "--min-first-5m-high-pct", str(thresholds[0]),
                "--min-first-15m-high-pct", str(thresholds[1]),
                "--min-or-range-pct", str(thresholds[2]),
            ]
        )
    if spec.name == "shs":
        command.extend(["--missed-runners-csv", str(args.output_dir / f"missed_runners_{session_date}.csv")])
    if spec.name == "nbas":
        command.extend(["--should-have-signaled-csv", str(args.output_dir / f"should_have_signaled_cases_{session_date}.csv")])
    if spec.name == "offline_runtime_pre_signal":
        command.extend(["--cases-csv", str(args.output_dir / f"no_buy_after_signal_cases_{session_date}.csv")])
    if spec.output_dir_arg:
        command.extend([spec.output_dir_arg, str(args.output_dir)])
    if spec.output_file_pattern:
        command.extend(["--output", str(args.output_dir / spec.output_file_pattern.format(date=session_date))])
    return command


def selected_specs(args: argparse.Namespace) -> list[AnalyzerSpec]:
    specs = list(ANALYZER_REGISTRY)
    names = {s.name for s in specs}
    only = set(args.only or [])
    skips = set(args.skip or [])
    for name in list(only | skips):
        if name not in names:
            raise SystemExit(f"unknown analyzer {name!r}; use --list")
    # Backward compatible dedicated skip flags.
    legacy_skips = {
        "coverage": args.skip_coverage,
        "missed": args.skip_missed,
        "bad_entries": args.skip_bad_entries,
        "shs": args.skip_shs,
        "nbas": args.skip_nbas,
        "offline_runtime_pre_signal": args.skip_offline_runtime_pre_signal,
        "top100_buy": getattr(args, "skip_top100_buy", False),
    }
    skips |= {name for name, enabled in legacy_skips.items() if enabled}
    if only:
        specs = [s for s in specs if s.name in only]
    return [s for s in specs if s.name not in skips]


def build_steps(session_date: str, args: argparse.Namespace) -> list[tuple[str, list[str], bool]]:
    return [(spec.name, command_for(spec, session_date, args), False) for spec in selected_specs(args)]


def expected_outputs(session_date: str, output_dir: Path | None = None) -> list[Path]:
    out = output_dir or DEFAULT_ANALYSIS_DIR
    paths: list[Path] = []
    for spec in ANALYZER_REGISTRY:
        paths.extend(output_paths(spec, session_date, out))
    return paths


def row_count_for_path(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() != ".csv" or path.stat().st_size <= 0:
        return None
    _fields, rows = csv_rows(path)
    return len(rows)


def output_row_counts(paths: list[Path]) -> dict[str, int | None]:
    return {rel(path): row_count_for_path(path) for path in paths}


def traceback_summary(stdout: str, stderr: str) -> str:
    text = "\n".join(part for part in (stdout, stderr) if part)
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tb_start = next((idx for idx, line in enumerate(lines) if line.startswith("Traceback ")), None)
    if tb_start is not None:
        return " | ".join(lines[tb_start:][-8:])[:2000]
    return " | ".join(lines[-5:])[:1000]


def missed_should_have_signaled_count(output_dir: Path, session_date: str) -> int:
    path = output_dir / f"missed_runners_{session_date}.csv"
    fields, rows = csv_rows(path)
    if "top100_no_signal_reason" not in fields:
        return 0
    return sum(1 for row in rows if str(row.get("top100_no_signal_reason") or "") == "should_have_signaled")


def shs_targets_count(output_dir: Path, session_date: str) -> int:
    path = output_dir / f"should_have_signaled_summary_{session_date}.csv"
    fields, rows = csv_rows(path)
    if not rows or "total_should_have_signaled" not in fields:
        return 0
    try:
        return int(float(rows[0].get("total_should_have_signaled") or 0))
    except ValueError:
        return 0


def validate_shs_handoff(result: StepResult, *, output_dir: Path, session_date: str) -> None:
    if result.name != "shs":
        return
    missed_count = missed_should_have_signaled_count(output_dir, session_date)
    target_count = shs_targets_count(output_dir, session_date)
    result.warnings.append(f"missed_should_have_signaled_count={missed_count}")
    result.warnings.append(f"shs_targets_count={target_count}")
    result.warnings.append(f"shs_target_handoff_match={int(missed_count == target_count)}")
    if missed_count > 0 and target_count == 0:
        result.status = "failed"
        result.failure_traceback_summary = (
            f"SHS target handoff failed: missed_should_have_signaled_count={missed_count} "
            f"but shs_targets_count={target_count}"
        )


def git_commit() -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
        return completed.stdout.strip()
    except Exception:
        return ""


def write_manifest(session_date: str, results: list[StepResult], *, args: argparse.Namespace, started_at: str, completed_at: str, final_failed: bool) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "run_manifest.json"
    payload = {
        "session_date": session_date,
        "started_at": started_at,
        "completed_at": completed_at,
        "git_commit": git_commit(),
        "sqlite_path": str(args.sqlite_path),
        "history_dir": str(args.history_dir),
        "signal_threshold_cli_override": {
            "effective_min_first5": getattr(args, "min_first_5m_high_pct", None),
            "effective_min_first15": getattr(args, "min_first_15m_high_pct", None),
            "effective_min_or_range": getattr(args, "min_or_range_pct", None),
            "config_source": "cli_explicit" if getattr(args, "min_first_5m_high_pct", None) is not None else "session_run_metadata",
        },
        "status": "FAILED" if final_failed else "OK",
        "successful_steps": [result.name for result in results if result.status == "ok"],
        "failed_steps": [result.name for result in results if result.status == "failed"],
        "degraded_steps": [result.name for result in results if result.status == "degraded" or result.warnings],
        "analysis_complete": not any(result.status == "failed" for result in results),
        "missed_should_have_signaled_count": missed_should_have_signaled_count(args.output_dir, session_date),
        "shs_targets_count": shs_targets_count(args.output_dir, session_date),
        "shs_target_handoff_match": missed_should_have_signaled_count(args.output_dir, session_date) == shs_targets_count(args.output_dir, session_date),
        "analyzers": [result.__dict__ for result in results],
        "data_quality_summary": collect_data_quality_summary(session_date, args.output_dir),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def collect_data_quality_summary(session_date: str, output_dir: Path) -> dict[str, object]:
    path = output_dir / f"bad_entries_data_quality_{session_date}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"warning": "failed_to_read_bad_entries_data_quality"}


def write_strategy_summary(session_date: str, results: list[StepResult], *, total_elapsed: float, final_failed: bool, args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"strategy_analysis_summary_{session_date}.md"
    lines = [
        f"# Strategy Analysis Summary {session_date}", "",
        f"- final_status: {'FAILED' if final_failed else 'OK'}",
        f"- elapsed_seconds: {total_elapsed:.2f}",
        f"- sqlite_path: `{args.sqlite_path}`",
        f"- history_dir: `{args.history_dir}`", "",
        "FACT: All steps are read-only analyzers over finalized canonical trades, parquet history, recorder evidence, and runtime SQLite.",
        "HYPOTHESIS: Filter and early-exit results are candidates for investigation, not live strategy changes.",
        "NOT AVAILABLE: Missing or all-null feature groups, including premarket features for baseline sessions, are marked unavailable_for_session.",
        "BASELINE ONLY: Single-session results are baseline diagnostics.",
        "REQUIRES MULTI-DAY VALIDATION: Any proposed filter must be tested across more sessions.",
        "POSSIBLE OVERFITTING: One-day high-performing filters can overfit.", "",
        "## Step Status", "",
        "| analyzer | status | exit_code | elapsed_seconds | outputs |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(f"| {result.name} | {result.status} | {'' if result.exit_code is None else result.exit_code} | {result.elapsed_seconds:.2f} | {len(result.output_files)} |")
    sections = [
        ("Data quality", [args.output_dir / f"bad_entries_data_quality_{session_date}.json"]),
        ("Baseline results", [args.output_dir / f"bad_entries_{session_date}.csv", args.output_dir / f"coverage_report_{session_date}.csv"]),
        ("Entry timing", [args.output_dir / f"bad_entries_time_buckets_{session_date}.csv"]),
        ("Bad entry patterns", [args.output_dir / f"bad_entries_feature_buckets_{session_date}.csv", args.output_dir / f"bad_entries_filter_simulation_{session_date}.csv", args.output_dir / f"bad_entries_recommendations_{session_date}.md"]),
        ("Early loser exits", [args.output_dir / f"early_loser_rules_{session_date}.csv", args.output_dir / f"early_loser_summary_{session_date}.md"]),
        ("Stop loss strategy", [
            args.output_dir / f"stop_loss_fixed_grid_{session_date}.csv",
            args.output_dir / f"stop_loss_activation_delay_{session_date}.csv",
            args.output_dir / f"stop_loss_slippage_sensitivity_{session_date}.csv",
            args.output_dir / f"stop_loss_segment_analysis_{session_date}.csv",
            args.output_dir / f"stop_loss_dynamic_rules_{session_date}.csv",
            args.output_dir / f"stop_loss_hybrid_rules_{session_date}.csv",
            args.output_dir / f"stop_loss_data_quality_{session_date}.json",
            args.output_dir / f"stop_loss_recommendations_{session_date}.md",
        ]),
        ("Missed runners", [args.output_dir / f"missed_runners_{session_date}.csv"]),
        ("Top100 coverage", [args.output_dir / f"coverage_report_{session_date}.csv"]),
        ("Should-have-signal gaps", [args.output_dir / f"should_have_signaled_summary_{session_date}.csv", args.output_dir / f"no_buy_after_signal_summary_{session_date}.csv", args.output_dir / f"offline_runtime_pre_signal_summary_{session_date}.csv"]),
        ("Recommended next experiments", [args.output_dir / f"bad_entries_recommendations_{session_date}.md"]),
    ]
    for title, targets in sections:
        lines.extend(["", f"## {title}", ""])
        for target in targets:
            lines.extend(summarize_csv(target))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_summary(session_date: str, results: list[StepResult], *, total_elapsed: float, final_failed: bool, output_dir: Path | None = None) -> Path:
    out = output_dir or DEFAULT_ANALYSIS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"daily_analysis_summary_{session_date}.md"
    generated = [p for p in expected_outputs(session_date, out) if p.exists()]
    lines = [f"# Daily Analysis Summary {session_date}", "", f"- final_status: {'FAILED' if final_failed else 'OK'}", f"- elapsed_seconds: {total_elapsed:.2f}", "", "## Steps", "", "| step | status | exit_code | elapsed_seconds | command |", "|---|---:|---:|---:|---|"]
    for result in results:
        lines.append(f"| {result.name} | {result.status} | {'' if result.exit_code is None else result.exit_code} | {result.elapsed_seconds:.2f} | `{' '.join(result.command)}` |")
    lines.extend(["", "## Generated Output Files", ""])
    lines.extend([f"- `{rel(p)}`" for p in generated] or ["- none found"])
    lines.extend(["", "## CSV Summaries", ""])
    for spec in ANALYZER_REGISTRY:
        for target in summary_paths(spec, session_date, out):
            lines.extend(summarize_csv(target))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_step(spec: AnalyzerSpec, command: list[str], *, args: argparse.Namespace, session_date: str) -> StepResult:
    print(f"DAILY_ANALYSIS_STEP_START name={spec.name} command={' '.join(command)}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    elapsed = time.monotonic() - started
    status = "ok" if completed.returncode == 0 else "failed"
    print(("DAILY_ANALYSIS_STEP_DONE" if completed.returncode == 0 else "DAILY_ANALYSIS_STEP_FAILED") + f" name={spec.name} exit_code={completed.returncode} elapsed={elapsed:.2f}", flush=True)
    expected = output_paths(spec, session_date, args.output_dir)
    existing = [path for path in expected if path.exists()]
    return StepResult(
        spec.name,
        command,
        False,
        status,
        completed.returncode,
        elapsed,
        [rel(path) for path in existing],
        [],
        [rel(path) for path in expected],
        [rel(path) for path in existing],
        output_row_counts(expected),
        "" if completed.returncode == 0 else traceback_summary(completed.stdout, completed.stderr),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full read-only daily strategy analysis suite for one completed session.")
    parser.add_argument("--date", required=False, type=parse_date, help="Completed session date, YYYY-MM-DD.")
    parser.add_argument("--sqlite-path", type=Path, default=Path("data/runtime/trading_runtime.sqlite"))
    parser.add_argument("--history-dir", type=Path, default=Path("data/history/universe_1m"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--min-first-5m-high-pct", type=float, default=None)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=None)
    parser.add_argument("--min-or-range-pct", type=float, default=None)
    parser.add_argument("--only", action="append", help="Run only the named analyzer. Can be repeated.")
    parser.add_argument("--skip", action="append", help="Skip the named analyzer. Can be repeated.")
    parser.add_argument("--list", action="store_true", help="List registered analyzers and exit.")
    parser.add_argument("--fail-fast", "--stop-on-failure", dest="stop_on_failure", action="store_true")
    parser.add_argument("--no-force", action="store_true", help="Do not pass --force to analyzers that support it.")
    # Backward-compatible dedicated skip flags.
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-missed", action="store_true")
    parser.add_argument("--skip-bad-entries", action="store_true")
    parser.add_argument("--skip-bad-entry-details", action="store_true")
    parser.add_argument("--skip-bad-entry-patterns", action="store_true")
    parser.add_argument("--skip-shs", action="store_true")
    parser.add_argument("--skip-nbas", action="store_true")
    parser.add_argument("--skip-offline-runtime-pre-signal", action="store_true")
    parser.add_argument("--skip-top100-buy", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.list:
        for spec in ANALYZER_REGISTRY:
            print(f"{spec.name}\t{spec.category}\t{spec.script}")
        return 0
    if not args.date:
        parser.error("--date is required unless --list is used")
    threshold_override = (
        args.min_first_5m_high_pct,
        args.min_first_15m_high_pct,
        args.min_or_range_pct,
    )
    if any(value is not None for value in threshold_override) and not all(value is not None for value in threshold_override):
        parser.error(
            "--min-first-5m-high-pct, --min-first-15m-high-pct and "
            "--min-or-range-pct must be supplied together"
        )
    session_date = args.date
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"DAILY_ANALYSIS_START date={session_date} output_dir={args.output_dir}", flush=True)
    total_started = time.monotonic()
    results: list[StepResult] = []
    failed = False
    specs = selected_specs(args)
    for spec in specs:
        result = run_step(spec, command_for(spec, session_date, args), args=args, session_date=session_date)
        validate_shs_handoff(result, output_dir=args.output_dir, session_date=session_date)
        results.append(result)
        if result.status == "failed":
            failed = True
            if args.stop_on_failure:
                break
    total_elapsed = time.monotonic() - total_started
    completed_at = datetime.now(timezone.utc).isoformat()
    daily_summary = write_summary(session_date, results, total_elapsed=total_elapsed, final_failed=failed, output_dir=args.output_dir)
    strategy_summary = write_strategy_summary(session_date, results, total_elapsed=total_elapsed, final_failed=failed, args=args)
    manifest = write_manifest(session_date, results, args=args, started_at=started_at, completed_at=completed_at, final_failed=failed)
    print(f"DAILY_ANALYSIS_SUMMARY_WRITTEN path={rel(daily_summary)}", flush=True)
    print(f"STRATEGY_ANALYSIS_SUMMARY_WRITTEN path={rel(strategy_summary)}", flush=True)
    print(f"DAILY_ANALYSIS_MANIFEST_WRITTEN path={rel(manifest)}", flush=True)
    print(f"DAILY_ANALYSIS_DONE date={session_date} elapsed={total_elapsed:.2f} status={'FAILED' if failed else 'OK'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
