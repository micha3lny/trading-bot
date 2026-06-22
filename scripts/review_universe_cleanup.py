#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_UNIVERSE = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_DIAGNOSTICS_GLOB = "data/universe/daily_top100_*_diagnostics.csv"
DEFAULT_COLLECTOR_STATUS = "data/history/collector_status.json"
DEFAULT_COLLECTOR_FAILURES = "data/history/collector_failures.json"
DEFAULT_OUTPUT = "data/universe/universe_cleanup_candidates.csv"

OUTPUT_COLUMNS = [
    "symbol",
    "reason",
    "count",
    "first_seen_date",
    "last_seen_date",
    "suggested_action",
    "top100_missing_count",
    "top100_rejected_count",
    "top100_error_count",
    "collector_no_data_count",
    "collector_partial_count",
    "collector_failed_count",
    "statuses",
    "examples",
    "notes",
    "approved",
]


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip()


def parse_date_safe(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def date_from_diagnostics_path(path: Path) -> date | None:
    stem = path.stem
    prefix = "daily_top100_"
    suffix = "_diagnostics"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None
    return parse_date_safe(stem[len(prefix) : -len(suffix)])


def include_date(day: date | None, *, start_date: date | None, end_date: date | None) -> bool:
    if day is None:
        return True
    if start_date and day < start_date:
        return False
    if end_date and day > end_date:
        return False
    return True


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_reason(status: str, reason: str) -> tuple[str, str]:
    reason_l = str(reason or "").lower()
    status_l = str(status or "").lower()
    if "kid" in reason_l or "priip" in reason_l or "no_trading_permission" in reason_l or "no trading permission" in reason_l:
        return "kid_priip_ineligible", "denylist"
    if "error 200" in reason_l or "no security definition" in reason_l or "qualify_failed" in reason_l:
        return "invalid_contract", "remove_from_universe"
    if "error 162" in reason_l or "historical market data service error" in reason_l or status_l in {"no_data", "no_data_permanent"}:
        return "ibkr_no_data", "remove_from_universe"
    if "empty_bars" in reason_l:
        return "ibkr_no_data", "remove_from_universe"
    if "too_few_bars" in reason_l or "rows_below_threshold" in reason_l:
        return "too_few_bars", "review"
    if "price_too_low" in reason_l:
        return "price_too_low", "review"
    if "volume_too_low" in reason_l:
        return "volume_too_low", "review"
    if "dollar_volume_too_low" in reason_l:
        return "dollar_volume_too_low", "review"
    if any(token in reason_l for token in ["warrant", "unit", "rights", "preferred", "non_stock_sectype", "product_keyword"]):
        return "non_common_stock_product", "remove_from_universe"
    if status_l == "missing" or "missing_history" in reason_l:
        return "missing_history", "investigate_history"
    if status_l == "partial":
        return "partial_history", "retry_history"
    if status_l in {"failed", "failed_permanent", "error"}:
        return "collector_or_ranking_error", "investigate_history"
    return reason_l.replace(" ", "_")[:80] or status_l or "unknown", "review"


@dataclass
class CandidateStats:
    symbol: str
    reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    actions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    statuses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    examples: list[str] = field(default_factory=list)
    first_seen: date | None = None
    last_seen: date | None = None
    top100_missing_count: int = 0
    top100_rejected_count: int = 0
    top100_error_count: int = 0
    collector_no_data_count: int = 0
    collector_partial_count: int = 0
    collector_failed_count: int = 0

    def add(self, *, day: date | None, status: str, reason: str, source: str) -> None:
        normalized_reason, action = classify_reason(status, reason)
        self.reasons[normalized_reason] += 1
        self.actions[action] += 1
        self.statuses[str(status or "unknown")] += 1
        if day is not None:
            self.first_seen = day if self.first_seen is None else min(self.first_seen, day)
            self.last_seen = day if self.last_seen is None else max(self.last_seen, day)
        example = f"{source}:{day.isoformat() if day else 'unknown'}:{status}:{reason}"
        if len(self.examples) < 5 and example not in self.examples:
            self.examples.append(example)
        status_l = str(status or "").lower()
        if source == "top100":
            if status_l == "missing":
                self.top100_missing_count += 1
            elif status_l == "rejected":
                self.top100_rejected_count += 1
            elif status_l == "error":
                self.top100_error_count += 1
        else:
            if status_l in {"no_data", "no_data_permanent"}:
                self.collector_no_data_count += 1
            elif status_l == "partial":
                self.collector_partial_count += 1
            elif status_l in {"failed", "failed_permanent"}:
                self.collector_failed_count += 1

    @property
    def count(self) -> int:
        return sum(self.reasons.values())

    def primary_reason(self) -> str:
        return sorted(self.reasons.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def suggested_action(self) -> str:
        priority = ["denylist", "remove_from_universe", "retry_history", "investigate_history", "review"]
        ranked = sorted(self.actions.items(), key=lambda item: (priority.index(item[0]) if item[0] in priority else 99, -item[1], item[0]))
        return ranked[0][0] if ranked else "review"

    def to_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "reason": self.primary_reason(),
            "count": self.count,
            "first_seen_date": self.first_seen.isoformat() if self.first_seen else "",
            "last_seen_date": self.last_seen.isoformat() if self.last_seen else "",
            "suggested_action": self.suggested_action(),
            "top100_missing_count": self.top100_missing_count,
            "top100_rejected_count": self.top100_rejected_count,
            "top100_error_count": self.top100_error_count,
            "collector_no_data_count": self.collector_no_data_count,
            "collector_partial_count": self.collector_partial_count,
            "collector_failed_count": self.collector_failed_count,
            "statuses": ";".join(f"{k}:{v}" for k, v in sorted(self.statuses.items())),
            "examples": " | ".join(self.examples),
            "notes": "",
            "approved": "",
        }


def load_top100_diagnostics(
    pattern: str,
    *,
    start_date: date | None,
    end_date: date | None,
) -> dict[str, CandidateStats]:
    candidates: dict[str, CandidateStats] = {}
    for raw_path in sorted(glob.glob(pattern)):
        path = Path(raw_path)
        day = date_from_diagnostics_path(path)
        if not include_date(day, start_date=start_date, end_date=end_date):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for row in df.to_dict("records"):
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            status = str(row.get("status") or "")
            reason = str(row.get("reason") or "")
            if status.lower() not in {"missing", "rejected", "error", "excluded_ineligible"}:
                continue
            stats = candidates.setdefault(symbol, CandidateStats(symbol=symbol))
            stats.add(day=parse_date_safe(row.get("date")) or day, status=status, reason=reason, source="top100")
    return candidates


def load_collector_status(
    status_path: str | Path,
    failures_path: str | Path,
    *,
    start_date: date | None,
    end_date: date | None,
) -> dict[str, CandidateStats]:
    out: dict[str, CandidateStats] = {}
    status_rows = load_json(status_path)
    failure_rows = load_json(failures_path)
    if not isinstance(status_rows, dict):
        status_rows = {}
    if not isinstance(failure_rows, dict):
        failure_rows = {}
    for key, row in status_rows.items():
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol") or str(key).split("_", 1)[0])
        day = parse_date_safe(row.get("date"))
        if not symbol or not include_date(day, start_date=start_date, end_date=end_date):
            continue
        status = str(row.get("status") or "")
        if status.lower() not in {"partial", "no_data", "no_data_permanent", "failed", "failed_permanent"}:
            continue
        failure = failure_rows.get(key) if isinstance(failure_rows, dict) else {}
        last_error = ""
        if isinstance(failure, dict):
            last_error = str(failure.get("last_error") or "")
        reason = str(row.get("last_error") or last_error or status)
        stats = out.setdefault(symbol, CandidateStats(symbol=symbol))
        stats.add(day=day, status=status, reason=reason, source="collector")
    return out


def merge_candidates(*groups: dict[str, CandidateStats]) -> dict[str, CandidateStats]:
    merged: dict[str, CandidateStats] = {}
    for group in groups:
        for symbol, stats in group.items():
            target = merged.setdefault(symbol, CandidateStats(symbol=symbol))
            for reason, count in stats.reasons.items():
                target.reasons[reason] += count
            for action, count in stats.actions.items():
                target.actions[action] += count
            for status, count in stats.statuses.items():
                target.statuses[status] += count
            target.examples.extend([ex for ex in stats.examples if ex not in target.examples][: max(0, 5 - len(target.examples))])
            target.first_seen = stats.first_seen if target.first_seen is None else (min(target.first_seen, stats.first_seen) if stats.first_seen else target.first_seen)
            target.last_seen = stats.last_seen if target.last_seen is None else (max(target.last_seen, stats.last_seen) if stats.last_seen else target.last_seen)
            target.top100_missing_count += stats.top100_missing_count
            target.top100_rejected_count += stats.top100_rejected_count
            target.top100_error_count += stats.top100_error_count
            target.collector_no_data_count += stats.collector_no_data_count
            target.collector_partial_count += stats.collector_partial_count
            target.collector_failed_count += stats.collector_failed_count
    return merged


def build_review_rows(
    *,
    diagnostics_glob: str,
    collector_status: str | Path,
    collector_failures: str | Path,
    min_count: int,
    start_date: date | None,
    end_date: date | None,
) -> list[dict[str, Any]]:
    top100 = load_top100_diagnostics(diagnostics_glob, start_date=start_date, end_date=end_date)
    collector = load_collector_status(collector_status, collector_failures, start_date=start_date, end_date=end_date)
    merged = merge_candidates(top100, collector)
    rows = [stats.to_row() for stats in merged.values() if stats.count >= int(min_count)]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["reason"]), str(row["symbol"])))
    return rows


def write_review_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output, index=False)


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "approved", "apply"}


def apply_review_cleanup(
    *,
    universe_path: str | Path,
    review_path: str | Path,
    output_universe: str | Path | None = None,
    backup: bool = True,
    apply_all_candidates: bool = False,
) -> dict[str, Any]:
    universe = Path(universe_path)
    review = Path(review_path)
    if not universe.exists():
        raise FileNotFoundError(f"Universe file not found: {universe}")
    if not review.exists():
        raise FileNotFoundError(f"Review file not found: {review}")
    universe_df = pd.read_csv(universe)
    if "symbol" not in universe_df.columns:
        raise ValueError("Universe CSV must contain symbol column")
    review_df = pd.read_csv(review).fillna("")
    if "symbol" not in review_df.columns:
        raise ValueError("Review CSV must contain symbol column")
    if apply_all_candidates:
        approved = review_df
    elif "approved" in review_df.columns:
        approved = review_df[review_df["approved"].map(truthy)]
    else:
        approved = review_df.iloc[0:0]
    removable_actions = {"remove_from_universe", "denylist"}
    if "suggested_action" in approved.columns:
        approved = approved[approved["suggested_action"].astype(str).isin(removable_actions)]
    symbols_to_remove = sorted({normalize_symbol(s) for s in approved["symbol"].tolist() if normalize_symbol(s)})
    before = len(universe_df)
    clean = universe_df[~universe_df["symbol"].astype(str).str.upper().str.strip().isin(symbols_to_remove)].copy()
    destination = Path(output_universe) if output_universe else universe
    if backup and destination == universe:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = universe.with_name(f"{universe.stem}.{stamp}.bak{universe.suffix}")
        shutil.copy2(universe, backup_path)
    else:
        backup_path = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(destination, index=False)
    return {
        "universe": str(universe),
        "output_universe": str(destination),
        "review": str(review),
        "before": before,
        "after": len(clean),
        "removed": before - len(clean),
        "removed_symbols": symbols_to_remove,
        "backup": str(backup_path) if backup_path else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Review recurring unusable symbols before cleaning the Top100 universe")
    parser.add_argument("--diagnostics-glob", default=DEFAULT_DIAGNOSTICS_GLOB)
    parser.add_argument("--collector-status", default=DEFAULT_COLLECTOR_STATUS)
    parser.add_argument("--collector-failures", default=DEFAULT_COLLECTOR_FAILURES)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--apply", action="store_true", help="Apply approved cleanup rows from the review CSV")
    parser.add_argument("--review-input", default=None, help="CSV to apply; defaults to --output")
    parser.add_argument("--output-universe", default=None, help="Write cleaned universe here instead of replacing --universe")
    parser.add_argument("--apply-all-candidates", action="store_true", help="Apply every remove/denylist candidate without approved=1")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    end_date = parse_date_safe(args.end_date) or date.today()
    start_date = parse_date_safe(args.start_date)
    if start_date is None and int(args.lookback_days or 0) > 0:
        start_date = end_date - timedelta(days=int(args.lookback_days) - 1)

    rows = build_review_rows(
        diagnostics_glob=args.diagnostics_glob,
        collector_status=args.collector_status,
        collector_failures=args.collector_failures,
        min_count=int(args.min_count),
        start_date=start_date,
        end_date=end_date,
    )
    write_review_csv(args.output, rows)
    print(
        json.dumps(
            {
                "output": args.output,
                "candidates": len(rows),
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "min_count": int(args.min_count),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.apply:
        result = apply_review_cleanup(
            universe_path=args.universe,
            review_path=args.review_input or args.output,
            output_universe=args.output_universe,
            backup=not bool(args.no_backup),
            apply_all_candidates=bool(args.apply_all_candidates),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
