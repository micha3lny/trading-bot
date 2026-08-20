from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.live_trading.analysis.common import load_top100, normalize_symbol, parse_dt
from src.live_trading.analysis.light_snapshot_scanner import BoundedLightScanner, snapshot_timestamp
from src.live_trading.market_calendar import get_us_equity_session, previous_us_equity_trading_day


INVALID_TOP100_PARITY = "INVALID_TOP100_PARITY"
TOP100_PARITY_OK = "OK"
_SOURCE_FIELDS = (
    "ranking_source_date",
    "runtime_top100_ranking_source_date",
    "top100_source_date",
    "top100_ranking_source_date",
)


def _valid_date(value: Any) -> str | None:
    text = str(value or "")[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _source_date_from_path(path: Path | None) -> str | None:
    if path is None or path.name == "daily_top100_latest.csv":
        return None
    return _valid_date(path.stem.removeprefix("daily_top100_"))


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or ""))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class Top100Generation:
    process_start_id: str
    ranking_source_date: str
    effective_at: pd.Timestamp
    last_seen_at: pd.Timestamp
    symbols: frozenset[str]
    entry_symbols: frozenset[str]
    retained_symbols: frozenset[str]
    scan_count: int
    first_scan_id: str
    last_scan_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "process_start_id": self.process_start_id,
            "ranking_source_date": self.ranking_source_date,
            "effective_at": self.effective_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "symbol_count": len(self.symbols),
            "entry_symbol_count": len(self.entry_symbols),
            "retained_symbol_count": len(self.retained_symbols),
            "scan_count": self.scan_count,
            "first_scan_id": self.first_scan_id,
            "last_scan_id": self.last_scan_id,
        }


@dataclass
class CausalTop100Resolution:
    trading_session_date: str
    runtime_top100_ranking_source_date: str | None
    analysis_top100_ranking_source_date: str | None
    top100_runtime_parity: str
    top100_source_path: Path | None
    top100_source_reason: str
    generations: list[Top100Generation] = field(default_factory=list)
    source_paths_by_date: dict[str, Path] = field(default_factory=dict)
    _symbols_by_source_date: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    @property
    def valid(self) -> bool:
        return self.top100_runtime_parity == TOP100_PARITY_OK

    @property
    def top100_generation_count(self) -> int:
        return len(self.generations)

    def output_fields(self) -> dict[str, Any]:
        desired = None
        if self.generations:
            desired = self._symbols_by_source_date.get(self.generations[-1].ranking_source_date)
        transition = top100_transition_validation(self.generations, desired_new_symbols=desired)
        return {
            "trading_session_date": self.trading_session_date,
            "runtime_top100_ranking_source_date": self.runtime_top100_ranking_source_date or "",
            "analysis_top100_ranking_source_date": self.analysis_top100_ranking_source_date or "",
            "top100_runtime_parity": self.top100_runtime_parity,
            "top100_source_path": str(self.top100_source_path or ""),
            "top100_source_reason": self.top100_source_reason,
            "top100_generation_count": self.top100_generation_count,
            "top100_transition_timeline": json.dumps(
                [generation.as_dict() for generation in self.generations], sort_keys=True
            ),
            "top100_transition_validation": json.dumps(transition, sort_keys=True),
        }

    def transition_validation(self, process_start_id: str | None = None) -> dict[str, Any]:
        desired = None
        candidates = self.generations
        if process_start_id:
            candidates = [item for item in candidates if item.process_start_id == process_start_id]
        if candidates:
            desired = self._symbols_by_source_date.get(candidates[-1].ranking_source_date)
        return top100_transition_validation(
            candidates,
            desired_new_symbols=desired,
            process_start_id=process_start_id,
        )

    def primary_top100(self) -> pd.DataFrame:
        if self.top100_source_path is None:
            return pd.DataFrame()
        return load_top100(self.top100_source_path)

    def generation_at(self, timestamp: Any) -> Top100Generation | None:
        target = parse_dt(timestamp)
        if target is None:
            return None
        eligible = [generation for generation in self.generations if generation.effective_at <= target]
        return max(eligible, key=lambda generation: generation.effective_at) if eligible else None

    def membership_at(self, symbol: str, timestamp: Any) -> bool | None:
        if not self.valid:
            return None
        generation = self.generation_at(timestamp)
        if generation is not None:
            return normalize_symbol(symbol) in generation.symbols
        if self.generations:
            return None
        source = self.analysis_top100_ranking_source_date
        symbols = self._symbols_by_source_date.get(str(source or ""))
        return normalize_symbol(symbol) in symbols if symbols is not None else None


def _flush_scan(
    generations: list[Top100Generation],
    key: tuple[str, str] | None,
    timestamp: pd.Timestamp | None,
    source_dates: set[str],
    symbols: set[str],
    entry_symbols: set[str],
    retained_symbols: set[str],
    in_top100_field_seen: bool,
) -> None:
    if not in_top100_field_seen:
        symbols = set(entry_symbols)
    if key is None or timestamp is None or len(source_dates) != 1 or not symbols:
        return
    process, scan_id = key
    source_date = next(iter(source_dates))
    frozen_symbols = frozenset(symbols)
    frozen_entry_symbols = frozenset(entry_symbols)
    frozen_retained_symbols = frozenset(retained_symbols)
    previous = generations[-1] if generations else None
    if (
        previous
        and previous.process_start_id == process
        and previous.ranking_source_date == source_date
        and previous.symbols == frozen_symbols
        and previous.entry_symbols == frozen_entry_symbols
    ):
        generations[-1] = Top100Generation(
            process_start_id=previous.process_start_id,
            ranking_source_date=previous.ranking_source_date,
            effective_at=previous.effective_at,
            last_seen_at=timestamp,
            symbols=previous.symbols,
            entry_symbols=previous.entry_symbols,
            retained_symbols=previous.retained_symbols | frozen_retained_symbols,
            scan_count=previous.scan_count + 1,
            first_scan_id=previous.first_scan_id,
            last_scan_id=scan_id,
        )
    else:
        generations.append(
            Top100Generation(
                process_start_id=process,
                ranking_source_date=source_date,
                effective_at=timestamp,
                last_seen_at=timestamp,
                symbols=frozen_symbols,
                entry_symbols=frozen_entry_symbols,
                retained_symbols=frozen_retained_symbols,
                scan_count=1,
                first_scan_id=scan_id,
                last_scan_id=scan_id,
            )
        )


def _flag(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return int(value) == 1
    except (TypeError, ValueError, OverflowError):
        return bool(value)


def _light_generations(recorder_dir: Path, session_date: str) -> list[Top100Generation]:
    columns = {
        "timestamp", "scan_completed_at", "scan_started_at", "process_start_id", "scan_id",
        "symbol", "ranking_source_date", "in_runtime_top100", "entry_symbol_allowed",
        "already_open", "contract_present", "ticker_present",
    }
    with BoundedLightScanner(recorder_dir, session_date, log_prefix="CAUSAL_TOP100_LIGHT") as scanner:
        scanner.build_index()
        summary_path = scanner.cache_generation_summary_path
        if summary_path is not None and summary_path.exists():
            try:
                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                return [
                    Top100Generation(
                        process_start_id=str(item["process_start_id"]),
                        ranking_source_date=str(item["ranking_source_date"]),
                        effective_at=pd.Timestamp(item["effective_at"]),
                        last_seen_at=pd.Timestamp(item["last_seen_at"]),
                        symbols=frozenset(item["symbols"]),
                        entry_symbols=frozenset(item["entry_symbols"]),
                        retained_symbols=frozenset(item["retained_symbols"]),
                        scan_count=int(item["scan_count"]),
                        first_scan_id=str(item["first_scan_id"]),
                        last_scan_id=str(item["last_scan_id"]),
                    )
                    for item in cached
                ]
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                pass
        return _light_generations_from_scanner(scanner, columns)


def _light_generations_from_scanner(
    scanner: BoundedLightScanner,
    columns: Iterable[str] | None = None,
) -> list[Top100Generation]:
    requested = set(columns or {
        "timestamp", "scan_completed_at", "scan_started_at", "process_start_id", "scan_id",
        "symbol", "ranking_source_date", "in_runtime_top100", "entry_symbol_allowed",
        "already_open", "contract_present", "ticker_present",
    })
    generations: list[Top100Generation] = []
    current_key: tuple[str, str] | None = None
    current_timestamp: pd.Timestamp | None = None
    current_sources: set[str] = set()
    current_symbols: set[str] = set()
    current_entry_symbols: set[str] = set()
    current_retained_symbols: set[str] = set()
    current_in_top100_field_seen = False
    for row in scanner.iter_rows(requested):
        key = (str(row.get("process_start_id") or ""), str(row.get("scan_id") or ""))
        if current_key is not None and key != current_key:
            _flush_scan(
                generations,
                current_key,
                current_timestamp,
                current_sources,
                current_symbols,
                current_entry_symbols,
                current_retained_symbols,
                current_in_top100_field_seen,
            )
            current_timestamp, current_sources, current_symbols = None, set(), set()
            current_entry_symbols, current_retained_symbols = set(), set()
            current_in_top100_field_seen = False
        current_key = key
        current_timestamp = snapshot_timestamp(row) or current_timestamp
        source_date = _valid_date(row.get("ranking_source_date"))
        if source_date:
            current_sources.add(source_date)
        symbol = normalize_symbol(row.get("symbol"))
        raw_in_top100 = row.get("in_runtime_top100")
        if raw_in_top100 is not None and not pd.isna(raw_in_top100):
            current_in_top100_field_seen = True
        in_top100 = _flag(raw_in_top100)
        entry_allowed = _flag(row.get("entry_symbol_allowed"))
        if symbol:
            if in_top100:
                current_symbols.add(symbol)
            if entry_allowed:
                current_entry_symbols.add(symbol)
            if not in_top100 and not entry_allowed and (
                _flag(row.get("already_open"))
                or _flag(row.get("contract_present"))
                or _flag(row.get("ticker_present"))
            ):
                current_retained_symbols.add(symbol)
    _flush_scan(
        generations,
        current_key,
        current_timestamp,
        current_sources,
        current_symbols,
        current_entry_symbols,
        current_retained_symbols,
        current_in_top100_field_seen,
    )
    return generations


def _runtime_metadata_source(recorder_dir: Path, session_date: str) -> str | None:
    path = recorder_dir / session_date / "run_metadata.csv"
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    for row in reversed(rows):
        if str(row.get("session_date") or "")[:10] not in ("", session_date):
            continue
        metadata = {**_json_mapping(row.get("metadata_json")), **row}
        for field_name in _SOURCE_FIELDS:
            source_date = _valid_date(metadata.get(field_name))
            if source_date:
                return source_date
    return None


def _dated_path(top100_dir: Path, source_date: str | None) -> Path | None:
    if not source_date:
        return None
    path = top100_dir / f"daily_top100_{source_date}.csv"
    return path if path.exists() else None


def _session_generations(generations: Iterable[Top100Generation], session_date: str) -> list[Top100Generation]:
    session = get_us_equity_session(date.fromisoformat(session_date))
    if not session.open_utc or not session.close_utc:
        return []
    open_time, close_time = pd.Timestamp(session.open_utc), pd.Timestamp(session.close_utc)
    ordered = sorted(generations, key=lambda generation: generation.effective_at)
    before_open = [generation for generation in ordered if generation.effective_at <= open_time]
    during = [generation for generation in ordered if open_time < generation.effective_at <= close_time]
    selected = ([before_open[-1]] if before_open else []) + during
    if not selected:
        selected = [generation for generation in ordered if generation.effective_at <= close_time]
        selected = selected[-1:] if selected else []
    return selected


def resolve_causal_top100(
    *,
    session_date: str,
    top100_dir: str | Path,
    recorder_dir: str | Path,
    explicit_top100: str | Path | None = None,
) -> CausalTop100Resolution:
    root = Path(top100_dir)
    recorder_root = Path(recorder_dir)
    all_generations = _light_generations(recorder_root, session_date)
    causal_generations = _session_generations(all_generations, session_date)
    source_paths: dict[str, Path] = {}
    symbols_by_source: dict[str, frozenset[str]] = {}
    for generation in all_generations:
        path = _dated_path(root, generation.ranking_source_date)
        if path is not None:
            source_paths[generation.ranking_source_date] = path
            symbols_by_source[generation.ranking_source_date] = frozenset(load_top100(path)["symbol"].map(normalize_symbol))

    if causal_generations:
        runtime_source = causal_generations[0].ranking_source_date
        primary_path = source_paths.get(runtime_source)
        complete = all(generation.ranking_source_date in source_paths for generation in causal_generations)
        parity = TOP100_PARITY_OK if complete and primary_path is not None else INVALID_TOP100_PARITY
        reason = "causal_light_trading_window" if parity == TOP100_PARITY_OK else "causal_light_source_file_missing"
        return CausalTop100Resolution(
            session_date, runtime_source, runtime_source, parity, primary_path, reason,
            all_generations, source_paths, symbols_by_source,
        )

    metadata_source = _runtime_metadata_source(recorder_root, session_date)
    metadata_path = _dated_path(root, metadata_source)
    if metadata_source:
        parity = TOP100_PARITY_OK if metadata_path is not None else INVALID_TOP100_PARITY
        if metadata_path is not None:
            symbols_by_source[metadata_source] = frozenset(load_top100(metadata_path)["symbol"].map(normalize_symbol))
            source_paths[metadata_source] = metadata_path
        return CausalTop100Resolution(
            session_date, metadata_source, metadata_source, parity, metadata_path,
            "exact_session_runtime_metadata" if parity == TOP100_PARITY_OK else "runtime_metadata_source_file_missing",
            all_generations, source_paths, symbols_by_source,
        )

    explicit = Path(explicit_top100) if explicit_top100 is not None else None
    if explicit is not None:
        explicit_source = _source_date_from_path(explicit)
        if explicit.name == "daily_top100_latest.csv":
            return CausalTop100Resolution(
                session_date, None, None, INVALID_TOP100_PARITY, None,
                "latest_file_rejected_without_runtime_provenance", all_generations,
            )
        if explicit.exists() and explicit_source:
            symbols_by_source[explicit_source] = frozenset(load_top100(explicit)["symbol"].map(normalize_symbol))
            return CausalTop100Resolution(
                session_date, None, explicit_source, INVALID_TOP100_PARITY, explicit,
                "explicit_cli_override_unverified_against_runtime", all_generations,
                {explicit_source: explicit}, symbols_by_source,
            )

    fallback_source = previous_us_equity_trading_day(date.fromisoformat(session_date)).isoformat()
    fallback_path = _dated_path(root, fallback_source)
    if fallback_path is not None:
        symbols_by_source[fallback_source] = frozenset(load_top100(fallback_path)["symbol"].map(normalize_symbol))
        return CausalTop100Resolution(
            session_date, None, fallback_source, INVALID_TOP100_PARITY, fallback_path,
            "controlled_previous_session_fallback_unverified", all_generations,
            {fallback_source: fallback_path}, symbols_by_source,
        )
    return CausalTop100Resolution(
        session_date, None, None, INVALID_TOP100_PARITY, None, "no_causal_top100_source", all_generations,
    )


def attach_top100_provenance(frame: pd.DataFrame, resolution: CausalTop100Resolution) -> pd.DataFrame:
    out = frame.copy()
    for name, value in resolution.output_fields().items():
        out[name] = value
    return out


def top100_transition_validation(
    generations: Iterable[Top100Generation],
    *,
    desired_new_symbols: Iterable[str] | None = None,
    process_start_id: str | None = None,
) -> dict[str, Any]:
    ordered = sorted(generations, key=lambda generation: generation.effective_at)
    if process_start_id:
        ordered = [item for item in ordered if item.process_start_id == process_start_id]
    transition_index = None
    for index in range(1, len(ordered)):
        if ordered[index - 1].ranking_source_date != ordered[index].ranking_source_date:
            transition_index = index
    if len(ordered) < 2:
        return {
            "process_start_id": process_start_id or (ordered[0].process_start_id if ordered else ""),
            "old_generation_count": len(ordered[0].symbols) if ordered else 0,
            "new_generation_count": 0,
            "overlap_count": 0,
            "removed_symbols": [],
            "added_symbols": [],
            "first_timestamp_of_new_generation": "",
            "first_complete_post_reload_scan": "",
            "first_complete_post_reload_timestamp": "",
            "stale_old_symbols_after_reload": [],
            "stale_old_entry_symbols_after_reload": [],
            "old_symbols_retained_only_for_active_positions_or_subscriptions": [],
            "retained_symbol_evidence_status": "not_available_from_light_entry_universe",
            "classification": "no_transition_observed",
        }
    if transition_index is None:
        return {
            "process_start_id": process_start_id or ordered[-1].process_start_id,
            "old_generation_count": len(ordered[-1].symbols),
            "new_generation_count": 0,
            "overlap_count": 0,
            "removed_symbols": [],
            "added_symbols": [],
            "first_timestamp_of_new_generation": "",
            "first_complete_post_reload_scan": "",
            "first_complete_post_reload_timestamp": "",
            "stale_old_symbols_after_reload": [],
            "stale_old_entry_symbols_after_reload": [],
            "old_symbols_retained_only_for_active_positions_or_subscriptions": [],
            "retained_symbol_evidence_status": "not_available_from_light_entry_universe",
            "classification": "no_source_date_transition_observed",
        }
    old, first_new = ordered[transition_index - 1], ordered[transition_index]
    desired = (
        frozenset(normalize_symbol(symbol) for symbol in desired_new_symbols if normalize_symbol(symbol))
        if desired_new_symbols is not None else first_new.symbols
    )
    removed = sorted(old.symbols - desired)
    added = sorted(desired - old.symbols)
    post_reload = [
        item
        for item in ordered[transition_index:]
        if item.process_start_id == first_new.process_start_id
        and item.ranking_source_date == first_new.ranking_source_date
    ]
    first_complete = next((item for item in post_reload if desired.issubset(item.symbols)), None)
    stale = sorted((old.symbols - desired) & first_complete.symbols) if first_complete else []
    stale_entry = sorted((old.symbols - desired) & first_complete.entry_symbols) if first_complete else []
    retained = sorted((old.symbols - desired) & first_complete.retained_symbols) if first_complete else []
    if first_complete is None:
        classification = "post_reload_scan_incomplete"
    elif stale:
        classification = "runtime_top100_union_observed"
    elif stale_entry:
        classification = "runtime_entry_universe_union_observed"
    else:
        classification = "temporal_union_only"
    return {
        "process_start_id": process_start_id or first_new.process_start_id,
        "old_ranking_source_date": old.ranking_source_date,
        "new_ranking_source_date": first_new.ranking_source_date,
        "old_generation_count": len(old.symbols),
        "new_generation_count": len(desired),
        "first_observed_new_generation_count": len(first_new.symbols),
        "overlap_count": len(old.symbols & desired),
        "removed_symbols": removed,
        "added_symbols": added,
        "first_timestamp_of_new_generation": first_new.effective_at.isoformat(),
        "first_complete_post_reload_scan": first_complete.first_scan_id if first_complete else "",
        "first_complete_post_reload_timestamp": first_complete.effective_at.isoformat() if first_complete else "",
        "stale_old_symbols_after_reload": stale,
        "stale_old_entry_symbols_after_reload": stale_entry,
        "old_symbols_retained_only_for_active_positions_or_subscriptions": retained,
        "retained_symbol_evidence_status": (
            "observed_in_light" if retained else "not_available_from_light_entry_universe"
        ),
        "classification": classification,
    }
