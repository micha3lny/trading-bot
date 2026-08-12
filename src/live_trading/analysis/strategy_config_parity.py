from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


THRESHOLD_KEYS = (
    "min_first_5m_high_pct",
    "min_first_15m_high_pct",
    "min_or_range_pct",
)


@dataclass(frozen=True)
class EffectiveSignalThresholds:
    min_first5: float
    min_first15: float
    min_or_range: float
    config_source: str

    def output_fields(self) -> dict[str, Any]:
        return {
            "effective_min_first5": self.min_first5,
            "effective_min_first15": self.min_first15,
            "effective_min_or_range": self.min_or_range,
            "config_source": self.config_source,
        }


def add_threshold_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-first-5m-high-pct", type=float, default=None)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=None)
    parser.add_argument("--min-or-range-pct", type=float, default=None)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _walk_mappings(value: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield value
    for key in ("effective_config", "strategy_config", "signal_config", "args", "config"):
        nested = _mapping(value.get(key))
        if nested:
            yield from _walk_mappings(nested)


def _thresholds_from_mapping(value: dict[str, Any]) -> tuple[float, float, float] | None:
    aliases = {
        "min_first_5m_high_pct": ("min_first_5m_high_pct", "first5_threshold", "effective_min_first5"),
        "min_first_15m_high_pct": ("min_first_15m_high_pct", "first15_threshold", "effective_min_first15"),
        "min_or_range_pct": ("min_or_range_pct", "or_max_range_pct", "effective_min_or_range"),
    }
    for candidate in _walk_mappings(value):
        resolved: list[float] = []
        for logical in THRESHOLD_KEYS:
            raw = next((candidate.get(key) for key in aliases[logical] if candidate.get(key) not in (None, "")), None)
            if raw is None:
                break
            try:
                resolved.append(float(raw))
            except (TypeError, ValueError):
                break
        if len(resolved) == 3:
            return resolved[0], resolved[1], resolved[2]
    return None


def load_session_thresholds(recorder_dir: Path, session_date: str) -> EffectiveSignalThresholds | None:
    path = recorder_dir / session_date / "run_metadata.csv"
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return None
    for row in reversed(rows):
        row_session = str(row.get("session_date") or "")[:10]
        if row_session and row_session != session_date:
            continue
        metadata = _mapping(row.get("metadata_json"))
        values = _thresholds_from_mapping({**metadata, **row})
        if values is not None:
            return EffectiveSignalThresholds(*values, config_source="run_metadata")
    return None


def resolve_signal_thresholds(
    *,
    session_date: str,
    recorder_dir: Path,
    min_first5: float | None,
    min_first15: float | None,
    min_or_range: float | None,
    profile: str = "live_session",
) -> EffectiveSignalThresholds:
    explicit = (min_first5, min_first15, min_or_range)
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("all three signal thresholds must be supplied together")
        return EffectiveSignalThresholds(
            float(min_first5), float(min_first15), float(min_or_range), "cli_explicit"
        )
    if profile == "low_threshold_causal":
        return EffectiveSignalThresholds(0.5, 1.0, 5.0, "profile:low_threshold_causal")
    if profile == "legacy_offline":
        return EffectiveSignalThresholds(0.5, 1.0, 5.0, "profile:legacy_offline_historical")
    session = load_session_thresholds(recorder_dir, session_date)
    if session is not None:
        return session
    raise RuntimeError(
        "STRATEGY_CONFIG_PARITY_UNRESOLVED "
        f"date={session_date} recorder={recorder_dir / session_date / 'run_metadata.csv'}; "
        "provide all of --min-first-5m-high-pct, --min-first-15m-high-pct and --min-or-range-pct"
    )


def resolve_threshold_args(args: argparse.Namespace, session_date: str) -> EffectiveSignalThresholds:
    return resolve_signal_thresholds(
        session_date=session_date,
        recorder_dir=Path(args.recorder_dir),
        min_first5=getattr(args, "min_first_5m_high_pct", None),
        min_first15=getattr(args, "min_first_15m_high_pct", None),
        min_or_range=getattr(args, "min_or_range_pct", None),
    )


def runtime_threshold_metadata(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_first_5m_high_pct": float(args.min_first_5m_high_pct),
        "min_first_15m_high_pct": float(args.min_first_15m_high_pct),
        "min_or_range_pct": float(args.min_or_range_pct),
    }


def output_has_config_provenance(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix.lower() == ".json":
            value = _mapping(path.read_text(encoding="utf-8", errors="replace"))
            return all(key in value for key in EffectiveSignalThresholds(0, 0, 0, "").output_fields())
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return all(key in text for key in EffectiveSignalThresholds(0, 0, 0, "").output_fields())
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            fields = set(next(csv.reader(handle), []))
        return set(EffectiveSignalThresholds(0, 0, 0, "").output_fields()).issubset(fields)
    except Exception:
        return False
