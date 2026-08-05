from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


LIGHT_SCHEMA_VERSION = "top100_candidate_light_v1"
FULL_SCHEMA_VERSION = "top100_candidate_full_v1"
MANIFEST_SCHEMA_VERSION = "top100_candidate_manifest_v1"

LIGHT_FIELDS = (
    "schema_version", "session_date", "trading_session_date", "timestamp",
    "scan_started_at", "scan_completed_at", "scan_duration_ms", "run_id",
    "process_start_id", "scan_id", "scan_uid", "symbol", "source",
    "expected_in_runtime_top100", "in_runtime_top100", "top100_rank",
    "top100_score", "top100_source", "ranking_source_date", "contract_present",
    "contract_source", "ticker_present", "usable_price", "state_present",
    "first_tick_received", "current_price", "bid", "ask", "spread_bps",
    "last_live_update", "ticker_age_seconds", "stale_reason",
    "first_price_initialized", "first5_initialized", "first15_initialized",
    "first_5m_complete", "first_15m_complete", "first_5m_high_pct",
    "first_15m_high_pct", "rth_open", "first_5m_high", "first_15m_high",
    "or_high", "or_low", "or_range_pct", "distance_from_or_high_pct",
    "premarket_data_quality", "premarket_open", "premarket_high", "premarket_low",
    "premarket_last", "premarket_range_pct", "premarket_change_pct",
    "raw_ticker_volume", "premarket_volume", "premarket_volume_accumulator",
    "premarket_vwap", "volume_semantics_assessment",
    "distance_from_premarket_high_pct", "distance_from_premarket_low_pct",
    "distance_from_premarket_vwap_pct", "gap_from_previous_close_pct",
    "ranking_position", "live_rank", "live_entry_score", "ready", "ready_since",
    "signal_sent", "would_emit_signal_ready", "signal_ready_reason",
    "rejection_reason", "entry_symbol_allowed", "symbol_ineligible",
    "candidate_age_seconds", "entries_blocked", "entries_blocked_reason",
    "manual_block", "entry_delay_block", "restart_block", "reconnect_block",
    "disk_block", "top100_block", "risk_guard_block", "risk_guard_reason",
    "open_positions", "pending_orders", "max_open_positions", "available_slots",
    "max_entries_per_cycle", "max_entries_per_minute", "already_open", "quantity",
    "quantity_reason", "eligible_for_entry_selection", "selected_for_entry",
    "selection_rank", "selection_rejected_reason", "buy_decision_created",
    "buy_blocked", "buy_block_reason", "order_created", "order_submitted",
    "order_id", "perm_id", "order_status", "full_snapshot_ref",
)

FULL_FIELDS = (
    "schema_version", "session_date", "trading_session_date", "timestamp",
    "scan_started_at", "scan_completed_at", "scan_duration_ms", "run_id", "process_start_id",
    "scan_id", "scan_uid", "symbol", "source", "emit_reason", "emit_reasons",
    "candle_timestamp", "candle_open", "candle_high", "candle_low", "candle_close",
    "candle_volume", "candle_samples", "candle_source", "candle_is_completed", "session_phase",
    "feature_state_revision", "feature_state_hash", "first_price_initialized",
    "first5_initialized", "first15_initialized", "first_5m_complete",
    "first_15m_complete", "rth_open", "first_5m_high", "first_15m_high",
    "or_high", "or_low", "premarket_open", "premarket_high", "premarket_low",
    "premarket_last", "raw_ticker_volume", "premarket_volume",
    "premarket_volume_accumulator", "premarket_vwap_numerator",
    "premarket_vwap_denominator", "premarket_vwap", "premarket_data_quality",
    "volume_semantics_assessment", "full_snapshot_ref",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return token[:120] or "unknown"


def _normalize_row(row: dict[str, Any], fields: tuple[str, ...], schema_version: str) -> dict[str, Any]:
    out = {field: row.get(field) for field in fields}
    out["schema_version"] = schema_version
    return out


def normalize_light_row(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize_row(row, LIGHT_FIELDS, LIGHT_SCHEMA_VERSION)


def normalize_full_row(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize_row(row, FULL_FIELDS, FULL_SCHEMA_VERSION)


def feature_state_hash(values: Iterable[Any]) -> str:
    payload = repr(tuple(values)).encode("utf-8", errors="replace")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


@dataclass(frozen=True)
class CandidateSnapshotBatch:
    session_date: str
    process_start_id: str
    scan_id: int
    scan_uid: str
    expected_symbols: int
    light_rows: tuple[dict[str, Any], ...]
    full_rows: tuple[dict[str, Any], ...] = ()


@dataclass
class _SessionBuffer:
    session_date: str
    light_rows: list[dict[str, Any]] = field(default_factory=list)
    full_rows: list[dict[str, Any]] = field(default_factory=list)
    batches: list[CandidateSnapshotBatch] = field(default_factory=list)
    light_count_by_scan_uid: dict[str, int] = field(default_factory=dict)


@dataclass
class CandidateScanCollector:
    """Sidecar scan state. Updating it must not mutate trading inputs."""

    session_date: str
    run_id: str
    process_start_id: str
    scan_id: int
    scan_started_at: str
    expected_symbols: tuple[str, ...]
    rows: dict[str, dict[str, Any]] = field(init=False)
    full_rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        scan_uid = f"{self.process_start_id}:{self.scan_id}"
        self.rows = {
            symbol: {
                "session_date": self.session_date,
                "trading_session_date": self.session_date,
                "timestamp": self.scan_started_at,
                "scan_started_at": self.scan_started_at,
                "run_id": self.run_id,
                "process_start_id": self.process_start_id,
                "scan_id": self.scan_id,
                "scan_uid": scan_uid,
                "symbol": symbol,
                "source": "live_runtime",
                "expected_in_runtime_top100": 1,
                "in_runtime_top100": 1,
            }
            for symbol in self.expected_symbols
        }

    @property
    def scan_uid(self) -> str:
        return f"{self.process_start_id}:{self.scan_id}"

    def update(self, symbol: str, **values: Any) -> None:
        row = self.rows.get(symbol)
        if row is not None:
            row.update(values)

    def add_full(self, row: dict[str, Any]) -> None:
        self.full_rows.append(row)

    def finalize(self, scan_completed_at: str, scan_duration_ms: float) -> CandidateSnapshotBatch:
        for row in self.rows.values():
            row["scan_completed_at"] = scan_completed_at
            row["scan_duration_ms"] = round(float(scan_duration_ms), 3)
            row["timestamp"] = scan_completed_at
            for field in (
                "contract_present", "ticker_present", "usable_price", "state_present",
                "first_tick_received", "ready", "signal_sent", "would_emit_signal_ready",
                "first_price_initialized", "first5_initialized", "first15_initialized",
                "first_5m_complete", "first_15m_complete",
                "entry_symbol_allowed", "symbol_ineligible", "entries_blocked", "manual_block",
                "entry_delay_block", "restart_block", "reconnect_block", "disk_block",
                "top100_block", "risk_guard_block", "already_open", "eligible_for_entry_selection",
                "selected_for_entry", "buy_decision_created", "buy_blocked", "order_created",
                "order_submitted",
            ):
                if row.get(field) is None:
                    row[field] = 0
            if not row.get("eligible_for_entry_selection") and not row.get("selection_rejected_reason"):
                row["selection_rejected_reason"] = row.get("rejection_reason") or row.get("signal_ready_reason")
        for row in self.full_rows:
            row["scan_completed_at"] = scan_completed_at
            row["scan_duration_ms"] = round(float(scan_duration_ms), 3)
            row["timestamp"] = scan_completed_at
        return CandidateSnapshotBatch(
            session_date=self.session_date,
            process_start_id=self.process_start_id,
            scan_id=self.scan_id,
            scan_uid=self.scan_uid,
            expected_symbols=len(self.expected_symbols),
            light_rows=tuple(self.rows[symbol] for symbol in self.expected_symbols),
            full_rows=tuple(self.full_rows),
        )


@dataclass
class _FullState:
    revision: int = 0
    last_hash: str = ""
    last_emitted_at: float = 0.0
    last_candle_timestamp: str = ""
    initialization_flags: tuple[Any, ...] = ()
    completion_flags: tuple[Any, ...] = ()


class FullSnapshotTracker:
    def __init__(self) -> None:
        self._states: dict[str, _FullState] = {}

    def consider(
        self,
        symbol: str,
        *,
        feature_values: tuple[Any, ...],
        initialization_flags: tuple[Any, ...],
        completion_flags: tuple[Any, ...],
        candle_timestamp: str = "",
        now_monotonic: float | None = None,
        checkpoint_seconds: float = 60.0,
    ) -> dict[str, Any] | None:
        now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        current_hash = feature_state_hash(feature_values)
        previous = self._states.get(symbol)
        reasons: list[str] = []
        if previous is None:
            previous = _FullState()
            self._states[symbol] = previous
            reasons.append("state_created")
        if candle_timestamp and candle_timestamp != previous.last_candle_timestamp:
            reasons.insert(0, "new_completed_1m_bar")
        if previous.initialization_flags and initialization_flags != previous.initialization_flags:
            reasons.append("initialization_state_change")
        if previous.completion_flags and completion_flags != previous.completion_flags:
            reasons.append("completion_state_change")
        if not reasons and checkpoint_seconds > 0 and now_monotonic - previous.last_emitted_at >= checkpoint_seconds:
            reasons.append("explicit_feature_state_checkpoint")
        previous.initialization_flags = initialization_flags
        previous.completion_flags = completion_flags
        if candle_timestamp:
            previous.last_candle_timestamp = candle_timestamp
        if not reasons:
            return None
        previous.revision += 1
        previous.last_hash = current_hash
        previous.last_emitted_at = now_monotonic
        return {
            "emit_reason": reasons[0],
            "emit_reasons": ";".join(dict.fromkeys(reasons)),
            "feature_state_revision": previous.revision,
            "feature_state_hash": current_hash,
            "full_snapshot_ref": f"{symbol}:{previous.revision}:{current_hash}",
        }


class CandidateSnapshotWriter:
    """Non-blocking producer and dedicated Parquet writer for scan snapshots."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        process_start_id: str,
        queue_size: int = 16,
        chunk_rows: int = 5_000,
        start_thread: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.process_start_id = str(process_start_id)
        self.chunk_rows = max(1, int(chunk_rows))
        self._queue: queue.Queue[CandidateSnapshotBatch | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="candidate-snapshot-writer", daemon=True)
        self._buffer: _SessionBuffer | None = None
        self._chunk_sequence: dict[str, int] = {}
        self._seen_light_keys: set[tuple[str, str, int, str]] = set()
        self._seen_full_keys: set[tuple[Any, ...]] = set()
        self._expected_scan_uids: set[str] = set()
        self._accepted_scan_uids: set[str] = set()
        self._session_stats: dict[str, dict[str, Any]] = {}
        if start_thread:
            self._thread.start()

    def _stats(self, session_date: str) -> dict[str, Any]:
        with self._lock:
            stats = self._session_stats.get(session_date)
            if stats is None:
                stats = self._load_existing_stats(session_date) or {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "session_date": session_date,
                    "process_start_ids": [],
                    "expected_scan_batches": 0,
                    "enqueued_scan_batches": 0,
                    "written_scan_batches": 0,
                    "dropped_scan_batches": 0,
                    "expected_rows": 0,
                    "written_rows": 0,
                    "dropped_rows": 0,
                    "written_full_rows": 0,
                    "dropped_full_rows": 0,
                    "first_scan_uid": "",
                    "last_scan_uid": "",
                    "written_scan_ids_by_process": {},
                    "expected_scan_ids_by_process": {},
                    "expected_symbols_by_scan_uid": {},
                    "written_symbols_by_scan_uid": {},
                    "expected_symbols_per_scan": 0,
                    "written_symbols_per_scan": [],
                    "queue_depth": 0,
                    "queue_high_watermark": 0,
                    "last_write_latency_ms": None,
                    "max_write_latency_ms": 0.0,
                    "last_write_error": "",
                    "writer_error_count": 0,
                    "updated_at": utc_now_iso(),
                }
                self._session_stats[session_date] = stats
            if self.process_start_id not in stats["process_start_ids"]:
                stats["process_start_ids"].append(self.process_start_id)
            return stats

    def _load_existing_stats(self, session_date: str) -> dict[str, Any] | None:
        path = self.output_root / session_date / "top100_candidate_snapshots" / "candidate_snapshot_manifest.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        payload.setdefault("written_scan_ids_by_process", {})
        payload.setdefault("expected_scan_ids_by_process", {})
        payload.setdefault("written_symbols_per_scan", [])
        payload.setdefault("expected_symbols_by_scan_uid", {})
        payload.setdefault("written_symbols_by_scan_uid", {})
        payload.setdefault("process_start_ids", [])
        return payload

    def note_expected_batch(
        self,
        session_date: str,
        *,
        process_start_id: str,
        scan_id: int,
        scan_uid: str,
        expected_symbols: int,
    ) -> None:
        if scan_uid in self._expected_scan_uids:
            return
        self._expected_scan_uids.add(scan_uid)
        stats = self._stats(session_date)
        with self._lock:
            stats["expected_scan_batches"] += 1
            stats["expected_rows"] += int(expected_symbols)
            stats["expected_symbols_per_scan"] = max(int(stats["expected_symbols_per_scan"]), int(expected_symbols))
            stats["first_scan_uid"] = stats["first_scan_uid"] or scan_uid
            stats["last_scan_uid"] = scan_uid
            stats.setdefault("expected_scan_ids_by_process", {}).setdefault(process_start_id, []).append(int(scan_id))
            stats.setdefault("expected_symbols_by_scan_uid", {})[scan_uid] = int(expected_symbols)
            stats["updated_at"] = utc_now_iso()

    def enqueue(self, batch: CandidateSnapshotBatch) -> bool:
        self.note_expected_batch(
            batch.session_date,
            process_start_id=batch.process_start_id,
            scan_id=batch.scan_id,
            scan_uid=batch.scan_uid,
            expected_symbols=batch.expected_symbols,
        )
        stats = self._stats(batch.session_date)
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            with self._lock:
                stats["dropped_scan_batches"] += 1
                stats["dropped_rows"] += len(batch.light_rows)
                stats["dropped_full_rows"] += len(batch.full_rows)
                stats["queue_depth"] = self._queue.qsize()
                stats["queue_high_watermark"] = max(stats["queue_high_watermark"], self._queue.qsize())
                stats["updated_at"] = utc_now_iso()
            return False
        with self._lock:
            stats["enqueued_scan_batches"] += 1
            stats["queue_depth"] = self._queue.qsize()
            stats["queue_high_watermark"] = max(stats["queue_high_watermark"], self._queue.qsize())
            stats["updated_at"] = utc_now_iso()
        return True

    def health(self, session_date: str) -> dict[str, Any]:
        stats = self._stats(session_date)
        with self._lock:
            return self._manifest_payload(dict(stats))

    def _manifest_payload(self, stats: dict[str, Any]) -> dict[str, Any]:
        written_counts = [int(value) for value in stats.get("written_symbols_per_scan", [])]
        written_by_process = stats.get("written_scan_ids_by_process") or {}
        expected_by_process = stats.get("expected_scan_ids_by_process") or {}
        stats["min_written_symbols_per_scan"] = min(written_counts) if written_counts else 0
        stats["max_written_symbols_per_scan"] = max(written_counts) if written_counts else 0
        stats["average_written_symbols_per_scan"] = (
            round(sum(written_counts) / len(written_counts), 4) if written_counts else 0.0
        )
        stats["missing_scan_ranges"] = {}
        for process_id, expected_values in expected_by_process.items():
            expected_ids = sorted(set(int(value) for value in expected_values))
            written_ids = set(int(value) for value in written_by_process.get(process_id, []))
            stats["missing_scan_ranges"][str(process_id)] = _ranges_from_values(
                [value for value in expected_ids if value not in written_ids]
            )
        completeness = "IN_PROGRESS"
        if stats.get("writer_error_count"):
            completeness = "PARTIAL_WRITE_ERROR"
        elif stats.get("dropped_scan_batches"):
            completeness = "PARTIAL_QUEUE_DROP"
        elif len(stats.get("process_start_ids") or []) > 1:
            completeness = "PARTIAL_PROCESS_RESTART"
        elif self._closed and stats.get("written_scan_batches") != stats.get("expected_scan_batches"):
            completeness = "PARTIAL_MISSING_SYMBOL_ROWS"
        elif stats.get("written_scan_batches") == stats.get("expected_scan_batches") and stats.get("expected_scan_batches"):
            expected_by_scan = stats.get("expected_symbols_by_scan_uid") or {}
            written_by_scan = stats.get("written_symbols_by_scan_uid") or {}
            if any(int(written_by_scan.get(uid, -1)) != int(count) for uid, count in expected_by_scan.items()):
                completeness = "PARTIAL_MISSING_SYMBOL_ROWS"
            else:
                completeness = "COMPLETE"
        stats["session_completeness"] = completeness
        stats["snapshot_batches_written"] = stats.get("written_scan_batches", 0)
        stats["snapshot_rows_written"] = stats.get("written_rows", 0)
        stats["snapshot_batches_dropped"] = stats.get("dropped_scan_batches", 0)
        stats["snapshot_rows_dropped"] = stats.get("dropped_rows", 0)
        return stats

    def _write_manifest(self, session_date: str) -> None:
        stats = self._stats(session_date)
        with self._lock:
            payload = self._manifest_payload(dict(stats))
        session_dir = self.output_root / session_date / "top100_candidate_snapshots"
        session_dir.mkdir(parents=True, exist_ok=True)
        target = session_dir / "candidate_snapshot_manifest.json"
        temp = target.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp, target)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            try:
                self._accept(item)
            except Exception as exc:  # writer failures must not reach the trading thread
                self._record_writer_error(item, exc)
            finally:
                self._queue.task_done()
        self._flush()

    def _accept(self, batch: CandidateSnapshotBatch) -> None:
        if batch.scan_uid in self._accepted_scan_uids:
            return
        self._accepted_scan_uids.add(batch.scan_uid)
        if self._buffer is not None and self._buffer.session_date != batch.session_date:
            self._flush()
        if self._buffer is None:
            self._buffer = _SessionBuffer(session_date=batch.session_date)

        added_light_rows = 0
        for raw in batch.light_rows:
            row = normalize_light_row(raw)
            key = (str(row["session_date"]), str(row["process_start_id"]), int(row["scan_id"]), str(row["symbol"]))
            if key not in self._seen_light_keys:
                self._seen_light_keys.add(key)
                self._buffer.light_rows.append(row)
                added_light_rows += 1
        for raw in batch.full_rows:
            row = normalize_full_row(raw)
            key = _full_key(row)
            if key not in self._seen_full_keys:
                self._seen_full_keys.add(key)
                self._buffer.full_rows.append(row)
        self._buffer.batches.append(batch)
        self._buffer.light_count_by_scan_uid[batch.scan_uid] = added_light_rows
        if len(self._buffer.light_rows) + len(self._buffer.full_rows) >= self.chunk_rows:
            self._flush()

    def _flush(self) -> None:
        buffer = self._buffer
        if buffer is None or not buffer.batches:
            self._buffer = None
            return
        started = time.perf_counter()
        session_dir = self.output_root / buffer.session_date / "top100_candidate_snapshots"
        sequence = self._chunk_sequence.get(buffer.session_date, _next_chunk_sequence(session_dir, self.process_start_id))
        token = _safe_token(self.process_start_id)
        prepared: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            if buffer.light_rows:
                target = session_dir / "light" / f"part-{token}-{sequence:06d}.parquet"
                prepared.append(
                    (_prepare_parquet(target, buffer.light_rows), target)
                )
            if buffer.full_rows:
                target = session_dir / "full" / f"part-{token}-{sequence:06d}.parquet"
                prepared.append(
                    (_prepare_parquet(target, buffer.full_rows), target)
                )
            for temp, target in prepared:
                os.replace(temp, target)
                published.append(target)
        except Exception as exc:
            for temp, _target in prepared:
                temp.unlink(missing_ok=True)
            for target in published:
                target.unlink(missing_ok=True)
            for batch in buffer.batches:
                self._record_writer_error(batch, exc)
            self._buffer = None
            return

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats = self._stats(buffer.session_date)
        with self._lock:
            stats["written_scan_batches"] += len(buffer.batches)
            stats["written_rows"] += len(buffer.light_rows)
            stats["written_full_rows"] += len(buffer.full_rows)
            by_process = stats.setdefault("written_scan_ids_by_process", {})
            for batch in buffer.batches:
                by_process.setdefault(batch.process_start_id, []).append(batch.scan_id)
                written_count = int(buffer.light_count_by_scan_uid.get(batch.scan_uid, 0))
                stats["written_symbols_per_scan"].append(written_count)
                stats.setdefault("written_symbols_by_scan_uid", {})[batch.scan_uid] = written_count
            stats["last_write_latency_ms"] = round(elapsed_ms, 3)
            stats["max_write_latency_ms"] = max(float(stats["max_write_latency_ms"]), elapsed_ms)
            stats["queue_depth"] = self._queue.qsize()
            stats["updated_at"] = utc_now_iso()
        self._chunk_sequence[buffer.session_date] = sequence + 1
        try:
            self._write_manifest(buffer.session_date)
        except Exception as exc:
            self._record_manifest_error(buffer.session_date, exc)
        finally:
            self._buffer = None

    def _record_writer_error(self, batch: CandidateSnapshotBatch, exc: Exception) -> None:
        stats = self._stats(batch.session_date)
        with self._lock:
            stats["writer_error_count"] += 1
            stats["last_write_error"] = repr(exc)[:1000]
            stats["dropped_scan_batches"] += 1
            stats["dropped_rows"] += len(batch.light_rows)
            stats["dropped_full_rows"] += len(batch.full_rows)
            stats["updated_at"] = utc_now_iso()
        try:
            self._write_manifest(batch.session_date)
        except Exception:
            pass

    def _record_manifest_error(self, session_date: str, exc: Exception) -> None:
        stats = self._stats(session_date)
        with self._lock:
            stats["writer_error_count"] += 1
            stats["last_write_error"] = f"manifest:{exc!r}"[:1000]
            stats["updated_at"] = utc_now_iso()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))
        if not self._thread.is_alive():
            self._flush()
        self._closed = True
        for session_date in list(self._session_stats):
            try:
                self._write_manifest(session_date)
            except Exception as exc:
                self._record_manifest_error(session_date, exc)


def _prepare_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temp, compression="zstd", use_dictionary=True)
    return temp


def _next_chunk_sequence(session_dir: Path, process_start_id: str) -> int:
    token = _safe_token(process_start_id)
    sequences: list[int] = []
    for kind in ("light", "full"):
        for path in (session_dir / kind).glob(f"part-{token}-*.parquet"):
            try:
                sequences.append(int(path.stem.rsplit("-", 1)[-1]))
            except ValueError:
                continue
    return max(sequences, default=-1) + 1


def _missing_ranges(scan_ids: list[int]) -> list[str]:
    if len(scan_ids) < 2:
        return []
    missing: list[str] = []
    previous = scan_ids[0]
    for current in scan_ids[1:]:
        if current > previous + 1:
            start, end = previous + 1, current - 1
            missing.append(str(start) if start == end else f"{start}-{end}")
        previous = current
    return missing


def _ranges_from_values(values: list[int]) -> list[str]:
    if not values:
        return []
    values = sorted(set(values))
    ranges: list[str] = []
    start = previous = values[0]
    for current in values[1:]:
        if current != previous + 1:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = current
        previous = current
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ranges


def _full_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("session_date"), row.get("process_start_id"), row.get("symbol"),
        row.get("candle_timestamp"), row.get("emit_reason"), row.get("feature_state_revision"),
    )


def snapshot_chunk_paths(recorder_root: str | Path, session_date: str, kind: str) -> list[Path]:
    return sorted((Path(recorder_root) / session_date / "top100_candidate_snapshots" / kind).glob("*.parquet"))


def export_snapshot_csv(
    recorder_root: str | Path,
    session_date: str,
    kind: str,
    output_path: str | Path,
) -> dict[str, Any]:
    if kind not in {"light", "full"}:
        raise ValueError("kind must be light or full")
    fields = LIGHT_FIELDS if kind == "light" else FULL_FIELDS
    paths = snapshot_chunk_paths(recorder_root, session_date, kind)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    seen: set[tuple[Any, ...]] = set()
    rows_written = 0
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    key = (
                        (row.get("session_date"), row.get("process_start_id"), row.get("scan_id"), row.get("symbol"))
                        if kind == "light" else _full_key(row)
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.writerow({field: row.get(field) for field in fields})
                    rows_written += 1
    os.replace(temp, output)
    return {"kind": kind, "chunks": len(paths), "rows_written": rows_written, "output": str(output)}
