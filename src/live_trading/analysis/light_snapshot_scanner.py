from __future__ import annotations

import heapq
import resource
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.live_trading.analysis.common import normalize_symbol, parse_dt
from src.live_trading.candidate_snapshot_telemetry import snapshot_chunk_paths
from src.live_trading.analysis.persistent_light_index import ensure_persistent_light_index


LIGHT_KEY_COLUMNS = ("session_date", "process_start_id", "scan_id", "symbol")
LIGHT_TIME_COLUMNS = ("timestamp", "scan_completed_at", "scan_started_at")
PARTITION_MERGE_BATCH_SIZE = 256


def rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak /= 1024.0 * 1024.0
    else:
        peak /= 1024.0
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    return round(peak, 2)


def snapshot_timestamp(row: dict[str, Any]) -> pd.Timestamp | None:
    cached = row.get("_snapshot_timestamp")
    if isinstance(cached, pd.Timestamp) and not pd.isna(cached):
        return cached
    for column in LIGHT_TIME_COLUMNS:
        parsed = parse_dt(row.get(column))
        if parsed is not None:
            return parsed
    return None


def _session_matches(row: dict[str, Any], session_date: str) -> bool:
    explicit = str(
        row.get("session_date")
        or row.get("trading_session_date")
        or row.get("trade_session_date")
        or ""
    )[:10]
    if explicit:
        return explicit == session_date
    timestamp = snapshot_timestamp(row)
    return timestamp is not None and timestamp.date().isoformat() == session_date


def _key(row: dict[str, Any]) -> str:
    values: list[str] = []
    for column in LIGHT_KEY_COLUMNS:
        value = normalize_symbol(row.get(column)) if column == "symbol" else row.get(column)
        values.append("<NA>" if value is None or pd.isna(value) else str(value))
    return "\x1f".join(values)


@dataclass(frozen=True)
class SnapshotTarget:
    target_id: str
    symbol: str
    timestamp: pd.Timestamp


@dataclass
class SnapshotMatch:
    target: SnapshotTarget
    before: dict[str, Any] | None = None
    nearest: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    first_seen: pd.Timestamp | None = None
    last_seen: pd.Timestamp | None = None
    scan_count: int = 0


class BoundedLightScanner:
    """Disk-backed LIGHT scanner preserving last-row dedupe semantics."""

    def __init__(
        self,
        recorder_dir: str | Path,
        session_date: str,
        *,
        temp_dir: str | Path | None = None,
        batch_size: int = 65536,
        log_prefix: str = "LIGHT_SCANNER",
        index_filter_columns: Iterable[str] = (),
        index_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> None:
        self.recorder_dir = Path(recorder_dir)
        self.session_date = session_date
        self.paths = snapshot_chunk_paths(self.recorder_dir, session_date, "light")
        self.batch_size = batch_size
        self.log_prefix = log_prefix
        self.index_filter_columns = tuple(index_filter_columns)
        self.index_filter = index_filter
        self._temp_owner = tempfile.TemporaryDirectory(prefix="light-scanner-") if temp_dir is None else None
        self.temp_dir = Path(temp_dir or self._temp_owner.name)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.temp_dir / "light_index.sqlite")
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.execute(
            "CREATE TABLE retained (dedupe_key TEXT PRIMARY KEY, source_index INTEGER NOT NULL, row_index INTEGER NOT NULL, symbol TEXT NOT NULL, timestamp TEXT)"
        )
        self.conn.execute("CREATE INDEX retained_source ON retained(source_index, row_index)")
        self.conn.execute("CREATE INDEX retained_symbol ON retained(symbol, timestamp)")
        self.rows_scanned = 0
        self.rows_session_scoped = 0
        self.rows_wrong_session = 0
        self.parse_failures = 0
        self._built = False
        self._memory_retained: dict[str, tuple[int, int, str, str]] | None = None
        self._target_symbols: set[str] | None = None
        self.cache_hit = 0
        self.cache_fingerprint = ""
        self.cache_canonical_bytes = 0
        self.cache_canonical_partition_count = 0
        self.cache_generation_summary_path: Path | None = None
        self.canonical_partitions: dict[str, Path] = {}

    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None
        if self._temp_owner is not None:
            self._temp_owner.cleanup()
            self._temp_owner = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "BoundedLightScanner":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def retained_rows(self) -> int:
        if self._memory_retained is not None:
            return len(self._memory_retained)
        if self._target_symbols:
            placeholders = ",".join("?" for _ in self._target_symbols)
            return int(self.conn.execute(
                f"SELECT COUNT(*) FROM retained WHERE symbol IN ({placeholders})", tuple(sorted(self._target_symbols))
            ).fetchone()[0])
        return int(self.conn.execute("SELECT COUNT(*) FROM retained").fetchone()[0])

    def build_index(self, target_symbols: set[str] | None = None) -> None:
        if self._built:
            return
        targets = {normalize_symbol(symbol) for symbol in target_symbols} if target_symbols else None
        self._target_symbols = targets
        started = time.perf_counter()
        print(
            f"{self.log_prefix}_PHASE_START phase=index_light chunks={len(self.paths)} rss_mb={rss_mb()}",
            flush=True,
        )
        index_path, metadata = ensure_persistent_light_index(
            self.recorder_dir, self.session_date, self.paths, batch_size=self.batch_size,
        )
        self.conn.close()
        self.conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
        self.rows_scanned = int(metadata.get("rows_scanned", 0))
        self.rows_session_scoped = int(metadata.get("rows_session_scoped", 0))
        self.rows_wrong_session = int(metadata.get("rows_wrong_session", 0))
        self.parse_failures = int(metadata.get("timestamp_parse_failures", 0))
        self.cache_hit = int(metadata.get("cache_hit", 0))
        self.cache_fingerprint = str(metadata.get("fingerprint") or "")
        self.cache_canonical_bytes = int(metadata.get("canonical_bytes", 0))
        self.cache_canonical_partition_count = int(metadata.get("canonical_partition_count", 0))
        cache_root = index_path.parent
        summary_relative = metadata.get("top100_generation_summary")
        self.cache_generation_summary_path = cache_root / str(summary_relative) if summary_relative else None
        self.canonical_partitions = {
            normalize_symbol(symbol): cache_root / relative
            for symbol, relative in (metadata.get("canonical_partitions") or {}).items()
        }
        self._built = True
        print(
            f"{self.log_prefix}_PHASE_DONE phase=index_light elapsed={time.perf_counter() - started:.3f} "
            f"rows_scanned={self.rows_scanned} rows_session_scoped={self.rows_session_scoped} "
            f"rows_retained={self.retained_rows} wrong_session={self.rows_wrong_session} "
            f"timestamp_parse_failures={self.parse_failures} rss_mb={rss_mb()}",
            f"cache_hit={self.cache_hit} cache_fingerprint={self.cache_fingerprint}",
            flush=True,
        )

    def iter_rows(self, columns: Iterable[str]) -> Iterator[dict[str, Any]]:
        if not self._built:
            self.build_index()
        requested = set(columns) | set(LIGHT_KEY_COLUMNS) | set(LIGHT_TIME_COLUMNS)
        if self.canonical_partitions:
            symbols = sorted(self._target_symbols or self.canonical_partitions)
            iterators = []
            for symbol in symbols:
                path = self.canonical_partitions.get(symbol)
                if path is not None and path.exists():
                    iterators.append(self._iter_partition_rows(path, requested))
            heap: list[tuple[tuple[Any, ...], int, dict[str, Any], Iterator[dict[str, Any]]]] = []
            for ordinal, iterator in enumerate(iterators):
                try:
                    row = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(heap, (self._canonical_order_key(row), ordinal, row, iterator))
            yielded = 0
            while heap:
                _key, ordinal, row, iterator = heapq.heappop(heap)
                yield row
                yielded += 1
                if yielded % 10000 == 0:
                    pa.default_memory_pool().release_unused()
                try:
                    following = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(heap, (self._canonical_order_key(following), ordinal, following, iterator))
            pa.default_memory_pool().release_unused()
            return
        for source_index, path in enumerate(self.paths):
            if self._target_symbols:
                placeholders = ",".join("?" for _ in self._target_symbols)
                retained = {
                    int(row[0]) for row in self.conn.execute(
                        f"SELECT row_index FROM retained WHERE source_index=? AND symbol IN ({placeholders})",
                        (source_index, *sorted(self._target_symbols)),
                    )
                }
            else:
                retained = {
                    int(row[0]) for row in self.conn.execute(
                        "SELECT row_index FROM retained WHERE source_index = ?", (source_index,)
                    )
                }
            if not retained:
                continue
            parquet = pq.ParquetFile(path)
            selected = [column for column in parquet.schema_arrow.names if column in requested]
            source_row = 0
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=selected):
                local_indices = sorted(index - source_row for index in retained if source_row <= index < source_row + batch.num_rows)
                if not local_indices:
                    source_row += batch.num_rows
                    continue
                yield from self._rows_from_batch(batch.take(local_indices))
                source_row += batch.num_rows

    def iter_symbol_rows(self, symbol: str, columns: Iterable[str]) -> Iterator[dict[str, Any]]:
        """Stream one canonical symbol partition without pinning other partitions."""
        if not self._built:
            self.build_index()
        normalized = normalize_symbol(symbol)
        path = self.canonical_partitions.get(normalized)
        if path is None or not path.exists():
            return
        requested = set(columns) | set(LIGHT_KEY_COLUMNS) | set(LIGHT_TIME_COLUMNS)
        yield from self._iter_partition_rows(path, requested)

    def _iter_partition_rows(self, path: Path, requested: set[str]) -> Iterator[dict[str, Any]]:
        parquet = pq.ParquetFile(path)
        selected = [column for column in parquet.schema_arrow.names if column in requested]
        # The k-way merge keeps one batch alive per symbol. Keep those batches
        # deliberately small so a 100+ symbol telemetry universe stays bounded.
        merge_batch_size = min(self.batch_size, PARTITION_MERGE_BATCH_SIZE)
        for batch in parquet.iter_batches(batch_size=merge_batch_size, columns=selected):
            yield from self._rows_from_batch(batch)

    @staticmethod
    def _canonical_order_key(row: dict[str, Any]) -> tuple[Any, ...]:
        timestamp = row.get("_snapshot_timestamp")
        timestamp_ns = timestamp.value if isinstance(timestamp, pd.Timestamp) and not pd.isna(timestamp) else -1
        scan_id = str(row.get("scan_id") or "")
        try:
            scan_key: tuple[int, Any] = (0, int(scan_id))
        except ValueError:
            scan_key = (1, scan_id)
        return (
            timestamp_ns,
            str(row.get("process_start_id") or ""),
            scan_key,
            normalize_symbol(row.get("symbol")),
        )

    def _rows_from_batch(self, batch: Any) -> Iterator[dict[str, Any]]:
        kept = range(batch.num_rows)
        if self.index_filter is not None and batch.num_rows:
            filter_frame = batch.to_pandas()
            for column in self.index_filter_columns:
                if column not in filter_frame:
                    filter_frame[column] = None
            mask = pd.Series(self.index_filter(filter_frame), index=filter_frame.index).fillna(False).astype(bool)
            kept = [index for index, value in enumerate(mask.tolist()) if value]
        names = batch.schema.names
        columns = [batch.column(index) for index in range(batch.num_columns)]
        for row_index in kept:
            row = {name: column[row_index].as_py() for name, column in zip(names, columns)}
            row["symbol"] = normalize_symbol(row.get("symbol"))
            timestamp = snapshot_timestamp(row)
            if timestamp is not None:
                row["_snapshot_timestamp"] = timestamp
            yield row

    def consume(self, columns: Iterable[str], consumer: Callable[[dict[str, Any]], None]) -> None:
        for row in self.iter_rows(columns):
            consumer(row)

    @staticmethod
    def _timestamp_series(batch: Any) -> pd.Series:
        timestamps = pd.Series(pd.NaT, index=range(batch.num_rows), dtype="datetime64[ns, UTC]")
        for column in LIGHT_TIME_COLUMNS:
            if column in batch.schema.names:
                values = batch.column(batch.schema.get_field_index(column)).to_pandas()
                timestamps = timestamps.combine_first(
                    pd.to_datetime(values, errors="coerce", utc=True).reset_index(drop=True)
                )
        return timestamps

    @staticmethod
    def _row_at(batch: Any, index: int, timestamp: pd.Timestamp) -> dict[str, Any]:
        row = {
            name: batch.column(column_index)[index].as_py()
            for column_index, name in enumerate(batch.schema.names)
        }
        row["symbol"] = normalize_symbol(row.get("symbol"))
        row["_snapshot_timestamp"] = timestamp
        return row

    def _match_partition_targets(
        self,
        path: Path,
        requested: set[str],
        matches: list[SnapshotMatch],
    ) -> None:
        parquet = pq.ParquetFile(path)
        selected = [column for column in parquet.schema_arrow.names if column in requested]
        for batch in parquet.iter_batches(batch_size=min(self.batch_size, 8192), columns=selected):
            timestamps = self._timestamp_series(batch)
            valid = timestamps.notna()
            if not valid.any():
                continue
            valid_timestamps = timestamps.loc[valid]
            for match in matches:
                match.scan_count += int(valid.sum())
                batch_first = valid_timestamps.min()
                batch_last = valid_timestamps.max()
                match.first_seen = batch_first if match.first_seen is None else min(match.first_seen, batch_first)
                match.last_seen = batch_last if match.last_seen is None else max(match.last_seen, batch_last)
                center = match.target.timestamp

                before_values = valid_timestamps.loc[valid_timestamps <= center]
                if not before_values.empty:
                    candidate_timestamp = before_values.max()
                    candidate_index = int(before_values.index[before_values == candidate_timestamp][0])
                    prior = snapshot_timestamp(match.before or {})
                    if prior is None or candidate_timestamp > prior:
                        match.before = self._row_at(batch, candidate_index, candidate_timestamp)

                after_values = valid_timestamps.loc[valid_timestamps > center]
                if not after_values.empty:
                    candidate_timestamp = after_values.min()
                    candidate_index = int(after_values.index[after_values == candidate_timestamp][0])
                    prior = snapshot_timestamp(match.after or {})
                    if prior is None or candidate_timestamp < prior:
                        match.after = self._row_at(batch, candidate_index, candidate_timestamp)

                deltas = (valid_timestamps - center).abs()
                minimum = deltas.min()
                tied = valid_timestamps.loc[deltas == minimum]
                before_tied = tied.loc[tied <= center]
                if not before_tied.empty:
                    candidate_index = int(before_tied.index[-1])
                else:
                    candidate_index = int(tied.index[0])
                candidate_timestamp = timestamps.iloc[candidate_index]
                prior = snapshot_timestamp(match.nearest or {})
                new_delta = abs((candidate_timestamp - center).total_seconds())
                old_delta = abs((prior - center).total_seconds()) if prior is not None else float("inf")
                if new_delta < old_delta or (new_delta == old_delta and candidate_timestamp <= center):
                    match.nearest = self._row_at(batch, candidate_index, candidate_timestamp)
            del timestamps, valid_timestamps
        pa.default_memory_pool().release_unused()

    def match_targets(self, targets: Iterable[SnapshotTarget], columns: Iterable[str]) -> dict[str, SnapshotMatch]:
        target_list = list(targets)
        by_symbol: dict[str, list[SnapshotMatch]] = {}
        matches: dict[str, SnapshotMatch] = {}
        for target in target_list:
            match = SnapshotMatch(target=target)
            matches[target.target_id] = match
            by_symbol.setdefault(normalize_symbol(target.symbol), []).append(match)
        self.build_index(set(by_symbol))

        if self.canonical_partitions:
            requested = set(columns) | set(LIGHT_KEY_COLUMNS) | set(LIGHT_TIME_COLUMNS)
            for symbol, symbol_matches in by_symbol.items():
                path = self.canonical_partitions.get(symbol)
                if path is not None and path.exists():
                    self._match_partition_targets(path, requested, symbol_matches)
            return matches

        def accept(row: dict[str, Any]) -> None:
            timestamp = row.get("_snapshot_timestamp")
            if not isinstance(timestamp, pd.Timestamp):
                return
            for match in by_symbol.get(normalize_symbol(row.get("symbol")), []):
                match.scan_count += 1
                match.first_seen = timestamp if match.first_seen is None else min(match.first_seen, timestamp)
                match.last_seen = timestamp if match.last_seen is None else max(match.last_seen, timestamp)
                center = match.target.timestamp
                if timestamp <= center:
                    prior = snapshot_timestamp(match.before or {})
                    if prior is None or timestamp > prior:
                        match.before = row
                if timestamp > center:
                    prior = snapshot_timestamp(match.after or {})
                    if prior is None or timestamp < prior:
                        match.after = row
                prior = snapshot_timestamp(match.nearest or {})
                new_delta = abs((timestamp - center).total_seconds())
                old_delta = abs((prior - center).total_seconds()) if prior is not None else float("inf")
                if new_delta < old_delta or (new_delta == old_delta and timestamp <= center):
                    match.nearest = row

        self.consume(columns, accept)
        return matches


def match_light_snapshots(
    recorder_dir: str | Path,
    session_date: str,
    targets: Iterable[SnapshotTarget],
    columns: Iterable[str],
    *,
    log_prefix: str = "LIGHT_SCANNER",
) -> tuple[dict[str, SnapshotMatch], dict[str, int]]:
    target_list = list(targets)
    if not target_list:
        return {}, {
            "chunks": len(snapshot_chunk_paths(Path(recorder_dir), session_date, "light")),
            "rows_scanned": 0, "rows_session_scoped": 0, "rows_retained": 0,
            "rows_wrong_session": 0, "timestamp_parse_failures": 0,
        }
    with BoundedLightScanner(recorder_dir, session_date, log_prefix=log_prefix) as scanner:
        matches = scanner.match_targets(target_list, columns)
        stats = {
            "chunks": len(scanner.paths),
            "rows_scanned": scanner.rows_scanned,
            "rows_session_scoped": scanner.rows_session_scoped,
            "rows_retained": scanner.retained_rows,
            "rows_wrong_session": scanner.rows_wrong_session,
            "timestamp_parse_failures": scanner.parse_failures,
        }
        return matches, stats
