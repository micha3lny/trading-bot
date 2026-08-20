from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CACHE_SCHEMA_VERSION = 2
KEY_COLUMNS = ("session_date", "process_start_id", "scan_id", "symbol")
TIME_COLUMNS = ("timestamp", "scan_completed_at", "scan_started_at")
GENERATION_COLUMNS = (
    "ranking_source_date", "in_runtime_top100", "entry_symbol_allowed",
    "already_open", "contract_present", "ticker_present",
)
INDEX_SCAN_BATCH_SIZE = 8192
SQLITE_INSERT_BATCH_SIZE = 2048
CANONICAL_SCAN_BATCH_SIZE = 8192
SQLITE_CACHE_KIB = 32768
MAX_ACTIVE_PARQUET_WRITERS = 8


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return round(peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0, 2)


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return round(peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0, 2)


def _disk_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024.0 * 1024.0), 2)


def _phase_log(name: str, state: str, started: float, temp: Path, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"PERSISTENT_LIGHT_CACHE_PHASE_{state} phase={name} elapsed_seconds={time.perf_counter() - started:.3f} "
        f"rss_mb={_rss_mb()} peak_rss_mb={_peak_rss_mb()} arrow_allocated_mb={pa.total_allocated_bytes() / (1024 * 1024):.2f} "
        f"temp_disk_mb={_disk_mb(temp)} {details}".rstrip(),
        flush=True,
    )


def default_cache_root(recorder_dir: Path) -> Path:
    configured = os.environ.get("TRADING_ANALYSIS_LIGHT_CACHE_DIR")
    if configured:
        return Path(configured)
    resolved = recorder_dir.resolve()
    if resolved.name == "recorder" and resolved.parent.name == "live":
        return resolved.parent.parent / "analysis" / "cache" / "light_index"
    return recorder_dir.parent / ".analysis_cache" / "light_index"


def source_fingerprint(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        schema = pq.ParquetFile(path).schema_arrow
        sources.append({
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "schema": [(field.name, str(field.type)) for field in schema],
        })
    payload = json.dumps(
        {"version": CACHE_SCHEMA_VERSION, "sources": sources}, sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest(), sources


def _valid_cache(root: Path, fingerprint: str) -> bool:
    manifest = root / "manifest.json"
    database = root / "light_index.sqlite"
    if not manifest.exists() or not database.exists():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION or metadata.get("fingerprint") != fingerprint:
            return False
        partitions = metadata.get("canonical_partitions") or {}
        if any(not (root / relative).exists() for relative in partitions.values()):
            return False
        summary = metadata.get("top100_generation_summary")
        if not summary or not (root / str(summary)).exists():
            return False
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
            return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    except Exception:
        return False


def _unified_schema(paths: list[Path]) -> pa.Schema | None:
    schemas = []
    for path in paths:
        try:
            schemas.append(pq.ParquetFile(path).schema_arrow)
        except Exception:
            continue
    return pa.unify_schemas(schemas, promote_options="permissive") if schemas else None


def _truthy(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y", "on"})
    try:
        return int(int(value) == 1)
    except (TypeError, ValueError, OverflowError):
        return int(bool(value))


def _comma_set(value: Any) -> set[str]:
    return {part for part in str(value or "").split(",") if part}


def _build_generation_summary(conn: sqlite3.Connection, target: Path) -> int:
    query = """
        SELECT process_start_id, scan_id, MAX(timestamp) AS timestamp,
               GROUP_CONCAT(DISTINCT NULLIF(ranking_source_date,'')) AS source_dates,
               MAX(in_top100_present) AS in_top100_present,
               GROUP_CONCAT(DISTINCT CASE WHEN in_runtime_top100=1 THEN symbol END) AS symbols,
               GROUP_CONCAT(DISTINCT CASE WHEN entry_symbol_allowed=1 THEN symbol END) AS entry_symbols,
               GROUP_CONCAT(DISTINCT CASE WHEN in_runtime_top100=0 AND entry_symbol_allowed=0
                    AND (already_open=1 OR contract_present=1 OR ticker_present=1) THEN symbol END) AS retained_symbols
        FROM retained
        GROUP BY process_start_id, scan_id
        ORDER BY timestamp, process_start_id,
                 CASE WHEN scan_id<>'' AND scan_id NOT GLOB '*[^0-9]*' THEN 0 ELSE 1 END,
                 CAST(scan_id AS INTEGER), scan_id
    """
    generations: list[dict[str, Any]] = []
    for row in conn.execute(query):
        scan = {
            "process_start_id": str(row[0] or ""),
            "scan_id": str(row[1] or ""),
            "timestamp": str(row[2] or ""),
            "source_dates": _comma_set(row[3]),
            "in_top100_present": bool(row[4]),
            "symbols": _comma_set(row[5]),
            "entry_symbols": _comma_set(row[6]),
            "retained_symbols": _comma_set(row[7]),
        }
        symbols = scan["symbols"] if scan["in_top100_present"] else scan["entry_symbols"]
        timestamp = pd.to_datetime(scan["timestamp"], errors="coerce", utc=True)
        if pd.isna(timestamp) or len(scan["source_dates"]) != 1 or not symbols:
            continue
        source_date = next(iter(scan["source_dates"]))
        previous = generations[-1] if generations else None
        if (
            previous
            and previous["process_start_id"] == scan["process_start_id"]
            and previous["ranking_source_date"] == source_date
            and set(previous["symbols"]) == symbols
            and set(previous["entry_symbols"]) == scan["entry_symbols"]
        ):
            previous["last_seen_at"] = timestamp.isoformat()
            previous["retained_symbols"] = sorted(set(previous["retained_symbols"]) | scan["retained_symbols"])
            previous["scan_count"] += 1
            previous["last_scan_id"] = scan["scan_id"]
            continue
        generations.append({
            "process_start_id": scan["process_start_id"],
            "ranking_source_date": source_date,
            "effective_at": timestamp.isoformat(),
            "last_seen_at": timestamp.isoformat(),
            "symbols": sorted(symbols),
            "entry_symbols": sorted(scan["entry_symbols"]),
            "retained_symbols": sorted(scan["retained_symbols"]),
            "scan_count": 1,
            "first_scan_id": scan["scan_id"],
            "last_scan_id": scan["scan_id"],
        })
    target.write_text(json.dumps(generations, separators=(",", ":")), encoding="utf-8")
    return len(generations)


def _build_canonical_partitions(
    paths: list[Path],
    conn: sqlite3.Connection,
    target: Path,
) -> tuple[dict[str, str], int, float]:
    """Persist deduplicated raw LIGHT rows once, partitioned by symbol."""
    schema = _unified_schema(paths)
    if schema is None:
        return {}, 0, _disk_mb(target.parent)
    symbols = [
        str(row[0])
        for row in conn.execute("SELECT DISTINCT symbol FROM retained WHERE symbol<>'' ORDER BY symbol")
    ]
    ordinal = {symbol: index for index, symbol in enumerate(symbols)}
    target.mkdir(parents=True, exist_ok=True)
    fragments_root = target.parent / "canonical_fragments"
    fragments_root.mkdir(parents=True, exist_ok=True)
    bucket_by_symbol = {
        symbol: index // MAX_ACTIVE_PARQUET_WRITERS
        for index, symbol in enumerate(symbols)
    }
    bucket_symbols: dict[int, list[str]] = {}
    for symbol, bucket in bucket_by_symbol.items():
        bucket_symbols.setdefault(bucket, []).append(symbol)
    fragments: dict[int, list[Path]] = {bucket: [] for bucket in bucket_symbols}
    relative_paths: dict[str, str] = {}
    fragment_sequence = 0
    base_bytes = sum(
        item.stat().st_size
        for item in target.parent.iterdir()
        if item.is_file()
    )
    fragment_bytes = 0
    canonical_bytes = 0
    peak_temp_bytes = base_bytes
    for source_index, path in enumerate(paths):
        try:
            parquet = pq.ParquetFile(path)
        except Exception:
            continue
        source_row = 0
        for batch in parquet.iter_batches(batch_size=CANONICAL_SCAN_BATCH_SIZE):
            batch_end = source_row + batch.num_rows
            retained_rows = [
                int(row[0]) - source_row
                for row in conn.execute(
                    "SELECT row_index FROM retained WHERE source_index=? AND row_index>=? AND row_index<? ORDER BY row_index",
                    (source_index, source_row, batch_end),
                )
            ]
            source_row = batch_end
            if not retained_rows:
                continue
            table = pa.Table.from_batches([batch]).take(pa.array(retained_rows, type=pa.int64()))
            arrays = []
            for field in schema:
                if field.name in table.column_names:
                    column = table[field.name]
                    if field.name == "symbol":
                        column = pc.utf8_upper(pc.utf8_trim_whitespace(pc.fill_null(pc.cast(column, pa.string()), "")))
                    if not column.type.equals(field.type):
                        column = pc.cast(column, field.type)
                else:
                    column = pa.nulls(table.num_rows, type=field.type)
                arrays.append(column)
            normalized = pa.Table.from_arrays(arrays, schema=schema)
            present_buckets = {
                bucket_by_symbol[str(raw_symbol)]
                for raw_symbol in pc.unique(normalized["symbol"]).to_pylist()
                if str(raw_symbol or "") in bucket_by_symbol
            }
            for bucket in sorted(present_buckets):
                group = normalized.filter(
                    pc.is_in(normalized["symbol"], value_set=pa.array(bucket_symbols[bucket], type=pa.string()))
                )
                fragment_dir = fragments_root / f"bucket-{bucket:04d}"
                fragment_dir.mkdir(parents=True, exist_ok=True)
                fragment_path = fragment_dir / f"part-{fragment_sequence:08d}.parquet"
                pq.write_table(group, fragment_path, compression="zstd")
                fragments[bucket].append(fragment_path)
                fragment_bytes += fragment_path.stat().st_size
                peak_temp_bytes = max(peak_temp_bytes, base_bytes + fragment_bytes + canonical_bytes)
                fragment_sequence += 1
            del normalized, table, batch
        pa.default_memory_pool().release_unused()

    max_active_writers = 0
    for bucket in sorted(bucket_symbols):
        bucket_fragments = fragments.get(bucket) or []
        if not bucket_fragments:
            continue
        writers: dict[str, pq.ParquetWriter] = {}
        try:
            for symbol in bucket_symbols[bucket]:
                relative = Path("canonical") / f"symbol-{ordinal[symbol]:04d}.parquet"
                relative_paths[symbol] = str(relative)
                writers[symbol] = pq.ParquetWriter(target.parent / relative, schema, compression="zstd")
            max_active_writers = max(max_active_writers, len(writers))
            for fragment in bucket_fragments:
                fragment_file = pq.ParquetFile(fragment)
                for batch in fragment_file.iter_batches(batch_size=CANONICAL_SCAN_BATCH_SIZE):
                    table = pa.Table.from_batches([batch])
                    for raw_symbol in pc.unique(table["symbol"]).to_pylist():
                        symbol = str(raw_symbol or "")
                        writer = writers.get(symbol)
                        if writer is not None:
                            writer.write_table(table.filter(pc.equal(table["symbol"], symbol)))
        finally:
            for writer in writers.values():
                writer.close()
        bucket_canonical_bytes = sum(
            (target.parent / relative_paths[symbol]).stat().st_size
            for symbol in writers
        )
        canonical_bytes += bucket_canonical_bytes
        peak_temp_bytes = max(peak_temp_bytes, base_bytes + fragment_bytes + canonical_bytes)
        removed_fragment_bytes = sum(path.stat().st_size for path in bucket_fragments if path.exists())
        shutil.rmtree(fragments_root / f"bucket-{bucket:04d}", ignore_errors=True)
        fragment_bytes -= removed_fragment_bytes
        pa.default_memory_pool().release_unused()
    shutil.rmtree(fragments_root, ignore_errors=True)
    print(
        f"PERSISTENT_LIGHT_CACHE_WRITERS max_active_writers={max_active_writers} "
        f"writer_limit={MAX_ACTIVE_PARQUET_WRITERS} symbols={len(relative_paths)}",
        flush=True,
    )
    return relative_paths, max_active_writers, round(peak_temp_bytes / (1024.0 * 1024.0), 2)


def ensure_persistent_light_index(
    recorder_dir: Path,
    session_date: str,
    paths: list[Path],
    *,
    batch_size: int = 65536,
    cache_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    fingerprint_started = time.perf_counter()
    cache_base = cache_root or default_cache_root(recorder_dir)
    fingerprint_probe = Path(cache_base) / session_date / ".fingerprint-probe"
    _phase_log("fingerprint_schema", "START", fingerprint_started, fingerprint_probe, chunks=len(paths))
    fingerprint, sources = source_fingerprint(paths)
    root = Path(cache_base) / session_date / fingerprint
    _phase_log("fingerprint_schema", "DONE", fingerprint_started, fingerprint_probe, chunks=len(paths))
    if _valid_cache(root, fingerprint):
        metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return root / "light_index.sqlite", {**metadata, "cache_hit": 1}
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{fingerprint[:12]}-", dir=root.parent))
    database_path = temp / "light_index.sqlite"
    rows_scanned = rows_session_scoped = wrong_session = parse_failures = 0
    ranking_source_dates: set[str] = set()
    try:
        with sqlite3.connect(database_path) as conn:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KIB}")
            conn.execute(
                "CREATE TABLE retained (dedupe_key TEXT PRIMARY KEY, source_index INTEGER NOT NULL, "
                "row_index INTEGER NOT NULL, symbol TEXT NOT NULL, timestamp TEXT, "
                "process_start_id TEXT, scan_id TEXT, ranking_source_date TEXT, "
                "in_top100_present INTEGER NOT NULL, in_runtime_top100 INTEGER NOT NULL, "
                "entry_symbol_allowed INTEGER NOT NULL, already_open INTEGER NOT NULL, "
                "contract_present INTEGER NOT NULL, ticker_present INTEGER NOT NULL)"
            )
            conn.execute("CREATE INDEX retained_source ON retained(source_index, row_index)")
            conn.execute("CREATE INDEX retained_symbol ON retained(symbol,timestamp)")
            phase_started = time.perf_counter()
            _phase_log("sqlite_dedupe", "START", phase_started, temp, chunks=len(paths))
            effective_batch_size = min(batch_size, INDEX_SCAN_BATCH_SIZE)
            for source_index, path in enumerate(paths):
                parquet = pq.ParquetFile(path)
                available = set(parquet.schema_arrow.names)
                columns = [
                    name for name in (
                        *KEY_COLUMNS, "trading_session_date", "trade_session_date",
                        *GENERATION_COLUMNS, *TIME_COLUMNS,
                    ) if name in available
                ]
                source_row = 0
                for batch in parquet.iter_batches(batch_size=effective_batch_size, columns=columns):
                    frame = batch.to_pandas()
                    count = len(frame)
                    rows_scanned += count
                    frame["_row_index"] = np.arange(source_row, source_row + count, dtype=np.int64)
                    source_row += count
                    for column in (*KEY_COLUMNS, "trading_session_date", "trade_session_date", *GENERATION_COLUMNS):
                        if column not in frame:
                            frame[column] = None
                    timestamps = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
                    for column in TIME_COLUMNS:
                        if column in frame:
                            timestamps = timestamps.combine_first(pd.to_datetime(frame[column], errors="coerce", utc=True))
                    explicit = frame["session_date"].combine_first(frame["trading_session_date"]).combine_first(frame["trade_session_date"])
                    explicit = explicit.fillna("").astype(str).str[:10]
                    resolved = explicit.where(explicit.ne(""), timestamps.dt.strftime("%Y-%m-%d"))
                    valid = resolved.eq(session_date)
                    rows_session_scoped += int(valid.sum())
                    wrong_session += int((~valid).sum())
                    selected = frame.loc[valid]
                    if "ranking_source_date" in selected:
                        ranking_source_dates.update(selected["ranking_source_date"].dropna().astype(str))
                    selected_timestamps = timestamps.loc[valid]
                    parse_failures += int(selected_timestamps.isna().sum())
                    symbols = selected["symbol"].fillna("").astype(str).str.upper().str.strip()
                    process_ids = selected["process_start_id"].fillna("<NA>").astype(str)
                    scan_ids = selected["scan_id"].fillna("<NA>").astype(str)
                    keys = (
                        selected["session_date"].fillna(session_date).astype(str) + "\x1f"
                        + process_ids + "\x1f" + scan_ids + "\x1f" + symbols
                    )
                    ranking_sources = selected["ranking_source_date"].fillna("").astype(str)
                    in_top_present = selected["in_runtime_top100"].notna().astype(int)
                    flag_columns = {
                        column: selected[column].map(_truthy).astype(int)
                        for column in (
                            "in_runtime_top100", "entry_symbol_allowed", "already_open",
                            "contract_present", "ticker_present",
                        )
                    }
                    values = zip(
                        keys.tolist(),
                        [source_index] * len(selected),
                        selected["_row_index"].astype(int).tolist(),
                        symbols.tolist(),
                        selected_timestamps.astype(str).tolist(),
                        process_ids.tolist(), scan_ids.tolist(), ranking_sources.tolist(),
                        in_top_present.tolist(), flag_columns["in_runtime_top100"].tolist(),
                        flag_columns["entry_symbol_allowed"].tolist(), flag_columns["already_open"].tolist(),
                        flag_columns["contract_present"].tolist(), flag_columns["ticker_present"].tolist(),
                    )
                    insert_batch: list[tuple[Any, ...]] = []
                    for value in values:
                        insert_batch.append(value)
                        if len(insert_batch) >= SQLITE_INSERT_BATCH_SIZE:
                            conn.executemany("INSERT OR REPLACE INTO retained VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", insert_batch)
                            insert_batch.clear()
                    if insert_batch:
                        conn.executemany("INSERT OR REPLACE INTO retained VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", insert_batch)
                    del frame, selected, keys, symbols
                conn.commit()
                pa.default_memory_pool().release_unused()
            _phase_log("sqlite_dedupe", "DONE", phase_started, temp, rows_scanned=rows_scanned)
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("persistent LIGHT index integrity_check failed")
            summary_started = time.perf_counter()
            _phase_log("generation_summary", "START", summary_started, temp)
            generation_count = _build_generation_summary(conn, temp / "top100_generations.json")
            _phase_log("generation_summary", "DONE", summary_started, temp, generations=generation_count)
            phase_started = time.perf_counter()
            _phase_log("canonical_partitions", "START", phase_started, temp)
            canonical_partitions, max_active_writers, peak_temp_disk_mb = _build_canonical_partitions(
                paths, conn, temp / "canonical"
            )
            _phase_log(
                "canonical_partitions", "DONE", phase_started, temp,
                partitions=len(canonical_partitions),
                max_active_writers=max_active_writers,
                peak_temp_disk_mb=peak_temp_disk_mb,
            )
        publish_started = time.perf_counter()
        _phase_log("manifest_publish", "START", publish_started, temp)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "session_date": session_date,
            "source_chunks": len(paths),
            "sources": sources,
            "rows_scanned": rows_scanned,
            "rows_session_scoped": rows_session_scoped,
            "rows_wrong_session": wrong_session,
            "timestamp_parse_failures": parse_failures,
            "ranking_source_dates": sorted(ranking_source_dates),
            "top100_generation_summary": "top100_generations.json",
            "top100_generation_count": generation_count,
            "canonical_partitions": canonical_partitions,
            "canonical_partition_count": len(canonical_partitions),
            "max_active_writers": max_active_writers,
            "peak_temp_disk_mb": peak_temp_disk_mb,
            "canonical_bytes": sum(
                path.stat().st_size for path in (temp / "canonical").glob("*.parquet")
            ),
        }
        (temp / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.replace(temp, root)
        except OSError:
            if not _valid_cache(root, fingerprint):
                raise
            shutil.rmtree(temp, ignore_errors=True)
        _phase_log("manifest_publish", "DONE", publish_started, root, canonical_bytes=metadata["canonical_bytes"])
        return root / "light_index.sqlite", {**metadata, "cache_hit": 0}
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
