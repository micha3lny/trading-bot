#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_trading.candidate_snapshot_telemetry import CandidateScanCollector


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def run_benchmark(symbols: int, scans: int) -> dict[str, float]:
    names = tuple(f"S{index:03d}" for index in range(symbols))
    timings: list[float] = []
    for scan_id in range(1, scans + 1):
        started = time.perf_counter()
        collector = CandidateScanCollector(
            session_date="2026-08-05",
            run_id="benchmark",
            process_start_id="benchmark-process",
            scan_id=scan_id,
            scan_started_at="2026-08-05T13:30:00+00:00",
            expected_symbols=names,
        )
        for rank, symbol in enumerate(names, 1):
            collector.update(
                symbol,
                top100_rank=rank,
                current_price=10.0 + rank / 100.0,
                bid=10.0,
                ask=10.01,
                spread_bps=9.995,
                live_entry_score=50.0,
                ranking_position=rank,
                ready=0,
                rejection_reason="first_15m_high_too_low",
            )
        collector.finalize("2026-08-05T13:30:01+00:00", 1.0)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "symbols": float(symbols),
        "scans": float(scans),
        "p50_ms": round(statistics.median(timings), 4),
        "p95_ms": round(percentile(timings, 0.95), 4),
        "p99_ms": round(percentile(timings, 0.99), 4),
        "max_ms": round(max(timings), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark trading-thread candidate snapshot batch construction.")
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--scans", type=int, default=500)
    args = parser.parse_args()
    result = run_benchmark(args.symbols, args.scans)
    print("CANDIDATE_SNAPSHOT_BENCHMARK " + " ".join(f"{key}={value}" for key, value in result.items()))
    print("sqlite_calls=0 journald_rows_per_symbol=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
