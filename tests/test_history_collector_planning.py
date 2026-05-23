from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.live_trading.data.v68_universe_1m_parquet_collector import (
    CollectTask,
    build_pending_tasks,
    build_tasks,
    collect_existing_parquet_keys,
    completion_pct,
    count_existing_parquets_for_tasks,
    parquet_path,
)


class HistoryCollectorPlanningTests(unittest.TestCase):
    def test_build_tasks_skips_weekends_by_default(self) -> None:
        tasks = build_tasks(["AAA"], date(2026, 1, 2), date(2026, 1, 5), "RTH")
        self.assertEqual([task.session_date.isoformat() for task in tasks], ["2026-01-02", "2026-01-05"])

        with_weekends = build_tasks(["AAA"], date(2026, 1, 2), date(2026, 1, 5), "RTH", include_weekends=True)
        self.assertEqual(len(with_weekends), 4)

    def test_pending_plan_treats_existing_parquet_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = CollectTask("AAA", date(2026, 5, 15), "RTH")
            path = parquet_path(output_dir, task.symbol, task.session_date, task.session_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-empty")

            pending, stats = build_pending_tasks(
                [task],
                status={},
                failures={},
                output_dir=output_dir,
                max_attempts=5,
                retry_failed=False,
            )

            self.assertEqual(pending, [])
            self.assertEqual(stats["complete"], 1)
            self.assertEqual(stats["pending"], 0)

    def test_existing_parquet_inventory_can_sync_status_for_fast_future_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = CollectTask("AAA", date(2026, 5, 15), "RTH")
            path = parquet_path(output_dir, task.symbol, task.session_date, task.session_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-empty")
            status = {}

            existing_keys = collect_existing_parquet_keys(output_dir, "RTH")
            pending, stats = build_pending_tasks(
                [task],
                status=status,
                failures={},
                output_dir=output_dir,
                max_attempts=5,
                retry_failed=False,
                existing_parquet_keys=existing_keys,
                sync_existing_status=True,
            )

            self.assertEqual(pending, [])
            self.assertEqual(stats["complete"], 1)
            self.assertEqual(stats["synced_existing"], 1)
            self.assertEqual(status["AAA_2026-05-15_RTH"]["status"], "complete")

    def test_partial_status_overrides_existing_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = CollectTask("AAA", date(2026, 5, 15), "RTH")
            path = parquet_path(output_dir, task.symbol, task.session_date, task.session_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-empty")

            pending, stats = build_pending_tasks(
                [task],
                status={"AAA_2026-05-15_RTH": {"status": "partial"}},
                failures={},
                output_dir=output_dir,
                max_attempts=5,
                retry_failed=False,
            )

            self.assertEqual(pending, [task])
            self.assertEqual(stats["complete"], 0)
            self.assertEqual(stats["pending"], 1)

    def test_output_summary_counts_existing_parquets_for_requested_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            tasks = [
                CollectTask("AAA", date(2026, 5, 15), "RTH"),
                CollectTask("BBB", date(2026, 5, 15), "RTH"),
            ]
            path = parquet_path(output_dir, tasks[0].symbol, tasks[0].session_date, tasks[0].session_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-empty")

            self.assertEqual(count_existing_parquets_for_tasks(output_dir, tasks), 1)
            self.assertEqual(completion_pct(1, 2), 50.0)


if __name__ == "__main__":
    unittest.main()
