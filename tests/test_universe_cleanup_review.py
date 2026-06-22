from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.review_universe_cleanup import apply_review_cleanup, build_review_rows, write_review_csv


class UniverseCleanupReviewTests(unittest.TestCase):
    def test_review_candidates_from_top100_and_collector_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diag_dir = root / "universe"
            diag_dir.mkdir()
            pd.DataFrame(
                [
                    {"date": "2026-06-16", "symbol": "BADW", "status": "rejected", "reason": "warrant_suffix"},
                    {"date": "2026-06-17", "symbol": "BADW", "status": "rejected", "reason": "warrant_suffix"},
                    {"date": "2026-06-18", "symbol": "MISS", "status": "missing", "reason": "missing_history"},
                ]
            ).to_csv(diag_dir / "daily_top100_2026-06-18_diagnostics.csv", index=False)
            status = {
                "MISS_2026-06-17_RTH": {
                    "symbol": "MISS",
                    "date": "2026-06-17",
                    "session_type": "RTH",
                    "status": "no_data",
                    "last_error": "historical_failed: Error 162 HMDS query returned no data",
                },
                "MISS_2026-06-18_RTH": {
                    "symbol": "MISS",
                    "date": "2026-06-18",
                    "session_type": "RTH",
                    "status": "no_data_permanent",
                    "last_error": "empty_bars",
                },
            }
            status_path = root / "collector_status.json"
            failures_path = root / "collector_failures.json"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            failures_path.write_text("{}", encoding="utf-8")

            rows = build_review_rows(
                diagnostics_glob=str(diag_dir / "daily_top100_*_diagnostics.csv"),
                collector_status=status_path,
                collector_failures=failures_path,
                min_count=2,
                start_date=None,
                end_date=None,
            )

            by_symbol = {row["symbol"]: row for row in rows}
            self.assertEqual(by_symbol["BADW"]["reason"], "non_common_stock_product")
            self.assertEqual(by_symbol["BADW"]["suggested_action"], "remove_from_universe")
            self.assertEqual(by_symbol["MISS"]["reason"], "ibkr_no_data")
            self.assertEqual(by_symbol["MISS"]["collector_no_data_count"], 2)

    def test_apply_cleanup_only_removes_approved_review_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            review = root / "review.csv"
            output = root / "clean.csv"
            pd.DataFrame({"symbol": ["KEEP", "DROP", "WAIT"], "alpha_score": [3, 2, 1]}).to_csv(universe, index=False)
            write_review_csv(
                review,
                [
                    {
                        "symbol": "DROP",
                        "reason": "ibkr_no_data",
                        "count": 4,
                        "first_seen_date": "2026-06-15",
                        "last_seen_date": "2026-06-18",
                        "suggested_action": "remove_from_universe",
                        "top100_missing_count": 0,
                        "top100_rejected_count": 0,
                        "top100_error_count": 0,
                        "collector_no_data_count": 4,
                        "collector_partial_count": 0,
                        "collector_failed_count": 0,
                        "statuses": "no_data:4",
                        "examples": "",
                        "notes": "",
                        "approved": "1",
                    },
                    {
                        "symbol": "WAIT",
                        "reason": "missing_history",
                        "count": 4,
                        "first_seen_date": "2026-06-15",
                        "last_seen_date": "2026-06-18",
                        "suggested_action": "investigate_history",
                        "top100_missing_count": 4,
                        "top100_rejected_count": 0,
                        "top100_error_count": 0,
                        "collector_no_data_count": 0,
                        "collector_partial_count": 0,
                        "collector_failed_count": 0,
                        "statuses": "missing:4",
                        "examples": "",
                        "notes": "",
                        "approved": "1",
                    },
                ],
            )

            result = apply_review_cleanup(universe_path=universe, review_path=review, output_universe=output)
            cleaned = pd.read_csv(output)

            self.assertEqual(result["removed_symbols"], ["DROP"])
            self.assertEqual(cleaned["symbol"].tolist(), ["KEEP", "WAIT"])


if __name__ == "__main__":
    unittest.main()
