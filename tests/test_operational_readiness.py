from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa


def _cache_data(*_args, **_kwargs):
    def decorator(func):
        return func

    return decorator


sys.modules.setdefault(
    "streamlit",
    SimpleNamespace(set_page_config=lambda **_kwargs: None, cache_data=_cache_data),
)
from src.dashboard.runtime_dashboard import display_optional_number, infer_top100_source_date, load_eod_readiness
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class OperationalReadinessTests(unittest.TestCase):
    def test_optional_numeric_display_is_arrow_safe_with_missing_values(self) -> None:
        rendered = pd.DataFrame({"Peak %": pd.Series([1.25, None]).map(display_optional_number)})
        self.assertEqual(rendered["Peak %"].tolist(), ["1.25", "MISSING"])
        table = pa.Table.from_pandas(rendered, preserve_index=False)
        self.assertEqual(str(table.schema.field("Peak %").type), "string")

    def test_top100_source_date_prefers_matching_latest_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            stale = root / "daily_top100_2026-06-18.csv"
            expected = root / "daily_top100_2026-06-22.csv"

            stale.write_text("rank,symbol,score\n1,OLD,1\n", encoding="utf-8")
            expected.write_text("rank,symbol,score\n1,NEW,2\n", encoding="utf-8")
            latest.write_text(expected.read_text(encoding="utf-8"), encoding="utf-8")

            now = time.time()
            # Make the stale dated file closer by mtime; content matching should
            # still choose the real source file.
            stale_touch = now
            expected_touch = now - 3600
            latest_touch = now + 1
            os.utime(stale, (stale_touch, stale_touch))
            os.utime(expected, (expected_touch, expected_touch))
            os.utime(latest, (latest_touch, latest_touch))

            self.assertEqual(infer_top100_source_date(latest), "2026-06-22")

    def test_eod_readiness_is_ok_when_broker_and_sqlite_are_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.record_runtime_event(
                    event_time="2026-06-22T20:00:00+00:00",
                    event_type="EOD_FINAL_STATUS",
                    session_date="2026-06-22",
                    reason="failed",
                    raw_json={"clean": 0, "open_positions": 1},
                )
            finally:
                store.close()

            readiness = load_eod_readiness(
                str(db),
                "2026-06-23",
                broker_open_count=0,
                sqlite_active_count=0,
            )

            self.assertEqual(readiness["status"], "OK")
            self.assertTrue(readiness["flat_confirmed"])
            self.assertFalse(readiness["final_clean"])


if __name__ == "__main__":
    unittest.main()
