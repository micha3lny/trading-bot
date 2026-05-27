from __future__ import annotations

import gzip
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.live_trading.unified_logger import (
    append_unified_log,
    daily_log_path,
    log_event,
    monitor_disk_usage,
    run_log_retention,
)


class UnifiedLoggerTests(unittest.TestCase):
    def test_unified_log_file_created_and_events_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            append_unified_log("2026-05-27T19:45:01+00:00 EOD_FLATTEN_START positions=12", log_dir=tmp)
            log_event("ORDER", "EOD_FLATTEN_SENT", log_dir=tmp, symbol="AKAN", qty=4)

            content = daily_log_path(tmp).read_text(encoding="utf-8")
            self.assertIn("INFO EOD EOD_FLATTEN_START positions=12", content)
            self.assertIn("INFO ORDER EOD_FLATTEN_SENT symbol=AKAN qty=4", content)

    def test_retention_compresses_old_logs_and_deletes_old_gz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            today = datetime.now(timezone.utc).date()
            old_day = today - timedelta(days=15)
            very_old_day = today - timedelta(days=31)
            old_log = root / f"trading-bot-{old_day.isoformat()}.log"
            old_log.write_text("old\n", encoding="utf-8")
            very_old_gz = root / f"trading-bot-{very_old_day.isoformat()}.log.gz"
            with gzip.open(very_old_gz, "wt", encoding="utf-8") as fh:
                fh.write("very old\n")

            run_log_retention(tmp)

            self.assertFalse(old_log.exists())
            self.assertTrue((root / f"trading-bot-{old_day.isoformat()}.log.gz").exists())
            self.assertFalse(very_old_gz.exists())

    def test_logging_failure_does_not_crash(self) -> None:
        with patch("src.live_trading.unified_logger.daily_log_path", side_effect=OSError("boom")):
            append_unified_log("SOME_EVENT key=value")
            log_event("BOT", "BOT_START")

    def test_disk_critical_blocks_entries(self) -> None:
        runtime_state: dict = {}
        with patch("src.live_trading.unified_logger.disk_usage_pct", return_value=98.5):
            result = monitor_disk_usage(".", runtime_state, log_dir=tempfile.mkdtemp())
        self.assertEqual(result["level"], "CRITICAL")
        self.assertTrue(result["block_entries"])
        self.assertTrue(runtime_state["disk_full_entries_blocked"])


if __name__ == "__main__":
    unittest.main()
