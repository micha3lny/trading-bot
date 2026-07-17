from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_trades_from_executions.py"
SPEC = importlib.util.spec_from_file_location("repair_trades_from_executions", SCRIPT_PATH)
assert SPEC and SPEC.loader
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class RepairTradesApplyGuardTests(unittest.TestCase):
    def test_production_path_active_trader_refuses(self) -> None:
        with patch.object(repair, "trader_process_running", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "production DB"):
                repair.enforce_apply_guard(repair.production_db_path())

    def test_relative_production_path_active_trader_refuses(self) -> None:
        with patch.object(repair, "trader_process_running", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "production DB"):
                repair.enforce_apply_guard(Path("data/runtime/trading_runtime.sqlite"))

    def test_copied_db_active_trader_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "trading_runtime.canonical_test.sqlite"
            copied.write_text("")
            with patch.object(repair, "trader_process_running", return_value=True):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    repair.enforce_apply_guard(copied)
            self.assertIn("NON_PRODUCTION_DATABASE_APPLY", stdout.getvalue())
            self.assertIn(str(copied.resolve()), stdout.getvalue())

    def test_symlink_to_production_active_trader_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "prod-link.sqlite"
            link.symlink_to(repair.production_db_path())
            with patch.object(repair, "trader_process_running", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "production DB"):
                    repair.enforce_apply_guard(link)


if __name__ == "__main__":
    unittest.main()

