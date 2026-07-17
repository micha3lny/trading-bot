from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inventory_continuity_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("inventory_continuity_diagnostic", SCRIPT_PATH)
assert SPEC and SPEC.loader
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)


class InventoryContinuityDiagnosticTests(unittest.TestCase):
    def test_positive_to_zero_is_complete_cycle(self) -> None:
        self.assertEqual(
            diag.classify_cycle(
                initial_history_date="2026-07-01",
                position_before=10,
                position_after=0,
                current_open_qty=0,
                final_reconstructed_qty=0,
                latest_position_qty=0,
            ),
            "COMPLETE_HISTORY_CYCLE",
        )

    def test_small_residual_is_dust(self) -> None:
        self.assertEqual(
            diag.classify_cycle(
                initial_history_date="2026-07-01",
                position_before=10,
                position_after=0.5,
                current_open_qty=0.5,
                final_reconstructed_qty=0.5,
                latest_position_qty=0.5,
            ),
            "RESIDUAL_DUST",
        )

    def test_positive_to_positive_is_carry(self) -> None:
        self.assertEqual(
            diag.classify_cycle(
                initial_history_date="2026-07-01",
                position_before=10,
                position_after=3,
                current_open_qty=3,
                final_reconstructed_qty=3,
                latest_position_qty=3,
            ),
            "CARRIED_POSITION_CONFIRMED",
        )

    def test_final_quantity_diff_can_indicate_missing_execution(self) -> None:
        self.assertEqual(
            diag.classify_cycle(
                initial_history_date="2026-07-01",
                position_before=0,
                position_after=0,
                current_open_qty=0,
                final_reconstructed_qty=5,
                latest_position_qty=0,
            ),
            "POSSIBLE_MISSING_EXECUTION",
        )


if __name__ == "__main__":
    unittest.main()

