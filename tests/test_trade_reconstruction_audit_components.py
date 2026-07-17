from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_trade_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("audit_trade_reconstruction", SCRIPT_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class TradeReconstructionAuditComponentStatusTests(unittest.TestCase):
    def test_single_component_conserved_sell_is_exact_ok(self) -> None:
        status = audit.sell_level_status(
            {
                "component_count": 1,
                "sell_quantity_conservation_diff": 0.0,
                "problem_classification": "wrong_entry_quantity",
            }
        )

        self.assertEqual(status, "OK_COMPONENT_EXACT")

    def test_multiple_components_conserved_sell_is_aggregate_ok(self) -> None:
        status = audit.sell_level_status(
            {
                "component_count": 2,
                "sell_quantity_conservation_diff": 0.0,
                "problem_classification": "ambiguous_match",
            }
        )

        self.assertEqual(status, "OK_COMPONENT_AGGREGATE")

    def test_component_quantity_mismatch_remains_error(self) -> None:
        status = audit.sell_level_status(
            {
                "component_count": 2,
                "sell_quantity_conservation_diff": 1.0,
                "problem_classification": "ambiguous_match",
            }
        )

        self.assertEqual(status, "COMPONENT_QUANTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()
