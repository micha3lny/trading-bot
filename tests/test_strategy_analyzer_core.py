from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.bad_entries_analyzer import build_data_quality_report, build_feature_bucket_report
from src.live_trading.analysis.early_loser_exit_analyzer import build_rules
from src.live_trading.analysis.trade_loader import load_finalized_canonical_trades
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class StrategyAnalyzerCoreTests(unittest.TestCase):
    def test_finalized_loader_excludes_entry_pending_and_includes_pending_pnl_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            for symbol, status, complete in [
                ("CLOSED", "CLOSED", True),
                ("COMM", "COMMISSION_PENDING", True),
                ("PNL", "PNL_PENDING", True),
                ("ENTRY", "ENTRY_PENDING", False),
                ("BROKEN", "CLOSED", False),
            ]:
                row = {
                    "trade_id": f"T_{symbol}",
                    "session_date": "2026-07-17",
                    "strategy_name": "v67",
                    "symbol": symbol,
                    "status": status,
                    "entry_price": 10,
                    "quantity": 1,
                }
                if complete:
                    row.update({"entry_fill_time": "2026-07-17T13:35:00+00:00", "exit_fill_time": "2026-07-17T13:45:00+00:00", "exit_price": 9.9})
                store.upsert_trade(row)
            store.close()
            trades = load_finalized_canonical_trades(db, "2026-07-17", "2026-07-17")
            self.assertEqual(set(trades["symbol"].tolist()), {"CLOSED", "COMM", "PNL"})

    def test_premarket_features_unavailable_when_missing_not_zero(self) -> None:
        df = pd.DataFrame([
            {"symbol": "AAA", "net_pnl": 1.0, "net_pnl_pct": 1.0, "spread_bps_at_entry": 25.0, "top100_rank": 5},
            {"symbol": "BBB", "net_pnl": -1.0, "net_pnl_pct": -1.0, "spread_bps_at_entry": 80.0, "top100_rank": 50},
        ])
        quality = build_data_quality_report(df, "2026-07-17")
        self.assertEqual(quality["premarket_feature_coverage"], "unavailable_for_session")
        feature_rows = build_feature_bucket_report(df, "2026-07-17")
        premarket = feature_rows[feature_rows["feature"] == "premarket_range_pct"].iloc[0]
        self.assertEqual(premarket["coverage"], "unavailable_for_session")
        self.assertEqual(premarket["bucket"], "not_available")

    def test_premarket_feature_available_creates_buckets(self) -> None:
        df = pd.DataFrame([
            {"symbol": "AAA", "net_pnl": 1.0, "premarket_range_pct": 2.5},
            {"symbol": "BBB", "net_pnl": -1.0, "premarket_range_pct": 7.5},
        ])
        quality = build_data_quality_report(df, "2026-07-18")
        self.assertEqual(quality["premarket_feature_coverage"], "available")
        feature_rows = build_feature_bucket_report(df, "2026-07-18")
        rows = feature_rows[feature_rows["feature"] == "premarket_range_pct"]
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("available", set(rows["coverage"].tolist()))

    def test_early_loser_rules_do_not_treat_missing_as_zero(self) -> None:
        paths = pd.DataFrame([
            {"symbol": "AAA", "entry_price": 10.0, "quantity": 10, "net_pnl": -5.0, "final_pnl_pct": -5.0, "pnl_pct_at_5m": None, "positive_seen_to_5m": None},
            {"symbol": "BBB", "entry_price": 10.0, "quantity": 10, "net_pnl": 5.0, "final_pnl_pct": 5.0, "pnl_pct_at_5m": 1.0, "positive_seen_to_5m": 1},
        ])
        rules = build_rules(paths)
        self.assertFalse(rules.empty)
        never_5 = rules[rules["rule"] == "exit_if_never_positive_to_5m"].iloc[0]
        self.assertEqual(int(never_5["affected_trades"]), 0)


if __name__ == "__main__":
    unittest.main()
