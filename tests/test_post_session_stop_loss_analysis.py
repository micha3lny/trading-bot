from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.post_session_stop_loss_analysis import analyze_trades, selected_closed_trades
from src.live_trading.ranking.daily_top100_builder import parquet_path
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class PostSessionStopLossAnalysisTests(unittest.TestCase):
    def test_analyzes_trade_path_and_stop_loss_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "runtime.sqlite"
            history_dir = root / "history" / "universe_1m"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "T1",
                    "strategy_name": "v67",
                    "session_date": "2026-06-18",
                    "symbol": "AAA",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-06-18T13:30:00+00:00",
                    "exit_fill_time": "2026-06-18T13:34:00+00:00",
                    "closed_at": "2026-06-18T13:34:00+00:00",
                    "entry_price": 10,
                    "exit_price": 10.5,
                    "quantity": 10,
                    "gross_pnl": 5,
                    "commission": 1,
                    "net_pnl": 4,
                    "exit_reason": "trailing_stop",
                })
            finally:
                store.close()
            path = parquet_path(history_dir, "AAA", pd.Timestamp("2026-06-18").date(), "RTH")
            path.parent.mkdir(parents=True, exist_ok=True)
            candles = pd.DataFrame([
                {"timestamp": "2026-06-18T13:30:00+00:00", "open": 10, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100},
                {"timestamp": "2026-06-18T13:31:00+00:00", "open": 10.1, "high": 11.0, "low": 10.0, "close": 10.8, "volume": 100},
                {"timestamp": "2026-06-18T13:32:00+00:00", "open": 10.8, "high": 10.9, "low": 9.6, "close": 9.8, "volume": 100},
                {"timestamp": "2026-06-18T13:34:00+00:00", "open": 10.3, "high": 10.6, "low": 10.2, "close": 10.5, "volume": 100},
            ])
            try:
                candles.to_parquet(path, index=False)
            except Exception as exc:
                raise unittest.SkipTest("pyarrow/fastparquet is required for parquet integration test") from exc

            trades = selected_closed_trades(db, "2026-06-18")
            trade_rows, variants, summary = analyze_trades(trades, history_dir=history_dir, session_type="RTH", stop_losses=(3.0, 8.0))

            self.assertEqual(len(trade_rows), 1)
            row = trade_rows.iloc[0]
            self.assertAlmostEqual(row["peak_price_since_entry"], 11.0)
            self.assertAlmostEqual(row["low_price_since_entry"], 9.6)
            self.assertAlmostEqual(row["mfe_pct"], 10.0)
            self.assertAlmostEqual(row["mae_pct"], -4.0)
            self.assertEqual(row["actual_exit_reason"], "trailing_stop")

            sl3 = variants[variants["stop_loss_pct"] == 3.0].iloc[0]
            self.assertEqual(sl3["simulated_stop_hit"], 1)
            self.assertAlmostEqual(sl3["simulated_exit_price"], 9.7)
            self.assertAlmostEqual(sl3["simulated_net_pnl"], -4.0)

            sl8_summary = summary[summary["stop_loss_pct"] == 8.0].iloc[0]
            self.assertEqual(sl8_summary["stop_hits"], 0)
            self.assertAlmostEqual(sl8_summary["total_net_pnl"], 4.0)


if __name__ == "__main__":
    unittest.main()
