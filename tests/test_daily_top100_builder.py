from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.live_trading.ranking.daily_top100_builder import build_daily_top, parquet_path, write_output_csv
from src.live_trading.ranking.ranking_store import RankingStore


def write_universe(path: Path, symbols: list[str]) -> None:
    pd.DataFrame({"symbol": symbols}).to_csv(path, index=False)


def session_frame(symbol: str, start_price: float, close_price: float, volume: int = 1_000) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 5, 15, 13, 30, tzinfo=timezone.utc)
    steps = 210
    for idx in range(steps):
        frac = idx / (steps - 1)
        price = start_price + (close_price - start_price) * frac
        rows.append(
            {
                "symbol": symbol,
                "bar_time_utc": (start + timedelta(minutes=idx)).isoformat(),
                "open": price,
                "high": price * (1.005 + 0.0005 * (idx % 5)),
                "low": price * 0.997,
                "close": price,
                "volume": volume,
                "wap": price,
                "trade_count": 10,
                "session_type": "RTH",
            }
        )
    return pd.DataFrame(rows)


def write_session(history_dir: Path, symbol: str, session_date: date, df: pd.DataFrame) -> None:
    path = parquet_path(history_dir, symbol, session_date, "RTH")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise unittest.SkipTest("pyarrow/fastparquet is required for parquet integration test") from exc


class DailyTop100BuilderTests(unittest.TestCase):
    def test_build_daily_top_outputs_compatible_ranked_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            history = root / "history"
            output = root / "daily_top.csv"
            write_universe(universe, ["AAA", "BBB", "MISS"])
            write_session(history, "AAA", date(2026, 5, 15), session_frame("AAA", 10.0, 13.0, 5_000))
            write_session(history, "BBB", date(2026, 5, 15), session_frame("BBB", 20.0, 20.5, 2_000))
            write_session(history, "AAA", date(2026, 5, 14), session_frame("AAA", 9.0, 9.5, 3_000))

            rows, stats = build_daily_top(
                ranking_date=date(2026, 5, 15),
                universe_path=universe,
                history_dir=history,
                top_n=2,
                session_type="RTH",
                min_price=5.0,
                min_bars=180,
                min_volume=100_000,
                min_dollar_volume=500_000,
                prior_sessions=5,
            )
            write_output_csv(output, rows)

            self.assertEqual(len(rows), 2)
            self.assertEqual(stats["missing"], 1)
            self.assertEqual(rows[0]["rank"], 1)
            self.assertEqual(rows[0]["symbol"], "AAA")
            self.assertIn("alpha_score", rows[0])
            self.assertEqual(rows[0]["score"], rows[0]["alpha_score"])
            loaded = pd.read_csv(output)
            self.assertIn("components_json", loaded.columns)
            self.assertEqual(loaded["symbol"].tolist(), ["AAA", "BBB"])

    def test_ranking_store_replaces_one_day_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RankingStore(Path(tmp) / "rankings.sqlite")
            store.replace_daily_rankings(
                "2026-05-15",
                [
                    {"rank": 1, "symbol": "AAA", "score": 80.0, "components_json": "{}"},
                    {"rank": 2, "symbol": "BBB", "score": 70.0, "components_json": "{}"},
                ],
            )
            store.replace_daily_rankings(
                "2026-05-15",
                [{"rank": 1, "symbol": "CCC", "score": 90.0, "components_json": "{}"}],
            )

            rows = store.load_daily_rankings("2026-05-15")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "CCC")
            self.assertEqual(rows[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()

