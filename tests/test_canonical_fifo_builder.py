from __future__ import annotations

import unittest

from src.live_trading.storage.canonical_fifo import build_canonical_fifo, sort_execution_rows, timestamp_diagnostics


def ex(execution_id: str, side: str, qty: float, price: float, ts: str, **extra):
    return {
        "execution_id": execution_id,
        "strategy_name": "v67",
        "session_date": ts[:10],
        "symbol": extra.pop("symbol", "FIFO"),
        "side": side,
        "quantity": qty,
        "price": price,
        "executed_at": ts,
        "recorded_at": extra.pop("recorded_at", ts),
        "commission": extra.pop("commission", 0.0),
        "realized_pnl": extra.pop("realized_pnl", None),
        **extra,
    }


class CanonicalFifoBuilderTests(unittest.TestCase):
    def test_one_entry_one_exit(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B1", "BOT", 10, 10, "2026-07-15T13:30:00+00:00"),
                ex("S1", "SLD", 10, 11, "2026-07-15T13:40:00+00:00", realized_pnl=10),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertEqual(len(rebuild.components), 1)
        self.assertAlmostEqual(rebuild.trades[0].quantity, 10)
        self.assertAlmostEqual(rebuild.open_quantity, 0)
        self.assertEqual(rebuild.unmatched_sells, [])

    def test_multiple_entry_partials_one_exit_collapses_to_one_trade(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B1", "BOT", 4, 10, "2026-07-15T13:30:00+00:00"),
                ex("B2", "BOT", 6, 12, "2026-07-15T13:31:00+00:00"),
                ex("S1", "SLD", 10, 13, "2026-07-15T13:40:00+00:00", realized_pnl=18),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertEqual(len(rebuild.components), 2)
        self.assertAlmostEqual(rebuild.trades[0].entry_price, 11.2)
        self.assertEqual(rebuild.trades[0].buy_execution_ids, ["B1", "B2"])

    def test_one_entry_multiple_exit_partials_collapses_to_one_trade(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B1", "BOT", 10, 10, "2026-07-15T13:30:00+00:00"),
                ex("S1", "SLD", 4, 11, "2026-07-15T13:40:00+00:00", realized_pnl=4),
                ex("S2", "SLD", 6, 12, "2026-07-15T13:45:00+00:00", realized_pnl=12),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertEqual(len(rebuild.components), 2)
        self.assertAlmostEqual(rebuild.trades[0].exit_price, 11.6)
        self.assertEqual(rebuild.trades[0].sell_execution_ids, ["S1", "S2"])

    def test_old_open_buy_is_consumed_next_day(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B_OLD", "BOT", 5, 20, "2026-07-14T18:00:00+00:00"),
                ex("S_NEXT", "SLD", 5, 21, "2026-07-15T13:35:00+00:00", realized_pnl=5),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertEqual(rebuild.trades[0].session_date, "2026-07-14")
        self.assertEqual(rebuild.components[0].buy_execution_id, "B_OLD")

    def test_old_closed_buy_is_not_reused(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B_OLD", "BOT", 5, 20, "2026-07-14T13:30:00+00:00"),
                ex("S_OLD", "SLD", 5, 21, "2026-07-14T13:40:00+00:00", realized_pnl=5),
                ex("B_NEW", "BOT", 5, 30, "2026-07-15T13:30:00+00:00"),
                ex("S_NEW", "SLD", 5, 31, "2026-07-15T13:40:00+00:00", realized_pnl=5),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 2)
        self.assertEqual(rebuild.trades[1].buy_execution_ids, ["B_NEW"])

    def test_sell_larger_than_available_reports_unmatched_quantity(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B1", "BOT", 5, 10, "2026-07-15T13:30:00+00:00"),
                ex("S1", "SLD", 8, 11, "2026-07-15T13:40:00+00:00", realized_pnl=8),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertAlmostEqual(rebuild.trades[0].quantity, 5)
        self.assertAlmostEqual(rebuild.sell_unmatched["S1"], 3)

    def test_out_of_order_executions_and_equal_timestamps_are_deterministic(self) -> None:
        rows = [
            ex("S1", "SLD", 10, 12, "2026-07-15T13:40:00+00:00", realized_pnl=15),
            ex("B2", "BOT", 5, 11, "2026-07-15T13:30:00+00:00", recorded_at="2026-07-15T13:30:02+00:00"),
            ex("B1", "BOT", 5, 10, "2026-07-15T13:30:00+00:00", recorded_at="2026-07-15T13:30:01+00:00"),
        ]

        rebuild = build_canonical_fifo(rows, symbol="FIFO")

        self.assertEqual([component.buy_execution_id for component in rebuild.components], ["B1", "B2"])
        self.assertEqual(rebuild.trades[0].buy_execution_ids, ["B1", "B2"])

    def test_position_zero_then_second_round_trip_starts_new_trade(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("B1", "BOT", 10, 10, "2026-07-15T13:30:00+00:00"),
                ex("S1", "SLD", 10, 11, "2026-07-15T13:40:00+00:00", realized_pnl=10),
                ex("B2", "BOT", 10, 12, "2026-07-15T13:50:00+00:00"),
                ex("S2", "SLD", 10, 13, "2026-07-15T14:00:00+00:00", realized_pnl=10),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 2)
        self.assertNotEqual(rebuild.trades[0].trade_id, rebuild.trades[1].trade_id)

    def test_space_separator_sell_sorts_after_t_separator_buy(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("S_SPACE", "SLD", 10, 11, "2026-07-02 17:46:15+00:00", realized_pnl=10),
                ex("B_T", "BOT", 10, 10, "2026-07-02T16:49:16+00:00"),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertEqual(rebuild.components[0].buy_execution_id, "B_T")
        self.assertEqual(rebuild.components[0].sell_execution_id, "S_SPACE")
        self.assertAlmostEqual(rebuild.open_quantity, 0)

    def test_mixed_timestamp_formats_close_normal_trade(self) -> None:
        rebuild = build_canonical_fifo(
            [
                ex("S_SPACE", "SLD", 5, 22, "2026-07-15 17:45:23+00:00", realized_pnl=10),
                ex("B_T", "BOT", 5, 20, "2026-07-15T13:38:23+00:00"),
            ],
            symbol="FIFO",
        )

        self.assertEqual(len(rebuild.trades), 1)
        self.assertEqual(rebuild.unmatched_sells, [])
        self.assertTrue(rebuild.timestamp_diagnostics["mixed_timestamp_formats"])
        self.assertTrue(rebuild.timestamp_diagnostics["raw_string_order_differs_from_parsed"])

    def test_timestamp_sort_handles_space_t_microseconds_z_and_tie_break(self) -> None:
        rows = [
            ex("D", "BOT", 1, 10, "2026-07-15T13:30:00.000001+00:00"),
            ex("B", "BOT", 1, 10, "2026-07-15 13:30:00+00:00", recorded_at="2026-07-15T13:30:00.000002Z"),
            ex("A", "BOT", 1, 10, "2026-07-15T13:30:00+00:00", recorded_at="2026-07-15T13:30:00.000001Z"),
            ex("C", "BOT", 1, 10, "2026-07-15T13:30:00Z", recorded_at="2026-07-15T13:30:00.000002Z"),
        ]

        ordered = sort_execution_rows(rows)
        diagnostics = timestamp_diagnostics(rows)

        self.assertEqual([row["execution_id"] for row in ordered], ["A", "B", "C", "D"])
        self.assertIn("space_separator", diagnostics["timestamp_formats"])
        self.assertIn("t_separator", diagnostics["timestamp_formats"])
        self.assertIn("z_suffix", diagnostics["timestamp_formats"])
        self.assertEqual(diagnostics["timestamp_parse_failures"], 0)

    def test_naive_timestamp_is_assumed_utc_and_reported(self) -> None:
        diagnostics = timestamp_diagnostics(
            [
                ex("B_NAIVE", "BOT", 1, 10, "2026-07-15T13:30:00", recorded_at="2026-07-15T13:30:00+00:00"),
                ex("S_Z", "SLD", 1, 11, "2026-07-15T13:31:00Z"),
            ]
        )

        self.assertEqual(diagnostics["naive_timestamp_assumed_utc_count"], 1)
        self.assertEqual(diagnostics["timestamp_parse_failures"], 0)


if __name__ == "__main__":
    unittest.main()
