from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    SymbolState,
    reload_top100_universe_if_requested,
)


class FakeContract:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.conId = abs(hash(symbol)) % 100000
        self.currency = "USD"


class FakeIB:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.cancelled: list[str] = []

    def qualifyContracts(self, contract):
        return [FakeContract(contract.symbol)]

    def reqMktData(self, contract, *_args):
        self.subscribed.append(contract.symbol)
        return SimpleNamespace(contract=contract)

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)


class Top100ReloadTests(unittest.TestCase):
    def test_reload_subscribes_new_top100_and_carries_active_positions_for_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            latest.write_text(
                "symbol,alpha_score,last_close\n"
                "BBB,10,12\n"
                "CCC,9,15\n"
                "EEE,8,20\n",
                encoding="utf-8",
            )
            recorder = LiveDataRecorder(root / "recorder", session_date="2026-05-22")
            ib = FakeIB()
            old_active_contract = FakeContract("DDD")
            contracts = [("AAA", FakeContract("AAA")), ("DDD", old_active_contract)]
            contract_by_symbol = {symbol: contract for symbol, contract in contracts}
            tickers = {
                "AAA": SimpleNamespace(contract=contracts[0][1]),
                "DDD": SimpleNamespace(contract=old_active_contract),
            }
            states = {"AAA": SymbolState("AAA"), "DDD": SymbolState("DDD")}
            latest_snapshots = {"AAA": {"price": 10.0}, "DDD": {"price": 11.0}}
            managed_positions = {
                "DDD": ManagedPosition("DDD", old_active_contract, 1, 11.0, "2026-05-22T12:00:00+00:00", 11.0),
            }
            runtime_state = {
                "top100_reload_requested": True,
                "top100_reload_path": str(latest),
                "top100_reload_ranking_date": "2026-05-21",
            }
            args = SimpleNamespace(
                top_n=3,
                min_price=5.0,
                max_one_trade_per_symbol_per_day=True,
                market_open_utc="13:30",
                alpha_rank_csv=str(latest),
                daily_top100_latest_output=str(latest),
            )

            changed = reload_top100_universe_if_requested(
                ib,
                recorder,
                states,
                contracts,
                contract_by_symbol,
                tickers,
                latest_snapshots,
                managed_positions,
                runtime_state,
                args,
            )

            self.assertTrue(changed)
            self.assertEqual([symbol for symbol, _ in contracts], ["BBB", "CCC", "EEE", "DDD"])
            self.assertEqual(runtime_state["entry_symbols"], {"BBB", "CCC", "EEE"})
            self.assertIn("AAA", ib.cancelled)
            self.assertNotIn("DDD", ib.cancelled)
            self.assertEqual(ib.subscribed, ["BBB", "CCC", "EEE"])
            self.assertIn("DDD", tickers)
            self.assertNotIn("AAA", tickers)
            self.assertFalse(runtime_state["top100_reload_requested"])
            self.assertFalse(runtime_state["entries_blocked"])


if __name__ == "__main__":
    unittest.main()
