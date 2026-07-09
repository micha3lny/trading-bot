from __future__ import annotations

import contextlib
import io
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
    def __init__(self, fail_symbols: set[str] | None = None) -> None:
        self.subscribed: list[str] = []
        self.cancelled: list[str] = []
        self.fail_symbols = fail_symbols or set()

    def qualifyContracts(self, contract):
        return [FakeContract(contract.symbol)]

    def reqMktData(self, contract, *_args):
        if contract.symbol in self.fail_symbols:
            raise RuntimeError("Error 101: Max number of tickers has been reached")
        self.subscribed.append(contract.symbol)
        return SimpleNamespace(contract=contract)

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)


class Top100ReloadTests(unittest.TestCase):
    def write_latest(self, path: Path, symbols: list[str]) -> None:
        rows = ["symbol,alpha_score,last_close"]
        for idx, symbol in enumerate(symbols):
            rows.append(f"{symbol},{1000 - idx},10")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def args_for(self, latest: Path, *, top_n: int, cap: int = 100) -> SimpleNamespace:
        root = latest.parent
        return SimpleNamespace(
            top_n=top_n,
            min_price=5.0,
            max_one_trade_per_symbol_per_day=True,
            market_open_utc="13:30",
            alpha_rank_csv=str(latest),
            daily_top100_latest_output=str(latest),
            max_market_data_subscriptions=cap,
            symbol_denylist=str(root / "symbol_denylist.csv"),
            runtime_ineligible_path=str(root / "ineligible_symbols.json"),
        )

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
            args = self.args_for(latest, top_n=3)

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
            self.assertEqual([symbol for symbol, _ in contracts], ["DDD", "BBB", "CCC", "EEE"])
            self.assertEqual(runtime_state["entry_symbols"], {"BBB", "CCC", "EEE"})
            self.assertIn("AAA", ib.cancelled)
            self.assertNotIn("DDD", ib.cancelled)
            self.assertEqual(ib.subscribed, ["BBB", "CCC", "EEE"])
            self.assertIn("DDD", tickers)
            self.assertNotIn("AAA", tickers)
            self.assertFalse(runtime_state["top100_reload_requested"])
            self.assertFalse(runtime_state["entries_blocked"])

    def test_reload_emits_subscription_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            self.write_latest(latest, ["AAA", "BBB"])
            recorder = LiveDataRecorder(root / "recorder", session_date="2026-05-22")
            ib = FakeIB()
            runtime_state = {"top100_reload_requested": True, "top100_reload_path": str(latest), "top100_reload_ranking_date": "2026-05-21"}
            contracts: list[tuple[str, FakeContract]] = []
            contract_by_symbol: dict[str, FakeContract] = {}
            tickers = {}
            states: dict[str, SymbolState] = {}

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                changed = reload_top100_universe_if_requested(
                    ib,
                    recorder,
                    states,
                    contracts,
                    contract_by_symbol,
                    tickers,
                    {},
                    {},
                    runtime_state,
                    self.args_for(latest, top_n=2, cap=100),
                )

            self.assertTrue(changed)
            text = output.getvalue()
            self.assertIn("TOP100_REFRESH_DIFF", text)
            self.assertIn("TOP100_SUBSCRIPTION_RECONCILE", text)
            self.assertIn("MARKET_DATA_SUBSCRIPTION_ACTIONS", text)
            self.assertIn("RUNTIME_STATE_SESSION_BOUNDARY_CHECK", text)

    def test_reload_caps_top100_after_active_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            top_symbols = [f"T{i:03d}" for i in range(100)]
            active_symbols = [f"A{i:03d}" for i in range(14)]
            self.write_latest(latest, top_symbols)
            recorder = LiveDataRecorder(root / "recorder", session_date="2026-05-22")
            ib = FakeIB()
            active_contracts = {symbol: FakeContract(symbol) for symbol in active_symbols}
            contracts = [(symbol, contract) for symbol, contract in active_contracts.items()]
            contract_by_symbol = dict(contracts)
            tickers = {symbol: SimpleNamespace(contract=contract) for symbol, contract in active_contracts.items()}
            states = {symbol: SymbolState(symbol) for symbol in active_symbols}
            managed_positions = {
                symbol: ManagedPosition(symbol, contract, 1, 10.0, "2026-05-22T12:00:00+00:00", 10.0)
                for symbol, contract in active_contracts.items()
            }
            runtime_state = {"top100_reload_requested": True, "top100_reload_path": str(latest), "top100_reload_ranking_date": "2026-05-21"}

            changed = reload_top100_universe_if_requested(
                ib, recorder, states, contracts, contract_by_symbol, tickers, {}, managed_positions, runtime_state, self.args_for(latest, top_n=100, cap=100)
            )

            self.assertTrue(changed)
            self.assertEqual(len(contracts), 100)
            self.assertTrue(set(active_symbols).issubset({symbol for symbol, _ in contracts}))
            self.assertEqual(len([symbol for symbol, _ in contracts if symbol.startswith("T")]), 86)
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["skipped_due_to_subscription_cap"], 14)
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["skipped_symbols_due_to_cap"], top_symbols[86:])
            self.assertEqual(len(ib.subscribed), 86)

    def test_reload_counts_active_top100_overlap_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            top_symbols = [f"T{i:03d}" for i in range(5)]
            self.write_latest(latest, top_symbols)
            recorder = LiveDataRecorder(root / "recorder", session_date="2026-05-22")
            ib = FakeIB()
            active_contracts = {"T000": FakeContract("T000"), "ACTIVE": FakeContract("ACTIVE")}
            contracts = list(active_contracts.items())
            contract_by_symbol = dict(contracts)
            tickers = {symbol: SimpleNamespace(contract=contract) for symbol, contract in active_contracts.items()}
            managed_positions = {
                symbol: ManagedPosition(symbol, contract, 1, 10.0, "2026-05-22T12:00:00+00:00", 10.0)
                for symbol, contract in active_contracts.items()
            }
            runtime_state = {"top100_reload_requested": True, "top100_reload_path": str(latest), "top100_reload_ranking_date": "2026-05-21"}

            changed = reload_top100_universe_if_requested(
                ib, recorder, {}, contracts, contract_by_symbol, tickers, {}, managed_positions, runtime_state, self.args_for(latest, top_n=5, cap=5)
            )

            self.assertTrue(changed)
            self.assertEqual([symbol for symbol, _ in contracts], ["ACTIVE", "T000", "T001", "T002", "T003"])
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["subscribed_total"], 5)
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["subscribed_active"], 2)
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["subscribed_top100"], 4)
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["skipped_symbols_due_to_cap"], ["T004"])

    def test_reload_records_error_101_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            self.write_latest(latest, ["AAA", "ERR", "BBB"])
            recorder = LiveDataRecorder(root / "recorder", session_date="2026-05-22")
            ib = FakeIB(fail_symbols={"ERR"})
            runtime_state = {"top100_reload_requested": True, "top100_reload_path": str(latest), "top100_reload_ranking_date": "2026-05-21"}
            contracts: list[tuple[str, FakeContract]] = []
            contract_by_symbol: dict[str, FakeContract] = {}
            tickers = {}

            changed = reload_top100_universe_if_requested(
                ib, recorder, {}, contracts, contract_by_symbol, tickers, {}, {}, runtime_state, self.args_for(latest, top_n=3, cap=100)
            )

            self.assertTrue(changed)
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["ibkr_error_101_count"], 1)
            self.assertNotIn("ERR", tickers)
            self.assertEqual([symbol for symbol, _ in contracts], ["AAA", "BBB"])

    def test_reload_skips_denylisted_symbol_in_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "daily_top100_latest.csv"
            self.write_latest(latest, ["CONL", "AAA", "BBB"])
            denylist = root / "symbol_denylist.csv"
            denylist.write_text(
                "symbol,reason,source,first_seen_at,last_seen_at,notes\n"
                "CONL,kid_priip_ineligible,ibkr_error_201,2026-05-29T00:00:00+00:00,2026-05-29T00:00:00+00:00,\n",
                encoding="utf-8",
            )
            recorder = LiveDataRecorder(root / "recorder", session_date="2026-05-22")
            ib = FakeIB()
            runtime_state = {"top100_reload_requested": True, "top100_reload_path": str(latest), "top100_reload_ranking_date": "2026-05-21"}
            contracts: list[tuple[str, FakeContract]] = []
            contract_by_symbol: dict[str, FakeContract] = {}
            tickers = {}

            changed = reload_top100_universe_if_requested(
                ib, recorder, {}, contracts, contract_by_symbol, tickers, {}, {}, runtime_state, self.args_for(latest, top_n=3, cap=100)
            )

            self.assertTrue(changed)
            self.assertNotIn("CONL", runtime_state["entry_symbols"])
            self.assertIn("CONL", runtime_state["ineligible_symbols"])
            self.assertEqual([symbol for symbol, _ in contracts], ["AAA", "BBB"])
            self.assertEqual(runtime_state["top100_reload_diagnostics"]["excluded_ineligible_count"], 1)


if __name__ == "__main__":
    unittest.main()
