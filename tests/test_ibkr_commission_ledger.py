from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_trading.analytics.v67_daily_report import reconstruct_closed_trades_from_fills
from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v66_ibkr_account_recorder import (
    install_commission_report_handler,
    record_commission_report,
    record_recent_fills,
)


def fake_fill(
    exec_id: str,
    *,
    symbol: str = "RKLB",
    side: str = "BOT",
    shares: float = 10,
    price: float = 10.0,
    commission: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol),
        execution=SimpleNamespace(
            execId=exec_id,
            side=side,
            shares=shares,
            price=price,
            orderId=101,
            permId=202,
            exchange="SMART",
            lastLiquidity=1,
            time="2026-05-22T13:30:00+00:00",
        ),
        commissionReport=SimpleNamespace(
            execId=exec_id,
            commission=commission,
            currency="USD",
            realizedPNL=0.0,
        ),
    )


class FakeIB:
    def __init__(self, fills: list[SimpleNamespace]) -> None:
        self._fills = fills

    def reqExecutions(self, _filter):
        return list(self._fills)


class DualSourceIB:
    def __init__(self, req_fills: list[SimpleNamespace], fills: list[SimpleNamespace]) -> None:
        self._req_fills = req_fills
        self._fills = fills

    def reqExecutions(self, _filter):
        return list(self._req_fills)

    def fills(self):
        return list(self._fills)


class FakeEvent:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def emit(self, *args):
        for handler in list(self.handlers):
            handler(*args)


class FakeEventIB:
    def __init__(self) -> None:
        self.commissionReportEvent = FakeEvent()


class CountingStore:
    def __init__(self) -> None:
        self.upsert_count = 0
        self.rows = []
        self.finalize_count = 0
        self.pending_counts = {}
        self.status_calls = []

    def mark_operation_status(self, *args, **kwargs):
        self.status_calls.append((args, kwargs))
        return None

    def upsert_execution(self, _row):
        self.upsert_count += 1
        self.rows.append(dict(_row))

    def runtime_pending_counts(self):
        return dict(self.pending_counts)

    def finalize_pending_trades(self):
        self.finalize_count += 1
        self.pending_counts = {}
        return {"pending_before": 1, "pending_after": 0, "resolved": 1}


class InterruptingPendingCountStore(CountingStore):
    def runtime_pending_counts(self):
        raise KeyboardInterrupt


def read_fills(recorder: LiveDataRecorder) -> list[dict]:
    with recorder.path("fills.csv").open(errors="replace") as fh:
        return list(csv.DictReader(fh))


class IbkrCommissionLedgerTests(unittest.TestCase):
    def test_execution_recorded_once_duplicate_exec_id_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            fill = fake_fill("E1", commission=0.35)

            count = record_recent_fills(FakeIB([fill, fill]), recorder, seen=set())

            rows = read_fills(recorder)
            self.assertEqual(count, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["execution_id"], "E1")
            self.assertEqual(rows[0]["commission_source"], "ibkr")

    def test_commission_report_updates_existing_missing_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            record_recent_fills(FakeIB([fake_fill("E1", commission=None)]), recorder, seen=set())

            status = record_commission_report(
                recorder,
                SimpleNamespace(execId="E1", commission=0.42, currency="USD", realizedPNL=1.23),
            )

            rows = read_fills(recorder)
            self.assertEqual(status, "matched")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["execution_id"], "E1")
            self.assertEqual(rows[0]["commission"], "0.42")
            self.assertEqual(rows[0]["commission_source"], "ibkr")
            self.assertEqual(rows[0]["realized_pnl"], "1.23")

    def test_record_recent_fills_finalizes_stale_pending_trades_after_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            store = CountingStore()
            store.pending_counts = {"pending_trade_finalization_count": 1}
            recorder.sqlite_store = store

            count = record_recent_fills(FakeIB([fake_fill("E1", commission=0.35)]), recorder, seen=set())

            self.assertEqual(count, 1)
            self.assertEqual(store.finalize_count, 1)
            self.assertTrue(store.status_calls)
            self.assertEqual(store.status_calls[-1][1]["pending_counts"], {})
            self.assertEqual(store.status_calls[-1][1]["finalized_pending_trades"]["resolved"], 1)

    def test_record_recent_fills_marks_interrupted_on_keyboard_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            store = InterruptingPendingCountStore()
            recorder.sqlite_store = store

            with self.assertRaises(KeyboardInterrupt):
                record_recent_fills(FakeIB([fake_fill("E1", commission=0.35)]), recorder, seen=set())

            statuses = [call[0][1] for call in store.status_calls if len(call[0]) >= 2]
            self.assertIn("running", statuses)
            self.assertIn("interrupted", statuses)

    def test_duplicate_commission_report_is_ignored_after_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            record_recent_fills(FakeIB([fake_fill("E1", commission=None)]), recorder, seen=set())
            report = SimpleNamespace(execId="E1", commission=0.42, currency="USD", realizedPNL=1.23)

            self.assertEqual(record_commission_report(recorder, report), "matched")
            self.assertEqual(record_commission_report(recorder, report), "duplicate")

            rows = read_fills(recorder)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["commission"], "0.42")
            self.assertEqual(rows[0]["commission_source"], "ibkr")

    def test_replayed_complete_fill_is_skipped_before_sqlite_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            store = CountingStore()
            setattr(recorder, "sqlite_store", store)
            fill = fake_fill("E1", commission=0.42)
            seen: set[str] = set()

            self.assertEqual(record_recent_fills(FakeIB([fill]), recorder, seen=seen), 1)
            for _ in range(5):
                self.assertEqual(record_recent_fills(FakeIB([fill]), recorder, seen=seen), 0)

            self.assertEqual(store.upsert_count, 1)
            self.assertEqual(len(read_fills(recorder)), 1)

    def test_record_recent_fills_prefers_ib_fills_commission_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            req_fill = fake_fill("E1", commission=None)
            commissioned_fill = fake_fill("E1", commission=0.42)

            count = record_recent_fills(DualSourceIB([req_fill], [commissioned_fill]), recorder, seen=set())

            rows = read_fills(recorder)
            self.assertEqual(count, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["execution_id"], "E1")
            self.assertEqual(rows[0]["commission"], "0.42")
            self.assertEqual(rows[0]["commission_source"], "ibkr")

    def test_startup_seen_complete_fill_repairs_sqlite_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            fill = fake_fill("E1", commission=0.42)
            record_recent_fills(FakeIB([fill]), recorder, seen=set())
            store = CountingStore()
            setattr(recorder, "sqlite_store", store)
            seen = {"E1"}

            self.assertEqual(record_recent_fills(FakeIB([fill]), recorder, seen=seen), 0)
            self.assertEqual(record_recent_fills(FakeIB([fill]), recorder, seen=seen), 0)

            self.assertEqual(store.upsert_count, 1)
            self.assertEqual(len(read_fills(recorder)), 1)

    def test_commission_report_event_handler_accepts_ib_insync_event_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            record_recent_fills(FakeIB([fake_fill("E1", commission=None)]), recorder, seen=set())
            ib = FakeEventIB()

            install_commission_report_handler(ib, recorder)
            install_commission_report_handler(ib, recorder)
            ib.commissionReportEvent.emit(
                object(),
                object(),
                SimpleNamespace(execId="E1", commission=0.55, currency="USD", realizedPNL=2.0),
            )

            rows = read_fills(recorder)
            self.assertEqual(len(ib.commissionReportEvent.handlers), 1)
            self.assertEqual(rows[0]["commission"], "0.55")
            self.assertEqual(rows[0]["commission_source"], "ibkr")

    def test_execution_after_commission_placeholder_syncs_canonical_sqlite_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            store = CountingStore()
            setattr(recorder, "sqlite_store", store)

            self.assertEqual(
                record_commission_report(
                    recorder,
                    SimpleNamespace(execId="E_PARTIAL", commission=0.42, currency="USD", realizedPNL=0.0),
                ),
                "placeholder",
            )
            count = record_recent_fills(
                FakeIB([fake_fill("E_PARTIAL", symbol="FTCI", shares=98, price=2.34, commission=None)]),
                recorder,
                seen=set(),
            )

            rows = read_fills(recorder)
            self.assertEqual(count, 1)
            self.assertEqual(rows[0]["symbol"], "FTCI")
            self.assertAlmostEqual(float(rows[0]["quantity"]), 98.0)
            self.assertEqual(rows[0]["fill_price"], "2.34")
            self.assertEqual(rows[0]["commission"], "0.42")
            self.assertEqual(rows[0]["commission_source"], "ibkr")
            self.assertGreaterEqual(store.upsert_count, 2)
            self.assertEqual(store.rows[-1]["execution_id"], "E_PARTIAL")
            self.assertEqual(store.rows[-1]["symbol"], "FTCI")
            self.assertAlmostEqual(float(store.rows[-1]["quantity"]), 98.0)
            self.assertEqual(store.rows[-1]["fill_price"], "2.34")
            self.assertEqual(store.rows[-1]["commission"], "0.42")
            self.assertEqual(store.rows[-1]["commission_source"], "ibkr")

    def test_daily_report_fill_ledger_uses_actual_commission(self) -> None:
        rows = [
            {
                "execution_id": "B1",
                "symbol": "RKLB",
                "action": "BOT",
                "quantity": "10",
                "fill_price": "10",
                "commission": "0.35",
                "commission_source": "ibkr",
                "recorded_at": "2026-05-22T13:30:00+00:00",
            },
            {
                "execution_id": "S1",
                "symbol": "RKLB",
                "action": "SLD",
                "quantity": "10",
                "fill_price": "11",
                "commission": "0.35",
                "commission_source": "ibkr",
                "recorded_at": "2026-05-22T13:35:00+00:00",
            },
        ]

        closed = reconstruct_closed_trades_from_fills(rows, commission_per_roundtrip=1.0)

        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0]["gross"], 10.0)
        self.assertAlmostEqual(closed[0]["actual_commission"], 0.70)
        self.assertAlmostEqual(closed[0]["estimated_commission_fallback"], 0.0)
        self.assertAlmostEqual(closed[0]["net_actual"], 9.30)
        self.assertAlmostEqual(closed[0]["net_estimated"], 9.30)
        self.assertEqual(closed[0]["commission_source"], "ibkr")

    def test_daily_report_fill_ledger_falls_back_when_commission_missing(self) -> None:
        rows = [
            {
                "execution_id": "B1",
                "symbol": "RKLB",
                "action": "BOT",
                "quantity": "10",
                "fill_price": "10",
                "commission": "",
                "commission_source": "missing",
                "recorded_at": "2026-05-22T13:30:00+00:00",
            },
            {
                "execution_id": "S1",
                "symbol": "RKLB",
                "action": "SLD",
                "quantity": "10",
                "fill_price": "11",
                "commission": "",
                "commission_source": "missing",
                "recorded_at": "2026-05-22T13:35:00+00:00",
            },
        ]

        closed = reconstruct_closed_trades_from_fills(rows, commission_per_roundtrip=1.0)

        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0]["gross"], 10.0)
        self.assertAlmostEqual(closed[0]["actual_commission"], 0.0)
        self.assertAlmostEqual(closed[0]["estimated_commission_fallback"], 1.0)
        self.assertAlmostEqual(closed[0]["net_actual"], 10.0)
        self.assertAlmostEqual(closed[0]["net_estimated"], 9.0)
        self.assertEqual(closed[0]["commission_source"], "estimated")


if __name__ == "__main__":
    unittest.main()
