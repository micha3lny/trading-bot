from __future__ import annotations

import csv
import ast
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.analysis.should_have_signaled_investigator import (
    EvidenceBundle,
    build_pipeline_symbol_index,
    investigate_case,
    pipeline_has_event,
    pipeline_records,
    summarize_symbol_pipeline,
)
from src.live_trading.symbol_pipeline_telemetry import SYMBOL_PIPELINE_EVENT_TYPES
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore
from src.live_trading.v67_live_top100_expansion_paper_trader import emit_top100_symbol_pipeline_event


class _Store:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_runtime_event(self, **kwargs):
        self.events.append(kwargs)
        return len(self.events)


class _Recorder:
    def __init__(self, root: Path, store: _Store) -> None:
        self.root = root
        self.session_date = "2026-07-27"
        self.sqlite_store = store

    def path(self, name: str, **_kwargs) -> Path:
        path = self.root / self.session_date / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class SymbolPipelineTelemetryTests(unittest.TestCase):
    def test_live_runtime_has_call_sites_for_every_pipeline_event(self) -> None:
        source_path = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        emitted: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "emit_top100_symbol_pipeline_event" or len(node.args) < 3:
                continue
            if isinstance(node.args[2], ast.Constant) and isinstance(node.args[2].value, str):
                emitted.add(node.args[2].value)
        self.assertEqual(set(SYMBOL_PIPELINE_EVENT_TYPES) - emitted, set())

    def test_every_pipeline_event_is_durable_and_symbol_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store()
            recorder = _Recorder(Path(tmp), store)
            runtime_state = {
                "top100_pipeline_run_id": "ranking:2026-07-27:1",
                "runtime_scan_id": 17,
                "top100_entry_metadata_by_symbol": {
                    "NUAI": {"top100_rank": 4, "top100_score": 91.25},
                },
            }

            for event_type in sorted(SYMBOL_PIPELINE_EVENT_TYPES):
                emit_top100_symbol_pipeline_event(
                    recorder,
                    runtime_state,
                    event_type,
                    "NUAI",
                    session_date="2026-07-27",
                    outcome="observed",
                    status="test",
                    reason="fixture",
                )

            lifecycle_path = Path(tmp) / "2026-07-27" / "trade_lifecycle.csv"
            with lifecycle_path.open(newline="", encoding="utf-8") as handle:
                recorder_rows = list(csv.DictReader(handle))

            self.assertEqual({row["event"] for row in recorder_rows}, set(SYMBOL_PIPELINE_EVENT_TYPES))
            self.assertEqual({event["event_type"] for event in store.events}, set(SYMBOL_PIPELINE_EVENT_TYPES))
            self.assertEqual(len(recorder_rows), len(SYMBOL_PIPELINE_EVENT_TYPES))
            self.assertEqual(len(store.events), len(SYMBOL_PIPELINE_EVENT_TYPES))
            for row in recorder_rows:
                payload = json.loads(row["raw_json"])
                self.assertEqual(payload["session_date"], "2026-07-27")
                self.assertEqual(payload["symbol"], "NUAI")
                self.assertEqual(payload["scan_id"], 17)
                self.assertEqual(payload["ranking_generation"], "ranking:2026-07-27:1")
                self.assertEqual(payload["top100_rank"], 4)
                self.assertEqual(payload["top100_score"], 91.25)
                self.assertEqual(payload["reason"], "fixture")
            self.assertTrue(all(event["force_persist"] for event in store.events))

    def test_force_persist_keeps_each_symbol_pipeline_buy_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                for scan_id in (1, 2):
                    store.record_runtime_event(
                        event_time=f"2026-07-27T13:45:0{scan_id}Z",
                        event_type="BUY_BLOCKED",
                        session_date="2026-07-27",
                        symbol="NUAI",
                        reason="max_entries_per_cycle",
                        force_persist=True,
                        raw_json={"scan_id": scan_id},
                    )
                rows = store.query("SELECT event_type, raw_json FROM runtime_events ORDER BY event_time")
            finally:
                store.close()

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["event_type"] for row in rows}, {"BUY_BLOCKED"})
            self.assertEqual([json.loads(row["raw_json"])["scan_id"] for row in rows], [1, 2])

    def test_unchanged_signal_transition_is_suppressed_but_reason_change_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store()
            recorder = _Recorder(Path(tmp), store)
            runtime_state = {
                "top100_pipeline_run_id": "ranking:2026-07-27:1",
                "runtime_scan_id": 1,
                "top100_entry_metadata_by_symbol": {"NUAI": {"top100_rank": 4, "top100_score": 91.25}},
            }
            emit_top100_symbol_pipeline_event(
                recorder, runtime_state, "SIGNAL_REJECTED", "NUAI",
                status="rejected", outcome="rejected", reason="first5_not_ready",
            )
            runtime_state["runtime_scan_id"] = 2
            emit_top100_symbol_pipeline_event(
                recorder, runtime_state, "SIGNAL_REJECTED", "NUAI",
                status="rejected", outcome="rejected", reason="first5_not_ready",
            )
            runtime_state["runtime_scan_id"] = 3
            emit_top100_symbol_pipeline_event(
                recorder, runtime_state, "SIGNAL_REJECTED", "NUAI",
                status="rejected", outcome="rejected", reason="spread_too_wide",
            )

            self.assertEqual(len(store.events), 2)
            self.assertEqual([event["reason"] for event in store.events], ["first5_not_ready", "spread_too_wide"])

    def test_shs_accepts_only_exact_symbol_session_pipeline_events(self) -> None:
        rows = pd.DataFrame([
            {
                "event_time": "2026-07-27T13:45:01Z",
                "session_date": "2026-07-27",
                "event_type": "SIGNAL_EVALUATED",
                "symbol": "NUAI",
                "raw_json": json.dumps({"symbol": "NUAI", "session_date": "2026-07-27", "status": "ready"}),
            },
            {
                "event_time": "2026-07-28T13:45:01Z",
                "session_date": "2026-07-28",
                "event_type": "BUY_DECISION",
                "symbol": "NUAI",
                "raw_json": json.dumps({"symbol": "NUAI", "session_date": "2026-07-28", "status": "order_submitted"}),
            },
            {
                "event_time": "2026-07-27T13:45:02Z",
                "session_date": "2026-07-27",
                "event_type": "BUY_DECISION",
                "symbol": "IREN",
                "raw_json": json.dumps({"symbol": "IREN", "session_date": "2026-07-27", "status": "order_submitted"}),
            },
            {
                "event_time": "2026-07-27T13:45:03Z",
                "session_date": "2026-07-27",
                "event_type": "COLLECTOR_COMPLETE",
                "symbol": "NUAI",
                "raw_json": json.dumps({"symbol": "NUAI", "session_date": "2026-07-27"}),
            },
        ])

        indexed = build_pipeline_symbol_index(
            {"runtime_events": rows},
            {"NUAI", "IREN"},
            "2026-07-27",
        )
        nuai = pipeline_records({"runtime_events": indexed["runtime_events"]["NUAI"]})
        iren = pipeline_records({"runtime_events": indexed["runtime_events"]["IREN"]})

        self.assertEqual(len(nuai), 1)
        self.assertTrue(pipeline_has_event(nuai, "SIGNAL_EVALUATED", "ready"))
        self.assertFalse(pipeline_has_event(nuai, "BUY_DECISION", "order_submitted"))
        self.assertEqual(len(iren), 1)
        self.assertTrue(pipeline_has_event(iren, "BUY_DECISION", "order_submitted"))
        self.assertEqual(summarize_symbol_pipeline(nuai)["pipeline_first_missing_event"], "TOP100_SYMBOL_LOAD")

    def test_shs_does_not_infer_runtime_buy_from_offline_target(self) -> None:
        empty_index = {"runtime_events": {"NUAI": pd.DataFrame()}}
        evidence = EvidenceBundle(
            sqlite_sources={"runtime_events": pd.DataFrame()},
            recorder_sources={"trade_lifecycle": pd.DataFrame()},
            sqlite_by_symbol=empty_index,
            recorder_by_symbol={"trade_lifecycle": {"NUAI": pd.DataFrame()}},
            journal_lines=["2026-07-27T13:45:00Z SIGNAL_READY symbol=NUAI"],
            heartbeat_states=[],
        )
        result = investigate_case(
            target={
                "symbol": "NUAI",
                "possible_signal_time": "2026-07-27T13:45:00Z",
                "was_bought": 1,
            },
            session_date="2026-07-27",
            evidence=evidence,
        )

        self.assertEqual(result["runtime_evidence_found"], 0)
        self.assertEqual(result["signal_ready_seen"], 0)
        self.assertEqual(result["buy_attempt_seen"], 0)
        self.assertEqual(result["final_classification"], "runtime_never_processed_symbol")

    def test_shs_reconstructs_successful_pipeline_from_telemetry_only(self) -> None:
        events = [
            "TOP100_SYMBOL_LOAD",
            "SYMBOL_REGISTERED",
            "CONTRACT_REQUESTED",
            "CONTRACT_RESOLVED",
            "MKT_DATA_REQUESTED",
            "MKT_DATA_SUBSCRIBED",
            "FIRST_TICK_RECEIVED",
            "STATE_CREATED",
            "SIGNAL_EVALUATED",
            "BUY_DECISION",
        ]
        rows = []
        for index, event in enumerate(events):
            status = "ready" if event == "SIGNAL_EVALUATED" else "observed"
            if event == "BUY_DECISION":
                status = "order_submitted"
            payload = {
                "session_date": "2026-07-27",
                "symbol": "NUAI",
                "event_type": event,
                "scan_id": 9,
                "status": status,
                "outcome": status,
            }
            rows.append({
                "event_time": f"2026-07-27T13:45:{index:02d}Z",
                "session_date": "2026-07-27",
                "event_type": event,
                "symbol": "NUAI",
                "raw_json": json.dumps(payload),
            })
        frame = pd.DataFrame(rows)
        index = build_pipeline_symbol_index({"runtime_events": frame}, {"NUAI"}, "2026-07-27")
        evidence = EvidenceBundle(
            sqlite_sources={"runtime_events": frame},
            recorder_sources={"trade_lifecycle": pd.DataFrame()},
            sqlite_by_symbol=index,
            recorder_by_symbol={"trade_lifecycle": {"NUAI": pd.DataFrame()}},
            journal_lines=[],
            heartbeat_states=[],
        )

        result = investigate_case(
            target={"symbol": "NUAI", "possible_signal_time": "2026-07-27T13:45:00Z"},
            session_date="2026-07-27",
            evidence=evidence,
        )

        self.assertEqual(result["signal_ready_seen"], 1)
        self.assertEqual(result["buy_attempt_seen"], 1)
        self.assertEqual(result["pipeline_complete"], 1)
        self.assertEqual(result["pipeline_first_missing_event"], "")
        self.assertEqual(result["final_classification"], "bought_late")


if __name__ == "__main__":
    unittest.main()
