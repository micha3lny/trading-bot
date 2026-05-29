#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.runtime_queries import DateWindow, load_dashboard_snapshot
from src.live_trading.storage.sqlite_store import resolve_sqlite_path


def metric_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = snapshot.get("summary") or {}
    diagnostics = snapshot.get("diagnostics") or {}
    data_quality = snapshot.get("data_quality_summary") or {}
    return {
        "closed_trades": int(summary.get("closed_trades") or 0),
        "open_trades": int(summary.get("open_trades") or 0),
        "gross_pnl": round(float(summary.get("gross_pnl") or 0.0), 6),
        "commissions": round(float(summary.get("commissions") or 0.0), 6),
        "net_actual_pnl": round(float(summary.get("net_actual_pnl") or 0.0), 6),
        "commission_ok": int(data_quality.get("commission_ok") or 0),
        "commission_partial": int(data_quality.get("commission_partial") or 0),
        "commission_missing": int(data_quality.get("commission_missing") or 0),
        "trades_count": int(diagnostics.get("trades_count") or 0),
        "reconstructed_trades_count": int(diagnostics.get("reconstructed_trades_count") or 0),
        "displayed_closed_trades_count": int(diagnostics.get("displayed_closed_trades_count") or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repeated dashboard snapshots for stability.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--date", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--strategy", default="All")
    parser.add_argument("--include-reconstructed", action="store_true")
    args = parser.parse_args()

    sqlite_path = resolve_sqlite_path(args.sqlite_path)
    snapshots: list[dict[str, Any]] = []
    for idx in range(max(1, args.samples)):
        snapshot = load_dashboard_snapshot(
            sqlite_path,
            DateWindow(args.date, args.date),
            args.strategy,
            include_reconstructed=args.include_reconstructed,
        )
        metrics = metric_snapshot(snapshot)
        snapshots.append(metrics)
        print(json.dumps({"sample": idx + 1, **metrics}, sort_keys=True), flush=True)
        if idx + 1 < args.samples and args.sleep > 0:
            time.sleep(args.sleep)

    baseline = snapshots[0]
    unstable = [sample for sample in snapshots[1:] if sample != baseline]
    if unstable:
        print(json.dumps({"ok": False, "baseline": baseline, "unstable_samples": unstable}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "samples": len(snapshots), "metrics": baseline}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
