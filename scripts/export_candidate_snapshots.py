#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_trading.candidate_snapshot_telemetry import export_snapshot_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one session of Top100 candidate Parquet snapshots to CSV.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--recorder-root", default="data/live/recorder")
    parser.add_argument("--kind", choices=("light", "full", "both"), default="both")
    parser.add_argument("--output-dir", default="data/analysis")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    kinds = ("light", "full") if args.kind == "both" else (args.kind,)
    results = []
    for kind in kinds:
        output = Path(args.output_dir) / f"top100_buy_candidate_snapshots_{kind}_{args.date}.csv"
        results.append(export_snapshot_csv(args.recorder_root, args.date, kind, output))
    print(f"CANDIDATE_SNAPSHOT_EXPORT_DONE {json.dumps(results, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
