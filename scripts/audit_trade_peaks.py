#!/usr/bin/env python3
from __future__ import annotations

from rebuild_trade_peaks import build_parser, run


def main() -> int:
    parser = build_parser("Audit canonical trade peak/giveback metrics from 1m candles.")
    parser.set_defaults(apply=False)
    args = parser.parse_args()
    if getattr(args, "apply", False):
        parser.error("audit_trade_peaks.py is read-only; use rebuild_trade_peaks.py --apply on a copied DB")
    return run(args, audit=True)


if __name__ == "__main__":
    raise SystemExit(main())
