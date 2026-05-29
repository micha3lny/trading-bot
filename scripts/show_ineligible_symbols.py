#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys

from src.live_trading.ineligible_symbols import (
    DEFAULT_RUNTIME_INELIGIBLE,
    DEFAULT_SYMBOL_DENYLIST,
    combined_ineligible_symbols,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show persistent and runtime ineligible symbols")
    parser.add_argument("--symbol-denylist", default=DEFAULT_SYMBOL_DENYLIST)
    parser.add_argument("--runtime-ineligible-path", default=DEFAULT_RUNTIME_INELIGIBLE)
    args = parser.parse_args()

    rows = combined_ineligible_symbols(args.symbol_denylist, args.runtime_ineligible_path)
    fieldnames = ["symbol", "reason", "source", "ibkr_error_code", "conId", "first_seen_at", "last_seen_at", "notes"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for symbol in sorted(rows):
        row = rows[symbol]
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

