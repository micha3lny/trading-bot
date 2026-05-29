#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.live_trading.ineligible_symbols import DEFAULT_SYMBOL_DENYLIST, add_symbol_denylist


def main() -> int:
    parser = argparse.ArgumentParser(description="Add or update a symbol in the persistent denylist")
    parser.add_argument("symbol")
    parser.add_argument("reason")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--notes", default="")
    parser.add_argument("--symbol-denylist", default=DEFAULT_SYMBOL_DENYLIST)
    args = parser.parse_args()
    row = add_symbol_denylist(
        args.symbol,
        args.reason,
        source=args.source,
        notes=args.notes,
        path=args.symbol_denylist,
    )
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

