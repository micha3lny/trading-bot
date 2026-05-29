#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.live_trading.ineligible_symbols import DEFAULT_SYMBOL_DENYLIST, remove_symbol_denylist


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove a symbol from the persistent denylist")
    parser.add_argument("symbol")
    parser.add_argument("--symbol-denylist", default=DEFAULT_SYMBOL_DENYLIST)
    args = parser.parse_args()
    removed = remove_symbol_denylist(args.symbol, path=args.symbol_denylist)
    print(f"removed={1 if removed else 0} symbol={args.symbol.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

