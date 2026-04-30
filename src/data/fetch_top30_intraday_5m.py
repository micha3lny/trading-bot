"""Fetch 5-minute intraday historical data for the trading universe.

This is a dedicated wrapper around the existing IBKR intraday downloader.
It avoids requiring manual environment variables in the shell.

Output files are written to:
    data/market_data_intraday/{SYMBOL}_5m.parquet

No orders are placed.
"""

from __future__ import annotations

import os


# Configure before importing the shared downloader module, because it reads env
# values at import time.
os.environ["INTRADAY_BAR_SIZE"] = "5 mins"
os.environ.setdefault("INTRADAY_DURATION", "90 D")
os.environ.setdefault("INTRADAY_OUTPUT_DIR", "data/market_data_intraday")
os.environ.setdefault("INTRADAY_USE_RTH", "true")

from src.data.fetch_top30_intraday import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
