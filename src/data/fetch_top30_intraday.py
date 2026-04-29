"""Fetch intraday historical data for the top 30 stock universe.

This script is intentionally simple and safe:
- connects to local IB Gateway / TWS
- fetches historical intraday candles only
- writes local Parquet files under data/market_data_intraday
- does not place orders
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from ib_insync import IB, util

from src.data.fetch_top30 import UNIVERSE, make_contract


load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))

# Use a different client id than the daily downloader.
CLIENT_ID = int(os.getenv("IB_INTRADAY_CLIENT_ID", "2"))

# Conservative defaults for IBKR pacing limits.
# 15-minute bars over 90 days give a better first validation sample than 30 days.
INTRADAY_DURATION = os.getenv("INTRADAY_DURATION", "90 D")
INTRADAY_BAR_SIZE = os.getenv("INTRADAY_BAR_SIZE", "15 mins")
WHAT_TO_SHOW = os.getenv("INTRADAY_WHAT_TO_SHOW", "TRADES")
USE_RTH = os.getenv("INTRADAY_USE_RTH", "true").lower() == "true"
REQUEST_SLEEP_SECONDS = float(os.getenv("IB_REQUEST_SLEEP_SECONDS", "1.5"))

OUTPUT_DIR = os.getenv("INTRADAY_OUTPUT_DIR", "data/market_data_intraday")


def bar_size_to_filename_part(bar_size: str) -> str:
    """Convert IBKR bar size text to a compact filename suffix."""
    return (
        bar_size.lower()
        .replace(" ", "")
        .replace("mins", "m")
        .replace("min", "m")
        .replace("hours", "h")
        .replace("hour", "h")
        .replace("days", "d")
        .replace("day", "d")
    )


def main() -> int:
    ib = IB()

    print("Connecting to IBKR...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved: list[str] = []
    skipped: list[str] = []
    interval_name = bar_size_to_filename_part(INTRADAY_BAR_SIZE)

    try:
        for spec in UNIVERSE:
            print(f"Fetching {spec.symbol} ({INTRADAY_BAR_SIZE}, {INTRADAY_DURATION})...")

            try:
                contract = make_contract(spec)
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    print(f"  SKIP {spec.symbol}: contract could not be qualified")
                    skipped.append(spec.symbol)
                    continue

                bars = ib.reqHistoricalData(
                    qualified[0],
                    endDateTime="",
                    durationStr=INTRADAY_DURATION,
                    barSizeSetting=INTRADAY_BAR_SIZE,
                    whatToShow=WHAT_TO_SHOW,
                    useRTH=USE_RTH,
                    formatDate=1,
                )

                df = util.df(bars)
                if df is None or df.empty:
                    print(f"  SKIP {spec.symbol}: no data returned")
                    skipped.append(spec.symbol)
                    continue

                file_path = f"{OUTPUT_DIR}/{spec.symbol}_{interval_name}.parquet"
                df.to_parquet(file_path, index=False)

                saved.append(spec.symbol)
                print(f"  Saved {spec.symbol}: {len(df)} rows -> {file_path}")

            except Exception as exc:  # noqa: BLE001 - CLI script should continue fetching others
                print(f"  ERROR {spec.symbol}: {exc}")
                skipped.append(spec.symbol)

            time.sleep(REQUEST_SLEEP_SECONDS)

    finally:
        if ib.isConnected():
            ib.disconnect()

    print()
    print("Summary")
    print("-------")
    print(f"Saved: {len(saved)} symbols")
    print(f"Skipped/errors: {len(skipped)} symbols")
    if skipped:
        print("Skipped:", ", ".join(skipped))

    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
