"""Fetch initial daily historical data for a small liquid stock universe.

This script is intentionally simple and safe:
- connects to local IB Gateway / TWS
- fetches historical candles only
- writes local Parquet files under data/market_data
- does not place orders
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from ib_insync import IB, Stock, util


load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

# Current scope: daily candles, 3 years back.
DURATION = os.getenv("HISTORY_DURATION", "3 Y")
BAR_SIZE = os.getenv("HISTORY_BAR_SIZE", "1 day")
WHAT_TO_SHOW = os.getenv("HISTORY_WHAT_TO_SHOW", "TRADES")
USE_RTH = os.getenv("HISTORY_USE_RTH", "true").lower() == "true"
REQUEST_SLEEP_SECONDS = float(os.getenv("IB_REQUEST_SLEEP_SECONDS", "1.0"))


@dataclass(frozen=True)
class StockSpec:
    symbol: str
    primary_exchange: str
    currency: str = "USD"


UNIVERSE = [
    StockSpec("AAPL", "NASDAQ"),
    StockSpec("MSFT", "NASDAQ"),
    StockSpec("NVDA", "NASDAQ"),
    StockSpec("AMD", "NASDAQ"),
    StockSpec("META", "NASDAQ"),
    StockSpec("GOOGL", "NASDAQ"),
    StockSpec("AMZN", "NASDAQ"),
    StockSpec("TSLA", "NASDAQ"),
    StockSpec("NFLX", "NASDAQ"),
    StockSpec("INTC", "NASDAQ"),
    StockSpec("AVGO", "NASDAQ"),
    StockSpec("ADBE", "NASDAQ"),
    StockSpec("CRM", "NYSE"),
    StockSpec("CSCO", "NASDAQ"),
    StockSpec("QCOM", "NASDAQ"),
    StockSpec("TXN", "NASDAQ"),
    StockSpec("MU", "NASDAQ"),
    StockSpec("AMAT", "NASDAQ"),
    StockSpec("PYPL", "NASDAQ"),
    StockSpec("SHOP", "NYSE"),
    StockSpec("PLTR", "NASDAQ"),
    StockSpec("SNOW", "NYSE"),
    StockSpec("UBER", "NYSE"),
    StockSpec("LYFT", "NASDAQ"),
    StockSpec("COIN", "NASDAQ"),
    StockSpec("XYZ", "NYSE"),  # Block Inc. formerly SQ
    StockSpec("ROKU", "NASDAQ"),
    StockSpec("ZM", "NASDAQ"),
    StockSpec("DOCU", "NASDAQ"),
    StockSpec("PINS", "NYSE"),
]


def make_contract(spec: StockSpec) -> Stock:
    return Stock(
        symbol=spec.symbol,
        exchange="SMART",
        currency=spec.currency,
        primaryExchange=spec.primary_exchange,
    )


def main() -> int:
    ib = IB()
    print("Connecting to IBKR...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    os.makedirs("data/market_data", exist_ok=True)

    saved: list[str] = []
    skipped: list[str] = []

    try:
        for spec in UNIVERSE:
            print(f"Fetching {spec.symbol} ({BAR_SIZE}, {DURATION})...")

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
                    durationStr=DURATION,
                    barSizeSetting=BAR_SIZE,
                    whatToShow=WHAT_TO_SHOW,
                    useRTH=USE_RTH,
                    formatDate=1,
                )

                df = util.df(bars)
                if df is None or df.empty:
                    print(f"  SKIP {spec.symbol}: no data returned")
                    skipped.append(spec.symbol)
                    continue

                file_path = f"data/market_data/{spec.symbol}_1D.parquet"
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
