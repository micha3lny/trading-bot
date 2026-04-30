"""Fetch 1-minute intraday data in smaller IBKR chunks.

Why this exists:
- IBKR often times out on one large 90D / 1m request.
- This script fetches smaller chunks and merges them into one parquet per symbol.

Output:
    data/market_data_intraday/{SYMBOL}_1m.parquet

No orders are placed.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from ib_insync import IB, Stock, util
import pandas as pd

from src.data.fetch_top30 import UNIVERSE, StockSpec


load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

OUTPUT_DIR = os.getenv("INTRADAY_OUTPUT_DIR", "data/market_data_intraday")
WHAT_TO_SHOW = os.getenv("INTRADAY_WHAT_TO_SHOW", "TRADES")
USE_RTH = os.getenv("INTRADAY_USE_RTH", "true").lower() == "true"
REQUEST_SLEEP_SECONDS = float(os.getenv("IB_REQUEST_SLEEP_SECONDS", "1.5"))
RETRY_SLEEP_SECONDS = float(os.getenv("IB_RETRY_SLEEP_SECONDS", "5.0"))
MAX_RETRIES = int(os.getenv("IB_MAX_RETRIES", "3"))

BAR_SIZE = "1 min"
CHUNK_DURATION = os.getenv("INTRADAY_1M_CHUNK_DURATION", "10 D")
TOTAL_DAYS = int(os.getenv("INTRADAY_1M_TOTAL_DAYS", "90"))
CHUNK_DAYS = int(os.getenv("INTRADAY_1M_CHUNK_DAYS", "10"))


def make_contract(spec: StockSpec) -> Stock:
    return Stock(
        symbol=spec.symbol,
        exchange="SMART",
        currency=spec.currency,
        primaryExchange=spec.primary_exchange,
    )


def request_chunk(ib: IB, contract, end_dt: datetime):
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=CHUNK_DURATION,
                barSizeSetting=BAR_SIZE,
                whatToShow=WHAT_TO_SHOW,
                useRTH=USE_RTH,
                formatDate=1,
                timeout=60,
            )
            df = util.df(bars)
            if df is not None and not df.empty:
                return df
            print(f"    Empty chunk ending {end_str} (attempt {attempt}/{MAX_RETRIES})")
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR chunk ending {end_str} attempt {attempt}/{MAX_RETRIES}: {exc}")

        time.sleep(RETRY_SLEEP_SECONDS)

    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_symbol(ib: IB, spec: StockSpec) -> pd.DataFrame | None:
    contract = make_contract(spec)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        print(f"  SKIP {spec.symbol}: contract could not be qualified")
        return None

    contract = qualified[0]
    chunks = []
    now = datetime.now(timezone.utc)

    # Fetch backwards: now, now-10d, now-20d, ...
    for offset in range(0, TOTAL_DAYS, CHUNK_DAYS):
        end_dt = now - timedelta(days=offset)
        print(f"  Chunk ending {end_dt.strftime('%Y-%m-%d')} ({CHUNK_DURATION})")
        df = request_chunk(ib, contract, end_dt)
        if df is not None and not df.empty:
            chunks.append(df)
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not chunks:
        return None

    merged = normalize_df(pd.concat(chunks, ignore_index=True))
    cutoff = pd.Timestamp(now - timedelta(days=TOTAL_DAYS)).tz_localize(None)
    if merged["date"].dt.tz is not None:
        cutoff = pd.Timestamp(now - timedelta(days=TOTAL_DAYS))
    merged = merged[merged["date"] >= cutoff].reset_index(drop=True)
    return merged


def main() -> int:
    ib = IB()
    print("Connecting to IBKR...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved: list[str] = []
    skipped: list[str] = []

    try:
        for spec in UNIVERSE:
            output_path = os.path.join(OUTPUT_DIR, f"{spec.symbol}_1m.parquet")
            print(f"Fetching {spec.symbol} ({BAR_SIZE}, chunked {TOTAL_DAYS}D)...")

            try:
                df = fetch_symbol(ib, spec)
                if df is None or df.empty:
                    print(f"  SKIP {spec.symbol}: no data returned")
                    skipped.append(spec.symbol)
                    continue

                df.to_parquet(output_path, index=False)
                saved.append(spec.symbol)
                print(f"  Saved {spec.symbol}: {len(df)} rows -> {output_path}")

            except Exception as exc:  # noqa: BLE001
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
