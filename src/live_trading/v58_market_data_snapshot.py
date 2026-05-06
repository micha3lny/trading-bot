from __future__ import annotations

import argparse
import csv
import time
from datetime import UTC, datetime
from pathlib import Path

from ib_insync import IB, Stock


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 59
DEFAULT_SYMBOLS = ["QQQ", "SPY", "IWM", "NVDA", "TSLA", "META", "NBIS", "RKLB"]
DEFAULT_OUTPUT = "data/live/market_data_snapshots.csv"


def safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if value != value:  # NaN
            return None
        return value
    except Exception:
        return None


def snapshot_row(symbol: str, ticker) -> dict[str, object]:
    bid = safe_float(ticker.bid)
    ask = safe_float(ticker.ask)
    last = safe_float(ticker.last)
    close = safe_float(ticker.close)
    volume = safe_float(ticker.volume)

    mid = None
    spread = None
    spread_bps = None
    if bid is not None and ask is not None and ask > 0 and bid > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        if mid > 0:
            spread_bps = spread / mid * 10_000.0

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
        "close": close,
        "volume": volume,
        "spread": spread,
        "spread_bps": spread_bps,
        "bid_size": safe_float(ticker.bidSize),
        "ask_size": safe_float(ticker.askSize),
        "last_size": safe_float(ticker.lastSize),
        "market_data_type": ticker.marketDataType,
    }


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "symbol",
        "bid",
        "ask",
        "mid",
        "last",
        "close",
        "volume",
        "spread",
        "spread_bps",
        "bid_size",
        "ask_size",
        "last_size",
        "market_data_type",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="v58 IBKR live market data snapshot logger")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--market-data-type", type=int, default=1, help="1=live, 2=frozen, 3=delayed, 4=delayed frozen")
    args = parser.parse_args()

    print("=== v58 IBKR market data snapshot logger ===")
    print(f"Connecting to {args.host}:{args.port} client_id={args.client_id}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Duration: {args.duration_seconds}s, interval: {args.interval_seconds}s")
    print(f"Output: {args.output}")

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as exc:
        print("Connection failed")
        print(repr(exc))
        return 1

    print("Connected")
    ib.reqMarketDataType(args.market_data_type)

    contracts = []
    tickers = {}
    try:
        for symbol in args.symbols:
            contract = Stock(symbol.upper(), "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                print(f"Could not qualify contract: {symbol}")
                continue
            q = qualified[0]
            contracts.append((symbol.upper(), q))
            tickers[symbol.upper()] = ib.reqMktData(q, "", False, False)
            print(f"Subscribed: {symbol.upper()} conId={q.conId}")

        if not contracts:
            print("No contracts subscribed")
            return 1

        start = time.time()
        output_path = Path(args.output)
        while time.time() - start < args.duration_seconds:
            ib.sleep(args.interval_seconds)
            rows = [snapshot_row(symbol, tickers[symbol]) for symbol, _ in contracts]
            append_rows(output_path, rows)

            printable = []
            for row in rows:
                spread_bps = row["spread_bps"]
                spread_txt = "NA" if spread_bps is None else f"{spread_bps:.1f}bps"
                bid = "NA" if row["bid"] is None else f"{row['bid']:.2f}"
                ask = "NA" if row["ask"] is None else f"{row['ask']:.2f}"
                last = "NA" if row["last"] is None else f"{row['last']:.2f}"
                printable.append(f"{row['symbol']} bid={bid} ask={ask} last={last} spread={spread_txt}")
            print(" | ".join(printable), flush=True)

    finally:
        for ticker in tickers.values():
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        ib.disconnect()
        print("Disconnected")

    print("v58 market data snapshot complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
