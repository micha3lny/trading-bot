from __future__ import annotations

import argparse
from datetime import datetime

from ib_insync import IB


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 58


def main() -> int:
    parser = argparse.ArgumentParser(description="v58 IBKR paper connection check")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    print("=== v58 IBKR paper connection check ===")
    print(f"Connecting to {args.host}:{args.port} client_id={args.client_id}")

    ib = IB()

    try:
        ib.connect(
            args.host,
            args.port,
            clientId=args.client_id,
            timeout=args.timeout,
        )
    except Exception as exc:
        print("Connection failed")
        print(repr(exc))
        return 1

    print("\nConnected successfully")

    try:
        server_time = ib.reqCurrentTime()
        print(f"IBKR server time: {server_time}")
    except Exception as exc:
        print(f"Could not fetch server time: {exc}")

    print("\n=== Managed accounts ===")
    try:
        accounts = ib.managedAccounts()
        for account in accounts:
            print(f"- {account}")
    except Exception as exc:
        print(f"Could not fetch accounts: {exc}")

    print("\n=== Account summary ===")
    try:
        summary = ib.accountSummary()
        important_tags = {
            "NetLiquidation",
            "TotalCashValue",
            "BuyingPower",
            "ExcessLiquidity",
            "AvailableFunds",
        }

        for row in summary:
            if row.tag in important_tags:
                print(f"{row.tag}: {row.value} {row.currency}")
    except Exception as exc:
        print(f"Could not fetch account summary: {exc}")

    print("\n=== Open positions ===")
    try:
        positions = ib.positions()
        if not positions:
            print("No open positions")
        else:
            for pos in positions:
                print(
                    f"{pos.contract.symbol} qty={pos.position} avgCost={pos.avgCost}"
                )
    except Exception as exc:
        print(f"Could not fetch positions: {exc}")

    print("\n=== Connection health ===")
    print(f"Connected: {ib.isConnected()}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")

    ib.disconnect()

    print("\nDisconnected cleanly")
    print("v58 connection check complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
