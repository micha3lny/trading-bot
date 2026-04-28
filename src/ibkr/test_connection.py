"""Minimal, safe IBKR connection test.

This script connects to local IB Gateway / TWS, reads basic account and
server-time information, then disconnects. It does not place orders.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import suppress

from dotenv import load_dotenv
from ib_insync import IB


load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
TIMEOUT_SECONDS = int(os.getenv("IB_TIMEOUT_SECONDS", "10"))

# Keep ib_insync / ibapi noise quiet during this basic check.
logging.basicConfig(level=logging.WARNING)
for logger_name in ("ib_insync", "ibapi", "asyncio"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def mask_account(account: str) -> str:
    """Avoid printing full account identifiers in terminal output."""
    if len(account) <= 4:
        return "****"
    return f"{account[:3]}***{account[-3:]}"


def main() -> int:
    ib = IB()

    print("IBKR connection test")
    print("--------------------")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Client ID: {CLIENT_ID}")
    print()

    try:
        print("Connecting...")
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=TIMEOUT_SECONDS)

        if not ib.isConnected():
            print("❌ Connection failed: IBKR did not report an active connection.")
            return 1

        server_time = ib.reqCurrentTime()
        accounts = ib.managedAccounts()
        masked_accounts = [mask_account(account) for account in accounts]

        print("✅ Connected")
        print(f"Server time: {server_time}")
        print(f"Accounts: {masked_accounts if masked_accounts else 'none returned'}")
        print()
        print("No orders were sent. This was a read-only connection check from our code.")
        return 0

    except ConnectionRefusedError:
        print("❌ Connection refused.")
        print("Check that IB Gateway/TWS is running, logged into Paper Trading, and API port is correct.")
        return 1
    except TimeoutError:
        print("❌ Connection timed out.")
        print("Check IB Gateway/TWS, API settings, localhost permission, and port.")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic script
        print(f"❌ IBKR connection test failed: {exc}")
        return 1
    finally:
        if ib.isConnected():
            with suppress(Exception):
                ib.disconnect()
            print("Disconnected")


if __name__ == "__main__":
    sys.exit(main())
