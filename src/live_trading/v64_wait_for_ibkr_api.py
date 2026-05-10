from __future__ import annotations

import argparse
import socket
import sys
import time
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def can_connect(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait until IBKR API TCP port is reachable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_seconds
    print(f"[{now()}] Waiting for IBKR API at {args.host}:{args.port}", flush=True)

    while time.monotonic() < deadline:
        if can_connect(args.host, args.port, args.connect_timeout):
            print(f"[{now()}] IBKR API is reachable at {args.host}:{args.port}", flush=True)
            return 0
        print(f"[{now()}] IBKR API not ready yet; retrying in {args.interval_seconds}s", flush=True)
        time.sleep(args.interval_seconds)

    print(f"[{now()}] ERROR: timed out waiting for IBKR API at {args.host}:{args.port}", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
