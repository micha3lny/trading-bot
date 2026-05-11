from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd


DEFAULT_RECORDER_DIR = "data/live/recorder"


def latest_session_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(base)
    dirs = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No recorder session dirs found in {base}")
    return dirs[0]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return ""


def print_table(title: str, df: pd.DataFrame, cols: list[str], limit: int = 10) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("empty")
        return
    show = df.tail(limit).copy()
    existing = [c for c in cols if c in show.columns]
    if not existing:
        print(show.tail(limit).to_string(index=False))
        return
    print(show[existing].to_string(index=False))


def render(session: Path, *, clear_screen: bool, table_limit: int, render_count: int) -> None:
    if clear_screen:
        os.system("clear")
    else:
        print("\n" + "=" * 120)

    print("=== v63 live portfolio monitor ===")
    print(f"Session: {session}")
    print(f"Refresh count: {render_count}")
    print(f"Mode: {'clear-screen' if clear_screen else 'append/no-clear'}")

    portfolio = read_csv_safe(session / "portfolio_snapshots.csv")
    fills = read_csv_safe(session / "fills.csv")
    lifecycle = read_csv_safe(session / "trade_lifecycle.csv")

    print("\n=== Portfolio latest ===")
    if portfolio.empty:
        print("empty")
    else:
        last = portfolio.tail(1).iloc[0]
        for field in [
            "account",
            "cash",
            "net_liquidation",
            "buying_power",
            "gross_exposure",
            "open_positions",
            "daily_pnl",
            "unrealized_pnl",
            "realized_pnl",
            "recorded_at",
        ]:
            print(f"{field}: {last.get(field, '')}")

    print_table(
        "Recent fills",
        fills,
        ["recorded_at", "symbol", "action", "quantity", "fill_price", "commission"],
        limit=table_limit,
    )

    print_table(
        "Recent lifecycle",
        lifecycle,
        ["recorded_at", "event", "symbol", "action", "quantity", "price", "pnl_pct", "reason"],
        limit=table_limit,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="v63 live portfolio/trade monitor")
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--session-dir", default="")
    parser.add_argument("--refresh", type=float, default=20.0)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--table-limit", type=int, default=8)
    args = parser.parse_args()

    session = Path(args.session_dir) if args.session_dir else latest_session_dir(args.recorder_dir)

    render_count = 0
    while True:
        render_count += 1
        render(
            session,
            clear_screen=args.clear,
            table_limit=args.table_limit,
            render_count=render_count,
        )
        time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
