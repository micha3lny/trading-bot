from __future__ import annotations

import argparse
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


def pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
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


def render(session: Path) -> None:
    os.system("clear")
    print("=== v63 live portfolio monitor ===")
    print(f"Session: {session}")
    print(f"Refresh: live recorder files")

    portfolio = read_csv_safe(session / "portfolio_snapshots.csv")
    fills = read_csv_safe(session / "fills.csv")
    intents = read_csv_safe(session / "order_intents.csv")
    signals = read_csv_safe(session / "signal_snapshots.csv")
    selections = read_csv_safe(session / "selection_events.csv")
    errors = read_csv_safe(session / "error_events.csv")

    print("\n=== Portfolio latest ===")
    if portfolio.empty:
        print("empty")
    else:
        last = portfolio.tail(1).iloc[0]
        print(f"Account:          {last.get('account', '')}")
        print(f"Cash:             {money(last.get('cash'))}")
        print(f"Net liquidation:  {money(last.get('net_liquidation'))}")
        print(f"Buying power:     {money(last.get('buying_power'))}")
        print(f"Gross exposure:   {money(last.get('gross_exposure'))}")
        print(f"Open positions:   {last.get('open_positions', '')}")
        print(f"Daily PnL:        {money(last.get('daily_pnl'))}")
        print(f"Unrealized PnL:   {money(last.get('unrealized_pnl'))}")
        print(f"Realized PnL:     {money(last.get('realized_pnl'))}")
        print(f"Recorded at:      {last.get('recorded_at', '')}")

    if not fills.empty:
        fills_num = fills.copy()
        for c in ["quantity", "fill_price", "commission", "realized_pnl", "slippage_bps"]:
            if c in fills_num.columns:
                fills_num[c] = pd.to_numeric(fills_num[c], errors="coerce")
        print("\n=== Today fills summary ===")
        print(f"Fills:            {len(fills_num)}")
        if "realized_pnl" in fills_num.columns:
            print(f"Realized PnL:     {money(fills_num['realized_pnl'].sum())}")
        if "commission" in fills_num.columns:
            print(f"Commission:       {money(fills_num['commission'].sum())}")
        if "slippage_bps" in fills_num.columns:
            print(f"Avg slippage bps: {fills_num['slippage_bps'].mean():.2f}")

    print_table(
        "Recent fills",
        fills,
        ["recorded_at", "symbol", "action", "quantity", "fill_price", "commission", "realized_pnl", "slippage_bps"],
        limit=12,
    )
    print_table(
        "Recent order intents",
        intents,
        ["recorded_at", "symbol", "action", "quantity", "notional_usd", "order_type", "limit_price", "strategy", "reason"],
        limit=12,
    )
    print_table(
        "Recent signals",
        signals,
        ["recorded_at", "symbol", "signal_name", "action", "score", "threshold", "reasons"],
        limit=12,
    )
    print_table(
        "Recent selection events",
        selections,
        ["recorded_at", "symbol", "stage", "decision", "rank", "score", "reason", "first_5m_high_pct", "first_15m_high_pct", "or_range_pct"],
        limit=12,
    )
    print_table(
        "Recent errors",
        errors,
        ["recorded_at", "severity", "component", "symbol", "message", "exception_type"],
        limit=8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="v63 live portfolio/trade monitor from recorder CSV files")
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--session-dir", default="")
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    session = Path(args.session_dir) if args.session_dir else latest_session_dir(args.recorder_dir)

    while True:
        render(session)
        if args.once:
            break
        time.sleep(args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
