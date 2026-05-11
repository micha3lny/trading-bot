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


def parse_positions_json(value) -> list[dict]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def render_strategy_section(session: Path) -> None:
    lifecycle = read_csv_safe(session / "trade_lifecycle.csv")
    equity = read_csv_safe(session / "strategy_equity.csv")

    print("\n=== Strategy v67 PnL ===")
    if lifecycle.empty and equity.empty:
        print("empty")
        return

    realized = 0.0
    closed_trades = pd.DataFrame()
    if not lifecycle.empty and "event" in lifecycle.columns:
        closed_trades = lifecycle[lifecycle["event"].astype(str).eq("SELL_ORDER_SENT")].copy()
        for c in ["quantity", "price", "entry_price", "peak_price", "pnl_pct"]:
            if c in closed_trades.columns:
                closed_trades[c] = pd.to_numeric(closed_trades[c], errors="coerce")
        if {"quantity", "price", "entry_price"}.issubset(closed_trades.columns):
            closed_trades["gross_pnl_usd"] = (closed_trades["price"] - closed_trades["entry_price"]) * closed_trades["quantity"]
            realized = float(closed_trades["gross_pnl_usd"].fillna(0).sum())

    unrealized = 0.0
    active_positions = ""
    gross_exposure = 0.0
    latest_positions = []
    if not equity.empty:
        eq = equity.tail(1).iloc[0]
        unrealized = float(pd.to_numeric(eq.get("unrealized_pnl"), errors="coerce") or 0.0)
        gross_exposure = float(pd.to_numeric(eq.get("gross_exposure"), errors="coerce") or 0.0)
        active_positions = eq.get("active_positions", "")
        latest_positions = parse_positions_json(eq.get("positions_json"))

    print(f"Realized gross PnL:   {money(realized)}")
    print(f"Unrealized PnL:       {money(unrealized)}")
    print(f"Total strategy PnL:   {money(realized + unrealized)}")
    print(f"Strategy exposure:    {money(gross_exposure)}")
    print(f"Managed open:         {active_positions}")
    print("Fees/slippage:        not netted yet; use execution_quality once real fills/commissions are calibrated")

    if latest_positions:
        pos_df = pd.DataFrame(latest_positions)
        print_table(
            "Strategy open positions",
            pos_df,
            ["symbol", "qty", "entry", "price", "peak", "unrealized_pnl", "unrealized_pct"],
            limit=12,
        )

    if not closed_trades.empty:
        print_table(
            "Strategy closed trades",
            closed_trades,
            ["recorded_at", "symbol", "quantity", "entry_price", "price", "peak_price", "pnl_pct", "gross_pnl_usd", "reason"],
            limit=12,
        )

    print_table(
        "Recent strategy lifecycle",
        lifecycle,
        ["recorded_at", "event", "symbol", "action", "quantity", "price", "entry_price", "peak_price", "pnl_pct", "reason"],
        limit=12,
    )


def render_execution_quality(session: Path) -> None:
    eq = read_csv_safe(session / "execution_quality.csv")
    print("\n=== Execution quality ===")
    if eq.empty:
        print("empty")
        return
    for c in ["spread_bps", "slippage_bps", "commission", "fill_latency_ms", "quantity", "fill_price", "decision_mid"]:
        if c in eq.columns:
            eq[c] = pd.to_numeric(eq[c], errors="coerce")
    print(f"Events:              {len(eq)}")
    if "slippage_bps" in eq.columns:
        print(f"Avg slippage bps:    {eq['slippage_bps'].mean():.2f}")
    if "spread_bps" in eq.columns:
        print(f"Avg spread bps:      {eq['spread_bps'].mean():.2f}")
    if "commission" in eq.columns:
        print(f"Commission total:    {money(eq['commission'].fillna(0).sum())}")
    if "fill_latency_ms" in eq.columns:
        print(f"Avg fill latency ms: {eq['fill_latency_ms'].mean():.0f}")
    print_table(
        "Recent execution quality",
        eq,
        ["recorded_at", "symbol", "action", "quantity", "decision_bid", "decision_ask", "decision_mid", "decision_last", "spread_bps", "fill_price", "slippage_bps", "fill_latency_ms", "commission"],
        limit=10,
    )


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

    print("\n=== IBKR Account / Portfolio latest ===")
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

    render_strategy_section(session)
    render_execution_quality(session)

    if not fills.empty:
        fills_num = fills.copy()
        for c in ["quantity", "fill_price", "commission", "realized_pnl", "slippage_bps"]:
            if c in fills_num.columns:
                fills_num[c] = pd.to_numeric(fills_num[c], errors="coerce")
        print("\n=== IBKR fills summary ===")
        print(f"Fills:            {len(fills_num)}")
        if "realized_pnl" in fills_num.columns:
            print(f"IBKR Realized PnL:{money(fills_num['realized_pnl'].sum())}")
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
