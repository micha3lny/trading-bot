from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

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


def to_num(value: Any, default: float = 0.0) -> float:
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def money(x: Any) -> str:
    try:
        if pd.isna(x):
            return "$0.00"
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "0.00%"
        return f"{float(x):.2f}%"
    except Exception:
        return "0.00%"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    return out.fillna("")


def print_table(title: str, df: pd.DataFrame, cols: list[str], limit: int = 10) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("empty")
        return
    show = clean_df(df.tail(limit).copy())
    existing = [c for c in cols if c in show.columns]
    if not existing:
        print(show.tail(limit).to_string(index=False))
        return
    print(show[existing].to_string(index=False))


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_latest_strategy_positions(session: Path) -> pd.DataFrame:
    equity = read_csv_safe(session / "strategy_equity.csv")
    if equity.empty or "positions_json" not in equity.columns:
        return pd.DataFrame()
    rows = parse_json_list(equity.tail(1).iloc[0].get("positions_json"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ["qty", "entry", "price", "peak", "unrealized_pnl", "unrealized_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"qty", "price"}.issubset(df.columns):
        df["market_value"] = df["qty"] * df["price"]
    if {"qty", "entry"}.issubset(df.columns):
        df["cost_basis"] = df["qty"] * df["entry"]
    if {"entry", "price"}.issubset(df.columns):
        df["current_pct"] = (df["price"] / df["entry"] - 1.0) * 100.0
    return df


def build_closed_trades(lifecycle: pd.DataFrame, assumed_roundtrip_commission: float) -> pd.DataFrame:
    if lifecycle.empty or "event" not in lifecycle.columns:
        return pd.DataFrame()
    closed = lifecycle[lifecycle["event"].astype(str).eq("SELL_ORDER_SENT")].copy()
    if closed.empty:
        return closed
    for c in ["quantity", "price", "entry_price", "peak_price", "pnl_pct"]:
        if c in closed.columns:
            closed[c] = pd.to_numeric(closed[c], errors="coerce")
    if {"quantity", "price", "entry_price"}.issubset(closed.columns):
        closed["gross_pnl_usd"] = (closed["price"] - closed["entry_price"]) * closed["quantity"]
        closed["entry_notional"] = closed["entry_price"] * closed["quantity"]
        closed["exit_notional"] = closed["price"] * closed["quantity"]
        closed["assumed_commission"] = assumed_roundtrip_commission
        closed["estimated_net_pnl_usd"] = closed["gross_pnl_usd"] - assumed_roundtrip_commission
        closed["estimated_net_pct"] = (closed["estimated_net_pnl_usd"] / closed["entry_notional"]) * 100.0
    return closed


def render_ibkr_section(session: Path) -> None:
    portfolio = read_csv_safe(session / "portfolio_snapshots.csv")
    print("\n=== IBKR Account / Portfolio ===")
    if portfolio.empty:
        print("empty")
        return
    last = portfolio.tail(1).iloc[0]
    print(f"Account:          {last.get('account', '')}")
    print(f"Cash:             {money(last.get('cash'))}")
    print(f"Net liquidation:  {money(last.get('net_liquidation'))}")
    print(f"Buying power:     {money(last.get('buying_power'))}")
    print(f"Gross exposure:   {money(last.get('gross_exposure'))}")
    print(f"Open positions:   {last.get('open_positions', '')}")
    print(f"Daily PnL:        {money(last.get('daily_pnl'))}")
    print(f"Unrealized PnL:   {money(last.get('unrealized_pnl'))}")
    print(f"IBKR Realized PnL:{money(last.get('realized_pnl'))}")
    print(f"Recorded at:      {last.get('recorded_at', '')}")


def render_strategy_section(session: Path, table_limit: int, assumed_roundtrip_commission: float) -> None:
    lifecycle = read_csv_safe(session / "trade_lifecycle.csv")
    equity = read_csv_safe(session / "strategy_equity.csv")
    open_positions = load_latest_strategy_positions(session)
    closed = build_closed_trades(lifecycle, assumed_roundtrip_commission)

    print("\n=== Bot Strategy v67 ===")
    if lifecycle.empty and equity.empty and open_positions.empty:
        print("empty")
        return

    realized_gross = to_num(closed["gross_pnl_usd"].sum()) if not closed.empty and "gross_pnl_usd" in closed.columns else 0.0
    realized_net = to_num(closed["estimated_net_pnl_usd"].sum()) if not closed.empty and "estimated_net_pnl_usd" in closed.columns else 0.0
    unrealized = to_num(open_positions["unrealized_pnl"].sum()) if not open_positions.empty and "unrealized_pnl" in open_positions.columns else 0.0
    exposure = to_num(open_positions["market_value"].sum()) if not open_positions.empty and "market_value" in open_positions.columns else 0.0

    print(f"Open bot positions:       {len(open_positions)}")
    print(f"Bot exposure now:         {money(exposure)}")
    print(f"Closed trades:            {len(closed)}")
    print(f"Realized gross PnL:       {money(realized_gross)}")
    print(f"Assumed commissions:      {money(len(closed) * assumed_roundtrip_commission)}")
    print(f"Realized estimated net:   {money(realized_net)}")
    print(f"Unrealized PnL open:      {money(unrealized)}")
    print(f"Total strategy est. PnL:  {money(realized_net + unrealized)}")
    print(f"Commission assumption:    {money(assumed_roundtrip_commission)} per roundtrip trade")

    if not open_positions.empty:
        show = open_positions.copy()
        for c in ["entry", "price", "peak", "market_value", "cost_basis", "unrealized_pnl", "current_pct", "unrealized_pct"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce")
        print_table(
            "Bot open positions",
            show.sort_values("unrealized_pnl", ascending=True) if "unrealized_pnl" in show.columns else show,
            ["symbol", "qty", "cost_basis", "market_value", "entry", "price", "peak", "current_pct", "unrealized_pnl"],
            limit=table_limit,
        )

    if not closed.empty:
        print_table(
            "Bot closed trades",
            closed,
            [
                "recorded_at", "symbol", "quantity", "entry_notional", "exit_notional",
                "entry_price", "price", "peak_price", "pnl_pct", "gross_pnl_usd",
                "assumed_commission", "estimated_net_pnl_usd", "estimated_net_pct", "reason",
            ],
            limit=table_limit,
        )

    print_table(
        "Recent bot lifecycle",
        lifecycle,
        ["recorded_at", "event", "symbol", "action", "quantity", "price", "entry_price", "peak_price", "pnl_pct", "reason"],
        limit=table_limit,
    )


def render_fills_section(session: Path, table_limit: int) -> None:
    fills = read_csv_safe(session / "fills.csv")
    print("\n=== IBKR fills / executions ===")
    if fills.empty:
        print("empty")
        return
    fills_num = fills.copy()
    for c in ["quantity", "fill_price", "commission", "realized_pnl", "slippage_bps"]:
        if c in fills_num.columns:
            fills_num[c] = pd.to_numeric(fills_num[c], errors="coerce")
    print(f"Fills rows:        {len(fills_num)}")
    if "commission" in fills_num.columns:
        print(f"IBKR commission:   {money(fills_num['commission'].fillna(0).sum())}")
    if "realized_pnl" in fills_num.columns:
        print(f"IBKR fills PnL:    {money(fills_num['realized_pnl'].fillna(0).sum())}")
    print_table(
        "Recent IBKR fills",
        fills_num,
        ["recorded_at", "symbol", "action", "quantity", "fill_price", "commission", "realized_pnl", "exchange", "order_id", "execution_id"],
        limit=table_limit,
    )


def render_execution_quality(session: Path, table_limit: int) -> None:
    eq = read_csv_safe(session / "execution_quality.csv")
    print("\n=== Execution quality ===")
    if eq.empty:
        print("empty - next patch will start writing decision_bid/ask/mid, slippage and latency")
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
        limit=table_limit,
    )


def render(session: Path, *, clear_screen: bool, table_limit: int, render_count: int, assumed_roundtrip_commission: float) -> None:
    if clear_screen:
        os.system("clear")
    else:
        print("\n" + "=" * 140)

    print("=== v63 live portfolio monitor ===")
    print(f"Session: {session}")
    print(f"Refresh count: {render_count}")
    print(f"Mode: {'clear-screen' if clear_screen else 'append/no-clear'}")

    render_ibkr_section(session)
    render_strategy_section(session, table_limit=table_limit, assumed_roundtrip_commission=assumed_roundtrip_commission)
    render_execution_quality(session, table_limit=table_limit)
    render_fills_section(session, table_limit=table_limit)

    errors = read_csv_safe(session / "error_events.csv")
    print_table(
        "Recent errors",
        errors,
        ["recorded_at", "severity", "component", "symbol", "message", "exception_type"],
        limit=min(table_limit, 8),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="v63 live portfolio/trade monitor")
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--session-dir", default="")
    parser.add_argument("--refresh", type=float, default=20.0)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--table-limit", type=int, default=10)
    parser.add_argument("--assumed-roundtrip-commission", type=float, default=1.00)
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
            assumed_roundtrip_commission=args.assumed_roundtrip_commission,
        )
        time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
