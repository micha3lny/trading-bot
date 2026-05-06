from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR


def load_trades(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    required = {"symbol", "session_date", "entry_time", "exit_time", "position_usd", "profit_usd", "pnl_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["entry_dt", "exit_dt"]).sort_values(["entry_dt", "symbol"]).reset_index(drop=True)
    return df


def exposure_timeline(df: pd.DataFrame) -> pd.DataFrame:
    events = []
    for idx, row in df.iterrows():
        pos = float(row["position_usd"])
        events.append({"time": row["entry_dt"], "delta_exposure": pos, "delta_positions": 1, "event": "entry", "trade_index": idx})
        events.append({"time": row["exit_dt"], "delta_exposure": -pos, "delta_positions": -1, "event": "exit", "trade_index": idx})
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev
    # Process entries before exits at the same timestamp to be conservative for capital requirement.
    ev["event_order"] = ev["event"].map({"entry": 0, "exit": 1}).fillna(9)
    ev = ev.sort_values(["time", "event_order", "trade_index"]).reset_index(drop=True)
    ev["open_exposure_usd"] = ev["delta_exposure"].cumsum()
    ev["open_positions"] = ev["delta_positions"].cumsum()
    ev["session_date"] = ev["time"].dt.date.astype(str)
    return ev


def daily_summary(df: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    trades = df.groupby("session_date").agg(
        trades=("symbol", "count"),
        symbols=("symbol", "nunique"),
        day_profit_usd=("profit_usd", "sum"),
        avg_position_usd=("position_usd", "mean"),
        max_single_position_usd=("position_usd", "max"),
        gross_position_sum_usd=("position_usd", "sum"),
    )
    if timeline.empty:
        trades["peak_open_exposure_usd"] = 0.0
        trades["peak_open_positions"] = 0
        return trades.reset_index()
    exp = timeline.groupby("session_date").agg(
        peak_open_exposure_usd=("open_exposure_usd", "max"),
        peak_open_positions=("open_positions", "max"),
    )
    out = trades.join(exp, how="left").fillna({"peak_open_exposure_usd": 0.0, "peak_open_positions": 0})
    return out.reset_index().sort_values("day_profit_usd", ascending=False)


def simulate_cash_constraints(df: pd.DataFrame, starting_cash: float, allow_margin: bool, max_gross_exposure: float | None) -> tuple[pd.DataFrame, dict[str, float]]:
    cash = starting_cash
    open_positions: dict[int, float] = {}
    executed = []
    skipped = []

    events = []
    for idx, row in df.iterrows():
        events.append({"time": row["entry_dt"], "type": "entry", "idx": idx})
        events.append({"time": row["exit_dt"], "type": "exit", "idx": idx})
    ev = pd.DataFrame(events)
    if ev.empty:
        return df.iloc[0:0].copy(), {"executed": 0, "skipped": 0, "ending_cash": starting_cash, "max_open_exposure": 0.0}
    ev["order"] = ev["type"].map({"exit": 0, "entry": 1}).fillna(9)
    ev = ev.sort_values(["time", "order", "idx"]).reset_index(drop=True)

    max_open = 0.0
    for _, event in ev.iterrows():
        idx = int(event["idx"])
        row = df.loc[idx]
        if event["type"] == "exit":
            if idx in open_positions:
                pos = open_positions.pop(idx)
                cash += pos + float(row["profit_usd"])
        else:
            pos = float(row["position_usd"])
            current_open = sum(open_positions.values())
            gross_after = current_open + pos
            if max_gross_exposure is not None and gross_after > max_gross_exposure:
                skipped.append({**row.to_dict(), "skip_reason": "max_gross_exposure"})
                continue
            if not allow_margin and cash < pos:
                skipped.append({**row.to_dict(), "skip_reason": "insufficient_cash"})
                continue
            cash -= pos
            open_positions[idx] = pos
            executed.append(row.to_dict())
            max_open = max(max_open, sum(open_positions.values()))

    executed_df = pd.DataFrame(executed)
    skipped_df = pd.DataFrame(skipped)
    stats = {
        "starting_cash": float(starting_cash),
        "ending_cash": float(cash + sum(open_positions.values())),
        "realized_plus_open_equity": float(cash + sum(open_positions.values())),
        "executed": float(len(executed_df)),
        "skipped": float(len(skipped_df)),
        "executed_profit_usd": float(executed_df["profit_usd"].sum()) if not executed_df.empty else 0.0,
        "max_open_exposure_usd": float(max_open),
    }
    return executed_df, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v52 capital requirements and portfolio exposure simulator for v51 adaptive trades.")
    parser.add_argument("--trades-csv", default="data/backtests/v51_aggressive_exposure_trades_recent90_or5_wide_trail_QQQ_aggressive.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cash-tests", nargs="*", type=float, default=[5000, 10000, 15000, 20000, 30000, 50000])
    parser.add_argument("--allow-margin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-gross-exposure", type=float, default=None)
    args = parser.parse_args()

    print("Experiment: v52 capital requirements")
    print("Input: v51 adaptive trades")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Allow margin: {args.allow_margin}")
    print(f"Max gross exposure cap: {args.max_gross_exposure}")

    trades = load_trades(args.trades_csv)
    timeline = exposure_timeline(trades)
    daily = daily_summary(trades, timeline)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_timeline = output_dir / "v52_capital_exposure_timeline.csv"
    out_daily = output_dir / "v52_capital_daily_summary.csv"
    timeline.to_csv(out_timeline, index=False)
    daily.to_csv(out_daily, index=False)

    total_profit = float(trades["profit_usd"].sum())
    peak_exposure = float(timeline["open_exposure_usd"].max()) if not timeline.empty else 0.0
    peak_positions = int(timeline["open_positions"].max()) if not timeline.empty else 0
    max_daily_trades = int(daily["trades"].max()) if not daily.empty else 0
    max_daily_gross = float(daily["gross_position_sum_usd"].max()) if not daily.empty else 0.0
    max_daily_peak_exposure = float(daily["peak_open_exposure_usd"].max()) if not daily.empty else 0.0

    print("\n=== Capital requirement summary ===")
    print(f"Trades: {len(trades)}")
    print(f"Total profit USD: {total_profit:.2f}")
    print(f"Max trades in one day: {max_daily_trades}")
    print(f"Max gross daily traded notional: ${max_daily_gross:.2f}")
    print(f"Peak simultaneous open positions: {peak_positions}")
    print(f"Peak simultaneous exposure: ${peak_exposure:.2f}")
    print(f"Peak daily simultaneous exposure: ${max_daily_peak_exposure:.2f}")
    print(f"Saved exposure timeline: {out_timeline}")
    print(f"Saved daily summary: {out_daily}")

    print("\n=== Top days by trade count ===")
    print(daily.sort_values("trades", ascending=False).head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Top days by peak exposure ===")
    print(daily.sort_values("peak_open_exposure_usd", ascending=False).head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    cash_rows = []
    for cash in args.cash_tests:
        executed_df, stats = simulate_cash_constraints(
            trades,
            starting_cash=float(cash),
            allow_margin=args.allow_margin,
            max_gross_exposure=args.max_gross_exposure,
        )
        cash_rows.append(stats)
    cash_summary = pd.DataFrame(cash_rows)
    out_cash = output_dir / "v52_capital_cash_test_summary.csv"
    cash_summary.to_csv(out_cash, index=False)

    print("\n=== Cash constraint tests ===")
    print(cash_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"Saved cash tests: {out_cash}")

    print("\nInterpretation hints:")
    print("- Peak simultaneous exposure is the minimum cash needed to take every trade without margin and without cash recycling issues.")
    print("- Max gross daily traded notional is NOT required account balance; it is turnover through the day.")
    print("- For live/paper trading, start with cash below peak exposure and let the simulator skip trades that exceed cash/exposure limits.")
    print("- Next step: run this on paper account with small size and strict max exposure / max daily loss limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
