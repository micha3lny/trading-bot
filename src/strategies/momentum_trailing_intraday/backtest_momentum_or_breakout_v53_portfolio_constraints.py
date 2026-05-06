from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v52_capital_requirements import load_trades


def quality_rank(row: pd.Series) -> float:
    """Rank candidates using only fields already known at/near entry in v51 output."""
    score = 0.0
    regime = str(row.get("market_regime", "unknown"))
    quality = str(row.get("setup_quality", "C"))
    v45 = float(row.get("v45_score", 0.0) or 0.0)
    or_strength = float(row.get("or_close_strength", 0.0) or 0.0)
    market_ret = float(row.get("market_return_from_open_pct", 0.0) or 0.0)
    entry_minutes = float(row.get("entry_minutes_from_open", 999.0) or 999.0)
    pre_break = float(row.get("pre_entry_break_below_or_pct", -99.0) or -99.0)

    score += {"strong": 5.0, "good": 3.0, "neutral": 2.0, "weak": 0.5, "bad": -3.0}.get(regime, 0.0)
    score += {"A+": 5.0, "A": 4.0, "B": 2.0, "C": -1.0}.get(quality, 0.0)
    score += min(max(v45 - 7.0, 0.0), 6.0) * 0.5
    score += or_strength * 2.0
    score += market_ret * 2.0

    if 5 <= entry_minutes <= 10:
        score += 1.0
    elif 10 < entry_minutes < 15:
        score -= 1.5
    elif entry_minutes > 30:
        score -= 1.0

    if pre_break >= -0.50:
        score += 1.0
    elif pre_break < -1.25:
        score -= 1.0

    return score


def simulate_portfolio(trades: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = trades.copy()
    df["candidate_rank"] = df.apply(quality_rank, axis=1)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df = df.dropna(subset=["entry_dt", "exit_dt"]).sort_values(["entry_dt", "candidate_rank"], ascending=[True, False]).reset_index(drop=True)

    cash = float(args.starting_cash)
    realized_pnl = 0.0
    open_positions: dict[int, dict[str, object]] = {}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    equity_points: list[dict[str, object]] = []

    events = []
    for idx, row in df.iterrows():
        events.append({"time": row["entry_dt"], "type": "entry", "idx": idx})
        events.append({"time": row["exit_dt"], "type": "exit", "idx": idx})
    ev = pd.DataFrame(events)
    if ev.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), pd.DataFrame()

    # At the same timestamp, free cash first, then evaluate new entries by best rank.
    ev["order"] = ev["type"].map({"exit": 0, "entry": 1}).fillna(9)
    ev = ev.merge(df[["candidate_rank"]], left_on="idx", right_index=True, how="left")
    ev = ev.sort_values(["time", "order", "candidate_rank"], ascending=[True, True, False]).reset_index(drop=True)

    for _, event in ev.iterrows():
        idx = int(event["idx"])
        row = df.loc[idx]
        now = event["time"]

        if event["type"] == "exit":
            if idx in open_positions:
                pos = open_positions.pop(idx)
                position_usd = float(pos["position_usd"])
                profit = float(pos["profit_usd"])
                cash += position_usd + profit
                realized_pnl += profit
        else:
            position_usd = float(row["position_usd"])
            open_exposure = sum(float(p["position_usd"]) for p in open_positions.values())
            open_count = len(open_positions)
            day = str(row["session_date"])
            day_realized = sum(float(r.get("profit_usd", 0.0)) for r in accepted if str(r.get("session_date")) == day and pd.to_datetime(r.get("exit_time")) <= now)

            reasons = []
            if row.get("candidate_rank", 0.0) < args.min_candidate_rank:
                reasons.append("candidate_rank_too_low")
            if open_count >= args.max_positions:
                reasons.append("max_positions")
            if open_exposure + position_usd > args.max_gross_exposure:
                reasons.append("max_gross_exposure")
            if cash < position_usd and not args.allow_margin:
                reasons.append("insufficient_cash")
            if day_realized <= -abs(args.max_daily_loss_usd):
                reasons.append("daily_loss_stop")

            if reasons:
                rec = row.to_dict()
                rec["skip_reason"] = ";".join(reasons)
                rec["cash_before"] = cash
                rec["open_exposure_before"] = open_exposure
                rec["open_positions_before"] = open_count
                rejected.append(rec)
            else:
                cash -= position_usd
                rec = row.to_dict()
                rec["cash_before"] = cash + position_usd
                rec["cash_after_entry"] = cash
                rec["open_exposure_before"] = open_exposure
                rec["open_positions_before"] = open_count
                accepted.append(rec)
                open_positions[idx] = rec

        current_open_exposure = sum(float(p["position_usd"]) for p in open_positions.values())
        equity_points.append({
            "time": now,
            "cash": cash,
            "open_exposure_usd": current_open_exposure,
            "open_positions": len(open_positions),
            "realized_pnl_usd": realized_pnl,
            "equity_approx_usd": cash + current_open_exposure,
        })

    return pd.DataFrame(accepted), pd.DataFrame(rejected), pd.DataFrame(equity_points)


def summarize(label: str, df: pd.DataFrame, equity: pd.DataFrame, starting_cash: float) -> dict[str, object]:
    if df.empty:
        return {"strategy": label, "trades": 0, "profit_usd": 0.0, "return_on_starting_cash_pct": 0.0, "active_days": 0, "symbols": 0, "win_rate": 0.0, "avg_profit_per_trade_usd": 0.0, "max_drawdown_usd": 0.0, "peak_exposure_usd": 0.0, "peak_positions": 0}
    profit = pd.to_numeric(df["profit_usd"], errors="coerce").fillna(0.0)
    if not equity.empty:
        eq = pd.to_numeric(equity["equity_approx_usd"], errors="coerce").fillna(starting_cash)
        dd = eq - eq.cummax()
        max_dd = float(dd.min())
        peak_exposure = float(equity["open_exposure_usd"].max())
        peak_positions = int(equity["open_positions"].max())
    else:
        max_dd = 0.0
        peak_exposure = 0.0
        peak_positions = 0
    return {
        "strategy": label,
        "trades": int(len(df)),
        "profit_usd": float(profit.sum()),
        "return_on_starting_cash_pct": float(profit.sum() / starting_cash * 100.0),
        "active_days": int(df["session_date"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "win_rate": float((profit > 0).mean() * 100.0),
        "avg_profit_per_trade_usd": float(profit.mean()),
        "max_drawdown_usd": max_dd,
        "peak_exposure_usd": peak_exposure,
        "peak_positions": peak_positions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v53 portfolio constraints: take best candidates under cash/exposure limits.")
    parser.add_argument("--trades-csv", default="data/backtests/v51_aggressive_exposure_trades_recent90_or5_wide_trail_QQQ_aggressive.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--starting-cash", type=float, default=20000.0)
    parser.add_argument("--max-gross-exposure", type=float, default=20000.0)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--max-daily-loss-usd", type=float, default=400.0)
    parser.add_argument("--min-candidate-rank", type=float, default=0.0)
    parser.add_argument("--allow-margin", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    print("Experiment: v53 portfolio constraints")
    print("Goal: choose best candidates under realistic cash/exposure/position limits.")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Starting cash: ${args.starting_cash:.2f}")
    print(f"Max gross exposure: ${args.max_gross_exposure:.2f}")
    print(f"Max positions: {args.max_positions}")
    print(f"Max daily loss: ${args.max_daily_loss_usd:.2f}")

    trades = load_trades(args.trades_csv)
    accepted, rejected, equity = simulate_portfolio(trades, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"cash{int(args.starting_cash)}_exposure{int(args.max_gross_exposure)}_pos{args.max_positions}"
    out_accepted = output_dir / f"v53_portfolio_accepted_{suffix}.csv"
    out_rejected = output_dir / f"v53_portfolio_rejected_{suffix}.csv"
    out_equity = output_dir / f"v53_portfolio_equity_{suffix}.csv"
    out_summary = output_dir / f"v53_portfolio_summary_{suffix}.csv"
    accepted.to_csv(out_accepted, index=False)
    rejected.to_csv(out_rejected, index=False)
    equity.to_csv(out_equity, index=False)

    summary = pd.DataFrame([summarize("v53_portfolio_constrained", accepted, equity, args.starting_cash)])
    summary.to_csv(out_summary, index=False)

    print("\n=== Portfolio constrained summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"Saved accepted: {out_accepted}")
    print(f"Saved rejected: {out_rejected}")
    print(f"Saved equity: {out_equity}")
    print(f"Saved summary: {out_summary}")

    print("\n=== Rejection reasons ===")
    if rejected.empty:
        print("none")
    else:
        print(rejected["skip_reason"].value_counts().head(20).to_string())

    if not accepted.empty:
        print("\n=== Top days by constrained profit ===")
        print(accepted.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(20).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Worst days by constrained profit ===")
        print(accepted.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=True).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- This is closer to paper trading: not every signal is taken when cash/exposure is full.")
    print("- If too many good trades are skipped by max_positions, increase --max-positions or use smaller sizing.")
    print("- If drawdown is too high, lower --max-gross-exposure or --max-daily-loss-usd.")
    print("- Next step after this: connect scanner/executor to IBKR paper account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
