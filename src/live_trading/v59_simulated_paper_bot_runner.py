from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


DEFAULT_INTENTS = "data/live/order_intents.csv"
DEFAULT_SNAPSHOTS = "data/live/live_signal_snapshots.csv"
DEFAULT_OUTPUT_DIR = "data/live"


@dataclass
class SimPosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    position_usd: float
    stop_price: float
    entry_commission_usd: float
    trailing_stop_price: float | None = None
    max_price: float = 0.0
    reason: str = ""
    score: float = 0.0

    def unrealized_pnl(self, price: float) -> float:
        return self.quantity * (price - self.entry_price)


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def read_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def prepare_intents(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0)
    out["reference_price"] = pd.to_numeric(out.get("reference_price"), errors="coerce")
    out["quantity"] = pd.to_numeric(out.get("quantity"), errors="coerce").fillna(0).astype(int)
    out["position_usd"] = pd.to_numeric(out.get("position_usd"), errors="coerce").fillna(0.0)
    out = out.dropna(subset=["timestamp_utc", "symbol", "reference_price"])
    return out.sort_values(["timestamp_utc", "score"], ascending=[True, False]).reset_index(drop=True)


def prepare_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    for col in ["reference_price", "last", "mid", "bid", "ask", "score", "spread_bps"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp_utc", "symbol"])
    return out.sort_values(["timestamp_utc", "symbol"]).reset_index(drop=True)


def current_price_from_snapshot(row: pd.Series) -> float | None:
    for col in ["reference_price", "last", "mid", "ask", "bid"]:
        val = row.get(col)
        try:
            if pd.notna(val) and float(val) > 0:
                return float(val)
        except Exception:
            pass
    return None


def estimate_commission(quantity: int, args: argparse.Namespace) -> float:
    if quantity <= 0:
        return 0.0
    if args.commission_model == "none":
        return 0.0
    commission = abs(quantity) * args.commission_per_share
    return max(args.min_commission_per_order, commission)


def estimate_sell_reg_fees(quantity: int, sell_notional_usd: float, args: argparse.Namespace) -> float:
    if quantity <= 0 or sell_notional_usd <= 0:
        return 0.0
    sec_fee = sell_notional_usd * args.sec_fee_rate
    finra_taf = min(abs(quantity) * args.finra_taf_per_share, args.finra_taf_cap)
    return sec_fee + finra_taf


def enter_position(
    *,
    ts: pd.Timestamp,
    symbol: str,
    price: float,
    intent: dict[str, object],
    args: argparse.Namespace,
    cash: float,
    open_positions: dict[str, SimPosition],
    trade_events: list[dict[str, object]],
) -> tuple[float, bool]:
    if symbol in open_positions:
        return cash, False
    if len(open_positions) >= args.max_positions:
        return cash, False

    entry_price = price * (1.0 + args.simulated_entry_slippage_bps / 10_000.0)
    quantity = int(intent.get("quantity") or 0)
    if quantity <= 0:
        quantity = int(float(intent.get("position_usd") or args.base_position_usd) // entry_price)
    if quantity <= 0:
        return cash, False

    notional = quantity * entry_price
    if notional > args.max_position_usd:
        quantity = int(args.max_position_usd // entry_price)
        notional = quantity * entry_price
    if quantity <= 0 or notional < args.min_position_usd:
        return cash, False

    open_exposure = sum(p.quantity * p.entry_price for p in open_positions.values())
    if open_exposure + notional > args.max_gross_exposure:
        return cash, False

    entry_commission = estimate_commission(quantity, args)
    total_cash_required = notional + entry_commission
    if cash < total_cash_required:
        return cash, False

    cash -= total_cash_required
    score = float(intent.get("score") or 0.0)
    stop_price = entry_price * (1.0 - args.stop_loss_pct / 100.0)
    open_positions[symbol] = SimPosition(
        symbol=symbol,
        entry_time=ts,
        entry_price=entry_price,
        quantity=quantity,
        position_usd=notional,
        stop_price=stop_price,
        entry_commission_usd=entry_commission,
        max_price=entry_price,
        reason=str(intent.get("reason") or "BUY_INTENT"),
        score=score,
    )
    trade_events.append({
        "timestamp_utc": ts.isoformat(),
        "event": "ENTRY",
        "symbol": symbol,
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": None,
        "gross_pnl_usd": None,
        "commission_usd": entry_commission,
        "reg_fees_usd": 0.0,
        "total_costs_usd": entry_commission,
        "net_pnl_usd": -entry_commission,
        "pnl_pct": None,
        "reason": str(intent.get("reason") or "BUY_INTENT"),
        "cash_after": cash,
        "score": score,
    })
    return cash, True


def exit_position(
    *,
    ts: pd.Timestamp,
    symbol: str,
    price: float,
    reason: str,
    args: argparse.Namespace,
    cash: float,
    realized_gross_pnl: float,
    realized_net_pnl: float,
    open_positions: dict[str, SimPosition],
    trade_events: list[dict[str, object]],
) -> tuple[float, float, float]:
    pos = open_positions[symbol]
    gross_pnl = pos.quantity * (price - pos.entry_price)
    exit_notional = pos.quantity * price
    exit_commission = estimate_commission(pos.quantity, args)
    reg_fees = estimate_sell_reg_fees(pos.quantity, exit_notional, args)
    total_costs = pos.entry_commission_usd + exit_commission + reg_fees
    net_pnl = gross_pnl - total_costs

    cash += exit_notional - exit_commission - reg_fees
    realized_gross_pnl += gross_pnl
    realized_net_pnl += net_pnl

    trade_events.append({
        "timestamp_utc": ts.isoformat(),
        "event": "EXIT",
        "symbol": symbol,
        "quantity": pos.quantity,
        "entry_price": pos.entry_price,
        "exit_price": price,
        "gross_pnl_usd": gross_pnl,
        "commission_usd": pos.entry_commission_usd + exit_commission,
        "reg_fees_usd": reg_fees,
        "total_costs_usd": total_costs,
        "net_pnl_usd": net_pnl,
        "pnl_pct": (price / pos.entry_price - 1.0) * 100.0,
        "reason": reason,
        "cash_after": cash,
        "score": pos.score,
    })
    del open_positions[symbol]
    return cash, realized_gross_pnl, realized_net_pnl


def build_daily_report(events_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    exits = events_df[events_df["event"] == "EXIT"].copy()
    if exits.empty:
        return pd.DataFrame()
    exits["date"] = pd.to_datetime(exits["timestamp_utc"], utc=True).dt.date.astype(str)
    for col in ["gross_pnl_usd", "net_pnl_usd", "commission_usd", "reg_fees_usd", "total_costs_usd"]:
        exits[col] = pd.to_numeric(exits[col], errors="coerce").fillna(0.0)
    daily = exits.groupby("date").agg(
        trades=("symbol", "count"),
        symbols=("symbol", "nunique"),
        gross_pnl_usd=("gross_pnl_usd", "sum"),
        net_pnl_usd=("net_pnl_usd", "sum"),
        commission_usd=("commission_usd", "sum"),
        reg_fees_usd=("reg_fees_usd", "sum"),
        total_costs_usd=("total_costs_usd", "sum"),
        win_rate=("net_pnl_usd", lambda s: float((s > 0).mean() * 100.0)),
        avg_net_trade_usd=("net_pnl_usd", "mean"),
        best_net_trade_usd=("net_pnl_usd", "max"),
        worst_net_trade_usd=("net_pnl_usd", "min"),
    ).reset_index()

    if not equity_df.empty:
        eq = equity_df.copy()
        eq["date"] = pd.to_datetime(eq["timestamp_utc"], utc=True).dt.date.astype(str)
        eq["equity_usd"] = pd.to_numeric(eq["equity_usd"], errors="coerce")
        eq["open_positions"] = pd.to_numeric(eq["open_positions"], errors="coerce").fillna(0)
        eq_daily = eq.groupby("date").agg(
            start_equity_usd=("equity_usd", "first"),
            end_equity_usd=("equity_usd", "last"),
            min_equity_usd=("equity_usd", "min"),
            max_equity_usd=("equity_usd", "max"),
            max_open_positions=("open_positions", "max"),
        ).reset_index()
        daily = daily.merge(eq_daily, on="date", how="left")
        daily["intraday_drawdown_usd"] = daily["min_equity_usd"] - daily["max_equity_usd"]
    return daily


def simulate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intents = prepare_intents(read_csv_if_exists(args.intents_csv))
    snapshots = prepare_snapshots(read_csv_if_exists(args.snapshots_csv))

    if snapshots.empty:
        raise RuntimeError("No snapshots found. Run v58_live_signal_engine or v59_fake_intent_generator first.")
    if intents.empty:
        print("No order intents found. Runner will only produce empty summary.")

    cash = float(args.starting_cash)
    realized_gross_pnl = 0.0
    realized_net_pnl = 0.0
    open_positions: dict[str, SimPosition] = {}
    trade_events: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    last_prices: dict[str, tuple[pd.Timestamp, float]] = {}
    pending_intents: list[dict[str, object]] = []
    intents_sorted = intents.to_dict("records") if not intents.empty else []
    intent_idx = 0
    daily_loss_triggered = False

    for _, snap in snapshots.iterrows():
        ts = snap["timestamp_utc"]
        symbol = str(snap["symbol"])
        price = current_price_from_snapshot(snap)
        if price is None:
            continue
        last_prices[symbol] = (ts, price)

        while intent_idx < len(intents_sorted) and intents_sorted[intent_idx]["timestamp_utc"] <= ts:
            pending_intents.append(intents_sorted[intent_idx])
            intent_idx += 1

        if symbol in open_positions:
            pos = open_positions[symbol]
            pos.max_price = max(pos.max_price, price)
            max_gain_pct = (pos.max_price / pos.entry_price - 1.0) * 100.0
            if max_gain_pct >= args.trailing_activation_pct:
                trail_price = pos.max_price * (1.0 - args.trailing_stop_pct / 100.0)
                pos.trailing_stop_price = max(pos.trailing_stop_price or 0.0, trail_price)

            exit_reason = None
            if price <= pos.stop_price:
                exit_reason = "stop_loss"
            elif pos.trailing_stop_price is not None and price <= pos.trailing_stop_price:
                exit_reason = "trailing_stop"

            if exit_reason is not None:
                cash, realized_gross_pnl, realized_net_pnl = exit_position(
                    ts=ts,
                    symbol=symbol,
                    price=price,
                    reason=exit_reason,
                    args=args,
                    cash=cash,
                    realized_gross_pnl=realized_gross_pnl,
                    realized_net_pnl=realized_net_pnl,
                    open_positions=open_positions,
                    trade_events=trade_events,
                )

        still_pending: list[dict[str, object]] = []
        symbol_intents = [i for i in pending_intents if str(i.get("symbol")) == symbol]
        other_intents = [i for i in pending_intents if str(i.get("symbol")) != symbol]
        symbol_intents = sorted(symbol_intents, key=lambda i: float(i.get("score") or 0.0), reverse=True)

        for intent in symbol_intents:
            if daily_loss_triggered:
                still_pending.append(intent)
                continue
            cash, accepted = enter_position(
                ts=ts,
                symbol=symbol,
                price=price,
                intent=intent,
                args=args,
                cash=cash,
                open_positions=open_positions,
                trade_events=trade_events,
            )
            if not accepted:
                still_pending.append(intent)
        pending_intents = other_intents + still_pending

        open_value = 0.0
        open_unrealized_gross = 0.0
        entry_costs_open = 0.0
        for p_symbol, pos in open_positions.items():
            p_price = last_prices.get(p_symbol, (ts, pos.entry_price))[1]
            open_value += pos.quantity * p_price
            open_unrealized_gross += pos.unrealized_pnl(p_price)
            entry_costs_open += pos.entry_commission_usd

        equity = cash + open_value
        open_unrealized_net_est = open_unrealized_gross - entry_costs_open
        day_net_pnl = realized_net_pnl + open_unrealized_net_est
        if day_net_pnl <= -abs(args.max_daily_loss_usd):
            daily_loss_triggered = True

        equity_rows.append({
            "timestamp_utc": ts.isoformat(),
            "cash": cash,
            "open_positions": len(open_positions),
            "open_value_usd": open_value,
            "open_unrealized_gross_pnl_usd": open_unrealized_gross,
            "open_unrealized_net_est_pnl_usd": open_unrealized_net_est,
            "realized_gross_pnl_usd": realized_gross_pnl,
            "realized_net_pnl_usd": realized_net_pnl,
            "equity_usd": equity,
            "daily_loss_triggered": daily_loss_triggered,
            "pending_intents": len(pending_intents),
        })

    for symbol, pos in list(open_positions.items()):
        ts, price = last_prices.get(symbol, (snapshots.iloc[-1]["timestamp_utc"], pos.entry_price))
        exit_price = price * (1.0 - args.simulated_exit_slippage_bps / 10_000.0)
        cash, realized_gross_pnl, realized_net_pnl = exit_position(
            ts=ts,
            symbol=symbol,
            price=exit_price,
            reason="end_of_run_flatten",
            args=args,
            cash=cash,
            realized_gross_pnl=realized_gross_pnl,
            realized_net_pnl=realized_net_pnl,
            open_positions=open_positions,
            trade_events=trade_events,
        )

    events_df = pd.DataFrame(trade_events)
    equity_df = pd.DataFrame(equity_rows)
    daily_df = build_daily_report(events_df, equity_df)

    if events_df.empty:
        summary = pd.DataFrame([{
            "strategy": "v59_simulated_paper_bot",
            "entries": 0,
            "exits": 0,
            "gross_pnl_usd": 0.0,
            "commission_usd": 0.0,
            "reg_fees_usd": 0.0,
            "total_costs_usd": 0.0,
            "net_pnl_usd": 0.0,
            "ending_cash": cash,
            "return_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_open_positions": 0,
            "pending_intents_left": len(pending_intents),
        }])
    else:
        exits = events_df[events_df["event"] == "EXIT"].copy()
        for col in ["gross_pnl_usd", "net_pnl_usd", "commission_usd", "reg_fees_usd", "total_costs_usd"]:
            exits[col] = pd.to_numeric(exits.get(col), errors="coerce").fillna(0.0) if not exits.empty else pd.Series(dtype=float)
        net = exits["net_pnl_usd"] if not exits.empty else pd.Series(dtype=float)
        if not equity_df.empty:
            eq = pd.to_numeric(equity_df["equity_usd"], errors="coerce")
            dd = eq - eq.cummax()
            max_dd = float(dd.min())
            max_open = int(equity_df["open_positions"].max())
        else:
            max_dd = 0.0
            max_open = 0
        summary = pd.DataFrame([{
            "strategy": "v59_simulated_paper_bot",
            "entries": int((events_df["event"] == "ENTRY").sum()),
            "exits": int((events_df["event"] == "EXIT").sum()),
            "gross_pnl_usd": float(exits["gross_pnl_usd"].sum()) if not exits.empty else 0.0,
            "commission_usd": float(exits["commission_usd"].sum()) if not exits.empty else 0.0,
            "reg_fees_usd": float(exits["reg_fees_usd"].sum()) if not exits.empty else 0.0,
            "total_costs_usd": float(exits["total_costs_usd"].sum()) if not exits.empty else 0.0,
            "net_pnl_usd": float(net.sum()) if not net.empty else 0.0,
            "ending_cash": float(cash),
            "return_pct": float((cash / args.starting_cash - 1.0) * 100.0),
            "win_rate": float((net > 0).mean() * 100.0) if not net.empty else 0.0,
            "avg_net_trade_pnl_usd": float(net.mean()) if not net.empty else 0.0,
            "max_drawdown_usd": max_dd,
            "max_open_positions": max_open,
            "pending_intents_left": len(pending_intents),
        }])

    return events_df, equity_df, summary, daily_df


def main() -> int:
    parser = argparse.ArgumentParser(description="v59 simulated paper bot runner from v58 order intents")
    parser.add_argument("--intents-csv", default=DEFAULT_INTENTS)
    parser.add_argument("--snapshots-csv", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=25_000.0)
    parser.add_argument("--base-position-usd", type=float, default=100.0)
    parser.add_argument("--min-position-usd", type=float, default=50.0)
    parser.add_argument("--max-position-usd", type=float, default=250.0)
    parser.add_argument("--max-gross-exposure", type=float, default=1_000.0)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-daily-loss-usd", type=float, default=100.0)
    parser.add_argument("--stop-loss-pct", type=float, default=2.0)
    parser.add_argument("--trailing-activation-pct", type=float, default=1.0)
    parser.add_argument("--trailing-stop-pct", type=float, default=0.7)
    parser.add_argument("--simulated-entry-slippage-bps", type=float, default=3.0)
    parser.add_argument("--simulated-exit-slippage-bps", type=float, default=3.0)
    parser.add_argument("--commission-model", choices=["ibkr_fixed", "none"], default="ibkr_fixed")
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--min-commission-per-order", type=float, default=1.0)
    parser.add_argument("--sec-fee-rate", type=float, default=0.0000278)
    parser.add_argument("--finra-taf-per-share", type=float, default=0.000166)
    parser.add_argument("--finra-taf-cap", type=float, default=8.30)
    args = parser.parse_args()

    print("=== v59 simulated paper bot runner ===")
    print(f"Intents: {args.intents_csv}")
    print(f"Snapshots: {args.snapshots_csv}")
    print(f"Starting cash: ${args.starting_cash:.2f}")
    print(f"Commission model: {args.commission_model}")

    events, equity, summary, daily = simulate(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "v59_simulated_trade_events.csv"
    equity_path = out_dir / "v59_simulated_equity_curve.csv"
    summary_path = out_dir / "v59_simulated_summary.csv"
    daily_path = out_dir / "v59_daily_report.csv"

    events.to_csv(events_path, index=False)
    equity.to_csv(equity_path, index=False)
    summary.to_csv(summary_path, index=False)
    daily.to_csv(daily_path, index=False)

    print("\n=== v59 summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if not daily.empty:
        print("\n=== v59 daily report ===")
        print(daily.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved events: {events_path}")
    print(f"Saved equity: {equity_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved daily report: {daily_path}")

    print("\nInterpretation hints:")
    print("- Net PnL includes simulated IBKR fixed commissions and sell-side regulatory fees.")
    print("- Real paper/live execution should later replace simulated fills and estimated fees with IBKR execution/commission reports.")
    print("- If commissions dominate, increase position size, reduce trade count, or require stronger signals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
