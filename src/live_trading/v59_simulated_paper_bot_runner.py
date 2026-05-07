from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
    trailing_stop_price: float | None = None
    max_price: float = 0.0
    reason: str = ""
    score: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.quantity * (price - self.entry_price)


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def read_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    df.to_csv(path, mode="a", index=False, header=not exists)


def prepare_intents(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0)
    out["reference_price"] = pd.to_numeric(out.get("reference_price"), errors="coerce")
    out["quantity"] = pd.to_numeric(out.get("quantity"), errors="coerce").fillna(0).astype(int)
    out["position_usd"] = pd.to_numeric(out.get("position_usd"), errors="coerce").fillna(0.0)
    out = out.dropna(subset=["timestamp_utc", "reference_price"])
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
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def current_price_from_snapshot(row: pd.Series) -> float | None:
    for col in ["reference_price", "last", "mid", "ask", "bid"]:
        val = row.get(col)
        try:
            if pd.notna(val) and float(val) > 0:
                return float(val)
        except Exception:
            pass
    return None


def simulate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intents = prepare_intents(read_csv_if_exists(args.intents_csv))
    snapshots = prepare_snapshots(read_csv_if_exists(args.snapshots_csv))

    if snapshots.empty:
        raise RuntimeError("No snapshots found. Run v58_live_signal_engine first.")

    if intents.empty:
        print("No order intents found. Runner will only produce empty summary.")

    cash = float(args.starting_cash)
    realized_pnl = 0.0
    open_positions: dict[str, SimPosition] = {}
    trade_events: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    intents_by_time = intents.groupby("timestamp_utc") if not intents.empty else None
    intent_idx = 0
    intents_sorted = intents.to_dict("records") if not intents.empty else []

    daily_loss_triggered = False

    for _, snap in snapshots.iterrows():
        ts = snap["timestamp_utc"]
        symbol = str(snap["symbol"])
        price = current_price_from_snapshot(snap)
        if price is None:
            continue

        # Manage exits for this symbol.
        if symbol in open_positions:
            pos = open_positions[symbol]
            pos.max_price = max(pos.max_price, price)
            max_gain_pct = (pos.max_price / pos.entry_price - 1.0) * 100.0

            if max_gain_pct >= args.trailing_activation_pct:
                trail_price = pos.max_price * (1.0 - args.trailing_stop_pct / 100.0)
                pos.trailing_stop_price = max(pos.trailing_stop_price or 0.0, trail_price)

            exit_reason = None
            exit_price = None
            if price <= pos.stop_price:
                exit_reason = "stop_loss"
                exit_price = price
            elif pos.trailing_stop_price is not None and price <= pos.trailing_stop_price:
                exit_reason = "trailing_stop"
                exit_price = price

            if exit_reason is not None:
                pnl = pos.quantity * (exit_price - pos.entry_price)
                cash += pos.quantity * exit_price
                realized_pnl += pnl
                trade_events.append({
                    "timestamp_utc": ts.isoformat(),
                    "event": "EXIT",
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "pnl_usd": pnl,
                    "pnl_pct": (exit_price / pos.entry_price - 1.0) * 100.0,
                    "reason": exit_reason,
                    "cash_after": cash,
                    "score": pos.score,
                })
                del open_positions[symbol]

        # Process intents up to current timestamp.
        while intent_idx < len(intents_sorted) and intents_sorted[intent_idx]["timestamp_utc"] <= ts:
            intent = intents_sorted[intent_idx]
            intent_idx += 1
            i_symbol = str(intent.get("symbol"))
            if i_symbol != symbol:
                # Only fill intent when we have a current snapshot for that symbol.
                continue
            if daily_loss_triggered:
                continue
            if i_symbol in open_positions:
                continue
            if len(open_positions) >= args.max_positions:
                continue

            ref_price = float(intent.get("reference_price") or price)
            entry_price = price * (1.0 + args.simulated_entry_slippage_bps / 10_000.0)
            quantity = int(intent.get("quantity") or 0)
            if quantity <= 0:
                quantity = int(float(intent.get("position_usd") or args.base_position_usd) // entry_price)
            if quantity <= 0:
                continue

            notional = quantity * entry_price
            open_exposure = sum(p.quantity * p.entry_price for p in open_positions.values())
            if notional > args.max_position_usd:
                quantity = int(args.max_position_usd // entry_price)
                notional = quantity * entry_price
            if notional < args.min_position_usd or quantity <= 0:
                continue
            if open_exposure + notional > args.max_gross_exposure:
                continue
            if cash < notional:
                continue

            cash -= notional
            stop_price = entry_price * (1.0 - args.stop_loss_pct / 100.0)
            score = float(intent.get("score") or 0.0)
            open_positions[i_symbol] = SimPosition(
                symbol=i_symbol,
                entry_time=ts,
                entry_price=entry_price,
                quantity=quantity,
                position_usd=notional,
                stop_price=stop_price,
                max_price=entry_price,
                reason=str(intent.get("reason") or ""),
                score=score,
            )
            trade_events.append({
                "timestamp_utc": ts.isoformat(),
                "event": "ENTRY",
                "symbol": i_symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": None,
                "pnl_usd": None,
                "pnl_pct": None,
                "reason": str(intent.get("reason") or "BUY_INTENT"),
                "cash_after": cash,
                "score": score,
            })

        open_value = 0.0
        open_unrealized = 0.0
        # Approximate equity using last seen price only for current symbol; entry price for others.
        for p_symbol, pos in open_positions.items():
            p_price = price if p_symbol == symbol else pos.entry_price
            open_value += pos.quantity * p_price
            open_unrealized += pos.unrealized_pnl(p_price)

        equity = cash + open_value
        day_pnl = realized_pnl + open_unrealized
        if day_pnl <= -abs(args.max_daily_loss_usd):
            daily_loss_triggered = True

        equity_rows.append({
            "timestamp_utc": ts.isoformat(),
            "cash": cash,
            "open_positions": len(open_positions),
            "open_value_usd": open_value,
            "open_unrealized_pnl_usd": open_unrealized,
            "realized_pnl_usd": realized_pnl,
            "equity_usd": equity,
            "daily_loss_triggered": daily_loss_triggered,
        })

    # End-of-run flatten using last available price per symbol.
    last_prices = {}
    for _, row in snapshots.iterrows():
        price = current_price_from_snapshot(row)
        if price is not None:
            last_prices[str(row["symbol"])] = (row["timestamp_utc"], price)

    for symbol, pos in list(open_positions.items()):
        ts, price = last_prices.get(symbol, (snapshots.iloc[-1]["timestamp_utc"], pos.entry_price))
        exit_price = price * (1.0 - args.simulated_exit_slippage_bps / 10_000.0)
        pnl = pos.quantity * (exit_price - pos.entry_price)
        cash += pos.quantity * exit_price
        realized_pnl += pnl
        trade_events.append({
            "timestamp_utc": ts.isoformat(),
            "event": "EXIT",
            "symbol": symbol,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_usd": pnl,
            "pnl_pct": (exit_price / pos.entry_price - 1.0) * 100.0,
            "reason": "end_of_run_flatten",
            "cash_after": cash,
            "score": pos.score,
        })
        del open_positions[symbol]

    events_df = pd.DataFrame(trade_events)
    equity_df = pd.DataFrame(equity_rows)

    if events_df.empty:
        summary = pd.DataFrame([{
            "strategy": "v59_simulated_paper_bot",
            "entries": 0,
            "exits": 0,
            "realized_pnl_usd": 0.0,
            "ending_cash": cash,
            "return_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_open_positions": 0,
        }])
    else:
        exits = events_df[events_df["event"] == "EXIT"].copy()
        pnl = pd.to_numeric(exits.get("pnl_usd"), errors="coerce").fillna(0.0) if not exits.empty else pd.Series(dtype=float)
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
            "realized_pnl_usd": float(pnl.sum()) if not pnl.empty else 0.0,
            "ending_cash": float(cash),
            "return_pct": float((cash / args.starting_cash - 1.0) * 100.0),
            "win_rate": float((pnl > 0).mean() * 100.0) if not pnl.empty else 0.0,
            "avg_trade_pnl_usd": float(pnl.mean()) if not pnl.empty else 0.0,
            "max_drawdown_usd": max_dd,
            "max_open_positions": max_open,
        }])

    return events_df, equity_df, summary


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
    args = parser.parse_args()

    print("=== v59 simulated paper bot runner ===")
    print(f"Intents: {args.intents_csv}")
    print(f"Snapshots: {args.snapshots_csv}")
    print(f"Starting cash: ${args.starting_cash:.2f}")

    events, equity, summary = simulate(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "v59_simulated_trade_events.csv"
    equity_path = out_dir / "v59_simulated_equity_curve.csv"
    summary_path = out_dir / "v59_simulated_summary.csv"

    events.to_csv(events_path, index=False)
    equity.to_csv(equity_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== v59 summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved events: {events_path}")
    print(f"Saved equity: {equity_path}")
    print(f"Saved summary: {summary_path}")

    print("\nInterpretation hints:")
    print("- If there are zero entries, run v58_live_signal_engine during active market hours or loosen thresholds.")
    print("- This module validates portfolio/risk lifecycle before broker execution is connected.")
    print("- Once live market data is active, compare this simulated lifecycle with real paper fills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
