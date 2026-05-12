from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def f(x, default=0.0):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%F"))
    ap.add_argument("--recorder-dir", default="data/live/recorder")
    ap.add_argument("--commission-per-roundtrip", type=float, default=1.0)
    args = ap.parse_args()

    session = Path(args.recorder_dir) / args.date
    lifecycle = session / "trade_lifecycle.csv"

    if not lifecycle.exists():
        raise SystemExit(f"Missing {lifecycle}")

    buys = {}
    closed = []
    events = []

    with lifecycle.open() as fh:
        for r in csv.DictReader(fh):
            event = r.get("event")
            sym = r.get("symbol")
            if not sym:
                continue

            if event == "BUY_ORDER_SENT":
                buys.setdefault(sym, []).append(r)
                events.append(r)

            elif event == "SELL_ORDER_SENT":
                events.append(r)
                if buys.get(sym):
                    b = buys[sym].pop(0)

                    qty = f(r.get("quantity"))
                    buy_px = f(b.get("price") or b.get("entry_price"))
                    sell_px = f(r.get("price"))
                    gross = (sell_px - buy_px) * qty
                    net = gross - args.commission_per_roundtrip
                    pnl_pct = ((sell_px / buy_px) - 1) * 100 if buy_px else 0

                    closed.append({
                        "symbol": sym,
                        "qty": qty,
                        "buy_time": b.get("recorded_at"),
                        "sell_time": r.get("recorded_at"),
                        "buy": buy_px,
                        "sell": sell_px,
                        "gross": gross,
                        "net": net,
                        "pnl_pct": pnl_pct,
                        "reason": r.get("reason"),
                    })

    open_positions = []
    for sym, rows in buys.items():
        for b in rows:
            open_positions.append({
                "symbol": sym,
                "qty": f(b.get("quantity")),
                "buy_time": b.get("recorded_at"),
                "buy": f(b.get("price") or b.get("entry_price")),
            })

    wins = [x for x in closed if x["gross"] > 0]
    losses = [x for x in closed if x["gross"] <= 0]

    gross_total = sum(x["gross"] for x in closed)
    net_total = sum(x["net"] for x in closed)
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_win = sum(x["gross"] for x in wins) / len(wins) if wins else 0
    avg_loss = sum(x["gross"] for x in losses) / len(losses) if losses else 0
    expectancy = gross_total / len(closed) if closed else 0

    print(f"=== v67 Daily Report {args.date} ===")
    print(f"Closed trades: {len(closed)}")
    print(f"Open trades:   {len(open_positions)}")
    print(f"Win rate:      {win_rate:.1f}%")
    print(f"Gross PnL:     ${gross_total:.2f}")
    print(f"Net est PnL:   ${net_total:.2f}")
    print(f"Avg win:       ${avg_win:.2f}")
    print(f"Avg loss:      ${avg_loss:.2f}")
    print(f"Expectancy:    ${expectancy:.2f}/trade")
    print()

    print("=== Closed trades ===")
    print(f"{'SYM':<7} {'QTY':>5} {'BUY':>10} {'SELL':>10} {'GROSS':>10} {'NET':>10} {'PNL%':>8}  REASON")
    print("-" * 95)

    for x in sorted(closed, key=lambda r: r["gross"]):
        print(
            f"{x['symbol']:<7} "
            f"{x['qty']:>5.0f} "
            f"{x['buy']:>10.4f} "
            f"{x['sell']:>10.4f} "
            f"{x['gross']:>10.2f} "
            f"{x['net']:>10.2f} "
            f"{x['pnl_pct']:>7.2f}%  "
            f"{x['reason']}"
        )

    print()
    print("=== Open trades from today ===")
    print(f"{'SYM':<7} {'QTY':>5} {'BUY':>10}  BUY_TIME")
    print("-" * 60)

    for x in sorted(open_positions, key=lambda r: r["symbol"]):
        print(f"{x['symbol']:<7} {x['qty']:>5.0f} {x['buy']:>10.4f}  {x['buy_time']}")


if __name__ == "__main__":
    main()
