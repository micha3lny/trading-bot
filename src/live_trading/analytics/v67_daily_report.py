from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MARKET_OPEN_UTC = "13:30"


def f(x, default=0.0):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def parse_dt(value: str | None):
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def utc_hhmm(value: str | None) -> str:
    dt = parse_dt(value)
    return dt.strftime("%H:%M") if dt else ""


def minutes_from_market_open(value: str | None, market_open_utc: str = MARKET_OPEN_UTC) -> int | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    try:
        hh, mm = [int(x) for x in market_open_utc.split(":", 1)]
        open_dt = dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return int((dt - open_dt).total_seconds() // 60)
    except Exception:
        return None


def buy_time_bucket(value: str | None) -> str:
    mins = minutes_from_market_open(value)
    if mins is None:
        return "unknown"
    if mins < 0:
        return "pre-open"
    if mins < 30:
        return "13:30-14:00"
    if mins < 90:
        return "14:00-15:00"
    if mins < 150:
        return "15:00-16:00"
    if mins < 210:
        return "16:00-17:00"
    if mins < 270:
        return "17:00-18:00"
    if mins < 330:
        return "18:00-19:00"
    return "19:00+"


def load_latest_portfolio(session: Path):
    p = session / "portfolio_snapshots.csv"
    if not p.exists():
        return {}
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    row = rows[-1]
    out = {}
    try:
        for pos in json.loads(row.get("positions_json") or "[]"):
            sym = str(pos.get("symbol", "")).upper()
            out[sym] = pos
    except Exception:
        pass
    return out


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

    latest_portfolio = load_latest_portfolio(session)

    buys = {}
    closed = []

    with lifecycle.open() as fh:
        for r in csv.DictReader(fh):
            event = r.get("event")
            sym = str(r.get("symbol", "")).upper()
            if not sym:
                continue

            if event == "BUY_ORDER_SENT":
                buys.setdefault(sym, []).append(r)

            elif event == "SELL_ORDER_SENT":
                if buys.get(sym):
                    b = buys[sym].pop(0)

                    qty = f(r.get("quantity"))
                    buy_px = f(b.get("price") or b.get("entry_price"))
                    sell_px = f(r.get("price"))
                    peak = f(r.get("peak_price"), sell_px)
                    buy_time = b.get("recorded_at")

                    gross = (sell_px - buy_px) * qty
                    net = gross - args.commission_per_roundtrip
                    pnl_pct = ((sell_px / buy_px) - 1) * 100 if buy_px else 0
                    peak_pct = ((peak / buy_px) - 1) * 100 if buy_px else 0
                    giveback_pct = ((sell_px / peak) - 1) * 100 if peak else 0

                    closed.append({
                        "symbol": sym,
                        "qty": qty,
                        "buy_time": buy_time,
                        "buy_utc": utc_hhmm(buy_time),
                        "buy_bucket": buy_time_bucket(buy_time),
                        "minutes_from_open": minutes_from_market_open(buy_time),
                        "sell_time": r.get("recorded_at"),
                        "buy": buy_px,
                        "sell": sell_px,
                        "peak": peak,
                        "gross": gross,
                        "net": net,
                        "pnl_pct": pnl_pct,
                        "peak_pct": peak_pct,
                        "giveback_pct": giveback_pct,
                        "reason": r.get("reason"),
                    })

    open_positions = []
    for sym, rows in buys.items():
        for b in rows:
            qty = f(b.get("quantity"))
            buy_px = f(b.get("price") or b.get("entry_price"))
            buy_time = b.get("recorded_at")
            pf = latest_portfolio.get(sym, {})
            current = f(pf.get("marketPrice"), 0)
            market_value = f(pf.get("marketValue"), current * qty)
            unrealized = f(pf.get("unrealizedPNL"), (current - buy_px) * qty if current else 0)
            current_pct = ((current / buy_px) - 1) * 100 if buy_px and current else 0

            peak = f(b.get("peak_price"), buy_px)
            # fallback: if lifecycle has no current peak, current may be above entry
            peak = max(peak, buy_px, current)
            peak_pct = ((peak / buy_px) - 1) * 100 if buy_px else 0
            from_peak_pct = ((current / peak) - 1) * 100 if peak and current else 0

            open_positions.append({
                "symbol": sym,
                "qty": qty,
                "buy_time": buy_time,
                "buy_utc": utc_hhmm(buy_time),
                "buy_bucket": buy_time_bucket(buy_time),
                "minutes_from_open": minutes_from_market_open(buy_time),
                "buy": buy_px,
                "current": current,
                "peak": peak,
                "market_value": market_value,
                "unrealized": unrealized,
                "current_pct": current_pct,
                "peak_pct": peak_pct,
                "from_peak_pct": from_peak_pct,
            })

    wins = [x for x in closed if x["gross"] > 0]
    losses = [x for x in closed if x["gross"] <= 0]

    gross_total = sum(x["gross"] for x in closed)
    net_total = sum(x["net"] for x in closed)
    open_unrealized = sum(x["unrealized"] for x in open_positions)
    total_est = net_total + open_unrealized

    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_win = sum(x["gross"] for x in wins) / len(wins) if wins else 0
    avg_loss = sum(x["gross"] for x in losses) / len(losses) if losses else 0
    expectancy = gross_total / len(closed) if closed else 0

    print(f"=== v67 Daily Report {args.date} ===")
    print(f"Closed trades:        {len(closed)}")
    print(f"Open trades:          {len(open_positions)}")
    print(f"Win rate:             {win_rate:.1f}%")
    print(f"Gross closed PnL:     ${gross_total:.2f}")
    print(f"Net closed est PnL:   ${net_total:.2f}")
    print(f"Open unrealized PnL:  ${open_unrealized:.2f}")
    print(f"Total est PnL:        ${total_est:.2f}")
    print(f"Avg win:              ${avg_win:.2f}")
    print(f"Avg loss:             ${avg_loss:.2f}")
    print(f"Expectancy:           ${expectancy:.2f}/trade")
    print()

    print("=== PnL by buy time bucket ===")
    bucket_stats = defaultdict(lambda: {"closed": 0, "open": 0, "wins": 0, "closed_gross": 0.0, "closed_net": 0.0, "open_upnl": 0.0})
    for x in closed:
        b = x.get("buy_bucket") or "unknown"
        bucket_stats[b]["closed"] += 1
        bucket_stats[b]["closed_gross"] += x["gross"]
        bucket_stats[b]["closed_net"] += x["net"]
        if x["gross"] > 0:
            bucket_stats[b]["wins"] += 1
    for x in open_positions:
        b = x.get("buy_bucket") or "unknown"
        bucket_stats[b]["open"] += 1
        bucket_stats[b]["open_upnl"] += x["unrealized"]

    preferred_order = ["pre-open", "13:30-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00", "17:00-18:00", "18:00-19:00", "19:00+", "unknown"]
    print(f"{'BUCKET':<13} {'CLOSED':>7} {'OPEN':>5} {'WIN%':>7} {'GROSS':>10} {'NET':>10} {'OPEN_UPNL':>11} {'TOTAL_EST':>11}")
    print("-" * 86)
    for bucket in preferred_order:
        s = bucket_stats.get(bucket)
        if not s:
            continue
        closed_n = s["closed"]
        win_pct = (s["wins"] / closed_n * 100) if closed_n else 0.0
        total = s["closed_net"] + s["open_upnl"]
        print(
            f"{bucket:<13} "
            f"{closed_n:>7} "
            f"{s['open']:>5} "
            f"{win_pct:>6.1f}% "
            f"{s['closed_gross']:>10.2f} "
            f"{s['closed_net']:>10.2f} "
            f"{s['open_upnl']:>11.2f} "
            f"{total:>11.2f}"
        )
    print()

    print("=== Closed trades ===")
    print(f"{'SYM':<7} {'UTC':>5} {'MIN':>5} {'BUCKET':<13} {'QTY':>5} {'BUY':>10} {'SELL':>10} {'PEAK':>10} {'GROSS':>10} {'NET':>10} {'PNL%':>8} {'PEAK%':>8} {'DROP':>8}  REASON")
    print("-" * 150)

    for x in sorted(closed, key=lambda r: r["gross"]):
        mins = x.get("minutes_from_open")
        mins_txt = "" if mins is None else str(mins)
        print(
            f"{x['symbol']:<7} "
            f"{x.get('buy_utc',''):>5} "
            f"{mins_txt:>5} "
            f"{x.get('buy_bucket',''):<13} "
            f"{x['qty']:>5.0f} "
            f"{x['buy']:>10.4f} "
            f"{x['sell']:>10.4f} "
            f"{x['peak']:>10.4f} "
            f"{x['gross']:>10.2f} "
            f"{x['net']:>10.2f} "
            f"{x['pnl_pct']:>7.2f}% "
            f"{x['peak_pct']:>7.2f}% "
            f"{x['giveback_pct']:>7.2f}%  "
            f"{x['reason']}"
        )

    print()
    print("=== Open trades from today ===")
    print(f"{'SYM':<7} {'UTC':>5} {'MIN':>5} {'BUCKET':<13} {'QTY':>5} {'BUY':>10} {'NOW':>10} {'PEAK':>10} {'UPNL':>10} {'NOW%':>8} {'PEAK%':>8} {'FROM_PK':>9}  BUY_TIME")
    print("-" * 145)

    for x in sorted(open_positions, key=lambda r: r["unrealized"]):
        mins = x.get("minutes_from_open")
        mins_txt = "" if mins is None else str(mins)
        print(
            f"{x['symbol']:<7} "
            f"{x.get('buy_utc',''):>5} "
            f"{mins_txt:>5} "
            f"{x.get('buy_bucket',''):<13} "
            f"{x['qty']:>5.0f} "
            f"{x['buy']:>10.4f} "
            f"{x['current']:>10.4f} "
            f"{x['peak']:>10.4f} "
            f"{x['unrealized']:>10.2f} "
            f"{x['current_pct']:>7.2f}% "
            f"{x['peak_pct']:>7.2f}% "
            f"{x['from_peak_pct']:>8.2f}%  "
            f"{x['buy_time']}"
        )


if __name__ == "__main__":
    main()
