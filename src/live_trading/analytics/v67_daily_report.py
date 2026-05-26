from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MARKET_OPEN_UTC = "13:30"
STRICT_SETUP_NAME = "v67_original_600usd_setup"
STRICT_MIN_FIRST_5M_HIGH_PCT = 4.0
STRICT_MIN_FIRST_15M_HIGH_PCT = 6.5
STRICT_MIN_OR_RANGE_PCT = 5.0
STRICT_MAX_SPREAD_BPS = 50.0
STRICT_MIN_PRICE = 5.0


def f(x, default=0.0):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def b(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def raw_json_dict(row: dict) -> dict:
    try:
        raw = row.get("raw_json") or "{}"
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)
    except Exception:
        return {}


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


def is_strict_payload(payload: dict, row: dict | None = None) -> bool:
    row = row or {}
    if b(row.get("strict_setup_ready")) or b(payload.get("strict_setup_ready")):
        return True
    first_5m = f(row.get("first_5m_high_pct"), None)
    first_15m = f(row.get("first_15m_high_pct"), None)
    or_range = f(row.get("or_range_pct"), None)
    spread_pct = f(row.get("spread_pct"), None)
    spread_bps = spread_pct * 100.0 if spread_pct is not None else None
    if first_5m is None:
        first_5m = f(payload.get("first_5m_high_pct"), None)
    if first_15m is None:
        first_15m = f(payload.get("first_15m_high_pct"), None)
    if or_range is None:
        or_range = f(payload.get("or_range_pct"), None)
    if spread_bps is None:
        spread_bps = f(payload.get("spread_bps"), None)
    price = f(payload.get("entry_price"), f(row.get("price"), 0.0))
    return (
        first_5m is not None and first_5m >= STRICT_MIN_FIRST_5M_HIGH_PCT
        and first_15m is not None and first_15m >= STRICT_MIN_FIRST_15M_HIGH_PCT
        and or_range is not None and or_range >= STRICT_MIN_OR_RANGE_PCT
        and price >= STRICT_MIN_PRICE
        and (spread_bps is None or spread_bps <= STRICT_MAX_SPREAD_BPS)
    )


def load_latest_portfolio(session: Path):
    p = session / "portfolio_snapshots.csv"
    if not p.exists():
        return {}
    with p.open(errors="replace") as fh:
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


def load_json_file(path: Path, default):
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_eod_summary(session: Path) -> dict:
    return load_json_file(session / "eod_summary.json", {})


def load_managed_positions_summary(session: Path) -> dict:
    data = load_json_file(session / "managed_positions.json", {})
    if isinstance(data, dict):
        rows = data.get("positions", data)
    else:
        rows = data
    if isinstance(rows, dict):
        values = rows.values()
    elif isinstance(rows, list):
        values = rows
    else:
        values = []
    active = []
    for row in values:
        if not isinstance(row, dict):
            continue
        if row.get("active", True):
            active.append(row)
    return {"active_count": len(active), "active_symbols": sorted(str(x.get("symbol", "")).upper() for x in active if x.get("symbol"))}


def load_signal_features(session: Path) -> dict[str, list[dict]]:
    lifecycle = session / "trade_lifecycle.csv"
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    if not lifecycle.exists():
        return by_symbol
    with lifecycle.open(errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("event") != "SIGNAL_READY":
                continue
            sym = str(row.get("symbol", "")).upper()
            payload = raw_json_dict(row)
            payload.update({
                "recorded_at": row.get("recorded_at"),
                "strict_setup_ready": row.get("strict_setup_ready") or payload.get("strict_setup_ready"),
                "first_5m_high_pct": row.get("first_5m_high_pct") or payload.get("first_5m_high_pct"),
                "first_15m_high_pct": row.get("first_15m_high_pct") or payload.get("first_15m_high_pct"),
                "or_range_pct": row.get("or_range_pct") or payload.get("or_range_pct"),
                "score": row.get("score") or payload.get("score"),
            })
            if sym:
                by_symbol[sym].append(payload)
    return by_symbol


def best_signal_payload_for_buy(signal_features: dict[str, list[dict]], symbol: str, buy_time: str | None) -> dict:
    signals = signal_features.get(symbol, [])
    if not signals:
        return {}
    buy_dt = parse_dt(buy_time)
    if buy_dt is None:
        return signals[-1]
    best = {}
    best_delta = None
    for sig in signals:
        sig_dt = parse_dt(sig.get("recorded_at"))
        if sig_dt is None:
            continue
        delta = abs((buy_dt - sig_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = sig
    return best or signals[-1]


def avg(items, key):
    vals = [key(x) for x in items]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def load_fills(session: Path) -> list[dict]:
    p = session / "fills.csv"
    if not p.exists():
        return []
    with p.open(errors="replace") as fh:
        return list(csv.DictReader(fh))


def normalized_fill_action(action: str | None) -> str:
    value = str(action or "").strip().upper()
    if value in {"BOT", "BUY"}:
        return "BUY"
    if value in {"SLD", "SELL"}:
        return "SELL"
    return value


def actual_commission(row: dict) -> float | None:
    if str(row.get("commission_source") or "").strip().lower() != "ibkr":
        return None
    commission = f(row.get("commission"), None)
    return commission if commission is not None else None


def reconstruct_closed_trades_from_fills(
    fills: list[dict],
    commission_per_roundtrip: float = 1.0,
) -> list[dict]:
    """Reconstruct long-side closed trades from execution fills using FIFO lots.

    Commission accounting is conservative: known IBKR commissions are used when
    present, and only the missing side(s) receive the estimated fallback.
    """
    per_side_estimate = commission_per_roundtrip / 2.0
    open_lots: dict[str, list[dict]] = defaultdict(list)
    closed: list[dict] = []

    def sort_key(row: dict) -> tuple[str, str]:
        raw = raw_json_dict(row)
        execution = raw.get("execution") if isinstance(raw, dict) else {}
        return (
            str(row.get("recorded_at") or execution.get("time") or ""),
            str(row.get("execution_id") or ""),
        )

    for row in sorted(fills, key=sort_key):
        symbol = str(row.get("symbol") or "").upper()
        action = normalized_fill_action(row.get("action"))
        qty = f(row.get("quantity"), 0.0)
        price = f(row.get("fill_price"), 0.0)
        if not symbol or qty <= 0 or price <= 0:
            continue

        commission = actual_commission(row)
        if action == "BUY":
            open_lots[symbol].append({
                "execution_id": row.get("execution_id"),
                "remaining_qty": qty,
                "original_qty": qty,
                "price": price,
                "recorded_at": row.get("recorded_at"),
                "commission": commission,
                "commission_source": row.get("commission_source") or "missing",
            })
            continue

        if action != "SELL":
            continue

        remaining_sell_qty = qty
        sell_commission = commission
        sell_source = row.get("commission_source") or "missing"
        while remaining_sell_qty > 0 and open_lots[symbol]:
            lot = open_lots[symbol][0]
            matched_qty = min(remaining_sell_qty, lot["remaining_qty"])
            buy_fraction = matched_qty / lot["original_qty"] if lot["original_qty"] else 0.0
            sell_fraction = matched_qty / qty if qty else 0.0

            buy_actual = (lot["commission"] * buy_fraction) if lot["commission"] is not None else 0.0
            sell_actual = (sell_commission * sell_fraction) if sell_commission is not None else 0.0
            buy_fallback = 0.0 if lot["commission"] is not None else per_side_estimate * buy_fraction
            sell_fallback = 0.0 if sell_commission is not None else per_side_estimate * sell_fraction
            gross = (price - lot["price"]) * matched_qty
            actual_total = buy_actual + sell_actual
            fallback_total = buy_fallback + sell_fallback
            all_actual = lot["commission"] is not None and sell_commission is not None

            closed.append({
                "symbol": symbol,
                "qty": matched_qty,
                "buy_execution_id": lot.get("execution_id") or "",
                "sell_execution_id": row.get("execution_id") or "",
                "buy_time": lot.get("recorded_at") or "",
                "sell_time": row.get("recorded_at") or "",
                "buy": lot["price"],
                "sell": price,
                "gross": gross,
                "actual_commission": actual_total,
                "estimated_commission_fallback": fallback_total,
                "estimated_commission": actual_total + fallback_total,
                "net_actual": gross - actual_total,
                "net_estimated": gross - actual_total - fallback_total,
                "commission_source": "ibkr" if all_actual else "estimated",
                "buy_commission_source": lot.get("commission_source") or "missing",
                "sell_commission_source": sell_source,
            })

            lot["remaining_qty"] -= matched_qty
            remaining_sell_qty -= matched_qty
            if lot["remaining_qty"] <= 1e-9:
                open_lots[symbol].pop(0)

    return closed


def effective_commission_per_trade(fill_closed: list[dict], fallback: float) -> float:
    if not fill_closed:
        return fallback
    total = sum(x.get("estimated_commission", fallback) for x in fill_closed)
    return total / len(fill_closed) if fill_closed else fallback


def simulate_exit_strategies(closed: list[dict], commission_per_trade: float) -> list[dict]:
    scenarios = [
        ("actual trailing", None, "actual"),
        ("fixed TP +2%", 2.0, "fixed"),
        ("fixed TP +3%", 3.0, "fixed"),
        ("fixed TP +4%", 4.0, "fixed"),
        ("fixed TP +5%", 5.0, "fixed"),
        ("partial 50%@+3%", 3.0, "partial"),
    ]
    results: list[dict] = []
    for name, target_pct, mode in scenarios:
        gross_total = 0.0
        captured = 0
        for trade in closed:
            qty = f(trade.get("qty"), 0.0)
            buy = f(trade.get("buy"), 0.0)
            sell = f(trade.get("sell"), 0.0)
            mfe = f(trade.get("peak_gain_pct"), 0.0)
            actual_gross = f(trade.get("gross"), (sell - buy) * qty)
            if mode == "actual" or not target_pct or buy <= 0 or qty <= 0:
                gross = actual_gross
            elif mfe >= target_pct:
                target_price = buy * (1.0 + target_pct / 100.0)
                if mode == "partial":
                    gross = ((target_price - buy) * qty * 0.5) + ((sell - buy) * qty * 0.5)
                else:
                    gross = (target_price - buy) * qty
                captured += 1
            else:
                gross = actual_gross
            gross_total += gross
        estimated_commission = len(closed) * commission_per_trade
        results.append({
            "name": name,
            "trades": len(closed),
            "captured": captured,
            "gross": gross_total,
            "net": gross_total - estimated_commission,
        })
    return results


def lifecycle_event_counts(session: Path) -> Counter:
    counts: Counter = Counter()
    lifecycle = session / "trade_lifecycle.csv"
    if not lifecycle.exists():
        return counts
    try:
        with lifecycle.open(errors="replace") as fh:
            for row in csv.DictReader(fh):
                event = str(row.get("event") or "").strip()
                if event:
                    counts[event] += 1
    except Exception:
        return counts
    return counts


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
    signal_features = load_signal_features(session)
    buys: dict[str, list[dict]] = {}
    closed = []

    with lifecycle.open(errors="replace") as fh:
        for r in csv.DictReader(fh):
            event = r.get("event")
            sym = str(r.get("symbol", "")).upper()
            if not sym:
                continue
            if event == "BUY_ORDER_SENT":
                buys.setdefault(sym, []).append(r)
            elif event == "SELL_ORDER_SENT" and buys.get(sym):
                b_row = buys[sym].pop(0)
                buy_px = f(b_row.get("fill_price") or b_row.get("price") or b_row.get("entry_price"))
                sell_px = f(r.get("fill_price") or r.get("price"))
                qty = f(r.get("quantity"))
                buy_time = b_row.get("recorded_at")
                payload = raw_json_dict(b_row) or best_signal_payload_for_buy(signal_features, sym, buy_time)
                strict = is_strict_payload(payload, b_row)
                peak = f(r.get("peak_price"), max(buy_px, sell_px))
                peak_gain_pct = f(r.get("peak_gain_pct"), ((peak / buy_px - 1) * 100 if buy_px else 0))
                giveback_pct = f(r.get("giveback_pct"), ((sell_px / peak - 1) * 100 if peak else 0))
                gross = (sell_px - buy_px) * qty
                net = gross - args.commission_per_roundtrip
                closed.append({
                    "symbol": sym,
                    "qty": qty,
                    "buy_time": buy_time,
                    "buy_utc": utc_hhmm(buy_time),
                    "buy_bucket": buy_time_bucket(buy_time),
                    "minutes_from_open": minutes_from_market_open(buy_time),
                    "strict_setup_ready": strict,
                    "buy": buy_px,
                    "sell": sell_px,
                    "peak": peak,
                    "gross": gross,
                    "net": net,
                    "pnl_pct": ((sell_px / buy_px - 1) * 100 if buy_px else 0),
                    "peak_gain_pct": peak_gain_pct,
                    "giveback_pct": giveback_pct,
                    "reason": r.get("reason"),
                })

    open_positions = []
    for sym, rows in buys.items():
        for row in rows:
            buy_px = f(row.get("fill_price") or row.get("price") or row.get("entry_price"))
            qty = f(row.get("quantity"))
            buy_time = row.get("recorded_at")
            payload = raw_json_dict(row) or best_signal_payload_for_buy(signal_features, sym, buy_time)
            strict = is_strict_payload(payload, row)
            pf = latest_portfolio.get(sym, {})
            current = f(pf.get("marketPrice"), 0)
            unrealized = f(pf.get("unrealizedPNL"), (current - buy_px) * qty if current else 0)
            peak = max(f(row.get("peak_price"), buy_px), buy_px, current)
            open_positions.append({
                "symbol": sym,
                "qty": qty,
                "buy_time": buy_time,
                "buy_utc": utc_hhmm(buy_time),
                "buy_bucket": buy_time_bucket(buy_time),
                "minutes_from_open": minutes_from_market_open(buy_time),
                "strict_setup_ready": strict,
                "buy": buy_px,
                "current": current,
                "peak": peak,
                "unrealized": unrealized,
                "current_pct": ((current / buy_px - 1) * 100 if buy_px and current else 0),
                "peak_gain_pct": ((peak / buy_px - 1) * 100 if buy_px else 0),
                "from_peak_pct": ((current / peak - 1) * 100 if peak and current else 0),
            })

    wins = [x for x in closed if x["gross"] > 0]
    losses = [x for x in closed if x["gross"] <= 0]
    gross_total = sum(x["gross"] for x in closed)
    net_total = sum(x["net"] for x in closed)
    open_upnl = sum(x["unrealized"] for x in open_positions)
    strict_closed = [x for x in closed if x["strict_setup_ready"]]
    strict_open = [x for x in open_positions if x["strict_setup_ready"]]
    fill_rows = load_fills(session)
    fill_closed = reconstruct_closed_trades_from_fills(fill_rows, args.commission_per_roundtrip)
    fill_gross = sum(x["gross"] for x in fill_closed)
    fill_actual_commission = sum(x["actual_commission"] for x in fill_closed)
    fill_estimated_fallback = sum(x["estimated_commission_fallback"] for x in fill_closed)
    fill_net_actual = sum(x["net_actual"] for x in fill_closed)
    fill_net_estimated = sum(x["net_estimated"] for x in fill_closed)
    commission_per_trade = effective_commission_per_trade(fill_closed, args.commission_per_roundtrip)
    exit_simulations = simulate_exit_strategies(closed, commission_per_trade)
    fills_without_commission = [
        x for x in fill_rows
        if str(x.get("commission_source") or "").strip().lower() != "ibkr"
    ]
    eod_summary = load_eod_summary(session)
    managed_summary = load_managed_positions_summary(session)
    final_fractional_positions = [
        sym for sym, pos in latest_portfolio.items()
        if f(pos.get("position"), 0.0) != 0 and abs(f(pos.get("position"), 0.0) - round(f(pos.get("position"), 0.0))) > 1e-9
    ]
    final_whole_positions = [
        sym for sym, pos in latest_portfolio.items()
        if f(pos.get("position"), 0.0) != 0 and sym not in set(final_fractional_positions)
    ]
    diagnostic_fractional_orphans = (eod_summary.get("fractional_orphans") or []) if "fractional_orphans" in eod_summary else final_fractional_positions
    diagnostic_whole_share_orphans = (eod_summary.get("whole_share_orphans") or []) if "whole_share_orphans" in eod_summary else final_whole_positions
    event_counts = lifecycle_event_counts(session)

    print(f"=== v67 Daily Report {args.date} ===")
    print(f"Closed trades:        {len(closed)}")
    print(f"Open trades:          {len(open_positions)}")
    print(f"Win rate:             {(len(wins) / len(closed) * 100 if closed else 0):.1f}%")
    print(f"Gross closed PnL:     ${gross_total:.2f}")
    print(f"Net closed est PnL:   ${net_total:.2f}")
    print(f"Open unrealized PnL:  ${open_upnl:.2f}")
    print(f"Total est PnL:        ${net_total + open_upnl:.2f}")
    print(f"Avg win:              ${(sum(x['gross'] for x in wins) / len(wins) if wins else 0):.2f}")
    print(f"Avg loss:             ${(sum(x['gross'] for x in losses) / len(losses) if losses else 0):.2f}")
    print(f"Expectancy:           ${(gross_total / len(closed) if closed else 0):.2f}/trade")
    print()

    print("=== IBKR fill ledger ===")
    print(f"Fill rows:            {len(fill_rows)}")
    print(f"Fill closed trades:   {len(fill_closed)}")
    print(f"Gross fill PnL:       ${fill_gross:.2f}")
    print(f"IBKR commission:      ${fill_actual_commission:.2f}")
    print(f"Estimated fallback:   ${fill_estimated_fallback:.2f}")
    print(f"Net actual:           ${fill_net_actual:.2f}")
    print(f"Net estimated:        ${fill_net_estimated:.2f}")
    print(f"Fills without comm:   {len(fills_without_commission)}")
    print()

    if fill_closed:
        print("=== Closed trades from fills ===")
        print(f"{'SYM':<7} {'QTY':>8} {'BUY':>10} {'SELL':>10} {'GROSS':>10} {'IBKR_COMM':>10} {'EST_FB':>10} {'NET_ACT':>10} {'NET_EST':>10} {'SRC':>9}")
        print("-" * 109)
        for x in sorted(fill_closed, key=lambda r: r["gross"]):
            print(
                f"{x['symbol']:<7} {x['qty']:>8.2f} {x['buy']:>10.4f} {x['sell']:>10.4f} "
                f"{x['gross']:>10.2f} {x['actual_commission']:>10.2f} {x['estimated_commission_fallback']:>10.2f} "
                f"{x['net_actual']:>10.2f} {x['net_estimated']:>10.2f} {x['commission_source']:>9}"
            )
        print()

    print("=== Post-session diagnostics ===")
    print(f"Final IBKR positions: {len(latest_portfolio)}")
    print(f"Final managed count:  {managed_summary['active_count']}")
    print(f"Fractional orphans:   {len(diagnostic_fractional_orphans)}")
    print(f"Whole-share orphans:  {len(diagnostic_whole_share_orphans)}")
    print(f"Pending orders:       {eod_summary.get('pending_orders', '')}")
    clean_value = eod_summary.get("clean")
    print(f"EOD clean:            {'' if clean_value is None else int(bool(clean_value))}")
    print(f"Partial entries:      {event_counts.get('ENTRY_ORDER_PARTIAL', 0)}")
    print(f"Partial exits:        {event_counts.get('EXIT_ORDER_PARTIAL', 0)}")
    print(f"Delayed fill/cancel:  {event_counts.get('DELAYED_FILL_AFTER_CANCEL', 0)}")
    print(f"Cancel but position:  {event_counts.get('ORDER_CANCEL_BUT_POSITION_EXISTS', 0)}")
    if latest_portfolio:
        print(f"Final symbols:        {', '.join(sorted(latest_portfolio))}")
    print()

    print("=== Exit simulation only ===")
    print(f"{'SCENARIO':<18} {'TRADES':>7} {'CAPTURED':>8} {'GROSS':>10} {'NET_EST':>10}")
    print("-" * 58)
    for row in exit_simulations:
        print(f"{row['name']:<18} {row['trades']:>7} {row['captured']:>8} {row['gross']:>10.2f} {row['net']:>10.2f}")
    print()

    print("=== Strict/original setup subset ===")
    strict_wins = [x for x in strict_closed if x["gross"] > 0]
    print(f"Name:                 {STRICT_SETUP_NAME}")
    print(f"Thresholds:           5m>={STRICT_MIN_FIRST_5M_HIGH_PCT}%, 15m>={STRICT_MIN_FIRST_15M_HIGH_PCT}%, OR>={STRICT_MIN_OR_RANGE_PCT}%, spread<={STRICT_MAX_SPREAD_BPS}bps")
    print(f"Strict closed trades: {len(strict_closed)}")
    print(f"Strict open trades:   {len(strict_open)}")
    print(f"Strict win rate:      {(len(strict_wins) / len(strict_closed) * 100 if strict_closed else 0):.1f}%")
    print(f"Strict gross closed:  ${sum(x['gross'] for x in strict_closed):.2f}")
    print(f"Strict net closed:    ${sum(x['net'] for x in strict_closed):.2f}")
    print(f"Strict open UPNL:     ${sum(x['unrealized'] for x in strict_open):.2f}")
    print(f"Strict total est:     ${sum(x['net'] for x in strict_closed) + sum(x['unrealized'] for x in strict_open):.2f}")
    print()

    print("=== Peak / giveback analytics ===")
    groups = [("ALL", closed), ("STRICT", strict_closed), ("NONSTRICT", [x for x in closed if not x["strict_setup_ready"]])]
    print(f"{'GROUP':<10} {'TRADES':>7} {'AVG_MFE%':>10} {'AVG_EXIT%':>10} {'AVG_DROP%':>10} {'GROSS':>10} {'NET':>10}")
    print("-" * 76)
    for name, rows in groups:
        print(
            f"{name:<10} {len(rows):>7} "
            f"{avg(rows, lambda x: x['peak_gain_pct']):>10.2f} "
            f"{avg(rows, lambda x: x['pnl_pct']):>10.2f} "
            f"{avg(rows, lambda x: x['giveback_pct']):>10.2f} "
            f"{sum(x['gross'] for x in rows):>10.2f} "
            f"{sum(x['net'] for x in rows):>10.2f}"
        )
    print()

    print("=== PnL by buy time bucket ===")
    bucket_stats = defaultdict(lambda: {"closed": 0, "open": 0, "wins": 0, "gross": 0.0, "net": 0.0, "open_upnl": 0.0, "strict": 0})
    for x in closed:
        s = bucket_stats[x["buy_bucket"]]
        s["closed"] += 1
        s["gross"] += x["gross"]
        s["net"] += x["net"]
        s["wins"] += 1 if x["gross"] > 0 else 0
        s["strict"] += 1 if x["strict_setup_ready"] else 0
    for x in open_positions:
        s = bucket_stats[x["buy_bucket"]]
        s["open"] += 1
        s["open_upnl"] += x["unrealized"]
        s["strict"] += 1 if x["strict_setup_ready"] else 0
    order = ["pre-open", "13:30-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00", "17:00-18:00", "18:00-19:00", "19:00+", "unknown"]
    print(f"{'BUCKET':<13} {'CLOSED':>7} {'OPEN':>5} {'STRICT':>7} {'WIN%':>7} {'GROSS':>10} {'NET':>10} {'OPEN_UPNL':>11} {'TOTAL_EST':>11}")
    print("-" * 96)
    for bucket in order:
        s = bucket_stats.get(bucket)
        if not s:
            continue
        win_pct = s["wins"] / s["closed"] * 100 if s["closed"] else 0
        print(f"{bucket:<13} {s['closed']:>7} {s['open']:>5} {s['strict']:>7} {win_pct:>6.1f}% {s['gross']:>10.2f} {s['net']:>10.2f} {s['open_upnl']:>11.2f} {s['net'] + s['open_upnl']:>11.2f}")
    print()

    print("=== Closed trades ===")
    print(f"{'SYM':<7} {'UTC':>5} {'MIN':>5} {'BUCKET':<13} {'STRICT':>6} {'QTY':>5} {'BUY':>10} {'SELL':>10} {'PEAK':>10} {'GROSS':>10} {'NET':>10} {'PNL%':>8} {'MFE%':>8} {'DROP':>8}  REASON")
    print("-" * 165)
    for x in sorted(closed, key=lambda r: r["gross"]):
        mins = x.get("minutes_from_open")
        print(f"{x['symbol']:<7} {x['buy_utc']:>5} {str(mins if mins is not None else ''):>5} {x['buy_bucket']:<13} {str(x['strict_setup_ready']):>6} {x['qty']:>5.0f} {x['buy']:>10.4f} {x['sell']:>10.4f} {x['peak']:>10.4f} {x['gross']:>10.2f} {x['net']:>10.2f} {x['pnl_pct']:>7.2f}% {x['peak_gain_pct']:>7.2f}% {x['giveback_pct']:>7.2f}%  {x['reason']}")

    print()
    print("=== Open trades from today ===")
    print(f"{'SYM':<7} {'UTC':>5} {'MIN':>5} {'BUCKET':<13} {'STRICT':>6} {'QTY':>5} {'BUY':>10} {'NOW':>10} {'PEAK':>10} {'UPNL':>10} {'NOW%':>8} {'MFE%':>8} {'FROM_PK':>9}  BUY_TIME")
    print("-" * 158)
    for x in sorted(open_positions, key=lambda r: r["unrealized"]):
        mins = x.get("minutes_from_open")
        print(f"{x['symbol']:<7} {x['buy_utc']:>5} {str(mins if mins is not None else ''):>5} {x['buy_bucket']:<13} {str(x['strict_setup_ready']):>6} {x['qty']:>5.0f} {x['buy']:>10.4f} {x['current']:>10.4f} {x['peak']:>10.4f} {x['unrealized']:>10.2f} {x['current_pct']:>7.2f}% {x['peak_gain_pct']:>7.2f}% {x['from_peak_pct']:>8.2f}%  {x['buy_time']}")


if __name__ == "__main__":
    main()
