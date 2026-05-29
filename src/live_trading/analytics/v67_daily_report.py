from __future__ import annotations

import argparse
import csv
import json
import sqlite3
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


def load_rejected_entries(session: Path) -> list[dict]:
    lifecycle = session / "trade_lifecycle.csv"
    if not lifecycle.exists():
        return []
    rows: list[dict] = []
    with lifecycle.open(errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("event") != "ENTRY_ORDER_REJECTED":
                continue
            raw = raw_json_dict(row)
            rows.append(
                {
                    "recorded_at": row.get("recorded_at"),
                    "symbol": str(row.get("symbol") or "").upper(),
                    "quantity": row.get("quantity") or raw.get("quantity"),
                    "order_id": row.get("order_id") or raw.get("order_id"),
                    "reason": row.get("reason") or raw.get("reject_reason") or raw.get("reason"),
                    "ibkr_error_code": raw.get("ibkr_error_code"),
                }
            )
    rows.sort(key=lambda item: str(item.get("recorded_at") or ""), reverse=True)
    return rows


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


def load_sqlite_executions(sqlite_path: str | None, session_date: str) -> list[dict]:
    if not sqlite_path:
        return []
    path = Path(sqlite_path)
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT execution_id, symbol, side AS action, quantity, price AS fill_price,
                   order_id, perm_id, exchange, liquidity, commission,
                   commission_currency, realized_pnl, commission_source,
                   raw_json, recorded_at, executed_at
            FROM executions
            WHERE session_date = ?
            ORDER BY COALESCE(executed_at, recorded_at), execution_id
            """,
            (session_date,),
        ).fetchall()
        conn.close()
        out = []
        for row in rows:
            data = dict(row)
            data["recorded_at"] = data.get("recorded_at") or data.get("executed_at")
            out.append(data)
        return out
    except Exception:
        return []


def load_sqlite_positions(sqlite_path: str | None, session_date: str) -> list[dict]:
    if not sqlite_path:
        return []
    path = Path(sqlite_path)
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM positions
            WHERE session_date = ? AND COALESCE(active, 0) = 1
            ORDER BY symbol
            """,
            (session_date,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


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
        ("fixed TP +2.5%", 2.5, "fixed"),
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


def hold_minutes(start: str | None, end: str | None = None) -> float:
    start_dt = parse_dt(start)
    end_dt = parse_dt(end) or datetime.now(timezone.utc)
    if start_dt is None:
        return 0.0
    return max(0.0, (end_dt - start_dt).total_seconds() / 60.0)


def latest_lifecycle_rows(session: Path) -> dict[str, dict]:
    lifecycle = session / "trade_lifecycle.csv"
    latest: dict[str, dict] = {}
    if not lifecycle.exists():
        return latest
    try:
        with lifecycle.open(errors="replace") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("symbol") or "").upper()
                if sym:
                    latest[sym] = row
    except Exception:
        return latest
    return latest


def lifecycle_open_status(session: Path) -> dict[str, dict]:
    lifecycle = session / "trade_lifecycle.csv"
    status: dict[str, dict] = defaultdict(lambda: {"exit_order_exists": False, "partial_exit": False, "status": "OPEN"})
    if not lifecycle.exists():
        return status
    try:
        with lifecycle.open(errors="replace") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("symbol") or "").upper()
                event = str(row.get("event") or "")
                if not sym:
                    continue
                if event in {"SELL_ORDER_SENT", "EXIT_ORDER_SENT", "EOD_FLATTEN_SUBMIT", "MANUAL_FLATTEN_SENT"}:
                    status[sym]["exit_order_exists"] = True
                    status[sym]["status"] = "EXIT_SENT"
                if event in {"EXIT_ORDER_PARTIAL", "EXIT_PARTIAL"}:
                    status[sym]["partial_exit"] = True
                    status[sym]["status"] = "EXIT_PARTIAL"
    except Exception:
        return status
    return status


def enrich_closed_with_lifecycle(closed: list[dict], lifecycle_closed: list[dict]) -> list[dict]:
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in lifecycle_closed:
        by_symbol[row["symbol"]].append(row)
    for row in closed:
        candidates = by_symbol.get(row["symbol"], [])
        match = candidates.pop(0) if candidates else {}
        buy = f(row.get("buy"), 0.0)
        sell = f(row.get("sell"), 0.0)
        peak = f(match.get("peak"), max(buy, sell))
        peak_pct = f(match.get("peak_gain_pct"), ((peak / buy - 1.0) * 100.0 if buy else 0.0))
        pnl_pct = f(row.get("pnl_pct"), ((sell / buy - 1.0) * 100.0 if buy else 0.0))
        drop_from_peak = f(match.get("giveback_pct"), pnl_pct - peak_pct)
        row.update({
            "peak": peak,
            "peak_gain_pct": peak_pct,
            "drop_from_peak_pct": drop_from_peak,
            "giveback_pct": drop_from_peak,
            "pnl_pct": pnl_pct,
            "buy_utc": utc_hhmm(row.get("buy_time")),
            "buy_bucket": buy_time_bucket(row.get("buy_time")),
            "minutes_from_open": minutes_from_market_open(row.get("buy_time")),
            "strict_setup_ready": bool(match.get("strict_setup_ready", False)),
            "reason": row.get("reason") or match.get("reason") or "",
            "hold_min": hold_minutes(row.get("buy_time"), row.get("sell_time")),
        })
    return closed


def money(value: float) -> str:
    return f"{value:.2f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt(value, width: int = 8, decimals: int = 2) -> str:
    if value is None or value == "":
        return "NA".rjust(width)
    return f"{f(value, 0.0):>{width}.{decimals}f}"


def fmt_pct(value, width: int = 6, decimals: int = 1) -> str:
    if value is None or value == "":
        return "NA".rjust(width + 1)
    return f"{f(value, 0.0):>{width}.{decimals}f}%"


def primary_net(row: dict) -> float:
    return f(row.get("net_estimated"), f(row.get("net_actual"), f(row.get("net"), f(row.get("gross"), 0.0))))


def effective_status(row: dict) -> str:
    if row.get("partial_exit"):
        return "PARTIAL_EXIT"
    if row.get("exit_order_exists"):
        return "EXIT_ORDER"
    return str(row.get("status") or "OPEN")


def commission_coverage(closed: list[dict]) -> tuple[int, int]:
    sides = len(closed) * 2
    confirmed = 0
    for row in closed:
        if str(row.get("buy_commission_source") or "").strip().lower() == "ibkr":
            confirmed += 1
        if str(row.get("sell_commission_source") or "").strip().lower() == "ibkr":
            confirmed += 1
    return confirmed, sides


def closed_sort_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("sell_time") or row.get("closed_at") or ""),
        str(row.get("symbol") or ""),
        str(row.get("buy_execution_id") or ""),
        str(row.get("sell_execution_id") or ""),
    )


def choose_fill_snapshot(
    *,
    sqlite_fills: list[dict],
    csv_fills: list[dict],
    lifecycle_closed_count: int,
    commission_per_roundtrip: float,
) -> dict:
    sqlite_closed = reconstruct_closed_trades_from_fills(sqlite_fills, commission_per_roundtrip)
    csv_closed = reconstruct_closed_trades_from_fills(csv_fills, commission_per_roundtrip)
    if sqlite_fills and (lifecycle_closed_count == 0 or len(sqlite_closed) >= lifecycle_closed_count):
        source = "sqlite"
        fills = sqlite_fills
        reconstructed = sqlite_closed
    elif csv_fills:
        source = "csv"
        fills = csv_fills
        reconstructed = csv_closed
    elif sqlite_fills:
        source = "mixed"
        fills = sqlite_fills
        reconstructed = sqlite_closed
    else:
        source = "csv"
        fills = []
        reconstructed = []
    return {
        "source": source,
        "fills": list(fills),
        "reconstructed": list(reconstructed),
        "sqlite_executions_count": len(sqlite_fills),
        "csv_executions_count": len(csv_fills),
        "sqlite_reconstructed_count": len(sqlite_closed),
        "csv_reconstructed_count": len(csv_closed),
    }


def open_sort_key(row: dict, mode: str) -> tuple:
    symbol = str(row.get("symbol") or "")
    if mode == "symbol":
        return (symbol,)
    if mode == "peak":
        return (-f(row.get("peak_gain_pct"), 0.0), symbol)
    return (f(row.get("unrealized"), 0.0), symbol)


def closed_report_sort_key(row: dict, mode: str) -> tuple:
    symbol = str(row.get("symbol") or "")
    if mode == "time":
        return closed_sort_key(row)
    if mode == "symbol":
        return (symbol, str(row.get("sell_time") or row.get("closed_at") or ""))
    if mode == "peak":
        return (-f(row.get("peak_gain_pct"), 0.0), symbol, str(row.get("sell_time") or ""))
    return (primary_net(row), symbol, str(row.get("sell_time") or ""))


def is_active_rth_report(report_date: str) -> bool:
    now = datetime.now(timezone.utc)
    if report_date != now.strftime("%F"):
        return False
    start = now.replace(hour=13, minute=30, second=0, microsecond=0)
    end = now.replace(hour=20, minute=0, second=0, microsecond=0)
    return start <= now <= end


def build_position_diagnostics(latest_portfolio: dict, managed_summary: dict, open_positions: list[dict]) -> dict:
    ibkr_symbols = {
        str(sym).upper()
        for sym, pos in latest_portfolio.items()
        if f(pos.get("position"), 0.0) != 0
    }
    managed_symbols = {
        str(sym).upper()
        for sym in managed_summary.get("active_symbols", [])
        if str(sym).strip()
    }
    if not managed_symbols:
        managed_symbols = {str(row.get("symbol") or "").upper() for row in open_positions if row.get("symbol")}
    matched = sorted(ibkr_symbols & managed_symbols)
    true_orphans = sorted(ibkr_symbols - managed_symbols)
    missing_in_ibkr = sorted(managed_symbols - ibkr_symbols)
    fractional_orphans = [
        sym for sym in true_orphans
        if abs(f(latest_portfolio.get(sym, {}).get("position"), 0.0) - round(f(latest_portfolio.get(sym, {}).get("position"), 0.0))) > 1e-9
    ]
    whole_share_orphans = [sym for sym in true_orphans if sym not in set(fractional_orphans)]
    return {
        "active_managed_positions": len(managed_symbols),
        "ibkr_positions": len(ibkr_symbols),
        "matched_positions": len(matched),
        "true_orphans": true_orphans,
        "missing_in_ibkr": missing_in_ibkr,
        "fractional_orphans": fractional_orphans,
        "whole_share_orphans": whole_share_orphans,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%F"))
    ap.add_argument("--recorder-dir", default="data/live/recorder")
    ap.add_argument("--commission-per-roundtrip", type=float, default=1.0)
    ap.add_argument("--sqlite-path", default="data/runtime/trading_runtime.sqlite")
    ap.add_argument("--disable-sqlite", action="store_true")
    ap.add_argument("--watch-summary", action="store_true")
    ap.add_argument("--watch-full", action="store_true")
    ap.add_argument("--watch-open-limit", type=int, default=30)
    ap.add_argument("--watch-closed-limit", type=int, default=50)
    ap.add_argument("--sort-open", choices=["upnl", "symbol", "peak"], default="upnl")
    ap.add_argument("--sort-closed", choices=["net", "time", "symbol", "peak"], default="net")
    args = ap.parse_args()

    snapshot_loaded_at = datetime.now(timezone.utc).isoformat()
    session = Path(args.recorder_dir) / args.date
    lifecycle = session / "trade_lifecycle.csv"
    sqlite_path = None if args.disable_sqlite else args.sqlite_path
    sqlite_fills = load_sqlite_executions(sqlite_path, args.date)
    if not lifecycle.exists() and not sqlite_fills:
        raise SystemExit(f"Missing {lifecycle}")

    latest_portfolio = load_latest_portfolio(session)
    rejected_entries = load_rejected_entries(session)
    signal_features = load_signal_features(session)
    latest_lifecycle = latest_lifecycle_rows(session)
    open_status = lifecycle_open_status(session)
    buys: dict[str, list[dict]] = {}
    lifecycle_closed = []

    if lifecycle.exists():
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
                    pnl_pct = ((sell_px / buy_px - 1) * 100 if buy_px else 0)
                    giveback_pct = f(r.get("giveback_pct"), pnl_pct - peak_gain_pct)
                    gross = (sell_px - buy_px) * qty
                    net = gross - args.commission_per_roundtrip
                    lifecycle_closed.append({
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
                        "pnl_pct": pnl_pct,
                        "peak_gain_pct": peak_gain_pct,
                        "giveback_pct": giveback_pct,
                        "drop_from_peak_pct": giveback_pct,
                        "hold_min": hold_minutes(buy_time, r.get("recorded_at")),
                        "reason": r.get("reason"),
                    })

    csv_fills = load_fills(session)
    fill_snapshot = choose_fill_snapshot(
        sqlite_fills=sqlite_fills,
        csv_fills=csv_fills,
        lifecycle_closed_count=len(lifecycle_closed),
        commission_per_roundtrip=args.commission_per_roundtrip,
    )
    fill_rows = fill_snapshot["fills"]
    fill_closed = fill_snapshot["reconstructed"]
    closed = enrich_closed_with_lifecycle(fill_closed, lifecycle_closed) if fill_closed else lifecycle_closed
    closed = sorted(closed, key=closed_sort_key)

    open_positions = []
    sqlite_positions = load_sqlite_positions(sqlite_path, args.date)
    sqlite_open_symbols = {str(row.get("symbol") or "").upper() for row in sqlite_positions}
    for sym, rows in buys.items():
        for row in rows:
            buy_px = f(row.get("fill_price") or row.get("price") or row.get("entry_price"))
            qty = f(row.get("quantity"))
            buy_time = row.get("recorded_at")
            payload = raw_json_dict(row) or best_signal_payload_for_buy(signal_features, sym, buy_time)
            strict = is_strict_payload(payload, row)
            pf = latest_portfolio.get(sym, {})
            has_ibkr_position = sym in latest_portfolio or sym in sqlite_open_symbols
            current = f(pf.get("marketPrice"), None) if has_ibkr_position else None
            unrealized = f(pf.get("unrealizedPNL"), (current - buy_px) * qty if current else None) if has_ibkr_position else None
            lifecycle_row = latest_lifecycle.get(sym, {})
            peak = max(f(row.get("peak_price"), f(lifecycle_row.get("peak_price"), buy_px)), buy_px, current or buy_px)
            peak_pct = ((peak / buy_px - 1) * 100 if buy_px else 0)
            current_pct = ((current / buy_px - 1) * 100 if buy_px and current else None)
            status = open_status.get(sym, {})
            status_value = status.get("status") or "OPEN"
            if not has_ibkr_position:
                status_value = "MISSING_IN_IBKR/RECONCILING"
            buy_comm = sum(
                actual_commission(fill) or 0.0
                for fill in fill_rows
                if str(fill.get("symbol") or "").upper() == sym and normalized_fill_action(fill.get("action")) == "BUY"
            )
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
                "current_pct": current_pct,
                "peak_gain_pct": peak_pct,
                "from_peak_pct": (current_pct - peak_pct) if current_pct is not None else None,
                "ibkr_commission": buy_comm,
                "hold_min": hold_minutes(buy_time),
                "exit_order_exists": bool(status.get("exit_order_exists")),
                "partial_exit": bool(status.get("partial_exit")),
                "status": status_value,
            })
    existing_open_symbols = {row["symbol"] for row in open_positions}
    for row in sqlite_positions:
        sym = str(row.get("symbol") or "").upper()
        if not sym or sym in existing_open_symbols:
            continue
        raw = raw_json_dict(row)
        buy_px = f(row.get("avg_price") or raw.get("entry_price"), 0.0)
        qty = f(row.get("quantity") or row.get("ibkr_quantity"), 0.0)
        pf = latest_portfolio.get(sym, {})
        current = f(pf.get("marketPrice") or raw.get("market_price"), buy_px)
        unrealized = f(pf.get("unrealizedPNL"), (current - buy_px) * qty if current else 0.0)
        peak = max(f(raw.get("peak_price"), buy_px), buy_px, current)
        current_pct = ((current / buy_px - 1) * 100 if buy_px and current else 0.0)
        peak_pct = ((peak / buy_px - 1) * 100 if buy_px else 0.0)
        buy_time = raw.get("entry_time") or row.get("updated_at")
        buy_comm = sum(
            actual_commission(fill) or 0.0
            for fill in fill_rows
            if str(fill.get("symbol") or "").upper() == sym and normalized_fill_action(fill.get("action")) == "BUY"
        )
        open_positions.append({
            "symbol": sym,
            "qty": qty,
            "buy_time": buy_time,
            "buy_utc": utc_hhmm(buy_time),
            "buy_bucket": buy_time_bucket(buy_time),
            "minutes_from_open": minutes_from_market_open(buy_time),
            "strict_setup_ready": False,
            "buy": buy_px,
            "current": current,
            "peak": peak,
            "unrealized": unrealized,
            "current_pct": current_pct,
            "peak_gain_pct": peak_pct,
            "from_peak_pct": current_pct - peak_pct,
            "ibkr_commission": buy_comm,
            "hold_min": hold_minutes(buy_time),
            "exit_order_exists": bool(row.get("exit_sent")),
            "partial_exit": False,
            "status": row.get("status") or "OPEN",
        })

    wins = [x for x in closed if x["gross"] > 0]
    losses = [x for x in closed if x["gross"] <= 0]
    gross_total = sum(x["gross"] for x in closed)
    ibkr_commission_total = sum(f(x.get("actual_commission"), 0.0) for x in closed)
    fallback_commission_total = sum(f(x.get("estimated_commission_fallback"), 0.0) for x in closed)
    net_actual_total = sum(primary_net(x) for x in closed)
    open_upnl = sum(f(x.get("unrealized"), 0.0) for x in open_positions)
    strict_closed = [x for x in closed if x["strict_setup_ready"]]
    strict_open = [x for x in open_positions if x["strict_setup_ready"]]
    commission_per_trade = effective_commission_per_trade(fill_closed, args.commission_per_roundtrip)
    exit_simulations = simulate_exit_strategies(closed, commission_per_trade)
    fills_without_commission = [
        x for x in fill_rows
        if str(x.get("commission_source") or "").strip().lower() != "ibkr"
    ]
    eod_summary = load_eod_summary(session)
    managed_summary = load_managed_positions_summary(session)
    position_diagnostics = build_position_diagnostics(latest_portfolio, managed_summary, open_positions)
    active_rth_report = is_active_rth_report(args.date)
    event_counts = lifecycle_event_counts(session)
    avg_win = sum(x["gross"] for x in wins) / len(wins) if wins else 0.0
    avg_loss = sum(x["gross"] for x in losses) / len(losses) if losses else 0.0
    expectancy = gross_total / len(closed) if closed else 0.0
    avg_peak = avg(closed, lambda x: x.get("peak_gain_pct"))
    avg_giveback = avg(closed, lambda x: x.get("drop_from_peak_pct", x.get("giveback_pct")))
    avg_hold = avg(closed, lambda x: x.get("hold_min"))
    best_trade = max(closed, key=primary_net, default=None)
    worst_trade = min(closed, key=primary_net, default=None)
    confirmed_commission_sides, total_commission_sides = commission_coverage(closed)
    show_fallback_column = any(f(row.get("estimated_commission_fallback"), 0.0) > 0 for row in closed)
    net_column_label = "NET_ACTUAL*" if show_fallback_column else "NET_ACTUAL"

    if args.watch_summary or args.watch_full:
        print(f"SESSION SUMMARY {args.date}")
        print(f"closed trades:        {len(closed)}")
        print(f"open trades:          {len(open_positions)}")
        print(f"win rate:             {(len(wins) / len(closed) * 100 if closed else 0):.1f}%")
        print(f"gross closed pnl:     ${gross_total:.2f}")
        print(f"ibkr commissions:     ${ibkr_commission_total:.2f}")
        print(f"net actual pnl:       ${net_actual_total:.2f}")
        print(f"open unrealized pnl:  ${open_upnl:.2f}")
        print(f"total actual pnl:     ${net_actual_total + open_upnl:.2f}")
        print(f"avg win/loss:         ${avg_win:.2f} / ${avg_loss:.2f}")
        print(f"expectancy:           ${expectancy:.2f}/trade")
        print(f"average peak:         {avg_peak:.1f}%")
        print(f"average giveback:     {avg_giveback:.1f}%")
        print(f"commission coverage:  {confirmed_commission_sides}/{total_commission_sides}")
        print(f"source:               {fill_snapshot['source']}")
        print(f"executions_count:     {len(fill_rows)}")
        print(f"lifecycle_closed_count: {len(lifecycle_closed)}")
        print(f"reconstructed_closed_count: {len(fill_snapshot['reconstructed'])}")
        print()
        if open_positions:
            print("OPEN POSITIONS")
            print(f"{'SYM':<6} {'QTY':>5} {'BUY':>8} {'NOW':>8} {'UPNL':>8} {'NOW%':>7} {'PEAK%':>7} {'FROM_PEAK':>10} {'IBKR_COMM':>9} {'BUY_TIME':>8} STATUS")
            open_limit = max(0, args.watch_open_limit)
            open_rows = sorted(open_positions, key=lambda x: open_sort_key(x, args.sort_open))
            open_rows = open_rows if open_limit == 0 else open_rows[:open_limit]
            for row in open_rows:
                print(
                    f"{row['symbol']:<6} {f(row.get('qty'), 0.0):>5.0f} {fmt(row.get('buy'), 8, 3)} {fmt(row.get('current'), 8, 3)} "
                    f"{fmt(row.get('unrealized'), 8, 2)} {fmt_pct(row.get('current_pct'), 6, 1)} {fmt_pct(row.get('peak_gain_pct'), 6, 1)} "
                    f"{fmt_pct(row.get('from_peak_pct'), 9, 1)} {fmt(row.get('ibkr_commission'), 9, 2)} {row['buy_utc']:>8} {effective_status(row)}"
                )
            print()
        if rejected_entries:
            print("REJECTED ENTRIES")
            print(f"{'SYM':<6} {'QTY':>5} {'ORDER':>8} {'ERR':>5} {'TIME':>8} REASON")
            for row in rejected_entries[:20]:
                print(
                    f"{row.get('symbol', ''):<6} {f(row.get('quantity'), 0.0):>5.0f} "
                    f"{str(row.get('order_id') or ''):>8} {str(row.get('ibkr_error_code') or ''):>5} "
                    f"{utc_hhmm(row.get('recorded_at')):>8} {row.get('reason') or ''}"
                )
            print()
        closed_limit = max(0, args.watch_closed_limit)
        closed_rows = sorted(closed, key=lambda x: closed_report_sort_key(x, args.sort_closed))
        closed_rows = closed_rows if closed_limit == 0 else closed_rows[:closed_limit]
        if closed_rows:
            print("CLOSED POSITIONS")
            if show_fallback_column:
                print(f"{'SYM':<6} {'QTY':>5} {'BUY':>8} {'SELL':>8} {'GROSS':>8} {'IBKR_COMM':>9} {'EST_FB':>8} {net_column_label:>11} {'PNL%':>7} {'PEAK%':>7} {'DROP_FROM_PEAK':>14} {'HOLD_MIN':>8} EXIT_REASON")
            else:
                print(f"{'SYM':<6} {'QTY':>5} {'BUY':>8} {'SELL':>8} {'GROSS':>8} {'IBKR_COMM':>9} {net_column_label:>11} {'PNL%':>7} {'PEAK%':>7} {'DROP_FROM_PEAK':>14} {'HOLD_MIN':>8} EXIT_REASON")
            for row in closed_rows:
                base = (
                    f"{row['symbol']:<6} {f(row.get('qty'), 0.0):>5.0f} {f(row.get('buy'), 0.0):>8.3f} {f(row.get('sell'), 0.0):>8.3f} "
                    f"{f(row.get('gross'), 0.0):>8.2f} {f(row.get('actual_commission'), 0.0):>9.2f} "
                )
                if show_fallback_column:
                    base += f"{f(row.get('estimated_commission_fallback'), 0.0):>8.2f} {primary_net(row):>11.2f} "
                else:
                    base += f"{primary_net(row):>11.2f} "
                print(
                    f"{base}{f(row.get('pnl_pct'), 0.0):>6.1f}% {f(row.get('peak_gain_pct'), 0.0):>6.1f}% "
                    f"{f(row.get('drop_from_peak_pct'), 0.0):>13.1f}% {f(row.get('hold_min'), 0.0):>7.0f}m "
                    f"{str(row.get('reason') or '')[:18]}"
                )
            if show_fallback_column:
                print("* includes estimated fallback for missing commissions")
            print()
        print("EXIT SIMULATION")
        for name in ["actual trailing", "fixed TP +2.5%", "fixed TP +3%", "partial 50%@+3%"]:
            row = next((x for x in exit_simulations if x["name"] == name), None)
            if row:
                print(f"{row['name']:<17} gross={row['gross']:.2f} net_est={row['net']:.2f} captured={row['captured']}/{row['trades']}")
        print()
        print("CURRENT POSITION DIAGNOSTICS")
        print(
            f"active_managed={position_diagnostics['active_managed_positions']} "
            f"ibkr={position_diagnostics['ibkr_positions']} matched={position_diagnostics['matched_positions']} "
            f"true_orphans={len(position_diagnostics['true_orphans'])} "
            f"missing_in_ibkr={len(position_diagnostics['missing_in_ibkr'])} "
            f"whole_orphans={len(position_diagnostics['whole_share_orphans'])}"
        )
        if position_diagnostics["true_orphans"]:
            print(f"orphans={','.join(position_diagnostics['true_orphans'])}")
        if position_diagnostics["missing_in_ibkr"]:
            print(f"missing={','.join(position_diagnostics['missing_in_ibkr'])}")
        return

    print("=== SESSION SUMMARY ===")
    print(f"date:                 {args.date}")
    print(f"closed trades:        {len(closed)}")
    print(f"open trades:          {len(open_positions)}")
    print(f"win rate:             {(len(wins) / len(closed) * 100 if closed else 0):.1f}%")
    print(f"gross closed pnl:     ${gross_total:.2f}")
    print(f"ibkr_commissions_confirmed: ${ibkr_commission_total:.2f}")
    print(f"estimated_fallback:   ${fallback_commission_total:.2f}")
    print(f"net_after_commissions:${net_actual_total:.2f}")
    print(f"open unrealized pnl:  ${open_upnl:.2f}")
    print(f"total actual pnl:     ${net_actual_total + open_upnl:.2f}")
    print(f"avg win:              ${avg_win:.2f}")
    print(f"avg loss:             ${avg_loss:.2f}")
    print(f"expectancy:           ${expectancy:.2f}/trade")
    print(f"average peak:         {avg_peak:.2f}%")
    print(f"average giveback:     {avg_giveback:.2f}%")
    print(f"average hold:         {avg_hold:.1f} min")
    print(f"best trade:           {(best_trade or {}).get('symbol', '')} ${primary_net(best_trade or {}):.2f}")
    print(f"worst trade:          {(worst_trade or {}).get('symbol', '')} ${primary_net(worst_trade or {}):.2f}")
    print(f"commissions/gross:    {(abs(ibkr_commission_total) / abs(gross_total) * 100 if gross_total else 0):.1f}%")
    print(f"commission_coverage:  {confirmed_commission_sides}/{total_commission_sides} sides")
    print(f"fills without comm:   {len(fills_without_commission)}")
    print(f"source:               {fill_snapshot['source']}")
    print(f"executions_count:     {len(fill_rows)}")
    print(f"lifecycle_closed_count: {len(lifecycle_closed)}")
    print(f"reconstructed_closed_count: {len(fill_snapshot['reconstructed'])}")
    print(f"snapshot_loaded_at:   {snapshot_loaded_at}")
    print()

    print("=== OPEN POSITIONS ===")
    print(f"{'SYM':<7} {'QTY':>6} {'BUY':>9} {'NOW':>9} {'UPNL':>9} {'NOW%':>7} {'PEAK%':>7} {'FROM_PEAK%':>10} {'IBKR_COMM':>10} {'BUY_TIME':>8}  STATUS")
    print("-" * 126)
    for x in sorted(open_positions, key=lambda r: (f(r.get("unrealized"), 0.0), str(r.get("symbol") or ""))):
        print(
            f"{x['symbol']:<7} {f(x.get('qty'), 0.0):>6.0f} {fmt(x.get('buy'), 9, 4)} {fmt(x.get('current'), 9, 4)} {fmt(x.get('unrealized'), 9, 2)} "
            f"{fmt_pct(x.get('current_pct'), 6, 1)} {fmt_pct(x.get('peak_gain_pct'), 6, 1)} {fmt_pct(x.get('from_peak_pct'), 9, 1)} "
            f"{fmt(x.get('ibkr_commission'), 10, 2)} {x['buy_utc']:>8}  {effective_status(x)}"
        )
    print()

    print("=== REJECTED ENTRIES ===")
    print(f"{'SYM':<7} {'QTY':>6} {'ORDER_ID':>10} {'IBKR_ERR':>8} {'TIME':>8}  REASON")
    print("-" * 82)
    for row in rejected_entries:
        print(
            f"{row.get('symbol', ''):<7} {f(row.get('quantity'), 0.0):>6.0f} {str(row.get('order_id') or ''):>10} "
            f"{str(row.get('ibkr_error_code') or ''):>8} {utc_hhmm(row.get('recorded_at')):>8}  {row.get('reason') or ''}"
        )
    print()

    print("=== EXIT SIMULATION ===")
    print(f"{'SCENARIO':<18} {'TRADES':>7} {'CAPTURED':>8} {'GROSS':>10} {'NET_EST':>10}")
    print("-" * 58)
    for row in exit_simulations:
        print(f"{row['name']:<18} {row['trades']:>7} {row['captured']:>8} {row['gross']:>10.2f} {row['net']:>10.2f}")
    print()

    print("=== CLOSED POSITIONS ===")
    print(f"{'SYM':<7} {'QTY':>6} {'BUY':>9} {'SELL':>9} {'GROSS':>9} {'IBKR_COMM':>10} {'EST_FB':>9} {'NET_ACTUAL_OR_EST':>17} {'PNL%':>7} {'PEAK%':>7} {'DROP_FROM_PEAK%':>15} {'HOLD_MIN':>9}  EXIT_REASON")
    print("-" * 158)
    for x in sorted(closed, key=closed_sort_key):
        print(
            f"{x['symbol']:<7} {x['qty']:>6.0f} {x['buy']:>9.4f} {x['sell']:>9.4f} {x['gross']:>9.2f} "
            f"{f(x.get('actual_commission'), 0.0):>10.2f} {f(x.get('estimated_commission_fallback'), 0.0):>9.2f} {primary_net(x):>17.2f} "
            f"{x['pnl_pct']:>6.1f}% {x['peak_gain_pct']:>6.1f}% {f(x.get('drop_from_peak_pct'), 0.0):>14.1f}% "
            f"{f(x.get('hold_min'), 0.0):>9.1f}  {x.get('reason') or ''}"
        )
    print()

    print("=== CURRENT POSITION DIAGNOSTICS ===" if active_rth_report or open_positions else "=== POST-SESSION DIAGNOSTICS ===")
    print(f"active_managed_positions: {position_diagnostics['active_managed_positions']}")
    print(f"ibkr_positions:           {position_diagnostics['ibkr_positions']}")
    print(f"matched_positions:        {position_diagnostics['matched_positions']}")
    print(f"true_orphans:             {len(position_diagnostics['true_orphans'])}")
    print(f"missing_in_ibkr:          {len(position_diagnostics['missing_in_ibkr'])}")
    print(f"fractional orphans:       {len(position_diagnostics['fractional_orphans'])}")
    print(f"whole-share orphans:      {len(position_diagnostics['whole_share_orphans'])}")
    print(f"pending orders:       {eod_summary.get('pending_orders', '')}")
    clean_value = eod_summary.get("clean")
    print(f"eod clean:            {'' if clean_value is None else int(bool(clean_value))}")
    print(f"partial entries:      {event_counts.get('ENTRY_ORDER_PARTIAL', 0)}")
    print(f"partial exits:        {event_counts.get('EXIT_ORDER_PARTIAL', 0)}")
    print(f"delayed fill/cancel:  {event_counts.get('DELAYED_FILL_AFTER_CANCEL', 0)}")
    print(f"cancel but position:  {event_counts.get('ORDER_CANCEL_BUT_POSITION_EXISTS', 0)}")
    if position_diagnostics["true_orphans"]:
        print(f"orphan symbols:       {', '.join(position_diagnostics['true_orphans'])}")
    if position_diagnostics["missing_in_ibkr"]:
        print(f"missing symbols:      {', '.join(position_diagnostics['missing_in_ibkr'])}")
    print()

    print("=== Strict/original setup subset ===")
    strict_wins = [x for x in strict_closed if x["gross"] > 0]
    print(f"Name:                 {STRICT_SETUP_NAME}")
    print(f"Thresholds:           5m>={STRICT_MIN_FIRST_5M_HIGH_PCT}%, 15m>={STRICT_MIN_FIRST_15M_HIGH_PCT}%, OR>={STRICT_MIN_OR_RANGE_PCT}%, spread<={STRICT_MAX_SPREAD_BPS}bps")
    print(f"Strict closed trades: {len(strict_closed)}")
    print(f"Strict open trades:   {len(strict_open)}")
    print(f"Strict win rate:      {(len(strict_wins) / len(strict_closed) * 100 if strict_closed else 0):.1f}%")
    print(f"Strict gross closed:  ${sum(x['gross'] for x in strict_closed):.2f}")
    print(f"Strict net closed:    ${sum(primary_net(x) for x in strict_closed):.2f}")
    print(f"Strict open UPNL:     ${sum(f(x.get('unrealized'), 0.0) for x in strict_open):.2f}")
    print(f"Strict total actual:  ${sum(primary_net(x) for x in strict_closed) + sum(f(x.get('unrealized'), 0.0) for x in strict_open):.2f}")
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
            f"{sum(primary_net(x) for x in rows):>10.2f}"
        )
    print()

    print("=== PnL by buy time bucket ===")
    bucket_stats = defaultdict(lambda: {"closed": 0, "open": 0, "wins": 0, "gross": 0.0, "net": 0.0, "open_upnl": 0.0, "strict": 0})
    for x in closed:
        s = bucket_stats[x["buy_bucket"]]
        s["closed"] += 1
        s["gross"] += x["gross"]
        s["net"] += primary_net(x)
        s["wins"] += 1 if x["gross"] > 0 else 0
        s["strict"] += 1 if x["strict_setup_ready"] else 0
    for x in open_positions:
        s = bucket_stats[x["buy_bucket"]]
        s["open"] += 1
        s["open_upnl"] += f(x.get("unrealized"), 0.0)
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


if __name__ == "__main__":
    main()
