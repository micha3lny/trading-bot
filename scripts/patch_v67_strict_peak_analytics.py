from __future__ import annotations

from pathlib import Path

V67 = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')
REPORT = Path('src/live_trading/analytics/v67_daily_report.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def patch_v67() -> None:
    txt = V67.read_text()

    # 1) Add stable columns to lifecycle CSV schema.
    if '"strict_setup_ready"' not in txt or '"peak_gain_pct"' not in txt:
        old = '''        "spread_pct", "fill_price", "fill_latency_ms",
        "estimated_commission", "realized_slippage_bps",
        "raw_json",'''
        new = '''        "spread_pct", "strict_setup_ready", "strict_setup_name",
        "first_5m_high_pct", "first_15m_high_pct", "or_range_pct", "score",
        "peak_gain_pct", "giveback_pct", "from_peak_pct",
        "fill_price", "fill_latency_ms",
        "estimated_commission", "realized_slippage_bps",
        "raw_json",'''
        txt = replace_once(txt, old, new, 'extend lifecycle csv fields')

    # 2) Ensure strict setup is computed inside compute_live_safe_features.
    if 'strict_setup_ready = (' not in txt:
        marker = '    reasons: list[str] = []'
        strict_calc = '''    strict_setup_ready = (
        first_5m_high_pct is not None
        and first_15m_high_pct is not None
        and or_range_pct is not None
        and first_5m_high_pct >= getattr(args, "strict_min_first_5m_high_pct", 4.0)
        and first_15m_high_pct >= getattr(args, "strict_min_first_15m_high_pct", 6.5)
        and or_range_pct >= getattr(args, "strict_min_or_range_pct", 5.0)
        and price is not None
        and price >= getattr(args, "strict_min_price", 5.0)
        and (spread_bps is None or spread_bps <= getattr(args, "strict_max_spread_bps", 50.0))
    )

'''
        txt = replace_once(txt, marker, strict_calc + marker, 'insert strict setup calculation')

    if '"strict_setup_ready": strict_setup_ready' not in txt:
        old = '''        "ready": ready,
        "score": round(score, 4),'''
        new = '''        "ready": ready,
        "strict_setup_ready": strict_setup_ready,
        "strict_setup_name": getattr(args, "strict_setup_name", "v67_original_600usd_setup"),
        "strict_min_first_5m_high_pct": getattr(args, "strict_min_first_5m_high_pct", 4.0),
        "strict_min_first_15m_high_pct": getattr(args, "strict_min_first_15m_high_pct", 6.5),
        "strict_min_or_range_pct": getattr(args, "strict_min_or_range_pct", 5.0),
        "strict_min_price": getattr(args, "strict_min_price", 5.0),
        "strict_max_spread_bps": getattr(args, "strict_max_spread_bps", 50.0),
        "score": round(score, 4),'''
        txt = replace_once(txt, old, new, 'add strict fields to features payload')

    # 3) Add strict parser args if missing.
    if '--strict-setup-name' not in txt:
        markers = [
            '    parser.add_argument("--min-or-range-pct", type=float, default=5.0)',
            '    parser.add_argument("--min-or-range-pct", type=float, default=0.5)',
        ]
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, marker + '''
    parser.add_argument("--strict-setup-name", default="v67_original_600usd_setup")
    parser.add_argument("--strict-min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--strict-min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--strict-min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--strict-min-price", type=float, default=5.0)
    parser.add_argument("--strict-max-spread-bps", type=float, default=50.0)''', 1)
                break

    # 4) Add strict/setup metric kwargs to SIGNAL_READY record if marker exists.
    if 'event="SIGNAL_READY"' in txt and 'strict_setup_ready=features.get("strict_setup_ready")' not in txt:
        # Conservative: add extra kwargs near raw_json=features in SIGNAL_READY block.
        old = 'raw_json=features,'
        new = '''strict_setup_ready=features.get("strict_setup_ready"),
                    strict_setup_name=features.get("strict_setup_name"),
                    first_5m_high_pct=features.get("first_5m_high_pct"),
                    first_15m_high_pct=features.get("first_15m_high_pct"),
                    or_range_pct=features.get("or_range_pct"),
                    score=features.get("score"),
                    raw_json=features,'''
        if old in txt:
            txt = txt.replace(old, new, 1)

    # 5) Ensure BUY_ORDER_SENT carries raw_json=features and metric columns if marker exists.
    if 'strict_setup_ready=features.get("strict_setup_ready")' not in txt.split('BUY_ORDER_SENT')[-1]:
        markers = [
            '                        spread_pct=(snap.get("spread_bps") or 0) / 100.0 if snap.get("spread_bps") is not None else None,\n                    )',
            '                        spread_pct=spread_pct,\n                    )',
            '                        spread_pct=spread_pct,\n                        raw_json=features,\n                    )',
        ]
        addition = '''                        strict_setup_ready=features.get("strict_setup_ready"),
                        strict_setup_name=features.get("strict_setup_name"),
                        first_5m_high_pct=features.get("first_5m_high_pct"),
                        first_15m_high_pct=features.get("first_15m_high_pct"),
                        or_range_pct=features.get("or_range_pct"),
                        score=features.get("score"),
                        raw_json=features,
                    )'''
        for old in markers:
            if old in txt:
                prefix = old.split('\n                    )')[0]
                txt = txt.replace(old, prefix + '\n' + addition, 1)
                break

    # 6) Add peak analytics to SELL_ORDER_SENT inside send_exit_order.
    if 'peak_gain_pct=peak_gain_pct' not in txt:
        old = '''    pnl_pct = ((price / pos.entry_price - 1.0) * 100.0) if price and pos.entry_price > 0 else None
    record_lifecycle('''
        new = '''    pnl_pct = ((price / pos.entry_price - 1.0) * 100.0) if price and pos.entry_price > 0 else None
    peak_gain_pct = ((pos.peak_price / pos.entry_price - 1.0) * 100.0) if pos.entry_price > 0 and pos.peak_price else None
    giveback_pct = ((price / pos.peak_price - 1.0) * 100.0) if price and pos.peak_price and pos.peak_price > 0 else None
    from_peak_pct = giveback_pct
    record_lifecycle('''
        if old in txt:
            txt = txt.replace(old, new, 1)
            old2 = '''        peak_price=pos.peak_price,
        pnl_pct=pnl_pct,'''
            new2 = '''        peak_price=pos.peak_price,
        pnl_pct=pnl_pct,
        peak_gain_pct=peak_gain_pct,
        giveback_pct=giveback_pct,
        from_peak_pct=from_peak_pct,'''
            txt = replace_once(txt, old2, new2, 'sell lifecycle peak analytics')

    V67.write_text(txt)
    print('patched live v67 strict telemetry + peak analytics columns')


def patch_report() -> None:
    # Replace report with a robust version that tolerates old/new lifecycle schemas and computes strict + peak stats.
    REPORT.write_text(r'''from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
''')
    print('replaced daily report with strict + peak analytics version')


def main() -> None:
    patch_v67()
    patch_report()


if __name__ == '__main__':
    main()
