from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, time as dtime, timezone
from pathlib import Path

MARKET_OPEN_UTC = "13:30"
MARKET_CLOSE_UTC = "20:00"


def f(x, default=0.0):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def parse_bar_time(value: str | None):
    if not value:
        return None
    raw = str(value).strip().replace("Z", "+00:00")
    raw = raw.replace("US/Eastern", "").replace("America/New_York", "").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y%m%d  %H:%M:%S",
        "%Y%m%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_ts(value: str | None):
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def hhmm_to_time(value: str) -> dtime:
    hh, mm = [int(x) for x in value.split(":", 1)]
    return dtime(hour=hh, minute=mm, tzinfo=timezone.utc)


def raw_json_dict(row: dict) -> dict:
    try:
        raw = row.get("raw_json") or "{}"
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)
    except Exception:
        return {}


def pct(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base <= 0:
        return None
    return (value / base - 1.0) * 100.0


def load_lifecycle(session: Path):
    path = session / "trade_lifecycle.csv"
    signals: dict[str, list[dict]] = defaultdict(list)
    buys: dict[str, list[dict]] = defaultdict(list)
    sells: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return signals, buys, sells
    with path.open(errors="replace") as fh:
        for row in csv.DictReader(fh):
            sym = str(row.get("symbol", "")).upper().strip()
            event = row.get("event")
            if not sym:
                continue
            if event == "SIGNAL_READY":
                payload = raw_json_dict(row)
                payload.update({
                    "recorded_at": row.get("recorded_at"),
                    "reason": row.get("reason") or payload.get("reason"),
                    "score": row.get("score") or payload.get("score"),
                    "strict_setup_ready": row.get("strict_setup_ready") or payload.get("strict_setup_ready"),
                    "first_5m_high_pct": row.get("first_5m_high_pct") or payload.get("first_5m_high_pct"),
                    "first_15m_high_pct": row.get("first_15m_high_pct") or payload.get("first_15m_high_pct"),
                    "or_range_pct": row.get("or_range_pct") or payload.get("or_range_pct"),
                    "spread_bps": payload.get("spread_bps"),
                })
                signals[sym].append(payload)
            elif event == "BUY_ORDER_SENT":
                buys[sym].append(row)
            elif event in {"SELL_ORDER_SENT", "BUY_TO_COVER_SENT"}:
                sells[sym].append(row)
    return signals, buys, sells


def infer_reject_reasons(metrics: dict, args: argparse.Namespace) -> list[str]:
    reasons = []
    if metrics.get("first_5m_high_pct") is None or metrics["first_5m_high_pct"] < args.min_first_5m_high_pct:
        reasons.append("first_5m_high_too_low")
    if metrics.get("first_15m_high_pct") is None or metrics["first_15m_high_pct"] < args.min_first_15m_high_pct:
        reasons.append("first_15m_high_too_low")
    if metrics.get("or_range_pct") is None or metrics["or_range_pct"] < args.min_or_range_pct:
        reasons.append("or_range_too_low")
    if metrics.get("open_price") is None or metrics["open_price"] < args.min_price:
        reasons.append("price_too_low_at_open")
    # spread requires live quotes; candle-only report cannot reconstruct it.
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%F"))
    ap.add_argument("--recorder-dir", default="data/live/recorder")
    ap.add_argument("--runner-threshold-pct", type=float, default=10.0)
    ap.add_argument("--market-open-utc", default=MARKET_OPEN_UTC)
    ap.add_argument("--market-close-utc", default=MARKET_CLOSE_UTC)
    ap.add_argument("--min-first-5m-high-pct", type=float, default=0.5)
    ap.add_argument("--min-first-15m-high-pct", type=float, default=1.0)
    ap.add_argument("--min-or-range-pct", type=float, default=0.5)
    ap.add_argument("--min-price", type=float, default=5.0)
    args = ap.parse_args()

    session = Path(args.recorder_dir) / args.date
    candles = session / "candles_1m.csv"
    if not candles.exists():
        raise SystemExit(f"Missing {candles}")

    market_open = hhmm_to_time(args.market_open_utc)
    market_close = hhmm_to_time(args.market_close_utc)
    by_symbol: dict[str, list[dict]] = defaultdict(list)

    with candles.open(errors="replace") as fh:
        for row in csv.DictReader(fh):
            sym = str(row.get("symbol", "")).upper().strip()
            ts = parse_bar_time(row.get("bar_time"))
            if not sym or ts is None:
                continue
            if ts.date().isoformat() != args.date:
                continue
            if not (market_open <= ts.timetz() <= market_close):
                continue
            o = f(row.get("open"), None)
            h = f(row.get("high"), None)
            l = f(row.get("low"), None)
            c = f(row.get("close"), None)
            if o is None or h is None or l is None or c is None:
                continue
            by_symbol[sym].append({"ts": ts, "open": o, "high": h, "low": l, "close": c})

    signals, buys, sells = load_lifecycle(session)
    rows = []

    for sym, bars in by_symbol.items():
        bars.sort(key=lambda x: x["ts"])
        if not bars:
            continue
        open_price = bars[0]["open"]
        session_high_bar = max(bars, key=lambda x: x["high"])
        session_high_pct = pct(open_price, session_high_bar["high"])
        if session_high_pct is None or session_high_pct < args.runner_threshold_pct:
            continue

        first_5 = [b for b in bars if (b["ts"] - bars[0]["ts"]).total_seconds() < 5 * 60]
        first_15 = [b for b in bars if (b["ts"] - bars[0]["ts"]).total_seconds() < 15 * 60]
        first_30 = [b for b in bars if (b["ts"] - bars[0]["ts"]).total_seconds() < 30 * 60]
        first_5m_high_pct = pct(open_price, max([b["high"] for b in first_5], default=None))
        first_15m_high_pct = pct(open_price, max([b["high"] for b in first_15], default=None))
        or_high = max([b["high"] for b in first_30], default=None)
        or_low = min([b["low"] for b in first_30], default=None)
        or_range_pct = ((or_high / or_low - 1.0) * 100.0) if or_high and or_low and or_low > 0 else None

        bought = sym in buys and len(buys[sym]) > 0
        first_buy = buys[sym][0] if bought else {}
        buy_time = first_buy.get("recorded_at")
        buy_px = f(first_buy.get("fill_price") or first_buy.get("price") or first_buy.get("entry_price"), None)
        buy_ts = parse_ts(buy_time)
        before_buy_high_pct = None
        after_buy_high_pct = None
        if buy_ts and buy_px:
            before = [b["high"] for b in bars if b["ts"] <= buy_ts]
            after = [b["high"] for b in bars if b["ts"] >= buy_ts]
            before_buy_high_pct = pct(open_price, max(before, default=None))
            after_buy_high_pct = pct(buy_px, max(after, default=None))

        metrics = {
            "open_price": open_price,
            "first_5m_high_pct": first_5m_high_pct,
            "first_15m_high_pct": first_15m_high_pct,
            "or_range_pct": or_range_pct,
        }
        inferred_rejects = infer_reject_reasons(metrics, args)
        signal_payloads = signals.get(sym, [])
        had_signal_ready = bool(signal_payloads)
        last_signal = signal_payloads[-1] if had_signal_ready else {}
        signal_reason = str(last_signal.get("reason") or "")

        if bought:
            why = "BOUGHT"
        elif had_signal_ready:
            why = f"SIGNAL_READY_BUT_NOT_BOUGHT:{signal_reason or 'unknown_after_signal'}"
        else:
            why = ";".join(inferred_rejects) if inferred_rejects else "NO_SIGNAL_READY_PROBABLY_SPREAD_OR_POSITION_LIMIT_OR_LATE_STATE"

        rows.append({
            "symbol": sym,
            "open": open_price,
            "session_high": session_high_bar["high"],
            "session_high_pct": session_high_pct,
            "high_time_utc": session_high_bar["ts"].strftime("%H:%M"),
            "first_5m_high_pct": first_5m_high_pct,
            "first_15m_high_pct": first_15m_high_pct,
            "or_range_pct": or_range_pct,
            "bought": bought,
            "buy_time_utc": buy_ts.strftime("%H:%M") if buy_ts else "",
            "buy_price": buy_px,
            "before_buy_high_pct": before_buy_high_pct,
            "after_buy_high_from_buy_pct": after_buy_high_pct,
            "buy_count": len(buys.get(sym, [])),
            "sell_count": len(sells.get(sym, [])),
            "had_signal_ready": had_signal_ready,
            "why_not_bought_or_status": why,
        })

    rows.sort(key=lambda r: r["session_high_pct"], reverse=True)

    print(f"=== v67 Missed Runners Report {args.date} ===")
    print(f"Runner threshold: +{args.runner_threshold_pct:.1f}% intraday from first RTH open")
    print(f"Runners found: {len(rows)}")
    print()
    bought_rows = [r for r in rows if r["bought"]]
    missed_rows = [r for r in rows if not r["bought"]]
    print(f"Bought runners: {len(bought_rows)}")
    print(f"Missed runners: {len(missed_rows)}")
    print()

    reason_counts = defaultdict(int)
    for r in missed_rows:
        reason_counts[r["why_not_bought_or_status"]] += 1
    print("=== Missed reason summary ===")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{count:>4}  {reason}")
    print()

    print("=== Runners detail ===")
    print(f"{'SYM':<7} {'RUN%':>7} {'HIGH':>8} {'H_TIME':>6} {'BGT':>4} {'BUY':>5} {'BUY_PX':>9} {'5M%':>7} {'15M%':>7} {'OR%':>7} {'WHY'}")
    print("-" * 132)
    for r in rows:
        def fmt(v):
            return "" if v is None else f"{v:.2f}"
        print(
            f"{r['symbol']:<7} {r['session_high_pct']:>6.2f}% {r['session_high']:>8.2f} {r['high_time_utc']:>6} "
            f"{str(r['bought']):>4} {r['buy_time_utc']:>5} {fmt(r['buy_price']):>9} "
            f"{fmt(r['first_5m_high_pct']):>7} {fmt(r['first_15m_high_pct']):>7} {fmt(r['or_range_pct']):>7} "
            f"{r['why_not_bought_or_status']}"
        )


if __name__ == "__main__":
    main()
