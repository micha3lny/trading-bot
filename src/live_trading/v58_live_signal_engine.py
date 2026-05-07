from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from ib_insync import IB, Stock


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 61
DEFAULT_SYMBOLS = ["QQQ", "SPY", "IWM", "NVDA", "TSLA", "META", "RKLB", "SOUN", "PLTR", "NBIS"]
DEFAULT_OUTPUT = "data/live/order_intents.csv"
DEFAULT_SNAPSHOT_OUTPUT = "data/live/live_signal_snapshots.csv"
DEFAULT_FLOW_SIGNALS = "data/live/v61_flow_signals.csv"


@dataclass
class SymbolState:
    symbol: str
    bars: list[dict[str, float]] = field(default_factory=list)
    intent_sent: bool = False


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if value != value:
            return None
        return value
    except Exception:
        return None


def load_latest_flow_scores(path: str) -> dict[str, dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if df.empty or "symbol" not in df.columns:
        return {}
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values("timestamp_utc")
    latest = df.groupby("symbol", as_index=False).tail(1)
    out: dict[str, dict[str, object]] = {}
    for _, row in latest.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        out[symbol] = {
            "flow_score": safe_float(row.get("flow_score")) or 0.0,
            "relative_volume": safe_float(row.get("relative_volume")),
            "relative_strength_6": safe_float(row.get("relative_strength_6")),
            "momentum_acceleration": safe_float(row.get("momentum_acceleration")),
            "flow_reasons": row.get("flow_reasons", ""),
        }
    return out


def market_snapshot(symbol: str, ticker) -> dict[str, object]:
    bid = safe_float(ticker.bid)
    ask = safe_float(ticker.ask)
    last = safe_float(ticker.last)
    close = safe_float(ticker.close)
    volume = safe_float(ticker.volume)
    bid_size = safe_float(ticker.bidSize)
    ask_size = safe_float(ticker.askSize)

    mid = None
    spread = None
    spread_bps = None
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        if mid > 0:
            spread_bps = spread / mid * 10_000.0

    reference_price = last or mid or close

    return {
        "timestamp_utc": now_utc(),
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
        "close": close,
        "volume": volume,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread": spread,
        "spread_bps": spread_bps,
        "reference_price": reference_price,
    }


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def update_bar_state(state: SymbolState, snap: dict[str, object], max_bars: int = 120) -> None:
    price = safe_float(snap.get("reference_price"))
    if price is None or price <= 0:
        return
    volume = safe_float(snap.get("volume")) or 0.0
    state.bars.append({
        "timestamp": time.time(),
        "price": price,
        "volume": volume,
    })
    if len(state.bars) > max_bars:
        state.bars = state.bars[-max_bars:]


def apply_flow_boost(features: dict[str, object], flow: dict[str, object] | None, args: argparse.Namespace) -> dict[str, object]:
    base_score = float(features.get("score", 0.0) or 0.0)
    flow_score = float((flow or {}).get("flow_score", 0.0) or 0.0)
    flow_boost = min(args.max_flow_score_boost, flow_score * args.flow_score_multiplier)

    # Flow is additive, but it cannot override required market structure filters.
    final_score = base_score + flow_boost
    features["base_score"] = base_score
    features["flow_score"] = flow_score
    features["flow_boost"] = flow_boost
    features["score"] = final_score
    features["relative_volume"] = (flow or {}).get("relative_volume")
    features["relative_strength_6"] = (flow or {}).get("relative_strength_6")
    features["momentum_acceleration"] = (flow or {}).get("momentum_acceleration")
    features["flow_reasons"] = (flow or {}).get("flow_reasons", "")
    return features


def compute_features(state: SymbolState, snap: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    bars = state.bars
    price = safe_float(snap.get("reference_price"))
    spread_bps = safe_float(snap.get("spread_bps"))
    if not bars or price is None:
        return {
            "ready": False,
            "reason": "no_price_history",
            "score": 0.0,
            "base_score": 0.0,
            "flow_score": 0.0,
            "flow_boost": 0.0,
        }

    prices = pd.Series([b["price"] for b in bars], dtype="float64")
    opening_window = prices.iloc[: max(1, min(args.opening_range_samples, len(prices)))]
    or_high = float(opening_window.max())
    or_low = float(opening_window.min())
    first_price = float(prices.iloc[0])
    last_price = float(prices.iloc[-1])

    intraday_from_first_pct = (last_price / first_price - 1.0) * 100.0 if first_price > 0 else 0.0
    or_breakout_pct = (last_price / or_high - 1.0) * 100.0 if or_high > 0 else 0.0
    or_range_pct = (or_high / or_low - 1.0) * 100.0 if or_low > 0 else 999.0

    momentum_5 = 0.0
    if len(prices) >= 5:
        base = float(prices.iloc[-5])
        if base > 0:
            momentum_5 = (last_price / base - 1.0) * 100.0

    spread_ok = spread_bps is not None and spread_bps <= args.max_spread_bps
    price_ok = last_price >= args.min_price
    or_ready = len(prices) >= args.opening_range_samples
    breakout = or_ready and or_breakout_pct >= args.min_breakout_pct
    momentum_ok = momentum_5 >= args.min_momentum_5_pct
    or_range_ok = or_range_pct <= args.max_or_range_pct

    score = 0.0
    if breakout:
        score += 3.0
    if momentum_ok:
        score += 2.0
    if spread_ok:
        score += 2.0
    if price_ok:
        score += 1.0
    if or_range_ok:
        score += 1.0
    if intraday_from_first_pct >= 1.0:
        score += 1.0
    if intraday_from_first_pct >= 2.0:
        score += 1.0

    reasons = []
    if not or_ready:
        reasons.append("opening_range_not_ready")
    if not breakout:
        reasons.append("no_or_breakout")
    if not momentum_ok:
        reasons.append("momentum_too_weak")
    if not spread_ok:
        reasons.append("spread_too_wide")
    if not price_ok:
        reasons.append("price_too_low")
    if not or_range_ok:
        reasons.append("or_range_too_wide")
    if score < args.min_signal_score:
        reasons.append("score_too_low")

    return {
        "ready": False,  # final readiness is computed after flow boost
        "structure_ready": bool(or_ready and breakout and momentum_ok and spread_ok and price_ok and or_range_ok),
        "reason": ";".join(reasons) if reasons else "signal_ready",
        "score": score,
        "base_score": score,
        "flow_score": 0.0,
        "flow_boost": 0.0,
        "last_price": last_price,
        "first_price": first_price,
        "or_high": or_high,
        "or_low": or_low,
        "or_range_pct": or_range_pct,
        "or_breakout_pct": or_breakout_pct,
        "momentum_5_pct": momentum_5,
        "intraday_from_first_pct": intraday_from_first_pct,
        "spread_bps": spread_bps,
        "samples": len(prices),
    }


def finalize_ready(features: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    structure_ready = bool(features.get("structure_ready"))
    score = float(features.get("score", 0.0) or 0.0)
    ready = bool(structure_ready and score >= args.min_signal_score)
    reasons = [r for r in str(features.get("reason", "")).split(";") if r]
    if score >= args.min_signal_score:
        reasons = [r for r in reasons if r != "score_too_low"]
    elif "score_too_low" not in reasons:
        reasons.append("score_too_low")
    features["ready"] = ready
    features["reason"] = ";".join(reasons) if reasons else "signal_ready"
    return features


def calc_position_usd(features: dict[str, object], args: argparse.Namespace) -> float:
    score = float(features.get("score", 0.0) or 0.0)
    flow_score = float(features.get("flow_score", 0.0) or 0.0)
    if score >= 10 and flow_score >= args.min_flow_score_for_max_size:
        return args.max_position_usd
    if score >= 8:
        return min(args.max_position_usd, args.base_position_usd * 1.5)
    return args.base_position_usd


def create_intent(symbol: str, snap: dict[str, object], features: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    price = safe_float(snap.get("ask")) or safe_float(snap.get("reference_price"))
    if price is None or price <= 0:
        price = safe_float(features.get("last_price")) or 0.0
    position_usd = calc_position_usd(features, args)
    quantity = int(max(1, position_usd // price)) if price > 0 else 0
    limit_price = round(price * (1.0 + args.limit_offset_bps / 10_000.0), 2) if price > 0 else None

    return {
        "timestamp_utc": now_utc(),
        "intent_type": "BUY_INTENT",
        "symbol": symbol,
        "side": "BUY",
        "quantity": quantity,
        "position_usd": position_usd,
        "reference_price": price,
        "limit_price": limit_price,
        "order_type": "MARKETABLE_LIMIT_INTENT",
        "score": features.get("score"),
        "base_score": features.get("base_score"),
        "flow_score": features.get("flow_score"),
        "flow_boost": features.get("flow_boost"),
        "flow_reasons": features.get("flow_reasons"),
        "relative_volume": features.get("relative_volume"),
        "relative_strength_6": features.get("relative_strength_6"),
        "momentum_acceleration": features.get("momentum_acceleration"),
        "reason": features.get("reason"),
        "or_high": features.get("or_high"),
        "or_low": features.get("or_low"),
        "or_breakout_pct": features.get("or_breakout_pct"),
        "momentum_5_pct": features.get("momentum_5_pct"),
        "intraday_from_first_pct": features.get("intraday_from_first_pct"),
        "spread_bps": features.get("spread_bps"),
        "bid": snap.get("bid"),
        "ask": snap.get("ask"),
        "last": snap.get("last"),
        "mid": snap.get("mid"),
        "status": "NOT_SENT_TO_BROKER",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v58 live signal engine - generates order intents only")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--market-data-type", type=int, default=3, help="1=live, 3=delayed")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-output", default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--flow-signals", default=DEFAULT_FLOW_SIGNALS)
    parser.add_argument("--enable-flow-boost", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--opening-range-samples", type=int, default=6, help="6 samples at 5s interval = 30 seconds test OR")
    parser.add_argument("--min-breakout-pct", type=float, default=0.10)
    parser.add_argument("--min-momentum-5-pct", type=float, default=0.05)
    parser.add_argument("--max-spread-bps", type=float, default=15.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-or-range-pct", type=float, default=3.0)
    parser.add_argument("--min-signal-score", type=float, default=7.0)
    parser.add_argument("--flow-score-multiplier", type=float, default=1.0)
    parser.add_argument("--max-flow-score-boost", type=float, default=3.0)
    parser.add_argument("--min-flow-score-for-max-size", type=float, default=3.0)

    parser.add_argument("--base-position-usd", type=float, default=100.0)
    parser.add_argument("--max-position-usd", type=float, default=250.0)
    parser.add_argument("--limit-offset-bps", type=float, default=2.0)
    parser.add_argument("--one-intent-per-symbol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    print("=== v58 live signal engine ===")
    print("Mode: signal/order-intent only. No broker orders are sent.")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Market data type: {args.market_data_type}")
    print(f"Output intents: {args.output}")
    print(f"Flow boost: {args.enable_flow_boost} file={args.flow_signals}")

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as exc:
        print("Connection failed")
        print(repr(exc))
        return 1

    tickers = {}
    states = {}
    contracts = []

    try:
        ib.reqMarketDataType(args.market_data_type)
        for symbol in [s.upper() for s in args.symbols]:
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                print(f"Could not qualify: {symbol}")
                continue
            q = qualified[0]
            contracts.append((symbol, q))
            tickers[symbol] = ib.reqMktData(q, "", False, False)
            states[symbol] = SymbolState(symbol=symbol)
            print(f"Subscribed: {symbol} conId={q.conId}")

        intent_fields = [
            "timestamp_utc", "intent_type", "symbol", "side", "quantity", "position_usd", "reference_price",
            "limit_price", "order_type", "score", "base_score", "flow_score", "flow_boost", "flow_reasons",
            "relative_volume", "relative_strength_6", "momentum_acceleration", "reason", "or_high", "or_low",
            "or_breakout_pct", "momentum_5_pct", "intraday_from_first_pct", "spread_bps", "bid", "ask", "last",
            "mid", "status",
        ]
        snapshot_fields = [
            "timestamp_utc", "symbol", "bid", "ask", "mid", "last", "close", "volume", "bid_size",
            "ask_size", "spread", "spread_bps", "reference_price", "ready", "structure_ready", "reason", "score",
            "base_score", "flow_score", "flow_boost", "flow_reasons", "relative_volume", "relative_strength_6",
            "momentum_acceleration", "or_high", "or_low", "or_breakout_pct", "momentum_5_pct", "samples",
        ]

        start = time.time()
        while time.time() - start < args.duration_seconds:
            ib.sleep(args.interval_seconds)
            flow_scores = load_latest_flow_scores(args.flow_signals) if args.enable_flow_boost else {}
            snapshot_rows = []
            intent_rows = []

            for symbol, _ in contracts:
                snap = market_snapshot(symbol, tickers[symbol])
                state = states[symbol]
                update_bar_state(state, snap)
                features = compute_features(state, snap, args)
                if args.enable_flow_boost:
                    features = apply_flow_boost(features, flow_scores.get(symbol), args)
                features = finalize_ready(features, args)

                snapshot_rows.append({**snap, **features})

                if features.get("ready") and (not args.one_intent_per_symbol or not state.intent_sent):
                    intent = create_intent(symbol, snap, features, args)
                    intent_rows.append(intent)
                    state.intent_sent = True

            if snapshot_rows:
                append_csv(Path(args.snapshot_output), snapshot_rows, snapshot_fields)
            if intent_rows:
                append_csv(Path(args.output), intent_rows, intent_fields)
                for intent in intent_rows:
                    print(
                        f"BUY_INTENT {intent['symbol']} qty={intent['quantity']} "
                        f"limit={intent['limit_price']} score={float(intent['score']):.1f} "
                        f"base={float(intent.get('base_score') or 0.0):.1f} flow={float(intent.get('flow_score') or 0.0):.1f} "
                        f"spread={intent['spread_bps']}bps"
                    )

            status_parts = []
            for row in snapshot_rows:
                score = row.get("score")
                base_score = row.get("base_score")
                flow_score = row.get("flow_score")
                score_txt = "NA" if score is None else f"{float(score):.1f}"
                base_txt = "NA" if base_score is None else f"{float(base_score):.1f}"
                flow_txt = "NA" if flow_score is None else f"{float(flow_score):.1f}"
                spread = row.get("spread_bps")
                spread_txt = "NA" if spread is None else f"{float(spread):.1f}"
                status_parts.append(
                    f"{row['symbol']} score={score_txt} base={base_txt} flow={flow_txt} "
                    f"spread={spread_txt} reason={row.get('reason')}"
                )
            print(" | ".join(status_parts), flush=True)

    finally:
        for ticker in tickers.values():
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        ib.disconnect()
        print("Disconnected")

    print("v58 live signal engine complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
