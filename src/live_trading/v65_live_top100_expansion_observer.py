from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ib_insync import IB, Stock

from src.live_trading.v62_live_data_recorder import (
    ExtendedHoursCandle1m,
    LiveCandle1m,
    LiveDataRecorder,
    MarketDataSnapshot,
    OrderIntent,
    SelectionEvent,
    SignalSnapshot,
    SpreadSnapshot,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4001
DEFAULT_CLIENT_ID = 65
DEFAULT_ALPHA_RANK = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_RECORDER_DIR = "data/live/recorder"


@dataclass
class SymbolState:
    symbol: str
    open_price: float | None = None
    first_price: float | None = None
    high: float | None = None
    low: float | None = None
    last_price: float | None = None
    first_5m_high: float | None = None
    first_15m_high: float | None = None
    or_high: float | None = None
    or_low: float | None = None
    first_seen_ts: float | None = None
    latest_volume: float | None = None
    signal_sent: bool = False
    bars: list[dict[str, Any]] = field(default_factory=list)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def load_top_symbols(alpha_rank_csv: str, top_n: int, min_price: float | None = None) -> list[str]:
    p = Path(alpha_rank_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing alpha rank file: {alpha_rank_csv}")
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("alpha rank csv must contain symbol column")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if "alpha_score" in df.columns:
        df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("alpha_score", ascending=False)
    if min_price is not None and "last_close" in df.columns:
        df["last_close"] = pd.to_numeric(df["last_close"], errors="coerce")
        df = df[df["last_close"] >= min_price]
    return df["symbol"].dropna().drop_duplicates().head(top_n).tolist()


def snapshot_from_ticker(symbol: str, ticker) -> dict[str, Any]:
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
    if bid is not None and ask is not None and ask >= bid and bid > 0:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10_000.0 if mid else None
    price = last or mid or close
    return {
        "symbol": symbol,
        "price": price,
        "bid": bid,
        "ask": ask,
        "last": last,
        "close": close,
        "volume": volume,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid_price": mid,
        "spread": spread,
        "spread_bps": spread_bps,
    }


def update_state(state: SymbolState, snap: dict[str, Any], elapsed: float, opening_range_seconds: int) -> None:
    price = safe_float(snap.get("price"))
    if price is None or price <= 0:
        return
    now_ts = time.time()
    if state.first_seen_ts is None:
        state.first_seen_ts = now_ts
        state.first_price = price
        state.open_price = price
        state.high = price
        state.low = price
    state.last_price = price
    state.high = max(state.high or price, price)
    state.low = min(state.low or price, price)
    state.latest_volume = safe_float(snap.get("volume"))
    if elapsed <= 5 * 60:
        state.first_5m_high = max(state.first_5m_high or price, price)
    if elapsed <= 15 * 60:
        state.first_15m_high = max(state.first_15m_high or price, price)
    if elapsed <= opening_range_seconds:
        state.or_high = max(state.or_high or price, price)
        state.or_low = min(state.or_low or price, price)
    state.bars.append({"recorded_at": now_utc(), "price": price, "volume": state.latest_volume})
    if len(state.bars) > 500:
        state.bars = state.bars[-500:]


def pct_from(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base <= 0:
        return None
    return (value / base - 1.0) * 100.0


def compute_live_safe_features(state: SymbolState, snap: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    first = state.first_price or state.open_price
    last = state.last_price or safe_float(snap.get("price"))
    first_5m_high_pct = pct_from(first, state.first_5m_high)
    first_15m_high_pct = pct_from(first, state.first_15m_high)
    or_range_pct = None
    if state.or_low is not None and state.or_high is not None and state.or_low > 0:
        or_range_pct = (state.or_high / state.or_low - 1.0) * 100.0
    price = last
    spread_bps = safe_float(snap.get("spread_bps"))
    ready = (
        first_5m_high_pct is not None
        and first_15m_high_pct is not None
        and or_range_pct is not None
        and first_5m_high_pct >= args.min_first_5m_high_pct
        and first_15m_high_pct >= args.min_first_15m_high_pct
        and or_range_pct >= args.min_or_range_pct
        and price is not None
        and price >= args.min_price
        and (spread_bps is None or spread_bps <= args.max_spread_bps)
    )
    score = 0.0
    for value, weight in [
        (first_5m_high_pct, 2.0),
        (first_15m_high_pct, 2.0),
        (or_range_pct, 1.0),
    ]:
        if value is not None:
            score += value * weight
    if spread_bps is not None:
        score += max(0.0, args.max_spread_bps - spread_bps) / args.max_spread_bps * 5.0
    reasons = []
    if first_5m_high_pct is None or first_5m_high_pct < args.min_first_5m_high_pct:
        reasons.append("first_5m_high_too_low")
    if first_15m_high_pct is None or first_15m_high_pct < args.min_first_15m_high_pct:
        reasons.append("first_15m_high_too_low")
    if or_range_pct is None or or_range_pct < args.min_or_range_pct:
        reasons.append("or_range_too_low")
    if price is None or price < args.min_price:
        reasons.append("price_too_low")
    if spread_bps is not None and spread_bps > args.max_spread_bps:
        reasons.append("spread_too_wide")
    return {
        "ready": ready,
        "score": round(score, 4),
        "reason": ";".join(reasons) if reasons else "live_safe_expansion_ready",
        "first_5m_high_pct": first_5m_high_pct,
        "first_15m_high_pct": first_15m_high_pct,
        "or_range_pct": or_range_pct,
        "entry_price": price,
        "spread_bps": spread_bps,
        "open_price": first,
        "or_high": state.or_high,
        "or_low": state.or_low,
    }


def record_snapshot(recorder: LiveDataRecorder, snap: dict[str, Any], session_type: str) -> None:
    recorder.record_market_snapshot(MarketDataSnapshot(
        symbol=snap["symbol"],
        price=snap.get("price"),
        bid=snap.get("bid"),
        ask=snap.get("ask"),
        last=snap.get("last"),
        bid_size=snap.get("bid_size"),
        ask_size=snap.get("ask_size"),
        volume=snap.get("volume"),
        close=snap.get("close"),
        spread_bps=snap.get("spread_bps"),
    ))
    if snap.get("spread_bps") is not None:
        recorder.record_spread_snapshot(SpreadSnapshot(
            symbol=snap["symbol"],
            timestamp=now_utc(),
            bid=snap.get("bid"),
            ask=snap.get("ask"),
            spread=snap.get("spread"),
            spread_bps=snap.get("spread_bps"),
            mid_price=snap.get("mid_price"),
            session_type=session_type,
        ))


def main() -> int:
    parser = argparse.ArgumentParser(description="v65 observe-only live top100 + live-safe expansion bot")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--alpha-rank-csv", default=DEFAULT_ALPHA_RANK)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--market-data-type", type=int, default=1, help="1=live, 3=delayed")
    parser.add_argument("--session-type", default="regular", choices=["regular", "premarket", "afterhours"])
    parser.add_argument("--opening-range-seconds", type=int, default=15 * 60)
    parser.add_argument("--min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-spread-bps", type=float, default=50.0)
    parser.add_argument("--position-usd", type=float, default=1000.0)
    parser.add_argument("--observe-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    symbols = load_top_symbols(args.alpha_rank_csv, args.top_n, min_price=args.min_price)
    if not symbols:
        raise SystemExit("No symbols loaded for live observer")

    recorder = LiveDataRecorder(args.recorder_dir)
    recorder.record_run_metadata({
        "module": "v65_live_top100_expansion_observer",
        "strategy": "v59_top100_live_safe_expansion_observe_only",
        "top_n": args.top_n,
        "symbols": symbols,
        "host": args.host,
        "port": args.port,
        "market_data_type": args.market_data_type,
        "session_type": args.session_type,
        "observe_only": args.observe_only,
    })

    print("=== v65 live top100 expansion observer ===")
    print("Mode: observe-only. No broker orders are sent.")
    print(f"Symbols: {len(symbols)} top ranked from {args.alpha_rank_csv}")
    print(f"Recorder: {recorder.session_dir}")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    ib.reqMarketDataType(args.market_data_type)

    tickers = {}
    states = {symbol: SymbolState(symbol=symbol) for symbol in symbols}
    contracts = []
    try:
        for symbol in symbols:
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                recorder.record_selection(SelectionEvent(symbol=symbol, stage="contract_qualification", decision="rejected", reason="could_not_qualify"))
                continue
            q = qualified[0]
            contracts.append((symbol, q))
            tickers[symbol] = ib.reqMktData(q, "", False, False)
            recorder.record_selection(SelectionEvent(symbol=symbol, stage="top100", decision="accepted", reason="alpha_rank_top_n"))
            print(f"Subscribed {symbol} conId={q.conId}")

        start = time.time()
        while time.time() - start < args.duration_seconds:
            ib.sleep(args.interval_seconds)
            elapsed = time.time() - start
            ready_count = 0
            for symbol, _ in contracts:
                snap = snapshot_from_ticker(symbol, tickers[symbol])
                if snap.get("price") is None:
                    continue
                record_snapshot(recorder, snap, args.session_type)
                state = states[symbol]
                update_state(state, snap, elapsed, args.opening_range_seconds)
                features = compute_live_safe_features(state, snap, args)

                recorder.record_selection(SelectionEvent(
                    symbol=symbol,
                    stage="live_safe_expansion",
                    decision="accepted" if features["ready"] else "rejected",
                    score=features["score"],
                    reason=features["reason"],
                    first_5m_high_pct=features.get("first_5m_high_pct"),
                    first_15m_high_pct=features.get("first_15m_high_pct"),
                    or_range_pct=features.get("or_range_pct"),
                    entry_price=features.get("entry_price"),
                    features_json=json.dumps(features, default=str),
                ))

                if features["ready"] and not state.signal_sent:
                    ready_count += 1
                    recorder.record_signal(SignalSnapshot(
                        symbol=symbol,
                        signal_name="v59_top100_live_safe_expansion",
                        action="BUY_INTENT",
                        score=features["score"],
                        threshold=0.0,
                        reasons=features["reason"],
                        features_json=json.dumps(features, default=str),
                    ))
                    recorder.record_order_intent(OrderIntent(
                        symbol=symbol,
                        action="BUY",
                        notional_usd=args.position_usd,
                        order_type="OBSERVE_ONLY_INTENT",
                        strategy="v59_top100_live_safe_expansion",
                        reason=features["reason"],
                        signal_score=features["score"],
                        features_json=json.dumps(features, default=str),
                    ))
                    state.signal_sent = True
                    print(f"BUY_INTENT observe-only {symbol} score={features['score']} price={features.get('entry_price')}")
            print(f"{now_utc()} scanned={len(contracts)} ready_new={ready_count}", flush=True)

    finally:
        for ticker in tickers.values():
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        ib.disconnect()
        print("Disconnected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
