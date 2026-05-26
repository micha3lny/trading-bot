from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.control.control_api import process_control_api_commands, process_history_collector_commands, start_control_api
from src.live_trading.market_calendar import get_us_equity_session, is_us_equity_trading_day, previous_us_equity_trading_day
from src.live_trading.v66_ibkr_account_recorder import (
    install_commission_report_handler,
    record_account_snapshot,
    record_recent_fills,
)
from src.live_trading.order_lifecycle.models import (
    ExecutionRecord,
    LifecycleEvent,
    LifecycleEventType,
    OrderSide,
    OrderState,
    PositionRecord,
    PositionState,
)
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore
from src.live_trading.order_lifecycle.reducer import reduce_lifecycle_events
from src.live_trading.order_lifecycle.reconciliation import build_reconciliation_report, log_reconciliation_report

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 65
DEFAULT_ALPHA_RANK = "data/universe/daily_top100_latest.csv"
DEFAULT_UNIVERSE = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_HISTORY_DIR = "data/history/universe_1m"
DEFAULT_RECORDER_DIR = "data/live/recorder"
STRATEGY_NAME = "v67_top100_live_safe_expansion_v46_wide_trail"


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
    first_seen_utc: str | None = None
    latest_seen_utc: str | None = None
    latest_volume: float | None = None
    signal_sent: bool = False
    bars: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ManagedPosition:
    symbol: str
    contract: Any
    quantity: int
    entry_price: float
    entry_time: str
    peak_price: float
    active: bool = True
    exit_sent: bool = False
    source: str = "live_buy"
    exit_order_id: int | None = None
    last_exit_order_ts: float | None = None
    eod_retry_count: int = 0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def append_dict_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


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


def snapshot_from_ticker(symbol: str, ticker: Any) -> dict[str, Any]:
    bid = safe_float(getattr(ticker, "bid", None))
    ask = safe_float(getattr(ticker, "ask", None))
    last = safe_float(getattr(ticker, "last", None))
    close = safe_float(getattr(ticker, "close", None))
    volume = safe_float(getattr(ticker, "volume", None))
    bid_size = safe_float(getattr(ticker, "bidSize", None))
    ask_size = safe_float(getattr(ticker, "askSize", None))

    mid = None
    spread = None
    spread_bps = None
    if bid is not None and ask is not None and ask >= bid and bid > 0:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10_000.0 if mid else None

    price = last or mid or close or bid or ask
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


def market_open_datetime_utc(args: argparse.Namespace, now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
    return now.replace(hour=hh, minute=mm, second=0, microsecond=0)


def update_state(
    state: SymbolState,
    snap: dict[str, Any],
    session_elapsed: float,
    opening_range_seconds: int,
    *,
    observed_at: datetime | None = None,
) -> None:
    price = safe_float(snap.get("price"))
    if price is None or price <= 0:
        return
    observed_at = observed_at or datetime.now(timezone.utc)
    now_ts = observed_at.timestamp()
    observed_iso = observed_at.isoformat()
    state.last_price = price
    state.latest_seen_utc = observed_iso
    state.latest_volume = safe_float(snap.get("volume"))
    state.bars.append(
        {
            "bar_time_utc": observed_iso,
            "price": price,
            "session_elapsed_seconds": round(float(session_elapsed), 3),
            "source": "live_ticker_snapshot",
        }
    )
    if len(state.bars) > 500:
        del state.bars[:-500]

    if session_elapsed < 0:
        return

    if state.first_seen_ts is None:
        state.first_seen_ts = now_ts
        state.first_seen_utc = observed_iso
        state.first_price = price
        state.open_price = price
        state.high = price
        state.low = price
    state.high = max(state.high or price, price)
    state.low = min(state.low or price, price)
    if 0 <= session_elapsed < 5 * 60:
        state.first_5m_high = max(state.first_5m_high or price, price)
    if 0 <= session_elapsed < 15 * 60:
        state.first_15m_high = max(state.first_15m_high or price, price)
    if 0 <= session_elapsed < opening_range_seconds:
        state.or_high = max(state.or_high or price, price)
        state.or_low = min(state.or_low or price, price)


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
    for value, weight in [(first_5m_high_pct, 2.0), (first_15m_high_pct, 2.0), (or_range_pct, 1.0)]:
        if value is not None:
            score += value * weight
    if spread_bps is not None and args.max_spread_bps > 0:
        score += max(0.0, args.max_spread_bps - spread_bps) / args.max_spread_bps * 5.0

    reasons: list[str] = []
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
    }


def _feature_value_is_zeroish(value: Any) -> bool:
    numeric = safe_float(value)
    return numeric is None or abs(numeric) <= 1e-9


def features_are_all_zeroish(features: dict[str, Any]) -> bool:
    return (
        _feature_value_is_zeroish(features.get("first_5m_high_pct"))
        and _feature_value_is_zeroish(features.get("first_15m_high_pct"))
        and _feature_value_is_zeroish(features.get("or_range_pct"))
        and _feature_value_is_zeroish(features.get("score"))
    )


def log_live_feature_debug(
    *,
    runtime_state: dict[str, Any],
    symbol: str,
    state: SymbolState,
    snap: dict[str, Any],
    features: dict[str, Any],
    market_open: datetime,
    session_elapsed: float,
    reason: str,
    interval_seconds: float = 60.0,
) -> None:
    now_ts = time.time()
    key = f"live_feature_debug_last_{reason}"
    if now_ts - float(runtime_state.get(key) or 0.0) < interval_seconds:
        return
    runtime_state[key] = now_ts
    first_bar = state.bars[0] if state.bars else {}
    last_bar = state.bars[-1] if state.bars else {}
    print(
        f"{now_utc()} LIVE_FEATURE_DEBUG reason={reason} symbol={symbol} "
        f"session_start={market_open.isoformat()} session_elapsed_seconds={session_elapsed:.1f} "
        f"open_price={state.open_price} first_price={state.first_price} current_price={snap.get('price')} "
        f"last_price={state.last_price} first_5m_high={state.first_5m_high} "
        f"first_15m_high={state.first_15m_high} or_high={state.or_high} or_low={state.or_low} "
        f"first_5m_high_pct={features.get('first_5m_high_pct')} "
        f"first_15m_high_pct={features.get('first_15m_high_pct')} "
        f"or_range_pct={features.get('or_range_pct')} score={features.get('score')} "
        f"spread_bps={features.get('spread_bps')} first_seen_utc={state.first_seen_utc} "
        f"latest_seen_utc={state.latest_seen_utc} first_candle_ts={first_bar.get('bar_time_utc')} "
        f"latest_candle_ts={last_bar.get('bar_time_utc')} candle_samples={len(state.bars)}",
        flush=True,
    )


def record_lifecycle(recorder: LiveDataRecorder, event: str, symbol: str, **kwargs: Any) -> None:
    fields = [
        "recorded_at", "strategy", "event", "symbol", "action", "quantity", "price", "order_id",
        "execution_id", "reason", "entry_price", "peak_price", "pnl_pct",
        "decision_bid", "decision_ask", "decision_mid", "decision_last",
        "spread_pct", "fill_price", "fill_latency_ms",
        "estimated_commission", "realized_slippage_bps",
        "raw_json",
    ]
    row = {"recorded_at": now_utc(), "strategy": STRATEGY_NAME, "event": event, "symbol": symbol, **kwargs}
    raw = row.get("raw_json")
    if raw and not isinstance(raw, str):
        row["raw_json"] = json.dumps(raw, ensure_ascii=False, default=str)
    append_dict_csv(recorder.path("trade_lifecycle.csv"), row, fields)


def formal_event_type_for_legacy_event(event: str) -> LifecycleEventType | None:
    mapping = {
        "SIGNAL_READY": LifecycleEventType.ENTRY_SIGNAL,
        "BUY_ORDER_SENT": LifecycleEventType.ENTRY_ORDER_SUBMITTED,
        "SELL_ORDER_SENT": LifecycleEventType.EXIT_ORDER_SUBMITTED,
        "EOD_FLATTEN_SUBMIT": LifecycleEventType.EXIT_ORDER_SUBMITTED,
        "MANUAL_FLATTEN_SENT": LifecycleEventType.EXIT_ORDER_SUBMITTED,
        "MANUAL_FLATTEN_QUEUED": LifecycleEventType.EXIT_ORDER_PREPARED,
        "MANUAL_FLATTEN_DRY_RUN": LifecycleEventType.EXIT_ORDER_PREPARED,
        "MANUAL_FLATTEN_FAILED": LifecycleEventType.EXIT_ORDER_REJECTED,
        "EOD_FLATTEN_FAILED": LifecycleEventType.EXIT_ORDER_REJECTED,
        "EXIT_ORDER_CANCEL_REQUESTED": LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED,
        "EXIT_ORDER_CANCELLED": LifecycleEventType.EXIT_ORDER_CANCELLED,
        "ORDER_CANCEL_CONFIRMED": LifecycleEventType.EXIT_ORDER_CANCELLED,
        "ORDER_STALE": LifecycleEventType.EXIT_ORDER_STALE,
        "ADOPTED_POSITION": LifecycleEventType.POSITION_ADOPTED,
        "RESTORED_MANAGED_POSITION": LifecycleEventType.POSITION_ADOPTED,
        "POSITION_VERIFIED_CLOSED": LifecycleEventType.POSITION_CLOSED,
        "POSITION_QUANTITY_DRIFT": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "POSITION_MISSING_IN_IBKR": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "ORPHAN_IBKR_POSITION_OBSERVED": LifecycleEventType.POSITION_DRIFT_DETECTED,
        "FRACTIONAL_ORPHAN_MANUAL_REQUIRED": LifecycleEventType.POSITION_DRIFT_DETECTED,
    }
    return mapping.get(event)


def record_formal_lifecycle(
    recorder: LiveDataRecorder,
    event_type: LifecycleEventType,
    symbol: str,
    *,
    strategy: str = STRATEGY_NAME,
    order_state: OrderState | None = None,
    position_state: PositionState | None = None,
    state_before: PositionState | None = None,
    state_after: PositionState | None = None,
    client_order_id: str = "",
    order_id: Any = "",
    execution_id: str = "",
    quantity: Any = None,
    price: Any = None,
    reason: str = "",
    raw_json: dict[str, Any] | None = None,
) -> None:
    try:
        store = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl"))
        store.append_event(
            LifecycleEvent(
                event_type=event_type,
                symbol=symbol,
                strategy=strategy,
                state_before=state_before,
                state_after=state_after,
                client_order_id=client_order_id,
                ib_order_id=str(order_id or ""),
                execution_id=execution_id,
                order_state=order_state,
                position_state=position_state,
                quantity=safe_float(quantity),
                price=safe_float(price),
                reason=reason,
                raw_json=raw_json or {},
            )
        )
    except Exception as exc:
        print(f"{now_utc()} formal_lifecycle_record_error event={event_type} symbol={symbol} error={exc!r}", flush=True)


def record_lifecycle_with_formal(recorder: LiveDataRecorder, event: str, symbol: str, **kwargs: Any) -> None:
    record_lifecycle(recorder, event, symbol, **kwargs)
    event_type = formal_event_type_for_legacy_event(event)
    if event_type is None:
        return
    order_state = None
    position_state = None
    state_before = None
    state_after = None
    if event_type == LifecycleEventType.ENTRY_SIGNAL:
        position_state = PositionState.NONE
    elif event_type == LifecycleEventType.ENTRY_ORDER_SUBMITTED:
        order_state = OrderState.SUBMITTED
        state_before = PositionState.NONE
        state_after = PositionState.ENTRY_PENDING
        position_state = PositionState.ENTRY_PENDING
    elif event_type == LifecycleEventType.EXIT_ORDER_SUBMITTED:
        order_state = OrderState.SUBMITTED
        state_before = PositionState.OPEN
        state_after = PositionState.EXIT_PENDING
        position_state = PositionState.EXIT_PENDING
    elif event_type == LifecycleEventType.EXIT_ORDER_REJECTED:
        order_state = OrderState.REJECTED
        state_before = PositionState.OPEN
        state_after = PositionState.RECONCILING
        position_state = PositionState.RECONCILING
    elif event_type in {LifecycleEventType.ENTRY_ORDER_CANCELLED, LifecycleEventType.EXIT_ORDER_CANCELLED}:
        order_state = OrderState.CANCELLED
        state_after = PositionState.RECONCILING
        position_state = PositionState.RECONCILING
    elif event_type in {LifecycleEventType.POSITION_ADOPTED, LifecycleEventType.POSITION_DRIFT_DETECTED}:
        position_state = PositionState.RECONCILING if event_type == LifecycleEventType.POSITION_DRIFT_DETECTED else PositionState.OPEN
    elif event_type == LifecycleEventType.POSITION_CLOSED:
        state_after = PositionState.CLOSED
        position_state = PositionState.CLOSED
    record_formal_lifecycle(
        recorder,
        event_type,
        symbol,
        order_state=order_state,
        position_state=position_state,
        state_before=state_before,
        state_after=state_after,
        order_id=kwargs.get("order_id", ""),
        execution_id=str(kwargs.get("execution_id", "") or ""),
        quantity=kwargs.get("quantity"),
        price=kwargs.get("price"),
        reason=str(kwargs.get("reason", "") or ""),
        raw_json=kwargs.get("raw_json") if isinstance(kwargs.get("raw_json"), dict) else {"legacy_event": event},
    )


def load_existing_fill_keys(recorder: LiveDataRecorder) -> set[str]:
    path = recorder.path("fills.csv")
    seen: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return seen
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("execution_id") or f"{row.get('order_id','')}-{row.get('symbol','')}-{row.get('recorded_at','')}"
                if key.strip("-"):
                    seen.add(key)
    except Exception:
        return seen
    return seen


def load_exit_sent_symbols(recorder: LiveDataRecorder) -> set[str]:
    path = recorder.path("trade_lifecycle.csv")
    symbols: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return symbols
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("event", "")).strip() == "SELL_ORDER_SENT":
                    sym = str(row.get("symbol", "")).upper().strip()
                    if sym and sym != "__ALL__":
                        symbols.add(sym)
    except Exception:
        return symbols
    return symbols


def managed_position_payload(pos: ManagedPosition) -> dict[str, Any]:
    return {
        "symbol": pos.symbol,
        "quantity": pos.quantity,
        "entry_price": pos.entry_price,
        "entry_time": pos.entry_time,
        "peak_price": pos.peak_price,
        "active": pos.active,
        "exit_sent": pos.exit_sent,
        "source": pos.source,
        "exit_order_id": pos.exit_order_id,
        "last_exit_order_ts": pos.last_exit_order_ts,
        "eod_retry_count": pos.eod_retry_count,
    }


def persist_managed_positions(recorder: LiveDataRecorder, positions: dict[str, ManagedPosition]) -> None:
    payload = {
        "recorded_at": now_utc(),
        "strategy": STRATEGY_NAME,
        "positions": {symbol: managed_position_payload(pos) for symbol, pos in positions.items() if pos.active},
    }
    recorder.path("managed_positions.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def restore_managed_positions(recorder: LiveDataRecorder, contract_by_symbol: dict[str, Any]) -> dict[str, ManagedPosition]:
    path = recorder.path("managed_positions.json")
    restored: dict[str, ManagedPosition] = {}
    if not path.exists():
        return restored
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for symbol, row in payload.get("positions", {}).items():
            symbol = str(symbol).upper()
            if symbol not in contract_by_symbol:
                continue
            qty = int(float(row.get("quantity", 0)))
            entry = safe_float(row.get("entry_price"))
            peak = safe_float(row.get("peak_price")) or entry
            if qty <= 0 or entry is None or entry <= 0:
                continue
            restored[symbol] = ManagedPosition(
                symbol=symbol,
                contract=contract_by_symbol[symbol],
                quantity=qty,
                entry_price=float(entry),
                entry_time=str(row.get("entry_time") or f"restored:{now_utc()}"),
                peak_price=float(peak or entry),
                active=bool(row.get("active", True)),
                exit_sent=bool(row.get("exit_sent", False)),
                source=str(row.get("source") or "restored"),
                exit_order_id=int(float(row["exit_order_id"])) if row.get("exit_order_id") not in (None, "", "None") else None,
                last_exit_order_ts=safe_float(row.get("last_exit_order_ts")),
                eod_retry_count=int(float(row.get("eod_retry_count") or 0)),
            )
    except Exception as exc:
        print(f"{now_utc()} managed_positions_restore_error={exc!r}", flush=True)
    return restored


def record_strategy_equity(recorder: LiveDataRecorder, positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]]) -> None:
    unrealized = 0.0
    gross = 0.0
    active_count = 0
    rows = []
    for symbol, pos in positions.items():
        if not pos.active or pos.exit_sent:
            continue
        price = safe_float((latest_snapshots.get(symbol) or {}).get("price"))
        if price is None:
            continue
        active_count += 1
        market_value = price * pos.quantity
        cost = pos.entry_price * pos.quantity
        pnl = market_value - cost
        unrealized += pnl
        gross += abs(market_value)
        rows.append({
            "symbol": symbol,
            "qty": pos.quantity,
            "entry": pos.entry_price,
            "price": price,
            "peak": pos.peak_price,
            "unrealized_pnl": pnl,
            "unrealized_pct": (price / pos.entry_price - 1.0) * 100.0 if pos.entry_price else None,
        })
    append_dict_csv(
        recorder.path("strategy_equity.csv"),
        {
            "recorded_at": now_utc(),
            "strategy": STRATEGY_NAME,
            "active_positions": active_count,
            "gross_exposure": gross,
            "unrealized_pnl": unrealized,
            "positions_json": json.dumps(rows, ensure_ascii=False, default=str),
        },
        ["recorded_at", "strategy", "active_positions", "gross_exposure", "unrealized_pnl", "positions_json"],
    )


def existing_candle_keys(recorder: LiveDataRecorder) -> set[tuple[str, str]]:
    path = recorder.path("candles_1m.csv")
    keys: set[tuple[str, str]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym = str(row.get("symbol", "")).upper()
                ts = str(row.get("bar_time", ""))
                if sym and ts:
                    keys.add((sym, ts))
    except Exception:
        return keys
    return keys


def backfill_recent_1m(ib: IB, recorder: LiveDataRecorder, contracts: list[tuple[str, Any]], args: argparse.Namespace) -> int:
    if not args.backfill_1m_on_start:
        return 0
    keys = existing_candle_keys(recorder)
    total = 0
    subset = contracts[: max(0, args.backfill_top_n)]
    for symbol, contract in subset:
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=args.backfill_duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            print(f"{now_utc()} backfill_1m_error symbol={symbol} error={exc!r}", flush=True)
            continue
        rows = []
        for bar in bars:
            bar_time = str(getattr(bar, "date", ""))
            key = (symbol, bar_time)
            if key in keys:
                continue
            keys.add(key)
            rows.append({
                "symbol": symbol,
                "bar_time": bar_time,
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "source": "ibkr_backfill_1m",
                "recorded_at": now_utc(),
            })
        if rows:
            total += recorder.record_candles_1m(rows)
        ib.sleep(args.backfill_pause_seconds)
    print(f"{now_utc()} backfill_1m_done symbols={len(subset)} rows={total}", flush=True)
    return total


def current_session_candle_count(recorder: LiveDataRecorder, args: argparse.Namespace) -> int:
    path = recorder.path("candles_1m.csv")
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        today = datetime.now(timezone.utc).date()
        market_open = datetime(today.year, today.month, today.day, hh, mm, tzinfo=timezone.utc)
    except Exception:
        return 0
    count = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_bar_time_utc(row.get("bar_time", ""))
                if ts is not None and ts >= market_open:
                    count += 1
    except Exception:
        return 0
    return count


def backfill_current_session_1m(
    ib: IB,
    recorder: LiveDataRecorder,
    contracts: list[tuple[str, Any]],
    args: argparse.Namespace,
) -> int:
    if not getattr(args, "backfill_current_session_on_rebuild_miss", True):
        return 0

    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        now = datetime.now(timezone.utc)
        market_open = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception as exc:
        print(f"{now_utc()} current_session_backfill_skipped reason=bad_market_open error={exc!r}", flush=True)
        return 0

    if now < market_open:
        print(f"{now_utc()} current_session_backfill_skipped reason=before_market_open", flush=True)
        return 0

    duration_seconds = max(60, int((now - market_open).total_seconds()) + 300)
    duration = f"{duration_seconds} S"
    keys = existing_candle_keys(recorder)
    total = 0
    subset = contracts[: max(0, int(getattr(args, "backfill_top_n", len(contracts))))]
    print(f"{now_utc()} current_session_backfill_start symbols={len(subset)} duration={duration}", flush=True)

    for symbol, contract in subset:
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            print(f"{now_utc()} current_session_backfill_error symbol={symbol} error={exc!r}", flush=True)
            continue

        rows = []
        for bar in bars:
            bar_time = str(getattr(bar, "date", ""))
            ts = _parse_bar_time_utc(bar_time)
            if ts is None or ts < market_open:
                continue
            key = (symbol, bar_time)
            if key in keys:
                continue
            keys.add(key)
            rows.append({
                "symbol": symbol,
                "bar_time": bar_time,
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "source": "ibkr_current_session_backfill_1m",
                "recorded_at": now_utc(),
            })
        if rows:
            total += recorder.record_candles_1m(rows)
        ib.sleep(float(getattr(args, "backfill_pause_seconds", 0.15)))

    print(f"{now_utc()} current_session_backfill_done symbols={len(subset)} rows={total}", flush=True)
    return total


def is_eod_flatten_time(flatten_utc: str) -> bool:
    try:
        hh, mm = [int(x) for x in flatten_utc.split(":", 1)]
        return datetime.now(timezone.utc).time() >= dtime(hour=hh, minute=mm, tzinfo=timezone.utc)
    except Exception:
        return False


def send_exit_order(ib: IB, recorder: LiveDataRecorder, pos: ManagedPosition, reason: str, price: float | None) -> bool:
    if not pos.active or pos.quantity <= 0:
        return False
    order = MarketOrder("SELL", pos.quantity)
    order.tif = "DAY"
    order.outsideRth = False
    trade = ib.placeOrder(pos.contract, order)
    order_id = trade.order.orderId
    pnl_pct = ((price / pos.entry_price - 1.0) * 100.0) if price and pos.entry_price > 0 else None
    record_lifecycle_with_formal(
        recorder,
        "SELL_ORDER_SENT",
        pos.symbol,
        action="SELL",
        quantity=pos.quantity,
        price=price,
        order_id=order_id,
        reason=reason,
        entry_price=pos.entry_price,
        peak_price=pos.peak_price,
        pnl_pct=pnl_pct,
        decision_bid=None,
        decision_ask=None,
        decision_mid=None,
        decision_last=price,
        spread_pct=None,
    )
    pos.exit_sent = True
    pos.exit_order_id = order_id
    pos.last_exit_order_ts = time.time()
    pnl_txt = f" pnl_pct={pnl_pct:.2f}" if pnl_pct is not None else ""
    print(
        f"PAPER SELL SENT symbol={pos.symbol} qty={pos.quantity} "
        f"reason={reason} entry={pos.entry_price:.2f} price={price if price else 0:.2f}"
        f"{pnl_txt} orderId={order_id} tif={order.tif} outsideRth={order.outsideRth}",
        flush=True,
    )
    return True


def ibkr_position_quantities(ib: IB) -> dict[str, float]:
    quantities: dict[str, float] = {}
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = safe_float(getattr(item, "position", None)) or 0.0
        if abs(qty) > 0:
            quantities[symbol] = quantities.get(symbol, 0.0) + qty
    return quantities


def ibkr_portfolio_position_rows(ib: IB) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ib.portfolio():
        contract = getattr(item, "contract", None)
        symbol = str(getattr(contract, "symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = safe_float(getattr(item, "position", None)) or 0.0
        if abs(qty) <= 0:
            continue
        rows.append({
            "symbol": symbol,
            "quantity": qty,
            "contract": contract,
            "market_price": safe_float(getattr(item, "marketPrice", None)),
            "market_value": safe_float(getattr(item, "marketValue", None)),
            "average_cost": safe_float(getattr(item, "averageCost", None)),
        })
    return rows


def same_position_direction(managed_qty: float, ibkr_qty: float) -> bool:
    return (managed_qty > 0 and ibkr_qty > 0) or (managed_qty < 0 and ibkr_qty < 0)


def whole_share_quantity(quantity: float) -> bool:
    return not is_fractional_position_quantity(quantity)


def open_ibkr_order_trades(ib: IB) -> list[Any]:
    out: list[Any] = []
    try:
        out.extend(list(ib.openTrades()))
    except Exception:
        pass
    seen_order_ids = {order_trade_id(trade) for trade in out if order_trade_id(trade) is not None}
    try:
        for order in ib.openOrders():
            order_id = getattr(order, "orderId", None)
            if order_id in seen_order_ids:
                continue
            out.append(type("OpenOrderTrade", (), {"order": order, "contract": None})())
    except Exception:
        pass
    return out


def order_trade_symbol(trade: Any) -> str:
    return str(getattr(getattr(trade, "contract", None), "symbol", "") or "").upper().strip()


def order_trade_id(trade: Any) -> Any:
    return getattr(getattr(trade, "order", None), "orderId", None)


def order_trade_action(trade: Any) -> str:
    return str(getattr(getattr(trade, "order", None), "action", "") or "").upper().strip()


def order_trade_quantity(trade: Any) -> float | None:
    return safe_float(getattr(getattr(trade, "order", None), "totalQuantity", None))


def startup_reconcile_runtime_state(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    contract_by_symbol: dict[str, Any],
    runtime_state: dict[str, Any],
    *,
    cancel_stale_orders: bool = True,
    submit_orphan_flatten: bool = True,
    log_prefix: str = "STARTUP_RECONCILIATION",
    reason_prefix: str = "startup_reconciliation",
) -> dict[str, Any]:
    print(f"{now_utc()} {log_prefix}_START", flush=True)
    runtime_state["entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = reason_prefix

    lifecycle_events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
    reducer_snapshot = reduce_lifecycle_events(lifecycle_events)
    reducer_positions = reducer_snapshot.positions
    ibkr_rows = ibkr_portfolio_position_rows(ib)
    ibkr_qty_by_symbol = {row["symbol"]: float(row["quantity"]) for row in ibkr_rows}
    ibkr_row_by_symbol = {row["symbol"]: row for row in ibkr_rows}

    closed_local: list[str] = []
    drift_symbols: list[str] = []
    orphans: list[str] = []
    fractional_orphans: list[str] = []
    whole_share_orphans: list[str] = []
    pending_orders: list[str] = []
    submitted_flatten_order_ids: set[Any] = set()

    candidate_symbols = sorted(set(managed_positions) | set(reducer_positions))
    for symbol in candidate_symbols:
        pos = managed_positions.get(symbol)
        reduced = reducer_positions.get(symbol)
        lifecycle_open = reduced is not None and reduced.state in {PositionState.OPEN, PositionState.ENTRY_PENDING, PositionState.EXIT_PENDING, PositionState.RECONCILING}
        local_open = pos is not None and pos.active
        ibkr_qty = float(ibkr_qty_by_symbol.get(symbol, 0.0))
        if (local_open or lifecycle_open) and abs(ibkr_qty) <= 0:
            if pos is not None and pos.active:
                pos.active = False
                pos.exit_sent = False
            closed_local.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_VERIFIED_CLOSED",
                symbol,
                action="VERIFY",
                quantity=getattr(pos, "quantity", None) if pos is not None else getattr(reduced, "open_quantity", None),
                reason=f"{reason_prefix}_ibkr_flat",
                raw_json={
                    "managed_active": bool(local_open),
                    "reducer_state": getattr(getattr(reduced, "state", None), "value", None),
                    "ibkr_quantity": ibkr_qty,
                },
            )
            print(f"{now_utc()} {log_prefix}_LOCAL_CLOSED symbol={symbol}", flush=True)
            continue
        if pos is not None and pos.active and abs(ibkr_qty) > 0 and abs(float(pos.quantity) - abs(ibkr_qty)) > 1e-9:
            drift_symbols.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_QUANTITY_DRIFT",
                symbol,
                action="VERIFY",
                quantity=pos.quantity,
                reason=f"{reason_prefix}_quantity_drift",
                raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ibkr_qty},
            )
            if same_position_direction(float(pos.quantity), ibkr_qty) and whole_share_quantity(ibkr_qty):
                old_qty = pos.quantity
                pos.quantity = int(abs(round(ibkr_qty)))
                pos.source = f"{pos.source}:{reason_prefix}_qty"
                print(
                    f"{now_utc()} {log_prefix}_DRIFT symbol={symbol} managed_quantity={old_qty} "
                    f"ibkr_quantity={ibkr_qty:.4f} action=managed_quantity_updated",
                    flush=True,
                )
            else:
                print(
                    f"{now_utc()} {log_prefix}_DRIFT symbol={symbol} managed_quantity={pos.quantity} "
                    f"ibkr_quantity={ibkr_qty:.4f} action=reconciling_manual_review",
                    flush=True,
                )

    for symbol in sorted(set(ibkr_qty_by_symbol) - {s for s, p in managed_positions.items() if p.active}):
        row = ibkr_row_by_symbol[symbol]
        ibkr_qty = float(row["quantity"])
        orphans.append(symbol)
        if is_fractional_position_quantity(ibkr_qty):
            fractional_orphans.append(symbol)
            record_fractional_orphan_manual_required(
                recorder,
                runtime_state,
                symbol=symbol,
                quantity=ibkr_qty,
                price=row.get("market_price"),
                reason=f"{reason_prefix}_fractional_orphan_manual_required",
                raw_json={"average_cost": row.get("average_cost"), "market_value": row.get("market_value")},
                log_prefix=f"{log_prefix}_FRACTIONAL_ORPHAN_MANUAL_REQUIRED",
            )
        else:
            whole_share_orphans.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "ORPHAN_IBKR_POSITION_OBSERVED",
                symbol,
                action="ALERT",
                quantity=ibkr_qty,
                price=row.get("market_price"),
                reason=f"{reason_prefix}_orphan_ibkr_position",
                raw_json={"ibkr_quantity": ibkr_qty, "average_cost": row.get("average_cost"), "market_value": row.get("market_value")},
            )
            print(f"{now_utc()} {log_prefix}_ORPHAN_DETECTED symbol={symbol} quantity={ibkr_qty:.4f}", flush=True)
        if submit_orphan_flatten:
            if whole_share_quantity(ibkr_qty):
                submitted_order_id = submit_eod_flatten_order(
                    ib,
                    recorder,
                    symbol=symbol,
                    contract=row["contract"],
                    ibkr_quantity=ibkr_qty,
                    reason=f"{reason_prefix}_orphan_flatten",
                    attempt=1,
                    runtime_state=runtime_state,
                )
                if submitted_order_id is not None:
                    submitted_flatten_order_ids.add(submitted_order_id)
            else:
                runtime_state["startup_reconciliation_fractional_manual_required"] = True

    for trade in open_ibkr_order_trades(ib):
        order_id = order_trade_id(trade)
        symbol = order_trade_symbol(trade)
        action = order_trade_action(trade)
        quantity = order_trade_quantity(trade)
        pending_orders.append(str(order_id or ""))
        if order_id in submitted_flatten_order_ids:
            print(f"{now_utc()} {log_prefix}_PENDING_FLATTEN_ORDER order_id={order_id} symbol={symbol}", flush=True)
            continue
        print(f"{now_utc()} {log_prefix}_PENDING_ORDER order_id={order_id} symbol={symbol}", flush=True)
        record_lifecycle_with_formal(
            recorder,
            "ORDER_STALE",
            symbol or "__UNKNOWN__",
            action=action,
            quantity=quantity,
            order_id=order_id,
            reason=f"{reason_prefix}_stale_open_order",
            raw_json={"order_id": order_id, "symbol": symbol, "action": action, "quantity": quantity},
        )
        if cancel_stale_orders:
            record_lifecycle_with_formal(
                recorder,
                "EXIT_ORDER_CANCEL_REQUESTED",
                symbol or "__UNKNOWN__",
                action=action,
                quantity=quantity,
                order_id=order_id,
                reason=f"{reason_prefix}_cancel_stale_open_order",
            )
            try:
                ib.cancelOrder(getattr(trade, "order", trade))
                print(f"{now_utc()} ORDER_CANCEL_CONFIRMED order_id={order_id} symbol={symbol}", flush=True)
                record_lifecycle_with_formal(
                    recorder,
                    "EXIT_ORDER_CANCELLED",
                    symbol or "__UNKNOWN__",
                    action=action,
                    quantity=quantity,
                    order_id=order_id,
                    reason=f"{reason_prefix}_cancel_sent",
                )
                record_lifecycle_with_formal(
                    recorder,
                    "ORDER_CANCEL_CONFIRMED",
                    symbol or "__UNKNOWN__",
                    action=action,
                    quantity=quantity,
                    order_id=order_id,
                    reason=f"{reason_prefix}_cancel_confirmed",
                )
            except Exception as exc:
                record_lifecycle_with_formal(
                    recorder,
                    "EXIT_ORDER_REJECTED",
                    symbol or "__UNKNOWN__",
                    action=action,
                    quantity=quantity,
                    order_id=order_id,
                    reason=f"{reason_prefix}_cancel_failed:{exc!r}",
                )

    clean = not closed_local and not drift_symbols and not orphans and not pending_orders and not reducer_snapshot.anomalies
    runtime_state["startup_reconciliation_done"] = True
    runtime_state["startup_reconciliation_clean"] = bool(clean)
    runtime_state["startup_reconciliation_orphans"] = sorted(orphans)
    runtime_state["startup_reconciliation_fractional_orphans"] = sorted(fractional_orphans)
    runtime_state["startup_reconciliation_whole_share_orphans"] = sorted(whole_share_orphans)
    runtime_state["startup_reconciliation_closed_local"] = sorted(closed_local)
    runtime_state["startup_reconciliation_pending_orders"] = sorted([p for p in pending_orders if p])
    runtime_state["startup_reconciliation_drift_symbols"] = sorted(drift_symbols)
    runtime_state["startup_reconciliation_anomalies"] = list(reducer_snapshot.anomalies)
    runtime_state["entries_blocked"] = False
    if runtime_state.get("entries_blocked_reason") == reason_prefix:
        runtime_state["entries_blocked_reason"] = ""
    persist_managed_positions(recorder, managed_positions)
    print(
        f"{now_utc()} {log_prefix}_DONE clean={int(clean)} "
        f"closed_local={len(closed_local)} orphans={len(orphans)} fractional_orphans={len(fractional_orphans)} "
        f"whole_share_orphans={len(whole_share_orphans)} drift={len(drift_symbols)} "
        f"pending_orders={len(pending_orders)} anomalies={len(reducer_snapshot.anomalies)}",
        flush=True,
    )
    return {
        "clean": clean,
        "closed_local": sorted(closed_local),
        "orphans": sorted(orphans),
        "fractional_orphans": sorted(fractional_orphans),
        "whole_share_orphans": sorted(whole_share_orphans),
        "drift_symbols": sorted(drift_symbols),
        "pending_orders": sorted([p for p in pending_orders if p]),
        "anomalies": list(reducer_snapshot.anomalies),
    }


def _flatten_action_for_quantity(quantity: float) -> str:
    return "SELL" if quantity > 0 else "BUY"


def is_fractional_position_quantity(quantity: float) -> bool:
    qty = abs(float(quantity))
    return not math.isclose(qty, round(qty), rel_tol=0.0, abs_tol=1e-9)


def record_fractional_orphan_manual_required(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    symbol: str,
    quantity: float,
    price: float | None = None,
    reason: str,
    raw_json: dict[str, Any] | None = None,
    log_prefix: str = "FRACTIONAL_ORPHAN_MANUAL_REQUIRED",
) -> bool:
    seen = runtime_state.setdefault("fractional_orphan_manual_required_seen", {})
    suppressed = runtime_state.setdefault("fractional_orphan_manual_required_suppressed", {})
    if not isinstance(seen, dict):
        seen = {}
        runtime_state["fractional_orphan_manual_required_seen"] = seen
    if not isinstance(suppressed, dict):
        suppressed = {}
        runtime_state["fractional_orphan_manual_required_suppressed"] = suppressed

    key = f"{symbol}:{float(quantity):.8f}"
    if seen.get(key):
        suppressed[key] = int(suppressed.get(key, 0) or 0) + 1
        runtime_state["fractional_orphan_manual_required_suppressed_total"] = sum(int(v or 0) for v in suppressed.values())
        return False

    seen[key] = now_utc()
    payload = {
        "ibkr_quantity": quantity,
        "manual_action_required": "close_fractional_position_in_ibkr_desktop",
        **(raw_json or {}),
    }
    record_lifecycle_with_formal(
        recorder,
        "FRACTIONAL_ORPHAN_MANUAL_REQUIRED",
        symbol,
        action="ALERT",
        quantity=quantity,
        price=price,
        reason=reason,
        raw_json=payload,
    )
    print(
        f"{now_utc()} {log_prefix} symbol={symbol} quantity={quantity:.4f} "
        f"reason={reason} manual_action_required=ibkr_desktop",
        flush=True,
    )
    return True


def smart_stock_contract_for_flatten(symbol: str, contract: Any | None) -> Any:
    currency = str(getattr(contract, "currency", "") or "USD")
    return Stock(symbol, "SMART", currency)


def _order_quantity_from_position(quantity: float) -> int | float:
    qty = abs(float(quantity))
    rounded = round(qty)
    if abs(qty - rounded) <= 1e-6:
        return int(rounded)
    return qty


def open_flatten_order_keys(ib: IB) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    done_statuses = {"Cancelled", "ApiCancelled", "Filled", "Inactive"}
    try:
        trades = list(ib.openTrades())
    except Exception:
        trades = []
    for trade in trades:
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        status = getattr(getattr(trade, "orderStatus", None), "status", "")
        if status in done_statuses:
            continue
        symbol = str(getattr(contract, "symbol", "") or "").upper().strip()
        action = str(getattr(order, "action", "") or "").upper().strip()
        if symbol and action in {"BUY", "SELL"}:
            keys.add((symbol, action))
    return keys


def write_eod_final_status(
    recorder: LiveDataRecorder,
    runtime_state: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    pending_orders: int,
    managed_positions: dict[str, ManagedPosition],
    reason: str,
) -> dict[str, Any]:
    managed_open_symbols = sorted(symbol for symbol, pos in managed_positions.items() if pos.active)
    managed_open_set = set(managed_open_symbols)
    fractional_orphans = sorted(
        row["symbol"] for row in rows
        if is_fractional_position_quantity(float(row["quantity"]))
    )
    whole_share_orphans = sorted(
        row["symbol"] for row in rows
        if not is_fractional_position_quantity(float(row["quantity"])) and row["symbol"] not in managed_open_set
    )
    summary = {
        "recorded_at": now_utc(),
        "reason": reason,
        "clean": not rows and pending_orders == 0 and not managed_open_symbols,
        "open_positions": len(rows),
        "open_symbols": sorted(row["symbol"] for row in rows),
        "fractional_orphans": fractional_orphans,
        "whole_share_orphans": whole_share_orphans,
        "pending_orders": pending_orders,
        "managed_open": len(managed_open_symbols),
        "managed_open_symbols": managed_open_symbols,
    }
    runtime_state["eod_final_status"] = summary
    recorder.path("eod_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(
        f"{now_utc()} EOD_FINAL_STATUS clean={int(summary['clean'])} "
        f"open_positions={summary['open_positions']} fractional_orphans={len(fractional_orphans)} "
        f"whole_share_orphans={len(whole_share_orphans)} pending_orders={pending_orders} "
        f"managed_open={summary['managed_open']}",
        flush=True,
    )
    return summary


def submit_eod_flatten_order(
    ib: IB,
    recorder: LiveDataRecorder,
    *,
    symbol: str,
    contract: Any,
    ibkr_quantity: float,
    reason: str,
    attempt: int,
    runtime_state: dict[str, Any] | None = None,
) -> int | None:
    if abs(ibkr_quantity) <= 0:
        return None
    if is_fractional_position_quantity(ibkr_quantity):
        action = _flatten_action_for_quantity(ibkr_quantity)
        if runtime_state is not None:
            record_fractional_orphan_manual_required(
                recorder,
                runtime_state,
                symbol=symbol,
                quantity=ibkr_quantity,
                reason=f"{reason}_fractional_quantity_api_unsupported",
                raw_json={"attempt": attempt, "action": action},
            )
        else:
            record_lifecycle_with_formal(
                recorder,
                "EOD_FLATTEN_FAILED",
                symbol,
                action=action,
                quantity=abs(float(ibkr_quantity)),
                reason="fractional_quantity_api_unsupported",
                raw_json={
                    "ibkr_quantity": ibkr_quantity,
                    "attempt": attempt,
                    "manual_action_required": "close_fractional_position_in_ibkr_desktop",
                },
            )
            print(
                f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} quantity={ibkr_quantity:.4f} "
                "reason=fractional_quantity_api_unsupported manual_action_required=ibkr_desktop",
                flush=True,
            )
        return None
    action = _flatten_action_for_quantity(ibkr_quantity)
    quantity = _order_quantity_from_position(ibkr_quantity)
    if quantity <= 0:
        return None
    order_contract = smart_stock_contract_for_flatten(symbol, contract)
    try:
        qualified = ib.qualifyContracts(order_contract)
        if qualified:
            order_contract = qualified[0]
    except Exception as exc:
        record_lifecycle_with_formal(
            recorder,
            "EOD_FLATTEN_FAILED",
            symbol,
            action=action,
            quantity=quantity,
            reason=f"qualify_failed:{exc!r}",
            raw_json={"ibkr_quantity": ibkr_quantity, "attempt": attempt},
        )
        print(f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} reason=qualify_failed error={exc!r}", flush=True)
        return None
    order = MarketOrder(action, quantity)
    order.tif = "DAY"
    order.outsideRth = False
    trade = ib.placeOrder(order_contract, order)
    try:
        ib.sleep(0.5)
    except Exception:
        pass
    order_id = getattr(getattr(trade, "order", None), "orderId", None)
    status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
    if status.lower() == "cancelled":
        log = getattr(trade, "log", None)
        error_message = str(log[-1].message) if log else "cancelled"
        record_lifecycle_with_formal(
            recorder,
            "EOD_FLATTEN_FAILED",
            symbol,
            action=action,
            quantity=quantity,
            order_id=order_id,
            reason=f"ibkr_cancelled:{error_message}",
            raw_json={"ibkr_quantity": ibkr_quantity, "attempt": attempt, "status": status},
        )
        print(
            f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} action={action} quantity={quantity} "
            f"orderId={order_id} reason=ibkr_cancelled error={error_message}",
            flush=True,
        )
        return None
    record_lifecycle_with_formal(
        recorder,
        "EOD_FLATTEN_SUBMIT",
        symbol,
        action=action,
        quantity=quantity,
        order_id=order_id,
        reason=reason,
        raw_json={"ibkr_quantity": ibkr_quantity, "attempt": attempt, "tif": order.tif, "outsideRth": order.outsideRth, "status": status},
    )
    print(
        f"{now_utc()} EOD_FLATTEN_SUBMIT symbol={symbol} action={action} "
        f"quantity={quantity} ibkr_quantity={ibkr_quantity:.4f} attempt={attempt} "
        f"orderId={order_id} reason={reason}",
        flush=True,
    )
    return int(order_id) if order_id is not None else 0


def hard_eod_flatten_portfolio(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
    *,
    reason: str,
) -> int:
    try:
        if hasattr(ib, "isConnected") and not ib.isConnected():
            print(f"{now_utc()} EOD_FLATTEN_FAILED reason=ibkr_not_connected", flush=True)
            return 0
    except Exception as exc:
        print(f"{now_utc()} EOD_FLATTEN_FAILED reason=ibkr_connection_check_error error={exc!r}", flush=True)
        return 0

    now_ts = time.time()
    rows = ibkr_portfolio_position_rows(ib)
    open_keys = open_flatten_order_keys(ib)
    pending_order_count = len(open_ibkr_order_trades(ib))
    attempt_by_symbol = runtime_state.setdefault("eod_flatten_attempts_by_symbol", {})
    last_submit_by_symbol = runtime_state.setdefault("eod_flatten_last_submit_ts_by_symbol", {})
    if not isinstance(attempt_by_symbol, dict):
        attempt_by_symbol = {}
        runtime_state["eod_flatten_attempts_by_symbol"] = attempt_by_symbol
    if not isinstance(last_submit_by_symbol, dict):
        last_submit_by_symbol = {}
        runtime_state["eod_flatten_last_submit_ts_by_symbol"] = last_submit_by_symbol

    print(
        f"{now_utc()} EOD_FLATTEN_VERIFY open_positions={len(rows)} open_flatten_orders={len(open_keys)} reason={reason}",
        flush=True,
    )
    runtime_state["eod_last_verification"] = {
        "recorded_at": now_utc(),
        "reason": reason,
        "ibkr_open_symbols": sorted(row["symbol"] for row in rows),
        "managed_open_symbols": sorted(symbol for symbol, pos in managed_positions.items() if pos.active),
        "open_flatten_orders": sorted(f"{symbol}:{action}" for symbol, action in open_keys),
    }

    if not rows:
        for pos in managed_positions.values():
            if pos.active:
                pos.active = False
        runtime_state["manual_eod_flatten_requested"] = False
        runtime_state["manual_eod_flatten_force"] = False
        runtime_state["eod_flatten_attempts_by_symbol"] = {}
        runtime_state["eod_flatten_last_submit_ts_by_symbol"] = {}
        print(f"{now_utc()} EOD_FLATTEN_SUCCESS open_positions=0", flush=True)
        write_eod_final_status(
            recorder,
            runtime_state,
            rows=[],
            pending_orders=pending_order_count,
            managed_positions=managed_positions,
            reason=reason,
        )
        return 0

    submitted = 0
    force = bool(runtime_state.get("manual_eod_flatten_force", False))
    retry_seconds = float(getattr(args, "eod_retry_seconds", 60))
    max_retries = int(getattr(args, "eod_max_retries", 5))

    for row in rows:
        symbol = row["symbol"]
        ibkr_qty = float(row["quantity"])
        action = _flatten_action_for_quantity(ibkr_qty)
        key = (symbol, action)
        print(
            f"{now_utc()} EOD_POSITION_STILL_OPEN symbol={symbol} quantity={ibkr_qty:.4f} "
            f"action={action} has_open_flatten_order={int(key in open_keys)}",
            flush=True,
        )
        if key in open_keys:
            continue

        attempts = int(attempt_by_symbol.get(symbol, 0) or 0)
        last_submit = safe_float(last_submit_by_symbol.get(symbol))
        retry_due = last_submit is None or (now_ts - last_submit) >= retry_seconds
        if attempts > 0 and not retry_due and not force:
            continue
        if attempts > 0:
            print(f"{now_utc()} EOD_FLATTEN_RETRY symbol={symbol} attempt={attempts + 1}", flush=True)
        if attempts >= max_retries and not force:
            print(
                f"{now_utc()} EOD_FLATTEN_FAILED symbol={symbol} quantity={ibkr_qty:.4f} "
                f"attempts={attempts} reason=max_retries_exceeded_continuing",
                flush=True,
            )
        if submit_eod_flatten_order(
            ib,
            recorder,
            symbol=symbol,
            contract=row["contract"],
            ibkr_quantity=ibkr_qty,
            reason=reason,
            attempt=attempts + 1,
            runtime_state=runtime_state,
        ):
            submitted += 1
            attempt_by_symbol[symbol] = attempts + 1
            last_submit_by_symbol[symbol] = now_ts
            pos = managed_positions.get(symbol)
            if pos is not None:
                pos.exit_sent = True
                pos.last_exit_order_ts = now_ts
                pos.eod_retry_count = attempts + 1

    if reason == "manual_eod_flatten":
        blocking_symbols = sorted(
            row["symbol"]
            for row in rows
            if (managed_positions.get(row["symbol"]) is not None and managed_positions[row["symbol"]].active and not managed_positions[row["symbol"]].exit_sent)
        )
        if not blocking_symbols:
            orphan_symbols = sorted(row["symbol"] for row in rows)
            runtime_state["manual_eod_flatten_requested"] = False
            runtime_state["manual_eod_flatten_force"] = False
            runtime_state["entries_blocked"] = False
            print(
                f"{now_utc()} EOD_FLATTEN_ORPHANS_REMAINING_NOT_BLOCKING symbols={','.join(orphan_symbols)}",
                flush=True,
            )

    runtime_state["manual_eod_flatten_force"] = False
    try:
        pending_order_count = len(open_ibkr_order_trades(ib))
    except Exception:
        pass
    write_eod_final_status(
        recorder,
        runtime_state,
        rows=rows,
        pending_orders=pending_order_count,
        managed_positions=managed_positions,
        reason=reason,
    )
    return submitted


def verify_managed_positions_against_ibkr(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    *,
    reason: str,
) -> dict[str, Any]:
    quantities = ibkr_position_quantities(ib)
    open_symbols: list[str] = []
    closed_symbols: list[str] = []
    drift_symbols: list[str] = []

    for symbol, pos in managed_positions.items():
        if not pos.active:
            continue
        ib_qty = quantities.get(symbol, 0.0)
        if abs(ib_qty) <= 0:
            if not pos.exit_sent:
                drift_symbols.append(symbol)
                record_lifecycle_with_formal(
                    recorder,
                    "POSITION_MISSING_IN_IBKR",
                    symbol,
                    action="VERIFY",
                    quantity=pos.quantity,
                    reason=reason,
                    raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ib_qty},
                )
                continue
            pos.active = False
            closed_symbols.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_VERIFIED_CLOSED",
                symbol,
                action="VERIFY",
                quantity=pos.quantity,
                reason=reason,
                entry_price=pos.entry_price,
                peak_price=pos.peak_price,
            )
            continue
        open_symbols.append(symbol)
        if int(abs(ib_qty)) != int(abs(pos.quantity)):
            drift_symbols.append(symbol)
            record_lifecycle_with_formal(
                recorder,
                "POSITION_QUANTITY_DRIFT",
                symbol,
                action="VERIFY",
                quantity=pos.quantity,
                reason=reason,
                raw_json={"managed_quantity": pos.quantity, "ibkr_quantity": ib_qty},
            )

    result = {
        "recorded_at": now_utc(),
        "reason": reason,
        "ibkr_open_symbols": sorted(quantities.keys()),
        "managed_open_symbols": sorted(open_symbols),
        "closed_symbols": sorted(closed_symbols),
        "drift_symbols": sorted(drift_symbols),
    }
    return result


def managed_positions_to_lifecycle_positions(
    managed_positions: dict[str, ManagedPosition],
) -> dict[str, PositionRecord]:
    positions: dict[str, PositionRecord] = {}
    session_date = datetime.now(timezone.utc).date().isoformat()
    for symbol, pos in managed_positions.items():
        if not pos.active:
            continue
        open_qty = 0.0 if pos.exit_sent else float(pos.quantity)
        state = PositionState.EXIT_PENDING if pos.exit_sent else PositionState.OPEN
        positions[symbol] = PositionRecord(
            symbol=symbol,
            strategy=STRATEGY_NAME,
            session_date=session_date,
            state=state,
            target_quantity=float(pos.quantity),
            entry_filled_quantity=float(pos.quantity),
            exit_filled_quantity=0.0,
            avg_entry_price=pos.entry_price,
            open_quantity=open_qty,
            peak_price=pos.peak_price,
        )
    return positions


def ibkr_open_order_rows(ib: IB) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        trades = list(ib.openTrades())
    except Exception:
        trades = []
    for trade in trades:
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        status = getattr(trade, "orderStatus", None)
        rows.append({
            "symbol": str(getattr(contract, "symbol", "") or "").upper(),
            "ib_order_id": str(getattr(order, "orderId", "") or ""),
            "perm_id": str(getattr(order, "permId", "") or ""),
            "action": str(getattr(order, "action", "") or ""),
            "total_quantity": safe_float(getattr(order, "totalQuantity", None)),
            "status": str(getattr(status, "status", "") or ""),
            "filled": safe_float(getattr(status, "filled", None)),
            "remaining": safe_float(getattr(status, "remaining", None)),
        })
    if rows:
        return rows
    try:
        orders = list(ib.openOrders())
    except Exception:
        orders = []
    for order in orders:
        rows.append({
            "symbol": "",
            "ib_order_id": str(getattr(order, "orderId", "") or ""),
            "perm_id": str(getattr(order, "permId", "") or ""),
            "action": str(getattr(order, "action", "") or ""),
            "total_quantity": safe_float(getattr(order, "totalQuantity", None)),
            "status": "",
            "filled": None,
            "remaining": None,
        })
    return rows


def run_dry_run_reconciliation_report(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
    *,
    reason: str,
) -> None:
    try:
        positions = managed_positions_to_lifecycle_positions(managed_positions)
        ibkr_quantities = ibkr_position_quantities(ib)
        open_orders = ibkr_open_order_rows(ib)
        lifecycle_events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
        report = build_reconciliation_report(
            positions,
            ibkr_quantities,
            lifecycle_events=lifecycle_events,
            open_orders=open_orders,
        )
        runtime_state["reconciliation_last_report"] = report.to_dict()
        runtime_state["reconciliation_last_reason"] = reason
        log_reconciliation_report(report, prefix=now_utc())
    except Exception as exc:
        print(f"{now_utc()} RECONCILIATION_ERROR reason={reason} error={exc!r}", flush=True)


def adopt_existing_long_positions(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    latest_snapshots: dict[str, dict[str, Any]],
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any] | None = None,
) -> int:
    adopted = 0
    exit_sent_symbols = load_exit_sent_symbols(recorder)
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper().strip()
        if not symbol or symbol in managed_positions:
            continue
        if symbol in exit_sent_symbols:
            record_lifecycle_with_formal(
                recorder,
                "ADOPT_DESPITE_EXIT_SENT",
                symbol,
                action="ADOPT",
                reason="ibkr_portfolio_still_has_position_after_lifecycle_sell_order",
            )
        quantity = safe_float(getattr(item, "position", None))
        avg_cost = safe_float(getattr(item, "averageCost", None))
        market_price = safe_float(getattr(item, "marketPrice", None))
        if quantity is None or quantity <= 0:
            continue
        if is_fractional_position_quantity(quantity):
            state = runtime_state if runtime_state is not None else {}
            record_fractional_orphan_manual_required(
                recorder,
                state,
                symbol=symbol,
                quantity=quantity,
                price=market_price,
                reason="fractional_ibkr_position_requires_manual_desktop_close",
                raw_json={"average_cost": avg_cost, "market_price": market_price},
            )
            continue
        in_runtime_universe = symbol in contract_by_symbol
        if not in_runtime_universe:
            record_lifecycle_with_formal(
                recorder,
                "ORPHAN_IBKR_POSITION_OBSERVED",
                symbol,
                action="ALERT",
                quantity=quantity,
                price=market_price,
                reason="external_ibkr_position_not_adopted_into_strategy",
                raw_json={"ibkr_quantity": quantity, "average_cost": avg_cost, "market_price": market_price},
            )
            print(f"{now_utc()} ORPHAN_IBKR_POSITION_OBSERVED symbol={symbol} quantity={quantity:.4f} reason=external_not_adopted", flush=True)
            continue
        entry_price = avg_cost or market_price
        if entry_price is None or entry_price <= 0:
            continue
        contract = contract_by_symbol.get(symbol) or item.contract
        if contract is None:
            contract = Stock(symbol, "SMART", "USD")
        contract_by_symbol.setdefault(symbol, contract)
        snap_price = safe_float((latest_snapshots.get(symbol) or {}).get("price"))
        peak_price = max(entry_price, snap_price or market_price or entry_price)
        managed_positions[symbol] = ManagedPosition(
            symbol=symbol,
            contract=contract,
            quantity=int(quantity),
            entry_price=float(entry_price),
            entry_time=f"adopted_on_restart:{now_utc()}",
            peak_price=float(peak_price),
            source="adopted_from_ibkr_portfolio_top100" if in_runtime_universe else "adopted_from_ibkr_portfolio_external",
        )
        record_lifecycle_with_formal(
            recorder,
            "ADOPTED_POSITION",
            symbol,
            action="ADOPT",
            quantity=int(quantity),
            price=snap_price or market_price,
            entry_price=entry_price,
            peak_price=peak_price,
            reason="restart_recovery_from_ibkr_portfolio",
        )
        print(f"ADOPTED EXISTING POSITION symbol={symbol} qty={int(quantity)} entry={entry_price:.2f} peak={peak_price:.2f}", flush=True)
        adopted += 1
    return adopted


def manage_exits(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    runtime_state: dict[str, Any],
) -> int:
    exits = 0
    manual_eod = bool(runtime_state.get("manual_eod_flatten_requested", False))
    manual_eod_force = bool(runtime_state.get("manual_eod_flatten_force", False))
    eod = args.enable_eod_flatten and (manual_eod or is_eod_flatten_time(args.eod_flatten_utc))
    if not eod and not is_after_utc(getattr(args, "manage_exits_start_utc", "13:30")):
        return 0

    if eod:
        runtime_state["entries_blocked"] = True
        try:
            exits += hard_eod_flatten_portfolio(
                ib,
                recorder,
                managed_positions,
                args,
                runtime_state,
                reason="manual_eod_flatten" if manual_eod else "scheduled_eod_flatten",
            )
            persist_managed_positions(recorder, managed_positions)
            return exits
        except Exception as exc:
            print(f"{now_utc()} EOD_FLATTEN_FAILED reason=portfolio_flatten_exception error={exc!r}", flush=True)

    for symbol, pos in list(managed_positions.items()):
        if not pos.active or pos.exit_sent:
            continue
        snap = latest_snapshots.get(symbol) or {}
        price = safe_float(snap.get("price"))

        if price is None or price <= 0:
            continue

        old_peak = pos.peak_price
        pos.peak_price = max(pos.peak_price, price)
        if pos.peak_price != old_peak:
            record_lifecycle(recorder, "PEAK_UPDATED", symbol, price=price, entry_price=pos.entry_price, peak_price=pos.peak_price)

        stop_price = pos.entry_price * (1.0 - args.exit_stop_loss_pct / 100.0)
        peak_pnl_pct = (pos.peak_price / pos.entry_price - 1.0) * 100.0
        reason = None
        if price <= stop_price:
            reason = "v46_wide_trail_stop_loss"
        elif peak_pnl_pct >= args.exit_trailing_activation_pct:
            trail_price = pos.peak_price * (1.0 - args.exit_trailing_stop_pct / 100.0)
            if price <= trail_price:
                reason = "v46_wide_trail_trailing_stop"
        if reason is None and eod:
            reason = "v46_wide_trail_close_exit_eod"
        if reason:
            if not pos.exit_sent and send_exit_order(ib, recorder, pos, reason, price):
                exits += 1
    return exits


def utc_minutes_now():
    from datetime import datetime, timezone
    t = datetime.now(timezone.utc).time()
    return t.hour * 60 + t.minute


def parse_utc_hhmm(value):
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    if not 0 <= hh <= 23 or not 0 <= mm <= 59:
        raise ValueError(f"Invalid UTC time HH:MM: {value}")
    return hh * 60 + mm


def is_after_utc(value):
    return utc_minutes_now() >= parse_utc_hhmm(value)


def parse_utc_schedule(value: str) -> list[str]:
    slots: list[str] = []
    for raw in str(value or "").split(","):
        slot = raw.strip()
        if not slot:
            continue
        parse_utc_hhmm(slot)
        slots.append(slot)
    return slots


def is_utc_slot_due(now: datetime, slot: str, *, window_minutes: int = 10) -> bool:
    now_min = now.hour * 60 + now.minute
    slot_min = parse_utc_hhmm(slot)
    return slot_min <= now_min < slot_min + max(1, int(window_minutes))


def latest_completed_trading_day(now: datetime, market_close_utc: str) -> date:
    close_min = parse_utc_hhmm(market_close_utc)
    now_min = now.hour * 60 + now.minute
    cur = now.date()
    if is_us_equity_trading_day(cur) and now_min >= close_min:
        return cur
    return previous_us_equity_trading_day(cur)


def _runtime_set(runtime_state: dict[str, Any], key: str) -> set[str]:
    value = runtime_state.setdefault(key, set())
    if isinstance(value, set):
        return value
    if isinstance(value, list):
        out = {str(item) for item in value}
        runtime_state[key] = out
        return out
    out: set[str] = set()
    runtime_state[key] = out
    return out


def overnight_backlog_start_date(end_date: str, args: argparse.Namespace) -> str:
    lookback_days = int(getattr(args, "overnight_backlog_lookback_days", 30) or 0)
    if lookback_days > 0:
        try:
            end = date.fromisoformat(str(end_date))
            return (end - timedelta(days=lookback_days)).isoformat()
        except Exception:
            pass
    return str(getattr(args, "overnight_collector_start_date", "2026-01-01"))


def enqueue_overnight_collector_if_due(runtime_state: dict[str, Any], args: argparse.Namespace, now: datetime | None = None) -> None:
    if not bool(getattr(args, "enable_overnight_automation", True)):
        return
    now = now or datetime.now(timezone.utc)
    slots = parse_utc_schedule(getattr(args, "overnight_collector_times_utc", ""))
    if not slots:
        return

    run_keys = _runtime_set(runtime_state, "overnight_collector_run_keys")
    skip_keys = _runtime_set(runtime_state, "overnight_collector_skip_logged_keys")
    queue = runtime_state.setdefault("history_collector_commands", [])
    if not isinstance(queue, list):
        queue = []
        runtime_state["history_collector_commands"] = queue

    backlog_slots = set(parse_utc_schedule(getattr(args, "overnight_backlog_collector_times_utc", "")))
    prioritize_previous_day = bool(getattr(args, "overnight_prioritize_previous_day", True))

    for slot in slots:
        if not is_utc_slot_due(now, slot):
            continue
        end_date = latest_completed_trading_day(now, getattr(args, "market_close_utc", "20:00")).isoformat()
        modes = ["daily"]
        if slot in backlog_slots:
            modes.append("backlog")
        if not prioritize_previous_day:
            modes = ["backlog"]
        key = f"{end_date}_{slot}_{'+'.join(modes)}"
        if key in run_keys:
            continue
        if runtime_state.get("history_collector_process") is not None or queue:
            if key not in skip_keys:
                print(
                    f"{now_utc()} OVERNIGHT_COLLECTOR_SKIPPED slot={slot} end={end_date} "
                    f"reason=collector_already_running_or_queued pending={len(queue)}",
                    flush=True,
                )
                skip_keys.add(key)
            continue
        queued_commands = []
        for mode in modes:
            command_id = f"overnight_{mode}_{end_date}_{slot.replace(':', '')}"
            start_date = end_date if mode == "daily" else overnight_backlog_start_date(end_date, args)
            max_tasks = (
                int(getattr(args, "overnight_daily_collector_max_tasks", 3000))
                if mode == "daily"
                else int(getattr(args, "overnight_collector_max_tasks", 3000))
            )
            command = {
                "id": command_id,
                "type": "history_collector",
                "source": "overnight_scheduler",
                "collector_mode": mode,
                "schedule_slot_utc": slot,
                "start_date": start_date,
                "end_date": end_date,
                "session_type": "RTH",
                "max_tasks": max_tasks,
                "max_attempts": int(getattr(args, "overnight_collector_max_attempts", 5)),
                "limit_symbols": 0,
                "client_id": int(getattr(args, "history_collector_client_id", 168)),
                "force": False,
                "allow_live_session": False,
                "plan_only": False,
                "include_weekends": False,
                "retry_failed": bool(getattr(args, "overnight_collector_retry_failed", False)),
            }
            queue.append(command)
            queued_commands.append(command)
        run_keys.add(key)
        skip_keys.discard(key)
        for command in queued_commands:
            print(
                f"{now_utc()} OVERNIGHT_COLLECTOR_QUEUED command_id={command['id']} mode={command['collector_mode']} "
                f"slot={slot} start={command['start_date']} end={end_date} max_tasks={command['max_tasks']}",
                flush=True,
            )


def process_daily_top100_build(runtime_state: dict[str, Any], args: argparse.Namespace, now: datetime | None = None) -> None:
    if not bool(getattr(args, "enable_overnight_automation", True)):
        return
    proc = runtime_state.get("daily_top100_process")
    if proc is not None:
        rc = proc.poll()
        if rc is None:
            return
        command = runtime_state.get("daily_top100_running_command") or {}
        if rc == 0:
            print(
                f"{now_utc()} DAILY_TOP100_BUILD_DONE ranking_date={command.get('ranking_date')} "
                f"returncode={rc} latest_output={command.get('latest_output')}",
                flush=True,
            )
            runtime_state["top100_reload_requested"] = True
            runtime_state["top100_reload_path"] = command.get("latest_output")
            runtime_state["top100_reload_ranking_date"] = command.get("ranking_date")
        else:
            print(
                f"{now_utc()} DAILY_TOP100_BUILD_FAILED ranking_date={command.get('ranking_date')} "
                f"returncode={rc} latest_output={command.get('latest_output')}",
                flush=True,
            )
        runtime_state["daily_top100_process"] = None
        runtime_state["daily_top100_running_command"] = None

    now = now or datetime.now(timezone.utc)
    slot = str(getattr(args, "daily_top100_build_utc", "12:45"))
    if not is_utc_slot_due(now, slot):
        return

    ranking_date = latest_completed_trading_day(now, getattr(args, "market_close_utc", "20:00")).isoformat()
    key = f"{ranking_date}_{slot}"
    run_keys = _runtime_set(runtime_state, "daily_top100_build_run_keys")
    if key in run_keys:
        return
    if runtime_state.get("daily_top100_process") is not None:
        return

    output_dir = Path(getattr(args, "daily_top100_output_dir", "data/universe"))
    dated_output = output_dir / f"daily_top100_{ranking_date}.csv"
    diagnostics_output = output_dir / f"daily_top100_{ranking_date}_diagnostics.csv"
    latest_output = Path(getattr(args, "daily_top100_latest_output", "data/universe/daily_top100_latest.csv"))
    command = [
        sys.executable,
        "-m",
        "src.live_trading.ranking.daily_top100_builder",
        "--date", ranking_date,
        "--universe", str(getattr(args, "daily_top100_universe", DEFAULT_UNIVERSE)),
        "--history-dir", str(getattr(args, "daily_top100_history_dir", DEFAULT_HISTORY_DIR)),
        "--output", str(dated_output),
        "--latest-output", str(latest_output),
        "--diagnostics-output", str(diagnostics_output),
        "--top-n", str(int(getattr(args, "daily_top100_top_n", 100))),
    ]
    sqlite_path = str(getattr(args, "daily_top100_sqlite_path", "data/runtime/rankings.sqlite"))
    if sqlite_path:
        command.extend(["--sqlite-path", sqlite_path])
    print(
        f"{now_utc()} DAILY_TOP100_BUILD_START ranking_date={ranking_date} "
        f"output={dated_output} latest_output={latest_output}",
        flush=True,
    )
    try:
        proc = subprocess.Popen(command)
    except Exception as exc:
        print(f"{now_utc()} DAILY_TOP100_BUILD_FAILED ranking_date={ranking_date} error={exc!r}", flush=True)
        run_keys.add(key)
        return
    runtime_state["daily_top100_process"] = proc
    runtime_state["daily_top100_running_command"] = {
        "ranking_date": ranking_date,
        "latest_output": str(latest_output),
        "command": command,
    }
    run_keys.add(key)


def process_overnight_automation(runtime_state: dict[str, Any], args: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc)
    enqueue_overnight_collector_if_due(runtime_state, args, now)
    process_daily_top100_build(runtime_state, args, now)


def reload_top100_universe_if_requested(
    ib: IB,
    recorder: LiveDataRecorder,
    states: dict[str, SymbolState],
    contracts: list[tuple[str, Any]],
    contract_by_symbol: dict[str, Any],
    tickers: dict[str, Any],
    latest_snapshots: dict[str, dict[str, Any]],
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not bool(runtime_state.get("top100_reload_requested", False)):
        return False

    reload_path = str(runtime_state.get("top100_reload_path") or getattr(args, "daily_top100_latest_output", args.alpha_rank_csv))
    ranking_date = runtime_state.get("top100_reload_ranking_date")
    print(f"{now_utc()} TOP100_RELOAD_START ranking_date={ranking_date} path={reload_path}", flush=True)
    runtime_state["entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = "top100_reload"

    try:
        entry_symbols = load_top_symbols(reload_path, int(args.top_n), min_price=args.min_price)
        if len(entry_symbols) < int(args.top_n):
            raise RuntimeError(f"top100_reload_too_few_symbols rows={len(entry_symbols)} required={int(args.top_n)}")

        active_symbols = sorted(symbol for symbol, pos in managed_positions.items() if pos.active)
        subscription_symbols = list(dict.fromkeys(entry_symbols + [s for s in active_symbols if s not in set(entry_symbols)]))
        previous_symbols = [symbol for symbol, _ in contracts]
        previous_symbol_set = set(previous_symbols)
        subscription_symbol_set = set(subscription_symbols)

        for symbol in sorted(previous_symbol_set - subscription_symbol_set):
            ticker = tickers.pop(symbol, None)
            contract = getattr(ticker, "contract", None) or contract_by_symbol.get(symbol)
            if contract is not None:
                try:
                    ib.cancelMktData(contract)
                except Exception as exc:
                    print(f"{now_utc()} TOP100_RELOAD_CANCEL_MKTDATA_FAILED symbol={symbol} error={exc!r}", flush=True)
            latest_snapshots.pop(symbol, None)

        new_contracts: list[tuple[str, Any]] = []
        new_contract_by_symbol: dict[str, Any] = {}
        subscribed = 0
        reused = 0
        failed_symbols: list[str] = []
        for symbol in subscription_symbols:
            contract = contract_by_symbol.get(symbol)
            if contract is None:
                contract = Stock(symbol, "SMART", "USD")
                try:
                    qualified = ib.qualifyContracts(contract)
                    if not qualified:
                        failed_symbols.append(symbol)
                        print(f"{now_utc()} TOP100_RELOAD_CONTRACT_FAILED symbol={symbol} reason=not_qualified", flush=True)
                        continue
                    contract = qualified[0]
                except Exception as exc:
                    failed_symbols.append(symbol)
                    print(f"{now_utc()} TOP100_RELOAD_CONTRACT_FAILED symbol={symbol} error={exc!r}", flush=True)
                    continue

            new_contracts.append((symbol, contract))
            new_contract_by_symbol[symbol] = contract
            states.setdefault(symbol, SymbolState(symbol=symbol))
            if symbol not in tickers:
                tickers[symbol] = ib.reqMktData(contract, "", False, False)
                subscribed += 1
                print(f"{now_utc()} TOP100_RELOAD_SUBSCRIBED symbol={symbol} conId={getattr(contract, 'conId', '')}", flush=True)
            else:
                reused += 1

        if len([s for s, _ in new_contracts if s in set(entry_symbols)]) < int(args.top_n):
            raise RuntimeError(
                f"top100_reload_qualified_too_few_entry_symbols "
                f"qualified={len([s for s, _ in new_contracts if s in set(entry_symbols)])} required={int(args.top_n)}"
            )

        for symbol in list(states):
            if symbol not in subscription_symbol_set and symbol not in active_symbols:
                states.pop(symbol, None)

        contracts[:] = new_contracts
        contract_by_symbol.clear()
        contract_by_symbol.update(new_contract_by_symbol)
        runtime_state["entry_symbols"] = set(entry_symbols)
        runtime_state["top100_reload_requested"] = False
        runtime_state["top100_reload_done_at"] = now_utc()
        runtime_state["top100_reload_last_error"] = ""
        runtime_state["top100_reload_symbols"] = list(entry_symbols)

        traded_symbols_today = load_traded_symbols_today(recorder)
        if traded_symbols_today and args.max_one_trade_per_symbol_per_day:
            for symbol in traded_symbols_today:
                if symbol in states:
                    states[symbol].signal_sent = True
        rebuilt = rebuild_symbol_states_from_1m_candles(recorder, states, args)

        print(
            f"{now_utc()} TOP100_RELOAD_DONE ranking_date={ranking_date} entry_symbols={len(entry_symbols)} "
            f"subscriptions={len(contracts)} added={subscribed} reused={reused} removed={len(previous_symbol_set - subscription_symbol_set)} "
            f"active_carried={len(active_symbols)} failed={len(failed_symbols)} state_rebuilt={rebuilt}",
            flush=True,
        )
        return True
    except Exception as exc:
        runtime_state["top100_reload_requested"] = False
        runtime_state["top100_reload_last_error"] = repr(exc)
        print(f"{now_utc()} TOP100_RELOAD_FAILED ranking_date={ranking_date} path={reload_path} error={exc!r}", flush=True)
        return False
    finally:
        runtime_state["entries_blocked"] = False
        if runtime_state.get("entries_blocked_reason") == "top100_reload":
            runtime_state["entries_blocked_reason"] = ""


def enrich_lifecycle_with_fills(recorder: LiveDataRecorder) -> int:
    lifecycle_path = recorder.path("trade_lifecycle.csv")
    fills_path = recorder.path("fills.csv")
    if not lifecycle_path.exists() or not fills_path.exists():
        return 0

    try:
        with lifecycle_path.open("r", newline="", encoding="utf-8") as f:
            lifecycle_rows = list(csv.DictReader(f))
            lifecycle_fields = list(f.reader.fieldnames or [])
        with fills_path.open("r", newline="", encoding="utf-8") as f:
            fills_rows = list(csv.DictReader(f))
    except Exception:
        return 0

    if not lifecycle_rows or not fills_rows:
        return 0

    needed = ["fill_price", "fill_latency_ms"]
    for field in needed:
        if field not in lifecycle_fields:
            lifecycle_fields.append(field)

    fills_by_order = {}
    for fill in fills_rows:
        oid = str(fill.get("order_id", "")).strip()
        if oid:
            fills_by_order[oid] = fill

    updated = 0
    for row in lifecycle_rows:
        event = str(row.get("event", ""))
        if event not in {"BUY_ORDER_SENT", "SELL_ORDER_SENT"}:
            continue
        if str(row.get("fill_price", "")).strip():
            continue

        oid = str(row.get("order_id", "")).strip()
        fill = fills_by_order.get(oid)
        if not fill:
            continue

        fill_price = (
            fill.get("price")
            or fill.get("avg_price")
            or fill.get("avgPrice")
            or fill.get("fill_price")
        )
        if fill_price:
            row["fill_price"] = fill_price
            side = str(row.get("action") or "").upper()
            event_type = LifecycleEventType.ENTRY_ORDER_FILLED if side == "BUY" else LifecycleEventType.EXIT_ORDER_FILLED
            position_state = PositionState.OPEN if side == "BUY" else PositionState.CLOSED
            try:
                execution_id = str(fill.get("execution_id") or fill.get("execId") or f"{oid}-{row.get('symbol','')}-{fill_price}")
                execution = ExecutionRecord(
                    execution_id=execution_id,
                    client_order_id="",
                    symbol=str(row.get("symbol") or ""),
                    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                    quantity=abs(float(row.get("quantity") or fill.get("quantity") or 0)),
                    price=float(fill_price),
                    ib_order_id=oid,
                    commission=safe_float(fill.get("commission")),
                    raw_json={"source": "enrich_lifecycle_with_fills"},
                )
                store = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl"))
                store.append_execution_once(
                    execution,
                    LifecycleEvent(
                        event_type=event_type,
                        symbol=execution.symbol,
                        strategy=STRATEGY_NAME,
                        state_after=position_state,
                        ib_order_id=oid,
                        execution_id=execution.execution_id,
                        order_state=OrderState.FILLED,
                        position_state=position_state,
                        quantity=execution.quantity,
                        price=execution.price,
                        reason="ibkr_execution_recorded",
                        raw_json={"legacy_order_event": event},
                    ),
                )
                record_formal_lifecycle(
                    recorder,
                    LifecycleEventType.POSITION_OPENED if side == "BUY" else LifecycleEventType.POSITION_CLOSED,
                    execution.symbol,
                    position_state=position_state,
                    state_after=position_state,
                    order_id=oid,
                    execution_id=execution.execution_id,
                    quantity=execution.quantity,
                    price=execution.price,
                    reason="fill_enriched_lifecycle",
                )
            except Exception as exc:
                print(f"{now_utc()} formal_fill_record_error order_id={oid} error={exc!r}", flush=True)

            try:
                fill_px = float(fill_price)
                qty = abs(float(row.get("quantity") or 0))
                decision_mid = row.get("decision_mid")

                # IBKR-style rough estimate
                est_commission = max(0.35, qty * 0.0035)
                row["estimated_commission"] = round(est_commission, 4)

                if decision_mid not in (None, "", "None"):
                    decision_mid = float(decision_mid)
                    if decision_mid > 0:
                        side = str(row.get("action") or "").upper()

                        if side == "BUY":
                            slippage_bps = ((fill_px - decision_mid) / decision_mid) * 10000.0
                        else:
                            slippage_bps = ((decision_mid - fill_px) / decision_mid) * 10000.0

                        row["realized_slippage_bps"] = round(slippage_bps, 2)
            except Exception:
                pass

        try:
            order_ts = datetime.fromisoformat(str(row.get("recorded_at")).replace("Z", "+00:00"))
            fill_ts_raw = fill.get("recorded_at") or fill.get("time") or fill.get("execution_time")
            fill_ts = datetime.fromisoformat(str(fill_ts_raw).replace("Z", "+00:00"))
            row["fill_latency_ms"] = int((fill_ts - order_ts).total_seconds() * 1000)
        except Exception:
            row["fill_latency_ms"] = ""

        updated += 1

    if updated:
        with lifecycle_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=lifecycle_fields)
            writer.writeheader()
            writer.writerows(lifecycle_rows)

    return updated


def load_traded_symbols_today(recorder: LiveDataRecorder) -> set[str]:
    path = recorder.path("trade_lifecycle.csv")
    symbols: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return symbols
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("event", "")).strip() in {"BUY_ORDER_SENT", "SIGNAL_READY"}:
                    sym = str(row.get("symbol", "")).upper().strip()
                    if sym:
                        symbols.add(sym)
    except Exception:
        return symbols
    return symbols


def count_entry_orders_today(recorder: LiveDataRecorder) -> int:
    path = recorder.path("trade_lifecycle.csv")
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("event", "")).strip() == "BUY_ORDER_SENT":
                    count += 1
    except Exception:
        return 0
    return count


def latest_strategy_equity_metrics(recorder: LiveDataRecorder) -> dict[str, float]:
    path = recorder.path("strategy_equity.csv")
    if not path.exists() or path.stat().st_size == 0:
        return {"unrealized_pnl": 0.0, "gross_exposure": 0.0}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {"unrealized_pnl": 0.0, "gross_exposure": 0.0}
    row = rows[-1] if rows else {}
    return {
        "unrealized_pnl": safe_float(row.get("unrealized_pnl")) or 0.0,
        "gross_exposure": safe_float(row.get("gross_exposure")) or 0.0,
        "active_positions": safe_float(row.get("active_positions")) or 0.0,
    }


def latest_portfolio_pnl_metrics(recorder: LiveDataRecorder) -> dict[str, float]:
    path = recorder.path("portfolio_snapshots.csv")
    if not path.exists() or path.stat().st_size == 0:
        return {"realized_pnl": 0.0, "unrealized_pnl": 0.0}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {"realized_pnl": 0.0, "unrealized_pnl": 0.0}
    row = rows[-1] if rows else {}
    return {
        "realized_pnl": safe_float(row.get("realized_pnl")) or 0.0,
        "unrealized_pnl": safe_float(row.get("unrealized_pnl")) or 0.0,
    }


def active_position_count(positions: dict[str, ManagedPosition]) -> int:
    return sum(1 for pos in positions.values() if bool(pos.active))


def managed_gross_exposure(positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]]) -> float:
    gross = 0.0
    for symbol, pos in positions.items():
        if not bool(pos.active):
            continue
        price = safe_float((latest_snapshots.get(symbol) or {}).get("price")) or safe_float(getattr(pos, "entry_price", None))
        qty = safe_float(getattr(pos, "quantity", None)) or 0.0
        if price and qty:
            gross += abs(price * qty)
    return gross


def evaluate_risk_guard(
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    symbol: str = "",
    candidate_notional: float = 0.0,
) -> dict[str, Any]:
    enabled = bool(getattr(args, "risk_guard_enabled", True))
    equity = latest_strategy_equity_metrics(recorder)
    portfolio = latest_portfolio_pnl_metrics(recorder)
    active_positions = active_position_count(managed_positions)
    gross_exposure = managed_gross_exposure(managed_positions, latest_snapshots) or safe_float(equity.get("gross_exposure")) or 0.0
    trades_today = count_entry_orders_today(recorder)
    realized_pnl = safe_float(portfolio.get("realized_pnl")) or 0.0
    unrealized_pnl = safe_float(portfolio.get("unrealized_pnl")) or safe_float(equity.get("unrealized_pnl")) or 0.0
    total_pnl = realized_pnl + unrealized_pnl
    effective_single_cap = float(getattr(args, "max_single_position_usd", 0.0) or 0.0)
    if effective_single_cap <= 0:
        effective_single_cap = float(getattr(args, "position_usd", 0.0) or 0.0)

    metrics: dict[str, Any] = {
        "enabled": enabled,
        "blocked": False,
        "reason": "",
        "symbol": symbol,
        "candidate_notional": round(float(candidate_notional or 0.0), 4),
        "daily_pnl": round(total_pnl, 4),
        "realized_pnl": round(realized_pnl, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "trades_today": trades_today,
        "active_positions": active_positions,
        "gross_exposure": round(gross_exposure, 4),
        "max_daily_loss_usd": float(getattr(args, "max_daily_loss_usd", 0.0) or 0.0),
        "max_trades_per_day": int(getattr(args, "max_trades_per_day", 0) or 0),
        "max_open_positions": int(getattr(args, "max_open_positions", 0) or 0),
        "max_gross_exposure_usd": float(getattr(args, "max_gross_exposure_usd", 0.0) or 0.0),
        "max_single_position_usd": effective_single_cap,
    }
    if not enabled:
        return metrics

    checks = [
        ("max_daily_loss", metrics["max_daily_loss_usd"] > 0 and total_pnl <= -metrics["max_daily_loss_usd"]),
        ("max_trades_per_day", metrics["max_trades_per_day"] > 0 and trades_today >= metrics["max_trades_per_day"]),
        ("max_open_positions", metrics["max_open_positions"] > 0 and active_positions >= metrics["max_open_positions"]),
        ("max_gross_exposure", metrics["max_gross_exposure_usd"] > 0 and gross_exposure >= metrics["max_gross_exposure_usd"]),
        ("max_single_position", effective_single_cap > 0 and candidate_notional > effective_single_cap),
    ]
    for reason, blocked in checks:
        if blocked:
            metrics["blocked"] = True
            metrics["reason"] = reason
            break
    return metrics


def _read_lifecycle_rows(recorder: LiveDataRecorder) -> list[dict[str, Any]]:
    path = recorder.path("trade_lifecycle.csv")
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _read_fill_rows(recorder: LiveDataRecorder) -> list[dict[str, Any]]:
    path = recorder.path("fills.csv")
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def process_fill_lifecycle_diagnostics(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    runtime_state: dict[str, Any],
) -> int:
    fills = _read_fill_rows(recorder)
    if not fills:
        return 0
    lifecycle_rows = _read_lifecycle_rows(recorder)
    orders: dict[str, dict[str, Any]] = {}
    cancelled_order_ids: set[str] = set()
    for row in lifecycle_rows:
        order_id = str(row.get("order_id") or "").strip()
        if not order_id:
            continue
        event = str(row.get("event") or "").strip()
        if event in {"BUY_ORDER_SENT", "SELL_ORDER_SENT", "EOD_FLATTEN_SUBMIT", "MANUAL_FLATTEN_SENT"}:
            orders.setdefault(order_id, row)
        if event in {"ENTRY_ORDER_CANCELLED", "EXIT_ORDER_CANCELLED", "ORDER_CANCEL_CONFIRMED", "ORDER_STALE"}:
            cancelled_order_ids.add(order_id)

    processed = runtime_state.setdefault("fill_diagnostic_execution_ids", set())
    if not isinstance(processed, set):
        processed = set(processed or [])
        runtime_state["fill_diagnostic_execution_ids"] = processed
    portfolio_qty = ibkr_position_quantities(ib)
    emitted = 0
    for fill in fills:
        execution_id = str(fill.get("execution_id") or "").strip()
        if not execution_id or execution_id in processed:
            continue
        processed.add(execution_id)
        order_id = str(fill.get("order_id") or "").strip()
        symbol = str(fill.get("symbol") or "").upper().strip()
        action = str(fill.get("action") or "").upper().strip()
        fill_qty = abs(safe_float(fill.get("quantity")) or 0.0)
        order_row = orders.get(order_id) or {}
        order_qty = abs(safe_float(order_row.get("quantity")) or 0.0)
        fill_price = safe_float(fill.get("fill_price"))

        if order_id and order_id in cancelled_order_ids:
            record_lifecycle(
                recorder,
                "DELAYED_FILL_AFTER_CANCEL",
                symbol,
                action=action,
                quantity=fill_qty,
                price=fill_price,
                order_id=order_id,
                execution_id=execution_id,
                reason="fill_arrived_after_cancelled_order",
                raw_json={"fill": fill, "order": order_row},
            )
            record_formal_lifecycle(
                recorder,
                LifecycleEventType.EXECUTION_RECORDED,
                symbol,
                order_id=order_id,
                execution_id=execution_id,
                quantity=fill_qty,
                price=fill_price,
                reason="delayed_fill_after_cancel",
                raw_json={"fill": fill, "order": order_row},
            )
            remaining = portfolio_qty.get(symbol, 0.0)
            if abs(remaining) > 0:
                record_lifecycle(
                    recorder,
                    "ORDER_CANCEL_BUT_POSITION_EXISTS",
                    symbol,
                    action=action,
                    quantity=abs(remaining),
                    order_id=order_id,
                    execution_id=execution_id,
                    reason="ibkr_position_exists_after_cancelled_order_fill",
                    raw_json={"ibkr_quantity": remaining},
                )
                runtime_state["cancel_but_position_exists_count"] = int(runtime_state.get("cancel_but_position_exists_count") or 0) + 1
            runtime_state["delayed_fill_after_cancel_count"] = int(runtime_state.get("delayed_fill_after_cancel_count") or 0) + 1
            try:
                runtime_state["delayed_fill_after_cancel_verification"] = verify_managed_positions_against_ibkr(
                    ib,
                    recorder,
                    managed_positions,
                    reason="delayed_fill_after_cancel",
                )
            except Exception as exc:
                runtime_state["delayed_fill_after_cancel_verification_error"] = repr(exc)
            print(f"{now_utc()} DELAYED_FILL_AFTER_CANCEL symbol={symbol} order_id={order_id} execution_id={execution_id} quantity={fill_qty}", flush=True)
            emitted += 1

        if order_qty > 0 and fill_qty > 0 and fill_qty + 1e-9 < order_qty:
            partial_event = "ENTRY_ORDER_PARTIAL" if action in {"BOT", "BUY"} else "EXIT_ORDER_PARTIAL"
            record_lifecycle(
                recorder,
                partial_event,
                symbol,
                action=action,
                quantity=fill_qty,
                price=fill_price,
                order_id=order_id,
                execution_id=execution_id,
                reason="ibkr_partial_fill",
                raw_json={"order_quantity": order_qty, "fill": fill, "order": order_row},
            )
            record_formal_lifecycle(
                recorder,
                LifecycleEventType.ENTRY_ORDER_PARTIAL if partial_event == "ENTRY_ORDER_PARTIAL" else LifecycleEventType.EXIT_ORDER_PARTIAL,
                symbol,
                order_state=OrderState.PARTIAL,
                position_state=PositionState.ENTRY_PENDING if partial_event == "ENTRY_ORDER_PARTIAL" else PositionState.EXIT_PENDING,
                order_id=order_id,
                execution_id=execution_id,
                quantity=fill_qty,
                price=fill_price,
                reason="ibkr_partial_fill",
                raw_json={"order_quantity": order_qty},
            )
            key = "partial_entry_count" if partial_event == "ENTRY_ORDER_PARTIAL" else "partial_exit_count"
            runtime_state[key] = int(runtime_state.get(key) or 0) + 1
            partial_states = runtime_state.setdefault("partial_fill_states", {})
            if isinstance(partial_states, dict):
                partial_states[str(order_id or execution_id)] = {
                    "state": "ENTRY_PARTIAL" if partial_event == "ENTRY_ORDER_PARTIAL" else "EXIT_PARTIAL",
                    "event": partial_event,
                    "symbol": symbol,
                    "order_id": order_id,
                    "execution_id": execution_id,
                    "fill_quantity": fill_qty,
                    "order_quantity": order_qty,
                }
            if partial_event == "EXIT_ORDER_PARTIAL" and symbol in managed_positions:
                remaining = portfolio_qty.get(symbol, 0.0)
                pos = managed_positions[symbol]
                if abs(remaining) > 0:
                    pos.quantity = int(round(abs(remaining)))
                    pos.active = True
                else:
                    pos.active = False
            print(f"{now_utc()} {partial_event} symbol={symbol} order_id={order_id} execution_id={execution_id} fill_quantity={fill_qty} order_quantity={order_qty}", flush=True)
            emitted += 1
    return emitted


def _parse_bar_time_utc(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None

    raw = raw.replace("US/Eastern", "").replace("America/New_York", "").strip()

    for fmt in (
        "%Y%m%d  %H:%M:%S",
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def rebuild_symbol_states_from_1m_candles(
    recorder: LiveDataRecorder,
    states: dict[str, SymbolState],
    args: argparse.Namespace,
) -> int:
    path = recorder.path("candles_1m.csv")
    if not path.exists() or path.stat().st_size == 0:
        print(f"{now_utc()} state_rebuild_skipped reason=no_candles_1m", flush=True)
        return 0

    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        now = datetime.now(timezone.utc)
        market_open = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception as exc:
        print(f"{now_utc()} state_rebuild_skipped reason=bad_market_open error={exc!r}", flush=True)
        return 0

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                symbol = str(row.get("symbol", "")).upper().strip()
                if symbol not in states:
                    continue
                ts = _parse_bar_time_utc(row.get("bar_time", ""))
                if ts is None or ts < market_open:
                    continue
                by_symbol.setdefault(symbol, []).append(row)
    except Exception as exc:
        print(f"{now_utc()} state_rebuild_error error={exc!r}", flush=True)
        return 0

    rebuilt = 0
    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: str(r.get("bar_time", "")))
        if not rows:
            continue

        first = safe_float(rows[0].get("open")) or safe_float(rows[0].get("close"))
        if first is None or first <= 0:
            continue

        st = states[symbol]
        st.first_seen_ts = time.time()
        first_ts = _parse_bar_time_utc(rows[0].get("bar_time", ""))
        st.first_seen_utc = first_ts.isoformat() if first_ts is not None else None
        st.latest_seen_utc = st.first_seen_utc
        st.first_price = first
        st.open_price = first
        st.high = first
        st.low = first
        st.first_5m_high = None
        st.first_15m_high = None
        st.or_high = None
        st.or_low = None
        st.bars = []

        for row in rows:
            ts = _parse_bar_time_utc(row.get("bar_time", ""))
            if ts is None:
                continue

            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            close = safe_float(row.get("close"))
            vol = safe_float(row.get("volume"))

            if high is None or low is None:
                continue

            minutes = (ts - market_open).total_seconds() / 60.0

            st.high = max(st.high or high, high)
            st.low = min(st.low or low, low)
            st.last_price = close or st.last_price
            st.latest_volume = vol or st.latest_volume
            st.latest_seen_utc = ts.isoformat()
            st.bars.append(
                {
                    "bar_time_utc": ts.isoformat(),
                    "open": safe_float(row.get("open")),
                    "high": high,
                    "low": low,
                    "close": close,
                    "session_elapsed_seconds": round(minutes * 60.0, 3),
                    "source": "candles_1m_rebuild",
                }
            )

            if 0 <= minutes < 5:
                st.first_5m_high = max(st.first_5m_high or high, high)
            if 0 <= minutes < 15:
                st.first_15m_high = max(st.first_15m_high or high, high)
            if 0 <= minutes < (args.opening_range_seconds / 60.0):
                st.or_high = max(st.or_high or high, high)
                st.or_low = min(st.or_low or low, low)

        rebuilt += 1

    print(f"{now_utc()} state_rebuild_done symbols={rebuilt} source=candles_1m market_open_utc={args.market_open_utc}", flush=True)
    return rebuilt


def connect_ibkr_with_retry(ib: IB, args: argparse.Namespace) -> None:
    attempts = 0
    while True:
        attempts += 1
        try:
            if ib.isConnected():
                return
            print(f"{now_utc()} RECONNECT_START attempt={attempts}", flush=True)
            ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
            ib.reqMarketDataType(args.market_data_type)
            print(f"{now_utc()} RECONNECT_DONE attempt={attempts}", flush=True)
            return
        except Exception as exc:
            print(f"{now_utc()} RECONNECT_FAILED attempt={attempts} error={exc!r}", flush=True)
            try:
                ib.disconnect()
            except Exception:
                pass
            if attempts >= getattr(args, "reconnect_max_attempts", 999999):
                raise
            time.sleep(getattr(args, "reconnect_wait_seconds", 15.0))


def resubscribe_market_data(ib: IB, contracts: list[tuple[str, Any]]) -> dict[str, Any]:
    tickers: dict[str, Any] = {}
    for symbol, contract in contracts:
        tickers[symbol] = ib.reqMktData(contract, "", False, False)
        print(f"RESUBSCRIBED {symbol} conId={getattr(contract, 'conId', '')}", flush=True)
    print(f"{now_utc()} RESUBSCRIBE_DONE symbols={len(tickers)}", flush=True)
    return tickers


def ibkr_connection_alive(ib: IB) -> bool:
    try:
        return bool(ib.isConnected())
    except Exception:
        return False


def handle_ibkr_disconnect_and_recover(
    ib: IB,
    recorder: LiveDataRecorder,
    managed_positions: dict[str, ManagedPosition],
    contract_by_symbol: dict[str, Any],
    contracts: list[tuple[str, Any]],
    runtime_state: dict[str, Any],
    args: argparse.Namespace,
    *,
    reason: str,
    seen_fills: set[str] | None = None,
    connect_fn: Any | None = None,
    resubscribe_fn: Any | None = None,
    reconcile_fn: Any | None = None,
    record_account_snapshot_fn: Any | None = None,
    record_recent_fills_fn: Any | None = None,
) -> dict[str, Any]:
    print(f"{now_utc()} RECONNECT_SUPERVISOR_START reason={reason}", flush=True)
    runtime_state["ibkr_connected"] = False
    runtime_state["reconnect_active"] = True
    runtime_state["post_reconnect_reconciliation_done"] = False
    runtime_state["entries_blocked"] = True
    runtime_state["entries_blocked_reason"] = "ibkr_reconnect"

    if ibkr_connection_alive(ib):
        try:
            ib.disconnect()
        except Exception:
            pass
    print(f"{now_utc()} RECONNECT_SUPERVISOR_DISCONNECTED reason={reason}", flush=True)

    connect_impl = connect_fn or connect_ibkr_with_retry
    resubscribe_impl = resubscribe_fn or resubscribe_market_data
    reconcile_impl = reconcile_fn or startup_reconcile_runtime_state
    account_snapshot_impl = record_account_snapshot_fn or record_account_snapshot
    recent_fills_impl = record_recent_fills_fn or record_recent_fills

    attempt = int(runtime_state.get("reconnect_attempts") or 0) + 1
    runtime_state["reconnect_attempts"] = attempt
    print(f"{now_utc()} RECONNECT_SUPERVISOR_ATTEMPT attempt={attempt} reason={reason}", flush=True)

    try:
        connect_impl(ib, args)
        install_commission_report_handler(ib, recorder)
        runtime_state["ibkr_connected"] = True
        runtime_state["reconnect_last_error"] = ""
        runtime_state["reconnect_last_success_at"] = now_utc()
        print(f"{now_utc()} RECONNECT_SUPERVISOR_CONNECTED attempt={attempt}", flush=True)

        tickers = resubscribe_impl(ib, contracts)
        print(f"{now_utc()} RECONNECT_SUPERVISOR_RESUBSCRIBED symbols={len(tickers)}", flush=True)

        try:
            account_snapshot_impl(ib, recorder)
        except Exception as exc:
            print(f"{now_utc()} RECONNECT_SUPERVISOR_ACCOUNT_RECORD_FAILED error={exc!r}", flush=True)
        if seen_fills is not None:
            try:
                recent_fills_impl(ib, recorder, seen_fills)
            except Exception as exc:
                print(f"{now_utc()} RECONNECT_SUPERVISOR_FILLS_RECORD_FAILED error={exc!r}", flush=True)

        reconciliation = reconcile_impl(
            ib,
            recorder,
            managed_positions,
            contract_by_symbol,
            runtime_state,
            log_prefix="POST_RECONNECT_RECONCILIATION",
            reason_prefix="post_reconnect_reconciliation",
        )
        runtime_state["post_reconnect_reconciliation_done"] = True
        runtime_state["post_reconnect_reconciliation_clean"] = bool(reconciliation.get("clean")) if isinstance(reconciliation, dict) else False
        runtime_state["post_reconnect_reconciliation_orphans"] = list(reconciliation.get("orphans", [])) if isinstance(reconciliation, dict) else []
        runtime_state["post_reconnect_reconciliation_pending_orders"] = list(reconciliation.get("pending_orders", [])) if isinstance(reconciliation, dict) else []
        runtime_state["reconnect_active"] = False
        runtime_state["entries_blocked"] = False
        if runtime_state.get("entries_blocked_reason") in {"ibkr_reconnect", "post_reconnect_reconciliation"}:
            runtime_state["entries_blocked_reason"] = ""
        return {"ok": True, "tickers": tickers, "reconciliation": reconciliation}
    except Exception as exc:
        runtime_state["ibkr_connected"] = False
        runtime_state["reconnect_active"] = True
        runtime_state["reconnect_last_error"] = repr(exc)
        runtime_state["post_reconnect_reconciliation_done"] = False
        runtime_state["entries_blocked"] = True
        runtime_state["entries_blocked_reason"] = "ibkr_reconnect"
        print(f"{now_utc()} RECONNECT_SUPERVISOR_FAILED attempt={attempt} error={exc!r}", flush=True)
        return {"ok": False, "tickers": {}, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="v67 live top100 expansion paper trader")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--alpha-rank-csv", default=DEFAULT_ALPHA_RANK)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--recorder-dir", default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--duration-seconds", type=int, default=28800)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--portfolio-interval-seconds", type=float, default=10.0)
    parser.add_argument("--market-data-type", type=int, default=1)
    parser.add_argument("--opening-range-seconds", type=int, default=15 * 60)
    parser.add_argument("--min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-spread-bps", type=float, default=50.0)
    parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--position-usd", type=float, default=1000.0)
    parser.add_argument("--risk-guard-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-daily-loss-usd", type=float, default=0.0)
    parser.add_argument("--max-trades-per-day", type=int, default=0)
    parser.add_argument("--max-open-positions", type=int, default=0)
    parser.add_argument("--max-gross-exposure-usd", type=float, default=0.0)
    parser.add_argument("--max-single-position-usd", type=float, default=0.0)
    parser.add_argument("--exit-stop-loss-pct", type=float, default=8.0)
    parser.add_argument("--exit-trailing-activation-pct", type=float, default=3.0)
    parser.add_argument("--exit-trailing-stop-pct", type=float, default=3.0)
    parser.add_argument("--adopt-existing-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-eod-flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eod-flatten-utc", default="19:45")
    parser.add_argument("--eod-retry-seconds", type=float, default=60.0)
    parser.add_argument("--eod-max-retries", type=int, default=5)
    parser.add_argument("--no-new-entries-after-utc", default="19:30")
    parser.add_argument("--market-open-utc", default="13:30")
    parser.add_argument("--market-close-utc", default="20:00")
    parser.add_argument("--new-entries-start-utc", default="13:35")
    parser.add_argument("--manage-exits-start-utc", default="13:30")
    parser.add_argument("--restart-cooldown-seconds", type=float, default=300.0)
    parser.add_argument("--backfill-1m-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backfill-duration", default="1 D")
    parser.add_argument("--backfill-top-n", type=int, default=100)
    parser.add_argument("--backfill-pause-seconds", type=float, default=0.15)
    parser.add_argument("--backfill-current-session-on-rebuild-miss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reconnect-wait-seconds", type=float, default=15.0)
    parser.add_argument("--reconnect-max-attempts", type=int, default=999999)
    parser.add_argument("--enable-overnight-automation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overnight-collector-times-utc", default="20:15,23:00,03:00,07:00")
    parser.add_argument("--overnight-backlog-collector-times-utc", default="07:00")
    parser.add_argument("--overnight-prioritize-previous-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overnight-collector-start-date", default="2026-01-01")
    parser.add_argument("--overnight-backlog-lookback-days", type=int, default=30)
    parser.add_argument("--overnight-daily-collector-max-tasks", type=int, default=3000)
    parser.add_argument("--overnight-collector-max-tasks", type=int, default=3000)
    parser.add_argument("--overnight-collector-max-attempts", type=int, default=5)
    parser.add_argument("--overnight-collector-retry-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-collector-client-id", type=int, default=168)
    parser.add_argument("--history-collector-max-runtime-minutes", type=float, default=120.0)
    parser.add_argument("--daily-top100-build-utc", default="12:45")
    parser.add_argument("--daily-top100-universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--daily-top100-history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--daily-top100-output-dir", default="data/universe")
    parser.add_argument("--daily-top100-latest-output", default="data/universe/daily_top100_latest.csv")
    parser.add_argument("--daily-top100-sqlite-path", default="data/runtime/rankings.sqlite")
    parser.add_argument("--daily-top100-top-n", type=int, default=100)
    args = parser.parse_args()

    symbols = load_top_symbols(args.alpha_rank_csv, args.top_n, min_price=args.min_price)
    recorder = LiveDataRecorder(args.recorder_dir)

    print("=== v67 live top100 expansion paper trader ===", flush=True)
    print(f"Symbols loaded: {len(symbols)}", flush=True)
    print(f"Recorder dir: {recorder.session_dir}", flush=True)
    print(f"Portfolio/fills recorder: integrated every {args.portfolio_interval_seconds}s", flush=True)
    print(
        "Exit: v46 wide_trail "
        f"stop_loss={args.exit_stop_loss_pct}% trail_activation={args.exit_trailing_activation_pct}% "
        f"trail_stop={args.exit_trailing_stop_pct}% adopt_existing={args.adopt_existing_positions} "
        f"eod_flatten={args.enable_eod_flatten} at {args.eod_flatten_utc} UTC",
        flush=True,
    )
    print(f"Backfill 1m: {args.backfill_1m_on_start} duration={args.backfill_duration} top_n={args.backfill_top_n}", flush=True)
    print(
        f"Overnight automation: {args.enable_overnight_automation} "
        f"collector_slots={args.overnight_collector_times_utc} "
        f"backlog_slots={args.overnight_backlog_collector_times_utc} "
        f"prioritize_previous_day={args.overnight_prioritize_previous_day} "
        f"backlog_lookback_days={args.overnight_backlog_lookback_days} "
        f"daily_top100_build={args.daily_top100_build_utc}",
        flush=True,
    )

    ib = IB()
    connect_ibkr_with_retry(ib, args)
    install_commission_report_handler(ib, recorder)

    tickers: dict[str, Any] = {}
    states = {symbol: SymbolState(symbol=symbol) for symbol in symbols}
    contracts: list[tuple[str, Any]] = []
    contract_by_symbol: dict[str, Any] = {}
    seen_fills: set[str] = load_existing_fill_keys(recorder)
    managed_positions: dict[str, ManagedPosition] = {}
    runtime_state = {
        "entries_blocked": False,
        "entries_blocked_until": time.time() + max(0.0, args.restart_cooldown_seconds),
        "entries_blocked_reason": "restart_cooldown" if args.restart_cooldown_seconds > 0 else "",
        "control_api_commands": [],
        "history_collector_start_utc": "20:15",
        "history_collector_end_utc": str(args.market_open_utc),
        "history_collector_max_tasks": int(args.overnight_collector_max_tasks),
        "history_collector_max_attempts": int(args.overnight_collector_max_attempts),
        "history_collector_client_id": int(args.history_collector_client_id),
        "history_collector_max_runtime_minutes": float(args.history_collector_max_runtime_minutes),
        "market_open_utc": str(args.market_open_utc),
        "market_close_utc": str(args.market_close_utc),
        "manual_eod_flatten_requested": False,
        "manual_eod_flatten_force": False,
        "startup_reconciliation_done": False,
        "startup_reconciliation_clean": False,
        "startup_reconciliation_orphans": [],
        "startup_reconciliation_fractional_orphans": [],
        "startup_reconciliation_whole_share_orphans": [],
        "startup_reconciliation_closed_local": [],
        "startup_reconciliation_pending_orders": [],
        "fractional_orphan_manual_required_seen": {},
        "fractional_orphan_manual_required_suppressed": {},
        "fractional_orphan_manual_required_suppressed_total": 0,
        "eod_final_status": {},
        "ibkr_connected": True,
        "reconnect_active": False,
        "reconnect_attempts": 0,
        "reconnect_last_error": "",
        "reconnect_last_success_at": "",
        "post_reconnect_reconciliation_done": False,
        "post_reconnect_reconciliation_clean": False,
        "post_reconnect_reconciliation_orphans": [],
        "post_reconnect_reconciliation_pending_orders": [],
        "entry_symbols": set(symbols),
        "top100_reload_requested": False,
        "top100_reload_path": "",
        "top100_reload_ranking_date": "",
        "top100_reload_last_error": "",
        "top100_reload_symbols": list(symbols),
        "market_closed_logged_dates": set(),
        "risk_guard_last_status": {
            "enabled": bool(args.risk_guard_enabled),
            "blocked": False,
            "reason": "",
        },
        "partial_entry_count": 0,
        "partial_exit_count": 0,
        "partial_fill_states": {},
        "delayed_fill_after_cancel_count": 0,
        "cancel_but_position_exists_count": 0,
    }
    latest_snapshots: dict[str, dict[str, Any]] = {}
    last_portfolio_record = 0.0
    adopted_once = False

    try:
        for symbol in symbols:
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                continue
            q = qualified[0]
            contracts.append((symbol, q))
            contract_by_symbol[symbol] = q
            tickers[symbol] = ib.reqMktData(q, "", False, False)
            print(f"Subscribed {symbol} conId={q.conId}", flush=True)

        restored = restore_managed_positions(recorder, contract_by_symbol)
        if restored:
            managed_positions.update(restored)
            for symbol, pos in restored.items():
                states.get(symbol, SymbolState(symbol)).signal_sent = True
                record_lifecycle_with_formal(recorder, "RESTORED_MANAGED_POSITION", symbol, quantity=pos.quantity, entry_price=pos.entry_price, peak_price=pos.peak_price, reason="managed_positions_json")
            print(f"{now_utc()} restored_managed_positions={len(restored)}", flush=True)

        backfilled_rows = backfill_recent_1m(ib, recorder, contracts, args)
        state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        current_session_backfilled_rows = 0
        today_session = get_us_equity_session(datetime.now(timezone.utc).date())
        if not today_session.is_trading_day:
            print(
                f"{now_utc()} MARKET_CLOSED_HOLIDAY date={today_session.session_date.isoformat()} "
                f"reason={today_session.reason}",
                flush=True,
            )
            _runtime_set(runtime_state, "market_closed_logged_dates").add(today_session.session_date.isoformat())
        if today_session.is_trading_day and state_rebuild_count == 0 and current_session_candle_count(recorder, args) == 0:
            current_session_backfilled_rows = backfill_current_session_1m(ib, recorder, contracts, args)
            if current_session_backfilled_rows:
                state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        traded_symbols_today = load_traded_symbols_today(recorder)
        if traded_symbols_today:
            for symbol in traded_symbols_today:
                if symbol in states and args.max_one_trade_per_symbol_per_day:
                    states[symbol].signal_sent = True
            print(f"{now_utc()} traded_symbols_today_loaded={len(traded_symbols_today)} max_one_trade_per_symbol_per_day={args.max_one_trade_per_symbol_per_day}", flush=True)

        control_api_server = start_control_api(
            ib=ib,
            recorder=recorder,
            managed_positions=managed_positions,
            runtime_state=runtime_state,
            record_lifecycle_fn=record_lifecycle_with_formal,
            persist_managed_positions_fn=persist_managed_positions,
            host="127.0.0.1",
            port=8767,
        )

        startup_reconciliation = startup_reconcile_runtime_state(
            ib,
            recorder,
            managed_positions,
            contract_by_symbol,
            runtime_state,
        )

        recorder.record_run_metadata({
            "module": "v67_live_top100_expansion_paper_trader",
            "strategy": STRATEGY_NAME,
            "client_id": args.client_id,
            "top_n": args.top_n,
            "seen_fills_loaded": len(seen_fills),
            "restored_positions": len(restored),
            "backfilled_1m_rows": backfilled_rows,
            "state_rebuild_count": state_rebuild_count,
            "current_session_backfilled_1m_rows": current_session_backfilled_rows,
            "startup_reconciliation": startup_reconciliation,
        })
        run_dry_run_reconciliation_report(
            ib,
            recorder,
            managed_positions,
            runtime_state,
            reason="startup_after_restore_and_control_api_start",
        )

        start = time.time()
        while time.time() - start < args.duration_seconds:
            try:
                if not ibkr_connection_alive(ib):
                    recovery = handle_ibkr_disconnect_and_recover(
                        ib,
                        recorder,
                        managed_positions,
                        contract_by_symbol,
                        contracts,
                        runtime_state,
                        args,
                        reason="connection_health_check",
                        seen_fills=seen_fills,
                    )
                    if recovery.get("ok"):
                        tickers = recovery.get("tickers") or tickers
                    else:
                        time.sleep(float(args.reconnect_wait_seconds))
                    continue

                process_overnight_automation(runtime_state, args)

                reload_top100_universe_if_requested(
                    ib,
                    recorder,
                    states,
                    contracts,
                    contract_by_symbol,
                    tickers,
                    latest_snapshots,
                    managed_positions,
                    runtime_state,
                    args,
                )

                process_control_api_commands(
                    ib=ib,
                    recorder=recorder,
                    managed_positions=managed_positions,
                    runtime_state=runtime_state,
                    record_lifecycle_fn=record_lifecycle_with_formal,
                    persist_managed_positions_fn=persist_managed_positions,
                )

                process_history_collector_commands(
                    runtime_state=runtime_state,
                )

                ib.sleep(args.interval_seconds)
            except Exception as exc:
                print(f"{now_utc()} IBKR_DISCONNECTED during=sleep error={exc!r}", flush=True)
                recovery = handle_ibkr_disconnect_and_recover(
                    ib,
                    recorder,
                    managed_positions,
                    contract_by_symbol,
                    contracts,
                    runtime_state,
                    args,
                    reason="main_loop_exception",
                    seen_fills=seen_fills,
                )
                if recovery.get("ok"):
                    tickers = recovery.get("tickers") or tickers
                else:
                    time.sleep(float(args.reconnect_wait_seconds))
                continue
            loop_now = time.time()
            observed_at = datetime.now(timezone.utc)
            market_open = market_open_datetime_utc(args, observed_at)
            session_elapsed = (observed_at - market_open).total_seconds()
            today_session = get_us_equity_session(observed_at.date())
            market_closed_today = not today_session.is_trading_day
            if market_closed_today:
                runtime_state["entries_blocked_reason"] = "market_closed_holiday"
                closed_key = today_session.session_date.isoformat()
                logged_dates = _runtime_set(runtime_state, "market_closed_logged_dates")
                if closed_key not in logged_dates:
                    print(
                        f"{now_utc()} MARKET_CLOSED_HOLIDAY date={closed_key} reason={today_session.reason}",
                        flush=True,
                    )
                    logged_dates.add(closed_key)
            elif runtime_state.get("entries_blocked_reason") == "market_closed_holiday":
                runtime_state["entries_blocked_reason"] = ""
            ready_count = 0
            data_count = 0
            best_symbol = None
            best_score = -999999.0
            rejection_counter = Counter()
            ranked = []
            debug_symbol = contracts[0][0] if contracts else None
            debug_payload: tuple[str, SymbolState, dict[str, Any], dict[str, Any]] | None = None
            zeroish_feature_count = 0
            time_entries_blocked = (
                market_closed_today
                or
                not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            restart_entries_blocked = loop_now < float(runtime_state.get("entries_blocked_until") or 0.0)
            if not restart_entries_blocked and runtime_state.get("entries_blocked_reason") == "restart_cooldown":
                runtime_state["entries_blocked_reason"] = ""
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            reconnect_entries_blocked = bool(runtime_state.get("reconnect_active", False))
            entries_blocked = time_entries_blocked or manual_entries_blocked or restart_entries_blocked or reconnect_entries_blocked
            eod_active = args.enable_eod_flatten and (is_after_utc(args.eod_flatten_utc) or bool(runtime_state.get("manual_eod_flatten_requested", False)))

            for symbol, q in contracts:
                snap = snapshot_from_ticker(symbol, tickers[symbol])
                if snap.get("price") is None:
                    continue
                latest_snapshots[symbol] = snap
                data_count += 1
                state = states[symbol]
                update_state(state, snap, session_elapsed, args.opening_range_seconds, observed_at=observed_at)
                features = compute_live_safe_features(state, snap, args)
                if features_are_all_zeroish(features):
                    zeroish_feature_count += 1
                if symbol == debug_symbol:
                    debug_payload = (symbol, state, snap, features)
                ranked.append((symbol, features["score"], features))
                if features["score"] > best_score:
                    best_score = features["score"]
                    best_symbol = symbol
                if not features["ready"]:
                    for reason in features["reason"].split(";"):
                        rejection_counter[reason] += 1

                has_active_position = symbol in managed_positions and managed_positions[symbol].active and not managed_positions[symbol].exit_sent
                entry_symbol_allowed = symbol in _runtime_set(runtime_state, "entry_symbols")
                if features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed and entries_blocked:
                    record_lifecycle_with_formal(
                        recorder,
                        "BUY_BLOCKED",
                        symbol,
                        action="BUY",
                        price=features.get("entry_price"),
                        reason="entries_blocked_manual_or_time_window",
                        raw_json={
                            **features,
                            "manual_entries_blocked": manual_entries_blocked,
                            "time_entries_blocked": time_entries_blocked,
                            "restart_entries_blocked": restart_entries_blocked,
                            "reconnect_entries_blocked": reconnect_entries_blocked,
                            "entries_blocked_reason": runtime_state.get("entries_blocked_reason"),
                        },
                    )
                if features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed and not entries_blocked:
                    ready_count += 1
                    price = features.get("entry_price")
                    qty = max(1, int(args.position_usd // price)) if price and price > 0 else 0
                    record_lifecycle_with_formal(recorder, "SIGNAL_READY", symbol, action="BUY", quantity=qty, price=price, reason=features["reason"], raw_json=features)
                    candidate_notional = float(qty) * float(price) if qty and price else 0.0
                    risk_status = evaluate_risk_guard(
                        recorder,
                        managed_positions,
                        latest_snapshots,
                        args,
                        symbol=symbol,
                        candidate_notional=candidate_notional,
                    )
                    runtime_state["risk_guard_last_status"] = risk_status
                    if risk_status.get("blocked"):
                        reason = str(risk_status.get("reason") or "risk_guard")
                        print(
                            f"{now_utc()} RISK_GUARD_BLOCK_ENTRY symbol={symbol} reason={reason} "
                            f"daily_pnl={risk_status.get('daily_pnl')} trades_today={risk_status.get('trades_today')} "
                            f"active_positions={risk_status.get('active_positions')} gross_exposure={risk_status.get('gross_exposure')} "
                            f"candidate_notional={risk_status.get('candidate_notional')}",
                            flush=True,
                        )
                        record_lifecycle_with_formal(
                            recorder,
                            "RISK_GUARD_BLOCK_ENTRY",
                            symbol,
                            action="BUY",
                            quantity=qty,
                            price=price,
                            reason=reason,
                            raw_json=risk_status,
                        )
                        continue
                    order = MarketOrder("BUY", qty)
                    order.tif = "DAY"
                    order.outsideRth = False
                    trade = ib.placeOrder(q, order)
                    if qty > 0 and price and price > 0:
                        managed_positions[symbol] = ManagedPosition(symbol=symbol, contract=q, quantity=qty, entry_price=float(price), entry_time=now_utc(), peak_price=float(price))
                        persist_managed_positions(recorder, managed_positions)
                    record_lifecycle_with_formal(
                        recorder,
                        "BUY_ORDER_SENT",
                        symbol,
                        action="BUY",
                        quantity=qty,
                        price=price,
                        order_id=trade.order.orderId,
                        entry_price=price,
                        peak_price=price,
                        decision_bid=snap.get("bid"),
                        decision_ask=snap.get("ask"),
                        decision_mid=snap.get("mid_price"),
                        decision_last=snap.get("last"),
                        spread_pct=((snap.get("spread") / snap.get("mid_price") * 100.0) if snap.get("spread") and snap.get("mid_price") else None),
                    )
                    print(f"PAPER BUY SENT symbol={symbol} qty={qty} price={price:.2f} score={features['score']:.2f} orderId={trade.order.orderId} tif={order.tif} outsideRth={order.outsideRth}", flush=True)
                    state.signal_sent = True

            adopted_count = 0
            if args.adopt_existing_positions and not adopted_once and data_count > 0:
                adopted_count = adopt_existing_long_positions(ib, recorder, contract_by_symbol, latest_snapshots, managed_positions, runtime_state)
                for symbol in managed_positions:
                    if symbol in states:
                        states[symbol].signal_sent = True
                adopted_once = True
                if adopted_count:
                    persist_managed_positions(recorder, managed_positions)

            exit_count = manage_exits(ib, recorder, managed_positions, latest_snapshots, args, runtime_state)
            if exit_count:
                persist_managed_positions(recorder, managed_positions)

            new_fills = None
            if loop_now - last_portfolio_record >= args.portfolio_interval_seconds:
                try:
                    record_account_snapshot(ib, recorder)
                    new_fills = record_recent_fills(ib, recorder, seen_fills)
                    lifecycle_fills_updated = enrich_lifecycle_with_fills(recorder)
                    fill_diagnostics_updated = process_fill_lifecycle_diagnostics(
                        ib,
                        recorder,
                        managed_positions,
                        runtime_state,
                    )
                    if lifecycle_fills_updated:
                        print(f"{now_utc()} lifecycle_fills_updated={lifecycle_fills_updated}", flush=True)
                    if fill_diagnostics_updated:
                        print(f"{now_utc()} fill_lifecycle_diagnostics_updated={fill_diagnostics_updated}", flush=True)
                    verification = verify_managed_positions_against_ibkr(
                        ib,
                        recorder,
                        managed_positions,
                        reason="portfolio_recorder_verification",
                    )
                    runtime_state["portfolio_last_verification"] = verification
                    run_dry_run_reconciliation_report(
                        ib,
                        recorder,
                        managed_positions,
                        runtime_state,
                        reason="periodic_portfolio_record",
                    )
                    record_strategy_equity(recorder, managed_positions, latest_snapshots)
                    persist_managed_positions(recorder, managed_positions)
                    last_portfolio_record = loop_now
                except Exception as exc:
                    print(f"{now_utc()} portfolio_recorder_error={exc!r}", flush=True)
                    if "disconnect" in repr(exc).lower() or "connection" in repr(exc).lower() or "socket" in repr(exc).lower():
                        print(f"{now_utc()} IBKR_DISCONNECTED during=portfolio_recorder error={exc!r}", flush=True)
                        recovery = handle_ibkr_disconnect_and_recover(
                            ib,
                            recorder,
                            managed_positions,
                            contract_by_symbol,
                            contracts,
                            runtime_state,
                            args,
                            reason="portfolio_recorder_exception",
                            seen_fills=seen_fills,
                        )
                        if recovery.get("ok"):
                            tickers = recovery.get("tickers") or tickers
                        else:
                            time.sleep(float(args.reconnect_wait_seconds))

            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            if debug_payload is not None:
                symbol, state, snap, features = debug_payload
                log_live_feature_debug(
                    runtime_state=runtime_state,
                    symbol=symbol,
                    state=state,
                    snap=snap,
                    features=features,
                    market_open=market_open,
                    session_elapsed=session_elapsed,
                    reason="top_ranked_symbol",
                )
            if data_count > 0 and zeroish_feature_count == data_count and not market_closed_today:
                if debug_payload is not None:
                    symbol, state, snap, features = debug_payload
                    log_live_feature_debug(
                        runtime_state=runtime_state,
                        symbol=symbol,
                        state=state,
                        snap=snap,
                        features=features,
                        market_open=market_open,
                        session_elapsed=session_elapsed,
                        reason="all_features_zero",
                    )
                warning_key = "live_features_all_zero_warning_last"
                if loop_now - float(runtime_state.get(warning_key) or 0.0) >= 60.0:
                    runtime_state[warning_key] = loop_now
                    print(
                        f"{now_utc()} LIVE_FEATURES_ALL_ZERO_WARNING with_data={data_count} "
                        f"zeroish_features={zeroish_feature_count} session_start={market_open.isoformat()} "
                        f"session_elapsed_seconds={session_elapsed:.1f} top_symbol={debug_symbol}",
                        flush=True,
                    )
            top5_str = " | ".join([f"{s}:{score:.1f}" for s, score, _ in ranked[:5]])
            rejection_summary = ", ".join([f"{k}={v}" for k, v in rejection_counter.most_common(5)])
            portfolio_part = f" portfolio_recorded=1 new_fills={new_fills}" if new_fills is not None else ""
            active_managed = sum(1 for p in managed_positions.values() if p.active)
            time_entries_blocked = (
                market_closed_today
                or not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            restart_entries_blocked = loop_now < float(runtime_state.get("entries_blocked_until") or 0.0)
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            reconnect_entries_blocked = bool(runtime_state.get("reconnect_active", False))
            entries_blocked = time_entries_blocked or manual_entries_blocked or restart_entries_blocked or reconnect_entries_blocked
            eod_active = args.enable_eod_flatten and (is_after_utc(args.eod_flatten_utc) or bool(runtime_state.get("manual_eod_flatten_requested", False)))
            risk_status = runtime_state.get("risk_guard_last_status") if isinstance(runtime_state.get("risk_guard_last_status"), dict) else {}
            risk_guard_block = bool((risk_status or {}).get("blocked"))
            risk_guard_reason = str((risk_status or {}).get("reason") or "")
            print(
                f"{now_utc()} heartbeat scanned={len(contracts)} with_data={data_count} ready_new={ready_count} "
                f"adopted={adopted_count} exits_sent={exit_count} managed_open={active_managed} entries_blocked={int(entries_blocked)} "
                f"manual_block={int(manual_entries_blocked)} restart_block={int(restart_entries_blocked)} reconnect_block={int(reconnect_entries_blocked)} eod_active={int(eod_active)} "
                f"risk_guard_block={int(risk_guard_block)} risk_guard_reason={risk_guard_reason} "
                f"best={best_symbol}:{best_score:.2f} top5=[{top5_str}] rejects=[{rejection_summary}]"
                f"{portfolio_part}",
                flush=True,
            )

    finally:
        persist_managed_positions(recorder, managed_positions)
        for ticker in tickers.values():
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        ib.disconnect()
        print("Disconnected", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
