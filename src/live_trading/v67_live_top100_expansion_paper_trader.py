from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.control.control_api import process_control_api_commands, process_history_collector_commands, start_control_api
from src.live_trading.v66_ibkr_account_recorder import (
    record_account_snapshot,
    record_recent_fills,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 65
DEFAULT_ALPHA_RANK = "data/universe/v68_final_daytrading_universe.csv"
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
        "positions": {symbol: managed_position_payload(pos) for symbol, pos in positions.items() if pos.active and not pos.exit_sent},
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
    record_lifecycle(
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
                record_lifecycle(
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
            record_lifecycle(
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
            record_lifecycle(
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


def adopt_existing_long_positions(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    latest_snapshots: dict[str, dict[str, Any]],
    managed_positions: dict[str, ManagedPosition],
) -> int:
    adopted = 0
    exit_sent_symbols = load_exit_sent_symbols(recorder)
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper()
        if symbol not in contract_by_symbol or symbol in managed_positions:
            continue
        if symbol in exit_sent_symbols:
            record_lifecycle(
                recorder,
                "SKIP_ADOPT_EXIT_SENT",
                symbol,
                action="SKIP_ADOPT",
                reason="sell_order_already_sent_in_lifecycle",
            )
            print(f"SKIP ADOPT symbol={symbol} reason=sell_order_already_sent_in_lifecycle", flush=True)
            continue
        quantity = safe_float(getattr(item, "position", None))
        avg_cost = safe_float(getattr(item, "averageCost", None))
        market_price = safe_float(getattr(item, "marketPrice", None))
        if quantity is None or quantity <= 0:
            continue
        entry_price = avg_cost or market_price
        if entry_price is None or entry_price <= 0:
            continue
        snap_price = safe_float((latest_snapshots.get(symbol) or {}).get("price"))
        peak_price = max(entry_price, snap_price or market_price or entry_price)
        managed_positions[symbol] = ManagedPosition(
            symbol=symbol,
            contract=contract_by_symbol[symbol],
            quantity=int(quantity),
            entry_price=float(entry_price),
            entry_time=f"adopted_on_restart:{now_utc()}",
            peak_price=float(peak_price),
            source="adopted_from_ibkr_portfolio",
        )
        record_lifecycle(
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

    ibkr_quantities: dict[str, float] = {}
    if eod:
        runtime_state["entries_blocked"] = True
        try:
            ibkr_quantities = ibkr_position_quantities(ib)
        except Exception as exc:
            print(f"{now_utc()} eod_portfolio_verify_error={exc!r}", flush=True)

    for symbol, pos in list(managed_positions.items()):
        if not pos.active or pos.exit_sent:
            if not eod:
                continue
        snap = latest_snapshots.get(symbol) or {}
        price = safe_float(snap.get("price"))

        if eod:
            ibkr_qty = ibkr_quantities.get(symbol)
            if ibkr_qty is not None and abs(ibkr_qty) <= 0:
                pos.active = False
                continue
            retry_due = (
                pos.exit_sent
                and pos.last_exit_order_ts is not None
                and (time.time() - pos.last_exit_order_ts) >= args.eod_retry_seconds
                and pos.eod_retry_count < args.eod_max_retries
            )
            should_send = not pos.exit_sent or retry_due or manual_eod_force
            if should_send:
                if retry_due or pos.exit_sent:
                    pos.eod_retry_count += 1
                if send_exit_order(ib, recorder, pos, "v46_wide_trail_close_exit_eod", price):
                    exits += 1
            continue

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

    if eod:
        runtime_state["manual_eod_flatten_force"] = False
        try:
            verification = verify_managed_positions_against_ibkr(
                ib,
                recorder,
                managed_positions,
                reason="eod_flatten_verification_pass",
            )
            runtime_state["eod_last_verification"] = verification
            if not verification["managed_open_symbols"]:
                runtime_state["manual_eod_flatten_requested"] = False
                runtime_state["manual_eod_flatten_force"] = False
        except Exception as exc:
            print(f"{now_utc()} eod_verification_error={exc!r}", flush=True)
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
        st.first_price = first
        st.open_price = first
        st.high = first
        st.low = first
        st.first_5m_high = None
        st.first_15m_high = None
        st.or_high = None
        st.or_low = None

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

    ib = IB()
    connect_ibkr_with_retry(ib, args)

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
        "history_collector_end_utc": "15:00",
        "market_open_utc": "15:00",
        "market_close_utc": "20:00",
        "manual_eod_flatten_requested": False,
        "manual_eod_flatten_force": False,
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
                record_lifecycle(recorder, "RESTORED_MANAGED_POSITION", symbol, quantity=pos.quantity, entry_price=pos.entry_price, peak_price=pos.peak_price, reason="managed_positions_json")
            print(f"{now_utc()} restored_managed_positions={len(restored)}", flush=True)

        backfilled_rows = backfill_recent_1m(ib, recorder, contracts, args)
        state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        current_session_backfilled_rows = 0
        if state_rebuild_count == 0 and current_session_candle_count(recorder, args) == 0:
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
            record_lifecycle_fn=record_lifecycle,
            persist_managed_positions_fn=persist_managed_positions,
            host="127.0.0.1",
            port=8767,
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
        })

        start = time.time()
        while time.time() - start < args.duration_seconds:
            try:
                process_control_api_commands(
                    ib=ib,
                    recorder=recorder,
                    managed_positions=managed_positions,
                    runtime_state=runtime_state,
                    record_lifecycle_fn=record_lifecycle,
                    persist_managed_positions_fn=persist_managed_positions,
                )

                process_history_collector_commands(
                    runtime_state=runtime_state,
                )

                ib.sleep(args.interval_seconds)
            except Exception as exc:
                print(f"{now_utc()} IBKR_DISCONNECTED during=sleep error={exc!r}", flush=True)
                try:
                    ib.disconnect()
                except Exception:
                    pass
                connect_ibkr_with_retry(ib, args)
                tickers = resubscribe_market_data(ib, contracts)
                continue
            loop_now = time.time()
            elapsed = loop_now - start
            ready_count = 0
            data_count = 0
            best_symbol = None
            best_score = -999999.0
            rejection_counter = Counter()
            ranked = []
            time_entries_blocked = (
                not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            restart_entries_blocked = loop_now < float(runtime_state.get("entries_blocked_until") or 0.0)
            if not restart_entries_blocked and runtime_state.get("entries_blocked_reason") == "restart_cooldown":
                runtime_state["entries_blocked_reason"] = ""
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            entries_blocked = time_entries_blocked or manual_entries_blocked or restart_entries_blocked
            eod_active = args.enable_eod_flatten and (is_after_utc(args.eod_flatten_utc) or bool(runtime_state.get("manual_eod_flatten_requested", False)))

            for symbol, q in contracts:
                snap = snapshot_from_ticker(symbol, tickers[symbol])
                if snap.get("price") is None:
                    continue
                latest_snapshots[symbol] = snap
                data_count += 1
                state = states[symbol]
                update_state(state, snap, elapsed, args.opening_range_seconds)
                features = compute_live_safe_features(state, snap, args)
                ranked.append((symbol, features["score"], features))
                if features["score"] > best_score:
                    best_score = features["score"]
                    best_symbol = symbol
                if not features["ready"]:
                    for reason in features["reason"].split(";"):
                        rejection_counter[reason] += 1

                has_active_position = symbol in managed_positions and managed_positions[symbol].active and not managed_positions[symbol].exit_sent
                if features["ready"] and not state.signal_sent and not has_active_position and entries_blocked:
                    record_lifecycle(
                        recorder,
                        "BUY_BLOCKED",
                        symbol,
                        action="BUY",
                        price=features.get("entry_price"),
                        reason="entries_blocked_manual_or_time_window",
                        raw_json={**features, "manual_entries_blocked": manual_entries_blocked, "time_entries_blocked": time_entries_blocked, "restart_entries_blocked": restart_entries_blocked},
                    )
                if features["ready"] and not state.signal_sent and not has_active_position and not entries_blocked:
                    ready_count += 1
                    price = features.get("entry_price")
                    qty = max(1, int(args.position_usd // price)) if price and price > 0 else 0
                    record_lifecycle(recorder, "SIGNAL_READY", symbol, action="BUY", quantity=qty, price=price, reason=features["reason"], raw_json=features)
                    order = MarketOrder("BUY", qty)
                    order.tif = "DAY"
                    order.outsideRth = False
                    trade = ib.placeOrder(q, order)
                    if qty > 0 and price and price > 0:
                        managed_positions[symbol] = ManagedPosition(symbol=symbol, contract=q, quantity=qty, entry_price=float(price), entry_time=now_utc(), peak_price=float(price))
                        persist_managed_positions(recorder, managed_positions)
                    record_lifecycle(
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
                adopted_count = adopt_existing_long_positions(ib, recorder, contract_by_symbol, latest_snapshots, managed_positions)
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
                    if lifecycle_fills_updated:
                        print(f"{now_utc()} lifecycle_fills_updated={lifecycle_fills_updated}", flush=True)
                    verification = verify_managed_positions_against_ibkr(
                        ib,
                        recorder,
                        managed_positions,
                        reason="portfolio_recorder_verification",
                    )
                    runtime_state["portfolio_last_verification"] = verification
                    record_strategy_equity(recorder, managed_positions, latest_snapshots)
                    persist_managed_positions(recorder, managed_positions)
                    last_portfolio_record = loop_now
                except Exception as exc:
                    print(f"{now_utc()} portfolio_recorder_error={exc!r}", flush=True)
                    if "disconnect" in repr(exc).lower() or "connection" in repr(exc).lower() or "socket" in repr(exc).lower():
                        print(f"{now_utc()} IBKR_DISCONNECTED during=portfolio_recorder error={exc!r}", flush=True)
                        try:
                            ib.disconnect()
                        except Exception:
                            pass
                        connect_ibkr_with_retry(ib, args)
                        tickers = resubscribe_market_data(ib, contracts)

            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            top5_str = " | ".join([f"{s}:{score:.1f}" for s, score, _ in ranked[:5]])
            rejection_summary = ", ".join([f"{k}={v}" for k, v in rejection_counter.most_common(5)])
            portfolio_part = f" portfolio_recorded=1 new_fills={new_fills}" if new_fills is not None else ""
            active_managed = sum(1 for p in managed_positions.values() if p.active)
            time_entries_blocked = (
                not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            restart_entries_blocked = loop_now < float(runtime_state.get("entries_blocked_until") or 0.0)
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            entries_blocked = time_entries_blocked or manual_entries_blocked or restart_entries_blocked
            eod_active = args.enable_eod_flatten and (is_after_utc(args.eod_flatten_utc) or bool(runtime_state.get("manual_eod_flatten_requested", False)))
            print(
                f"{now_utc()} heartbeat scanned={len(contracts)} with_data={data_count} ready_new={ready_count} "
                f"adopted={adopted_count} exits_sent={exit_count} managed_open={active_managed} entries_blocked={int(entries_blocked)} "
                f"manual_block={int(manual_entries_blocked)} restart_block={int(restart_entries_blocked)} eod_active={int(eod_active)} "
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
