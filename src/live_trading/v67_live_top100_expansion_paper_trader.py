from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v66_ibkr_account_recorder import (
    record_account_snapshot,
    record_recent_fills,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4001
DEFAULT_CLIENT_ID = 65
DEFAULT_ALPHA_RANK = "data/universe/v64_universe_alpha_ranked.csv"
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


def record_lifecycle(recorder: LiveDataRecorder, event: str, symbol: str, **kwargs: Any) -> None:
    fields = [
        "recorded_at", "strategy", "event", "symbol", "action", "quantity", "price", "order_id",
        "execution_id", "reason", "entry_price", "peak_price", "pnl_pct", "raw_json",
    ]
    row = {
        "recorded_at": now_utc(),
        "strategy": STRATEGY_NAME,
        "event": event,
        "symbol": symbol,
        **kwargs,
    }
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


def managed_position_payload(pos: ManagedPosition) -> dict[str, Any]:
    out = asdict(pos)
    out.pop("contract", None)
    return out


def persist_managed_positions(recorder: LiveDataRecorder, positions: dict[str, ManagedPosition]) -> None:
    payload = {
        "recorded_at": now_utc(),
        "strategy": STRATEGY_NAME,
        "positions": {
            symbol: managed_position_payload(pos)
            for symbol, pos in positions.items()
            if pos.active and not pos.exit_sent
        },
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


def is_eod_flatten_time(flatten_utc: str) -> bool:
    try:
        hh, mm = [int(x) for x in flatten_utc.split(":", 1)]
        return datetime.now(timezone.utc).time() >= dtime(hour=hh, minute=mm, tzinfo=timezone.utc)
    except Exception:
        return False


def send_exit_order(ib: IB, recorder: LiveDataRecorder, pos: ManagedPosition, reason: str, price: float | None) -> None:
    if not pos.active or pos.exit_sent or pos.quantity <= 0:
        return
    order = MarketOrder("SELL", pos.quantity)
    order.tif = "DAY"
    order.outsideRth = False
    trade = ib.placeOrder(pos.contract, order)
    pnl_pct = ((price / pos.entry_price - 1.0) * 100.0) if price and pos.entry_price > 0 else None
    record_lifecycle(
        recorder,
        "SELL_ORDER_SENT",
        pos.symbol,
        action="SELL",
        quantity=pos.quantity,
        price=price,
        order_id=trade.order.orderId,
        reason=reason,
        entry_price=pos.entry_price,
        peak_price=pos.peak_price,
        pnl_pct=pnl_pct,
    )
    pos.exit_sent = True
    pos.active = False
    pnl_txt = f" pnl_pct={pnl_pct:.2f}" if pnl_pct is not None else ""
    print(
        f"PAPER SELL SENT symbol={pos.symbol} qty={pos.quantity} "
        f"reason={reason} entry={pos.entry_price:.2f} price={price if price else 0:.2f}"
        f"{pnl_txt} orderId={trade.order.orderId} tif={order.tif} outsideRth={order.outsideRth}",
        flush=True,
    )


def adopt_existing_long_positions(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    latest_snapshots: dict[str, dict[str, Any]],
    managed_positions: dict[str, ManagedPosition],
) -> int:
    adopted = 0
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper()
        if symbol not in contract_by_symbol or symbol in managed_positions:
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
        print(
            f"ADOPTED EXISTING POSITION symbol={symbol} qty={int(quantity)} "
            f"entry={entry_price:.2f} peak={peak_price:.2f}",
            flush=True,
        )
        adopted += 1
    return adopted


def manage_exits(ib: IB, recorder: LiveDataRecorder, managed_positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:
    exits = 0
    eod = args.enable_eod_flatten and is_eod_flatten_time(args.eod_flatten_utc)

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
            send_exit_order(ib, recorder, pos, reason, price)
            exits += 1

    return exits


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
    parser.add_argument("--position-usd", type=float, default=1000.0)
    parser.add_argument("--exit-stop-loss-pct", type=float, default=8.0)
    parser.add_argument("--exit-trailing-activation-pct", type=float, default=3.0)
    parser.add_argument("--exit-trailing-stop-pct", type=float, default=3.0)
    parser.add_argument("--adopt-existing-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-eod-flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eod-flatten-utc", default="19:55")
    parser.add_argument("--backfill-1m-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backfill-duration", default="1 D")
    parser.add_argument("--backfill-top-n", type=int, default=100)
    parser.add_argument("--backfill-pause-seconds", type=float, default=0.15)
    args = parser.parse_args()

    symbols = load_top_symbols(args.alpha_rank_csv, args.top_n, min_price=args.min_price)
    recorder = LiveDataRecorder(args.recorder_dir)

    print("=== v67 live top100 expansion paper trader ===")
    print(f"Symbols loaded: {len(symbols)}")
    print(f"Recorder dir: {recorder.session_dir}")
    print(f"Portfolio/fills recorder: integrated every {args.portfolio_interval_seconds}s")
    print(
        "Exit: v46 wide_trail "
        f"stop_loss={args.exit_stop_loss_pct}% "
        f"trail_activation={args.exit_trailing_activation_pct}% "
        f"trail_stop={args.exit_trailing_stop_pct}% "
        f"adopt_existing={args.adopt_existing_positions} "
        f"eod_flatten={args.enable_eod_flatten} at {args.eod_flatten_utc} UTC"
    )
    print(f"Backfill 1m: {args.backfill_1m_on_start} duration={args.backfill_duration} top_n={args.backfill_top_n}")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    ib.reqMarketDataType(args.market_data_type)

    tickers = {}
    states = {symbol: SymbolState(symbol=symbol) for symbol in symbols}
    contracts = []
    contract_by_symbol: dict[str, Any] = {}
    seen_fills: set[str] = load_existing_fill_keys(recorder)
    managed_positions: dict[str, ManagedPosition] = {}
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
            print(f"Subscribed {symbol} conId={q.conId}")

        restored = restore_managed_positions(recorder, contract_by_symbol)
        if restored:
            managed_positions.update(restored)
            for symbol, pos in restored.items():
                record_lifecycle(recorder, "RESTORED_MANAGED_POSITION", symbol, quantity=pos.quantity, entry_price=pos.entry_price, peak_price=pos.peak_price, reason="managed_positions_json")
            print(f"{now_utc()} restored_managed_positions={len(restored)}", flush=True)

        backfilled_rows = backfill_recent_1m(ib, recorder, contracts, args)
        recorder.record_run_metadata({
            "module": "v67_live_top100_expansion_paper_trader",
            "strategy": STRATEGY_NAME,
            "client_id": args.client_id,
            "top_n": args.top_n,
            "seen_fills_loaded": len(seen_fills),
            "restored_positions": len(restored),
            "backfilled_1m_rows": backfilled_rows,
        })

        start = time.time()
        while time.time() - start < args.duration_seconds:
            ib.sleep(args.interval_seconds)
            loop_now = time.time()
            elapsed = loop_now - start

            ready_count = 0
            exit_count = 0
            data_count = 0
            best_symbol = None
            best_score = -999999.0
            rejection_counter = Counter()
            ranked = []

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

                if features["ready"] and not state.signal_sent:
                    ready_count += 1
                    price = features.get("entry_price")
                    qty = max(1, int(args.position_usd // price)) if price and price > 0 else 0
                    record_lifecycle(recorder, "SIGNAL_READY", symbol, action="BUY", quantity=qty, price=price, reason=features["reason"], raw_json=features)
                    order = MarketOrder("BUY", qty)
                    order.tif = "DAY"
                    order.outsideRth = False
                    trade = ib.placeOrder(q, order)

                    if qty > 0 and price and price > 0:
                        managed_positions[symbol] = ManagedPosition(
                            symbol=symbol,
                            contract=q,
                            quantity=qty,
                            entry_price=float(price),
                            entry_time=now_utc(),
                            peak_price=float(price),
                        )
                        persist_managed_positions(recorder, managed_positions)
                    record_lifecycle(recorder, "BUY_ORDER_SENT", symbol, action="BUY", quantity=qty, price=price, order_id=trade.order.orderId, entry_price=price, peak_price=price)
                    print(
                        f"PAPER BUY SENT symbol={symbol} qty={qty} price={price:.2f} score={features['score']:.2f} "
                        f"orderId={trade.order.orderId} tif={order.tif} outsideRth={order.outsideRth}"
                    )
                    state.signal_sent = True

            adopted_count = 0
            if args.adopt_existing_positions and not adopted_once and data_count > 0:
                adopted_count = adopt_existing_long_positions(ib, recorder, contract_by_symbol, latest_snapshots, managed_positions)
                adopted_once = True
                if adopted_count:
                    persist_managed_positions(recorder, managed_positions)

            exit_count = manage_exits(ib, recorder, managed_positions, latest_snapshots, args)
            if exit_count:
                persist_managed_positions(recorder, managed_positions)

            new_fills = None
            if loop_now - last_portfolio_record >= args.portfolio_interval_seconds:
                try:
                    record_account_snapshot(ib, recorder)
                    new_fills = record_recent_fills(ib, recorder, seen_fills)
                    record_strategy_equity(recorder, managed_positions, latest_snapshots)
                    persist_managed_positions(recorder, managed_positions)
                    last_portfolio_record = loop_now
                except Exception as exc:
                    print(f"{now_utc()} portfolio_recorder_error={exc!r}", flush=True)

            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            top5_str = " | ".join([f"{s}:{score:.1f}" for s, score, _ in ranked[:5]])
            rejection_summary = ", ".join([f"{k}={v}" for k, v in rejection_counter.most_common(5)])
            portfolio_part = f" portfolio_recorded=1 new_fills={new_fills}" if new_fills is not None else ""
            active_managed = sum(1 for p in managed_positions.values() if p.active)
            print(
                f"{now_utc()} heartbeat scanned={len(contracts)} with_data={data_count} ready_new={ready_count} "
                f"adopted={adopted_count} exits_sent={exit_count} managed_open={active_managed} "
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
        print("Disconnected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
