from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import (
    LIVE_SIGNAL_MAX_SPREAD_BPS,
    LIVE_SIGNAL_MIN_FIRST_15M_HIGH_PCT,
    LIVE_SIGNAL_MIN_FIRST_5M_HIGH_PCT,
    LIVE_SIGNAL_MIN_OR_RANGE_PCT,
    LIVE_SIGNAL_MIN_PRICE,
    LIVE_SIGNAL_OPENING_RANGE_SECONDS,
    fnum,
    iso_ts,
    load_session_candles,
    load_top100,
    normalize_symbol,
    parse_dt,
    pct,
    read_sql_table,
)
from src.live_trading.analysis.signal_opportunity_forensics import bar_available_at

DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/analysis")

TRADE_COLUMNS = [
    "date",
    "source",
    "symbol",
    "signal_time",
    "candidate_rank",
    "live_entry_score",
    "entry_decision",
    "block_reason",
    "entry_time",
    "entry_price",
    "quantity",
    "exit_time",
    "exit_price",
    "exit_reason",
    "gross_pnl",
    "commissions",
    "net_pnl",
    "mfe_pct",
    "mae_pct",
    "matched_live_trade",
    "divergence_type",
]

EVENT_COLUMNS = [
    "date",
    "timestamp",
    "event_type",
    "symbol",
    "candidate_rank",
    "score",
    "price",
    "quantity",
    "reason",
    "open_positions",
    "details",
]

COMPARISON_COLUMNS = [
    "date",
    "symbol",
    "offline_entry_time",
    "live_entry_time",
    "offline_net_pnl",
    "live_net_pnl",
    "pnl_difference",
    "matched_live_trade",
    "divergence_type",
    "first_divergence",
]


@dataclass
class ReplayPosition:
    symbol: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    candidate_rank: int
    score: float
    peak_price: float
    low_price: float
    gross_pnl: float | None = None
    commissions: float | None = None
    net_pnl: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""


@dataclass
class ReplayConfig:
    position_usd: float = 1000.0
    max_open_positions: int = 0
    max_entries_per_cycle: int = 5
    max_entries_per_minute: int = 5
    entry_delay_after_open_minutes: float = 5.0
    min_live_entry_score: float = 0.0
    max_one_trade_per_symbol_per_day: bool = True
    exit_stop_loss_pct: float = 8.0
    exit_trailing_activation_pct: float = 3.0
    exit_trailing_stop_pct: float = 3.0
    eod_flatten_utc: str = "19:45"
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    bar_timestamp_semantics: str = "bar_start"


@dataclass
class ReplayResult:
    trades: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    max_concurrent_positions: int = 0
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)


def _rows(candles: pd.DataFrame, semantics: str) -> pd.DataFrame:
    if candles.empty or "timestamp" not in candles.columns:
        return pd.DataFrame()
    rows = candles.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "spread_bps"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["available_at"] = bar_available_at(rows, semantics)
    return rows


def _feature_at(rows: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, Any]:
    if rows.empty:
        return {"ready": False, "reason": "missing_candles"}
    start = rows.iloc[0]["timestamp"]
    visible = rows[rows["available_at"] <= timestamp]
    if visible.empty:
        return {"ready": False, "reason": "no_completed_bar"}
    open_price = fnum(rows.iloc[0].get("open"))
    current = visible.iloc[-1]
    price = fnum(current.get("close"), fnum(current.get("open")))
    if open_price is None or open_price <= 0:
        return {"ready": False, "reason": "invalid_open"}
    first5_end = start + pd.Timedelta(minutes=5)
    first15_end = start + pd.Timedelta(minutes=15)
    or_end = start + pd.Timedelta(seconds=LIVE_SIGNAL_OPENING_RANGE_SECONDS)
    first5 = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < first5_end)] if timestamp >= first5_end else pd.DataFrame()
    first15 = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < first15_end)] if timestamp >= first15_end else pd.DataFrame()
    or_rows = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < or_end)] if timestamp >= or_end else pd.DataFrame()
    first5_high = fnum(first5["high"].max()) if not first5.empty else None
    first15_high = fnum(first15["high"].max()) if not first15.empty else None
    or_high = fnum(or_rows["high"].max()) if not or_rows.empty else None
    or_low = fnum(or_rows["low"].min()) if not or_rows.empty else None
    first5_pct = pct(first5_high, open_price)
    first15_pct = pct(first15_high, open_price)
    or_range = (or_high / or_low - 1.0) * 100.0 if or_high is not None and or_low is not None and or_low > 0 else None
    spread = fnum(current.get("spread_bps")) if "spread_bps" in visible.columns else None
    score = 0.0
    for value, weight in [(first5_pct, 2.0), (first15_pct, 2.0), (or_range, 1.0)]:
        if value is not None:
            score += float(value) * weight
    if spread is not None and LIVE_SIGNAL_MAX_SPREAD_BPS > 0:
        score += max(0.0, LIVE_SIGNAL_MAX_SPREAD_BPS - spread) / LIVE_SIGNAL_MAX_SPREAD_BPS * 5.0
    reasons = []
    if first5_pct is None or first5_pct < LIVE_SIGNAL_MIN_FIRST_5M_HIGH_PCT:
        reasons.append("first_5m_high_too_low")
    if first15_pct is None or first15_pct < LIVE_SIGNAL_MIN_FIRST_15M_HIGH_PCT:
        reasons.append("first_15m_high_too_low")
    if or_range is None or or_range < LIVE_SIGNAL_MIN_OR_RANGE_PCT:
        reasons.append("or_range_too_low")
    if price is None or price < LIVE_SIGNAL_MIN_PRICE:
        reasons.append("price_too_low")
    if spread is not None and spread > LIVE_SIGNAL_MAX_SPREAD_BPS:
        reasons.append("spread_too_wide")
    return {
        "ready": not reasons,
        "reason": ";".join(reasons) if reasons else "live_safe_expansion_ready",
        "entry_price": price,
        "score": round(score, 4),
        "spread_bps": spread,
        "first_5m_high_pct": first5_pct,
        "first_15m_high_pct": first15_pct,
        "or_range_pct": or_range,
    }


def _event(result: ReplayResult, session_date: str, timestamp: pd.Timestamp, event_type: str, symbol: str = "", **kwargs: Any) -> None:
    result.events.append({
        "date": session_date,
        "timestamp": iso_ts(timestamp),
        "event_type": event_type,
        "symbol": symbol,
        "candidate_rank": kwargs.get("candidate_rank", ""),
        "score": kwargs.get("score", ""),
        "price": kwargs.get("price", ""),
        "quantity": kwargs.get("quantity", ""),
        "reason": kwargs.get("reason", ""),
        "open_positions": kwargs.get("open_positions", ""),
        "details": json.dumps(kwargs.get("details", {}), sort_keys=True, default=str),
    })


def _close_position(pos: ReplayPosition, *, timestamp: pd.Timestamp, price: float, reason: str, config: ReplayConfig) -> dict[str, Any]:
    exit_price = price * (1.0 - config.slippage_bps / 10000.0)
    gross = (exit_price - pos.entry_price) * pos.quantity
    commission = max(config.min_commission, pos.quantity * config.commission_per_share) * 2.0
    net = gross - commission
    pos.exit_time = timestamp
    pos.exit_price = exit_price
    pos.exit_reason = reason
    pos.gross_pnl = gross
    pos.commissions = commission
    pos.net_pnl = net
    mfe = pct(pos.peak_price, pos.entry_price)
    mae = pct(pos.low_price, pos.entry_price)
    return {
        "date": "",
        "source": "offline_replay",
        "symbol": pos.symbol,
        "signal_time": iso_ts(pos.signal_time),
        "candidate_rank": pos.candidate_rank,
        "live_entry_score": pos.score,
        "entry_decision": "entered",
        "block_reason": "",
        "entry_time": iso_ts(pos.entry_time),
        "entry_price": pos.entry_price,
        "quantity": pos.quantity,
        "exit_time": iso_ts(timestamp),
        "exit_price": exit_price,
        "exit_reason": reason,
        "gross_pnl": gross,
        "commissions": commission,
        "net_pnl": net,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "matched_live_trade": "",
        "divergence_type": "",
    }


def _manage_positions(result: ReplayResult, session_date: str, timestamp: pd.Timestamp, open_positions: dict[str, ReplayPosition], candle_by_symbol: dict[str, pd.Series], config: ReplayConfig) -> None:
    for symbol, pos in list(open_positions.items()):
        row = candle_by_symbol.get(symbol)
        if row is None:
            continue
        high = fnum(row.get("high"), pos.entry_price) or pos.entry_price
        low = fnum(row.get("low"), pos.entry_price) or pos.entry_price
        close = fnum(row.get("close"), pos.entry_price) or pos.entry_price
        pos.peak_price = max(pos.peak_price, high)
        pos.low_price = min(pos.low_price, low)
        stop_price = pos.entry_price * (1.0 - config.exit_stop_loss_pct / 100.0)
        reason = ""
        exit_price = close
        if low <= stop_price:
            reason = "v46_wide_trail_stop_loss"
            exit_price = stop_price
        else:
            peak_pnl = (pos.peak_price / pos.entry_price - 1.0) * 100.0
            if peak_pnl >= config.exit_trailing_activation_pct:
                trail_price = pos.peak_price * (1.0 - config.exit_trailing_stop_pct / 100.0)
                if low <= trail_price:
                    reason = "v46_wide_trail_trailing_stop"
                    exit_price = trail_price
        if reason:
            row_out = _close_position(pos, timestamp=timestamp, price=exit_price, reason=reason, config=config)
            row_out["date"] = session_date
            result.trades.append(row_out)
            _event(result, session_date, timestamp, "EXIT", symbol, price=exit_price, reason=reason, open_positions=len(open_positions) - 1)
            del open_positions[symbol]


def _eod_timestamp(rows_by_symbol: dict[str, pd.DataFrame], config: ReplayConfig) -> pd.Timestamp | None:
    first_rows = next((rows for rows in rows_by_symbol.values() if not rows.empty), pd.DataFrame())
    if first_rows.empty:
        return None
    day = first_rows.iloc[0]["timestamp"].date()
    hh, mm = [int(part) for part in str(config.eod_flatten_utc).split(":", 1)]
    return pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz="UTC")


def replay_session(
    *,
    session_date: str,
    top100_path: Path,
    history_dir: Path,
    config: ReplayConfig,
) -> ReplayResult:
    result = ReplayResult()
    top100 = load_top100(top100_path)
    if top100.empty:
        return result
    symbols = [normalize_symbol(value) for value in top100["symbol"].tolist() if normalize_symbol(value)]
    rows_by_symbol = {symbol: _rows(load_session_candles(history_dir, symbol, session_date, "RTH"), config.bar_timestamp_semantics) for symbol in symbols}
    non_empty = [rows for rows in rows_by_symbol.values() if not rows.empty]
    if not non_empty:
        return result
    start = min(rows["available_at"].min() for rows in non_empty)
    end = max(rows["available_at"].max() for rows in non_empty)
    eod = _eod_timestamp(rows_by_symbol, config)
    if eod is not None:
        end = min(end, eod)
    entry_delay_until = min(rows["timestamp"].min() for rows in non_empty) + pd.Timedelta(minutes=config.entry_delay_after_open_minutes)
    open_positions: dict[str, ReplayPosition] = {}
    traded_symbols: set[str] = set()
    realized = 0.0
    current = start.floor("min")
    while current <= end.ceil("min"):
        candle_by_symbol = {}
        for symbol, rows in rows_by_symbol.items():
            visible = rows[rows["available_at"] <= current]
            if not visible.empty:
                candle_by_symbol[symbol] = visible.iloc[-1]
        _manage_positions(result, session_date, current, open_positions, candle_by_symbol, config)
        if eod is not None and current >= eod:
            for symbol, pos in list(open_positions.items()):
                row = candle_by_symbol.get(symbol)
                close = fnum(row.get("close"), pos.entry_price) if row is not None else pos.entry_price
                row_out = _close_position(pos, timestamp=current, price=close or pos.entry_price, reason="v46_wide_trail_close_exit_eod", config=config)
                row_out["date"] = session_date
                result.trades.append(row_out)
                del open_positions[symbol]
            break
        candidates = []
        for symbol, rows in rows_by_symbol.items():
            if symbol in open_positions:
                continue
            if config.max_one_trade_per_symbol_per_day and symbol in traded_symbols:
                continue
            features = _feature_at(rows, current)
            if not features.get("ready"):
                result.skipped[str(features.get("reason") or "not_ready")] = result.skipped.get(str(features.get("reason") or "not_ready"), 0) + 1
                continue
            if current < entry_delay_until:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, score=features.get("score"), price=features.get("entry_price"), reason="entry_delay_after_open", open_positions=len(open_positions))
                result.skipped["entry_delay_after_open"] = result.skipped.get("entry_delay_after_open", 0) + 1
                continue
            candidates.append((symbol, float(features.get("score") or 0.0), features))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        entries_this_minute = 0
        for rank, (symbol, score, features) in enumerate(candidates, start=1):
            if config.max_open_positions > 0 and len(open_positions) >= config.max_open_positions:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, candidate_rank=rank, score=score, price=features.get("entry_price"), reason="max_positions_full", open_positions=len(open_positions))
                result.skipped["max_positions_full"] = result.skipped.get("max_positions_full", 0) + 1
                continue
            if config.max_entries_per_cycle > 0 and entries_this_minute >= config.max_entries_per_cycle:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, candidate_rank=rank, score=score, price=features.get("entry_price"), reason="max_entries_per_cycle", open_positions=len(open_positions))
                result.skipped["max_entries_per_cycle"] = result.skipped.get("max_entries_per_cycle", 0) + 1
                continue
            if config.max_entries_per_minute > 0 and entries_this_minute >= config.max_entries_per_minute:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, candidate_rank=rank, score=score, price=features.get("entry_price"), reason="max_entries_per_minute", open_positions=len(open_positions))
                result.skipped["max_entries_per_minute"] = result.skipped.get("max_entries_per_minute", 0) + 1
                continue
            if score < config.min_live_entry_score:
                result.skipped["live_entry_score_too_low"] = result.skipped.get("live_entry_score_too_low", 0) + 1
                continue
            price = fnum(features.get("entry_price"))
            if price is None or price <= 0:
                result.skipped["invalid_price"] = result.skipped.get("invalid_price", 0) + 1
                continue
            entry_price = price * (1.0 + config.slippage_bps / 10000.0)
            qty = max(1, int(config.position_usd // entry_price))
            pos = ReplayPosition(symbol=symbol, signal_time=current, entry_time=current, entry_price=entry_price, quantity=qty, candidate_rank=rank, score=score, peak_price=entry_price, low_price=entry_price)
            open_positions[symbol] = pos
            traded_symbols.add(symbol)
            entries_this_minute += 1
            result.max_concurrent_positions = max(result.max_concurrent_positions, len(open_positions))
            _event(result, session_date, current, "ENTRY", symbol, candidate_rank=rank, score=score, price=entry_price, quantity=qty, reason="entered", open_positions=len(open_positions))
        realized = sum(float(row.get("net_pnl") or 0.0) for row in result.trades)
        result.equity_curve.append((current, realized))
        current += pd.Timedelta(minutes=1)
    return result


def load_live_trades(sqlite_path: Path, session_date: str) -> list[dict[str, Any]]:
    trades = read_sql_table(sqlite_path, "trades")
    if trades.empty:
        return []
    if "session_date" in trades.columns:
        trades = trades[trades["session_date"].astype(str).eq(session_date)]
    status = trades.get("status", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    trades = trades[status.isin(["CLOSED", "COMMISSION_PENDING", "PNL_PENDING"])]
    rows = []
    for row in trades.to_dict("records"):
        entry = parse_dt(row.get("entry_fill_time") or row.get("opened_at") or row.get("created_at"))
        exit_time = parse_dt(row.get("exit_fill_time") or row.get("closed_at"))
        rows.append({
            "date": session_date,
            "source": "live_sqlite",
            "symbol": normalize_symbol(row.get("symbol")),
            "signal_time": iso_ts(row.get("signal_time") or row.get("ready_since") or entry),
            "candidate_rank": row.get("live_entry_rank") or row.get("ranking_position") or row.get("top100_rank"),
            "live_entry_score": row.get("live_entry_score") or row.get("score"),
            "entry_decision": "live_entered",
            "block_reason": "",
            "entry_time": iso_ts(entry),
            "entry_price": row.get("entry_price"),
            "quantity": row.get("quantity"),
            "exit_time": iso_ts(exit_time),
            "exit_price": row.get("exit_price"),
            "exit_reason": row.get("exit_reason"),
            "gross_pnl": row.get("gross_pnl"),
            "commissions": row.get("commission") or row.get("commissions"),
            "net_pnl": row.get("net_pnl") or row.get("realized_pnl"),
            "mfe_pct": row.get("mfe_pct") or row.get("peak_pct"),
            "mae_pct": row.get("mae_pct"),
            "matched_live_trade": "",
            "divergence_type": "",
        })
    return rows


def compare_trades(session_date: str, offline: list[dict[str, Any]], live: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in live:
        live_by_symbol.setdefault(str(row.get("symbol") or ""), []).append(row)
    used: set[int] = set()
    out = []
    for off in offline:
        sym = str(off.get("symbol") or "")
        candidates = live_by_symbol.get(sym, [])
        match_idx = None
        for idx, live_row in enumerate(candidates):
            if idx not in used:
                match_idx = idx
                break
        live_row = candidates[match_idx] if match_idx is not None else None
        if match_idx is not None:
            used.add(match_idx)
        off_entry = parse_dt(off.get("entry_time"))
        live_entry = parse_dt(live_row.get("entry_time")) if live_row else None
        off_net = fnum(off.get("net_pnl"), 0.0) or 0.0
        live_net = fnum(live_row.get("net_pnl"), 0.0) if live_row else 0.0
        if live_row is None:
            div = "offline_only_trade"
        elif off_entry and live_entry and abs((off_entry - live_entry).total_seconds()) > 60:
            div = "same_symbol_different_entry_time"
        else:
            div = "matched_same_symbol"
        out.append({
            "date": session_date,
            "symbol": sym,
            "offline_entry_time": off.get("entry_time"),
            "live_entry_time": live_row.get("entry_time") if live_row else "",
            "offline_net_pnl": off_net,
            "live_net_pnl": live_net,
            "pnl_difference": off_net - (live_net or 0.0),
            "matched_live_trade": int(live_row is not None),
            "divergence_type": div,
            "first_divergence": div,
        })
    offline_symbols = {str(row.get("symbol") or "") for row in offline}
    for row in live:
        sym = str(row.get("symbol") or "")
        if sym not in offline_symbols:
            out.append({
                "date": session_date,
                "symbol": sym,
                "offline_entry_time": "",
                "live_entry_time": row.get("entry_time"),
                "offline_net_pnl": 0.0,
                "live_net_pnl": fnum(row.get("net_pnl"), 0.0) or 0.0,
                "pnl_difference": -(fnum(row.get("net_pnl"), 0.0) or 0.0),
                "matched_live_trade": 0,
                "divergence_type": "live_only_trade",
                "first_divergence": "live_only_trade",
            })
    return out


def summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = [fnum(row.get("net_pnl"), 0.0) or 0.0 for row in rows]
    wins = sum(1 for value in pnl if value > 0)
    losses = sum(1 for value in pnl if value <= 0)
    return {
        "entries": len(rows),
        "winners": wins,
        "losers": losses,
        "win_rate": wins / len(rows) * 100.0 if rows else 0.0,
        "net_pnl": sum(pnl),
        "average_pnl": sum(pnl) / len(pnl) if pnl else 0.0,
        "gross_pnl": sum(fnum(row.get("gross_pnl"), 0.0) or 0.0 for row in rows),
    }


def max_drawdown(equity_curve: list[tuple[pd.Timestamp, float]]) -> float:
    peak = 0.0
    dd = 0.0
    for _ts, equity in equity_curve:
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, session_date: str, replay: ReplayResult, live: list[dict[str, Any]], comparison: list[dict[str, Any]], focus_symbols: list[str]) -> None:
    offline_metrics = summary_metrics(replay.trades)
    live_metrics = summary_metrics(live)
    comp_counts: dict[str, int] = {}
    for row in comparison:
        key = str(row.get("divergence_type") or "unknown")
        comp_counts[key] = comp_counts.get(key, 0) + 1
    lines = [
        f"# Full Session v67 Offline Replay {session_date}",
        "",
        "FACT: This is a read-only causal replay over completed 1m bars. It does not alter live trading state.",
        "FACT: Bar-start candles become available one minute after their timestamp.",
        "FACT: v67 thresholds used here: first5=4.0%, first15=6.5%, OR range=5.0%, min_price=5.0, max_spread_bps=50.",
        "HYPOTHESIS: Differences versus live can come from intrabar tick prices, real spreads, IBKR permissions/subscriptions, or production blocks not intentionally simulated.",
        "",
        "## Offline Replay",
        f"- signals/entries={offline_metrics['entries']}",
        f"- winners={offline_metrics['winners']} losers={offline_metrics['losers']} win_rate={offline_metrics['win_rate']:.2f}%",
        f"- gross_pnl={offline_metrics['gross_pnl']:.4f} net_pnl={offline_metrics['net_pnl']:.4f} average_pnl={offline_metrics['average_pnl']:.4f}",
        f"- max_concurrent_positions={replay.max_concurrent_positions}",
        f"- max_drawdown={max_drawdown(replay.equity_curve):.4f}",
        f"- skipped_candidates={json.dumps(replay.skipped, sort_keys=True)}",
        "",
        "## Live SQLite",
        f"- entries={live_metrics['entries']}",
        f"- winners={live_metrics['winners']} losers={live_metrics['losers']} win_rate={live_metrics['win_rate']:.2f}%",
        f"- gross_pnl={live_metrics['gross_pnl']:.4f} net_pnl={live_metrics['net_pnl']:.4f}",
        "",
        "## Comparison",
    ]
    for key, value in sorted(comp_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Focus Symbols")
    by_symbol = {str(row.get("symbol") or ""): row for row in comparison}
    for symbol in focus_symbols:
        row = by_symbol.get(symbol, {})
        lines.append(f"- {symbol}: {row.get('divergence_type', 'not_entered_by_replay')} offline_entry={row.get('offline_entry_time', '')} live_entry={row.get('live_entry_time', '')} pnl_diff={row.get('pnl_difference', '')}")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    top100_path = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    config = ReplayConfig(
        position_usd=args.position_usd,
        max_open_positions=args.max_open_positions,
        max_entries_per_cycle=args.max_entries_per_cycle,
        max_entries_per_minute=args.max_entries_per_minute,
        entry_delay_after_open_minutes=args.entry_delay_after_open_minutes,
        min_live_entry_score=args.min_live_entry_score,
        slippage_bps=args.slippage_bps,
        bar_timestamp_semantics=args.bar_timestamp_semantics,
    )
    replay = replay_session(session_date=args.date, top100_path=top100_path, history_dir=args.history_dir, config=config)
    live = load_live_trades(args.sqlite_path, args.date)
    offline_rows = replay.trades
    for row in offline_rows:
        row["date"] = args.date
    combined = [*offline_rows, *live]
    comparison = compare_trades(args.date, offline_rows, live)
    output_dir = args.output_dir
    trades_path = output_dir / f"full_session_replay_{args.date}.csv"
    events_path = output_dir / f"full_session_replay_events_{args.date}.csv"
    comparison_path = output_dir / f"full_session_replay_comparison_{args.date}.csv"
    summary_path = output_dir / f"full_session_replay_summary_{args.date}.md"
    write_csv(trades_path, combined, TRADE_COLUMNS)
    write_csv(events_path, replay.events, EVENT_COLUMNS)
    write_csv(comparison_path, comparison, COMPARISON_COLUMNS)
    focus = [normalize_symbol(s) for s in str(args.focus_symbols or "NUAI,IREN,FBYD").split(",") if normalize_symbol(s)]
    write_summary(summary_path, args.date, replay, live, comparison, focus)
    print(
        f"FULL_SESSION_REPLAY_DONE date={args.date} offline_entries={len(offline_rows)} live_entries={len(live)} "
        f"output={trades_path} events={events_path} comparison={comparison_path} summary={summary_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only full-session causal v67 offline replay.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--position-usd", type=float, default=1000.0)
    parser.add_argument("--max-open-positions", type=int, default=0)
    parser.add_argument("--max-entries-per-cycle", type=int, default=5)
    parser.add_argument("--max-entries-per-minute", type=int, default=5)
    parser.add_argument("--entry-delay-after-open-minutes", type=float, default=5.0)
    parser.add_argument("--min-live-entry-score", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--bar-timestamp-semantics", choices=["bar_start", "bar_end"], default="bar_start")
    parser.add_argument("--focus-symbols", default="NUAI,IREN,FBYD")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
