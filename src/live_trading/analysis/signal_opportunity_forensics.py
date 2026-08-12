from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    live_signal_replay,
    load_session_candles,
    pct,
    parse_dt,
)
from src.live_trading.ranking.daily_top100_builder import parquet_path
from src.live_trading.analysis.strategy_config_parity import EffectiveSignalThresholds, add_threshold_cli, resolve_threshold_args

DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_ANALYSIS_DIR = Path("data/analysis")
DEFAULT_SYMBOLS_2026_07_20 = [
    "ADVB",
    "NUAI",
    "CCOI",
    "IREN",
    "CLSK",
    "XNDU",
    "FRMM",
    "CIFR",
    "TTAN",
    "MARA",
    "SBET",
    "AARD",
]

CASE_COLUMNS = [
    "date",
    "symbol",
    "source_possible_signal_time",
    "live_equivalent_possible_signal_time",
    "classification",
    "opportunity_classification",
    "open",
    "offline_high_price",
    "open_to_session_high_pct",
    "session_high_time",
    "session_high_after_signal",
    "pre_signal_session_high",
    "price_at_signal",
    "first_tradable_price_after_signal",
    "first_tradable_time_after_signal",
    "post_signal_high",
    "post_signal_high_time",
    "post_signal_low",
    "session_close",
    "signal_entry_to_post_signal_high_mfe_pct",
    "signal_entry_to_post_signal_low_mae_pct",
    "signal_entry_to_close_pct",
    "time_from_signal_to_mfe_seconds",
    "first5_high",
    "first15_high",
    "or_high",
    "or_low",
    "or_range_pct",
    "breakout_gate_used",
    "did_break_or_high",
    "signal_price_source",
    "bar_timestamp_semantics",
    "legacy_breakout_possible_time",
    "first_time_above_5pct",
    "v67_entry_price",
    "v67_exit_time",
    "v67_exit_price",
    "v67_exit_reason",
    "v67_gross_pnl",
    "v67_commission",
    "v67_net_pnl",
    "v67_gross_return_pct",
    "v67_net_return_pct",
    "v67_quantity",
    "parity_first_divergence",
    "effective_min_first5",
    "effective_min_first15",
    "effective_min_or_range",
    "config_source",
    "notes",
]

PARITY_COLUMNS = [
    "date",
    "symbol",
    "timestamp",
    "bar_open",
    "bar_high",
    "bar_low",
    "bar_close",
    "features_available_at_timestamp",
    "first5_high",
    "first15_high",
    "or_high",
    "or_low",
    "or_range_pct",
    "current_live_equivalent_price",
    "spread_bps",
    "score",
    "first5_gate",
    "first15_gate",
    "or_gate",
    "price_gate",
    "spread_gate",
    "legacy_breakout_gate",
    "offline_decision",
    "live_equivalent_decision",
    "first_divergence",
    "effective_min_first5",
    "effective_min_first15",
    "effective_min_or_range",
    "config_source",
]


@dataclass(frozen=True)
class ExitSimulation:
    entry_time: pd.Timestamp | None
    entry_price: float | None
    exit_time: pd.Timestamp | None
    exit_price: float | None
    exit_reason: str
    quantity: int
    gross_pnl: float | None
    commission: float | None
    net_pnl: float | None
    gross_return_pct: float | None
    net_return_pct: float | None


def _rows(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty or "timestamp" not in candles.columns:
        return pd.DataFrame()
    out = candles.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "spread_bps"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def bar_available_at(rows: pd.DataFrame, semantics: str = "bar_start") -> pd.Series:
    if semantics == "bar_end":
        return rows["timestamp"]
    return rows["timestamp"] + pd.Timedelta(minutes=1)


def first_time_above_pct(candles: pd.DataFrame, open_price: float | None, threshold_pct: float) -> pd.Timestamp | None:
    rows = _rows(candles)
    if rows.empty or open_price is None or open_price <= 0:
        return None
    mask = pd.to_numeric(rows.get("high", pd.Series(dtype=float)), errors="coerce") >= open_price * (1.0 + threshold_pct / 100.0)
    if not mask.any():
        return None
    return rows.loc[mask].iloc[0]["timestamp"]


def legacy_breakout_time(candles: pd.DataFrame, opening_range_seconds: int = LIVE_SIGNAL_OPENING_RANGE_SECONDS) -> pd.Timestamp | None:
    rows = _rows(candles)
    if rows.empty:
        return None
    start = rows.iloc[0]["timestamp"]
    or_end = start + pd.Timedelta(seconds=opening_range_seconds)
    opening = rows[(rows["timestamp"] >= start) & (rows["timestamp"] < or_end)]
    if opening.empty:
        return None
    or_high = fnum(opening["high"].max())
    if or_high is None:
        return None
    after = rows[rows["timestamp"] >= or_end]
    broke = after[pd.to_numeric(after["high"], errors="coerce") >= or_high]
    if broke.empty:
        return None
    return broke.iloc[0]["timestamp"]


def _window_features(rows: pd.DataFrame, timestamp: pd.Timestamp, *, opening_range_seconds: int, semantics: str, effective: EffectiveSignalThresholds | None = None) -> dict[str, Any]:
    effective = effective or EffectiveSignalThresholds(4.0, 6.5, 5.0, "programmatic_historical_strict_default")
    if rows.empty:
        return {}
    data = rows.copy()
    data["available_at"] = bar_available_at(data, semantics)
    start = data.iloc[0]["timestamp"]
    available = data[data["available_at"] <= timestamp]
    open_price = fnum(data.iloc[0].get("open"))
    first5_end = start + pd.Timedelta(minutes=5)
    first15_end = start + pd.Timedelta(minutes=15)
    or_end = start + pd.Timedelta(seconds=opening_range_seconds)
    if open_price is None or open_price <= 0 or available.empty:
        return {"open_price": open_price, "current_price": None}
    first5_available = timestamp >= first5_end
    first15_available = timestamp >= first15_end
    or_available = timestamp >= or_end
    first5 = available[(available["timestamp"] >= start) & (available["timestamp"] < first5_end)] if first5_available else pd.DataFrame()
    first15 = available[(available["timestamp"] >= start) & (available["timestamp"] < first15_end)] if first15_available else pd.DataFrame()
    or_rows = available[(available["timestamp"] >= start) & (available["timestamp"] < or_end)] if or_available else pd.DataFrame()
    last = available.iloc[-1]
    first5_high = fnum(first5["high"].max()) if not first5.empty else None
    first15_high = fnum(first15["high"].max()) if not first15.empty else None
    or_high = fnum(or_rows["high"].max()) if not or_rows.empty else None
    or_low = fnum(or_rows["low"].min()) if not or_rows.empty else None
    or_range = (or_high / or_low - 1.0) * 100.0 if or_high is not None and or_low is not None and or_low > 0 else None
    spread = fnum(last.get("spread_bps")) if "spread_bps" in available.columns else None
    current_price = fnum(last.get("close"), fnum(last.get("open")))
    score = 0.0
    for value, weight in [(pct(first5_high, open_price), 2.0), (pct(first15_high, open_price), 2.0), (or_range, 1.0)]:
        if value is not None:
            score += value * weight
    if spread is not None and LIVE_SIGNAL_MAX_SPREAD_BPS > 0:
        score += max(0.0, LIVE_SIGNAL_MAX_SPREAD_BPS - spread) / LIVE_SIGNAL_MAX_SPREAD_BPS * 5.0
    return {
        "open_price": open_price,
        "current_price": current_price,
        "first5_high": first5_high,
        "first15_high": first15_high,
        "or_high": or_high,
        "or_low": or_low,
        "or_range_pct": or_range,
        "spread_bps": spread,
        "score": round(score, 4),
        "first5_gate": int((pct(first5_high, open_price) or -999.0) >= effective.min_first5),
        "first15_gate": int((pct(first15_high, open_price) or -999.0) >= effective.min_first15),
        "or_gate": int((or_range or -999.0) >= effective.min_or_range),
        "price_gate": int(current_price is not None and current_price >= LIVE_SIGNAL_MIN_PRICE),
        "spread_gate": int(spread is None or spread <= LIVE_SIGNAL_MAX_SPREAD_BPS),
    }


def build_parity_rows(
    *,
    session_date: str,
    symbol: str,
    candles: pd.DataFrame,
    center: pd.Timestamp | None,
    semantics: str = "bar_start",
    opening_range_seconds: int = LIVE_SIGNAL_OPENING_RANGE_SECONDS,
    effective: EffectiveSignalThresholds | None = None,
) -> list[dict[str, Any]]:
    rows = _rows(candles)
    if rows.empty:
        return []
    start = rows.iloc[0]["timestamp"]
    if center is None or center <= start + pd.Timedelta(minutes=31):
        begin = start
        end = start + pd.Timedelta(minutes=30)
    else:
        begin = center - pd.Timedelta(minutes=10)
        end = center + pd.Timedelta(minutes=10)
    out = []
    current = begin.floor("min")
    while current <= end.ceil("min"):
        feats = _window_features(rows, current, opening_range_seconds=opening_range_seconds, semantics=semantics, effective=effective)
        available = rows.copy()
        available["available_at"] = bar_available_at(available, semantics)
        visible = available[available["available_at"] <= current]
        bar = visible.iloc[-1] if not visible.empty else {}
        legacy_breakout = int(feats.get("or_high") is not None and not visible.empty and fnum(bar.get("high")) is not None and fnum(bar.get("high")) >= fnum(feats.get("or_high"), 0.0))
        live_ready = all(int(feats.get(key, 0) or 0) == 1 for key in ["first5_gate", "first15_gate", "or_gate", "price_gate", "spread_gate"])
        legacy_ready = live_ready and legacy_breakout
        divergence = "" if legacy_ready == live_ready else "candle_high_breakout_gate_not_in_live"
        out.append({
            "date": session_date,
            "symbol": symbol,
            "timestamp": iso_ts(current),
            "bar_open": fnum(bar.get("open")) if isinstance(bar, pd.Series) else None,
            "bar_high": fnum(bar.get("high")) if isinstance(bar, pd.Series) else None,
            "bar_low": fnum(bar.get("low")) if isinstance(bar, pd.Series) else None,
            "bar_close": fnum(bar.get("close")) if isinstance(bar, pd.Series) else None,
            "features_available_at_timestamp": int(bool(feats and not visible.empty)),
            "first5_high": feats.get("first5_high"),
            "first15_high": feats.get("first15_high"),
            "or_high": feats.get("or_high"),
            "or_low": feats.get("or_low"),
            "or_range_pct": feats.get("or_range_pct"),
            "current_live_equivalent_price": feats.get("current_price"),
            "spread_bps": feats.get("spread_bps"),
            "score": feats.get("score"),
            "first5_gate": feats.get("first5_gate", 0),
            "first15_gate": feats.get("first15_gate", 0),
            "or_gate": feats.get("or_gate", 0),
            "price_gate": feats.get("price_gate", 0),
            "spread_gate": feats.get("spread_gate", 0),
            "legacy_breakout_gate": legacy_breakout,
            "offline_decision": int(legacy_ready),
            "live_equivalent_decision": int(live_ready),
            "first_divergence": divergence,
            **(effective or EffectiveSignalThresholds(4.0, 6.5, 5.0, "programmatic_historical_strict_default")).output_fields(),
        })
        current += pd.Timedelta(minutes=1)
    return out


def simulate_v67_exit(
    candles: pd.DataFrame,
    *,
    entry_time: pd.Timestamp | None,
    entry_price: float | None,
    notional: float = 1000.0,
    slippage_bps: float = 5.0,
    bar_timestamp_semantics: str = "bar_start",
    commission_per_share: float = 0.005,
    min_commission: float = 1.0,
    stop_loss_pct: float = 8.0,
    trailing_activation_pct: float = 3.0,
    trailing_stop_pct: float = 3.0,
) -> ExitSimulation:
    rows = _rows(candles)
    if rows.empty or entry_time is None or entry_price is None or entry_price <= 0:
        return ExitSimulation(entry_time, entry_price, None, None, "missing_entry_or_candles", 0, None, None, None, None, None)
    entry_fill = entry_price * (1.0 + slippage_bps / 10000.0)
    qty = int(notional // entry_fill)
    if qty <= 0:
        return ExitSimulation(entry_time, entry_fill, None, None, "quantity_zero", 0, None, None, None, None, None)
    rows = rows.copy()
    rows["available_at"] = bar_available_at(rows, bar_timestamp_semantics)
    after = rows[rows["available_at"] >= entry_time].copy()
    if after.empty:
        return ExitSimulation(entry_time, entry_fill, None, None, "no_candles_after_entry", qty, None, None, None, None, None)
    peak = entry_fill
    stop_price = entry_fill * (1.0 - stop_loss_pct / 100.0)
    exit_time = None
    exit_price = None
    reason = "v46_wide_trail_close_exit_eod"
    for _, row in after.iterrows():
        high = fnum(row.get("high"), entry_fill) or entry_fill
        low = fnum(row.get("low"), entry_fill) or entry_fill
        close = fnum(row.get("close"), entry_fill) or entry_fill
        peak = max(peak, high)
        if low <= stop_price:
            exit_time = row["available_at"]
            exit_price = stop_price * (1.0 - slippage_bps / 10000.0)
            reason = "v46_wide_trail_stop_loss"
            break
        peak_pnl_pct = (peak / entry_fill - 1.0) * 100.0
        if peak_pnl_pct >= trailing_activation_pct:
            trail_price = peak * (1.0 - trailing_stop_pct / 100.0)
            if low <= trail_price:
                exit_time = row["available_at"]
                exit_price = trail_price * (1.0 - slippage_bps / 10000.0)
                reason = "v46_wide_trail_trailing_stop"
                break
        exit_time = row["available_at"]
        exit_price = close * (1.0 - slippage_bps / 10000.0)
    if exit_price is None:
        return ExitSimulation(entry_time, entry_fill, None, None, "no_exit", qty, None, None, None, None, None)
    gross = (exit_price - entry_fill) * qty
    commission = max(min_commission, qty * commission_per_share) * 2.0
    net = gross - commission
    return ExitSimulation(entry_time, entry_fill, exit_time, exit_price, reason, qty, gross, commission, net, (exit_price / entry_fill - 1.0) * 100.0, (net / (entry_fill * qty)) * 100.0)


def load_case_rows(cases_csv: Path | None, symbols: Iterable[str], session_date: str) -> dict[str, dict[str, Any]]:
    out = {str(symbol).upper(): {"symbol": str(symbol).upper(), "date": session_date} for symbol in symbols}
    if cases_csv and cases_csv.exists() and cases_csv.stat().st_size > 1:
        loaded: dict[str, dict[str, Any]] = {}
        with cases_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                sym = str(row.get("symbol") or "").upper()
                if sym:
                    loaded[sym] = row
        if loaded:
            if out:
                out.update({sym: row for sym, row in loaded.items() if sym in out})
            else:
                out = loaded
    return out


def classify_case(*, replay_time: pd.Timestamp | None, source_time: pd.Timestamp | None, mfe_pct: float | None, net_pnl: float | None, divergence: str) -> tuple[str, str]:
    if replay_time is None and source_time is not None:
        return "OFFLINE_LOOKAHEAD_FALSE_POSITIVE", "NOT_PROFITABLE_AFTER_SIGNAL" if (mfe_pct or 0) <= 0 else ""
    if source_time is not None and replay_time is not None and source_time < replay_time:
        return "BAR_BOUNDARY_MISMATCH", ""
    if divergence:
        return "CANDLE_HIGH_VS_LIVE_PRICE_MISMATCH", ""
    if replay_time is None:
        return "OFFLINE_LOOKAHEAD_FALSE_POSITIVE", ""
    if (mfe_pct or 0.0) <= 0.0:
        return "NOT_PROFITABLE_AFTER_SIGNAL", "NOT_PROFITABLE_AFTER_SIGNAL"
    if net_pnl is not None and net_pnl > 0:
        return "VALID_LIVE_SIGNAL_MISSED", "PROFITABLE_LIVE_OPPORTUNITY"
    return "VALID_SIGNAL_BUT_RUNTIME_STATE_MISSING", "NOT_PROFITABLE_AFTER_SIGNAL"


def analyze_symbol(
    *,
    session_date: str,
    symbol: str,
    case: dict[str, Any],
    history_dir: Path,
    notional: float,
    slippage_bps: float,
    semantics: str,
    effective: EffectiveSignalThresholds | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    effective = effective or EffectiveSignalThresholds(4.0, 6.5, 5.0, "programmatic_historical_strict_default")
    candles = load_session_candles(history_dir, symbol, session_date, "RTH")
    rows = _rows(candles)
    if rows.empty:
        return {
            "date": session_date,
            "symbol": symbol,
            "classification": "MISSING_CANDLES",
            **effective.output_fields(),
            "notes": f"missing {parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), 'RTH')}",
        }, []
    replay = live_signal_replay(
        candles, bar_timestamp_semantics=semantics,
        min_first_5m_high_pct=effective.min_first5,
        min_first_15m_high_pct=effective.min_first15,
        min_or_range_pct=effective.min_or_range,
    )
    source_time = parse_dt(case.get("possible_signal_time")) or replay.possible_signal_time
    signal_time = source_time or replay.possible_signal_time
    open_price = fnum(rows.iloc[0].get("open"))
    high_idx = pd.to_numeric(rows["high"], errors="coerce").idxmax()
    session_high = fnum(rows.loc[high_idx, "high"])
    session_high_time = rows.loc[high_idx, "timestamp"]
    close_price = fnum(rows.iloc[-1].get("close"))
    before = rows[rows["timestamp"] <= signal_time] if signal_time is not None else pd.DataFrame()
    rows = rows.copy()
    rows["available_at"] = bar_available_at(rows, semantics)
    after = rows[rows["available_at"] >= signal_time] if signal_time is not None else pd.DataFrame()
    first_after = after.iloc[0] if not after.empty else None
    entry_price = fnum(first_after.get("close")) if isinstance(first_after, pd.Series) else replay.current_price
    pre_high = fnum(before["high"].max()) if not before.empty else None
    post_high = fnum(after["high"].max()) if not after.empty else None
    post_low = fnum(after["low"].min()) if not after.empty else None
    post_high_time = after.loc[pd.to_numeric(after["high"], errors="coerce").idxmax(), "available_at"] if not after.empty else None
    mfe = pct(post_high, entry_price)
    mae = pct(post_low, entry_price)
    to_close = pct(close_price, entry_price)
    time_to_mfe = (post_high_time - signal_time).total_seconds() if post_high_time is not None and signal_time is not None else None
    sim = simulate_v67_exit(candles, entry_time=signal_time, entry_price=entry_price, notional=notional, slippage_bps=slippage_bps, bar_timestamp_semantics=semantics)
    parity = build_parity_rows(session_date=session_date, symbol=symbol, candles=candles, center=signal_time, semantics=semantics, effective=effective)
    first_divergence = next((row["first_divergence"] for row in parity if row.get("first_divergence")), "")
    classification, opportunity = classify_case(replay_time=replay.possible_signal_time, source_time=source_time, mfe_pct=mfe, net_pnl=sim.net_pnl, divergence=first_divergence)
    return {
        "date": session_date,
        "symbol": symbol,
        "source_possible_signal_time": iso_ts(source_time),
        "live_equivalent_possible_signal_time": iso_ts(replay.possible_signal_time),
        "classification": classification,
        "opportunity_classification": opportunity,
        "open": open_price,
        "offline_high_price": session_high,
        "open_to_session_high_pct": pct(session_high, open_price),
        "session_high_time": iso_ts(session_high_time),
        "session_high_after_signal": int(bool(signal_time is not None and session_high_time >= signal_time)),
        "pre_signal_session_high": pre_high,
        "price_at_signal": replay.current_price if replay.possible_signal_time == signal_time else entry_price,
        "first_tradable_price_after_signal": entry_price,
        "first_tradable_time_after_signal": iso_ts(first_after.get("available_at") if isinstance(first_after, pd.Series) else None),
        "post_signal_high": post_high,
        "post_signal_high_time": iso_ts(post_high_time),
        "post_signal_low": post_low,
        "session_close": close_price,
        "signal_entry_to_post_signal_high_mfe_pct": mfe,
        "signal_entry_to_post_signal_low_mae_pct": mae,
        "signal_entry_to_close_pct": to_close,
        "time_from_signal_to_mfe_seconds": time_to_mfe,
        "first5_high": replay.first_5m_high,
        "first15_high": replay.first_15m_high,
        "or_high": replay.or_high,
        "or_low": replay.or_low,
        "or_range_pct": replay.or_range_pct,
        "breakout_gate_used": replay.breakout_gate_used,
        "did_break_or_high": replay.did_break_or_high,
        "signal_price_source": replay.signal_price_source,
        "bar_timestamp_semantics": replay.bar_timestamp_semantics,
        "legacy_breakout_possible_time": iso_ts(legacy_breakout_time(candles)),
        "first_time_above_5pct": iso_ts(first_time_above_pct(candles, open_price, 5.0)),
        "v67_entry_price": sim.entry_price,
        "v67_exit_time": iso_ts(sim.exit_time),
        "v67_exit_price": sim.exit_price,
        "v67_exit_reason": sim.exit_reason,
        "v67_gross_pnl": sim.gross_pnl,
        "v67_commission": sim.commission,
        "v67_net_pnl": sim.net_pnl,
        "v67_gross_return_pct": sim.gross_return_pct,
        "v67_net_return_pct": sim.net_return_pct,
        "v67_quantity": sim.quantity,
        "parity_first_divergence": first_divergence,
        **effective.output_fields(),
        "notes": "high is max RTH high; current price uses completed candle close as live-equivalent last price",
    }, parity


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, cases: list[dict[str, Any]], parity_path: Path) -> None:
    total = len(cases)
    valid = sum(1 for row in cases if row.get("classification") in {"VALID_LIVE_SIGNAL_MISSED", "VALID_SIGNAL_BUT_RUNTIME_STATE_MISSING"})
    profitable = sum(1 for row in cases if row.get("opportunity_classification") == "PROFITABLE_LIVE_OPPORTUNITY")
    net = sum(float(row.get("v67_net_pnl") or 0.0) for row in cases)
    avg_mfe = sum(float(row.get("signal_entry_to_post_signal_high_mfe_pct") or 0.0) for row in cases) / total if total else 0.0
    avg_mae = sum(float(row.get("signal_entry_to_post_signal_low_mae_pct") or 0.0) for row in cases) / total if total else 0.0
    lines = [
        "# Signal Opportunity Forensic Report",
        "",
        "FACT: `high/offline_high_price` is the maximum RTH 1-minute candle high in the loaded session history.",
        "FACT: `open_to_high_pct` is `(max_rth_high / first_rth_open - 1) * 100`.",
        "FACT: live-equivalent current price is the completed 1-minute candle close, standing in for live last/tick price.",
        "FACT: v67 has no extra breakout-above-OR-high entry gate; breakout is diagnostic only.",
        "FACT: `first_time_above_5pct` is a runner diagnostic, not a required live gate.",
        "FACT: effective thresholds=" + (
            f"{cases[0].get('effective_min_first5')}/{cases[0].get('effective_min_first15')}/{cases[0].get('effective_min_or_range')} "
            f"config_source={cases[0].get('config_source')}" if cases else "not available"
        ),
        "",
        f"cases={total}",
        f"causally_valid_live_signals={valid}",
        f"profitable_under_v67_exit_simulation={profitable}",
        f"total_simulated_net_pnl={net:.4f}",
        f"average_mfe_pct={avg_mfe:.4f}",
        f"average_mae_pct={avg_mae:.4f}",
        f"minute_parity_table={parity_path}",
        "",
        "## Classification Counts",
    ]
    counts: dict[str, int] = {}
    for row in cases:
        counts[str(row.get("classification") or "UNKNOWN")] = counts.get(str(row.get("classification") or "UNKNOWN"), 0) + 1
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Cases", ""])
    for row in cases:
        lines.append(f"- {row.get('symbol')}: {row.get('classification')} / {row.get('opportunity_classification')} net={row.get('v67_net_pnl')} MFE={row.get('signal_entry_to_post_signal_high_mfe_pct')} MAE={row.get('signal_entry_to_post_signal_low_mae_pct')}")
    lines.extend([
        "",
        "## Recommended Next Steps",
        "",
        "- Live code change: none until this report shows a causally valid live signal that runtime failed to process.",
        "- Offline correction: use `live_signal_replay(...)` for SHS/missed-runner labels; do not use candle-high OR breakout as a live gate.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    effective = resolve_threshold_args(args, args.date)
    explicit_symbols = [s.strip().upper() for s in str(args.symbols or "").split(",") if s.strip()]
    cases_csv = args.cases_csv or Path(f"data/analysis/should_have_signaled_cases_{args.date}.csv")
    cases = load_case_rows(cases_csv, explicit_symbols, args.date)
    symbols = explicit_symbols or sorted(cases) or DEFAULT_SYMBOLS_2026_07_20
    case_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        row, parity = analyze_symbol(
            session_date=args.date,
            symbol=symbol,
            case=cases.get(symbol, {"symbol": symbol, "date": args.date}),
            history_dir=args.history_dir,
            notional=args.notional,
            slippage_bps=args.slippage_bps,
            semantics=args.bar_timestamp_semantics,
            effective=effective,
        )
        case_rows.append(row)
        parity_rows.extend(parity)
    output_dir = args.output_dir
    cases_path = output_dir / f"signal_opportunity_cases_{args.date}.csv"
    parity_path = output_dir / f"signal_opportunity_parity_{args.date}.csv"
    summary_path = output_dir / f"signal_opportunity_summary_{args.date}.md"
    write_csv(cases_path, case_rows, CASE_COLUMNS)
    write_csv(parity_path, parity_rows, PARITY_COLUMNS)
    write_summary(summary_path, case_rows, parity_path)
    print(f"SIGNAL_OPPORTUNITY_DONE cases={len(case_rows)} output={cases_path} parity={parity_path} summary={summary_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only offline/live signal opportunity forensic report.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Defaults to the 12 2026-07-20 SHS symbols.")
    parser.add_argument("--cases-csv", type=Path, default=None)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=Path("data/live/recorder"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--notional", type=float, default=1000.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--bar-timestamp-semantics", choices=["bar_start", "bar_end"], default="bar_start")
    add_threshold_cli(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
