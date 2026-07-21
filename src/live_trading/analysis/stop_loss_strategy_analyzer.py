from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.bad_entries_analyzer import (
    DYNAMIC_FEATURES,
    PREMARKET_FEATURES,
    bucket_for_feature,
    classify_bad_entry,
    opening_range_features,
    row_value,
)
from src.live_trading.analysis.common import fnum, load_session_candles, parse_dt, parse_raw_json, pct
from src.live_trading.analysis.early_loser_exit_analyzer import build_rules as build_early_loser_rules
from src.live_trading.analysis.trade_loader import load_finalized_canonical_trades

DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
STOP_LOSS_PCTS = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00]
SLIPPAGE_BPS = [0, 10, 25, 50]
ACTIVATION_DELAYS_MIN = [0, 1, 3, 5]
BASELINE_STOP_PCT = 8.00
CONSERVATIVE_SLIPPAGE_BPS = 25
SEGMENT_FEATURES = [
    "entry_time_bucket",
    "spread_bps_at_entry",
    "top100_rank",
    "live_entry_score",
    "first_5m_high_pct",
    "first_15m_high_pct",
    "or_range_pct",
    "repeated_symbol_entry",
    "bad_entry_label",
    "peak_data_quality",
    "premarket_range_pct",
    "premarket_change_pct",
    "distance_from_premarket_high_pct",
    "distance_from_premarket_vwap_pct",
    "gap_from_previous_close_pct",
]


def iso(value: Any) -> str:
    ts = parse_dt(value)
    return ts.isoformat() if ts is not None else ""


def slippage_multiplier(bps: float) -> float:
    return 1.0 - (float(bps) / 10000.0)


def trade_commission(row: dict[str, Any]) -> float:
    for name in ("commission", "persisted_commission", "ibkr_commission"):
        value = fnum(row.get(name))
        if value is not None:
            return abs(value)
    return 0.0


def candle_window(history_dir: Path, symbol: str, session_date: str, entry_time: pd.Timestamp | None, exit_time: pd.Timestamp | None, session_type: str) -> pd.DataFrame:
    candles = load_session_candles(history_dir, symbol, session_date, session_type)
    if candles.empty or entry_time is None:
        return pd.DataFrame()
    out = candles[candles["timestamp"] >= entry_time]
    if exit_time is not None:
        out = out[out["timestamp"] <= exit_time]
    return out.reset_index(drop=True)


def full_session_candles(history_dir: Path, symbol: str, session_date: str, session_type: str) -> pd.DataFrame:
    return load_session_candles(history_dir, symbol, session_date, session_type)


def entry_time_bucket_et(entry_time: pd.Timestamp | None, session_date: str) -> str:
    if entry_time is None:
        return "missing"
    try:
        et = entry_time.tz_convert("America/New_York") if entry_time.tzinfo else entry_time.tz_localize("UTC").tz_convert("America/New_York")
    except Exception:
        return "missing"
    minutes = (et.hour * 60 + et.minute + et.second / 60.0) - (9 * 60 + 30)
    if minutes < 5:
        return "before_09:35"
    if minutes < 10:
        return "09:35-09:40"
    if minutes < 15:
        return "09:40-09:45"
    if minutes < 30:
        return "09:45-10:00"
    return "10:00+"


def normalized_candle_times(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty or "timestamp" not in candles.columns:
        return candles
    out = candles.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp")


def first_hit(candles: pd.DataFrame, stop_price: float, activation_time: pd.Timestamp | None) -> tuple[pd.Timestamp | None, dict[str, Any] | None, str]:
    candles = normalized_candle_times(candles)
    if candles.empty or activation_time is None:
        return None, None, "missing_candles"
    active = candles[candles["timestamp"] >= activation_time]
    for row in active.to_dict("records"):
        low = fnum(row.get("low"))
        high = fnum(row.get("high"))
        if low is None:
            continue
        if low <= stop_price:
            ambiguity = "conservative"
            if high is not None and high >= stop_price:
                ambiguity = "ambiguous"
            return parse_dt(row.get("timestamp")), row, ambiguity
    return None, None, "none"


def post_stop_outcome(candles: pd.DataFrame, hit_time: pd.Timestamp | None, entry_price: float) -> dict[str, Any]:
    candles = normalized_candle_times(candles)
    if candles.empty or hit_time is None or entry_price <= 0:
        return {
            "recovered_to_entry_after_stop": None,
            "positive_after_stop": None,
            "later_mfe_pct": None,
            "time_to_recovery_seconds": None,
        }
    later = candles[candles["timestamp"] >= hit_time]
    if later.empty:
        return {"recovered_to_entry_after_stop": 0, "positive_after_stop": 0, "later_mfe_pct": None, "time_to_recovery_seconds": None}
    high = pd.to_numeric(later.get("high"), errors="coerce")
    recovered = high >= entry_price
    recovery_time = None
    if recovered.any():
        recovery_time = parse_dt(later.loc[recovered, "timestamp"].iloc[0])
    max_high = fnum(high.max())
    return {
        "recovered_to_entry_after_stop": int(bool(recovered.any())),
        "positive_after_stop": int(bool((high > entry_price).any())),
        "later_mfe_pct": pct(max_high, entry_price) if max_high is not None else None,
        "time_to_recovery_seconds": (recovery_time - hit_time).total_seconds() if recovery_time is not None and hit_time is not None else None,
    }


def simulate_stop(row: dict[str, Any], candles: pd.DataFrame, *, stop_pct: float, slippage_bps: float, activation_delay_min: int) -> dict[str, Any]:
    entry_time = parse_dt(row.get("entry_time"))
    exit_time = parse_dt(row.get("exit_time"))
    entry_price = fnum(row.get("entry_price")) or 0.0
    exit_price = fnum(row.get("exit_price"))
    qty = abs(fnum(row.get("quantity"), 0.0) or 0.0)
    actual_net = fnum(row.get("actual_net_pnl"), fnum(row.get("net_pnl"), 0.0)) or 0.0
    actual_gross = fnum(row.get("actual_gross_pnl"), fnum(row.get("gross_pnl"), actual_net)) or 0.0
    commission = fnum(row.get("commission"), 0.0) or 0.0
    stop_price = entry_price * (1.0 - stop_pct / 100.0) if entry_price > 0 else None
    activation_time = entry_time + pd.Timedelta(minutes=activation_delay_min) if entry_time is not None else None
    hit_time, hit_row, ambiguity = first_hit(candles, stop_price or 0.0, activation_time)
    stop_hit = hit_time is not None
    fill_price = (stop_price or 0.0) * slippage_multiplier(slippage_bps) if stop_hit and stop_price else exit_price
    simulated_gross = ((fill_price or 0.0) - entry_price) * qty if entry_price and fill_price is not None else actual_gross
    simulated_net = simulated_gross - commission if commission else simulated_gross
    final_winner = actual_net > 0
    simulated_better = simulated_net > actual_net
    post = post_stop_outcome(candles, hit_time, entry_price)
    label = "no_stop"
    false_stop = 0
    saved_loser = 0
    if stop_hit:
        if final_winner:
            label = "false_stop"
            false_stop = 1
        elif simulated_better:
            label = "saved_loser"
            saved_loser = 1
        else:
            label = "neutral_stop"
    holding = None
    if entry_time is not None:
        end_time = hit_time if stop_hit else exit_time
        holding = (end_time - entry_time).total_seconds() / 60.0 if end_time is not None else None
    return {
        "stop_pct": stop_pct,
        "slippage_bps": slippage_bps,
        "activation_delay_min": activation_delay_min,
        "baseline_stop": int(abs(stop_pct - BASELINE_STOP_PCT) < 1e-9),
        "stop_price": stop_price,
        "stop_hit": int(stop_hit),
        "stop_outcome": label,
        "stop_hit_time": iso(hit_time),
        "stop_fill_price": fill_price,
        "ambiguity": ambiguity,
        "simulated_gross_pnl": simulated_gross,
        "simulated_net_pnl": simulated_net,
        "actual_gross_pnl": actual_gross,
        "actual_net_pnl": actual_net,
        "pnl_delta_vs_actual": simulated_net - actual_net,
        "final_winner": int(final_winner),
        "false_stop": false_stop,
        "saved_loser": saved_loser,
        "neutral_stop": int(stop_hit and not false_stop and not saved_loser),
        "holding_minutes_simulated": holding,
        **post,
    }


def enrich_trade_features(trades: pd.DataFrame, history_dir: Path, session_date: str, session_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for trade in trades.to_dict("records"):
        raw = parse_raw_json(trade.get("raw_json"))
        symbol = str(trade.get("symbol") or "").upper()
        counts[symbol] = counts.get(symbol, 0) + 1
        entry_time = parse_dt(trade.get("entry_fill_time"))
        exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
        entry_price = fnum(trade.get("entry_price"))
        exit_price = fnum(trade.get("exit_price"))
        qty = abs(fnum(trade.get("quantity"), 0.0) or 0.0)
        full = full_session_candles(history_dir, symbol, session_date, session_type)
        features = opening_range_features(full, entry_price)
        row = {
            "date": session_date,
            "trade_id": trade.get("trade_id"),
            "symbol": symbol,
            "entry_time": iso(entry_time),
            "exit_time": iso(exit_time),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": qty,
            "gross_pnl": fnum(trade.get("gross_pnl")),
            "net_pnl": fnum(trade.get("net_pnl")),
            "commission": trade_commission(trade),
            "actual_gross_pnl": fnum(trade.get("gross_pnl")),
            "actual_net_pnl": fnum(trade.get("net_pnl")),
            "actual_return_pct": pct(exit_price, entry_price) if exit_price is not None and entry_price else None,
            "entry_time_bucket": entry_time_bucket_et(entry_time, session_date),
            "top100_rank": row_value(trade, raw, ["top100_rank"]),
            "top100_score": row_value(trade, raw, ["top100_score"]),
            "live_entry_score": row_value(trade, raw, ["live_entry_score", "entry_score", "score"]),
            "live_entry_rank": row_value(trade, raw, ["live_entry_rank", "ranking_position"]),
            "spread_bps_at_entry": row_value(trade, raw, ["spread_bps_at_entry", "spread_bps", "bid_ask_spread_bps"]),
            "peak_pct": fnum(trade.get("mfe_pct") or raw.get("peak_pct")),
            "mae_pct": fnum(trade.get("mae_pct") or raw.get("mae_pct")),
            "peak_data_quality": raw.get("peak_data_quality"),
            "repeated_symbol_entry": int(counts[symbol] > 1),
            **features,
        }
        for name in PREMARKET_FEATURES:
            row[name] = row_value(trade, raw, [name])
        row["premarket_feature_coverage"] = "available" if any(row.get(name) not in (None, "") and not pd.isna(row.get(name)) for name in PREMARKET_FEATURES) else "unavailable_for_session"
        row["bad_entry_label"], row["bad_entry_reason"] = classify_bad_entry({**row, "net_pnl_pct": row.get("actual_return_pct")})
        rows.append(row)
    return pd.DataFrame(rows)


def build_trade_paths(*, date: str, sqlite_path: Path, history_dir: Path, session_type: str) -> pd.DataFrame:
    trades = load_finalized_canonical_trades(sqlite_path, date, date)
    enriched = enrich_trade_features(trades, history_dir, date, session_type) if not trades.empty else pd.DataFrame()
    if enriched.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    candle_cache: dict[str, pd.DataFrame] = {}
    for row in enriched.to_dict("records"):
        symbol = row["symbol"]
        if symbol not in candle_cache:
            entry = parse_dt(row.get("entry_time"))
            exit_time = parse_dt(row.get("exit_time"))
            candles = candle_window(history_dir, symbol, date, entry, exit_time, session_type)
            candle_cache[symbol] = candles
        candles = candle_cache[symbol]
        base = dict(row)
        base["candle_count"] = len(candles)
        base["missing_candles"] = int(candles.empty)
        for stop_pct in STOP_LOSS_PCTS:
            sim = simulate_stop(base, candles, stop_pct=stop_pct, slippage_bps=CONSERVATIVE_SLIPPAGE_BPS, activation_delay_min=0)
            rows.append({**base, **sim, "scenario_type": "fixed_conservative"})
    return pd.DataFrame(rows)


def numeric_column(group: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in group.columns:
        return pd.Series([default] * len(group), index=group.index, dtype="float64")
    return pd.to_numeric(group[column], errors="coerce").fillna(default)


def summarize_pnl(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        sim = numeric_column(group, "simulated_net_pnl")
        actual = numeric_column(group, "actual_net_pnl")
        stop_hit = numeric_column(group, "stop_hit")
        saved_loser = numeric_column(group, "saved_loser")
        false_stop = numeric_column(group, "false_stop")
        missing_candles = numeric_column(group, "missing_candles")
        wins = sim[sim > 0]
        losses = sim[sim < 0]
        row = {col: value for col, value in zip(group_cols, key)}
        row.update({
            "trades_analyzed": int(len(group)),
            "stop_hits": int(stop_hit.sum()),
            "stop_hit_rate": float(stop_hit.mean() * 100.0) if len(group) else 0.0,
            "simulated_gross_pnl": float(numeric_column(group, "simulated_gross_pnl").sum()),
            "simulated_net_pnl": float(sim.sum()),
            "actual_net_pnl": float(actual.sum()),
            "pnl_delta_vs_actual": float((sim - actual).sum()),
            "win_rate": float((sim > 0).mean() * 100.0) if len(sim) else 0.0,
            "expectancy": float(sim.mean()) if len(sim) else 0.0,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-9 else None,
            "average_winner": float(wins.mean()) if len(wins) else 0.0,
            "average_loser": float(losses.mean()) if len(losses) else 0.0,
            "max_loss": float(sim.min()) if len(sim) else 0.0,
            "drawdown": float((sim.cumsum() - sim.cumsum().cummax()).min()) if len(sim) else 0.0,
            "saved_losers": int(saved_loser.sum()),
            "false_stops": int(false_stop.sum()),
            "false_stop_rate": float(false_stop.mean() * 100.0) if len(group) else 0.0,
            "winners_sacrificed": int(((false_stop > 0) & (actual > 0)).sum()),
            "losers_improved": int(((sim - actual) > 0).sum()),
            "average_holding_time": float(numeric_column(group, "holding_minutes_simulated", default=float("nan")).mean()) if len(group) else None,
            "median_holding_time": float(numeric_column(group, "holding_minutes_simulated", default=float("nan")).median()) if len(group) else None,
            "ambiguity_count": int((group.get("ambiguity", pd.Series(dtype=str)).astype(str) == "ambiguous").sum()),
            "missing_candle_count": int(missing_candles.sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_slippage_sensitivity(base: pd.DataFrame, *, history_dir: Path, date: str, session_type: str) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    rows = []
    for row in base.drop_duplicates("trade_id").to_dict("records"):
        candles = candle_window(history_dir, row["symbol"], date, parse_dt(row.get("entry_time")), parse_dt(row.get("exit_time")), session_type)
        for slip in SLIPPAGE_BPS:
            for stop_pct in STOP_LOSS_PCTS:
                rows.append({**row, **simulate_stop(row, candles, stop_pct=stop_pct, slippage_bps=slip, activation_delay_min=0)})
    return summarize_pnl(pd.DataFrame(rows), ["stop_pct", "slippage_bps"])


def build_activation_delay(base: pd.DataFrame, *, history_dir: Path, date: str, session_type: str) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    rows = []
    for row in base.drop_duplicates("trade_id").to_dict("records"):
        candles = candle_window(history_dir, row["symbol"], date, parse_dt(row.get("entry_time")), parse_dt(row.get("exit_time")), session_type)
        for delay in ACTIVATION_DELAYS_MIN:
            for stop_pct in STOP_LOSS_PCTS:
                rows.append({**row, **simulate_stop(row, candles, stop_pct=stop_pct, slippage_bps=CONSERVATIVE_SLIPPAGE_BPS, activation_delay_min=delay)})
    return summarize_pnl(pd.DataFrame(rows), ["stop_pct", "activation_delay_min"])


def add_segment_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for feature in SEGMENT_FEATURES:
        if feature not in out.columns:
            out[f"{feature}_bucket"] = "unavailable_for_session" if feature in PREMARKET_FEATURES else "missing"
        elif feature == "entry_time_bucket" or feature == "bad_entry_label" or feature == "peak_data_quality" or feature == "repeated_symbol_entry":
            out[f"{feature}_bucket"] = out[feature].fillna("missing").astype(str)
        else:
            out[f"{feature}_bucket"] = out[feature].map(lambda value, feature=feature: bucket_for_feature(feature, value))
    return out


def build_segment_analysis(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    df = add_segment_buckets(paths)
    rows = []
    for feature in SEGMENT_FEATURES:
        if feature in PREMARKET_FEATURES and (feature not in paths.columns or paths[feature].isna().all()):
            continue
        bucket_col = f"{feature}_bucket"
        if bucket_col not in df.columns:
            continue
        summary = summarize_pnl(df, ["stop_pct", bucket_col])
        if not summary.empty:
            summary = summary.rename(columns={bucket_col: "segment_bucket"})
            summary.insert(1, "segment_feature", feature)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def dynamic_stop_price(row: dict[str, Any], candles: pd.DataFrame, rule: str) -> tuple[float | None, str]:
    entry = fnum(row.get("entry_price")) or 0.0
    if entry <= 0:
        return None, "missing_entry"
    entry_time = parse_dt(row.get("entry_time"))
    if candles.empty or entry_time is None:
        return None, "missing_candles"
    before = candles[candles["timestamp"] <= entry_time]
    first = candles.head(5)
    if rule == "entry_candle_low":
        current = candles[candles["timestamp"] >= entry_time].head(1)
        return (fnum(current["low"].min()), "ok") if not current.empty else (None, "missing_entry_candle")
    if rule == "last_swing_low":
        window = before.tail(5) if not before.empty else candles.head(1)
        return (fnum(window["low"].min()), "ok") if not window.empty else (None, "missing_swing")
    if rule == "opening_range_low":
        return (fnum(first["low"].min()), "ok") if not first.empty else (None, "missing_or")
    if rule == "or_range_pct":
        or_range = fnum(row.get("or_range_pct"))
        return (entry * (1.0 - or_range / 100.0), "ok") if or_range is not None else (None, "missing_or_range")
    if rule == "recent_volatility":
        window = before.tail(10) if not before.empty else candles.head(10)
        if window.empty:
            return None, "missing_volatility"
        rng = (fnum(window["high"].max()) or entry) - (fnum(window["low"].min()) or entry)
        return entry - rng, "ok"
    if rule == "spread_dependent":
        spread = fnum(row.get("spread_bps_at_entry"))
        return (entry * (1.0 - max(1.0, spread / 100.0) / 100.0), "ok") if spread is not None else (None, "missing_spread")
    return None, "unknown_rule"


def build_dynamic_rules(base: pd.DataFrame, *, history_dir: Path, date: str, session_type: str) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    rows = []
    rules = ["entry_candle_low", "last_swing_low", "opening_range_low", "or_range_pct", "recent_volatility", "spread_dependent"]
    for rule in rules:
        sim_rows = []
        coverage = 0
        missing = 0
        for row in base.drop_duplicates("trade_id").to_dict("records"):
            candles = candle_window(history_dir, row["symbol"], date, parse_dt(row.get("entry_time")), parse_dt(row.get("exit_time")), session_type)
            price, reason = dynamic_stop_price(row, candles, rule)
            if price is None or price <= 0 or not (fnum(row.get("entry_price")) or 0):
                missing += 1
                continue
            coverage += 1
            stop_pct = max(0.0, (1.0 - price / (fnum(row.get("entry_price")) or price)) * 100.0)
            sim_rows.append({**row, **simulate_stop(row, candles, stop_pct=stop_pct, slippage_bps=CONSERVATIVE_SLIPPAGE_BPS, activation_delay_min=0), "dynamic_rule": rule, "dynamic_stop_formula": rule, "dynamic_missing_reason": reason})
        summary = summarize_pnl(pd.DataFrame(sim_rows), ["dynamic_rule"]) if sim_rows else pd.DataFrame([{"dynamic_rule": rule}])
        summary["coverage"] = coverage
        summary["missing"] = missing
        summary["dynamic_stop_formula"] = rule
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def require_unique_columns(df: pd.DataFrame, *, context: str) -> None:
    duplicates = sorted({str(col) for col in df.columns if list(df.columns).count(col) > 1})
    if duplicates:
        raise ValueError(f"{context} duplicate columns: {duplicates}")


def require_series_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")
    for column in columns:
        value = df[column]
        if not isinstance(value, pd.Series):
            raise TypeError(f"{context} column {column!r} is not a Series")


def stop_loss_to_early_loser_adapter(base: pd.DataFrame) -> pd.DataFrame:
    """Build the exact early-loser input schema without duplicate renamed columns."""
    require_unique_columns(base, context="stop_loss_hybrid_base")
    out = pd.DataFrame(index=base.index)
    out["net_pnl"] = base["actual_net_pnl"] if "actual_net_pnl" in base.columns else base.get("net_pnl", pd.Series(pd.NA, index=base.index))
    out["final_pnl_pct"] = base["actual_return_pct"] if "actual_return_pct" in base.columns else base.get("final_pnl_pct", pd.Series(pd.NA, index=base.index))
    required_path_columns = ["quantity", "entry_price"]
    for column in required_path_columns:
        out[column] = base[column] if column in base.columns else pd.NA
    for minutes in [5, 10, 15, 20, 30, 45, 60]:
        for prefix in ("pnl_pct_at", "positive_seen_to"):
            column = f"{prefix}_{minutes}m"
            if column in base.columns:
                out[column] = base[column]
    for column in ("trade_id", "symbol", "entry_time", "exit_time"):
        if column in base.columns:
            out[column] = base[column]
    require_unique_columns(out, context="stop_loss_early_loser_adapter")
    require_series_columns(out, ["net_pnl", "final_pnl_pct", *required_path_columns], context="stop_loss_early_loser_adapter")
    return out


def build_hybrid_rules(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    base = paths[paths["slippage_bps"].eq(CONSERVATIVE_SLIPPAGE_BPS) & paths["activation_delay_min"].eq(0)].copy()
    print(
        "STOP_LOSS_HYBRID_INPUT_COLUMNS "
        f"columns={list(base.columns)} "
        f"duplicate_columns={sorted({col for col in base.columns if list(base.columns).count(col) > 1})}",
        flush=True,
    )
    # Add early-loser style rules over actual trade paths, then compare side by side. Hybrid execution is approximated conservatively.
    early = build_early_loser_rules(stop_loss_to_early_loser_adapter(base))
    if early.empty:
        return pd.DataFrame()
    rows = []
    for stop_pct in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        stop_summary = summarize_pnl(base[base["stop_pct"].eq(stop_pct)], ["stop_pct"])
        for early_row in early.head(20).to_dict("records"):
            row = {"hybrid_rule": f"hard_stop_{stop_pct:g}_plus_{early_row.get('rule')}", "stop_pct": stop_pct}
            if not stop_summary.empty:
                row.update({f"hard_stop_{k}": v for k, v in stop_summary.iloc[0].to_dict().items() if k != "stop_pct"})
            row.update({f"early_{k}": v for k, v in early_row.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def data_quality(paths: pd.DataFrame) -> dict[str, Any]:
    total = int(paths["trade_id"].nunique()) if not paths.empty and "trade_id" in paths.columns else 0
    feature_rows = []
    one_per_trade = paths.drop_duplicates("trade_id") if not paths.empty and "trade_id" in paths.columns else pd.DataFrame()
    for feature in DYNAMIC_FEATURES:
        non_null = int(pd.to_numeric(one_per_trade.get(feature, pd.Series(dtype=float)), errors="coerce").notna().sum()) if not one_per_trade.empty else 0
        feature_rows.append({"feature": feature, "non_null": non_null, "total": total, "coverage": "available" if non_null else "unavailable_for_session"})
    return {
        "trades": total,
        "missing_candle_count": int(pd.to_numeric(paths.get("missing_candles", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not paths.empty else 0,
        "premarket_feature_coverage": "available" if any(row["feature"] in PREMARKET_FEATURES and row["non_null"] for row in feature_rows) else "unavailable_for_session",
        "features": feature_rows,
        "conservative_slippage_bps": CONSERVATIVE_SLIPPAGE_BPS,
        "baseline_stop_pct": BASELINE_STOP_PCT,
    }


def write_recommendations(date: str, fixed: pd.DataFrame, dynamic: pd.DataFrame, quality: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Stop Loss Strategy Recommendations {date}", "",
        "FACT: This is a read-only candle-based stop-loss analysis over finalized canonical trades.",
        "BASELINE ONLY: The session is a baseline sample and is not enough for live strategy changes.",
        "REQUIRES MULTI-DAY VALIDATION: Validate any hard stop or dynamic stop across multiple sessions.",
        "POSSIBLE OVERFITTING: Best one-day stop settings can overfit noise.", "",
        f"conservative_slippage_bps={CONSERVATIVE_SLIPPAGE_BPS}",
        f"baseline_stop_pct={BASELINE_STOP_PCT}",
        f"premarket_feature_coverage={quality.get('premarket_feature_coverage')}", "",
        "## Conservative Fixed Stops", "",
    ]
    if not fixed.empty:
        for row in fixed.sort_values("simulated_net_pnl", ascending=False).head(10).to_dict("records"):
            marker = " baseline" if int(row.get("baseline_stop", 0) or 0) else ""
            lines.append(f"- stop={row.get('stop_pct')}%{marker}: net={row.get('simulated_net_pnl'):.4f}, delta={row.get('pnl_delta_vs_actual'):.4f}, false_stops={row.get('false_stops')}, saved_losers={row.get('saved_losers')}")
    else:
        lines.append("- no fixed stop results available")
    lines.extend(["", "## Dynamic Stops", ""])
    if not dynamic.empty:
        for row in dynamic.head(10).to_dict("records"):
            lines.append(f"- {row.get('dynamic_rule')}: coverage={row.get('coverage')}, missing={row.get('missing')}, net={row.get('simulated_net_pnl', '')}")
    else:
        lines.append("- no dynamic stop results available")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, date: str, sqlite_path: Path, history_dir: Path, output_dir: Path, session_type: str = "RTH") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_trade_paths(date=date, sqlite_path=sqlite_path, history_dir=history_dir, session_type=session_type)
    fixed = summarize_pnl(paths[(paths.get("slippage_bps") == CONSERVATIVE_SLIPPAGE_BPS) & (paths.get("activation_delay_min") == 0)] if not paths.empty else paths, ["stop_pct", "baseline_stop"])
    delay = build_activation_delay(paths, history_dir=history_dir, date=date, session_type=session_type)
    slippage = build_slippage_sensitivity(paths, history_dir=history_dir, date=date, session_type=session_type)
    segments = build_segment_analysis(paths[(paths.get("slippage_bps") == CONSERVATIVE_SLIPPAGE_BPS) & (paths.get("activation_delay_min") == 0)] if not paths.empty else paths)
    dynamic = build_dynamic_rules(paths, history_dir=history_dir, date=date, session_type=session_type)
    hybrid = build_hybrid_rules(paths)
    quality = data_quality(paths)
    outputs = {
        "trade_paths": output_dir / f"stop_loss_trade_paths_{date}.csv",
        "fixed_grid": output_dir / f"stop_loss_fixed_grid_{date}.csv",
        "activation_delay": output_dir / f"stop_loss_activation_delay_{date}.csv",
        "slippage": output_dir / f"stop_loss_slippage_sensitivity_{date}.csv",
        "segments": output_dir / f"stop_loss_segment_analysis_{date}.csv",
        "dynamic": output_dir / f"stop_loss_dynamic_rules_{date}.csv",
        "hybrid": output_dir / f"stop_loss_hybrid_rules_{date}.csv",
        "quality": output_dir / f"stop_loss_data_quality_{date}.json",
        "recommendations": output_dir / f"stop_loss_recommendations_{date}.md",
    }
    paths.to_csv(outputs["trade_paths"], index=False)
    fixed.to_csv(outputs["fixed_grid"], index=False)
    delay.to_csv(outputs["activation_delay"], index=False)
    slippage.to_csv(outputs["slippage"], index=False)
    segments.to_csv(outputs["segments"], index=False)
    dynamic.to_csv(outputs["dynamic"], index=False)
    hybrid.to_csv(outputs["hybrid"], index=False)
    outputs["quality"].write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_recommendations(date, fixed, dynamic, quality, outputs["recommendations"])
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only stop-loss strategy analyzer for finalized canonical trades.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-type", default="RTH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    outputs = run(date=args.date, sqlite_path=args.sqlite_path, history_dir=args.history_dir, output_dir=args.output_dir, session_type=args.session_type)
    print(f"STOP_LOSS_DONE date={args.date} elapsed_seconds={time.monotonic() - started:.1f} output={outputs['fixed_grid']} recommendations={outputs['recommendations']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
