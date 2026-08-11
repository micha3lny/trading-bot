from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.live_trading.analysis.common import fnum, load_session_candles, normalize_symbol, parse_dt, pct
from src.live_trading.analysis.full_session_replay_v67 import (
    PreparedCausalSession,
    PreparedCompletedBarFeatures,
    PreparedSessionCache,
    ReplayConfig,
    _feature_at,
    _rows,
    profile_config,
    record_completed_bar_full_frame_call,
    record_replay_snapshots_call,
    replay_session,
)
from src.live_trading.analysis.top100_analysis_common import (
    load_top100_source,
    read_snapshot_chunks,
    read_snapshot_manifest,
    safe_json,
    session_dates,
    write_dataframe,
)
from src.live_trading.analysis.trade_loader import load_finalized_canonical_trades


FEATURES = [
    "top100_rank", "top100_score", "live_rank", "live_entry_score", "spread_bps",
    "first_5m_high_pct", "first_15m_high_pct", "or_range_pct",
    "distance_from_or_high_pct", "gap_from_previous_close_pct", "candidate_age_seconds",
    "consecutive_scans_top10", "consecutive_scans_top20", "consecutive_scans_ready",
    "return_1m", "return_3m", "return_5m", "pullback_from_recent_high_pct",
]


def _bool(value: Any) -> int:
    if value in (None, "") or pd.isna(value):
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))


def _column(frame: pd.DataFrame, name: str, default: Any = None) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (np.ndarray, pd.Series)) else False
    except Exception:
        return False


def _consecutive(values: pd.Series) -> pd.Series:
    count = 0
    out: list[int] = []
    for value in values.fillna(False).astype(bool):
        count = count + 1 if value else 0
        out.append(count)
    return pd.Series(out, index=values.index)


def enrich_light_snapshots(light: pd.DataFrame) -> pd.DataFrame:
    if light.empty:
        return light
    out = light.copy()
    out["live_rank"] = pd.to_numeric(out.get("live_rank", out.get("ranking_position")), errors="coerce")
    out["live_entry_score"] = pd.to_numeric(out.get("live_entry_score"), errors="coerce")
    out = out.sort_values([column for column in ["symbol", "timestamp", "process_start_id", "scan_id"] if column in out.columns])
    grouped = out.groupby("symbol", sort=False, dropna=False)
    for lag in (1, 3, 5):
        out[f"live_rank_delta_{lag}_scan"] = out["live_rank"] - grouped["live_rank"].shift(lag)
        out[f"live_score_delta_{lag}_scan"] = out["live_entry_score"] - grouped["live_entry_score"].shift(lag)
    out["consecutive_scans_top5"] = grouped["live_rank"].transform(lambda s: _consecutive(s <= 5))
    out["consecutive_scans_top10"] = grouped["live_rank"].transform(lambda s: _consecutive(s <= 10))
    out["consecutive_scans_top20"] = grouped["live_rank"].transform(lambda s: _consecutive(s <= 20))
    ready = pd.to_numeric(_column(out, "ready", 0), errors="coerce").fillna(0).gt(0)
    out["consecutive_scans_ready"] = ready.groupby(out["symbol"], sort=False).transform(_consecutive)
    out["snapshot_source"] = "p1_light"
    out["runtime_observed"] = 1
    out["causal_valid"] = 1
    return out.reset_index(drop=True)


def completed_bar_features(candles: pd.DataFrame, timestamp: Any) -> dict[str, Any]:
    record_completed_bar_full_frame_call()
    when = parse_dt(timestamp)
    if candles.empty or when is None:
        return {}
    rows = candles.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    # Stored timestamps are bar starts. At T only bars with start < T are complete.
    visible = rows[rows["timestamp"] + pd.Timedelta(minutes=1) <= when].sort_values("timestamp")
    if visible.empty:
        return {}
    close = pd.to_numeric(visible["close"], errors="coerce")
    high = pd.to_numeric(visible["high"], errors="coerce")
    low = pd.to_numeric(visible["low"], errors="coerce")
    volume = pd.to_numeric(visible.get("volume", 0), errors="coerce")
    current = float(close.iloc[-1])
    def ret(periods: int) -> float | None:
        return pct(current, float(close.iloc[-periods - 1])) if len(close) > periods else None
    recent5 = visible.tail(5)
    recent3 = visible.tail(3)
    recent_high = fnum(high.tail(5).max())
    recent_low = fnum(low.tail(5).min())
    bar_range = fnum(visible.iloc[-1]["high"]) - fnum(visible.iloc[-1]["low"]) if fnum(visible.iloc[-1]["high"]) is not None and fnum(visible.iloc[-1]["low"]) is not None else None
    clv = None if not bar_range or bar_range <= 0 else (current - float(visible.iloc[-1]["low"])) / bar_range
    prior_volume = volume.iloc[:-1].tail(5).mean() if len(volume) > 1 else np.nan
    return {
        "return_1m": ret(1), "return_3m": ret(3), "return_5m": ret(5),
        "green_bars_last_3": int((pd.to_numeric(recent3["close"]) > pd.to_numeric(recent3["open"])).sum()),
        "green_bars_last_5": int((pd.to_numeric(recent5["close"]) > pd.to_numeric(recent5["open"])).sum()),
        "close_location_value": clv,
        "pullback_from_recent_high_pct": pct(current, recent_high),
        "recent_low_distance_pct": pct(current, recent_low),
        "volume_acceleration": None if pd.isna(prior_volume) or prior_volume <= 0 else float(volume.iloc[-1] / prior_volume),
    }


def replay_snapshots(
    session_date: str,
    top100: pd.DataFrame,
    history_dir: Path,
    config: ReplayConfig,
    *,
    prepared_rows_by_symbol: dict[str, pd.DataFrame] | None = None,
    prepared_sessions_by_symbol: dict[str, PreparedCausalSession] | None = None,
) -> pd.DataFrame:
    record_replay_snapshots_call()
    records: list[dict[str, Any]] = []
    for row in top100.to_dict("records"):
        symbol = normalize_symbol(row.get("symbol"))
        prepared_session = prepared_sessions_by_symbol.get(symbol) if prepared_sessions_by_symbol is not None else None
        if prepared_session is not None:
            bars = prepared_session.rows
        elif prepared_rows_by_symbol is None:
            candles = load_session_candles(history_dir, symbol, session_date)
            bars = _rows(candles, config.bar_timestamp_semantics)
        else:
            bars = prepared_rows_by_symbol.get(symbol, pd.DataFrame())
        if bars.empty:
            records.append({"session_date": session_date, "symbol": symbol, "snapshot_source": "replay", "runtime_observed": 0, "causal_valid": 0, "rejection_reason": "missing_history"})
            continue
        prepared_session = prepared_session or PreparedCausalSession(symbol, session_date, bars, config)
        for when, feature, completed in prepared_session.iter_features():
            records.append({
                "session_date": session_date, "symbol": symbol, "timestamp": when,
                "snapshot_source": "replay", "runtime_observed": 0,
                "causal_valid": int(config.bar_timestamp_semantics == "bar_start"),
                "schema_version": "top100_buy_replay_v1", "top100_rank": row.get("top100_rank") or row.get("rank"),
                "top100_score": row.get("top100_score") or row.get("score"), "live_rank": row.get("top100_rank") or row.get("rank"),
                "live_entry_score": feature.get("score"), "current_price": feature.get("entry_price"),
                "spread_bps": feature.get("spread_bps"), "ready": int(bool(feature.get("ready"))),
                "would_emit_signal_ready": int(bool(feature.get("ready"))), "signal_ready_reason": feature.get("reason"),
                "rejection_reason": "" if feature.get("ready") else feature.get("reason"),
                "first_5m_high_pct": feature.get("first_5m_high_pct"), "first_15m_high_pct": feature.get("first_15m_high_pct"),
                "first_5m_complete": feature.get("first_5m_complete"), "first_15m_complete": feature.get("first_15m_complete"),
                "or_range_pct": feature.get("or_range_pct"), "or_high": feature.get("or_high"), "or_low": feature.get("or_low"),
                "data_quality_flags": "replay_only;spread_and_broker_state_unavailable",
            } | completed)
    return enrich_light_snapshots(pd.DataFrame(records)) if records else pd.DataFrame()


def _representative_snapshots(snapshots: pd.DataFrame, top100: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for top in top100.to_dict("records"):
        symbol = normalize_symbol(top.get("symbol"))
        rows = snapshots[snapshots["symbol"].eq(symbol)] if not snapshots.empty else pd.DataFrame()
        if rows.empty:
            selected: dict[str, Any] = {"symbol": symbol, "snapshot_source": "missing", "runtime_observed": 0, "causal_valid": 0, "data_quality_flags": "missing_snapshot"}
        else:
            preferred = rows[pd.to_numeric(_column(rows, "would_emit_signal_ready", 0), errors="coerce").fillna(0).gt(0)]
            if preferred.empty:
                preferred = rows[pd.to_numeric(_column(rows, "selected_for_entry", 0), errors="coerce").fillna(0).gt(0)]
            if preferred.empty:
                preferred = rows.sort_values([column for column in ["live_rank", "timestamp"] if column in rows.columns], na_position="last")
            selected = preferred.iloc[0].to_dict()
        selected.update({key: value for key, value in top.items() if key not in selected or pd.isna(selected.get(key))})
        selected["session_date"] = str(selected.get("session_date") or top.get("session_date") or "")
        selected["symbol"] = symbol
        selected["top100_rank"] = selected.get("top100_rank") or top.get("rank")
        selected["top100_score"] = selected.get("top100_score") or top.get("score") or top.get("final_score")
        selected["ready_reason"] = selected.get("signal_ready_reason")
        selected["stale"] = int(bool(selected.get("stale_reason")))
        selected["position_qty"] = selected.get("quantity")
        current = fnum(selected.get("current_price"))
        selected["distance_from_or_low_pct"] = pct(current, fnum(selected.get("or_low")))
        selected["price_vs_open_pct"] = pct(current, fnum(selected.get("rth_open")))
        selected["or_complete"] = selected.get("first_15m_complete")
        records.append(selected)
    return pd.DataFrame(records)


def attach_full_feature_state(representative: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    if representative.empty or full.empty:
        return representative
    records: list[dict[str, Any]] = []
    full = full.copy()
    full["timestamp"] = pd.to_datetime(full.get("timestamp"), errors="coerce", utc=True)
    for row in representative.to_dict("records"):
        symbol_rows = full[full["symbol"].eq(row.get("symbol"))]
        when = parse_dt(row.get("timestamp"))
        if when is not None:
            symbol_rows = symbol_rows[symbol_rows["timestamp"] <= when]
        if not symbol_rows.empty:
            feature = symbol_rows.sort_values("timestamp").iloc[-1].to_dict()
            for key, value in feature.items():
                if key not in row or _missing(row.get(key)):
                    row[key] = value
            row["full_snapshot_attached"] = 1
        else:
            row["full_snapshot_attached"] = 0
        records.append(row)
    return pd.DataFrame(records)


def _actual_outcomes(trades: pd.DataFrame, session_date: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["session_date", "symbol", "actually_bought"])
    records: list[dict[str, Any]] = []
    for symbol, group in trades.groupby(trades["symbol"].map(normalize_symbol)):
        ordered = group.sort_values("entry_fill_time")
        qty = pd.to_numeric(ordered.get("quantity"), errors="coerce").fillna(0)
        net = pd.to_numeric(ordered.get("net_pnl", ordered.get("realized_pnl")), errors="coerce").fillna(0)
        entry_notional = (pd.to_numeric(ordered["entry_price"], errors="coerce") * qty).sum()
        records.append({
            "session_date": session_date, "symbol": symbol, "actually_bought": 1,
            "actual_entry_time": ordered.iloc[0].get("entry_fill_time"), "actual_entry_price": (entry_notional / qty.sum()) if qty.sum() else ordered.iloc[0].get("entry_price"),
            "actual_exit_time": ordered.iloc[-1].get("exit_fill_time"), "actual_exit_price": ordered.iloc[-1].get("exit_price"),
            "actual_net_pnl": float(net.sum()), "actual_return_pct": (float(net.sum()) / entry_notional * 100.0) if entry_notional else None,
            "actual_exit_reason": ";".join(dict.fromkeys(ordered.get("exit_reason", pd.Series(dtype=str)).fillna("").astype(str))),
            "canonical_trade_id": ";".join(ordered.get("trade_id", pd.Series(dtype=str)).fillna("").astype(str)),
        })
    return pd.DataFrame(records)


def _hypothetical(row: pd.Series, candles: pd.DataFrame, config: ReplayConfig) -> dict[str, Any]:
    signal = parse_dt(row.get("timestamp"))
    price = fnum(row.get("current_price"))
    if signal is None or price is None or candles.empty:
        return {"hypothetical_outcome_quality": "unavailable"}
    bars = candles.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    future = bars[bars["timestamp"] + pd.Timedelta(minutes=1) > signal].sort_values("timestamp")
    if future.empty:
        return {"hypothetical_outcome_quality": "unavailable"}
    entry_price = price * (1 + config.slippage_bps / 10000)
    qty = max(1, int(config.position_usd // entry_price))
    peak = entry_price
    low_seen = entry_price
    exit_time = future.iloc[-1]["timestamp"] + pd.Timedelta(minutes=1)
    exit_price = fnum(future.iloc[-1]["close"], entry_price) or entry_price
    reason = "replay_sell_model_eod"
    for _, bar in future.iterrows():
        peak = max(peak, fnum(bar.get("high"), entry_price) or entry_price)
        low_seen = min(low_seen, fnum(bar.get("low"), entry_price) or entry_price)
        stop = entry_price * (1 - config.exit_stop_loss_pct / 100)
        trail = peak * (1 - config.exit_trailing_stop_pct / 100)
        when = bar["timestamp"] + pd.Timedelta(minutes=1)
        if (fnum(bar.get("low"), entry_price) or entry_price) <= stop:
            exit_time, exit_price, reason = when, stop, "replay_sell_model_hard_stop"
            break
        if pct(peak, entry_price) is not None and pct(peak, entry_price) >= config.exit_trailing_activation_pct and (fnum(bar.get("low"), entry_price) or entry_price) <= trail:
            exit_time, exit_price, reason = when, trail, "replay_sell_model_trailing_stop"
            break
    exit_price *= 1 - config.slippage_bps / 10000
    gross = (exit_price - entry_price) * qty
    commission = max(config.min_commission, qty * config.commission_per_share) * 2
    close = pd.to_numeric(future["close"], errors="coerce")
    return {
        "hypothetical_entry_time": signal.isoformat(), "hypothetical_entry_price": entry_price,
        "hypothetical_exit_time": exit_time.isoformat(), "hypothetical_exit_price": exit_price,
        "hypothetical_gross_pnl": gross, "hypothetical_net_pnl": gross - commission,
        "hypothetical_return_pct": pct(exit_price, entry_price), "hypothetical_exit_reason": reason,
        "mfe_pct": pct(peak, entry_price), "mae_pct": pct(low_seen, entry_price),
        "minutes_to_mfe": int((future.loc[pd.to_numeric(future["high"], errors="coerce").idxmax(), "timestamp"] + pd.Timedelta(minutes=1) - signal).total_seconds() // 60),
        "minutes_to_mae": int((future.loc[pd.to_numeric(future["low"], errors="coerce").idxmin(), "timestamp"] + pd.Timedelta(minutes=1) - signal).total_seconds() // 60),
        "ever_positive": int(peak > entry_price),
        "positive_after_1m": int(len(close) >= 1 and close.iloc[0] > entry_price),
        "positive_after_3m": int(len(close) >= 3 and close.iloc[2] > entry_price),
        "positive_after_5m": int(len(close) >= 5 and close.iloc[4] > entry_price),
        "return_after_1m": pct(fnum(close.iloc[0]) if len(close) >= 1 else None, entry_price),
        "return_after_3m": pct(fnum(close.iloc[2]) if len(close) >= 3 else None, entry_price),
        "return_after_5m": pct(fnum(close.iloc[4]) if len(close) >= 5 else None, entry_price),
        "return_after_15m": pct(fnum(close.iloc[14]) if len(close) >= 15 else None, entry_price),
        "session_high_after_entry_pct": pct(fnum(pd.to_numeric(future["high"], errors="coerce").max()), entry_price),
        "session_low_after_entry_pct": pct(fnum(pd.to_numeric(future["low"], errors="coerce").min()), entry_price),
        "hypothetical_outcome_quality": "replay_sell_model",
    }


def feature_analysis(symbol_day: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcome = pd.to_numeric(symbol_day.get("hypothetical_net_pnl"), errors="coerce")
    for feature in FEATURES:
        if feature not in symbol_day or pd.to_numeric(symbol_day[feature], errors="coerce").notna().sum() < 2:
            continue
        values = pd.to_numeric(symbol_day[feature], errors="coerce")
        try:
            buckets = pd.qcut(values, min(5, values.nunique()), duplicates="drop")
        except ValueError:
            continue
        for bucket, group in symbol_day.assign(_bucket=buckets, _outcome=outcome).dropna(subset=["_bucket"]).groupby("_bucket", observed=True):
            pnl = pd.to_numeric(group["_outcome"], errors="coerce")
            returns = pd.to_numeric(group.get("hypothetical_return_pct"), errors="coerce")
            rows.append({
                "feature": feature, "bucket": str(bucket), "count": len(group), "win_rate": float((pnl > 0).mean() * 100),
                "median_return": returns.median(), "average_net_pnl": pnl.mean(), "total_net_pnl": pnl.sum(),
                "median_mfe": pd.to_numeric(group.get("mfe_pct"), errors="coerce").median(),
                "median_mae": pd.to_numeric(group.get("mae_pct"), errors="coerce").median(),
                "never_positive_count": int((pd.to_numeric(group.get("ever_positive"), errors="coerce") == 0).sum()),
                "immediate_failure_count": int((pd.to_numeric(group.get("return_after_1m"), errors="coerce") < -1).sum()),
                "interpretation": "INFERENCE; requires multi-day validation",
            })
    return pd.DataFrame(rows)


def _filter_specs(df: pd.DataFrame) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    eligible = lambda x: pd.to_numeric(_column(x, "potential_entry_eligible", 0), errors="coerce").fillna(0).gt(0)
    specs: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [("baseline", eligible)]
    for threshold in (10, 15, 20, 30, 40, 50):
        specs.append((f"top100_rank_le_{threshold}", lambda x, n=threshold: eligible(x) & (pd.to_numeric(_column(x, "top100_rank"), errors="coerce") <= n)))
    for threshold in (5, 10, 15, 20, 30):
        specs.append((f"live_rank_le_{threshold}", lambda x, n=threshold: eligible(x) & (pd.to_numeric(_column(x, "live_rank"), errors="coerce") <= n)))
    for threshold in (-0.25, -0.5, -1.0, -1.5):
        specs.append((f"distance_or_high_ge_{threshold}", lambda x, n=threshold: eligible(x) & (pd.to_numeric(_column(x, "distance_from_or_high_pct"), errors="coerce") >= n)))
    for threshold in (0, 1, 2, 3):
        specs.append((f"gap_ge_{threshold}", lambda x, n=threshold: eligible(x) & (pd.to_numeric(_column(x, "gap_from_previous_close_pct"), errors="coerce") >= n)))
    scores = pd.to_numeric(_column(df, "live_entry_score"), errors="coerce").dropna()
    for quantile in (0.5, 0.75):
        if not scores.empty:
            threshold = float(scores.quantile(quantile))
            specs.append((f"live_score_ge_p{int(quantile * 100)}", lambda x, n=threshold: eligible(x) & (pd.to_numeric(_column(x, "live_entry_score"), errors="coerce") >= n)))
    specs += [
        ("top10_persistent_2", lambda x: eligible(x) & (pd.to_numeric(_column(x, "consecutive_scans_top10"), errors="coerce") >= 2)),
        ("top20_persistent_3", lambda x: eligible(x) & (pd.to_numeric(_column(x, "consecutive_scans_top20"), errors="coerce") >= 3)),
        ("completed_bar_confirmation_positive", lambda x: eligible(x) & (pd.to_numeric(_column(x, "return_1m"), errors="coerce") > 0)),
        ("current_early_momentum_thresholds", lambda x: eligible(x) & (pd.to_numeric(_column(x, "first_5m_high_pct"), errors="coerce") >= 4.0) & (pd.to_numeric(_column(x, "first_15m_high_pct"), errors="coerce") >= 6.5)),
        ("late_bloomer_separate_setup", lambda x: eligible(x) & (pd.to_numeric(_column(x, "first_5m_high_pct"), errors="coerce") >= 0.5) & (pd.to_numeric(_column(x, "first_15m_high_pct"), errors="coerce") >= 1.0) & (pd.to_numeric(_column(x, "first_5m_high_pct"), errors="coerce") < 4.0)),
        ("top20_and_near_or_high", lambda x: eligible(x) & (pd.to_numeric(_column(x, "top100_rank"), errors="coerce") <= 20) & (pd.to_numeric(_column(x, "distance_from_or_high_pct"), errors="coerce") >= -0.5)),
    ]
    return specs


def portfolio_filter_simulation(symbol_day: pd.DataFrame, max_positions: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulation: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    baseline_symbols: set[str] = set()
    for name, predicate in _filter_specs(symbol_day):
        try:
            eligible = symbol_day[predicate(symbol_day).fillna(False)].copy()
        except Exception:
            eligible = symbol_day.iloc[0:0].copy()
        eligible["_entry"] = pd.to_datetime(eligible.get("hypothetical_entry_time"), errors="coerce", utc=True)
        eligible["_exit"] = pd.to_datetime(eligible.get("hypothetical_exit_time"), errors="coerce", utc=True)
        eligible = eligible.dropna(subset=["_entry", "_exit"]).sort_values(["_entry", "live_rank", "symbol"], na_position="last")
        active: list[pd.Timestamp] = []
        chosen: list[pd.Series] = []
        for _, row in eligible.iterrows():
            active = [value for value in active if value > row["_entry"]]
            if max_positions > 0 and len(active) >= max_positions:
                continue
            active.append(row["_exit"])
            chosen.append(row)
            replay_rows.append({"variant": name, "symbol": row["symbol"], "entry_time": row["_entry"], "exit_time": row["_exit"], "net_pnl": row.get("hypothetical_net_pnl"), "selection_rank": row.get("live_rank")})
        selected = pd.DataFrame(chosen)
        selected_symbols = set(selected.get("symbol", pd.Series(dtype=str)))
        if name == "baseline":
            baseline_symbols = selected_symbols
        pnl = pd.to_numeric(selected.get("hypothetical_net_pnl"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
        curve = pnl.fillna(0).cumsum()
        drawdown = (curve - curve.cummax()).min() if not curve.empty else 0.0
        all_winners = set(symbol_day.loc[pd.to_numeric(symbol_day.get("hypothetical_net_pnl"), errors="coerce") > 0, "symbol"])
        simulation.append({
            "variant": name, "eligible_symbols": len(eligible), "candidate_entries": len(eligible), "entries_selected": len(selected),
            "winners": int((pnl > 0).sum()), "losers": int((pnl <= 0).sum()), "win_rate": float((pnl > 0).mean() * 100) if len(pnl) else 0,
            "gross_pnl": pd.to_numeric(selected.get("hypothetical_gross_pnl"), errors="coerce").sum() if not selected.empty else 0,
            "net_pnl": pnl.sum(), "max_drawdown": drawdown, "max_concurrent_positions": max_positions,
            "missed_winners": len(all_winners - selected_symbols), "removed_losers": len(set(symbol_day.loc[pd.to_numeric(symbol_day.get("hypothetical_net_pnl"), errors="coerce") <= 0, "symbol"]) - selected_symbols),
            "false_exclusions": len((baseline_symbols - selected_symbols) & all_winners) if name != "baseline" else 0,
            "turnover": len(selected), "shared_trades": len(selected_symbols & baseline_symbols), "removed_trades": len(baseline_symbols - selected_symbols),
            "added_trades": len(selected_symbols - baseline_symbols), "interpretation": "HYPOTHESIS; replay_sell_model; requires multi-day validation",
        })
    result = pd.DataFrame(simulation)
    if not result.empty:
        base_pnl = fnum(result.loc[result["variant"].eq("baseline"), "net_pnl"].iloc[0], 0) or 0
        result["pnl_difference_vs_baseline"] = pd.to_numeric(result["net_pnl"], errors="coerce") - base_pnl
    return result, pd.DataFrame(replay_rows)


def analyze_session(
    session_date: str,
    *,
    sqlite_path: Path,
    history_dir: Path,
    recorder_dir: Path,
    top100_dir: Path,
    output_dir: Path,
    config: ReplayConfig | None = None,
) -> dict[str, Path]:
    config = config or profile_config("live")
    light = enrich_light_snapshots(read_snapshot_chunks(recorder_dir, session_date, "light"))
    ranking_hint = None
    if not light.empty and "ranking_source_date" in light:
        hints = light["ranking_source_date"].dropna().astype(str)
        ranking_hint = hints.iloc[0] if not hints.empty else None
    top100, top100_path, ranking_source_date = load_top100_source(top100_dir, session_date, ranking_hint)
    if top100.empty:
        raise RuntimeError(f"dated Top100 unavailable for session {session_date}")
    top100 = top100.copy()
    top100["session_date"] = session_date
    top100["ranking_source_date"] = ranking_source_date
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    prepared_rows_by_symbol: dict[str, pd.DataFrame] = {}
    prepared_sessions_by_symbol: dict[str, PreparedCausalSession] = {}
    prepared_cache = PreparedSessionCache(max_entries=max(1, len(top100)), max_bytes=256 * 1024 * 1024)
    for symbol in top100["symbol"].map(normalize_symbol):
        candles = load_session_candles(history_dir, symbol, session_date)
        candles_by_symbol[symbol] = candles
        rows = _rows(candles, config.bar_timestamp_semantics)
        prepared_rows_by_symbol[symbol] = rows
        if not rows.empty:
            prepared_sessions_by_symbol[symbol] = prepared_cache.get_or_build(symbol, session_date, rows, config)
    if light.empty:
        snapshots = replay_snapshots(
            session_date,
            top100,
            history_dir,
            config,
            prepared_rows_by_symbol=prepared_rows_by_symbol,
            prepared_sessions_by_symbol=prepared_sessions_by_symbol,
        )
    else:
        snapshots = light
        for symbol in top100["symbol"].map(normalize_symbol):
            if symbol not in set(snapshots["symbol"]):
                snapshots = pd.concat([snapshots, pd.DataFrame([{"session_date": session_date, "symbol": symbol, "snapshot_source": "missing", "runtime_observed": 0, "causal_valid": 0, "data_quality_flags": "missing_p1_symbol_snapshot"}])], ignore_index=True)
    representative = _representative_snapshots(snapshots, top100)
    full = read_snapshot_chunks(recorder_dir, session_date, "full")
    representative = attach_full_feature_state(representative, full)
    representative["session_date"] = session_date
    representative["ranking_source_date"] = ranking_source_date
    representative["top100_source"] = str(top100_path or "")
    representative["trading_session_date"] = session_date
    trades = load_finalized_canonical_trades(sqlite_path, session_date, session_date)
    actual = _actual_outcomes(trades, session_date)
    symbol_day = representative.merge(actual, on=["session_date", "symbol"], how="left")
    symbol_day["actually_bought"] = pd.to_numeric(_column(symbol_day, "actually_bought", 0), errors="coerce").fillna(0).astype(int)
    hypothetical: list[dict[str, Any]] = []
    history_symbols: set[str] = set()
    for _, row in symbol_day.iterrows():
        candles = candles_by_symbol.get(normalize_symbol(row["symbol"]), pd.DataFrame())
        if not candles.empty:
            history_symbols.add(row["symbol"])
        hypothetical.append(_hypothetical(row, candles, config))
    symbol_day = pd.concat([symbol_day.reset_index(drop=True), pd.DataFrame(hypothetical)], axis=1)
    symbol_day["potential_entry_eligible"] = (
        pd.to_numeric(_column(symbol_day, "would_emit_signal_ready", 0), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(_column(symbol_day, "ready", 0), errors="coerce").fillna(0).gt(0)
    ).astype(int)
    symbol_day["hypothetical_entry_basis"] = np.where(
        symbol_day["potential_entry_eligible"].eq(1), "causal_ready_snapshot", "representative_non_ready_observation"
    )
    aliases = {
        "first5_high_pct": "first_5m_high_pct", "first15_high_pct": "first_15m_high_pct",
        "first5_complete": "first_5m_complete", "first15_complete": "first_15m_complete",
        "live_score": "live_entry_score", "ready_reason": "signal_ready_reason",
    }
    for target, source in aliases.items():
        if source in symbol_day.columns:
            symbol_day[target] = symbol_day[source]
    features = feature_analysis(symbol_day)
    filters, portfolio = portfolio_filter_simulation(symbol_day, config.max_open_positions or 5)
    replay_top100 = top100[top100["symbol"].isin(history_symbols)].copy()
    if replay_top100.empty:
        full_replay = None
    else:
        with tempfile.TemporaryDirectory(prefix="top100-buy-replay-") as temp_dir:
            replay_top100_path = Path(temp_dir) / Path(top100_path).name
            replay_top100.to_csv(replay_top100_path, index=False)
            full_replay = replay_session(
                session_date=session_date,
                top100_path=replay_top100_path,
                history_dir=history_dir,
                config=config,
                prepared_rows_by_symbol=prepared_rows_by_symbol,
                prepared_sessions_by_symbol=prepared_sessions_by_symbol,
            )
    full_replay_rows = pd.DataFrame([
        {
            "variant": "full_session_v67_baseline", "symbol": row.get("symbol"),
            "entry_time": row.get("entry_time"), "exit_time": row.get("exit_time"),
            "net_pnl": row.get("net_pnl"), "gross_pnl": row.get("gross_pnl"),
            "selection_rank": row.get("candidate_rank"), "source": "causal_full_session_replay_v67",
        }
        for row in (full_replay.trades if full_replay is not None else [])
    ])
    if not full_replay_rows.empty:
        portfolio = pd.concat([portfolio, full_replay_rows], ignore_index=True)
    replay_pnl = pd.to_numeric(full_replay_rows.get("net_pnl"), errors="coerce").fillna(0) if not full_replay_rows.empty else pd.Series(dtype=float)
    replay_equity = pd.Series([value for _timestamp, value in full_replay.equity_curve], dtype=float) if full_replay is not None else pd.Series(dtype=float)
    replay_drawdown = float((replay_equity - replay_equity.cummax()).min()) if not replay_equity.empty else 0.0
    baseline_filter = filters[filters["variant"].eq("baseline")]
    filter_symbols = set(portfolio.loc[portfolio["variant"].eq("baseline"), "symbol"]) if not portfolio.empty else set()
    replay_symbols = set(full_replay_rows.get("symbol", pd.Series(dtype=str)))
    replay_summary_row = {
        "variant": "full_session_v67_baseline", "eligible_symbols": len(replay_top100), "candidate_entries": int(sum(1 for event in (full_replay.events if full_replay is not None else []) if event.get("event_type") == "SIGNAL")),
        "entries_selected": len(full_replay.trades) if full_replay is not None else 0, "winners": int((replay_pnl > 0).sum()), "losers": int((replay_pnl <= 0).sum()),
        "win_rate": float((replay_pnl > 0).mean() * 100) if len(replay_pnl) else 0.0,
        "gross_pnl": pd.to_numeric(full_replay_rows.get("gross_pnl"), errors="coerce").sum() if not full_replay_rows.empty else 0.0,
        "net_pnl": replay_pnl.sum(), "max_drawdown": replay_drawdown, "max_concurrent_positions": full_replay.max_concurrent_positions if full_replay is not None else 0,
        "shared_trades": len(filter_symbols & replay_symbols), "removed_trades": len(filter_symbols - replay_symbols), "added_trades": len(replay_symbols - filter_symbols),
        "baseline_parity_status": "exact_symbols" if filter_symbols == replay_symbols else "divergent_candidate_sampling",
        "interpretation": "FACT: existing causal full_session_replay_v67 baseline",
    }
    filters = pd.concat([filters, pd.DataFrame([replay_summary_row])], ignore_index=True)
    manifest = read_snapshot_manifest(recorder_dir, session_date)
    quality = {
        "session_date": session_date, "expected_top100_symbols": len(top100), "symbols_loaded": len(symbol_day),
        "symbols_with_history": len(history_symbols), "symbols_with_runtime_light": int(light["symbol"].nunique()) if not light.empty else 0,
        "symbols_with_full": int(full["symbol"].nunique()) if not full.empty else 0,
        "missing_scans": manifest.get("missing_scan_ranges"), "incomplete_session": manifest.get("session_completeness") not in {None, "COMPLETE"},
        "dropped_snapshot_batches": manifest.get("snapshot_batches_dropped", 0),
        "missing_spread": int(pd.to_numeric(_column(symbol_day, "spread_bps"), errors="coerce").isna().sum()),
        "missing_bid_ask": int((pd.to_numeric(_column(symbol_day, "bid"), errors="coerce").isna() | pd.to_numeric(_column(symbol_day, "ask"), errors="coerce").isna()).sum()),
        "missing_previous_close": int(pd.to_numeric(_column(symbol_day, "gap_from_previous_close_pct"), errors="coerce").isna().sum()),
        "full_replay_missing_history_symbols": sorted(set(top100["symbol"]) - history_symbols),
        "missing_premarket_data": int(symbol_day.get("premarket_data_quality", pd.Series(index=symbol_day.index, dtype=object)).fillna("").isin(["", "missing", "unavailable"]).sum()),
        "missing_canonical_outcome": int(symbol_day["actually_bought"].eq(0).sum()),
        "replay_only_symbols": int(symbol_day.get("snapshot_source", pd.Series(dtype=str)).eq("replay").sum()),
        "runtime_observed_symbols": int(pd.to_numeric(_column(symbol_day, "runtime_observed", 0), errors="coerce").fillna(0).gt(0).sum()),
        "causal_valid_count": int(pd.to_numeric(_column(symbol_day, "causal_valid", 0), errors="coerce").fillna(0).gt(0).sum()),
        "recommendation_strength": "insufficient" if len(symbol_day) < 300 else "exploratory",
    }
    prefix = output_dir / f"top100_buy_{{name}}_{session_date}"
    paths = {
        "symbol_day": Path(str(prefix).format(name="symbol_day") + ".csv"),
        "snapshots": Path(str(prefix).format(name="snapshots") + ".parquet"),
        "feature_analysis": Path(str(prefix).format(name="feature_analysis") + ".csv"),
        "filter_simulation": Path(str(prefix).format(name="filter_simulation") + ".csv"),
        "portfolio_replay": Path(str(prefix).format(name="portfolio_replay") + ".csv"),
        "summary": Path(str(prefix).format(name="summary") + ".md"),
        "data_quality": Path(str(prefix).format(name="data_quality") + ".json"),
    }
    for frame, key in [(symbol_day, "symbol_day"), (snapshots, "snapshots"), (features, "feature_analysis"), (filters, "filter_simulation"), (portfolio, "portfolio_replay")]:
        write_dataframe(frame, paths[key])
    paths["data_quality"].write_text(json.dumps(quality, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    baseline = filters[filters["variant"].eq("full_session_v67_baseline")].iloc[0].to_dict() if not filters.empty else {}
    paths["summary"].write_text("\n".join([
        f"# Top100 BUY Analysis {session_date}", "",
        f"FACT: dated_top100={top100_path} symbols={len(top100)} runtime_observed={quality['runtime_observed_symbols']} replay_only={quality['replay_only_symbols']}",
        f"FACT: actual_bought={int(symbol_day['actually_bought'].sum())} baseline_replay_entries={baseline.get('entries_selected', 0)} baseline_replay_net_pnl={baseline.get('net_pnl', 0)}",
        "INFERENCE: Feature buckets describe associations, not causal strategy improvements.",
        "HYPOTHESIS: Filter variants are experiments using replay_sell_model and portfolio slots.",
        "BASELINE ONLY: A single session cannot support production threshold changes.",
        "REQUIRES MULTI-DAY VALIDATION: Compare variants over independent completed sessions.",
        "POSSIBLE OVERFITTING: Filters were evaluated on the same outcomes they summarize.",
        "NOT AVAILABLE: Broker-only state is unavailable for replay-only symbols.",
    ]) + "\n", encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze BUY selection for every symbol in dated Top100 files.")
    parser.add_argument("--date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--sqlite-path", default="data/runtime/trading_runtime.sqlite")
    parser.add_argument("--history-dir", default="data/history/universe_1m")
    parser.add_argument("--recorder-dir", default="data/live/recorder")
    parser.add_argument("--top100-dir", default="data/universe")
    parser.add_argument("--output-dir", default="data/analysis")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dates = session_dates(args.date, args.start_date, args.end_date)
    completed: list[dict[str, Path]] = []
    for session_date in dates:
        summary = Path(args.output_dir) / f"top100_buy_summary_{session_date}.md"
        if summary.exists() and not args.force:
            print(f"TOP100_BUY_SKIP date={session_date} reason=output_exists", flush=True)
            continue
        print(f"TOP100_BUY_START date={session_date}", flush=True)
        paths = analyze_session(session_date, sqlite_path=Path(args.sqlite_path), history_dir=Path(args.history_dir), recorder_dir=Path(args.recorder_dir), top100_dir=Path(args.top100_dir), output_dir=Path(args.output_dir))
        completed.append(paths)
        print(f"TOP100_BUY_DONE date={session_date} output={paths['symbol_day']}", flush=True)
    if len(completed) > 1:
        suffix = f"{dates[0]}_to_{dates[-1]}"
        output_dir = Path(args.output_dir)
        for key, extension in (("symbol_day", ".csv"), ("snapshots", ".parquet"), ("feature_analysis", ".csv"), ("filter_simulation", ".csv"), ("portfolio_replay", ".csv")):
            frames = [pd.read_parquet(item[key]) if item[key].suffix == ".parquet" else pd.read_csv(item[key]) for item in completed if item[key].exists()]
            if frames:
                write_dataframe(pd.concat(frames, ignore_index=True), output_dir / f"top100_buy_{key}_{suffix}{extension}")
        qualities = [json.loads(item["data_quality"].read_text(encoding="utf-8")) for item in completed]
        (output_dir / f"top100_buy_data_quality_{suffix}.json").write_text(json.dumps({"sessions": qualities}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"top100_buy_summary_{suffix}.md").write_text(
            f"# Top100 BUY Analysis {suffix}\n\nFACT: sessions={len(completed)}.\nINFERENCE: Combined output preserves daily rows.\nREQUIRES MULTI-DAY VALIDATION.\nPOSSIBLE OVERFITTING.\n",
            encoding="utf-8",
        )
    return 0
