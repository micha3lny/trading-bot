from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
    "profile",
    "effective_config_json",
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
    "causal_valid",
    "matched_live_trade",
    "divergence_from_live",
    "divergence_type",
]

EVENT_COLUMNS = [
    "date",
    "profile",
    "effective_config_json",
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
    "profile",
    "effective_config_json",
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

PROFILE_COMPARISON_COLUMNS = [
    "date",
    "profile",
    "effective_config_json",
    "signals",
    "entries",
    "winners",
    "losers",
    "win_rate",
    "gross_pnl",
    "net_pnl",
    "max_drawdown",
    "average_mfe",
    "average_mae",
    "max_concurrent_positions",
    "blocked_candidates_by_reason",
    "trades_shared_with_live_profile",
    "trades_unique_to_profile",
    "symbols_entered_only_because_of_lower_thresholds",
    "symbols_entered_only_because_of_legacy_non_causal_behavior",
    "NUAI_status",
    "IREN_status",
    "FBYD_status",
]

PARITY_TRACE_COLUMNS = [
    "date",
    "profile",
    "symbol",
    "timestamp",
    "bar_open",
    "bar_high",
    "bar_low",
    "bar_close",
    "first_5m_high_pct",
    "first_15m_high_pct",
    "first_5m_complete",
    "first_15m_complete",
    "or_high",
    "or_low",
    "or_range_pct",
    "current_price",
    "spread_bps",
    "first5_gate",
    "first15_gate",
    "or_gate",
    "price_gate",
    "spread_gate",
    "ready",
    "reason",
    "replay_entry_time",
    "live_entry_time",
    "replay_entered_by_this_time",
    "live_entered_by_this_time",
    "first_divergence",
    "window_availability_mode",
    "effective_config_json",
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
    profile: str = "live"
    first5_threshold: float = LIVE_SIGNAL_MIN_FIRST_5M_HIGH_PCT
    first15_threshold: float = LIVE_SIGNAL_MIN_FIRST_15M_HIGH_PCT
    min_or_range_pct: float = LIVE_SIGNAL_MIN_OR_RANGE_PCT
    min_price: float = LIVE_SIGNAL_MIN_PRICE
    max_spread_bps: float = LIVE_SIGNAL_MAX_SPREAD_BPS
    breakout_mode: str = "live"
    entry_price_mode: str = "close"
    commission_model: str = "per_share"
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
    window_availability_mode: str = "live_partial"


def profile_config(profile: str) -> ReplayConfig:
    if profile == "live":
        return ReplayConfig(profile="live")
    if profile == "low_threshold_causal":
        return ReplayConfig(profile="low_threshold_causal", first5_threshold=0.5, first15_threshold=1.0)
    if profile == "legacy_offline":
        return ReplayConfig(
            profile="legacy_offline",
            first5_threshold=0.5,
            first15_threshold=1.0,
            breakout_mode="legacy_candle_high",
            bar_timestamp_semantics="bar_end",
            window_availability_mode="finalized_windows",
        )
    raise ValueError(f"unknown replay profile: {profile}")


def effective_config_dict(config: ReplayConfig) -> dict[str, Any]:
    return {
        "profile": config.profile,
        "first5_threshold": config.first5_threshold,
        "first15_threshold": config.first15_threshold,
        "or_max_range_pct": config.min_or_range_pct,
        "min_or_range_pct": config.min_or_range_pct,
        "min_price": config.min_price,
        "max_spread_bps": config.max_spread_bps,
        "breakout_mode": config.breakout_mode,
        "bar_timestamp_semantics": config.bar_timestamp_semantics,
        "window_availability_mode": config.window_availability_mode,
        "entry_price_mode": config.entry_price_mode,
        "notional": config.position_usd,
        "slippage_bps": config.slippage_bps,
        "commission_model": config.commission_model,
        "max_positions": config.max_open_positions,
        "max_entries_per_cycle": config.max_entries_per_cycle,
        "max_entries_per_minute": config.max_entries_per_minute,
        "entry_delay_after_open_minutes": config.entry_delay_after_open_minutes,
        "hard_stop_pct": config.exit_stop_loss_pct,
        "trailing_activation_pct": config.exit_trailing_activation_pct,
        "trailing_stop_pct": config.exit_trailing_stop_pct,
        "causal_valid": config.profile != "legacy_offline" and config.breakout_mode != "legacy_candle_high" and config.bar_timestamp_semantics == "bar_start",
    }


PREPARED_CAUSAL_SCHEMA_VERSION = "causal_session_v1"


@dataclass
class ReplayPerformanceCounters:
    prepared_session_builds: int = 0
    prepared_session_cache_hits: int = 0
    prepared_session_cache_misses: int = 0
    causal_cursor_advances: int = 0
    legacy_full_frame_feature_calls: int = 0
    completed_bar_full_frame_calls: int = 0
    replay_session_calls: int = 0
    replay_snapshots_calls: int = 0
    prepared_session_build_seconds: float = 0.0


_REPLAY_PERFORMANCE = ReplayPerformanceCounters()


def reset_replay_performance_counters() -> None:
    global _REPLAY_PERFORMANCE
    _REPLAY_PERFORMANCE = ReplayPerformanceCounters()


def replay_performance_counters() -> dict[str, Any]:
    return dict(vars(_REPLAY_PERFORMANCE))


def record_completed_bar_full_frame_call() -> None:
    _REPLAY_PERFORMANCE.completed_bar_full_frame_calls += 1


def record_replay_snapshots_call() -> None:
    _REPLAY_PERFORMANCE.replay_snapshots_calls += 1


@dataclass
class ReplayResult:
    trades: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    max_concurrent_positions: int = 0
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    effective_config_json: str = ""


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


def _feature_result(
    *,
    timestamp: pd.Timestamp,
    config: ReplayConfig,
    open_price: float | None,
    price: float | None,
    spread: float | None,
    current_high: float | None,
    first5_high: float | None,
    first15_high: float | None,
    or_high: float | None,
    or_low: float | None,
    start: pd.Timestamp,
) -> dict[str, Any]:
    if open_price is None or open_price <= 0:
        return {"ready": False, "reason": "invalid_open"}
    first5_end = start + pd.Timedelta(minutes=5)
    first15_end = start + pd.Timedelta(minutes=15)
    or_end = start + pd.Timedelta(seconds=LIVE_SIGNAL_OPENING_RANGE_SECONDS)
    first5_complete = timestamp >= first5_end
    first15_complete = timestamp >= first15_end
    or_complete = timestamp >= or_end
    if config.window_availability_mode == "finalized_windows":
        first5_high = first5_high if first5_complete else None
        first15_high = first15_high if first15_complete else None
        if not or_complete:
            or_high = None
            or_low = None
    first5_pct = pct(first5_high, open_price)
    first15_pct = pct(first15_high, open_price)
    or_range = (or_high / or_low - 1.0) * 100.0 if or_high is not None and or_low is not None and or_low > 0 else None
    score = 0.0
    for value, weight in [(first5_pct, 2.0), (first15_pct, 2.0), (or_range, 1.0)]:
        if value is not None:
            score += float(value) * weight
    if spread is not None and config.max_spread_bps > 0:
        score += max(0.0, config.max_spread_bps - spread) / config.max_spread_bps * 5.0
    reasons = []
    if first5_pct is None or first5_pct < config.first5_threshold:
        reasons.append("first_5m_high_too_low")
    if first15_pct is None or first15_pct < config.first15_threshold:
        reasons.append("first_15m_high_too_low")
    if or_range is None or or_range < config.min_or_range_pct:
        reasons.append("or_range_too_low")
    if price is None or price < config.min_price:
        reasons.append("price_too_low")
    if spread is not None and spread > config.max_spread_bps:
        reasons.append("spread_too_wide")
    breakout_gate_used = 0
    if config.breakout_mode == "legacy_candle_high":
        breakout_gate_used = 1
        if or_high is None or current_high is None or current_high < or_high:
            reasons.append("legacy_candle_high_breakout_not_met")
    elif config.breakout_mode == "current_price_or_high":
        breakout_gate_used = 1
        if or_high is None or price is None or price < or_high:
            reasons.append("current_price_breakout_not_met")
    return {
        "ready": not reasons,
        "reason": ";".join(reasons) if reasons else "live_safe_expansion_ready",
        "entry_price": price,
        "score": round(score, 4),
        "spread_bps": spread,
        "first_5m_high_pct": first5_pct,
        "first_15m_high_pct": first15_pct,
        "first_5m_complete": int(first5_complete),
        "first_15m_complete": int(first15_complete),
        "or_range_pct": or_range,
        "or_high": or_high,
        "or_low": or_low,
        "breakout_gate_used": breakout_gate_used,
    }


def _prefix_extreme(values: np.ndarray, included: np.ndarray, *, maximum: bool) -> list[float | None]:
    output: list[float | None] = []
    current: float | None = None
    for raw_value, use_value in zip(values, included):
        value = fnum(raw_value) if bool(use_value) else None
        if value is not None:
            current = value if current is None else (max(current, value) if maximum else min(current, value))
        output.append(current)
    return output


def _as_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return parse_dt(value)


class PreparedReplayFeatures:
    """Config-aware causal feature lookup over one pre-normalized candle frame."""

    def __init__(self, rows: pd.DataFrame, config: ReplayConfig):
        self.rows = rows
        self.config = config
        self._visible_index_cache: dict[int, int] = {}
        self._feature_cache: dict[int, dict[str, Any]] = {}
        self.start = rows.iloc[0]["timestamp"] if not rows.empty else None
        if rows.empty:
            self._available_ns = np.array([], dtype=np.int64)
            self._entry_values = np.array([], dtype=float)
            self._close_values = np.array([], dtype=float)
            self._open_values = np.array([], dtype=float)
            self._high_values = np.array([], dtype=float)
            self._spread_values = np.array([], dtype=float)
            self._has_spread = False
            return
        self._available_ns = rows["available_at"].astype("int64").to_numpy()
        timestamps = rows["timestamp"]
        timestamp_ns = timestamps.astype("int64").to_numpy()
        highs = rows["high"].to_numpy() if "high" in rows.columns else np.full(len(rows), np.nan)
        first5_end_ns = int((self.start + pd.Timedelta(minutes=5)).value)
        first15_end_ns = int((self.start + pd.Timedelta(minutes=15)).value)
        or_end_ns = int((self.start + pd.Timedelta(seconds=LIVE_SIGNAL_OPENING_RANGE_SECONDS)).value)
        self._first5_high = _prefix_extreme(highs, timestamp_ns < first5_end_ns, maximum=True)
        self._first15_high = _prefix_extreme(highs, timestamp_ns < first15_end_ns, maximum=True)
        self._or_high = _prefix_extreme(highs, timestamp_ns < or_end_ns, maximum=True)
        lows = rows["low"].to_numpy() if "low" in rows.columns else np.full(len(rows), np.nan)
        self._or_low = _prefix_extreme(lows, timestamp_ns < or_end_ns, maximum=False)
        self._open_price = fnum(rows.iloc[0].get("open"))
        self._entry_values = rows[config.entry_price_mode].to_numpy() if config.entry_price_mode in rows.columns else np.full(len(rows), np.nan)
        self._close_values = rows["close"].to_numpy() if "close" in rows.columns else np.full(len(rows), np.nan)
        self._open_values = rows["open"].to_numpy() if "open" in rows.columns else np.full(len(rows), np.nan)
        self._high_values = highs
        self._has_spread = "spread_bps" in rows.columns
        self._spread_values = rows["spread_bps"].to_numpy() if self._has_spread else np.full(len(rows), np.nan)

    def visible_index(self, timestamp: pd.Timestamp) -> int:
        if not len(self._available_ns):
            return -1
        when = _as_utc_timestamp(timestamp)
        if when is None:
            return -1
        key = int(when.value)
        if key not in self._visible_index_cache:
            self._visible_index_cache[key] = int(np.searchsorted(self._available_ns, key, side="right") - 1)
        return self._visible_index_cache[key]

    def latest_row(self, timestamp: pd.Timestamp) -> pd.Series | None:
        index = self.visible_index(timestamp)
        return None if index < 0 else self.rows.iloc[index]

    def at(self, timestamp: pd.Timestamp) -> dict[str, Any]:
        if self.rows.empty:
            return {"ready": False, "reason": "missing_candles"}
        when = _as_utc_timestamp(timestamp)
        index = -1 if when is None else self.visible_index(when)
        if when is None or index < 0:
            return {"ready": False, "reason": "no_completed_bar"}
        return self.at_index(index, when)

    def at_index(self, index: int, timestamp: pd.Timestamp) -> dict[str, Any]:
        when = _as_utc_timestamp(timestamp)
        if when is None or index < 0 or index >= len(self.rows):
            return {"ready": False, "reason": "no_completed_bar"}
        return self._at_prepared_index(index, when, int(when.value))

    def _at_prepared_index(self, index: int, when: pd.Timestamp, key: int) -> dict[str, Any]:
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        def array_value(values: np.ndarray, fallback: float | None = None) -> float | None:
            value = float(values[index])
            return fallback if np.isnan(value) else value

        open_value = array_value(self._open_values)
        close_value = array_value(self._close_values, open_value)
        price = array_value(self._entry_values, close_value)
        spread = array_value(self._spread_values) if self._has_spread else None
        result = _feature_result(
            timestamp=when,
            config=self.config,
            open_price=self._open_price,
            price=price,
            spread=spread,
            current_high=array_value(self._high_values),
            first5_high=self._first5_high[index],
            first15_high=self._first15_high[index],
            or_high=self._or_high[index],
            or_low=self._or_low[index],
            start=self.start,
        )
        self._feature_cache[key] = result
        return result

    @property
    def approximate_bytes(self) -> int:
        arrays = (
            self._available_ns, self._entry_values, self._close_values, self._open_values,
            self._high_values, self._spread_values,
        )
        return int(sum(value.nbytes for value in arrays if isinstance(value, np.ndarray)))


class PreparedCompletedBarFeatures:
    """Vectorized causal completed-bar features over normalized bar-start rows."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        if rows.empty:
            self._completed_ns = np.array([], dtype=np.int64)
            self._values: dict[str, np.ndarray] = {}
            return
        self._completed_ns = (rows["timestamp"] + pd.Timedelta(minutes=1)).astype("int64").to_numpy()
        open_values = rows["open"].to_numpy(dtype=float) if "open" in rows.columns else np.full(len(rows), np.nan)
        high_values = rows["high"].to_numpy(dtype=float) if "high" in rows.columns else np.full(len(rows), np.nan)
        low_values = rows["low"].to_numpy(dtype=float) if "low" in rows.columns else np.full(len(rows), np.nan)
        close_values = rows["close"].to_numpy(dtype=float) if "close" in rows.columns else np.full(len(rows), np.nan)
        volume_values = rows["volume"].to_numpy(dtype=float) if "volume" in rows.columns else np.zeros(len(rows))
        close = pd.Series(close_values, copy=False)
        high = pd.Series(high_values, copy=False)
        low = pd.Series(low_values, copy=False)
        volume = pd.Series(volume_values, copy=False)
        green = pd.Series((close_values > open_values).astype(np.int64), copy=False)
        bar_range = high_values - low_values
        close_location = np.divide(
            close_values - low_values,
            bar_range,
            out=np.full(len(rows), np.nan),
            where=bar_range > 0,
        )
        recent_high = high.rolling(5, min_periods=1).max().to_numpy()
        recent_low = low.rolling(5, min_periods=1).min().to_numpy()
        prior_volume = volume.shift(1).rolling(5, min_periods=1).mean().to_numpy()
        volume_acceleration = np.divide(
            volume_values,
            prior_volume,
            out=np.full(len(rows), np.nan),
            where=np.isfinite(prior_volume) & (prior_volume > 0),
        )

        def returns(periods: int) -> np.ndarray:
            previous = close.shift(periods).to_numpy()
            ratio = np.divide(
                close_values,
                previous,
                out=np.full(len(rows), np.nan),
                where=np.isfinite(previous) & (previous != 0),
            )
            return (ratio - 1.0) * 100.0

        recent_high_ratio = np.divide(
            close_values,
            recent_high,
            out=np.full(len(rows), np.nan),
            where=np.isfinite(recent_high) & (recent_high != 0),
        )
        recent_low_ratio = np.divide(
            close_values,
            recent_low,
            out=np.full(len(rows), np.nan),
            where=np.isfinite(recent_low) & (recent_low != 0),
        )

        self._values = {
            "return_1m": returns(1),
            "return_3m": returns(3),
            "return_5m": returns(5),
            "green_bars_last_3": green.rolling(3, min_periods=1).sum().to_numpy(dtype=np.int64),
            "green_bars_last_5": green.rolling(5, min_periods=1).sum().to_numpy(dtype=np.int64),
            "close_location_value": close_location,
            "pullback_from_recent_high_pct": (recent_high_ratio - 1.0) * 100.0,
            "recent_low_distance_pct": (recent_low_ratio - 1.0) * 100.0,
            "volume_acceleration": volume_acceleration,
        }

    def at(self, timestamp: Any) -> dict[str, Any]:
        when = _as_utc_timestamp(timestamp)
        if when is None or not len(self._completed_ns):
            return {}
        index = int(np.searchsorted(self._completed_ns, int(when.value), side="right") - 1)
        if index < 0:
            return {}
        return self.at_index(index)

    def at_index(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self._completed_ns):
            return {}
        def value(name: str) -> float | None:
            raw = self._values[name][index]
            return None if np.isnan(raw) else float(raw)

        return {
            "return_1m": value("return_1m"),
            "return_3m": value("return_3m"),
            "return_5m": value("return_5m"),
            "green_bars_last_3": int(self._values["green_bars_last_3"][index]),
            "green_bars_last_5": int(self._values["green_bars_last_5"][index]),
            "close_location_value": value("close_location_value"),
            "pullback_from_recent_high_pct": value("pullback_from_recent_high_pct"),
            "recent_low_distance_pct": value("recent_low_distance_pct"),
            "volume_acceleration": value("volume_acceleration"),
        }

    @property
    def approximate_bytes(self) -> int:
        return int(self._completed_ns.nbytes + sum(value.nbytes for value in self._values.values()))


class PreparedCausalSession:
    """One normalized symbol/session shared by snapshot and portfolio replay paths."""

    def __init__(self, symbol: str, session_date: str, rows: pd.DataFrame, config: ReplayConfig):
        started = time.perf_counter()
        self.symbol = normalize_symbol(symbol)
        self.session_date = session_date
        self.rows = rows
        self.config = config
        self.config_identity = json.dumps(effective_config_dict(config), sort_keys=True)
        self.replay_features = PreparedReplayFeatures(rows, config)
        self.completed_bar_features = PreparedCompletedBarFeatures(rows)
        self.available_timestamps = (
            pd.DatetimeIndex(rows["available_at"].dropna().unique()).sort_values()
            if not rows.empty
            else pd.DatetimeIndex([])
        )
        available_ns = self.available_timestamps.astype("int64").to_numpy()
        self._available_timestamp_ns = available_ns
        self._visible_indices = np.searchsorted(
            self.replay_features._available_ns, available_ns, side="right"
        ) - 1
        self._completed_indices = np.searchsorted(
            self.completed_bar_features._completed_ns, available_ns, side="right"
        ) - 1
        _REPLAY_PERFORMANCE.prepared_session_builds += 1
        _REPLAY_PERFORMANCE.prepared_session_build_seconds += time.perf_counter() - started

    def iter_features(self):
        for timestamp, timestamp_ns, visible_index, completed_index in zip(
            self.available_timestamps,
            self._available_timestamp_ns,
            self._visible_indices,
            self._completed_indices,
        ):
            _REPLAY_PERFORMANCE.causal_cursor_advances += 1
            when = pd.Timestamp(timestamp)
            yield (
                when,
                self.replay_features._at_prepared_index(int(visible_index), when, int(timestamp_ns)),
                self.completed_bar_features.at_index(int(completed_index)),
            )

    @property
    def approximate_bytes(self) -> int:
        frame_bytes = int(self.rows.memory_usage(index=True, deep=True).sum()) if not self.rows.empty else 0
        return (
            frame_bytes
            + int(getattr(self.replay_features, "approximate_bytes", 0))
            + int(getattr(self.completed_bar_features, "approximate_bytes", 0))
        )


class PreparedSessionCache:
    """Bounded cache keyed by symbol, session, schema and full replay configuration."""

    def __init__(self, *, max_entries: int = 128, max_bytes: int = 256 * 1024 * 1024):
        self.max_entries = max(1, max_entries)
        self.max_bytes = max(1, max_bytes)
        self._entries: OrderedDict[tuple[str, str, str, str, str], PreparedCausalSession] = OrderedDict()
        self.approximate_bytes = 0

    @staticmethod
    def key(symbol: str, session_date: str, session_type: str, config: ReplayConfig) -> tuple[str, str, str, str, str]:
        return (
            normalize_symbol(symbol),
            session_date,
            session_type.upper(),
            PREPARED_CAUSAL_SCHEMA_VERSION,
            json.dumps(effective_config_dict(config), sort_keys=True),
        )

    def get_or_build(
        self,
        symbol: str,
        session_date: str,
        rows: pd.DataFrame,
        config: ReplayConfig,
        *,
        session_type: str = "RTH",
    ) -> PreparedCausalSession:
        key = self.key(symbol, session_date, session_type, config)
        cached = self._entries.pop(key, None)
        if cached is not None and cached.rows is rows:
            self._entries[key] = cached
            _REPLAY_PERFORMANCE.prepared_session_cache_hits += 1
            return cached
        if cached is not None:
            self.approximate_bytes -= cached.approximate_bytes
        _REPLAY_PERFORMANCE.prepared_session_cache_misses += 1
        prepared = PreparedCausalSession(symbol, session_date, rows, config)
        self._entries[key] = prepared
        self.approximate_bytes += prepared.approximate_bytes
        while len(self._entries) > self.max_entries or self.approximate_bytes > self.max_bytes:
            _old_key, old = self._entries.popitem(last=False)
            self.approximate_bytes -= old.approximate_bytes
        return prepared

    @property
    def frame_count(self) -> int:
        return len(self._entries)


def _feature_at(rows: pd.DataFrame, timestamp: pd.Timestamp, config: ReplayConfig) -> dict[str, Any]:
    _REPLAY_PERFORMANCE.legacy_full_frame_feature_calls += 1
    if rows.empty:
        return {"ready": False, "reason": "missing_candles"}
    start = rows.iloc[0]["timestamp"]
    visible = rows[rows["available_at"] <= timestamp]
    if visible.empty:
        return {"ready": False, "reason": "no_completed_bar"}
    open_price = fnum(rows.iloc[0].get("open"))
    current = visible.iloc[-1]
    price = fnum(current.get(config.entry_price_mode), fnum(current.get("close"), fnum(current.get("open"))))
    first5_end = start + pd.Timedelta(minutes=5)
    first15_end = start + pd.Timedelta(minutes=15)
    or_end = start + pd.Timedelta(seconds=LIVE_SIGNAL_OPENING_RANGE_SECONDS)
    first5_complete = timestamp >= first5_end
    first15_complete = timestamp >= first15_end
    or_complete = timestamp >= or_end
    if config.window_availability_mode == "finalized_windows":
        first5 = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < first5_end)] if first5_complete else pd.DataFrame()
        first15 = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < first15_end)] if first15_complete else pd.DataFrame()
        or_rows = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < or_end)] if or_complete else pd.DataFrame()
    else:
        first5 = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < first5_end)]
        first15 = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < first15_end)]
        or_rows = visible[(visible["timestamp"] >= start) & (visible["timestamp"] < or_end)]
    first5_high = fnum(first5["high"].max()) if not first5.empty else None
    first15_high = fnum(first15["high"].max()) if not first15.empty else None
    or_high = fnum(or_rows["high"].max()) if not or_rows.empty else None
    or_low = fnum(or_rows["low"].min()) if not or_rows.empty else None
    spread = fnum(current.get("spread_bps")) if "spread_bps" in visible.columns else None
    return _feature_result(
        timestamp=timestamp,
        config=config,
        open_price=open_price,
        price=price,
        spread=spread,
        current_high=fnum(current.get("high")),
        first5_high=first5_high,
        first15_high=first15_high,
        or_high=or_high,
        or_low=or_low,
        start=start,
    )


def _event(result: ReplayResult, session_date: str, timestamp: pd.Timestamp, event_type: str, symbol: str = "", **kwargs: Any) -> None:
    result.events.append({
        "date": session_date,
        "profile": kwargs.get("profile", ""),
        "effective_config_json": kwargs.get("effective_config_json", ""),
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
    commission = 0.0 if config.commission_model == "none" else max(config.min_commission, pos.quantity * config.commission_per_share) * 2.0
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
        "profile": config.profile,
        "effective_config_json": json.dumps(effective_config_dict(config), sort_keys=True),
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
        "causal_valid": int(bool(effective_config_dict(config)["causal_valid"])),
        "matched_live_trade": "",
        "divergence_from_live": "",
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
            _event(result, session_date, timestamp, "EXIT", symbol, profile=config.profile, effective_config_json=json.dumps(effective_config_dict(config), sort_keys=True), price=exit_price, reason=reason, open_positions=len(open_positions) - 1)
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
    prepared_rows_by_symbol: dict[str, pd.DataFrame] | None = None,
    prepared_features_by_symbol: dict[str, PreparedReplayFeatures] | None = None,
    prepared_sessions_by_symbol: dict[str, PreparedCausalSession] | None = None,
) -> ReplayResult:
    _REPLAY_PERFORMANCE.replay_session_calls += 1
    result = ReplayResult()
    config_json = json.dumps(effective_config_dict(config), sort_keys=True)
    result.effective_config_json = config_json
    top100 = load_top100(top100_path)
    if top100.empty:
        return result
    symbols = [normalize_symbol(value) for value in top100["symbol"].tolist() if normalize_symbol(value)]
    if prepared_sessions_by_symbol is not None:
        rows_by_symbol = {
            symbol: prepared_sessions_by_symbol[symbol].rows
            for symbol in symbols
            if symbol in prepared_sessions_by_symbol
        }
    elif prepared_rows_by_symbol is None:
        rows_by_symbol = {symbol: _rows(load_session_candles(history_dir, symbol, session_date, "RTH"), config.bar_timestamp_semantics) for symbol in symbols}
    else:
        rows_by_symbol = {symbol: prepared_rows_by_symbol.get(symbol, pd.DataFrame()) for symbol in symbols}
    rows_by_symbol = {symbol: rows for symbol, rows in rows_by_symbol.items() if not rows.empty}
    if not rows_by_symbol:
        return result
    non_empty = list(rows_by_symbol.values())
    start = min(rows["available_at"].min() for rows in non_empty)
    end = max(rows["available_at"].max() for rows in non_empty)
    eod = _eod_timestamp(rows_by_symbol, config)
    if eod is not None:
        end = min(end, eod)
    entry_delay_until = min(rows["timestamp"].min() for rows in non_empty) + pd.Timedelta(minutes=config.entry_delay_after_open_minutes)
    open_positions: dict[str, ReplayPosition] = {}
    feature_lookups: dict[str, PreparedReplayFeatures] = {}
    for symbol, rows in rows_by_symbol.items():
        prepared_session = prepared_sessions_by_symbol.get(symbol) if prepared_sessions_by_symbol is not None else None
        prepared = (
            prepared_session.replay_features
            if prepared_session is not None and prepared_session.config == config and prepared_session.rows is rows
            else (prepared_features_by_symbol.get(symbol) if prepared_features_by_symbol is not None else None)
        )
        feature_lookups[symbol] = (
            prepared
            if prepared is not None and prepared.config == config and prepared.rows is rows
            else PreparedReplayFeatures(rows, config)
        )
    traded_symbols: set[str] = set()
    signaled_symbols: set[str] = set()
    realized = 0.0
    current = start.floor("min")
    while current <= end.ceil("min"):
        candle_by_symbol = {}
        for symbol, lookup in feature_lookups.items():
            latest = lookup.latest_row(current)
            if latest is not None:
                candle_by_symbol[symbol] = latest
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
        for symbol, lookup in feature_lookups.items():
            if symbol in open_positions:
                continue
            if config.max_one_trade_per_symbol_per_day and symbol in traded_symbols:
                continue
            features = lookup.at(current)
            if not features.get("ready"):
                result.skipped[str(features.get("reason") or "not_ready")] = result.skipped.get(str(features.get("reason") or "not_ready"), 0) + 1
                continue
            if symbol not in signaled_symbols:
                signaled_symbols.add(symbol)
                _event(result, session_date, current, "SIGNAL", symbol, profile=config.profile, effective_config_json=config_json, score=features.get("score"), price=features.get("entry_price"), reason=features.get("reason"), open_positions=len(open_positions))
            if current < entry_delay_until:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, profile=config.profile, effective_config_json=config_json, score=features.get("score"), price=features.get("entry_price"), reason="entry_delay_after_open", open_positions=len(open_positions))
                result.skipped["entry_delay_after_open"] = result.skipped.get("entry_delay_after_open", 0) + 1
                continue
            candidates.append((symbol, float(features.get("score") or 0.0), features))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        entries_this_minute = 0
        for rank, (symbol, score, features) in enumerate(candidates, start=1):
            if config.max_open_positions > 0 and len(open_positions) >= config.max_open_positions:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, profile=config.profile, effective_config_json=config_json, candidate_rank=rank, score=score, price=features.get("entry_price"), reason="max_positions_full", open_positions=len(open_positions))
                result.skipped["max_positions_full"] = result.skipped.get("max_positions_full", 0) + 1
                continue
            if config.max_entries_per_cycle > 0 and entries_this_minute >= config.max_entries_per_cycle:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, profile=config.profile, effective_config_json=config_json, candidate_rank=rank, score=score, price=features.get("entry_price"), reason="max_entries_per_cycle", open_positions=len(open_positions))
                result.skipped["max_entries_per_cycle"] = result.skipped.get("max_entries_per_cycle", 0) + 1
                continue
            if config.max_entries_per_minute > 0 and entries_this_minute >= config.max_entries_per_minute:
                _event(result, session_date, current, "ENTRY_BLOCKED", symbol, profile=config.profile, effective_config_json=config_json, candidate_rank=rank, score=score, price=features.get("entry_price"), reason="max_entries_per_minute", open_positions=len(open_positions))
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
            _event(result, session_date, current, "ENTRY", symbol, profile=config.profile, effective_config_json=config_json, candidate_rank=rank, score=score, price=entry_price, quantity=qty, reason="entered", open_positions=len(open_positions))
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
            "profile": "live_actual",
            "effective_config_json": json.dumps({"profile": "live_actual"}, sort_keys=True),
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
            "causal_valid": "",
            "matched_live_trade": "",
            "divergence_from_live": "",
            "divergence_type": "",
        })
    return rows



def build_parity_trace(
    *,
    session_date: str,
    symbols: list[str],
    top100_path: Path,
    history_dir: Path,
    config: ReplayConfig,
    replay: ReplayResult,
    live: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config_json = json.dumps(effective_config_dict(config), sort_keys=True)
    top100 = load_top100(top100_path)
    top100_symbols = {normalize_symbol(value) for value in top100.get("symbol", pd.Series(dtype=str)).tolist()}
    focus = [normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)]
    replay_entry_by_symbol = {
        normalize_symbol(event.get("symbol")): parse_dt(event.get("timestamp"))
        for event in replay.events
        if event.get("event_type") == "ENTRY"
    }
    live_entry_by_symbol: dict[str, pd.Timestamp] = {}
    for row in live:
        symbol = normalize_symbol(row.get("symbol"))
        entry_time = parse_dt(row.get("entry_time"))
        if symbol and entry_time is not None and symbol not in live_entry_by_symbol:
            live_entry_by_symbol[symbol] = entry_time
    out: list[dict[str, Any]] = []
    for symbol in focus:
        if top100_symbols and symbol not in top100_symbols:
            out.append({
                "date": session_date,
                "profile": config.profile,
                "symbol": symbol,
                "timestamp": "",
                "ready": 0,
                "reason": "symbol_not_in_top100",
                "replay_entry_time": iso_ts(replay_entry_by_symbol.get(symbol)),
                "live_entry_time": iso_ts(live_entry_by_symbol.get(symbol)),
                "first_divergence": "symbol_not_in_replay_top100",
                "window_availability_mode": config.window_availability_mode,
                "effective_config_json": config_json,
            })
            continue
        rows = _rows(load_session_candles(history_dir, symbol, session_date, "RTH"), config.bar_timestamp_semantics)
        if rows.empty:
            out.append({
                "date": session_date,
                "profile": config.profile,
                "symbol": symbol,
                "timestamp": "",
                "ready": 0,
                "reason": "missing_candles",
                "replay_entry_time": iso_ts(replay_entry_by_symbol.get(symbol)),
                "live_entry_time": iso_ts(live_entry_by_symbol.get(symbol)),
                "first_divergence": "missing_replay_candles",
                "window_availability_mode": config.window_availability_mode,
                "effective_config_json": config_json,
            })
            continue
        start = rows["available_at"].min().floor("min")
        end = rows["available_at"].max().ceil("min")
        replay_entry = replay_entry_by_symbol.get(symbol)
        live_entry = live_entry_by_symbol.get(symbol)
        first_divergence_seen = False
        current = start
        while current <= end:
            visible = rows[rows["available_at"] <= current]
            bar = visible.iloc[-1] if not visible.empty else pd.Series(dtype=object)
            features = _feature_at(rows, current, config)
            replay_entered = replay_entry is not None and replay_entry <= current
            live_entered = live_entry is not None and live_entry <= current
            divergence = ""
            if not first_divergence_seen and replay_entered != live_entered:
                divergence = "replay_entered_before_live" if replay_entered else "live_entered_before_replay"
                first_divergence_seen = True
            first5 = fnum(features.get("first_5m_high_pct"))
            first15 = fnum(features.get("first_15m_high_pct"))
            or_range = fnum(features.get("or_range_pct"))
            price = fnum(features.get("entry_price"))
            spread = fnum(features.get("spread_bps"))
            out.append({
                "date": session_date,
                "profile": config.profile,
                "symbol": symbol,
                "timestamp": iso_ts(current),
                "bar_open": bar.get("open", ""),
                "bar_high": bar.get("high", ""),
                "bar_low": bar.get("low", ""),
                "bar_close": bar.get("close", ""),
                "first_5m_high_pct": first5,
                "first_15m_high_pct": first15,
                "first_5m_complete": features.get("first_5m_complete", ""),
                "first_15m_complete": features.get("first_15m_complete", ""),
                "or_high": features.get("or_high", ""),
                "or_low": features.get("or_low", ""),
                "or_range_pct": or_range,
                "current_price": price,
                "spread_bps": spread,
                "first5_gate": int(first5 is not None and first5 >= config.first5_threshold),
                "first15_gate": int(first15 is not None and first15 >= config.first15_threshold),
                "or_gate": int(or_range is not None and or_range >= config.min_or_range_pct),
                "price_gate": int(price is not None and price >= config.min_price),
                "spread_gate": int(spread is None or spread <= config.max_spread_bps),
                "ready": int(bool(features.get("ready"))),
                "reason": features.get("reason"),
                "replay_entry_time": iso_ts(replay_entry),
                "live_entry_time": iso_ts(live_entry),
                "replay_entered_by_this_time": int(replay_entered),
                "live_entered_by_this_time": int(live_entered),
                "first_divergence": divergence,
                "window_availability_mode": config.window_availability_mode,
                "effective_config_json": config_json,
            })
            current += pd.Timedelta(minutes=1)
    return out

def compare_trades(session_date: str, offline: list[dict[str, Any]], live: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in live:
        live_by_symbol.setdefault(str(row.get("symbol") or ""), []).append(row)
    used: set[tuple[str, int]] = set()
    out = []
    for off in offline:
        sym = str(off.get("symbol") or "")
        candidates = live_by_symbol.get(sym, [])
        match_idx = None
        for idx, live_row in enumerate(candidates):
            if (sym, idx) not in used:
                match_idx = idx
                break
        live_row = candidates[match_idx] if match_idx is not None else None
        if match_idx is not None:
            used.add((sym, match_idx))
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
            "profile": off.get("profile", ""),
            "effective_config_json": off.get("effective_config_json", ""),
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
                "profile": "live_actual",
                "effective_config_json": json.dumps({"profile": "live_actual"}, sort_keys=True),
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


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [fnum(row.get(key)) for row in rows]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _focus_symbol_status(replay: ReplayResult, symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    if any(normalize_symbol(row.get("symbol")) == symbol for row in replay.trades):
        return "entered"
    blocks = [event for event in replay.events if normalize_symbol(event.get("symbol")) == symbol and event.get("event_type") == "ENTRY_BLOCKED"]
    if blocks:
        return f"blocked:{blocks[0].get('reason') or 'unknown'}"
    return "not_entered"


def profile_comparison_rows(session_date: str, profile_results: dict[str, ReplayResult]) -> list[dict[str, Any]]:
    live_symbols = {str(row.get("symbol") or "") for row in profile_results.get("live", ReplayResult()).trades}
    low_symbols = {str(row.get("symbol") or "") for row in profile_results.get("low_threshold_causal", ReplayResult()).trades}
    legacy_symbols = {str(row.get("symbol") or "") for row in profile_results.get("legacy_offline", ReplayResult()).trades}
    rows = []
    for profile, replay in profile_results.items():
        metrics = summary_metrics(replay.trades)
        symbols = {str(row.get("symbol") or "") for row in replay.trades}
        rows.append({
            "date": session_date,
            "profile": profile,
            "effective_config_json": replay.effective_config_json,
            "signals": len([event for event in replay.events if event.get("event_type") == "SIGNAL"]),
            "entries": metrics["entries"],
            "winners": metrics["winners"],
            "losers": metrics["losers"],
            "win_rate": metrics["win_rate"],
            "gross_pnl": metrics["gross_pnl"],
            "net_pnl": metrics["net_pnl"],
            "max_drawdown": max_drawdown(replay.equity_curve),
            "average_mfe": _avg(replay.trades, "mfe_pct"),
            "average_mae": _avg(replay.trades, "mae_pct"),
            "max_concurrent_positions": replay.max_concurrent_positions,
            "blocked_candidates_by_reason": json.dumps(replay.skipped, sort_keys=True),
            "trades_shared_with_live_profile": len(symbols & live_symbols) if profile != "live" else len(symbols),
            "trades_unique_to_profile": ",".join(sorted(symbols - live_symbols)) if profile != "live" else "",
            "symbols_entered_only_because_of_lower_thresholds": ",".join(sorted((low_symbols - live_symbols) & symbols)) if profile == "low_threshold_causal" else "",
            "symbols_entered_only_because_of_legacy_non_causal_behavior": ",".join(sorted((legacy_symbols - live_symbols - low_symbols) & symbols)) if profile == "legacy_offline" else "",
            "NUAI_status": _focus_symbol_status(replay, "NUAI"),
            "IREN_status": _focus_symbol_status(replay, "IREN"),
            "FBYD_status": _focus_symbol_status(replay, "FBYD"),
        })
    return rows


def write_summary(path: Path, session_date: str, replay: ReplayResult, live: list[dict[str, Any]], comparison: list[dict[str, Any]], focus_symbols: list[str], config: ReplayConfig) -> None:
    offline_metrics = summary_metrics(replay.trades)
    live_metrics = summary_metrics(live)
    comp_counts: dict[str, int] = {}
    for row in comparison:
        key = str(row.get("divergence_type") or "unknown")
        comp_counts[key] = comp_counts.get(key, 0) + 1
    lines = [
        f"# Full Session v67 Offline Replay {session_date}",
        "",
        f"Profile: `{config.profile}`",
        "",
        "FACT: This is a read-only causal replay over completed 1m bars. It does not alter live trading state.",
        f"FACT: Effective configuration: `{json.dumps(effective_config_dict(config), sort_keys=True)}`",
        "FACT: live and low_threshold_causal profiles use causal bar-start semantics. legacy_offline is diagnostic and may reproduce historical non-causal behavior.",
        "HYPOTHESIS: Differences versus live can come from intrabar tick prices, real spreads, IBKR permissions/subscriptions, or production blocks not intentionally simulated.",
        "",
        "## Offline Replay",
        f"- signals/entries={offline_metrics['entries']}",
        f"- winners={offline_metrics['winners']} losers={offline_metrics['losers']} win_rate={offline_metrics['win_rate']:.2f}%",
        f"- gross_pnl={offline_metrics['gross_pnl']:.4f} net_pnl={offline_metrics['net_pnl']:.4f} average_pnl={offline_metrics['average_pnl']:.4f}",
        f"- max_concurrent_positions={replay.max_concurrent_positions}",
        f"- max_drawdown={max_drawdown(replay.equity_curve):.4f}",
        f"- average_mfe={_avg(replay.trades, 'mfe_pct'):.4f} average_mae={_avg(replay.trades, 'mae_pct'):.4f}",
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


def build_effective_config(args: argparse.Namespace, profile: str) -> ReplayConfig:
    config = profile_config(profile)
    config.first5_threshold = args.first5_threshold if args.first5_threshold is not None else config.first5_threshold
    config.first15_threshold = args.first15_threshold if args.first15_threshold is not None else config.first15_threshold
    config.min_or_range_pct = args.or_max_range_pct if args.or_max_range_pct is not None else config.min_or_range_pct
    config.breakout_mode = args.breakout_mode or config.breakout_mode
    config.bar_timestamp_semantics = args.bar_timestamp_semantics or config.bar_timestamp_semantics
    config.window_availability_mode = args.window_availability_mode or config.window_availability_mode
    config.entry_price_mode = args.entry_price_mode or config.entry_price_mode
    config.position_usd = args.notional if args.notional is not None else (args.position_usd if args.position_usd is not None else config.position_usd)
    config.slippage_bps = args.slippage_bps if args.slippage_bps is not None else config.slippage_bps
    config.commission_model = args.commission_model or config.commission_model
    config.max_open_positions = args.max_positions if args.max_positions is not None else (args.max_open_positions if args.max_open_positions is not None else config.max_open_positions)
    config.exit_stop_loss_pct = args.hard_stop_pct if args.hard_stop_pct is not None else config.exit_stop_loss_pct
    config.exit_trailing_activation_pct = args.trailing_activation_pct if args.trailing_activation_pct is not None else config.exit_trailing_activation_pct
    config.exit_trailing_stop_pct = args.trailing_stop_pct if args.trailing_stop_pct is not None else config.exit_trailing_stop_pct
    config.max_entries_per_cycle = args.max_entries_per_cycle
    config.max_entries_per_minute = args.max_entries_per_minute
    config.entry_delay_after_open_minutes = args.entry_delay_after_open_minutes
    config.min_live_entry_score = args.min_live_entry_score
    return config


def _profile_suffix(profile: str) -> str:
    return "" if profile == "live" else f"_{profile}"


def run_one_profile(args: argparse.Namespace, profile: str, live: list[dict[str, Any]] | None = None) -> tuple[ReplayResult, list[dict[str, Any]]]:
    top100_path = args.top100 or Path(f"data/universe/daily_top100_{args.date}.csv")
    config = build_effective_config(args, profile)
    replay = replay_session(session_date=args.date, top100_path=top100_path, history_dir=args.history_dir, config=config)
    live = live if live is not None else load_live_trades(args.sqlite_path, args.date)
    offline_rows = replay.trades
    for row in offline_rows:
        row["date"] = args.date
        row["profile"] = profile
    comparison = compare_trades(args.date, offline_rows, live)
    divergence_by_symbol = {str(row.get("symbol") or ""): str(row.get("divergence_type") or "") for row in comparison if row.get("profile") == profile}
    for row in offline_rows:
        row["divergence_from_live"] = divergence_by_symbol.get(str(row.get("symbol") or ""), "")
        row["divergence_type"] = row["divergence_from_live"]
    combined = [*offline_rows, *live]
    output_dir = args.output_dir
    suffix = _profile_suffix(profile)
    trades_path = output_dir / f"full_session_replay{suffix}_{args.date}.csv"
    events_path = output_dir / f"full_session_replay_events{suffix}_{args.date}.csv"
    comparison_path = output_dir / f"full_session_replay_comparison{suffix}_{args.date}.csv"
    trace_path = output_dir / f"full_session_replay_parity_trace{suffix}_{args.date}.csv"
    summary_path = output_dir / f"full_session_replay_summary{suffix}_{args.date}.md"
    config_path = output_dir / f"full_session_replay_config{suffix}_{args.date}.json"
    write_csv(trades_path, combined, TRADE_COLUMNS)
    write_csv(events_path, replay.events, EVENT_COLUMNS)
    write_csv(comparison_path, comparison, COMPARISON_COLUMNS)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(effective_config_dict(config), indent=2, sort_keys=True) + "\n")
    focus = [normalize_symbol(s) for s in str(args.focus_symbols or "").split(",") if normalize_symbol(s)]
    trace_rows = build_parity_trace(
        session_date=args.date,
        symbols=focus,
        top100_path=top100_path,
        history_dir=args.history_dir,
        config=config,
        replay=replay,
        live=live,
    )
    write_csv(trace_path, trace_rows, PARITY_TRACE_COLUMNS)
    write_summary(summary_path, args.date, replay, live, comparison, focus, config)
    print(
        f"FULL_SESSION_REPLAY_DONE date={args.date} profile={profile} offline_entries={len(offline_rows)} live_entries={len(live)} "
        f"output={trades_path} events={events_path} comparison={comparison_path} trace={trace_path} summary={summary_path} config={config_path}",
        flush=True,
    )
    return replay, comparison


def run(args: argparse.Namespace) -> int:
    profiles = ["live", "low_threshold_causal", "legacy_offline"] if args.profile == "all" else [args.profile]
    live = load_live_trades(args.sqlite_path, args.date)
    profile_results: dict[str, ReplayResult] = {}
    all_comparison: list[dict[str, Any]] = []
    for profile in profiles:
        replay, comparison = run_one_profile(args, profile, live)
        profile_results[profile] = replay
        all_comparison.extend(comparison)
    if len(profiles) > 1:
        output_dir = args.output_dir
        profile_comparison_path = output_dir / f"full_session_replay_profile_comparison_{args.date}.csv"
        all_comparison_path = output_dir / f"full_session_replay_comparison_ALL_{args.date}.csv"
        write_csv(profile_comparison_path, profile_comparison_rows(args.date, profile_results), PROFILE_COMPARISON_COLUMNS)
        write_csv(all_comparison_path, all_comparison, COMPARISON_COLUMNS)
        print(f"FULL_SESSION_REPLAY_PROFILE_COMPARISON_DONE date={args.date} profiles={','.join(profiles)} output={profile_comparison_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only full-session causal v67 offline replay.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--profile", choices=["live", "low_threshold_causal", "legacy_offline", "all"], default="live")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--top100", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--first5-threshold", type=float, default=None)
    parser.add_argument("--first15-threshold", type=float, default=None)
    parser.add_argument("--or-max-range-pct", type=float, default=None, help="Override the live OR range threshold. The existing replay engine treats this as the OR range gate value.")
    parser.add_argument("--breakout-mode", choices=["live", "current_price_or_high", "legacy_candle_high"], default=None)
    parser.add_argument("--bar-timestamp-semantics", choices=["bar_start", "bar_end"], default=None)
    parser.add_argument("--window-availability-mode", choices=["live_partial", "finalized_windows"], default=None)
    parser.add_argument("--entry-price-mode", choices=["open", "high", "low", "close"], default=None)
    parser.add_argument("--notional", type=float, default=None)
    parser.add_argument("--position-usd", type=float, default=None, help="Backward-compatible alias for --notional.")
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--commission-model", choices=["per_share", "none"], default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-open-positions", type=int, default=None, help="Backward-compatible alias for --max-positions.")
    parser.add_argument("--hard-stop-pct", type=float, default=None)
    parser.add_argument("--trailing-activation-pct", type=float, default=None)
    parser.add_argument("--trailing-stop-pct", type=float, default=None)
    parser.add_argument("--max-entries-per-cycle", type=int, default=5)
    parser.add_argument("--max-entries-per-minute", type=int, default=5)
    parser.add_argument("--entry-delay-after-open-minutes", type=float, default=5.0)
    parser.add_argument("--min-live-entry-score", type=float, default=0.0)
    parser.add_argument("--focus-symbols", default="FCEL,AXTI,FRMI,FBYD,IREN,NUAI")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
