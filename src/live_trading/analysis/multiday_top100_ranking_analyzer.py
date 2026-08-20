from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.live_trading.analysis.common import load_session_candles, normalize_symbol, pct
from src.live_trading.analysis.full_session_replay_v67 import PreparedSessionCache, _rows, profile_config, replay_session
from src.live_trading.analysis.top100_analysis_common import find_dated_top100, load_top100_source, safe_json, session_dates, write_dataframe
from src.live_trading.ranking.daily_top100_builder import (
    DEFAULT_UNIVERSE,
    DEFAULT_PRIOR_SESSIONS,
    analyze_symbol,
    build_daily_top,
    combined_ineligible_symbols,
    load_universe,
    normalize_history_df,
    parquet_path,
    parse_partition_date,
    prior_session_paths,
    ranking_to_row,
)
from src.live_trading.market_calendar import previous_us_equity_trading_day


VARIANTS = (
    "production_baseline", "short_multiday_3d", "short_multiday_5d", "short_multiday_10d",
    "medium_trend_20d", "medium_trend_60d", "trend_agreement", "continuation",
    "reversal", "hybrid_70_30", "hybrid_80_20", "hybrid_50_50", "stabilized",
)

ELIGIBLE = "ELIGIBLE"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
MISSING_FEATURE_DATE = "MISSING_FEATURE_DATE"
LIQUIDITY_INELIGIBLE = "LIQUIDITY_INELIGIBLE"
DENYLISTED = "DENYLISTED"
OUTCOME_UNAVAILABLE = "OUTCOME_UNAVAILABLE"
RUNNER_THRESHOLDS = (5, 10, 15, 20)


@dataclass(frozen=True)
class BaselineSettings:
    top_n: int = 100
    min_price: float = 5.0
    min_bars: int = 180
    min_volume: float = 100_000.0
    min_dollar_volume: float = 500_000.0
    prior_sessions: int = DEFAULT_PRIOR_SESSIONS


@dataclass
class PerformanceDiagnostics:
    parquet_files_read: int = 0
    parquet_bytes_read: int = 0
    parquet_read_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    replay_session_calls: int = 0
    production_baseline_builds: int = 0
    matrix_rows: int = 0


class SessionHistoryCache:
    """Process-scoped, bounded cache for one requested P2 analysis window."""

    def __init__(
        self,
        history_dir: Path,
        *,
        max_feature_date: date,
        feature_dates: set[date],
        outcome_dates: set[date],
        baseline_prior_sessions: int,
        max_lookback: int = 252,
        max_session_frames: int = 512,
    ):
        self.history_dir = history_dir
        self.max_feature_date = max_feature_date
        self.feature_dates = set(feature_dates)
        self.outcome_dates = set(outcome_dates)
        self.baseline_prior_sessions = max(0, baseline_prior_sessions)
        self.max_lookback = max_lookback
        self.max_session_frames = max(1, max_session_frames)
        self.diagnostics = PerformanceDiagnostics()
        self.daily_histories: dict[str, pd.DataFrame] = {}
        self.daily_metrics: dict[tuple[str, date], dict[str, Any]] = {}
        self._session_frames: OrderedDict[tuple[str, date], pd.DataFrame] = OrderedDict()

    def _record_paths(self, paths: list[Path]) -> None:
        self.diagnostics.parquet_files_read += len(paths)
        self.diagnostics.parquet_bytes_read += sum(path.stat().st_size for path in paths if path.exists())
        self.diagnostics.parquet_read_operations += 1

    def _read_batch(self, paths: list[Path]) -> dict[date, pd.DataFrame]:
        if not paths:
            return {}
        self._record_paths(paths)
        try:
            combined = normalize_history_df(pd.read_parquet(paths))
            observed_dates = set(combined["timestamp"].dt.date) if not combined.empty else set()
            expected_dates = {parse_partition_date(path) for path in paths}
            expected_dates.discard(None)
            if not observed_dates.issubset(expected_dates):
                raise ValueError("history timestamp date does not match requested partitions")
            return {
                current: group.sort_values("timestamp").reset_index(drop=True)
                for current, group in combined.groupby(combined["timestamp"].dt.date, sort=True)
            }
        except Exception:
            frames: dict[date, pd.DataFrame] = {}
            for path in paths:
                current = parse_partition_date(path)
                if current is None:
                    continue
                self._record_paths([path])
                frame = load_session_candles(self.history_dir, path.parts[-4].split("=", 1)[1], current)
                if not frame.empty:
                    frames[current] = frame
            return frames

    def prepare_symbol(self, symbol: str) -> tuple[pd.DataFrame, dict[date, pd.DataFrame]]:
        symbol = normalize_symbol(symbol)
        if symbol in self.daily_histories:
            self.diagnostics.cache_hits += 1
            return self.daily_histories[symbol], {}
        self.diagnostics.cache_misses += 1
        history_dates = _available_dates(self.history_dir, symbol, self.max_feature_date)[-self.max_lookback:]
        # Session-D outcomes are loaded only after the D-1 ranking is frozen.
        required_dates = set(history_dates) | self.feature_dates
        for current in self.feature_dates:
            required_dates.update(
                parse_partition_date(path)
                for path in prior_session_paths(
                    self.history_dir, symbol, current, "RTH", limit=self.baseline_prior_sessions
                )
            )
        paths = []
        for current in sorted(value for value in required_dates if value is not None):
            path = parquet_path(self.history_dir, symbol, current, "RTH")
            if path.exists():
                paths.append(path)
        frames = self._read_batch(paths)
        sessions: list[dict[str, Any]] = []
        for current, frame in frames.items():
            metrics = _daily_metrics(frame)
            if not metrics:
                continue
            self.daily_metrics[(symbol, current)] = metrics
            if current in history_dates:
                sessions.append({"date": current, **metrics})
        daily = pd.DataFrame(sessions).sort_values("date").reset_index(drop=True) if sessions else pd.DataFrame()
        self.daily_histories[symbol] = daily
        return daily, frames

    def _remember_session(self, key: tuple[str, date], frame: pd.DataFrame) -> None:
        self._session_frames.pop(key, None)
        self._session_frames[key] = frame
        while len(self._session_frames) > self.max_session_frames:
            self._session_frames.popitem(last=False)

    def get_daily_history(self, symbol: str) -> pd.DataFrame:
        symbol = normalize_symbol(symbol)
        if symbol in self.daily_histories:
            self.diagnostics.cache_hits += 1
            return self.daily_histories[symbol]
        return self.prepare_symbol(symbol)[0]

    def get_daily_metrics(self, symbol: str, session_date: date) -> dict[str, Any]:
        key = (normalize_symbol(symbol), session_date)
        if key in self.daily_metrics:
            self.diagnostics.cache_hits += 1
            return self.daily_metrics[key]
        frame = self.get_session(symbol, session_date)
        metrics = _daily_metrics(frame)
        if metrics:
            self.daily_metrics[key] = metrics
        return metrics

    def get_session(self, symbol: str, session_date: date) -> pd.DataFrame:
        key = (normalize_symbol(symbol), session_date)
        if key in self._session_frames:
            self.diagnostics.cache_hits += 1
            frame = self._session_frames.pop(key)
            self._session_frames[key] = frame
            return frame
        self.diagnostics.cache_misses += 1
        path = parquet_path(self.history_dir, key[0], session_date, "RTH")
        if path.exists():
            self._record_paths([path])
        frame = load_session_candles(self.history_dir, key[0], session_date)
        self._remember_session(key, frame)
        return frame

    def prior_closes(self, symbol: str, feature_date: date, limit: int) -> list[float]:
        closes: list[float] = []
        for path in prior_session_paths(self.history_dir, symbol, feature_date, "RTH", limit=max(0, limit)):
            current = parse_partition_date(path)
            metrics = self.daily_metrics.get((normalize_symbol(symbol), current), {}) if current is not None else {}
            close = metrics.get("close")
            if close is not None and close > 0:
                closes.append(close)
        return closes


def _daily_metrics(candles: pd.DataFrame) -> dict[str, Any]:
    if candles.empty:
        return {}
    rows = candles.sort_values("timestamp")
    open_price = float(rows["open"].iloc[0])
    close = float(rows["close"].iloc[-1])
    high = float(rows["high"].max())
    low = float(rows["low"].min())
    volume = float(pd.to_numeric(rows["volume"], errors="coerce").fillna(0).sum())
    dollar_volume = float((pd.to_numeric(rows["close"], errors="coerce") * pd.to_numeric(rows["volume"], errors="coerce").fillna(0)).sum())
    ranges = ((pd.to_numeric(rows["high"]) - pd.to_numeric(rows["low"])) / pd.to_numeric(rows["close"]).replace(0, np.nan) * 100)
    return {
        "open": open_price, "close": close, "high": high, "low": low,
        "return_1d": pct(close, open_price), "high_open_pct": pct(high, open_price),
        "close_open_pct": pct(close, open_price), "close_location": (close - low) / (high - low) if high > low else None,
        "intraday_range_pct": pct(high, low), "dollar_volume": dollar_volume, "volume": volume,
        "volatility": float(pd.to_numeric(rows["close"]).pct_change().std() * math.sqrt(len(rows)) * 100),
        "atr_like_range": float(ranges.mean()),
    }


def _available_dates(history_dir: Path, symbol: str, end: date) -> list[date]:
    root = history_dir / "session_type=RTH" / f"symbol={symbol}"
    found: list[date] = []
    for path in root.glob("year=*/month=*/day=*.parquet"):
        try:
            values = {piece.split("=", 1)[0]: piece.split("=", 1)[1] for piece in path.parts if "=" in piece}
            current = date(int(values["year"]), int(values["month"]), int(values["day"]))
        except Exception:
            continue
        if current <= end:
            found.append(current)
    return sorted(set(found))


def load_symbol_daily_history(history_dir: Path, symbol: str, end_date: date, *, max_lookback: int = 252) -> pd.DataFrame:
    dates = _available_dates(history_dir, symbol, end_date)[-max_lookback:]
    sessions: list[dict[str, Any]] = []
    for current in dates:
        candles = load_session_candles(history_dir, symbol, current)
        metrics = _daily_metrics(candles)
        if metrics:
            metrics["date"] = current
            sessions.append(metrics)
    return pd.DataFrame(sessions).sort_values("date").reset_index(drop=True) if sessions else pd.DataFrame()


def build_symbol_features(
    history_dir: Path,
    symbol: str,
    feature_date: date,
    *,
    max_lookback: int = 252,
    daily_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    frame = daily_history.copy() if daily_history is not None else load_symbol_daily_history(history_dir, symbol, feature_date, max_lookback=max_lookback)
    if not frame.empty:
        frame = frame[pd.to_datetime(frame["date"]).dt.date <= feature_date].tail(max_lookback).reset_index(drop=True)
    if frame.empty:
        return {"symbol": symbol, "feature_date": feature_date.isoformat(), "history_sessions": 0}
    latest = frame.iloc[-1]
    result = {"symbol": symbol, "feature_date": feature_date.isoformat(), "history_sessions": len(frame)} | {key: latest.get(key) for key in latest.index if key != "date"}
    closes = pd.to_numeric(frame["close"], errors="coerce")
    highs = pd.to_numeric(frame["high"], errors="coerce")
    returns = closes.pct_change() * 100
    previous_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
    result["gap_from_previous_close_pct"] = pct(float(latest["open"]), previous_close) if previous_close else None
    for window in (2, 3, 5, 10, 20, 60, 120, 252):
        sample = frame.tail(window)
        result[f"feature_available_{window}d"] = int(len(sample) >= window)
        if len(sample) >= min(2, window):
            result[f"return_{window}d"] = pct(float(sample["close"].iloc[-1]), float(sample["close"].iloc[0]))
            result[f"positive_days_count_{window}d"] = int((pd.to_numeric(sample["close"]).pct_change() > 0).sum())
            result[f"distance_from_{window}d_high_pct"] = pct(float(sample["close"].iloc[-1]), float(pd.to_numeric(sample["high"]).max()))
            prior_highs = pd.to_numeric(sample["high"].iloc[:-1], errors="coerce")
            result[f"high_breakout_{window}d"] = int(
                not prior_highs.empty and float(sample["close"].iloc[-1]) > float(prior_highs.max())
            )
            result[f"avg_dollar_volume_{window}d"] = float(pd.to_numeric(sample["dollar_volume"], errors="coerce").mean())
            result[f"volatility_{window}d"] = float(pd.to_numeric(sample["return_1d"], errors="coerce").std())
            x = np.arange(len(sample), dtype=float)
            y = pd.to_numeric(sample["close"], errors="coerce").to_numpy(dtype=float)
            result[f"trend_slope_{window}d"] = float(np.polyfit(x, y, 1)[0] / np.nanmean(y) * 100) if len(sample) > 1 and np.nanmean(y) else None
            result[f"above_rolling_average_{window}d"] = int(float(sample["close"].iloc[-1]) >= float(pd.to_numeric(sample["close"]).mean()))
    result["momentum_consistency_5d"] = float((returns.tail(5) > 0).mean()) if len(returns.dropna()) else None
    result["volume_acceleration"] = float(latest["volume"] / pd.to_numeric(frame.tail(10)["volume"]).mean()) if pd.to_numeric(frame.tail(10)["volume"]).mean() else None
    result["drawdown_from_recent_high_pct"] = result.get("distance_from_20d_high_pct")
    result["trend_agreement_short_medium_long"] = int(all((result.get(f"trend_slope_{window}d") or -1) > 0 for window in (5, 20, 60)))
    return result


def reproduce_baseline(
    ranking_date: date,
    *,
    universe_path: Path,
    history_dir: Path,
    settings: BaselineSettings,
) -> pd.DataFrame:
    rows, _stats = build_daily_top(
        ranking_date=ranking_date,
        universe_path=universe_path,
        history_dir=history_dir,
        top_n=settings.top_n,
        session_type="RTH",
        min_price=settings.min_price,
        min_bars=settings.min_bars,
        min_volume=settings.min_volume,
        min_dollar_volume=settings.min_dollar_volume,
        prior_sessions=settings.prior_sessions,
        max_missing_log=0,
        max_reject_log=0,
        max_partial_history_log=0,
    )
    return pd.DataFrame(rows)


def reproduce_baselines_from_cache(
    ranking_dates: list[date],
    *,
    symbols: list[str],
    cache: SessionHistoryCache,
    settings: BaselineSettings,
) -> dict[date, pd.DataFrame]:
    """Reproduce production baselines while sharing history reads across dates."""
    requested = list(dict.fromkeys(ranking_dates))
    ranked_by_date: dict[date, list[Any]] = {current: [] for current in requested}
    ineligible = combined_ineligible_symbols()
    for symbol in symbols:
        _daily, frames = cache.prepare_symbol(symbol)
        if symbol in ineligible:
            continue
        for current in requested:
            frame = frames.get(current)
            if frame is None or frame.empty:
                continue
            item, _reason = analyze_symbol(
                symbol,
                frame,
                cache.prior_closes(symbol, current, settings.prior_sessions),
                min_price=settings.min_price,
                min_bars=settings.min_bars,
                min_volume=settings.min_volume,
                min_dollar_volume=settings.min_dollar_volume,
            )
            if item is not None:
                ranked_by_date[current].append(item)
    result: dict[date, pd.DataFrame] = {}
    for current in requested:
        ranked = ranked_by_date[current]
        ranked.sort(key=lambda item: (item.score, item.dollar_volume, item.intraday_high_pct), reverse=True)
        result[current] = pd.DataFrame([
            ranking_to_row(rank, item)
            for rank, item in enumerate(ranked[: max(0, settings.top_n)], 1)
        ])
    cache.diagnostics.production_baseline_builds += len(requested)
    return result


def build_production_populations_from_cache(
    ranking_dates: list[date],
    *,
    symbols: list[str],
    cache: SessionHistoryCache,
    settings: BaselineSettings,
) -> dict[date, pd.DataFrame]:
    """Build one row per symbol, ranking every production-eligible symbol globally."""
    requested = list(dict.fromkeys(ranking_dates))
    rows_by_date: dict[date, list[dict[str, Any]]] = {current: [] for current in requested}
    ineligible = combined_ineligible_symbols()
    for symbol in symbols:
        _daily, frames = cache.prepare_symbol(symbol)
        for current in requested:
            if symbol in ineligible:
                rows_by_date[current].append({
                    "symbol": symbol,
                    "ranking_eligibility_status": DENYLISTED,
                    "eligibility_reason": str(ineligible[symbol].get("reason") or "ineligible"),
                })
                continue
            frame = frames.get(current)
            if frame is None or frame.empty:
                rows_by_date[current].append({
                    "symbol": symbol,
                    "ranking_eligibility_status": MISSING_FEATURE_DATE,
                    "eligibility_reason": "missing_feature_date",
                })
                continue
            item, reason = analyze_symbol(
                symbol,
                frame,
                cache.prior_closes(symbol, current, settings.prior_sessions),
                min_price=settings.min_price,
                min_bars=settings.min_bars,
                min_volume=settings.min_volume,
                min_dollar_volume=settings.min_dollar_volume,
            )
            if item is not None:
                row = ranking_to_row(0, item)
                row.update({
                    "ranking_eligibility_status": ELIGIBLE,
                    "eligibility_reason": "",
                })
                rows_by_date[current].append(row)
            else:
                status = (
                    INSUFFICIENT_HISTORY
                    if str(reason or "").startswith("too_few_bars")
                    else LIQUIDITY_INELIGIBLE
                )
                rows_by_date[current].append({
                    "symbol": symbol,
                    "ranking_eligibility_status": status,
                    "eligibility_reason": reason or "rejected",
                })

    result: dict[date, pd.DataFrame] = {}
    for current in requested:
        population = pd.DataFrame(rows_by_date[current])
        if population.empty:
            result[current] = population
            continue
        eligible = population[population["ranking_eligibility_status"].eq(ELIGIBLE)].copy()
        eligible = eligible.sort_values(
            ["score", "dollar_volume", "intraday_high_pct", "symbol"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        global_ranks = pd.Series(
            range(1, len(eligible) + 1), index=eligible.index, dtype="Int64"
        )
        population["production_rank_global"] = pd.Series(
            pd.NA, index=population.index, dtype="Int64"
        )
        population.loc[global_ranks.index, "production_rank_global"] = global_ranks
        result[current] = population.sort_values(
            ["production_rank_global", "symbol"],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    cache.diagnostics.production_baseline_builds += len(requested)
    return result


def compare_baseline(reproduced: pd.DataFrame, saved: pd.DataFrame) -> dict[str, Any]:
    reproduced_symbols = [normalize_symbol(value) for value in reproduced.get("symbol", pd.Series(dtype=str))]
    saved_symbols = [normalize_symbol(value) for value in saved.get("symbol", pd.Series(dtype=str))]
    exact = reproduced_symbols == saved_symbols
    return {
        "baseline_match": exact,
        "reproduced_count": len(reproduced_symbols), "saved_count": len(saved_symbols),
        "missing_from_reproduced": sorted(set(saved_symbols) - set(reproduced_symbols)),
        "extra_in_reproduced": sorted(set(reproduced_symbols) - set(saved_symbols)),
        "rank_mismatch_count": sum(1 for index in range(min(len(reproduced_symbols), len(saved_symbols))) if reproduced_symbols[index] != saved_symbols[index]),
    }


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std()
    return (values - values.mean()) / std if std and not pd.isna(std) else pd.Series(0.0, index=values.index)


def _series(frame: pd.DataFrame, name: str, default: Any = None) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(default, index=frame.index)


def score_variants(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    baseline = pd.to_numeric(_series(out, "production_score"), errors="coerce").fillna(0)
    zbase = _zscore(baseline)
    z3, z5, z10 = (_zscore(out.get(f"return_{window}d", pd.Series(index=out.index, dtype=float))) for window in (3, 5, 10))
    z20, z60 = (_zscore(out.get(f"return_{window}d", pd.Series(index=out.index, dtype=float))) for window in (20, 60))
    continuation = (z3 + z5 + z10 + z20) / 4 + _zscore(out.get("volume_acceleration", pd.Series(index=out.index, dtype=float))) * 0.2
    reversal = -_zscore(out.get("drawdown_from_recent_high_pct", pd.Series(index=out.index, dtype=float))) - z5 * 0.25
    scores = {
        "production_baseline": zbase,
        "short_multiday_3d": zbase * 0.7 + z3 * 0.3,
        "short_multiday_5d": zbase * 0.7 + z5 * 0.3,
        "short_multiday_10d": zbase * 0.7 + z10 * 0.3,
        "medium_trend_20d": zbase * 0.7 + z20 * 0.3,
        "medium_trend_60d": zbase * 0.7 + z60 * 0.3,
        "trend_agreement": zbase + pd.to_numeric(_series(out, "trend_agreement_short_medium_long", 0), errors="coerce").fillna(0) * 0.5,
        "continuation": zbase * 0.6 + continuation * 0.4,
        "reversal": zbase * 0.5 + reversal * 0.5,
        "hybrid_70_30": zbase * 0.7 + continuation * 0.3,
        "hybrid_80_20": zbase * 0.8 + continuation * 0.2,
        "hybrid_50_50": zbase * 0.5 + continuation * 0.5,
        "stabilized": zbase + _zscore(out.get("consecutive_days_in_top100", pd.Series(index=out.index, dtype=float))) * 0.2,
    }
    for name, values in scores.items():
        out[f"score_{name}"] = values
        out[f"rank_{name}"] = values.rank(method="first", ascending=False).astype("Int64")
    return out


def attach_outcomes(
    matrix: pd.DataFrame,
    history_dir: Path,
    session_date: str,
    *,
    history_cache: SessionHistoryCache | None = None,
) -> pd.DataFrame:
    out = matrix.copy()
    metrics: list[dict[str, Any]] = []
    for symbol in out["symbol"]:
        daily = (
            history_cache.get_daily_metrics(symbol, date.fromisoformat(session_date))
            if history_cache is not None
            else _daily_metrics(load_session_candles(history_dir, symbol, session_date))
        )
        high_pct = daily.get("high_open_pct")
        available = high_pct is not None and not pd.isna(high_pct)
        row = {
            "outcome_open_to_high_pct": high_pct if available else None,
            "outcome_return_pct": daily.get("close_open_pct") if available else None,
            "outcome_dollar_volume": daily.get("dollar_volume") if available else None,
            "outcome_available": int(available),
        }
        for threshold in RUNNER_THRESHOLDS:
            row[f"outcome_runner_{threshold}"] = (
                int(float(high_pct) > threshold) if available else pd.NA
            )
        metrics.append(row)
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def _variant_symbols(matrix: pd.DataFrame, variant: str, top_n: int) -> list[str]:
    eligible = matrix[pd.to_numeric(matrix.get("production_rank"), errors="coerce").notna()]
    return eligible.sort_values([f"rank_{variant}", "symbol"])["symbol"].head(top_n).tolist()


def _ndcg(selected: pd.DataFrame, rank_column: str) -> float | None:
    ordered = selected.sort_values([rank_column, "symbol"])
    relevance = pd.to_numeric(ordered["outcome_open_to_high_pct"], errors="coerce").clip(lower=0).fillna(0).to_numpy()
    if not len(relevance) or relevance.max() <= 0:
        return None
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    dcg = float(np.sum((2 ** relevance - 1) / discounts))
    ideal = np.sort(relevance)[::-1]
    idcg = float(np.sum((2 ** ideal - 1) / discounts))
    return dcg / idcg if idcg else None


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(pair) < 2:
        return None
    value = pair["left"].rank(method="average").corr(pair["right"].rank(method="average"))
    return None if pd.isna(value) else float(value)


def comparison_metrics(matrix: pd.DataFrame, *, session_date: str, top_n: int, baseline_comparable: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    opportunity: list[dict[str, Any]] = []
    baseline = set(_variant_symbols(matrix, "production_baseline", top_n))
    for variant in VARIANTS:
        selected = matrix[matrix["symbol"].isin(_variant_symbols(matrix, variant, top_n))]
        selected_set = set(selected["symbol"])
        daily.append({
            "trading_session_date": session_date, "variant": variant, "symbols": len(selected),
            "baseline_comparable": int(baseline_comparable), "precision_runner_5": pd.to_numeric(selected["outcome_runner_5"]).mean(),
            "mean_open_to_high_pct": pd.to_numeric(selected["outcome_open_to_high_pct"], errors="coerce").mean(),
            "median_open_to_high_pct": pd.to_numeric(selected["outcome_open_to_high_pct"], errors="coerce").median(),
            "mean_close_return_pct": pd.to_numeric(selected["outcome_return_pct"], errors="coerce").mean(),
            "median_close_return_pct": pd.to_numeric(selected["outcome_return_pct"], errors="coerce").median(),
            "rank_correlation_future_move": _spearman(selected[f"rank_{variant}"], selected["outcome_open_to_high_pct"]),
            "ndcg_future_move": _ndcg(selected, f"rank_{variant}"),
            "top_decile_runner_5_count": int(pd.to_numeric(selected.sort_values(f"rank_{variant}").head(max(1, top_n // 10))["outcome_runner_5"], errors="coerce").sum()),
            "rank_bucket_distribution_json": safe_json([
                {
                    "rank_start": start, "rank_end": min(start + 9, top_n),
                    "count": len(bucket),
                    "mean_future_max_move": pd.to_numeric(bucket["outcome_open_to_high_pct"], errors="coerce").mean(),
                    "runner_5_count": int(pd.to_numeric(bucket["outcome_runner_5"], errors="coerce").sum()),
                }
                for start in range(1, top_n + 1, 10)
                for bucket in [selected[(pd.to_numeric(selected[f"rank_{variant}"], errors="coerce") >= start) & (pd.to_numeric(selected[f"rank_{variant}"], errors="coerce") <= start + 9)]]
            ]),
            "shared_with_baseline": len(selected_set & baseline), "unique_vs_baseline": len(selected_set - baseline),
            "interpretation": "FACT" if variant == "production_baseline" else ("INFERENCE" if baseline_comparable else "NOT COMPARABLE: BASELINE MISMATCH"),
        })
        for threshold in (5, 10, 15, 20):
            universe_count = int(pd.to_numeric(matrix[f"outcome_runner_{threshold}"], errors="coerce").sum())
            selected_count = int(pd.to_numeric(selected[f"outcome_runner_{threshold}"], errors="coerce").sum())
            threshold_rows.append({"trading_session_date": session_date, "variant": variant, "threshold_pct": threshold, "universe_runner_count": universe_count, "top100_runner_count": selected_count, "selected_runner_count": selected_count, "coverage_pct": selected_count / universe_count * 100 if universe_count else None, "precision_count": selected_count, "precision_pct": selected_count / len(selected) * 100 if len(selected) else None})
        missed = matrix[(~matrix["symbol"].isin(selected_set)) & pd.to_numeric(matrix["outcome_runner_5"], errors="coerce").eq(1)].sort_values("outcome_open_to_high_pct", ascending=False)
        for row in missed.head(20).to_dict("records"):
            opportunity.append({"trading_session_date": session_date, "variant": variant, "symbol": row["symbol"], "open_to_high_pct": row.get("outcome_open_to_high_pct"), "variant_rank": row.get(f"rank_{variant}"), "production_rank": row.get("production_rank"), "production_score": row.get("production_score"), "opportunity_cost_type": "runner_outside_top_n"})
    turnover = []
    for variant in VARIANTS:
        selected = set(_variant_symbols(matrix, variant, top_n))
        turnover.append({"trading_session_date": session_date, "variant": variant, "membership_turnover_vs_baseline": len(selected ^ baseline) / max(1, top_n), "rank_correlation_vs_baseline": _spearman(matrix[f"rank_{variant}"], matrix["rank_production_baseline"])})
    return pd.DataFrame(daily), pd.DataFrame(threshold_rows), pd.DataFrame(turnover), pd.DataFrame(opportunity)


def portfolio_replays(
    matrix: pd.DataFrame,
    *,
    session_date: str,
    history_dir: Path,
    output_dir: Path,
    top_n: int,
    baseline_comparable: bool,
    history_cache: SessionHistoryCache | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    symbols_by_variant = {variant: _variant_symbols(matrix, variant, top_n) for variant in VARIANTS}
    config = profile_config("live")
    prepared_rows_by_symbol: dict[str, pd.DataFrame] | None = None
    if history_cache is not None:
        trading_date = date.fromisoformat(session_date)
        prepared_rows_by_symbol = {
            symbol: _rows(history_cache.get_session(symbol, trading_date), config.bar_timestamp_semantics)
            for symbol in sorted({symbol for values in symbols_by_variant.values() for symbol in values})
        }
    prepared_sessions_by_symbol = None
    if prepared_rows_by_symbol is not None:
        prepared_cache = PreparedSessionCache(
            max_entries=max(1, len(prepared_rows_by_symbol)),
            max_bytes=256 * 1024 * 1024,
        )
        prepared_sessions_by_symbol = {
            symbol: prepared_cache.get_or_build(symbol, session_date, frame, config)
            for symbol, frame in prepared_rows_by_symbol.items()
            if not frame.empty
        }
    with tempfile.TemporaryDirectory(prefix="top100-ranking-") as temp:
        for variant in VARIANTS:
            symbols = symbols_by_variant[variant]
            path = Path(temp) / f"{variant}.csv"
            pd.DataFrame({"rank": range(1, len(symbols) + 1), "symbol": symbols, "score": range(len(symbols), 0, -1)}).to_csv(path, index=False)
            replay = replay_session(
                session_date=session_date,
                top100_path=path,
                history_dir=history_dir,
                config=config,
                prepared_rows_by_symbol=prepared_rows_by_symbol,
                prepared_sessions_by_symbol=prepared_sessions_by_symbol,
            )
            if history_cache is not None:
                history_cache.diagnostics.replay_session_calls += 1
            pnl = pd.to_numeric(pd.Series([row.get("net_pnl") for row in replay.trades]), errors="coerce").fillna(0)
            equity = pd.Series([value for _timestamp, value in replay.equity_curve], dtype=float)
            drawdown = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
            rows.append({
                "trading_session_date": session_date, "variant": variant, "entries": len(replay.trades), "net_pnl": pnl.sum(),
                "winners": int((pnl > 0).sum()), "losers": int((pnl <= 0).sum()), "max_concurrent_positions": replay.max_concurrent_positions,
                "max_drawdown": drawdown, "signals": int(sum(1 for event in replay.events if event.get("event_type") == "SIGNAL")),
                "skipped_candidates": safe_json(replay.skipped), "baseline_comparable": int(baseline_comparable), "replay_source": "causal_full_session_replay_v67",
            })
    return pd.DataFrame(rows)


def analyze_range(
    dates: list[str], *, history_dir: Path, top100_dir: Path, universe_path: Path, output_dir: Path, settings: BaselineSettings,
) -> dict[str, Path]:
    analysis_started = time.perf_counter()
    matrices: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []
    opportunity_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    quality_sessions: list[dict[str, Any]] = []
    symbols = load_universe(universe_path)
    max_feature_date = previous_us_equity_trading_day(date.fromisoformat(dates[-1]))
    feature_dates = [previous_us_equity_trading_day(date.fromisoformat(value)) for value in dates]
    history_cache = SessionHistoryCache(
        history_dir,
        max_feature_date=max_feature_date,
        feature_dates=set(feature_dates),
        outcome_dates={date.fromisoformat(value) for value in dates},
        baseline_prior_sessions=settings.prior_sessions,
        max_session_frames=max(512, settings.top_n * 2),
    )
    reproduced_by_date: dict[date, pd.DataFrame] | None = None
    phase_timings: dict[str, dict[str, float]] = {}
    first_feature_date = previous_us_equity_trading_day(date.fromisoformat(dates[0]))
    prior_feature_date = previous_us_equity_trading_day(first_feature_date)
    prior_top100, _prior_path, _prior_source = load_top100_source(
        top100_dir, first_feature_date.isoformat(), prior_feature_date.isoformat()
    )
    prior_rows = prior_top100.head(settings.top_n).to_dict("records") if not prior_top100.empty else []
    previous_membership: dict[str, int] = {
        normalize_symbol(row.get("symbol")): 1 for row in prior_rows
    }
    previous_rank: dict[str, int] = {
        normalize_symbol(row.get("symbol")): int(row.get("top100_rank") or row.get("rank") or index + 1)
        for index, row in enumerate(prior_rows)
    }
    previous_score: dict[str, float] = {
        normalize_symbol(row.get("symbol")): float(row.get("top100_score") or row.get("score") or 0.0)
        for row in prior_rows
    }
    previous_variant_sets: dict[str, set[str]] = {}
    variant_streaks: dict[str, dict[str, int]] = {variant: {} for variant in VARIANTS}
    for session_date in dates:
        session_started = time.perf_counter()
        print(f"P2_SESSION_START date={session_date}", flush=True)
        trading_date = date.fromisoformat(session_date)
        feature_date = previous_us_equity_trading_day(trading_date)

        baseline_started = time.perf_counter()
        print(f"BASELINE_START date={session_date} feature_date={feature_date.isoformat()}", flush=True)
        if reproduced_by_date is None:
            reproduced_by_date = reproduce_baselines_from_cache(
                feature_dates,
                symbols=symbols,
                cache=history_cache,
                settings=settings,
            )
        saved, saved_path, saved_source = load_top100_source(top100_dir, session_date, feature_date.isoformat())
        reproduced = reproduced_by_date[feature_date]
        comparison = compare_baseline(reproduced, saved)
        baseline_elapsed = time.perf_counter() - baseline_started
        print(
            f"BASELINE_DONE date={session_date} elapsed_seconds={baseline_elapsed:.3f} "
            f"rows={len(reproduced)} reused_existing=0 reuse_reason=artifact_identity_unverifiable "
            f"production_baseline_builds={history_cache.diagnostics.production_baseline_builds}",
            flush=True,
        )

        matrix_started = time.perf_counter()
        print(f"FEATURE_MATRIX_START date={session_date} symbols={len(symbols)}", flush=True)
        features = pd.DataFrame([
            build_symbol_features(history_dir, symbol, feature_date, daily_history=history_cache.get_daily_history(symbol))
            for symbol in symbols
        ])
        production_scores = reproduced[[column for column in ["symbol", "score", "rank", "components_json"] if column in reproduced.columns]].copy()
        production_scores = production_scores.rename(columns={"score": "production_score", "rank": "production_rank", "components_json": "production_components_json"})
        matrix = features.merge(production_scores, on="symbol", how="left")
        matrix["ranking_source_date"] = feature_date.isoformat()
        matrix["trading_session_date"] = session_date
        matrix["feature_max_date"] = feature_date.isoformat()
        matrix["leakage_check_passed"] = (pd.to_datetime(matrix["feature_max_date"]) < pd.Timestamp(trading_date)).astype(int)
        matrix["validation_segment"] = "baseline_only" if len(dates) < 20 else ("development" if dates.index(session_date) < len(dates) // 2 else "validation")
        matrix["previous_top100_membership"] = matrix["symbol"].map(lambda value: int(value in previous_membership))
        matrix["consecutive_days_in_top100"] = matrix["symbol"].map(lambda value: previous_membership.get(value, 0))
        matrix["previous_rank"] = matrix["symbol"].map(previous_rank)
        matrix["previous_score"] = matrix["symbol"].map(previous_score)
        matrix["rank_delta"] = pd.to_numeric(matrix["production_rank"], errors="coerce") - pd.to_numeric(matrix["previous_rank"], errors="coerce")
        matrix["score_delta"] = pd.to_numeric(matrix["production_score"], errors="coerce") - pd.to_numeric(matrix["previous_score"], errors="coerce")
        matrix["recently_added_symbol"] = ((matrix["production_rank"].notna()) & matrix["previous_top100_membership"].eq(0)).astype(int)
        matrix = attach_outcomes(score_variants(matrix), history_dir, session_date, history_cache=history_cache)
        history_cache.diagnostics.matrix_rows += len(matrix)
        matrices.append(matrix)
        matrix_elapsed = time.perf_counter() - matrix_started
        print(
            f"FEATURE_MATRIX_DONE date={session_date} elapsed_seconds={matrix_elapsed:.3f} "
            f"matrix_rows={len(matrix)} cache_hits={history_cache.diagnostics.cache_hits} "
            f"cache_misses={history_cache.diagnostics.cache_misses}",
            flush=True,
        )

        comparison_started = time.perf_counter()
        print(f"VARIANT_COMPARISON_START date={session_date} variants={len(VARIANTS)}", flush=True)
        daily, thresholds, _turnover_vs_baseline, opportunity = comparison_metrics(matrix, session_date=session_date, top_n=settings.top_n, baseline_comparable=bool(comparison["baseline_match"]))
        daily["validation_segment"] = matrix["validation_segment"].iloc[0]
        daily_frames.append(daily); threshold_frames.append(thresholds); opportunity_frames.append(opportunity)
        daily_turnover: list[dict[str, Any]] = []
        for variant in VARIANTS:
            current_set = set(_variant_symbols(matrix, variant, settings.top_n))
            previous_set = previous_variant_sets.get(variant, set())
            retained = current_set & previous_set
            streaks = variant_streaks[variant]
            variant_streaks[variant] = {symbol: streaks.get(symbol, 0) + 1 for symbol in current_set}
            daily_turnover.append({
                "trading_session_date": session_date, "variant": variant,
                "symbols_retained": len(retained), "symbols_added": len(current_set - previous_set), "symbols_removed": len(previous_set - current_set),
                "daily_turnover_pct": len(current_set ^ previous_set) / max(1, 2 * settings.top_n) * 100 if previous_set else None,
                "average_days_retained": float(np.mean(list(variant_streaks[variant].values()))) if current_set else 0.0,
            })
            previous_variant_sets[variant] = current_set
        turnover_frames.append(pd.DataFrame(daily_turnover))
        comparison_elapsed = time.perf_counter() - comparison_started
        print(
            f"VARIANT_COMPARISON_DONE date={session_date} elapsed_seconds={comparison_elapsed:.3f} "
            f"variants={len(VARIANTS)}",
            flush=True,
        )

        replay_started = time.perf_counter()
        print(f"PORTFOLIO_REPLAY_START date={session_date} variants={len(VARIANTS)}", flush=True)
        portfolio_frames.append(portfolio_replays(
            matrix,
            session_date=session_date,
            history_dir=history_dir,
            output_dir=output_dir,
            top_n=settings.top_n,
            baseline_comparable=bool(comparison["baseline_match"]),
            history_cache=history_cache,
        ))
        replay_elapsed = time.perf_counter() - replay_started
        print(
            f"PORTFOLIO_REPLAY_DONE date={session_date} elapsed_seconds={replay_elapsed:.3f} "
            f"replay_session_calls={history_cache.diagnostics.replay_session_calls}",
            flush=True,
        )
        phase_timings[session_date] = {
            "baseline_seconds": baseline_elapsed,
            "feature_matrix_seconds": matrix_elapsed,
            "variant_comparison_seconds": comparison_elapsed,
            "portfolio_replay_seconds": replay_elapsed,
        }
        quality_sessions.append({
            "trading_session_date": session_date,
            "feature_date": feature_date.isoformat(),
            "ranking_source_date": saved_source,
            "saved_top100_path": str(saved_path or ""),
            **comparison,
            "leakage_rows": int(matrix["leakage_check_passed"].eq(0).sum()),
            "symbols_with_features": int(matrix["history_sessions"].gt(0).sum()),
            "baseline_artifact_reuse": False,
            "baseline_artifact_reuse_reason": "artifact_identity_unverifiable",
        })
        selected = reproduced.head(settings.top_n)
        selected_symbols = set(selected.get("symbol", pd.Series(dtype=str)).map(normalize_symbol))
        previous_membership = {symbol: previous_membership.get(symbol, 0) + 1 for symbol in selected_symbols}
        previous_rank = {normalize_symbol(row.get("symbol")): int(row.get("rank")) for row in selected.to_dict("records")}
        previous_score = {normalize_symbol(row.get("symbol")): float(row.get("score")) for row in selected.to_dict("records")}
        session_elapsed = time.perf_counter() - session_started
        phase_timings[session_date]["total_seconds"] = session_elapsed
        print(
            f"P2_SESSION_DONE date={session_date} elapsed_seconds={session_elapsed:.3f} "
            f"parquet_files_read={history_cache.diagnostics.parquet_files_read} "
            f"parquet_bytes_read={history_cache.diagnostics.parquet_bytes_read} "
            f"parquet_read_operations={history_cache.diagnostics.parquet_read_operations} "
            f"cache_hits={history_cache.diagnostics.cache_hits} cache_misses={history_cache.diagnostics.cache_misses} "
            f"matrix_rows={history_cache.diagnostics.matrix_rows} "
            f"replay_session_calls={history_cache.diagnostics.replay_session_calls} "
            f"production_baseline_builds={history_cache.diagnostics.production_baseline_builds}",
            flush=True,
        )
    start, end = dates[0], dates[-1]
    suffix = start if start == end else f"{start}_to_{end}"
    paths = {
        "feature_matrix": output_dir / f"multiday_top100_feature_matrix_{suffix}.parquet",
        "daily_comparison": output_dir / f"multiday_top100_daily_comparison_{suffix}.csv",
        "threshold_coverage": output_dir / f"multiday_top100_threshold_coverage_{suffix}.csv",
        "turnover": output_dir / f"multiday_top100_turnover_{suffix}.csv",
        "opportunity_cost": output_dir / f"multiday_top100_opportunity_cost_{suffix}.csv",
        "portfolio_replay": output_dir / f"multiday_top100_portfolio_replay_{suffix}.csv",
        "summary": output_dir / f"multiday_top100_summary_{suffix}.md",
        "data_quality": output_dir / f"multiday_top100_data_quality_{suffix}.json",
    }
    frames = {
        "feature_matrix": pd.concat(matrices, ignore_index=True) if matrices else pd.DataFrame(),
        "daily_comparison": pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        "threshold_coverage": pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame(),
        "turnover": pd.concat(turnover_frames, ignore_index=True) if turnover_frames else pd.DataFrame(),
        "opportunity_cost": pd.concat(opportunity_frames, ignore_index=True) if opportunity_frames else pd.DataFrame(),
        "portfolio_replay": pd.concat(portfolio_frames, ignore_index=True) if portfolio_frames else pd.DataFrame(),
    }
    for name, frame in frames.items():
        write_dataframe(frame, paths[name])
    quality = {
        "sessions": quality_sessions,
        "all_baselines_match": all(item["baseline_match"] for item in quality_sessions),
        "all_leakage_checks_pass": all(item["leakage_rows"] == 0 for item in quality_sessions),
        "validation": "BASELINE ONLY" if len(dates) < 20 else "predefined_variants_walk_forward_reporting",
        "sample_warning": "REQUIRES MULTI-DAY VALIDATION" if len(dates) < 20 else "POSSIBLE OVERFITTING; validate on later period",
        "performance_diagnostics": asdict(history_cache.diagnostics),
        "phase_timings": phase_timings,
        "total_elapsed_seconds": time.perf_counter() - analysis_started,
    }
    paths["data_quality"].write_text(json.dumps(quality, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    paths["summary"].write_text("\n".join([
        f"# Multiday Top100 Ranking Analysis {suffix}", "",
        f"FACT: sessions={len(dates)} all_baselines_match={quality['all_baselines_match']} all_leakage_checks_pass={quality['all_leakage_checks_pass']}",
        "FACT: Production baseline is reproduced before variant comparisons.",
        "NOT AVAILABLE: Variant interpretation is blocked for any session with baseline mismatch.",
        "INFERENCE: Coverage, precision, turnover and opportunity cost compare predefined ranking variants.",
        "HYPOTHESIS: Continuation, hybrid, reversal and stabilization scores require independent validation.",
        "BASELINE ONLY: Results do not alter production daily Top100.",
        "REQUIRES MULTI-DAY VALIDATION: Short ranges are descriptive only.",
        "POSSIBLE OVERFITTING: Do not select weights from this same evaluation window.",
    ]) + "\n", encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare causal multiday Top100 ranking variants against the production baseline.")
    parser.add_argument("--date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--history-dir", default="data/history/universe_1m")
    parser.add_argument("--top100-dir", default="data/universe")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", default="data/analysis")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-bars", type=int, default=180)
    parser.add_argument("--min-volume", type=float, default=100_000.0)
    parser.add_argument("--min-dollar-volume", type=float, default=500_000.0)
    parser.add_argument("--prior-sessions", type=int, default=DEFAULT_PRIOR_SESSIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dates = session_dates(args.date, args.start_date, args.end_date)
    settings = BaselineSettings(top_n=args.top_n, min_price=args.min_price, min_bars=args.min_bars, min_volume=args.min_volume, min_dollar_volume=args.min_dollar_volume, prior_sessions=args.prior_sessions)
    print(f"MULTIDAY_TOP100_START start={dates[0]} end={dates[-1]} sessions={len(dates)}", flush=True)
    paths = analyze_range(dates, history_dir=Path(args.history_dir), top100_dir=Path(args.top100_dir), universe_path=Path(args.universe), output_dir=Path(args.output_dir), settings=settings)
    print(f"MULTIDAY_TOP100_DONE output={paths['daily_comparison']}", flush=True)
    return 0
