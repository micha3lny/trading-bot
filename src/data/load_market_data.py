"""Utilities for loading local market data from Parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DAILY_DATA_DIR = Path("data/market_data")
INTRADAY_DATA_DIR = Path("data/market_data_intraday")


@dataclass(frozen=True)
class MarketDataBundle:
    symbol: str
    daily: pd.DataFrame
    intraday: pd.DataFrame


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common IBKR dataframe columns for downstream code."""
    normalized = df.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])
        normalized = normalized.sort_values("date").reset_index(drop=True)

    return normalized


def load_daily(symbol: str, data_dir: Path = DAILY_DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{symbol}_1D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Daily data file not found: {path}")

    return _normalize_columns(pd.read_parquet(path))


def load_intraday(
    symbol: str,
    interval: str = "15m",
    data_dir: Path = INTRADAY_DATA_DIR,
) -> pd.DataFrame:
    path = data_dir / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Intraday data file not found: {path}")

    return _normalize_columns(pd.read_parquet(path))


def load_market_data_bundle(symbol: str, interval: str = "15m") -> MarketDataBundle:
    return MarketDataBundle(
        symbol=symbol,
        daily=load_daily(symbol),
        intraday=load_intraday(symbol, interval=interval),
    )
