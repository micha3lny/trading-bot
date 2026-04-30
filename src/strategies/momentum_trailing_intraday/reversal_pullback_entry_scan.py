# (header unchanged)
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_daily, load_intraday
from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.configs_v27 import V27_ENTRY_CORE

# (rest of file unchanged until PRESETS)

PRESETS: dict[str, EntryScanPreset] = {
    "loose": EntryScanPreset(
        name="loose",
        description="very permissive entry scan to estimate maximum opportunity count",
    ),
    "balanced": EntryScanPreset(
        name="balanced",
        description="moderate entry scan; target roughly one opportunity per active session",
        min_avg_daily_range_pct=4.0,
        max_daily_trend_pct=-3.0,
        min_breakout_pct=0.60,
        max_breakout_pct=3.00,
        min_confirmation_close_strength=0.20,
        max_confirmation_close_strength=1.00,
        max_setup_entry_risk_pct=12.0,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.85,
        max_5m_entry_risk_pct=10.0,
        max_5m_close_below_or_high_pct=1.00,
        min_1m_close_strength=0.65,
        max_1m_close_strength=0.95,
        max_1m_entry_risk_pct=10.0,
        max_1m_close_below_or_high_pct=1.00,
    ),
    "quality": EntryScanPreset(
        name="quality",
        description="stricter entry scan matching the current high-quality research direction",
        min_avg_daily_range_pct=4.5,
        max_daily_trend_pct=-3.0,
        min_breakout_pct=1.00,
        max_breakout_pct=2.20,
        min_confirmation_close_strength=0.30,
        max_confirmation_close_strength=1.00,
        max_setup_entry_risk_pct=10.0,
        min_pullback_from_confirmation_pct=0.20,
        max_pullback_from_confirmation_pct=3.00,
        max_5m_close_strength=0.75,
        max_5m_entry_risk_pct=8.0,
        max_5m_close_below_or_high_pct=0.75,
        min_1m_close_strength=0.75,
        max_1m_close_strength=0.90,
        max_1m_entry_risk_pct=8.0,
        max_1m_close_below_or_high_pct=0.75,
    ),
    "v27": EntryScanPreset(
        name="v27",
        description="entry-first config from configs_v27",
        min_avg_daily_range_pct=V27_ENTRY_CORE["min_avg_daily_range"],
        min_daily_trend_pct=V27_ENTRY_CORE["daily_trend_min"],
        max_daily_trend_pct=V27_ENTRY_CORE["daily_trend_max"],
        min_breakout_pct=V27_ENTRY_CORE["min_breakout_pct"],
        max_breakout_pct=V27_ENTRY_CORE["max_breakout_pct"],
        min_confirmation_close_strength=V27_ENTRY_CORE["min_15m_confirmation_close_strength"],
        max_confirmation_close_strength=V27_ENTRY_CORE["max_15m_confirmation_close_strength"],
        max_setup_entry_risk_pct=V27_ENTRY_CORE["max_entry_risk_pct"],
        min_pullback_from_confirmation_pct=V27_ENTRY_CORE["min_pullback_pct"],
        max_pullback_from_confirmation_pct=V27_ENTRY_CORE["max_pullback_pct"],
        max_5m_close_strength=V27_ENTRY_CORE["max_5m_close_strength"],
        max_5m_entry_risk_pct=V27_ENTRY_CORE["max_entry_risk_pct"],
        min_1m_close_strength=V27_ENTRY_CORE["min_1m_close_strength"],
        max_1m_close_strength=V27_ENTRY_CORE["max_1m_close_strength"],
        max_1m_entry_risk_pct=V27_ENTRY_CORE["max_entry_risk_pct"],
    ),
}
