from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = "data/live/recorder"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_date_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def safe_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return str(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    ensure_parent(path)
    exists = path.exists() and path.stat().st_size > 0
    clean = {k: row.get(k, "") for k in fieldnames}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(clean)


def append_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    ensure_parent(path)
    rows = list(rows)
    if not rows:
        return 0
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return len(rows)


@dataclass
class LiveCandle1m:
    symbol: str
    bar_time: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    wap: float | None = None
    trade_count: int | None = None
    source: str = "ibkr"
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class ExtendedHoursCandle1m:
    symbol: str
    bar_time: str
    session_type: str  # premarket or afterhours
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    wap: float | None = None
    trade_count: int | None = None
    bid: float | None = None
    ask: float | None = None
    mid_price: float | None = None
    spread_bps: float | None = None
    relative_volume: float | None = None
    market_regime: str = ""
    source: str = "ibkr"
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class MarketDataSnapshot:
    symbol: str
    price: float | None = None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    volume: float | None = None
    close: float | None = None
    spread_bps: float | None = None
    source: str = "ibkr"
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class SpreadSnapshot:
    symbol: str
    timestamp: str
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    mid_price: float | None = None
    spread_regime: str = ""
    liquidity_regime: str = ""
    session_type: str = ""
    source: str = "ibkr"
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class PremarketSummary:
    symbol: str
    session_date: str
    premarket_open: float | None = None
    premarket_close: float | None = None
    premarket_high: float | None = None
    premarket_low: float | None = None
    premarket_range_pct: float | None = None
    premarket_volume: float | None = None
    premarket_dollar_volume: float | None = None
    premarket_vwap: float | None = None
    premarket_trend_pct: float | None = None
    premarket_close_vs_high_pct: float | None = None
    premarket_close_vs_low_pct: float | None = None
    premarket_high_time: str = ""
    premarket_low_time: str = ""
    premarket_relative_volume: float | None = None
    premarket_gap_pct: float | None = None
    premarket_spread_avg_bps: float | None = None
    premarket_spread_max_bps: float | None = None
    premarket_breakout_distance_pct: float | None = None
    news_flag: bool | None = None
    earnings_flag: bool | None = None
    features_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class AfterhoursSummary:
    symbol: str
    session_date: str
    afterhours_open: float | None = None
    afterhours_close: float | None = None
    afterhours_high: float | None = None
    afterhours_low: float | None = None
    afterhours_range_pct: float | None = None
    afterhours_volume: float | None = None
    afterhours_dollar_volume: float | None = None
    afterhours_vwap: float | None = None
    afterhours_trend_pct: float | None = None
    afterhours_relative_volume: float | None = None
    afterhours_close_vs_high_pct: float | None = None
    afterhours_close_vs_low_pct: float | None = None
    afterhours_spread_avg_bps: float | None = None
    afterhours_spread_max_bps: float | None = None
    news_flag: bool | None = None
    earnings_flag: bool | None = None
    features_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class OvernightMarketContext:
    session_date: str
    spy_overnight_pct: float | None = None
    qqq_overnight_pct: float | None = None
    iwm_overnight_pct: float | None = None
    vix_change_pct: float | None = None
    spy_premarket_high_pct: float | None = None
    spy_premarket_low_pct: float | None = None
    qqq_premarket_high_pct: float | None = None
    qqq_premarket_low_pct: float | None = None
    futures_trend: str = ""
    market_gap_regime: str = ""
    market_volatility_regime: str = ""
    context_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class SelectionEvent:
    symbol: str
    stage: str
    decision: str
    rank: int | None = None
    score: float | None = None
    reason: str = ""
    first_5m_high_pct: float | None = None
    first_15m_high_pct: float | None = None
    or_range_pct: float | None = None
    entry_price: float | None = None
    market_regime: str = ""
    setup_quality: str = ""
    features_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class SignalSnapshot:
    symbol: str
    signal_name: str
    signal_value: float | str | None = None
    action: str = ""
    score: float | None = None
    threshold: float | None = None
    reasons: str = ""
    features_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class OrderIntent:
    symbol: str
    action: str
    quantity: float | None = None
    notional_usd: float | None = None
    order_type: str = ""
    limit_price: float | None = None
    stop_price: float | None = None
    strategy: str = ""
    reason: str = ""
    signal_score: float | None = None
    features_json: str = ""
    client_order_id: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class FillEvent:
    symbol: str
    action: str
    quantity: float | None = None
    fill_price: float | None = None
    commission: float | None = None
    order_id: str = ""
    client_order_id: str = ""
    execution_id: str = ""
    realized_pnl: float | None = None
    slippage_bps: float | None = None
    raw_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class PortfolioSnapshot:
    account: str = ""
    cash: float | None = None
    net_liquidation: float | None = None
    buying_power: float | None = None
    gross_exposure: float | None = None
    open_positions: int | None = None
    daily_pnl: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    positions_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass
class ErrorEvent:
    component: str
    message: str
    severity: str = "ERROR"
    symbol: str = ""
    exception_type: str = ""
    raw_json: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)


class LiveDataRecorder:
    """Append-only CSV recorder for paper/live trading experiments.

    This class intentionally has no IBKR dependency. The live engine should call these methods
    whenever it receives candles, market data, extended-hours data, selection decisions,
    signals, orders, fills, portfolio snapshots, or errors.
    """

    candle_fields = list(LiveCandle1m.__dataclass_fields__.keys())
    extended_candle_fields = list(ExtendedHoursCandle1m.__dataclass_fields__.keys())
    market_fields = list(MarketDataSnapshot.__dataclass_fields__.keys())
    spread_fields = list(SpreadSnapshot.__dataclass_fields__.keys())
    premarket_summary_fields = list(PremarketSummary.__dataclass_fields__.keys())
    afterhours_summary_fields = list(AfterhoursSummary.__dataclass_fields__.keys())
    overnight_context_fields = list(OvernightMarketContext.__dataclass_fields__.keys())
    selection_fields = list(SelectionEvent.__dataclass_fields__.keys())
    signal_fields = list(SignalSnapshot.__dataclass_fields__.keys())
    order_fields = list(OrderIntent.__dataclass_fields__.keys())
    fill_fields = list(FillEvent.__dataclass_fields__.keys())
    portfolio_fields = list(PortfolioSnapshot.__dataclass_fields__.keys())
    error_fields = list(ErrorEvent.__dataclass_fields__.keys())

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR, session_date: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.session_date = session_date or session_date_utc()
        self.session_dir = self.output_dir / self.session_date
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.session_dir / name

    def record_candle_1m(self, candle: LiveCandle1m | dict[str, Any]) -> None:
        row = asdict(candle) if isinstance(candle, LiveCandle1m) else dict(candle)
        append_csv_row(self.path("candles_1m.csv"), row, self.candle_fields)

    def record_candles_1m(self, candles: Iterable[LiveCandle1m | dict[str, Any]]) -> int:
        rows = [asdict(c) if isinstance(c, LiveCandle1m) else dict(c) for c in candles]
        return append_csv_rows(self.path("candles_1m.csv"), rows, self.candle_fields)

    def record_extended_hours_candle_1m(self, candle: ExtendedHoursCandle1m | dict[str, Any]) -> None:
        row = asdict(candle) if isinstance(candle, ExtendedHoursCandle1m) else dict(candle)
        session_type = str(row.get("session_type", "")).lower()
        filename = "premarket_1m.csv" if session_type == "premarket" else "afterhours_1m.csv"
        append_csv_row(self.path(filename), row, self.extended_candle_fields)

    def record_extended_hours_candles_1m(self, candles: Iterable[ExtendedHoursCandle1m | dict[str, Any]]) -> int:
        count = 0
        for candle in candles:
            self.record_extended_hours_candle_1m(candle)
            count += 1
        return count

    def record_premarket_candle_1m(self, candle: ExtendedHoursCandle1m | dict[str, Any]) -> None:
        row = asdict(candle) if isinstance(candle, ExtendedHoursCandle1m) else dict(candle)
        row["session_type"] = "premarket"
        append_csv_row(self.path("premarket_1m.csv"), row, self.extended_candle_fields)

    def record_afterhours_candle_1m(self, candle: ExtendedHoursCandle1m | dict[str, Any]) -> None:
        row = asdict(candle) if isinstance(candle, ExtendedHoursCandle1m) else dict(candle)
        row["session_type"] = "afterhours"
        append_csv_row(self.path("afterhours_1m.csv"), row, self.extended_candle_fields)

    def record_market_snapshot(self, snapshot: MarketDataSnapshot | dict[str, Any]) -> None:
        row = asdict(snapshot) if isinstance(snapshot, MarketDataSnapshot) else dict(snapshot)
        append_csv_row(self.path("market_snapshots.csv"), row, self.market_fields)

    def record_spread_snapshot(self, snapshot: SpreadSnapshot | dict[str, Any]) -> None:
        row = asdict(snapshot) if isinstance(snapshot, SpreadSnapshot) else dict(snapshot)
        append_csv_row(self.path("spread_snapshots.csv"), row, self.spread_fields)

    def record_premarket_summary(self, summary: PremarketSummary | dict[str, Any]) -> None:
        row = asdict(summary) if isinstance(summary, PremarketSummary) else dict(summary)
        if row.get("features_json") and not isinstance(row.get("features_json"), str):
            row["features_json"] = safe_json(row["features_json"])
        append_csv_row(self.path("premarket_summary.csv"), row, self.premarket_summary_fields)

    def record_afterhours_summary(self, summary: AfterhoursSummary | dict[str, Any]) -> None:
        row = asdict(summary) if isinstance(summary, AfterhoursSummary) else dict(summary)
        if row.get("features_json") and not isinstance(row.get("features_json"), str):
            row["features_json"] = safe_json(row["features_json"])
        append_csv_row(self.path("afterhours_summary.csv"), row, self.afterhours_summary_fields)

    def record_overnight_market_context(self, context: OvernightMarketContext | dict[str, Any]) -> None:
        row = asdict(context) if isinstance(context, OvernightMarketContext) else dict(context)
        if row.get("context_json") and not isinstance(row.get("context_json"), str):
            row["context_json"] = safe_json(row["context_json"])
        append_csv_row(self.path("overnight_market_context.csv"), row, self.overnight_context_fields)

    def record_selection(self, event: SelectionEvent | dict[str, Any]) -> None:
        row = asdict(event) if isinstance(event, SelectionEvent) else dict(event)
        if row.get("features_json") and not isinstance(row.get("features_json"), str):
            row["features_json"] = safe_json(row["features_json"])
        append_csv_row(self.path("selection_events.csv"), row, self.selection_fields)

    def record_signal(self, signal: SignalSnapshot | dict[str, Any]) -> None:
        row = asdict(signal) if isinstance(signal, SignalSnapshot) else dict(signal)
        if row.get("features_json") and not isinstance(row.get("features_json"), str):
            row["features_json"] = safe_json(row["features_json"])
        append_csv_row(self.path("signal_snapshots.csv"), row, self.signal_fields)

    def record_order_intent(self, intent: OrderIntent | dict[str, Any]) -> None:
        row = asdict(intent) if isinstance(intent, OrderIntent) else dict(intent)
        if row.get("features_json") and not isinstance(row.get("features_json"), str):
            row["features_json"] = safe_json(row["features_json"])
        append_csv_row(self.path("order_intents.csv"), row, self.order_fields)

    def record_fill(self, fill: FillEvent | dict[str, Any]) -> None:
        row = asdict(fill) if isinstance(fill, FillEvent) else dict(fill)
        if row.get("raw_json") and not isinstance(row.get("raw_json"), str):
            row["raw_json"] = safe_json(row["raw_json"])
        append_csv_row(self.path("fills.csv"), row, self.fill_fields)

    def record_portfolio(self, snapshot: PortfolioSnapshot | dict[str, Any]) -> None:
        row = asdict(snapshot) if isinstance(snapshot, PortfolioSnapshot) else dict(snapshot)
        if row.get("positions_json") and not isinstance(row.get("positions_json"), str):
            row["positions_json"] = safe_json(row["positions_json"])
        append_csv_row(self.path("portfolio_snapshots.csv"), row, self.portfolio_fields)

    def record_error(self, event: ErrorEvent | dict[str, Any]) -> None:
        row = asdict(event) if isinstance(event, ErrorEvent) else dict(event)
        if row.get("raw_json") and not isinstance(row.get("raw_json"), str):
            row["raw_json"] = safe_json(row["raw_json"])
        append_csv_row(self.path("error_events.csv"), row, self.error_fields)

    def record_run_metadata(self, metadata: dict[str, Any]) -> None:
        payload = {
            "recorded_at": utc_now_iso(),
            "session_date": self.session_date,
            "metadata_json": safe_json(metadata),
        }
        append_csv_row(self.path("run_metadata.csv"), payload, ["recorded_at", "session_date", "metadata_json"])

    def write_manifest(self) -> Path:
        manifest = {
            "session_date": self.session_date,
            "session_dir": str(self.session_dir),
            "created_at": utc_now_iso(),
            "files": [
                "candles_1m.csv",
                "premarket_1m.csv",
                "afterhours_1m.csv",
                "premarket_summary.csv",
                "afterhours_summary.csv",
                "overnight_market_context.csv",
                "spread_snapshots.csv",
                "market_snapshots.csv",
                "selection_events.csv",
                "signal_snapshots.csv",
                "order_intents.csv",
                "fills.csv",
                "portfolio_snapshots.csv",
                "error_events.csv",
                "run_metadata.csv",
            ],
        }
        path = self.path("manifest.json")
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def demo_records(recorder: LiveDataRecorder) -> None:
    recorder.record_run_metadata({
        "mode": "demo",
        "strategy": "v59_top100_live_safe_expansion",
        "extended_hours_enabled": True,
        "pid": os.getpid(),
    })
    recorder.record_candle_1m(LiveCandle1m(symbol="AAPL", bar_time=utc_now_iso(), open=100, high=101, low=99.5, close=100.5, volume=12345))
    recorder.record_premarket_candle_1m(ExtendedHoursCandle1m(symbol="AAPL", bar_time=utc_now_iso(), session_type="premarket", open=99.0, high=101.0, low=98.8, close=100.6, volume=25_000, wap=100.1, bid=100.55, ask=100.65, mid_price=100.60, spread_bps=9.94, relative_volume=3.2, market_regime="neutral"))
    recorder.record_afterhours_candle_1m(ExtendedHoursCandle1m(symbol="AAPL", bar_time=utc_now_iso(), session_type="afterhours", open=100.5, high=101.2, low=100.1, close=100.9, volume=18_000, wap=100.7, bid=100.85, ask=100.95, mid_price=100.90, spread_bps=9.91, relative_volume=2.1, market_regime="neutral"))
    recorder.record_market_snapshot(MarketDataSnapshot(symbol="AAPL", price=100.5, bid=100.49, ask=100.51, spread_bps=2.0))
    recorder.record_spread_snapshot(SpreadSnapshot(symbol="AAPL", timestamp=utc_now_iso(), bid=100.49, ask=100.51, spread=0.02, spread_bps=2.0, mid_price=100.50, spread_regime="tight", liquidity_regime="good", session_type="regular"))
    recorder.record_premarket_summary(PremarketSummary(symbol="AAPL", session_date=recorder.session_date, premarket_open=99.0, premarket_close=100.6, premarket_high=101.0, premarket_low=98.8, premarket_range_pct=2.22, premarket_volume=25_000, premarket_dollar_volume=2_500_000, premarket_vwap=100.1, premarket_trend_pct=1.62, premarket_close_vs_high_pct=-0.40, premarket_close_vs_low_pct=1.82, premarket_relative_volume=3.2, premarket_gap_pct=1.1, premarket_spread_avg_bps=10.0, premarket_spread_max_bps=18.0, news_flag=False, earnings_flag=False, features_json={"demo": True}))
    recorder.record_afterhours_summary(AfterhoursSummary(symbol="AAPL", session_date=recorder.session_date, afterhours_open=100.5, afterhours_close=100.9, afterhours_high=101.2, afterhours_low=100.1, afterhours_range_pct=1.09, afterhours_volume=18_000, afterhours_dollar_volume=1_814_000, afterhours_vwap=100.7, afterhours_trend_pct=0.40, afterhours_relative_volume=2.1, afterhours_close_vs_high_pct=-0.30, afterhours_spread_avg_bps=10.0, afterhours_spread_max_bps=20.0, news_flag=False, earnings_flag=False, features_json={"demo": True}))
    recorder.record_overnight_market_context(OvernightMarketContext(session_date=recorder.session_date, spy_overnight_pct=0.15, qqq_overnight_pct=0.22, iwm_overnight_pct=-0.05, vix_change_pct=-1.2, spy_premarket_high_pct=0.35, spy_premarket_low_pct=-0.10, qqq_premarket_high_pct=0.42, qqq_premarket_low_pct=-0.08, futures_trend="up", market_gap_regime="mild_gap_up", market_volatility_regime="normal", context_json={"demo": True}))
    recorder.record_selection(SelectionEvent(symbol="AAPL", stage="live_safe_expansion", decision="accepted", score=87.5, reason="first_5m_high_pct>=4;first_15m_high_pct>=6.5;or_range_pct>=5", features_json={"first_5m_high_pct": 4.8}))
    recorder.record_signal(SignalSnapshot(symbol="AAPL", signal_name="momentum_or_breakout", action="BUY", score=87.5, threshold=80, reasons="live_safe_expansion"))
    recorder.record_order_intent(OrderIntent(symbol="AAPL", action="BUY", quantity=10, notional_usd=1005, order_type="MKT", strategy="v59_top100_live_safe_expansion", reason="paper demo"))
    recorder.record_fill(FillEvent(symbol="AAPL", action="BUY", quantity=10, fill_price=100.52, commission=1.0, slippage_bps=2.0))
    recorder.record_portfolio(PortfolioSnapshot(account="DEMO", cash=24_000, net_liquidation=25_000, buying_power=100_000, gross_exposure=1_005, open_positions=1, positions_json={"AAPL": 10}))
    recorder.record_error(ErrorEvent(component="demo", severity="INFO", message="demo recorder event"))
    recorder.write_manifest()


def main() -> int:
    parser = argparse.ArgumentParser(description="v62 live data recorder utility")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--demo", action="store_true", help="write one demo row to each recorder file")
    args = parser.parse_args()

    recorder = LiveDataRecorder(args.output_dir, args.session_date)
    recorder.write_manifest()

    print("=== v62 live data recorder ===")
    print(f"Output dir: {recorder.session_dir}")

    if args.demo:
        demo_records(recorder)
        print("Demo rows written")

    print("Files:")
    for name in [
        "candles_1m.csv",
        "premarket_1m.csv",
        "afterhours_1m.csv",
        "premarket_summary.csv",
        "afterhours_summary.csv",
        "overnight_market_context.csv",
        "spread_snapshots.csv",
        "market_snapshots.csv",
        "selection_events.csv",
        "signal_snapshots.csv",
        "order_intents.csv",
        "fills.csv",
        "portfolio_snapshots.csv",
        "error_events.csv",
        "run_metadata.csv",
        "manifest.json",
    ]:
        print(f"- {recorder.path(name)}")

    print("\nUsage from live engine:")
    print("from src.live_trading.v62_live_data_recorder import LiveDataRecorder, SelectionEvent, ExtendedHoursCandle1m")
    print("rec = LiveDataRecorder(); rec.record_selection(SelectionEvent(symbol='AAPL', stage='top100', decision='accepted'))")
    print("rec.record_premarket_candle_1m(ExtendedHoursCandle1m(symbol='AAPL', bar_time='...', session_type='premarket'))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
