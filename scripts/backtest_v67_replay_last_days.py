#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.market_calendar import (  # noqa: E402
    get_us_equity_session,
    is_us_equity_trading_day,
    previous_us_equity_trading_day,
)
from src.live_trading.ranking.daily_top100_builder import normalize_history_df, parquet_path  # noqa: E402
from src.live_trading.v67_live_top100_expansion_paper_trader import (  # noqa: E402
    SymbolState,
    compute_live_safe_features,
    update_state,
)


DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_TOP100_DIR = Path("data/universe")
DEFAULT_REPORT_DIR = Path("reports")


@dataclass
class ReplayPosition:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    peak_price: float
    entry_rank: int | None
    entry_score: float
    first_5m_high_pct: float | None
    first_15m_high_pct: float | None
    or_range_pct: float | None


@dataclass
class ReplayTrade:
    session_date: str
    symbol: str
    quantity: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    gross_pnl: float
    estimated_commission: float
    net_pnl: float
    pnl_pct: float
    peak_pct: float
    drop_from_peak_pct: float
    exit_reason: str
    hold_minutes: float
    entry_rank: int | None
    entry_score: float
    first_5m_high_pct: float | None
    first_15m_high_pct: float | None
    or_range_pct: float | None
    top100_source_date: str
    top100_path: str

    def as_row(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": round(self.entry_price, 6),
            "exit_price": round(self.exit_price, 6),
            "gross_pnl": round(self.gross_pnl, 6),
            "estimated_commission": round(self.estimated_commission, 6),
            "net_pnl": round(self.net_pnl, 6),
            "pnl_pct": round(self.pnl_pct, 6),
            "peak_pct": round(self.peak_pct, 6),
            "drop_from_peak_pct": round(self.drop_from_peak_pct, 6),
            "exit_reason": self.exit_reason,
            "hold_minutes": round(self.hold_minutes, 3),
            "entry_rank": self.entry_rank or "",
            "entry_score": round(self.entry_score, 6),
            "first_5m_high_pct": _round_or_blank(self.first_5m_high_pct),
            "first_15m_high_pct": _round_or_blank(self.first_15m_high_pct),
            "or_range_pct": _round_or_blank(self.or_range_pct),
            "top100_source_date": self.top100_source_date,
            "top100_path": self.top100_path,
        }


def _round_or_blank(value: float | None, digits: int = 6) -> float | str:
    if value is None or pd.isna(value):
        return ""
    return round(float(value), digits)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_utc_time(value: str) -> time:
    hh, mm = [int(part) for part in str(value).split(":", 1)]
    return time(hour=hh, minute=mm, tzinfo=timezone.utc)


def trading_days_ending(end_date: date, count: int) -> list[date]:
    days: list[date] = []
    cur = end_date
    while len(days) < count:
        if is_us_equity_trading_day(cur):
            days.append(cur)
        cur = previous_us_equity_trading_day(cur)
    return list(reversed(days))


def trading_days_between(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")
    days: list[date] = []
    cur = start_date
    while cur <= end_date:
        if is_us_equity_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def latest_available_session(history_dir: Path, session_type: str, end_date: date | None = None) -> date | None:
    root = history_dir / f"session_type={session_type.upper()}"
    if not root.exists():
        return None
    latest: date | None = None
    for path in root.glob("symbol=*/year=*/month=*/day=*.parquet"):
        parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in path.parts if "=" in part}
        try:
            parsed = date(int(parts["year"]), int(parts["month"]), int(parts["day"]))
        except Exception:
            continue
        if end_date is not None and parsed > end_date:
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest


def top100_source_date_for_session(session_date: date, mode: str) -> date:
    if mode == "same-session":
        return session_date
    return previous_us_equity_trading_day(session_date)


def find_top100_file(top100_dir: Path, source_date: date, *, allow_latest_available: bool = True) -> tuple[Path | None, date | None]:
    exact = top100_dir / f"daily_top100_{source_date.isoformat()}.csv"
    if exact.exists():
        return exact, source_date
    if not allow_latest_available:
        return None, None
    candidates: list[tuple[date, Path]] = []
    for path in top100_dir.glob("daily_top100_*.csv"):
        stem = path.stem
        if stem.endswith("_diagnostics") or stem == "daily_top100_latest":
            continue
        raw = stem.replace("daily_top100_", "", 1)
        try:
            parsed = parse_date(raw)
        except Exception:
            continue
        if parsed <= source_date:
            candidates.append((parsed, path))
    if candidates:
        return sorted(candidates)[-1][1], sorted(candidates)[-1][0]
    latest = top100_dir / "daily_top100_latest.csv"
    if latest.exists():
        return latest, None
    return None, None


def load_top100(path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} missing symbol column")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out = out[out["symbol"].ne("") & out["symbol"].ne("NAN")]
    if "rank" not in out.columns:
        out["rank"] = range(1, len(out) + 1)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    score_col = "score" if "score" in out.columns else "final_score" if "final_score" in out.columns else "alpha_score"
    if score_col in out.columns:
        out["_top100_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    else:
        out["_top100_score"] = 0.0
    return out.sort_values(["rank", "_top100_score"], ascending=[True, False]).drop_duplicates("symbol").head(limit)


def read_history(history_dir: Path, symbol: str, session_date: date, session_type: str) -> pd.DataFrame:
    path = parquet_path(history_dir, symbol, session_date, session_type)
    if not path.exists():
        return pd.DataFrame()
    return normalize_history_df(pd.read_parquet(path))


def session_open_close(session_date: date, market_open_utc: str, market_close_utc: str) -> tuple[datetime, datetime]:
    session = get_us_equity_session(session_date)
    if session.is_trading_day and session.open_utc and session.close_utc:
        return session.open_utc, session.close_utc
    return (
        datetime.combine(session_date, parse_utc_time(market_open_utc), tzinfo=timezone.utc),
        datetime.combine(session_date, parse_utc_time(market_close_utc), tzinfo=timezone.utc),
    )


def estimated_commission(quantity: int, commission_per_share: float, min_commission: float) -> float:
    per_side = max(float(min_commission), abs(int(quantity)) * float(commission_per_share))
    return per_side * 2.0


def close_trade(
    *,
    session_date: date,
    pos: ReplayPosition,
    exit_time: datetime,
    exit_price: float,
    reason: str,
    commission_per_share: float,
    min_commission: float,
    top100_source_date: str,
    top100_path: str,
) -> ReplayTrade:
    gross = (float(exit_price) - float(pos.entry_price)) * int(pos.quantity)
    comm = estimated_commission(pos.quantity, commission_per_share, min_commission)
    net = gross - comm
    pnl_pct = (float(exit_price) / float(pos.entry_price) - 1.0) * 100.0
    peak_pct = (float(pos.peak_price) / float(pos.entry_price) - 1.0) * 100.0
    drop = (float(exit_price) / float(pos.peak_price) - 1.0) * 100.0 if pos.peak_price > 0 else 0.0
    hold_minutes = (exit_time - pos.entry_time).total_seconds() / 60.0
    return ReplayTrade(
        session_date=session_date.isoformat(),
        symbol=pos.symbol,
        quantity=pos.quantity,
        entry_time=pos.entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        entry_price=float(pos.entry_price),
        exit_price=float(exit_price),
        gross_pnl=gross,
        estimated_commission=comm,
        net_pnl=net,
        pnl_pct=pnl_pct,
        peak_pct=peak_pct,
        drop_from_peak_pct=drop,
        exit_reason=reason,
        hold_minutes=hold_minutes,
        entry_rank=pos.entry_rank,
        entry_score=pos.entry_score,
        first_5m_high_pct=pos.first_5m_high_pct,
        first_15m_high_pct=pos.first_15m_high_pct,
        or_range_pct=pos.or_range_pct,
        top100_source_date=top100_source_date,
        top100_path=top100_path,
    )


def simulate_session(
    session_date: date,
    *,
    top100: pd.DataFrame,
    top100_source_date: str,
    top100_path: Path,
    history_dir: Path,
    session_type: str,
    args: argparse.Namespace,
) -> tuple[list[ReplayTrade], dict[str, Any]]:
    market_open, market_close = session_open_close(session_date, args.market_open_utc, args.market_close_utc)
    eod_flatten_at = datetime.combine(session_date, parse_utc_time(args.eod_flatten_utc), tzinfo=timezone.utc)
    new_entries_start = datetime.combine(session_date, parse_utc_time(args.new_entries_start_utc), tzinfo=timezone.utc)
    no_new_entries_after = datetime.combine(session_date, parse_utc_time(args.no_new_entries_after_utc), tzinfo=timezone.utc)

    strategy_args = SimpleNamespace(
        min_first_5m_high_pct=args.min_first_5m_high_pct,
        min_first_15m_high_pct=args.min_first_15m_high_pct,
        min_or_range_pct=args.min_or_range_pct,
        min_price=args.min_price,
        max_spread_bps=args.max_spread_bps,
        opening_range_seconds=int(args.opening_range_seconds),
    )
    symbols = top100["symbol"].astype(str).str.upper().tolist()
    rank_by_symbol: dict[str, int | None] = {}
    score_by_symbol: dict[str, float] = {}
    for _, row in top100.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        rank = row.get("rank")
        rank_by_symbol[symbol] = int(rank) if rank is not None and not pd.isna(rank) else None
        score = row.get("_top100_score", 0.0)
        score_by_symbol[symbol] = float(score) if score is not None and not pd.isna(score) else 0.0

    frames: dict[str, pd.DataFrame] = {}
    missing_history_symbols: list[str] = []
    for symbol in symbols:
        df = read_history(history_dir, symbol, session_date, session_type)
        if df.empty:
            missing_history_symbols.append(symbol)
            continue
        df = df[(df["timestamp"] >= market_open) & (df["timestamp"] <= market_close)]
        if not df.empty:
            frames[symbol] = df
        else:
            missing_history_symbols.append(symbol)

    states = {symbol: SymbolState(symbol) for symbol in frames}
    open_positions: dict[str, ReplayPosition] = {}
    traded_symbols: set[str] = set()
    trades: list[ReplayTrade] = []
    bars_by_time: dict[pd.Timestamp, list[tuple[str, pd.Series]]] = {}
    for symbol, df in frames.items():
        for _, row in df.iterrows():
            bars_by_time.setdefault(row["timestamp"], []).append((symbol, row))

    ready_events = 0
    missing_history = len(missing_history_symbols)
    for ts in sorted(bars_by_time):
        current_time = pd.Timestamp(ts).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        for symbol, row in bars_by_time[ts]:
            state = states[symbol]
            price = float(row["close"])
            snap = {
                "symbol": symbol,
                "price": price,
                "last": price,
                "close": price,
                "volume": float(row.get("volume", 0.0) or 0.0),
                "spread_bps": None,
            }
            session_elapsed = (current_time - market_open).total_seconds()
            update_state(state, snap, session_elapsed, args.opening_range_seconds, observed_at=current_time, source="live")
            high = float(row["high"])
            low = float(row["low"])
            open_price = float(row["open"])
            if len(state.bars) == 1 and open_price > 0:
                state.first_price = open_price
                state.open_price = open_price
            state.high = max(state.high or high, high)
            state.low = min(state.low or low, low)
            if 0 <= session_elapsed < 5 * 60:
                state.first_5m_high = max(state.first_5m_high or high, high)
            if 0 <= session_elapsed < 15 * 60:
                state.first_15m_high = max(state.first_15m_high or high, high)
            if 0 <= session_elapsed < args.opening_range_seconds:
                state.or_high = max(state.or_high or high, high)
                state.or_low = min(state.or_low or low, low)
            features = compute_live_safe_features(state, snap, strategy_args)

            pos = open_positions.get(symbol)
            if pos is not None:
                close = float(row["close"])
                pos.peak_price = max(pos.peak_price, high)
                stop_price = pos.entry_price * (1.0 - args.stop_loss_pct / 100.0)
                peak_pnl_pct = (pos.peak_price / pos.entry_price - 1.0) * 100.0
                exit_price: float | None = None
                exit_reason: str | None = None
                if low <= stop_price:
                    exit_price = stop_price if args.stop_fill == "stop_price" else close
                    exit_reason = "v46_wide_trail_stop_loss"
                elif peak_pnl_pct >= args.trail_activation_pct:
                    trail_price = pos.peak_price * (1.0 - args.trail_stop_pct / 100.0)
                    if low <= trail_price:
                        exit_price = trail_price if args.stop_fill == "stop_price" else close
                        exit_reason = "v46_wide_trail_trailing_stop"
                if exit_reason is not None and exit_price is not None:
                    trades.append(
                        close_trade(
                            session_date=session_date,
                            pos=pos,
                            exit_time=current_time,
                            exit_price=exit_price,
                            reason=exit_reason,
                            commission_per_share=args.commission_per_share,
                            min_commission=args.min_commission,
                            top100_source_date=top100_source_date,
                            top100_path=str(top100_path),
                        )
                    )
                    del open_positions[symbol]
                    continue

            if current_time < new_entries_start or current_time >= no_new_entries_after or current_time >= eod_flatten_at:
                continue
            if symbol in traded_symbols and args.one_trade_per_symbol_per_day:
                continue
            if symbol in open_positions:
                continue
            if features.get("ready"):
                ready_events += 1
                entry_price = float(features["entry_price"])
                qty = int(args.position_usd // entry_price) if entry_price > 0 else 0
                if qty <= 0:
                    continue
                open_positions[symbol] = ReplayPosition(
                    symbol=symbol,
                    quantity=qty,
                    entry_price=entry_price,
                    entry_time=current_time,
                    peak_price=max(entry_price, float(row["high"])),
                    entry_rank=rank_by_symbol.get(symbol),
                    entry_score=float(features.get("score") or score_by_symbol.get(symbol, 0.0) or 0.0),
                    first_5m_high_pct=features.get("first_5m_high_pct"),
                    first_15m_high_pct=features.get("first_15m_high_pct"),
                    or_range_pct=features.get("or_range_pct"),
                )
                traded_symbols.add(symbol)

        if current_time >= eod_flatten_at and open_positions:
            for symbol, pos in list(open_positions.items()):
                row = next((item_row for item_symbol, item_row in bars_by_time[ts] if item_symbol == symbol), None)
                if row is None:
                    df = frames.get(symbol, pd.DataFrame())
                    prior = df[df["timestamp"] <= ts]
                    if prior.empty:
                        continue
                    row = prior.iloc[-1]
                exit_price = float(row["close"])
                pos.peak_price = max(pos.peak_price, float(row["high"]))
                trades.append(
                    close_trade(
                        session_date=session_date,
                        pos=pos,
                        exit_time=current_time,
                        exit_price=exit_price,
                        reason="v46_wide_trail_close_exit_eod",
                        commission_per_share=args.commission_per_share,
                        min_commission=args.min_commission,
                        top100_source_date=top100_source_date,
                        top100_path=str(top100_path),
                    )
                )
                del open_positions[symbol]

    if open_positions:
        for symbol, pos in list(open_positions.items()):
            df = frames.get(symbol, pd.DataFrame())
            if df.empty:
                continue
            last = df[df["timestamp"] <= eod_flatten_at]
            row = last.iloc[-1] if not last.empty else df.iloc[-1]
            exit_time = pd.Timestamp(row["timestamp"]).to_pydatetime()
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=timezone.utc)
            pos.peak_price = max(pos.peak_price, float(row["high"]))
            trades.append(
                close_trade(
                    session_date=session_date,
                    pos=pos,
                    exit_time=exit_time,
                    exit_price=float(row["close"]),
                    reason="v46_wide_trail_close_exit_eod",
                    commission_per_share=args.commission_per_share,
                    min_commission=args.min_commission,
                    top100_source_date=top100_source_date,
                    top100_path=str(top100_path),
                )
            )
            del open_positions[symbol]

    stats = {
        "session_date": session_date.isoformat(),
        "top100_source_date": top100_source_date,
        "top100_path": str(top100_path),
        "top100_symbols": len(symbols),
        "symbols_with_history": len(frames),
        "missing_history": missing_history,
        "_missing_history_symbols": missing_history_symbols,
        "ready_events": ready_events,
    }
    return trades, stats


def summarize_day(session_date: date, trades: list[ReplayTrade], stats: dict[str, Any]) -> dict[str, Any]:
    gross = sum(t.gross_pnl for t in trades)
    commission = sum(t.estimated_commission for t in trades)
    net = sum(t.net_pnl for t in trades)
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    top_winners = ";".join(f"{t.symbol}:{t.net_pnl:.2f}" for t in sorted(trades, key=lambda item: item.net_pnl, reverse=True)[:5])
    top_losers = ";".join(f"{t.symbol}:{t.net_pnl:.2f}" for t in sorted(trades, key=lambda item: item.net_pnl)[:5])
    return {
        "session_date": session_date.isoformat(),
        "top100_source_date": stats.get("top100_source_date", ""),
        "top100_path": stats.get("top100_path", ""),
        "top100_symbols": stats.get("top100_symbols", 0),
        "symbols_with_history": stats.get("symbols_with_history", 0),
        "missing_history": stats.get("missing_history", 0),
        "ready_events": stats.get("ready_events", 0),
        "trades": len(trades),
        "wins": len(winners),
        "losses": len(losers),
        "win_rate_pct": round(len(winners) / len(trades) * 100.0, 4) if trades else 0.0,
        "gross_pnl": round(gross, 6),
        "estimated_commission": round(commission, 6),
        "net_pnl": round(net, 6),
        "avg_trade_net": round(net / len(trades), 6) if trades else 0.0,
        "avg_win": round(sum(t.net_pnl for t in winners) / len(winners), 6) if winners else 0.0,
        "avg_loss": round(sum(t.net_pnl for t in losers) / len(losers), 6) if losers else 0.0,
        "top_winners": top_winners,
        "top_losers": top_losers,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay current v67 live strategy on historical 1m parquet data.")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--top100-dir", type=Path, default=DEFAULT_TOP100_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--session-type", default="RTH")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--top100-mode", choices=["previous-session", "same-session"], default="previous-session")
    parser.add_argument("--position-usd", type=float, default=1000.0)
    parser.add_argument("--min-first-5m-high-pct", type=float, default=0.5)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=1.0)
    parser.add_argument("--min-or-range-pct", type=float, default=0.5)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-spread-bps", type=float, default=50.0)
    parser.add_argument("--opening-range-seconds", type=int, default=15 * 60)
    parser.add_argument("--stop-loss-pct", type=float, default=8.0)
    parser.add_argument("--trail-activation-pct", type=float, default=3.0)
    parser.add_argument("--trail-stop-pct", type=float, default=3.0)
    parser.add_argument("--one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eod-flatten-utc", default="19:45")
    parser.add_argument("--new-entries-start-utc", default="13:35")
    parser.add_argument("--no-new-entries-after-utc", default="19:30")
    parser.add_argument("--market-open-utc", default="13:30")
    parser.add_argument("--market-close-utc", default="20:00")
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--min-commission", type=float, default=1.0)
    parser.add_argument("--stop-fill", choices=["stop_price", "close"], default="stop_price")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    end_date = args.end_date or latest_available_session(args.history_dir, args.session_type) or previous_us_equity_trading_day(datetime.now(timezone.utc).date())
    if args.start_date is not None:
        sessions = trading_days_between(args.start_date, end_date)
    else:
        sessions = trading_days_ending(end_date, max(1, int(args.days)))

    all_trades: list[ReplayTrade] = []
    daily_rows: list[dict[str, Any]] = []
    missing_top100_rows: list[dict[str, Any]] = []
    missing_history_rows: list[dict[str, Any]] = []
    for session_date in sessions:
        source_date = top100_source_date_for_session(session_date, args.top100_mode)
        top100_path, actual_source_date = find_top100_file(args.top100_dir, source_date)
        if top100_path is None:
            missing_top100_rows.append(
                {
                    "session_date": session_date.isoformat(),
                    "expected_top100_source_date": source_date.isoformat(),
                    "top100_dir": str(args.top100_dir),
                }
            )
            row = {
                "session_date": session_date.isoformat(),
                "top100_source_date": source_date.isoformat(),
                "top100_path": "",
                "top100_symbols": 0,
                "symbols_with_history": 0,
                "missing_history": 0,
                "ready_events": 0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "gross_pnl": 0.0,
                "estimated_commission": 0.0,
                "net_pnl": 0.0,
                "avg_trade_net": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "top_winners": "",
                "top_losers": "",
                "status": "missing_top100",
            }
            daily_rows.append(row)
            print(f"BACKTEST_DAY_SKIPPED date={session_date} reason=missing_top100 expected_source={source_date}")
            continue
        top100 = load_top100(top100_path, args.top_n)
        source_label = actual_source_date.isoformat() if actual_source_date else "latest"
        trades, stats = simulate_session(
            session_date,
            top100=top100,
            top100_source_date=source_label,
            top100_path=top100_path,
            history_dir=args.history_dir,
            session_type=args.session_type,
            args=args,
        )
        for symbol in stats.get("_missing_history_symbols", []):
            missing_history_rows.append(
                {
                    "session_date": session_date.isoformat(),
                    "symbol": symbol,
                    "expected_parquet": str(parquet_path(args.history_dir, symbol, session_date, args.session_type)),
                    "top100_source_date": source_label,
                    "top100_path": str(top100_path),
                }
            )
        all_trades.extend(trades)
        daily = summarize_day(session_date, trades, stats)
        daily["status"] = "ok"
        daily_rows.append(daily)
        print(
            f"BACKTEST_DAY_DONE date={session_date} trades={len(trades)} "
            f"gross={daily['gross_pnl']:.2f} net={daily['net_pnl']:.2f} "
            f"win_rate={daily['win_rate_pct']:.1f}% symbols_with_history={daily['symbols_with_history']} "
            f"missing_history={daily['missing_history']} top100={top100_path}"
        )
        if int(daily.get("symbols_with_history") or 0) == 0:
            print(f"BACKTEST_NO_HISTORY date={session_date} missing_history={daily.get('missing_history')}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    trades_path = args.reports_dir / f"backtest_v67_replay_{stamp}.csv"
    summary_path = args.reports_dir / "backtest_v67_replay_daily_summary.csv"
    missing_top100_path = args.reports_dir / "backtest_v67_replay_missing_top100.csv"
    missing_history_path = args.reports_dir / "backtest_v67_replay_missing_history.csv"
    overall_path = args.reports_dir / "backtest_v67_replay_overall_summary.csv"
    write_csv(trades_path, [trade.as_row() for trade in all_trades])
    write_csv(summary_path, daily_rows)
    write_csv(missing_top100_path, missing_top100_rows)
    write_csv(missing_history_path, missing_history_rows)

    total_gross = sum(trade.gross_pnl for trade in all_trades)
    total_commission = sum(trade.estimated_commission for trade in all_trades)
    total_net = sum(trade.net_pnl for trade in all_trades)
    winners = [trade for trade in all_trades if trade.net_pnl > 0]
    daily_ok = [row for row in daily_rows if row.get("status") == "ok"]
    best_day = max(daily_ok, key=lambda row: float(row.get("net_pnl") or 0.0), default=None)
    worst_day = min(daily_ok, key=lambda row: float(row.get("net_pnl") or 0.0), default=None)
    overall = {
        "start_date": sessions[0].isoformat() if sessions else "",
        "end_date": sessions[-1].isoformat() if sessions else "",
        "sessions": len(sessions),
        "sessions_with_no_history": sum(1 for row in daily_rows if int(row.get("symbols_with_history") or 0) == 0),
        "missing_top100_dates": len(missing_top100_rows),
        "missing_parquet_rows": len(missing_history_rows),
        "total_trades": len(all_trades),
        "wins": len(winners),
        "losses": len(all_trades) - len(winners),
        "win_rate_pct": round((len(winners) / len(all_trades) * 100.0) if all_trades else 0.0, 6),
        "total_gross_pnl": round(total_gross, 6),
        "total_estimated_commission": round(total_commission, 6),
        "total_net_pnl": round(total_net, 6),
        "average_daily_pnl": round((sum(float(row.get("net_pnl") or 0.0) for row in daily_ok) / len(daily_ok)) if daily_ok else 0.0, 6),
        "best_day": best_day.get("session_date", "") if best_day else "",
        "best_day_net_pnl": best_day.get("net_pnl", "") if best_day else "",
        "worst_day": worst_day.get("session_date", "") if worst_day else "",
        "worst_day_net_pnl": worst_day.get("net_pnl", "") if worst_day else "",
    }
    write_csv(overall_path, [overall])
    if missing_top100_rows:
        print(f"BACKTEST_MISSING_TOP100 count={len(missing_top100_rows)} csv={missing_top100_path}")
    no_history_dates = [row["session_date"] for row in daily_rows if int(row.get("symbols_with_history") or 0) == 0]
    if no_history_dates:
        print(f"BACKTEST_ZERO_HISTORY_DATES count={len(no_history_dates)} dates={','.join(no_history_dates)}")
    if missing_history_rows:
        print(f"BACKTEST_MISSING_HISTORY rows={len(missing_history_rows)} csv={missing_history_path}")
    print(
        f"BACKTEST_DONE sessions={len(sessions)} trades={len(all_trades)} "
        f"gross={total_gross:.2f} estimated_commission={total_commission:.2f} "
        f"net={total_net:.2f} win_rate={(len(winners) / len(all_trades) * 100.0 if all_trades else 0.0):.1f}% "
        f"avg_daily={overall['average_daily_pnl']:.2f} best_day={overall['best_day']}:{overall['best_day_net_pnl']} "
        f"worst_day={overall['worst_day']}:{overall['worst_day_net_pnl']} "
        f"trades_csv={trades_path} daily_summary={summary_path} overall_summary={overall_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
