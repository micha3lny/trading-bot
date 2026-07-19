from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analysis.common import fnum, load_session_candles, parse_dt, pct
from src.live_trading.analysis.trade_loader import load_finalized_canonical_trades

DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
HORIZONS_MINUTES = [1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
LOSS_THRESHOLDS = [-0.5, -1.0, -2.0, -3.0]


def iso(value: Any) -> str:
    ts = parse_dt(value)
    return ts.isoformat() if ts is not None else ""


def row_raw(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def candle_window(history_dir: Path, symbol: str, session_date: str, entry_time: pd.Timestamp | None, exit_time: pd.Timestamp | None, session_type: str) -> pd.DataFrame:
    candles = load_session_candles(history_dir, symbol, session_date, session_type)
    if candles.empty or entry_time is None:
        return pd.DataFrame()
    out = candles[candles["timestamp"] >= entry_time]
    if exit_time is not None:
        out = out[out["timestamp"] <= exit_time]
    return out.reset_index(drop=True)


def price_at_horizon(candles: pd.DataFrame, entry_time: pd.Timestamp | None, minutes: int) -> tuple[pd.Timestamp | None, float | None]:
    if candles.empty or entry_time is None:
        return None, None
    target = entry_time + pd.Timedelta(minutes=minutes)
    after = candles[candles["timestamp"] >= target]
    if after.empty:
        return None, None
    row = after.iloc[0]
    return row.get("timestamp"), fnum(row.get("close"))


def path_until(candles: pd.DataFrame, entry_time: pd.Timestamp | None, minutes: int | None) -> pd.DataFrame:
    if candles.empty or entry_time is None:
        return pd.DataFrame()
    if minutes is None:
        return candles
    return candles[candles["timestamp"] <= entry_time + pd.Timedelta(minutes=minutes)]


def path_stats(candles: pd.DataFrame, entry_price: float, entry_time: pd.Timestamp | None, minutes: int | None) -> dict[str, Any]:
    frame = path_until(candles, entry_time, minutes)
    if frame.empty or entry_price <= 0:
        return {"mfe": None, "mae": None, "positive_seen": None, "returned_to_entry": None}
    high = fnum(frame["high"].max())
    low = fnum(frame["low"].min())
    close = fnum(frame.iloc[-1].get("close"))
    return {
        "mfe": pct(high, entry_price) if high is not None else None,
        "mae": pct(low, entry_price) if low is not None else None,
        "positive_seen": int((high or 0) > entry_price),
        "returned_to_entry": int((low or math.inf) <= entry_price <= (high or -math.inf)),
        "last_close": close,
    }


def build_trade_paths(*, date: str, sqlite_path: Path, history_dir: Path, session_type: str = "RTH") -> pd.DataFrame:
    trades = load_finalized_canonical_trades(sqlite_path, date, date)
    rows: list[dict[str, Any]] = []
    for trade in trades.to_dict("records"):
        raw = row_raw(trade)
        symbol = str(trade.get("symbol") or "").upper()
        entry_time = parse_dt(trade.get("entry_fill_time"))
        exit_time = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
        entry_price = fnum(trade.get("entry_price")) or 0.0
        exit_price = fnum(trade.get("exit_price"))
        qty = abs(fnum(trade.get("quantity"), 0.0) or 0.0)
        net_pnl = fnum(trade.get("net_pnl"))
        gross_pnl = fnum(trade.get("gross_pnl"))
        final_pct = pct(exit_price, entry_price) if exit_price is not None and entry_price else None
        candles = candle_window(history_dir, symbol, date, entry_time, exit_time, session_type)
        base = {
            "date": date,
            "trade_id": trade.get("trade_id"),
            "symbol": symbol,
            "entry_time": iso(entry_time),
            "exit_time": iso(exit_time),
            "entry_price": entry_price if entry_price else None,
            "exit_price": exit_price,
            "quantity": qty,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "final_pnl_pct": final_pct,
            "peak_pct": fnum(trade.get("mfe_pct") or raw.get("peak_pct")),
            "drop_from_peak_pct": raw.get("drop_from_peak_pct"),
            "peak_data_quality": raw.get("peak_data_quality"),
            "candles_found": len(candles),
        }
        for minutes in HORIZONS_MINUTES:
            ts, price = price_at_horizon(candles, entry_time, minutes)
            stats = path_stats(candles, entry_price, entry_time, minutes)
            base[f"price_at_{minutes}m"] = price
            base[f"time_at_{minutes}m"] = iso(ts)
            base[f"pnl_pct_at_{minutes}m"] = pct(price, entry_price) if price is not None and entry_price else None
            base[f"unrealized_pnl_at_{minutes}m"] = ((price - entry_price) * qty) if price is not None and entry_price else None
            base[f"mfe_to_{minutes}m_pct"] = stats["mfe"]
            base[f"mae_to_{minutes}m_pct"] = stats["mae"]
            base[f"positive_seen_to_{minutes}m"] = stats["positive_seen"]
            base[f"returned_to_entry_to_{minutes}m"] = stats["returned_to_entry"]
        eod_stats = path_stats(candles, entry_price, entry_time, None)
        base["mfe_to_eod_pct"] = eod_stats["mfe"]
        base["mae_to_eod_pct"] = eod_stats["mae"]
        base["ever_positive"] = eod_stats["positive_seen"]
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_rule(df: pd.DataFrame, *, name: str, affected: pd.Series, simulated_pnl: pd.Series) -> dict[str, Any]:
    final = pd.to_numeric(df.get("net_pnl"), errors="coerce").fillna(0.0)
    affected = affected.fillna(False).astype(bool)
    sim = simulated_pnl.where(affected, final).fillna(final)
    winners = final > 0
    return {
        "rule": name,
        "affected_trades": int(affected.sum()),
        "losers_improved": int(((sim > final) & ~winners & affected).sum()),
        "winners_damaged": int(((sim < final) & winners & affected).sum()),
        "loss_avoided": float((sim - final).where(~winners & affected, 0).sum()),
        "profit_sacrificed": float((final - sim).where(winners & affected & (sim < final), 0).sum()),
        "net_improvement": float(sim.sum() - final.sum()),
        "baseline_net_pnl": float(final.sum()),
        "simulated_net_pnl": float(sim.sum()),
    }


def build_rules(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows = []
    final_pct = pd.to_numeric(paths.get("final_pnl_pct"), errors="coerce")
    qty = pd.to_numeric(paths.get("quantity"), errors="coerce").fillna(0.0)
    entry = pd.to_numeric(paths.get("entry_price"), errors="coerce")
    for minutes in [5, 10, 15, 20, 30, 45, 60]:
        pct_source = paths[f"pnl_pct_at_{minutes}m"] if f"pnl_pct_at_{minutes}m" in paths.columns else pd.Series([pd.NA] * len(paths), index=paths.index)
        pct_col = pd.to_numeric(pct_source, errors="coerce")
        pnl_at = (pct_col / 100.0) * entry * qty
        rows.append(summarize_rule(paths, name=f"exit_losers_after_{minutes}m", affected=(pct_col < 0) & (final_pct < 0), simulated_pnl=pnl_at))
        for threshold in LOSS_THRESHOLDS:
            rows.append(summarize_rule(paths, name=f"exit_if_pnl_lt_{threshold:g}_after_{minutes}m", affected=pct_col < threshold, simulated_pnl=pnl_at))
        positive_source = paths[f"positive_seen_to_{minutes}m"] if f"positive_seen_to_{minutes}m" in paths.columns else pd.Series([pd.NA] * len(paths), index=paths.index)
        positive = pd.to_numeric(positive_source, errors="coerce")
        rows.append(summarize_rule(paths, name=f"exit_if_never_positive_to_{minutes}m", affected=positive.eq(0) & positive.notna(), simulated_pnl=pnl_at))
    return pd.DataFrame(rows)


def write_summary(paths: pd.DataFrame, rules: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Early Loser Exit Summary {path.stem.split('_')[-1]}",
        "",
        "FACT: This is candle-based read-only path analysis using finalized canonical trades.",
        "HYPOTHESIS: Rule results are approximations and require multi-day validation before any live change.",
        "BASELINE ONLY: One session is not enough for strategy changes.",
        "POSSIBLE OVERFITTING: Treat best-performing filters as candidates, not decisions.",
        "",
        f"trades={len(paths)}",
        f"baseline_net_pnl={pd.to_numeric(paths.get('net_pnl'), errors='coerce').sum() if not paths.empty else 0:.4f}",
        "",
        "## Top Rules By Net Improvement",
        "",
    ]
    if not rules.empty:
        for row in rules.sort_values("net_improvement", ascending=False).head(12).to_dict("records"):
            lines.append(f"- {row['rule']}: net_improvement={row['net_improvement']:.4f}, affected={row['affected_trades']}, winners_damaged={row['winners_damaged']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, date: str, sqlite_path: Path, history_dir: Path, output_dir: Path, session_type: str = "RTH") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_trade_paths(date=date, sqlite_path=sqlite_path, history_dir=history_dir, session_type=session_type)
    rules = build_rules(paths)
    trade_path = output_dir / f"early_loser_trade_paths_{date}.csv"
    rules_path = output_dir / f"early_loser_rules_{date}.csv"
    summary_path = output_dir / f"early_loser_summary_{date}.md"
    paths.to_csv(trade_path, index=False)
    rules.to_csv(rules_path, index=False)
    write_summary(paths, rules, summary_path)
    return {"trade_paths": trade_path, "rules": rules_path, "summary": summary_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze early exit rules for losing finalized canonical trades.")
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
    print(f"EARLY_LOSER_DONE date={args.date} elapsed_seconds={time.monotonic() - started:.1f} output={outputs['trade_paths']} summary={outputs['summary']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
