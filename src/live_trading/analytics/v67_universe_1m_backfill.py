from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ib_insync import IB, Stock

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 766
DEFAULT_UNIVERSE = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_OUTPUT_DIR = "data/live/universe_candles_1m"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def load_symbols(path: str, limit: int | None = None, min_price: float | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing universe file: {p}")
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("Universe CSV must contain symbol column")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if min_price is not None and "last_close" in df.columns:
        df["last_close"] = pd.to_numeric(df["last_close"], errors="coerce")
        df = df[df["last_close"] >= min_price]
    if "alpha_score" in df.columns:
        df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("alpha_score", ascending=False)
    symbols = df["symbol"].dropna().drop_duplicates().tolist()
    return symbols[:limit] if limit else symbols


def existing_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("symbol", "")).upper()
                ts = str(row.get("bar_time", ""))
                if sym and ts:
                    keys.add((sym, ts))
    except Exception:
        return keys
    return keys


def append_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "trade_count",
        "source",
        "recorded_at",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return len(rows)


def backfill_symbol(ib: IB, symbol: str, duration: str, use_rth: bool, output_file: Path, keys: set[tuple[str, str]]) -> int:
    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    rows = []
    recorded_at = now_utc()
    for bar in bars:
        bar_time = str(getattr(bar, "date", ""))
        key = (symbol, bar_time)
        if key in keys:
            continue
        keys.add(key)
        rows.append(
            {
                "symbol": symbol,
                "bar_time": bar_time,
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "source": "ibkr_universe_post_session_backfill_1m",
                "recorded_at": recorded_at,
            }
        )
    return append_rows(output_file, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill 1m candles for the full v67 universe after session close.")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%F"))
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    ap.add_argument("--duration", default="1 D")
    ap.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=0, help="Optional test limit. 0 = full universe.")
    ap.add_argument("--min-price", type=float, default=None)
    ap.add_argument("--pause", type=float, default=0.25)
    ap.add_argument("--max-errors", type=int, default=50)
    args = ap.parse_args()

    symbols = load_symbols(args.universe, limit=args.limit or None, min_price=args.min_price)
    out_dir = Path(args.output_dir) / args.date
    output_file = out_dir / "universe_candles_1m.csv"
    keys = existing_keys(output_file)

    print(f"{now_utc()} universe_backfill_start date={args.date} symbols={len(symbols)} output={output_file} use_rth={args.use_rth}", flush=True)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)

    total_rows = 0
    errors = 0
    try:
        for idx, symbol in enumerate(symbols, start=1):
            try:
                rows = backfill_symbol(ib, symbol, args.duration, args.use_rth, output_file, keys)
                total_rows += rows
                print(f"{now_utc()} universe_backfill_symbol {idx}/{len(symbols)} symbol={symbol} rows={rows} total_rows={total_rows}", flush=True)
            except Exception as exc:
                errors += 1
                print(f"{now_utc()} universe_backfill_error {idx}/{len(symbols)} symbol={symbol} error={exc!r}", flush=True)
                if errors >= args.max_errors:
                    raise SystemExit(f"Too many errors: {errors}")
            time.sleep(args.pause)
    finally:
        ib.disconnect()

    print(f"{now_utc()} universe_backfill_done symbols={len(symbols)} rows={total_rows} errors={errors} output={output_file}", flush=True)


if __name__ == "__main__":
    main()
