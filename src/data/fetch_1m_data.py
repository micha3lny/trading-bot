"""
IBKR 1m historical data backfill.

Downloads REAL 1-minute OHLCV candles from Interactive Brokers via ib_insync.
Replaces the old placeholder/fake-data implementation.

Examples:
    python -m src.data.fetch_1m_data --universe nasdaq --days 90 --port 7497
    python -m src.data.fetch_1m_data --symbols AAPL TSLA NVDA --days 90 --port 7497
    python -m src.data.fetch_1m_data --symbols-file symbols.txt --days 90 --port 7497

Notes:
- Requires TWS or IB Gateway running and API enabled.
- TWS paper trading commonly uses port 7497, live TWS 7496.
- IBKR historical pacing limits are real. For large universes this can take many hours.
- The script is resumable and skips symbols with enough existing rows unless --force is used.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from ib_insync import IB, Stock, util


OUTPUT_DIR = Path("data/1m")
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass(frozen=True)
class FetchConfig:
    host: str
    port: int
    client_id: int
    days: int
    output_dir: Path
    chunk_days: int
    sleep_seconds: float
    timeout: int
    exchange: str
    currency: str
    what_to_show: str
    use_rth: bool
    force: bool
    min_rows_per_day: int
    skip_existing_any: bool
    failed_cache: set[str]
    max_empty_chunks_per_symbol: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol.replace("/", " ").replace(".", " ")


def parse_bool_flag(value: str | None) -> bool:
    return str(value or "").strip().upper() in {"Y", "YES", "TRUE", "1"}


def unique_symbols(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        sym = normalize_symbol(value)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        result.append(sym)
    return result


def read_symbols_file(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")

    text = file_path.read_text().strip()
    if not text:
        return []

    symbols: list[str] = []
    first_line = text.splitlines()[0]

    if "," in first_line:
        with file_path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            lower = [name.lower() for name in fieldnames]
            if "symbol" not in lower:
                raise ValueError(f"CSV must contain a symbol column: {path}")
            symbol_field = fieldnames[lower.index("symbol")]
            for row in reader:
                value = row.get(symbol_field)
                if value:
                    symbols.append(value)
    else:
        symbols.extend(line.strip() for line in text.splitlines() if line.strip())

    return unique_symbols(symbols)


def download_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec - public symbol list
        return response.read().decode("utf-8", errors="replace")


def load_nasdaq_symbols(include_etfs: bool, include_test_issues: bool) -> list[str]:
    raw = download_text(NASDAQ_LISTED_URL)
    symbols: list[str] = []

    reader = csv.DictReader(raw.splitlines(), delimiter="|")
    for row in reader:
        symbol = (row.get("Symbol") or "").strip()
        if not symbol or symbol.startswith("File Creation Time"):
            continue
        if not include_test_issues and parse_bool_flag(row.get("Test Issue")):
            continue
        if not include_etfs and parse_bool_flag(row.get("ETF")):
            continue
        symbols.append(symbol)

    return unique_symbols(symbols)


def load_us_symbols(include_etfs: bool, include_test_issues: bool) -> list[str]:
    symbols = load_nasdaq_symbols(include_etfs=include_etfs, include_test_issues=include_test_issues)

    raw = download_text(OTHER_LISTED_URL)
    reader = csv.DictReader(raw.splitlines(), delimiter="|")
    for row in reader:
        symbol = (row.get("ACT Symbol") or "").strip()
        if not symbol or symbol.startswith("File Creation Time"):
            continue
        if not include_test_issues and parse_bool_flag(row.get("Test Issue")):
            continue
        if not include_etfs and parse_bool_flag(row.get("ETF")):
            continue
        symbols.append(symbol)

    return unique_symbols(symbols)


def output_path(output_dir: Path, symbol: str) -> Path:
    safe_symbol = symbol.replace(" ", "_").replace("/", "_")
    return output_dir / f"{safe_symbol}.csv"


def existing_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def load_failed_cache(output_dir: Path) -> set[str]:
    failed_path = output_dir / "fetch_1m_failed_symbols.csv"
    if not failed_path.exists():
        return set()
    try:
        df = pd.read_csv(failed_path)
    except Exception:
        return set()
    if "symbol" not in df.columns:
        return set()
    return set(df["symbol"].dropna().astype(str).str.upper())


def append_failed_symbol(output_dir: Path, symbol: str, error: str) -> None:
    failed_path = output_dir / "fetch_1m_failed_symbols.csv"
    exists = failed_path.exists()
    with failed_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "error"])
        if not exists:
            writer.writeheader()
        writer.writerow({"symbol": symbol, "error": error})


def append_no_data_symbol(output_dir: Path, symbol: str, reason: str) -> None:
    no_data_path = output_dir / "fetch_1m_no_data_symbols.csv"
    exists = no_data_path.exists()
    with no_data_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "reason"])
        if not exists:
            writer.writeheader()
        writer.writerow({"symbol": symbol, "reason": reason})


def should_skip_existing(path: Path, days: int, min_rows_per_day: int, force: bool, skip_existing_any: bool) -> bool:
    if force:
        return False
    rows = existing_row_count(path)
    if skip_existing_any and rows > 0:
        return True
    expected_min_rows = max(1, days * min_rows_per_day)
    return rows >= expected_min_rows


def ib_to_dataframe(symbol: str, bars) -> pd.DataFrame:
    columns = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
    if not bars:
        return pd.DataFrame(columns=columns)

    df = util.df(bars)
    if df.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    out["symbol"] = symbol
    out["datetime"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype(str)
    out["open"] = pd.to_numeric(df["open"], errors="coerce")
    out["high"] = pd.to_numeric(df["high"], errors="coerce")
    out["low"] = pd.to_numeric(df["low"], errors="coerce")
    out["close"] = pd.to_numeric(df["close"], errors="coerce")
    out["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype("int64")
    return out.dropna(subset=["open", "high", "low", "close"])


def merge_and_save(symbol: str, output_dir: Path, new_df: pd.DataFrame) -> int:
    path = output_path(output_dir, symbol)

    frames = []
    if path.exists():
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"[WARN] Could not read existing {path}: {exc}")

    if not new_df.empty:
        frames.append(new_df)

    if not frames:
        return 0

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol", "datetime"], keep="last")
    merged = merged.sort_values(["symbol", "datetime"])
    merged.to_csv(path, index=False)
    return len(merged)


def fetch_symbol_1m(ib: IB, symbol: str, cfg: FetchConfig) -> pd.DataFrame:
    contract = Stock(symbol, cfg.exchange, cfg.currency)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError("IBKR contract qualification failed")
    contract = qualified[0]

    all_chunks: list[pd.DataFrame] = []
    empty_chunks = 0
    end = datetime.now(timezone.utc)
    start_limit = end - timedelta(days=cfg.days)

    while end > start_limit:
        chunk_start = max(start_limit, end - timedelta(days=cfg.chunk_days))
        duration_days = max(1, int((end - chunk_start).total_seconds() // 86400) + 1)

        print(f"  chunk {chunk_start.date()} -> {end.date()} duration={duration_days}D", flush=True)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=f"{duration_days} D",
            barSizeSetting="1 min",
            whatToShow=cfg.what_to_show,
            useRTH=cfg.use_rth,
            formatDate=1,
            keepUpToDate=False,
            timeout=cfg.timeout,
        )

        chunk_df = ib_to_dataframe(symbol, bars)
        if not chunk_df.empty:
            all_chunks.append(chunk_df)
            empty_chunks = 0
        else:
            empty_chunks += 1
            if cfg.max_empty_chunks_per_symbol > 0 and empty_chunks >= cfg.max_empty_chunks_per_symbol and not all_chunks:
                print(
                    f"  no data after {empty_chunks} empty chunks; stop symbol early",
                    flush=True,
                )
                append_no_data_symbol(cfg.output_dir, symbol, f"{empty_chunks}_empty_chunks")
                break

        end = chunk_start + timedelta(minutes=1)
        time.sleep(cfg.sleep_seconds)

        if duration_days <= 1 and chunk_start <= start_limit:
            break

    if not all_chunks:
        return pd.DataFrame(columns=["symbol", "datetime", "open", "high", "low", "close", "volume"])

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "datetime"], keep="last")
    combined = combined.sort_values("datetime")
    return combined


def build_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []

    if args.universe:
        if args.universe == "nasdaq":
            symbols.extend(load_nasdaq_symbols(args.include_etfs, args.include_test_issues))
        elif args.universe == "us":
            symbols.extend(load_us_symbols(args.include_etfs, args.include_test_issues))
        else:
            raise ValueError(f"Unsupported universe: {args.universe}")

    if args.symbols:
        symbols.extend(args.symbols)

    if args.symbols_file:
        symbols.extend(read_symbols_file(args.symbols_file))

    symbols = unique_symbols(symbols)

    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch REAL 1m candles from IBKR")
    parser.add_argument("--symbols", nargs="*", help="List of symbols")
    parser.add_argument("--symbols-file", help="File with symbols; one per line or CSV with symbol column")
    parser.add_argument("--universe", choices=["nasdaq", "us"], help="Download symbol list and fetch whole universe")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-symbols", type=int, help="Limit symbol count for smoke tests")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=31)
    parser.add_argument("--exchange", default="SMART")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--what-to-show", default="TRADES")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=1.5, help="Sleep between historical requests")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--include-etfs", action="store_true")
    parser.add_argument("--include-test-issues", action="store_true")
    parser.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Refetch even if local CSV looks complete")
    parser.add_argument(
        "--min-rows-per-day",
        type=int,
        default=250,
        help="Resume heuristic: skip existing files with at least days * this many rows",
    )
    parser.add_argument(
        "--skip-existing-any",
        action="store_true",
        help="Raw data-lake resume mode: skip any symbol that already has a non-empty CSV, even if partial.",
    )
    parser.add_argument(
        "--skip-failed-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip symbols already listed in data/1m/fetch_1m_failed_symbols.csv.",
    )
    parser.add_argument(
        "--max-empty-chunks-per-symbol",
        type=int,
        default=2,
        help="If a symbol returns this many empty chunks before any data, stop trying it early. Use 0 to disable.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    symbols = build_symbols(args)
    if not symbols:
        print("No symbols provided. Use --universe, --symbols, or --symbols-file.", file=sys.stderr)
        return 2

    failed_cache = load_failed_cache(output_dir) if args.skip_failed_cache else set()

    cfg = FetchConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        days=args.days,
        output_dir=output_dir,
        chunk_days=args.chunk_days,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        exchange=args.exchange,
        currency=args.currency,
        what_to_show=args.what_to_show,
        use_rth=args.use_rth,
        force=args.force,
        min_rows_per_day=args.min_rows_per_day,
        skip_existing_any=args.skip_existing_any,
        failed_cache=failed_cache,
        max_empty_chunks_per_symbol=args.max_empty_chunks_per_symbol,
    )

    print("IBKR 1m backfill")
    print(f"Symbols: {len(symbols)}")
    print(f"Days: {cfg.days}")
    print(f"Output: {cfg.output_dir}")
    print(f"IBKR: {cfg.host}:{cfg.port}, client_id={cfg.client_id}")
    print(f"Chunk days: {cfg.chunk_days}, useRTH={cfg.use_rth}, whatToShow={cfg.what_to_show}")
    print(f"skip_existing_any={cfg.skip_existing_any}, failed_cache_symbols={len(cfg.failed_cache)}")
    print(f"max_empty_chunks_per_symbol={cfg.max_empty_chunks_per_symbol}")

    ib = IB()
    ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=cfg.timeout)

    ok = 0
    skipped = 0
    skipped_failed = 0
    failed: list[tuple[str, str]] = []

    try:
        for idx, symbol in enumerate(symbols, start=1):
            if symbol.upper() in cfg.failed_cache and not cfg.force:
                skipped_failed += 1
                print(f"[{idx}/{len(symbols)}] SKIP_FAILED {symbol} -> in failed cache")
                continue

            path = output_path(output_dir, symbol)
            if should_skip_existing(path, cfg.days, cfg.min_rows_per_day, cfg.force, cfg.skip_existing_any):
                skipped += 1
                print(f"[{idx}/{len(symbols)}] SKIP {symbol} -> {path} already has {existing_row_count(path)} rows")
                continue

            print(f"[{idx}/{len(symbols)}] FETCH {symbol}")
            try:
                df = fetch_symbol_1m(ib, symbol, cfg)
                rows = merge_and_save(symbol, output_dir, df)
                ok += 1
                if rows == 0:
                    print(f"  saved {symbol}: 0 rows/no data", flush=True)
                else:
                    print(f"  saved {symbol}: {rows} total rows -> {path}", flush=True)
            except Exception as exc:
                error = str(exc)
                failed.append((symbol, error))
                append_failed_symbol(output_dir, symbol, error)
                print(f"  [FAIL] {symbol}: {exc}", flush=True)
                time.sleep(max(cfg.sleep_seconds, 2.0))
    finally:
        ib.disconnect()

    if failed:
        failed_path = output_dir / "fetch_1m_failed_symbols_last_run.csv"
        pd.DataFrame(failed, columns=["symbol", "error"]).to_csv(failed_path, index=False)
        print(f"Failed symbols from this run saved: {failed_path}")

    print("\nDone")
    print(f"Fetched/updated: {ok}")
    print(f"Skipped existing: {skipped}")
    print(f"Skipped failed cache: {skipped_failed}")
    print(f"Failed this run: {len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
