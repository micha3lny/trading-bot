from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
try:
    from ib_insync import IB, Stock
except ImportError:  # pragma: no cover - tests can exercise planning without IBKR deps
    class _MissingIbInsync:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("ib_insync is required for live IBKR collection")

    IB = Stock = _MissingIbInsync

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 168
DEFAULT_ALPHA_RANK = "data/universe/v68_final_daytrading_universe.csv"
DEFAULT_OUTPUT_DIR = "data/history/universe_1m"
DEFAULT_STATUS_DIR = "data/history"
DEFAULT_LOCK_PATH = "data/runtime/history_collector.lock"
LOCK_HELD_EXIT_CODE = 75

RTH_START_UTC = "13:30"
RTH_END_UTC = "20:00"


@dataclass(frozen=True)
class CollectTask:
    symbol: str
    session_date: date
    session_type: str


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def read_lock_info(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip().replace("\n", " | ")
    except Exception:
        return ""


def acquire_lock(path: Path) -> Any | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except Exception:
        handle.close()
        raise
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired_at={now_iso()}\n")
    handle.flush()
    return handle


def release_lock(handle: Any | None) -> None:
    if handle is None:
        return
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


def load_universe(alpha_rank_csv: str, limit: int | None = None) -> list[str]:
    path = Path(alpha_rank_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing universe file: {alpha_rank_csv}")
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError("Universe CSV must contain symbol column")
    if "alpha_score" in df.columns:
        df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("alpha_score", ascending=False)
    symbols = df["symbol"].astype(str).str.upper().str.strip().dropna().drop_duplicates().tolist()
    symbols = [s for s in symbols if s]
    if limit:
        symbols = symbols[: int(limit)]
    return symbols


def date_range(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def in_utc_window(start_hhmm: str, end_hhmm: str, now: datetime | None = None) -> bool:
    now = now or now_utc()
    sh, sm = [int(x) for x in start_hhmm.split(":", 1)]
    eh, em = [int(x) for x in end_hhmm.split(":", 1)]
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    if end <= start:
        return now >= start or now < end
    return start <= now < end


def is_trading_window(now: datetime | None = None) -> bool:
    return in_utc_window(RTH_START_UTC, RTH_END_UTC, now)


def parquet_path(output_dir: Path, symbol: str, session_date: date, session_type: str) -> Path:
    return (
        output_dir
        / f"session_type={session_type.upper()}"
        / f"symbol={symbol.upper()}"
        / f"year={session_date.year:04d}"
        / f"month={session_date.month:02d}"
        / f"day={session_date.day:02d}.parquet"
    )


def task_key(task: CollectTask) -> str:
    return f"{task.symbol}_{task.session_date.isoformat()}_{task.session_type.upper()}"


def status_is_complete(status: dict[str, Any], task: CollectTask) -> bool:
    row = status.get(task_key(task)) or {}
    return row.get("status") == "complete"


def parquet_exists(output_dir: Path, task: CollectTask) -> bool:
    path = parquet_path(output_dir, task.symbol, task.session_date, task.session_type)
    return path.exists() and path.stat().st_size > 0


def collect_existing_parquet_keys(output_dir: Path, session_type: str) -> set[str]:
    root = output_dir / f"session_type={session_type.upper()}"
    if not root.exists():
        return set()
    keys: set[str] = set()
    for path in root.glob("symbol=*/year=*/month=*/day=*.parquet"):
        try:
            if path.stat().st_size <= 0:
                continue
            symbol = path.parent.parent.parent.name.split("=", 1)[1].upper()
            year = int(path.parent.parent.name.split("=", 1)[1])
            month = int(path.parent.name.split("=", 1)[1])
            day = int(path.stem.split("=", 1)[1])
            keys.add(f"{symbol}_{date(year, month, day).isoformat()}_{session_type.upper()}")
        except Exception:
            continue
    return keys


def count_existing_parquets_for_tasks(output_dir: Path, tasks: list[CollectTask]) -> int:
    return sum(1 for task in tasks if parquet_exists(output_dir, task))


def completion_pct(done: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return round((float(done) / float(total)) * 100.0, 2)


def task_is_complete(
    status: dict[str, Any],
    output_dir: Path,
    task: CollectTask,
    *,
    existing_parquet_keys: set[str] | None = None,
) -> bool:
    row = status.get(task_key(task)) or {}
    if row.get("status") == "complete":
        return True
    if row.get("status") in {"partial", "failed", "failed_permanent", "no_data", "no_data_permanent"}:
        return False
    if existing_parquet_keys is not None and task_key(task) in existing_parquet_keys:
        return True
    return parquet_exists(output_dir, task)


def status_attempts(failures: dict[str, Any], task: CollectTask) -> int:
    row = failures.get(task_key(task)) or {}
    try:
        return int(row.get("attempts", 0))
    except Exception:
        return 0


def mark_status(status: dict[str, Any], task: CollectTask, state: str, rows: int = 0, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "symbol": task.symbol,
        "date": task.session_date.isoformat(),
        "session_type": task.session_type.upper(),
        "status": state,
        "rows": int(rows),
        "updated_at": now_iso(),
    }
    if extra:
        payload.update(extra)
    status[task_key(task)] = payload


def mark_failure(failures: dict[str, Any], task: CollectTask, error: str) -> int:
    key = task_key(task)
    row = failures.get(key) or {
        "symbol": task.symbol,
        "date": task.session_date.isoformat(),
        "session_type": task.session_type.upper(),
        "attempts": 0,
    }
    row["attempts"] = int(row.get("attempts", 0)) + 1
    row["last_error"] = str(error)[:1000]
    row["last_attempt_at"] = now_iso()
    failures[key] = row
    return int(row["attempts"])


def bars_to_df(symbol: str, bars: list[Any], session_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cutoff = now_utc() - timedelta(minutes=2)
    for bar in bars:
        raw_dt = getattr(bar, "date", None)
        if isinstance(raw_dt, datetime):
            ts = raw_dt
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts = ts.astimezone(timezone.utc)
        else:
            parsed = pd.to_datetime(str(raw_dt), errors="coerce", utc=True)
            if pd.isna(parsed):
                continue
            ts = parsed.to_pydatetime()
        if ts >= cutoff:
            continue
        rows.append(
            {
                "symbol": symbol.upper(),
                "bar_time_utc": ts.isoformat(),
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "session_type": session_type.upper(),
                "source": "ibkr_reqHistoricalData_1m",
                "downloaded_at": now_iso(),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["symbol", "bar_time_utc"]).sort_values("bar_time_utc")
    return df


def expected_min_rows(session_type: str) -> int:
    # RTH has 390 one-minute bars. Keep threshold loose to avoid false failures on halts/early closes.
    if session_type.upper() == "RTH":
        return 180
    return 1


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp, index=False)
    except ImportError as exc:
        raise RuntimeError("Missing parquet engine. Install pyarrow in venv: pip install pyarrow") from exc
    tmp.replace(path)


def collect_one(ib: IB, task: CollectTask, output_dir: Path, pause_seconds: float) -> tuple[str, int, str | None]:
    symbol = task.symbol.upper()
    contract = Stock(symbol, "SMART", "USD")
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            contract = qualified[0]
    except Exception as exc:
        return "failed", 0, f"qualify_failed: {exc!r}"

    # Ask IBKR for the requested session by using endDateTime as the next UTC midnight after the session date.
    # For RTH US stocks this returns the target date's regular session with useRTH=True.
    end_dt = datetime.combine(task.session_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S UTC")
    use_rth = task.session_type.upper() == "RTH"

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_str,
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception as exc:
        return "failed", 0, f"historical_failed: {exc!r}"

    df = bars_to_df(symbol, list(bars), task.session_type)
    ib.sleep(float(pause_seconds))

    if df.empty:
        return "no_data", 0, "empty_bars"

    rows = len(df)
    if rows < expected_min_rows(task.session_type):
        # Save partial data once, but mark partial so a future retry can attempt again.
        write_parquet(parquet_path(output_dir, symbol, task.session_date, task.session_type), df)
        return "partial", rows, f"rows_below_threshold rows={rows}"

    write_parquet(parquet_path(output_dir, symbol, task.session_date, task.session_type), df)
    return "complete", rows, None


def build_tasks(symbols: list[str], start: date, end: date, session_type: str, *, include_weekends: bool = False) -> list[CollectTask]:
    dates = [d for d in date_range(start, end) if include_weekends or d.weekday() < 5]
    return [CollectTask(symbol=s, session_date=d, session_type=session_type.upper()) for d in dates for s in symbols]


def build_pending_tasks(
    tasks: list[CollectTask],
    *,
    status: dict[str, Any],
    failures: dict[str, Any],
    output_dir: Path,
    max_attempts: int,
    retry_failed: bool,
    existing_parquet_keys: set[str] | None = None,
    sync_existing_status: bool = False,
    update_blocked_status: bool = True,
) -> tuple[list[CollectTask], dict[str, int]]:
    pending: list[CollectTask] = []
    stats = {"tasks": len(tasks), "complete": 0, "pending": 0, "blocked_by_attempts": 0, "synced_existing": 0}
    for task in tasks:
        key = task_key(task)
        status_row = status.get(key) or {}
        complete = task_is_complete(status, output_dir, task, existing_parquet_keys=existing_parquet_keys)
        if complete:
            stats["complete"] += 1
            if (
                sync_existing_status
                and update_blocked_status
                and status_row.get("status") != "complete"
                and existing_parquet_keys is not None
                and key in existing_parquet_keys
            ):
                mark_status(status, task, "complete", int(status_row.get("rows", 0) or 0), {"synced_from_parquet": True})
                stats["synced_existing"] += 1
            continue
        attempts = status_attempts(failures, task)
        if attempts >= int(max_attempts) and not retry_failed:
            if update_blocked_status:
                mark_status(status, task, "failed_permanent", 0, {"attempts": attempts})
            stats["blocked_by_attempts"] += 1
            continue
        pending.append(task)
    stats["pending"] = len(pending)
    return pending, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v68 full-universe 1m Parquet collector for IBKR")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--alpha-rank-csv", default=DEFAULT_ALPHA_RANK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-dir", default=DEFAULT_STATUS_DIR)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--session-type", choices=["RTH", "EXT"], default="RTH")
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--batch-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--allow-outside-window", action="store_true")
    parser.add_argument("--postmarket-start-utc", default="20:15")
    parser.add_argument("--postmarket-end-utc", default="15:00")
    parser.add_argument("--premarket-start-utc", default="20:15")
    parser.add_argument("--premarket-end-utc", default="15:00")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--include-weekends", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sync-status-from-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    args = parser.parse_args()
    started_monotonic = time.monotonic()

    current = now_utc()
    if not args.allow_outside_window:
        allowed = (
            in_utc_window(args.postmarket_start_utc, args.postmarket_end_utc, current)
            or in_utc_window(args.premarket_start_utc, args.premarket_end_utc, current)
        )
        if is_trading_window(current) or not allowed:
            print(
                f"{now_iso()} HISTORY_COLLECTOR_SKIPPED reason=outside_allowed_window "
                f"now={current.isoformat()} postmarket={args.postmarket_start_utc}-{args.postmarket_end_utc} "
                f"premarket={args.premarket_start_utc}-{args.premarket_end_utc}",
                flush=True,
            )
            return 0

    if args.date:
        start = end = parse_date(args.date)
    else:
        start = parse_date(args.start_date)
        end = parse_date(args.end_date) if args.end_date else current.date()

    output_dir = Path(args.output_dir)
    status_dir = Path(args.status_dir)
    status_path = status_dir / "collector_status.json"
    failures_path = status_dir / "collector_failures.json"
    lock_handle = acquire_lock(Path(args.lock_path))
    if lock_handle is None:
        lock_info = read_lock_info(Path(args.lock_path))
        print(
            f"{now_iso()} HISTORY_COLLECTOR_SKIPPED reason=lock_held lock_path={args.lock_path} "
            f"lock_info={lock_info!r}",
            flush=True,
        )
        return LOCK_HELD_EXIT_CODE

    try:
        status: dict[str, Any] = load_json(status_path, {})
        failures: dict[str, Any] = load_json(failures_path, {})

        symbols = load_universe(args.alpha_rank_csv, args.limit_symbols)
        tasks = build_tasks(symbols, start, end, args.session_type, include_weekends=bool(args.include_weekends))
        existing_parquet_keys = collect_existing_parquet_keys(output_dir, args.session_type)
        pending, plan = build_pending_tasks(
            tasks,
            status=status,
            failures=failures,
            output_dir=output_dir,
            max_attempts=int(args.max_attempts),
            retry_failed=bool(args.retry_failed),
            existing_parquet_keys=existing_parquet_keys,
            sync_existing_status=bool(args.sync_status_from_parquet),
            update_blocked_status=not bool(args.plan_only),
        )

        if args.max_tasks and args.max_tasks > 0:
            pending = pending[: int(args.max_tasks)]

        print(
            f"{now_iso()} HISTORY_COLLECTOR_START symbols={len(symbols)} total_symbols={len(symbols)} tasks={len(tasks)} "
            f"complete={plan['complete']} skipped_existing={plan['complete']} "
            f"synced_existing={plan['synced_existing']} "
            f"missing={plan['pending']} blocked_by_attempts={plan['blocked_by_attempts']} "
            f"pending={len(pending)} start={start} end={end} session={args.session_type} "
            f"include_weekends={bool(args.include_weekends)} output_dir={output_dir}",
            flush=True,
        )

        if args.plan_only:
            write_json_atomic(status_path, status)
            parquet_files = count_existing_parquets_for_tasks(output_dir, tasks)
            duration = int(time.monotonic() - started_monotonic)
            print(
                f"{now_iso()} HISTORY_COLLECTOR_OUTPUT_SUMMARY expected_symbols={len(tasks)} "
                f"parquet_files={parquet_files} completion_pct={completion_pct(parquet_files, len(tasks))}",
                flush=True,
            )
            print(
                f"{now_iso()} HISTORY_COLLECTOR_DONE duration_seconds={duration} total_symbols={len(symbols)} "
                f"processed=0 skipped_existing={plan['complete']} complete=0 partial=0 no_data=0 "
                f"failed=0 retries=0 parquet_files_written=0 output_dir={output_dir}",
                flush=True,
            )
            return 0

        if not pending:
            write_json_atomic(status_path, status)
            write_json_atomic(failures_path, failures)
            parquet_files = count_existing_parquets_for_tasks(output_dir, tasks)
            duration = int(time.monotonic() - started_monotonic)
            print(
                f"{now_iso()} HISTORY_COLLECTOR_OUTPUT_SUMMARY expected_symbols={len(tasks)} "
                f"parquet_files={parquet_files} completion_pct={completion_pct(parquet_files, len(tasks))}",
                flush=True,
            )
            print(
                f"{now_iso()} HISTORY_COLLECTOR_DONE duration_seconds={duration} total_symbols={len(symbols)} "
                f"processed=0 skipped_existing={plan['complete']} complete=0 partial=0 no_data=0 "
                f"failed=0 retries=0 parquet_files_written=0 output_dir={output_dir}",
                flush=True,
            )
            return 0

        ib = IB()
        ib.connect(args.host, int(args.port), clientId=int(args.client_id), timeout=20)

        completed = partial = failed = no_data = 0
        last_progress_idx = 0
        last_progress_monotonic = started_monotonic
        try:
            for idx, task in enumerate(pending, 1):
                try:
                    state, rows, error = collect_one(ib, task, output_dir, float(args.request_sleep_seconds))
                except Exception as exc:
                    state, rows, error = "failed", 0, f"unexpected: {exc!r}"
                    tb = traceback.format_exc(limit=3).replace("\n", " | ")
                    print(
                        f"{now_iso()} HISTORY_SYMBOL_EXCEPTION {task.symbol} {task.session_date} "
                        f"error={exc!r} traceback={tb}",
                        flush=True,
                    )

                if state == "complete":
                    completed += 1
                    failures.pop(task_key(task), None)
                    mark_status(status, task, "complete", rows)
                    print(f"{now_iso()} HISTORY_SYMBOL_OK {task.symbol} {task.session_date} rows={rows}", flush=True)
                elif state == "partial":
                    partial += 1
                    attempts = mark_failure(failures, task, error or "partial")
                    mark_status(status, task, "partial", rows, {"attempts": attempts, "last_error": error})
                    print(f"{now_iso()} HISTORY_SYMBOL_PARTIAL {task.symbol} {task.session_date} rows={rows} attempts={attempts}", flush=True)
                elif state == "no_data":
                    no_data += 1
                    attempts = mark_failure(failures, task, error or "no_data")
                    terminal = attempts >= int(args.max_attempts)
                    mark_status(status, task, "no_data_permanent" if terminal else "no_data", rows, {"attempts": attempts, "last_error": error})
                    print(f"{now_iso()} HISTORY_SYMBOL_NO_DATA {task.symbol} {task.session_date} attempts={attempts}", flush=True)
                else:
                    failed += 1
                    attempts = mark_failure(failures, task, error or "failed")
                    terminal = attempts >= int(args.max_attempts)
                    mark_status(status, task, "failed_permanent" if terminal else "failed", rows, {"attempts": attempts, "last_error": error})
                    print(f"{now_iso()} HISTORY_SYMBOL_FAILED {task.symbol} {task.session_date} attempts={attempts} error={error}", flush=True)

                progress_elapsed = time.monotonic() - last_progress_monotonic
                if idx == len(pending) or idx - last_progress_idx >= 100 or progress_elapsed >= 60.0:
                    print(
                        f"{now_iso()} HISTORY_COLLECTOR_PROGRESS processed={idx}/{len(pending)} "
                        f"complete={completed} partial={partial} no_data={no_data} failed={failed} "
                        f"elapsed_seconds={int(time.monotonic() - started_monotonic)}",
                        flush=True,
                    )
                    last_progress_idx = idx
                    last_progress_monotonic = time.monotonic()

                if idx % 10 == 0:
                    write_json_atomic(status_path, status)
                    write_json_atomic(failures_path, failures)
                    print(
                        f"{now_iso()} HISTORY_PROGRESS_SAVED idx={idx} pending={len(pending)} "
                        f"complete={completed} partial={partial} no_data={no_data} failed={failed}",
                        flush=True,
                    )

                if args.batch_size and idx % int(args.batch_size) == 0:
                    print(f"{now_iso()} HISTORY_BATCH_SLEEP idx={idx} sleep={args.batch_sleep_seconds}", flush=True)
                    ib.sleep(float(args.batch_sleep_seconds))

        finally:
            write_json_atomic(status_path, status)
            write_json_atomic(failures_path, failures)
            ib.disconnect()

        parquet_files = count_existing_parquets_for_tasks(output_dir, tasks)
        duration = int(time.monotonic() - started_monotonic)
        parquet_files_written = completed + partial
        print(
            f"{now_iso()} HISTORY_COLLECTOR_OUTPUT_SUMMARY expected_symbols={len(tasks)} "
            f"parquet_files={parquet_files} completion_pct={completion_pct(parquet_files, len(tasks))}",
            flush=True,
        )
        print(
            f"{now_iso()} HISTORY_COLLECTOR_DONE duration_seconds={duration} total_symbols={len(symbols)} processed={len(pending)} "
            f"skipped_existing={plan['complete']} complete={completed} partial={partial} "
            f"no_data={no_data} failed={failed} retries={partial + no_data + failed} "
            f"parquet_files_written={parquet_files_written} output_dir={output_dir}",
            flush=True,
        )
        return 0
    finally:
        release_lock(lock_handle)


if __name__ == "__main__":
    raise SystemExit(main())
