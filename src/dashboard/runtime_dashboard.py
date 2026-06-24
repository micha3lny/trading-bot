from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLITE_BUSY_TIMEOUT_MS, configure_sqlite_connection  # noqa: E402

from src.dashboard.runtime_queries import (  # noqa: E402
    DateWindow,
    aggregate_closed_positions,
    list_sessions,
    list_strategies,
    load_dashboard_snapshot,
    utc_today,
)
from src.dashboard.broker_reality import (  # noqa: E402
    ReconciliationResult,
    describe_asyncio_event_loop,
    empty_broker_executions,
    empty_closed_trades,
    fetch_ibkr_executions_diagnostic,
    fetch_ibkr_live_portfolio,
    closed_trades_from_commission_reports,
    load_sqlite_active_positions,
    load_sqlite_closed_trades,
    load_sqlite_executions,
    load_sqlite_trade_pnl,
    parse_ibkr_activity_csv,
    reconstruct_closed_trades_fifo,
    reconcile_broker_vs_sqlite,
    reconciliation_export_frames,
)
from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, resolve_sqlite_path  # noqa: E402
from src.live_trading.market_calendar import previous_us_equity_trading_day  # noqa: E402


st.set_page_config(
    page_title="Trading Runtime Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f14;
    color: #e5edf5;
}
[data-testid="stSidebar"] {
    background: #111821;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}
[data-testid="stMetric"] {
    background: #111821;
    border: 1px solid #223041;
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #223041;
    border-radius: 8px;
}
div[data-testid="stDataFrame"] [role="columnheader"] {
    position: sticky;
    top: 0;
    z-index: 2;
}
.small-note {
    color: #8fa3b8;
    font-size: 0.85rem;
}
</style>
"""


def is_current_window(window: DateWindow) -> bool:
    today = utc_today()
    return window.start_date <= today <= window.end_date and window.start_date == window.end_date


@st.cache_data(ttl=5)
def load_live_snapshot(
    sqlite_path: str,
    start_date: str,
    end_date: str,
    strategy: str,
    include_reconstructed: bool,
    broker_portfolio_records: tuple[dict, ...] = (),
) -> dict:
    broker_portfolio = pd.DataFrame(list(broker_portfolio_records))
    return load_dashboard_snapshot(
        sqlite_path,
        DateWindow(start_date, end_date),
        strategy,
        include_reconstructed=include_reconstructed,
        broker_portfolio=broker_portfolio,
    )


@st.cache_data(ttl=3600)
def load_historical_snapshot(sqlite_path: str, start_date: str, end_date: str, strategy: str, include_reconstructed: bool) -> dict:
    return load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), strategy, include_reconstructed=include_reconstructed)


@st.cache_data(ttl=15)
def cached_sessions(sqlite_path: str) -> list[str]:
    return list_sessions(sqlite_path)


@st.cache_data(ttl=15)
def cached_strategies(sqlite_path: str, start_date: str, end_date: str) -> list[str]:
    return list_strategies(sqlite_path, DateWindow(start_date, end_date))


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:,.1f}%"


def display_time(value) -> str:
    if value in (None, ""):
        return "MISSING"
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "MISSING"
    return ts.strftime("%d-%m-%Y %H:%M:%S")


def display_optional_number(value) -> object:
    if value in (None, ""):
        return "MISSING"
    try:
        if pd.isna(value):
            return "MISSING"
    except Exception:
        pass
    return value


def display_number_or_missing(value, *, decimals: int = 6) -> str:
    if value in (None, ""):
        return "MISSING"
    try:
        if pd.isna(value):
            return "MISSING"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def status_badge(label: str) -> None:
    normalized = str(label or "UNKNOWN").upper()
    if normalized == "OK":
        st.success(normalized)
    elif normalized in {"UNKNOWN", "PARTIAL", "STALE"}:
        st.warning(normalized)
    else:
        st.error(normalized)


def sqlite_connect_readonly(sqlite_path: str) -> sqlite3.Connection:
    path = Path(sqlite_path)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=max(1.0, SQLITE_BUSY_TIMEOUT_MS / 1000.0))
    conn.row_factory = sqlite3.Row
    return configure_sqlite_connection(conn, read_only=True)


def sqlite_scalar(sqlite_path: str, sql: str, params: list[object] | None = None, default: object = 0) -> object:
    try:
        with sqlite_connect_readonly(sqlite_path) as conn:
            row = conn.execute(sql, params or []).fetchone()
            if row is None:
                return default
            return row[0]
    except Exception:
        return default


def sqlite_row(sqlite_path: str, sql: str, params: list[object] | None = None) -> dict[str, object]:
    try:
        with sqlite_connect_readonly(sqlite_path) as conn:
            row = conn.execute(sql, params or []).fetchone()
            return dict(row) if row is not None else {}
    except Exception:
        return {}


def parse_raw_json(value: object) -> dict[str, object]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def parquet_files_for_session(history_dir: str | Path, session_date: str, session_type: str = "RTH") -> list[Path]:
    dt = pd.to_datetime(session_date).date()
    root = Path(history_dir) / f"session_type={session_type.upper()}"
    return sorted(root.glob(f"symbol=*/year={dt.year:04d}/month={dt.month:02d}/day={dt.day:02d}.parquet"))


def history_parquet_path(history_dir: str | Path, symbol: str, session_date: str, session_type: str = "RTH") -> Path:
    dt = pd.to_datetime(session_date).date()
    return (
        Path(history_dir)
        / f"session_type={session_type.upper()}"
        / f"symbol={str(symbol).upper()}"
        / f"year={dt.year:04d}"
        / f"month={dt.month:02d}"
        / f"day={dt.day:02d}.parquet"
    )


def history_task_key(symbol: str, session_date: str, session_type: str = "RTH") -> str:
    return f"{str(symbol).upper()}_{session_date}_{session_type.upper()}"


def load_json_file(path: str | Path, default: object) -> object:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_universe_symbols(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if "symbol" not in df.columns:
        return []
    return (
        df["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def load_history_readiness_summary(
    *,
    history_dir: str | Path,
    universe_path: str | Path,
    status_dir: str | Path,
    session_date: str,
    session_type: str = "RTH",
) -> dict[str, object]:
    symbols = load_universe_symbols(universe_path)
    status_rows = load_json_file(Path(status_dir) / "collector_status.json", {})
    if not isinstance(status_rows, dict):
        status_rows = {}
    complete = partial = no_data = failed = missing = 0
    for symbol in symbols:
        path = history_parquet_path(history_dir, symbol, session_date, session_type)
        row = status_rows.get(history_task_key(symbol, session_date, session_type)) or {}
        row_status = str(row.get("status") or "").lower() if isinstance(row, dict) else ""
        if path.exists() and path.stat().st_size > 0:
            complete += 1
        elif row_status == "complete":
            complete += 1
        elif row_status == "partial":
            partial += 1
        elif row_status in {"no_data", "no_data_permanent"}:
            no_data += 1
        elif row_status in {"failed", "failed_permanent"}:
            failed += 1
        else:
            missing += 1
    expected = len(symbols)
    terminal = complete + no_data
    completion_pct = round((terminal / expected) * 100.0, 2) if expected else 0.0
    ready = expected > 0 and missing == 0 and partial == 0 and failed == 0
    status = "OK" if ready else ("PARTIAL" if terminal or partial or failed else "MISSING")
    return {
        "expected_symbols": expected,
        "complete_symbols": complete,
        "partial_symbols": partial,
        "no_data_symbols": no_data,
        "failed_symbols": failed,
        "missing_symbols": missing,
        "terminal_symbols": terminal,
        "completion_pct": completion_pct,
        "status": status,
    }


def load_top100_diagnostics_summary(output_dir: str | Path, session_date: str) -> dict[str, object]:
    path = Path(output_dir) / f"daily_top100_{session_date}_diagnostics.csv"
    if not path.exists():
        return {
            "path": str(path),
            "rows": 0,
            "missing": 0,
            "rejected": 0,
            "error": 0,
            "excluded_ineligible": 0,
            "status": "MISSING",
        }
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"path": str(path), "rows": 0, "missing": 0, "rejected": 0, "error": 0, "excluded_ineligible": 0, "status": "ERROR"}
    statuses = df.get("status", pd.Series(dtype=str)).astype(str).str.lower()
    return {
        "path": str(path),
        "rows": int(len(df)),
        "missing": int((statuses == "missing").sum()),
        "rejected": int((statuses == "rejected").sum()),
        "error": int((statuses == "error").sum()),
        "excluded_ineligible": int((statuses == "excluded_ineligible").sum()),
        "status": "OK",
    }


def infer_top100_source_date(latest_path: str | Path) -> str:
    latest = Path(latest_path)
    if not latest.exists():
        return "MISSING"
    try:
        latest_digest = hashlib.sha256(latest.read_bytes()).hexdigest()
    except Exception:
        latest_digest = ""
    exact_matches: list[str] = []
    candidates: list[tuple[float, str]] = []
    for path in latest.parent.glob("daily_top100_*.csv"):
        if path.name == latest.name or path.name.endswith("_diagnostics.csv"):
            continue
        stem = path.stem.replace("daily_top100_", "")
        if len(stem) == 10:
            try:
                pd.to_datetime(stem)
            except Exception:
                continue
            if latest_digest:
                try:
                    if hashlib.sha256(path.read_bytes()).hexdigest() == latest_digest:
                        exact_matches.append(stem)
                except Exception:
                    pass
            delta = abs(path.stat().st_mtime - latest.stat().st_mtime)
            candidates.append((delta, stem))
    if exact_matches:
        return sorted(exact_matches)[-1]
    if not candidates:
        return "UNKNOWN"
    return sorted(candidates, key=lambda item: item[0])[0][1]


def load_top100_readiness(
    latest_path: str | Path,
    *,
    expected_symbols: int = 100,
    expected_source_session_date: str | None = None,
) -> dict[str, object]:
    path = Path(latest_path)
    if not path.exists():
        return {
            "path": str(path),
            "generated_at": "MISSING",
            "modified_at": "MISSING",
            "top100_source_session_date": "MISSING",
            "expected_source_session_date": expected_source_session_date or "UNKNOWN",
            "symbols": 0,
            "status": "MISSING",
        }
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    try:
        df = pd.read_csv(path)
        symbols = int(df["symbol"].dropna().astype(str).str.strip().ne("").sum()) if "symbol" in df.columns else int(len(df))
    except Exception:
        symbols = 0
    source_date = infer_top100_source_date(path)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600.0
    has_enough_symbols = symbols >= expected_symbols
    source_matches_expected = (
        not expected_source_session_date
        or source_date == expected_source_session_date
    )
    if not has_enough_symbols:
        status = "PARTIAL"
    elif not source_matches_expected:
        status = "STALE"
    elif age_hours > 36:
        status = "STALE"
    else:
        status = "OK"
    return {
        "path": str(path),
        "generated_at": modified.isoformat(),
        "modified_at": modified.strftime("%d-%m-%Y %H:%M:%S UTC"),
        "top100_source_session_date": source_date,
        "expected_source_session_date": expected_source_session_date or "UNKNOWN",
        "symbols": symbols,
        "status": status,
        "source_matches_expected": source_matches_expected,
    }


def load_eod_readiness(
    sqlite_path: str,
    session_date: str,
    *,
    broker_open_count: int | None = None,
    sqlite_active_count: int | None = None,
) -> dict[str, object]:
    final = sqlite_row(
        sqlite_path,
        """
        SELECT event_time, event_type, reason, raw_json
        FROM runtime_events
        WHERE event_type = 'EOD_FINAL_STATUS'
          AND substr(event_time, 1, 10) <= ?
        ORDER BY event_time DESC
        LIMIT 1
        """,
        [session_date],
    )
    flatten = sqlite_row(
        sqlite_path,
        """
        SELECT event_time, event_type, reason, raw_json
        FROM runtime_events
        WHERE event_type LIKE 'EOD_FLATTEN%'
          AND substr(event_time, 1, 10) <= ?
        ORDER BY event_time DESC
        LIMIT 1
        """,
        [session_date],
    )
    verified = sqlite_scalar(
        sqlite_path,
        """
        SELECT COUNT(*)
        FROM runtime_events
        WHERE event_type IN ('POSITION_VERIFIED_CLOSED', 'EOD_FLATTEN_SUCCESS')
          AND substr(event_time, 1, 10) = ?
        """,
        [session_date],
        0,
    )
    final_raw = parse_raw_json(final.get("raw_json"))
    clean = final_raw.get("clean")
    if clean is None:
        clean = 1 if str(final.get("reason") or "").lower() in {"clean", "eod_success"} else None
    final_clean = clean in {1, True, "1", "true", "True"}
    flat_confirmed = broker_open_count == 0 and sqlite_active_count == 0
    if final_clean or flat_confirmed:
        status = "OK"
    else:
        status = "FAILED" if final else "UNKNOWN"
    return {
        "last_flatten_event": flatten.get("event_type") or "UNKNOWN",
        "last_flatten_at": flatten.get("event_time") or "UNKNOWN",
        "last_final_status": final.get("event_type") or "UNKNOWN",
        "last_final_at": final.get("event_time") or "UNKNOWN",
        "positions_verified_closed": int(verified or 0),
        "status": status,
        "final_clean": final_clean,
        "broker_open_count": broker_open_count,
        "sqlite_active_count": sqlite_active_count,
        "flat_confirmed": flat_confirmed,
        "raw": final_raw,
    }


def post_json(url: str, payload: dict[str, object], timeout: float = 8.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0, repr(exc)


def get_json(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0, repr(exc)


def run_dashboard_command(command: list[str], timeout: int = 900) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
        return int(proc.returncode), output.strip()
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (("\nSTDERR:\n" + exc.stderr) if exc.stderr else "")
        return 124, f"COMMAND_TIMEOUT timeout_seconds={timeout}\n{output}".strip()
    except Exception as exc:
        return 1, repr(exc)


def style_pnl(df: pd.DataFrame, pnl_columns: list[str]):
    def color(value):
        try:
            v = float(value)
        except Exception:
            return ""
        if v > 0:
            return "color: #4ade80"
        if v < 0:
            return "color: #fb7185"
        return "color: #cbd5e1"

    existing = [col for col in pnl_columns if col in df.columns]
    return df.style.map(color, subset=existing) if existing else df


def filter_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    cols = st.columns([1, 1, 2])
    symbols = sorted(str(x) for x in df.get("symbol", pd.Series(dtype=str)).dropna().unique())
    strategies = sorted(str(x) for x in df.get("strategy", pd.Series(dtype=str)).dropna().unique())
    selected_symbols = cols[0].multiselect("Symbol", symbols, key=f"{prefix}_symbols")
    selected_strategies = cols[1].multiselect("Strategy", strategies, key=f"{prefix}_strategies")
    search = cols[2].text_input("Search", key=f"{prefix}_search")
    out = df.copy()
    if selected_symbols:
        out = out[out["symbol"].astype(str).isin(selected_symbols)]
    if selected_strategies and "strategy" in out.columns:
        out = out[out["strategy"].astype(str).isin(selected_strategies)]
    if search:
        needle = search.lower()
        out = out[out.astype(str).apply(lambda row: row.str.lower().str.contains(needle, regex=False).any(), axis=1)]
    return out


def render_summary(summary: dict) -> None:
    row1 = st.columns(6)
    row1[0].metric("Gross PnL", money(summary.get("gross_pnl", 0.0)))
    row1[1].metric("Net Actual PnL", money(summary.get("net_actual_pnl", 0.0)))
    row1[2].metric("Open UPNL", money(summary.get("open_upnl", 0.0)))
    row1[3].metric("Total PnL", money(summary.get("total_pnl", 0.0)))
    row1[4].metric("Win Rate", pct(summary.get("win_rate", 0.0)))
    row1[5].metric("Expectancy", money(summary.get("expectancy", 0.0)))
    row2 = st.columns(5)
    row2[0].metric("Avg Peak", pct(summary.get("avg_peak", 0.0)))
    row2[1].metric("Avg Giveback", pct(summary.get("avg_giveback", 0.0)))
    row2[2].metric("Commissions", money(summary.get("commissions", 0.0)))
    row2[3].metric("Closed Trades", int(summary.get("closed_trades", 0)))
    row2[4].metric("Open Trades", int(summary.get("open_trades", 0)))
    source = summary.get("main_pnl_source") or summary.get("closed_pnl_source") or ""
    if source:
        st.caption(f"main_pnl_source={source}")


def render_data_quality_summary(summary: dict) -> None:
    st.subheader("Closed Trade Data Quality")
    cols = st.columns(8)
    cols[0].metric("Closed", int(summary.get("closed_trades_count", 0)))
    cols[1].metric("Comm OK", int(summary.get("commission_ok", 0)))
    cols[2].metric("Comm Partial", int(summary.get("commission_partial", 0)))
    cols[3].metric("Comm Missing", int(summary.get("commission_missing", 0)))
    cols[4].metric("Peak Missing", int(summary.get("peak_missing", 0)))
    cols[5].metric("MAE Missing", int(summary.get("mae_missing", 0)))
    cols[6].metric("Peak Price Missing", int(summary.get("peak_price_missing", 0)))
    cols[7].metric("Warnings", int(summary.get("data_quality_warning_count", 0)))


def render_open_positions(df: pd.DataFrame, *, title: str = "Open Positions", prefix: str = "open") -> None:
    if title:
        st.subheader(title)
    if df.empty:
        st.info("No open positions in the selected window.")
        return
    cols = [
        "symbol", "qty", "entry_time", "buy", "now", "now_dollars", "now_pct", "peak_pct",
        "giveback_pct", "top100_rank", "top100_score", "live_entry_score", "live_entry_rank",
        "entry_order_id", "entry_perm_id",
        "status", "strategy", "data_quality", "entry_metadata_status", "ibkr_confirmed",
        "price_status", "now_price_source", "market_price_at", "last_update", "source", "exit_sent", "execution_ids",
    ]
    available = [col for col in cols if col in df.columns]
    out = filter_table(df[available].copy(), prefix).sort_values(["now_dollars", "symbol"], na_position="last")
    if len(out) != len(df):
        st.warning(f"Table filters are hiding {len(df) - len(out)} of {len(df)} open-position rows.")
    out["entry_time"] = out["entry_time"].map(display_time)
    if "last_update" in out.columns:
        out["last_update"] = out["last_update"].map(display_time)
    if "market_price_at" in out.columns:
        out["market_price_at"] = out["market_price_at"].map(display_time)
    for col in ("buy", "now", "now_dollars", "now_pct", "peak_pct", "giveback_pct", "top100_score", "live_entry_score"):
        if col in out.columns:
            out[col] = out[col].map(display_optional_number)
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "qty": "Qty",
            "entry_time": "Entry Time",
            "buy": "Buy",
            "now": "Now",
            "now_dollars": "Now $",
            "now_pct": "Now %",
            "peak_pct": "Peak %",
            "giveback_pct": "Drop from Peak %",
            "top100_rank": "Top100 Rank",
            "top100_score": "Top100 Score",
            "live_entry_score": "Live Entry Score",
            "live_entry_rank": "Live Entry Rank",
            "entry_order_id": "Entry Order ID",
            "entry_perm_id": "Entry Perm ID",
            "status": "Status",
            "strategy": "Strategy",
            "data_quality": "Data Quality",
            "entry_metadata_status": "Entry Metadata",
            "ibkr_confirmed": "IBKR Confirmed",
            "price_status": "Price Status",
            "now_price_source": "Now Source",
            "market_price_at": "Price Time",
            "last_update": "Last Update",
            "source": "Source",
            "exit_sent": "Exit Sent",
            "execution_ids": "Execution IDs",
        }
    )
    display_cols = [
        "Symbol", "Qty", "Entry Time", "Buy", "Now", "Now $", "Now %",
        "Peak %", "Drop from Peak %", "Top100 Rank", "Top100 Score",
        "Live Entry Score", "Live Entry Rank", "Entry Order ID", "Entry Perm ID",
        "Status", "Strategy", "Data Quality", "Entry Metadata",
        "IBKR Confirmed", "Price Status", "Now Source", "Price Time",
        "Last Update", "Source", "Exit Sent", "Execution IDs",
    ]
    display_cols = [col for col in display_cols if col in out.columns]
    out = out[display_cols]
    st.dataframe(
        style_pnl(out, ["Now $", "Now %"]),
        width="stretch",
        hide_index=True,
    )


def render_runtime_executions(df: pd.DataFrame) -> None:
    st.subheader("Executions")
    if df is None or df.empty:
        st.info("No executions in the selected session window.")
        return
    cols = [
        "time", "recorded_at", "symbol", "side", "qty", "price", "gross_value",
        "exchange", "liquidity", "order_id", "perm_id", "execution_id", "trade_id",
        "commission", "commission_currency", "realized_pnl", "commission_source",
        "strategy", "session_date", "data_quality",
    ]
    available = [col for col in cols if col in df.columns]
    out = filter_table(df[available].copy(), "runtime_exec")
    if "time" in out.columns:
        out["time"] = out["time"].map(display_time)
    if "recorded_at" in out.columns:
        out["recorded_at"] = out["recorded_at"].map(display_time)
    for col in ("qty", "price", "gross_value", "commission", "realized_pnl"):
        if col in out.columns:
            out[col] = out[col].map(display_optional_number)
    out = out.rename(
        columns={
            "time": "Time",
            "recorded_at": "Recorded At",
            "symbol": "Symbol",
            "side": "Side",
            "qty": "Qty",
            "price": "Price",
            "gross_value": "Gross Value",
            "exchange": "Exchange",
            "liquidity": "Liquidity",
            "order_id": "Order ID",
            "perm_id": "Perm ID",
            "execution_id": "Exec ID",
            "trade_id": "Trade ID",
            "commission": "Commission",
            "commission_currency": "Currency",
            "realized_pnl": "Realized PnL",
            "commission_source": "Commission Source",
            "strategy": "Strategy",
            "session_date": "Session Date",
            "data_quality": "Data Quality",
        }
    )
    st.dataframe(
        style_pnl(out, ["Realized PnL"]),
        width="stretch",
        hide_index=True,
    )


def render_open_position_sections(df: pd.DataFrame) -> None:
    render_open_positions(df, title="Open Positions", prefix="open")


def render_raw_active_positions(df: pd.DataFrame) -> None:
    st.subheader("Raw active positions from SQLite")
    if df is None or df.empty:
        st.info("No raw active positions in SQLite.")
        return
    out = df.copy().rename(
        columns={
            "symbol": "Symbol",
            "quantity": "Quantity",
            "avg_price": "Avg Price",
            "status": "Status",
            "active": "Active",
            "updated_at": "Updated At",
            "entry_time": "Entry Time",
            "session_date": "Session Date",
            "strategy": "Strategy",
            "source": "Source",
            "position_key": "Position Key",
        }
    )
    cols = [
        "Symbol", "Quantity", "Avg Price", "Status", "Active", "Updated At",
        "Entry Time", "Session Date", "Strategy", "Source", "Position Key",
    ]
    out = out[[col for col in cols if col in out.columns]]
    st.dataframe(out, width="stretch", hide_index=True)


def render_orphan_stale_positions(df: pd.DataFrame) -> None:
    st.subheader("Orphan Stale Positions")
    if df.empty:
        st.info("No orphan stale positions.")
        return
    cols = [
        "symbol", "qty", "buy", "entry_time", "entry_date", "age_days", "status",
        "strategy", "source", "data_quality", "cleanup_recommendation",
    ]
    available = [col for col in cols if col in df.columns]
    out = df[available].copy().sort_values(["age_days", "symbol"], ascending=[False, True], na_position="last")
    if "entry_time" in out.columns:
        out["entry_time"] = out["entry_time"].map(display_time)
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "qty": "Qty",
            "buy": "Buy",
            "entry_time": "Entry Time",
            "entry_date": "Entry Date",
            "age_days": "Age Days",
            "status": "Status",
            "strategy": "Strategy",
            "source": "Source",
            "data_quality": "Data Quality",
            "cleanup_recommendation": "Cleanup Recommendation",
        }
    )
    st.warning("These rows are excluded from Open Trades/Open PnL. Use cleanup only after confirming broker has no position.")
    st.dataframe(out, width="stretch", hide_index=True)


def render_excluded_open_positions(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    with st.expander("Excluded Open Positions", expanded=False):
        out = df.copy().rename(
            columns={
                "symbol": "Symbol",
                "position_key": "Position Key",
                "session_date": "Session Date",
                "status": "Status",
                "quantity": "Qty",
                "ibkr_quantity": "IBKR Qty",
                "updated_at": "Updated At",
                "source": "Source",
                "exclusion_reason": "Exclusion Reason",
            }
        )
        cols = [
            "Symbol", "Qty", "IBKR Qty", "Status", "Session Date", "Updated At",
            "Source", "Exclusion Reason", "Position Key",
        ]
        out = out[[col for col in cols if col in out.columns]]
        st.dataframe(out, width="stretch", hide_index=True)


def render_rejected_entries(df: pd.DataFrame) -> None:
    st.subheader("Rejected Entries")
    if df.empty:
        st.info("No rejected entries in the selected window.")
        return
    cols = ["symbol", "qty", "price", "order_id", "reason", "ibkr_error_code", "rejected_at", "strategy"]
    available = [col for col in cols if col in df.columns]
    out = filter_table(df[available].copy(), "rejected").sort_values(["rejected_at", "symbol"], ascending=[False, True], na_position="last")
    if "rejected_at" in out.columns:
        out["rejected_at"] = out["rejected_at"].map(display_time)
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "qty": "Qty",
            "price": "Price",
            "order_id": "Order ID",
            "reason": "Reason",
            "ibkr_error_code": "IBKR Error",
            "rejected_at": "Rejected At",
            "strategy": "Strategy",
        }
    )
    st.dataframe(out, width="stretch", hide_index=True)


def closed_normal_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    strategy = df.get("strategy", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    quality = df.get("data_quality", pd.Series("", index=df.index)).fillna("").astype(str)
    trusted = df.get("runtime_pnl_trusted", pd.Series(True, index=df.index)).fillna(True).astype(bool)
    carried = quality.str.contains("CARRIED_POSITION_CLOSED_TODAY|CARRY_BASIS_UNVERIFIED", regex=True, na=False)
    return trusted & (strategy != "unknown") & ~carried


def format_closed_positions(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        "symbol", "entry_date", "exit_date", "qty", "ibkr_commission", "commission_status", "buy", "sell", "gross", "net_actual", "net_pct", "peak_pct",
        "mae_pct", "peak_price", "low_price", "peak_unrealized_pnl", "max_adverse_unrealized_pnl",
        "giveback_from_peak", "drop_from_peak_pct", "top100_rank", "top100_score",
        "live_entry_score", "live_entry_rank", "entry_order_id", "entry_perm_id",
        "hold_minutes", "exit_reason", "strategy",
        "entry_time", "exit_time", "exit_reason_source", "matched_event_type",
        "matched_event_time", "matched_order_id", "data_quality", "partial_rows",
    ]
    available = [col for col in cols if col in df.columns]
    out = filter_table(df[available].copy(), prefix).sort_values(["net_actual", "symbol"], na_position="last")
    if "entry_time" in out.columns:
        out["entry_time"] = out["entry_time"].map(display_time)
    if "exit_time" in out.columns:
        out["exit_time"] = out["exit_time"].map(display_time)
    if "matched_event_time" in out.columns:
        out["matched_event_time"] = out["matched_event_time"].map(display_time)
    for col in (
        "peak_pct",
        "mae_pct",
        "peak_price",
        "low_price",
        "peak_unrealized_pnl",
        "max_adverse_unrealized_pnl",
        "giveback_from_peak",
        "drop_from_peak_pct",
        "hold_minutes",
        "top100_score",
        "live_entry_score",
    ):
        if col in out.columns:
            out[col] = out[col].map(display_number_or_missing)
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "entry_date": "Entry Date",
            "exit_date": "Exit Date",
            "qty": "Quantity",
            "ibkr_commission": "IBKR Comm",
            "commission_status": "Commission Status",
            "buy": "Buy",
            "sell": "Sell",
            "gross": "Gross",
            "net_actual": "Net",
            "net_pct": "Net %",
            "peak_pct": "Peak %",
            "mae_pct": "MAE %",
            "peak_price": "Peak Price",
            "low_price": "Low Price",
            "peak_unrealized_pnl": "Peak UPNL",
            "max_adverse_unrealized_pnl": "Max Adverse UPNL",
            "giveback_from_peak": "Giveback $",
            "drop_from_peak_pct": "Drop from Peak %",
            "top100_rank": "Top100 Rank",
            "top100_score": "Top100 Score",
            "live_entry_score": "Live Entry Score",
            "live_entry_rank": "Live Entry Rank",
            "entry_order_id": "Entry Order ID",
            "entry_perm_id": "Entry Perm ID",
            "hold_minutes": "Min",
            "exit_reason": "Exit Reason",
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "exit_reason_source": "Exit Reason Source",
            "matched_event_type": "Matched Event Type",
            "matched_event_time": "Matched Event Time",
            "matched_order_id": "Matched Order ID",
            "data_quality": "Data Quality",
            "strategy": "Strategy",
            "partial_rows": "Partial Rows",
        }
    )
    display_cols = [
        "Symbol", "Quantity", "IBKR Comm", "Commission Status", "Buy", "Sell", "Gross", "Net", "Net %",
        "Peak %", "Drop from Peak %", "Top100 Rank", "Top100 Score",
        "Live Entry Score", "Live Entry Rank", "Entry Order ID", "Entry Perm ID",
        "Min", "Exit Reason", "Exit Reason Source",
        "Matched Event Type", "Matched Event Time", "Matched Order ID",
        "Entry Date", "Exit Date", "Entry Time", "Exit Time", "Strategy",
        "Data Quality", "Partial Rows",
    ]
    return out[[col for col in display_cols if col in out.columns]]


def live_entry_score_bucket(value: object) -> str:
    try:
        score = float(value)
    except Exception:
        return "score missing"
    if pd.isna(score):
        return "score missing"
    if score >= 80:
        return "live_entry_score >= 80"
    if score >= 60:
        return "live_entry_score 60-80"
    if score >= 40:
        return "live_entry_score 40-60"
    if score >= 20:
        return "live_entry_score 20-40"
    return "live_entry_score <20"


def render_live_entry_score_summary(df: pd.DataFrame) -> None:
    if df.empty or "live_entry_score" not in df.columns:
        return
    rows: list[dict[str, object]] = []
    working = df.copy()
    working["_score_bucket"] = working["live_entry_score"].map(live_entry_score_bucket)
    for bucket in [
        "live_entry_score >= 80",
        "live_entry_score 60-80",
        "live_entry_score 40-60",
        "live_entry_score 20-40",
        "live_entry_score <20",
        "score missing",
    ]:
        group = working[working["_score_bucket"] == bucket]
        net = pd.to_numeric(group.get("net_actual"), errors="coerce").dropna()
        count = int(len(group))
        wins = int((net > 0).sum()) if not net.empty else 0
        avg_net = float(net.mean()) if not net.empty else 0.0
        rows.append(
            {
                "Bucket": bucket,
                "Trade Count": count,
                "Win Rate": pct((wins / len(net) * 100.0) if len(net) else 0.0),
                "Avg Net PnL": display_optional_number(avg_net),
                "Expectancy": display_optional_number(avg_net),
            }
        )
    st.subheader("Live Entry Score Buckets")
    st.dataframe(style_pnl(pd.DataFrame(rows), ["Avg Net PnL", "Expectancy"]), width="stretch", hide_index=True)


def render_closed_positions(df: pd.DataFrame) -> None:
    st.subheader("Closed Positions")
    if df.empty:
        st.info("No confirmed closed trades in trades table.")
        return
    normal = df[closed_normal_mask(df)].copy()
    diagnostics = df[~closed_normal_mask(df)].copy()
    if diagnostics.empty:
        st.caption("Showing normal strategy closed trades only.")
    else:
        st.warning(f"{len(diagnostics)} carried/unattributed closed rows are excluded from the main strategy table and shown below.")
    if normal.empty:
        st.info("No normal attributed strategy closed trades. See carry/unattributed diagnostics below.")
    else:
        render_live_entry_score_summary(normal)
    aggregate = aggregate_closed_positions(normal)
    out = format_closed_positions(aggregate, "closed") if not aggregate.empty else pd.DataFrame()
    if not out.empty:
        st.dataframe(
            style_pnl(out, ["Gross", "Net", "Net %"]),
            width="stretch",
            hide_index=True,
        )
    if not normal.empty:
        with st.expander("Closed trade partial execution details", expanded=False):
            details = format_closed_positions(normal, "closed_details")
            st.dataframe(style_pnl(details, ["Gross", "Net", "Net %"]), width="stretch", hide_index=True)
    if not diagnostics.empty:
        st.subheader("Carry / Unattributed Closed Diagnostics")
        aggregate_diag = aggregate_closed_positions(diagnostics)
        diag_out = format_closed_positions(aggregate_diag, "closed_carry_diag")
        st.dataframe(
            style_pnl(diag_out, ["Gross", "Net", "Net %"]),
            width="stretch",
            hide_index=True,
        )
    debug_cols = [
        "trade_id", "symbol", "entry_execution_id", "exit_execution_id", "source",
        "entry_execution_count", "exit_execution_count",
        "confirmed_commission_execution_count", "expected_commission_execution_count",
        "peak_source", "peak_match_quality", "exit_reason", "exit_reason_source",
        "matched_event_type", "matched_event_time", "matched_order_id",
        "commission_source_detail", "closed_source", "data_quality",
    ]
    available_debug_cols = [col for col in debug_cols if col in df.columns]
    if available_debug_cols:
        with st.expander("Closed trade source diagnostics", expanded=False):
            debug = df[available_debug_cols].copy().rename(
                columns={
                    "trade_id": "Trade ID",
                    "symbol": "Symbol",
                    "entry_execution_id": "Entry Execution ID",
                    "exit_execution_id": "Exit Execution ID",
                    "source": "Source",
                    "entry_execution_count": "Entry Execs",
                    "exit_execution_count": "Exit Execs",
                    "confirmed_commission_execution_count": "Confirmed Comm Execs",
                    "expected_commission_execution_count": "Expected Comm Execs",
                    "peak_source": "Peak Source",
                    "peak_match_quality": "Peak Match Quality",
                    "exit_reason": "Exit Reason",
                    "exit_reason_source": "Exit Reason Source",
                    "matched_event_type": "Matched Event Type",
                    "matched_event_time": "Matched Event Time",
                    "matched_order_id": "Matched Order ID",
                    "commission_source_detail": "Commission Source Detail",
                    "closed_source": "Closed Source",
                    "data_quality": "Data Quality",
                }
            )
            st.dataframe(debug, width="stretch", hide_index=True)


def render_pending_trades(df: pd.DataFrame) -> None:
    st.subheader("Pending Closed Trades")
    if df.empty:
        st.info("No pending commission/PnL trades in the selected window.")
        return
    cols = [
        "symbol", "status", "qty", "buy", "sell", "gross", "commission", "net_pnl",
        "entry_time", "exit_time", "closed_at", "strategy", "trade_id",
        "updated_at", "trade_reduction_version",
    ]
    available = [col for col in cols if col in df.columns]
    out = filter_table(df[available].copy(), "pending_trades")
    if "updated_at" in out.columns:
        out = out.sort_values(["updated_at", "symbol", "trade_id"], ascending=[False, True, True], na_position="last")
    elif "exit_time" in out.columns:
        out = out.sort_values(["exit_time", "symbol", "trade_id"], ascending=[False, True, True], na_position="last")
    for time_col in ["entry_time", "exit_time", "closed_at", "updated_at"]:
        if time_col in out.columns:
            out[time_col] = out[time_col].map(display_time)
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "status": "Status",
            "qty": "Quantity",
            "buy": "Buy",
            "sell": "Sell",
            "gross": "Gross",
            "commission": "Commission",
            "net_pnl": "Net",
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "closed_at": "Closed At",
            "strategy": "Strategy",
            "trade_id": "Trade ID",
            "updated_at": "Updated At",
            "trade_reduction_version": "Reduction Version",
        }
    )
    st.warning("Pending rows are excluded from Closed Positions and summary PnL until commission/PnL finalizes.")
    st.dataframe(
        style_pnl(out, [col for col in ["Gross", "Net"] if col in out.columns]),
        width="stretch",
        hide_index=True,
    )


def render_exit_simulation(df: pd.DataFrame) -> None:
    st.subheader("Exit Simulation")
    if df.empty:
        st.info("No closed trades available for simulation.")
        return
    wanted = ["actual trailing", "fixed TP +2%", "fixed TP +2.5%", "fixed TP +3%", "fixed TP +4%", "partial 50%@+3%"]
    out = df[df["scenario"].isin(wanted)].copy()
    st.dataframe(style_pnl(out, ["gross", "net"]), width="stretch", hide_index=True)


def render_peak_charts(closed: pd.DataFrame) -> None:
    st.subheader("Peak / Giveback Analytics")
    if closed.empty:
        st.info("No closed trades available for peak analytics.")
        return
    chart_cols = st.columns(3)
    peak_bins = pd.cut(closed["peak_pct"].fillna(0), bins=20).value_counts().sort_index()
    giveback_bins = pd.cut(closed["drop_from_peak_pct"].fillna(0), bins=20).value_counts().sort_index()
    chart_cols[0].caption("Histogram peak%")
    chart_cols[0].bar_chart(pd.DataFrame({"count": peak_bins.values}, index=[str(x) for x in peak_bins.index]))
    chart_cols[1].caption("Histogram giveback%")
    chart_cols[1].bar_chart(pd.DataFrame({"count": giveback_bins.values}, index=[str(x) for x in giveback_bins.index]))
    chart_cols[2].caption("Peak% vs realized PnL")
    scatter = closed[["peak_pct", "net_actual"]].rename(columns={"peak_pct": "peak_pct", "net_actual": "net_actual"})
    chart_cols[2].scatter_chart(scatter, x="peak_pct", y="net_actual")


def render_diagnostics(diag: dict) -> None:
    st.subheader("Diagnostics / Risk")
    stale_count = int(diag.get("stale_active_positions_count", 0) or 0)
    if stale_count > 0:
        st.warning(f"STALE_POSITION_ROWS_PRESENT stale_active_positions_count={stale_count}")
    if int(diag.get("active_positions_raw_count", 0) or 0) > 0 and int(diag.get("displayed_open_positions_count", 0) or 0) == 0:
        st.error("RUNTIME_OPEN_POSITION_FILTER_BUG: raw active positions exist in SQLite but Runtime displayed open positions is 0.")
    if int(diag.get("ibkr_positions_count", 0) or 0) > 0 and int(diag.get("displayed_open_positions_count", 0) or 0) == 0:
        st.error("BROKER_RUNTIME_OPEN_POSITION_MISMATCH: broker/reconciliation has open positions but Runtime displayed open positions is 0.")
    if int(diag.get("dropped_open_count", 0) or 0) > 0:
        st.warning(f"Runtime dropped open symbols: {diag.get('dropped_symbols', '')}")
    if int(diag.get("dropped_closed_trade_count", 0) or 0) > 0:
        st.warning(f"Runtime dropped closed trade IDs: {diag.get('dropped_closed_trade_ids', '')}")
    if int(diag.get("untrusted_carry_closed_count", 0) or 0) > 0:
        st.warning(
            "Runtime excluded unverified carry closed trades from PnL. "
            f"symbols={diag.get('untrusted_carry_closed_symbols', '')}. Broker Reality is source of truth for these rows."
        )
    cols = st.columns(8)
    labels = [
        ("Orphans", "orphans"),
        ("Missing IBKR", "missing_in_ibkr"),
        ("Rejected Entries", "rejected_entries"),
        ("Partial Exits", "partial_exits"),
        ("Delayed Fills", "delayed_fills"),
        ("Risk Blocks", "risk_guard_blocks"),
        ("SQLite Failures", "sqlite_failures"),
        ("Reconnects", "reconnect_events"),
    ]
    for col, (label, key) in zip(cols, labels):
        col.metric(label, int(diag.get(key, 0)))
    pos_cols = st.columns(12)
    position_labels = [
        ("Raw SQLite", "raw_active_sqlite_count"),
        ("Displayed", "displayed_open_count"),
        ("Dropped", "dropped_open_count"),
        ("Active Raw", "active_positions_raw_count"),
        ("Active Today", "active_positions_today_count"),
        ("After Orphan Filter", "active_positions_after_orphan_filter_count"),
        ("Displayed Open", "displayed_open_positions_count"),
        ("Displayed Today", "displayed_today_open_count"),
        ("Displayed Carry", "displayed_carry_open_count"),
        ("Orphan Stale", "orphan_stale_count"),
        ("Excluded Open", "excluded_open_positions_count"),
        ("IBKR Positions", "ibkr_positions_count"),
    ]
    for col, (label, key) in zip(pos_cols, position_labels):
        col.metric(label, int(diag.get(key, 0)))
    valuation_cols = st.columns(4)
    valuation_cols[0].metric("Broker Valuation Symbols", int(diag.get("broker_portfolio_valuation_symbols", 0) or 0))
    valuation_cols[1].metric("Price Mismatches", int(diag.get("dashboard_broker_price_mismatch_count", 0) or 0))
    valuation_cols[2].metric("Max Price Diff", display_number_or_missing(diag.get("dashboard_broker_price_max_abs_diff"), decimals=4))
    valuation_cols[3].metric("Broker Portfolio Rows", int(diag.get("runtime_broker_portfolio_rows", 0) or 0))
    closed_cols = st.columns(8)
    closed_labels = [
        ("Raw Closed", "raw_closed_trade_count"),
        ("Persisted Closed", "persisted_closed_trades_count"),
        ("Reconstructed Pairs", "reconstructed_execution_pairs_count"),
        ("Displayed Closed", "displayed_closed_trades_count"),
        ("Dropped Closed", "dropped_closed_trade_count"),
        ("Untrusted Carry", "untrusted_carry_closed_count"),
        ("Carried Closed Today", "carried_closed_today_count"),
    ]
    for col, (label, key) in zip(closed_cols, closed_labels):
        col.metric(label, int(diag.get(key, 0)))
    reducer_cols = st.columns(4)
    reducer_cols[0].metric("SQLite Closed Trades", int(diag.get("closed_trades_count", 0) or 0))
    reducer_cols[1].metric("Broker Closed Trades", str(diag.get("broker_closed_trades_count", "N/A")))
    reducer_cols[2].metric("Reducer Updated <60s", int(diag.get("trades_updated_last_60s", 0) or 0))
    reducer_cols[3].metric("Reducer Running", int(diag.get("reducer_running", 0) or 0))
    pnl_diag = pd.DataFrame(
        [
            {
                "Source": "Broker / executions truth",
                "Closed Trades Count": int(diag.get("broker_closed_trades_count", 0) or 0),
                "Net PnL": float(diag.get("broker_net_pnl", 0.0) or 0.0),
                "Commissions": float(diag.get("broker_commissions", 0.0) or 0.0),
            },
            {
                "Source": "Reducer / reconstructed rows",
                "Closed Trades Count": int(diag.get("reducer_closed_rows_count", 0) or 0),
                "Net PnL": float(diag.get("reducer_net_pnl", 0.0) or 0.0),
                "Commissions": float(diag.get("reducer_commissions", 0.0) or 0.0),
            },
        ]
    )
    st.markdown("**Closed PnL Source Diagnostics**")
    st.dataframe(
        style_pnl(pnl_diag, ["Net PnL"]).format({"Net PnL": "${:,.2f}", "Commissions": "${:,.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "runtime_trust_status="
        f"{diag.get('runtime_trust_status', 'UNKNOWN')} "
        f"last_reducer_run_at={diag.get('last_reducer_run_at', '')}"
    )


def dataframe_or_info(df: pd.DataFrame, message: str, *, key: str | None = None) -> None:
    if df.empty:
        st.info(message)
    else:
        st.dataframe(df, width="stretch", hide_index=True, key=key)


def export_reconciliation_csv(frames: dict[str, pd.DataFrame]) -> str:
    chunks = []
    for name, frame in frames.items():
        chunks.append(f"# {name}\n")
        if frame.empty:
            chunks.append("\n")
        else:
            chunks.append(frame.to_csv(index=False))
            chunks.append("\n")
    return "".join(chunks)


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]


def trade_difference_contributors(result: ReconciliationResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not result.trade_mismatches.empty:
        for row in result.trade_mismatches.to_dict("records"):
            rows.append(
                {
                    "symbol": row.get("symbol"),
                    "net_difference": float(row.get("net_difference") or 0.0),
                    "source": "trade_mismatch",
                }
            )
    if not result.missing_trades.empty:
        for row in result.missing_trades.to_dict("records"):
            rows.append(
                {
                    "symbol": row.get("symbol"),
                    "net_difference": float(row.get("net") or 0.0),
                    "source": "missing_sqlite",
                }
            )
    if not result.extra_trades.empty:
        for row in result.extra_trades.to_dict("records"):
            rows.append(
                {
                    "symbol": row.get("symbol"),
                    "net_difference": -float(row.get("net") or 0.0),
                    "source": "missing_broker",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "net_difference", "abs_net_difference", "rows"])
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("symbol", dropna=False)
        .agg(net_difference=("net_difference", "sum"), rows=("symbol", "size"))
        .reset_index()
    )
    grouped["abs_net_difference"] = pd.to_numeric(grouped["net_difference"], errors="coerce").abs()
    return grouped.sort_values("abs_net_difference", ascending=False, na_position="last").head(10)


def empty_reconciliation_result(selected_date: str, broker_status: str = "NOT_LOADED") -> ReconciliationResult:
    summary = {
        "status": "CSV_REQUIRED_FOR_HISTORICAL_DATE" if not str(broker_status).startswith("OK") else "NOT_RECONCILED",
        "broker_status": broker_status,
        "broker_execution_count": 0,
        "sqlite_execution_count": 0,
        "matched_executions": 0,
        "missing_in_sqlite": 0,
        "extra_in_sqlite": 0,
        "execution_mismatches": 0,
        "position_mismatches": 0,
        "matched_trades": 0,
        "missing_trades": 0,
        "extra_trades": 0,
        "trade_mismatches": 0,
    }
    return ReconciliationResult(
        summary=summary,
        matched=pd.DataFrame(),
        missing_in_sqlite=pd.DataFrame(),
        extra_in_sqlite=pd.DataFrame(),
        execution_mismatches=pd.DataFrame(),
        position_mismatches=pd.DataFrame(),
        broker_closed_trades=empty_closed_trades(),
        sqlite_closed_trades=empty_closed_trades(),
        matched_trades=pd.DataFrame(),
        missing_trades=pd.DataFrame(),
        extra_trades=pd.DataFrame(),
        trade_mismatches=pd.DataFrame(),
        pnl_comparison=pd.DataFrame(
            [
                {
                    "selected_date": selected_date,
                    "broker_gross_pnl": 0.0,
                    "broker_commission": 0.0,
                    "broker_net_pnl": 0.0,
                    "sqlite_gross": 0.0,
                    "sqlite_commission": 0.0,
                    "sqlite_net": 0.0,
                    "sqlite_closed_trades": 0,
                    "net_delta": 0.0,
                    "commission_delta": 0.0,
                }
            ]
        ),
    )


def render_runtime_tab(sqlite_path: str, start_date: str, end_date: str, strategy: str, include_reconstructed: bool, auto_refresh: bool) -> None:
    window = DateWindow(start_date, end_date)
    current = is_current_window(window)
    if current:
        broker_portfolio = pd.DataFrame()
        broker_status = "NOT_LOADED"
        try:
            host = str(st.session_state.get("broker_host") or "127.0.0.1")
            port = int(st.session_state.get("broker_port") or 4002)
            client_id = int(st.session_state.get("broker_client_id") or 177)
            timeout = float(st.session_state.get("broker_timeout") or 4.0)
            broker_portfolio, broker_status = fetch_ibkr_live_portfolio(host, port, client_id, timeout)
        except Exception as exc:
            broker_status = f"runtime_broker_portfolio_exception: {exc}"
            broker_portfolio = pd.DataFrame()
        snapshot = load_live_snapshot(
            sqlite_path,
            start_date,
            end_date,
            strategy,
            include_reconstructed,
            tuple(broker_portfolio.to_dict("records")) if not broker_portfolio.empty else (),
        )
        snapshot.setdefault("diagnostics", {})["runtime_broker_portfolio_status"] = broker_status
        snapshot.setdefault("diagnostics", {})["runtime_broker_portfolio_rows"] = int(len(broker_portfolio))
    else:
        snapshot = load_historical_snapshot(sqlite_path, start_date, end_date, strategy, include_reconstructed)

    loaded_at = snapshot.get("loaded_at", "")
    snapshot_version = snapshot.get("snapshot_version", "")
    trust_status = snapshot.get("trust_status", "")
    st.caption(
        f"SQLite: `{snapshot.get('source')}` | window: {start_date} to {end_date} | "
        f"strategy: {strategy} | loaded_at: {loaded_at} | snapshot_version: {snapshot_version}"
    )
    if trust_status == "SQLITE_UNTRUSTED_REDUCER_ACTIVE":
        st.warning(
            "Runtime Dashboard is SQLite-only and reducer rows changed in the last 60 seconds. "
            "Treat Runtime PnL as UNTRUSTED until Broker Reality reconciliation is stable."
        )
    elif trust_status:
        st.info(f"Runtime source: {trust_status}. Broker Reality remains the broker-truth reconciliation view.")

    render_summary(snapshot["summary"])
    st.divider()
    render_data_quality_summary(snapshot.get("data_quality_summary", {}))
    if int(snapshot.get("diagnostics", {}).get("execution_reconstruction_disabled", 0) or 0):
        st.warning("EXECUTION_RECONSTRUCTION_DISABLED_IN_DASHBOARD: executions exist, but no persisted closed trades exist for this window.")
    st.divider()
    render_open_position_sections(snapshot["open_positions"])
    render_raw_active_positions(snapshot.get("raw_active_positions", pd.DataFrame()))
    st.divider()
    render_runtime_executions(snapshot.get("executions", pd.DataFrame()))
    st.divider()
    render_orphan_stale_positions(snapshot.get("orphan_stale_positions", pd.DataFrame()))
    render_excluded_open_positions(snapshot.get("excluded_open_positions", pd.DataFrame()))
    st.divider()
    render_rejected_entries(snapshot.get("rejected_entries", pd.DataFrame()))
    st.divider()
    render_pending_trades(snapshot.get("pending_trades", pd.DataFrame()))
    st.divider()
    render_closed_positions(snapshot["closed_positions"])
    st.divider()
    render_exit_simulation(snapshot["exit_simulation"])
    st.divider()
    render_peak_charts(snapshot["closed_positions"])
    st.divider()
    render_diagnostics(snapshot["diagnostics"])

    if current and auto_refresh:
        st.markdown('<p class="small-note">Auto refresh disabled for Runtime stability. Refresh the browser manually after reducer/backfill settles.</p>', unsafe_allow_html=True)


def render_broker_reality_tab(sqlite_path: str) -> None:
    st.header("Broker Reality / IBKR Reconciliation")
    st.success("Broker Reality tab loaded")
    selected = st.date_input("Broker execution date", value=pd.to_datetime(utc_today()).date(), key="broker_reality_date")
    selected_date = selected.isoformat()
    debug_broker = st.toggle("Show broker debug diagnostics", value=False, key="broker_reality_debug")

    settings = st.expander("IBKR connection / Activity CSV fallback", expanded=False)
    with settings:
        col1, col2, col3, col4 = st.columns(4)
        host = col1.text_input("IBKR host", value="127.0.0.1", key="broker_host")
        port = col2.number_input("IBKR port", value=4002, step=1, key="broker_port")
        client_id = col3.number_input("Client ID", value=177, step=1, key="broker_client_id")
        timeout = col4.number_input("Timeout seconds", value=4.0, step=0.5, key="broker_timeout")
        use_api_executions = st.toggle("Try IBKR API executions for selected date", value=True)
        uploaded_csv = st.file_uploader("IBKR Activity Statement / Trades CSV fallback", type=["csv", "txt"], key="broker_activity_csv")
        csv_path = st.text_input("Or Activity CSV file path on server", value="", key="broker_activity_csv_path")

    portfolio = pd.DataFrame()
    portfolio_status = "NOT_LOADED"
    broker_executions = empty_broker_executions()
    raw_broker_executions = empty_broker_executions()
    broker_closed_trades = empty_closed_trades()
    broker_fifo_estimated_trades = empty_closed_trades()
    broker_status = "CSV_REQUIRED_FOR_HISTORICAL_DATE"
    broker_execution_diagnostics = {
        "selected_date": selected_date,
        "timezone_used": "UTC",
        "api_methods_used": [],
        "raw_execution_count_before_filtering": 0,
        "execution_count_after_date_filtering": 0,
        "raw_commission_report_count": 0,
        "fills_count": 0,
        "req_executions_count": 0,
        "req_executions_filter_count": 0,
        "executions_attr_count": 0,
        "unique_execution_client_ids": [],
        "first_5_raw_executions": [],
        "ibkr_messages": [],
        "session_only_warning": "IBKR_API_EXECUTIONS_SESSION_ONLY",
    }
    sqlite_executions = empty_broker_executions()
    sqlite_closed_trades = empty_closed_trades()
    sqlite_positions = pd.DataFrame()
    sqlite_trade_pnl = pd.DataFrame()
    result = empty_reconciliation_result(selected_date, broker_status)
    csv_loaded = False
    api_executions_count = 0

    try:
        portfolio, portfolio_status = fetch_ibkr_live_portfolio(host, int(port), int(client_id), float(timeout))
    except Exception as exc:
        portfolio = pd.DataFrame()
        portfolio_status = f"ibkr_portfolio_exception: {exc}"
        st.error("IBKR connection unavailable")
        st.exception(exc)

    if use_api_executions:
        try:
            execution_fetch = fetch_ibkr_executions_diagnostic(host, int(port), int(client_id), selected_date, float(timeout))
            raw_broker_executions = execution_fetch.raw_executions
            broker_executions = execution_fetch.filtered_executions
            broker_status = execution_fetch.status
            broker_execution_diagnostics = execution_fetch.diagnostics
            api_executions_count = len(broker_executions)
        except Exception as exc:
            raw_broker_executions = empty_broker_executions()
            broker_executions = empty_broker_executions()
            broker_status = f"ibkr_executions_exception: {exc}"
            broker_execution_diagnostics["ibkr_messages"].append(str(exc))
            st.error("IBKR execution request failed")
            st.exception(exc)
    if uploaded_csv is not None:
        try:
            raw_broker_executions = parse_ibkr_activity_csv(uploaded_csv)
            broker_executions = raw_broker_executions[raw_broker_executions["execution_time"].map(lambda x: str(x)[:10]) == selected_date].reset_index(drop=True)
            broker_status = "OK"
            csv_loaded = True
            broker_execution_diagnostics["raw_execution_count_before_filtering"] = int(len(raw_broker_executions))
            broker_execution_diagnostics["execution_count_after_date_filtering"] = int(len(broker_executions))
            broker_execution_diagnostics["api_methods_used"] = ["IBKR Activity CSV upload"]
        except Exception as exc:
            raw_broker_executions = empty_broker_executions()
            broker_executions = empty_broker_executions()
            broker_status = f"csv_upload_parse_error: {exc}"
            broker_execution_diagnostics["ibkr_messages"].append(str(exc))
            st.error("IBKR Activity CSV parse failed")
            st.exception(exc)
    elif csv_path.strip():
        path = Path(csv_path.strip()).expanduser()
        if path.exists():
            try:
                raw_broker_executions = parse_ibkr_activity_csv(path.read_text())
                broker_executions = raw_broker_executions[raw_broker_executions["execution_time"].map(lambda x: str(x)[:10]) == selected_date].reset_index(drop=True)
                broker_status = "OK"
                csv_loaded = True
                broker_execution_diagnostics["raw_execution_count_before_filtering"] = int(len(raw_broker_executions))
                broker_execution_diagnostics["execution_count_after_date_filtering"] = int(len(broker_executions))
                broker_execution_diagnostics["api_methods_used"] = [f"IBKR Activity CSV path {path}"]
            except Exception as exc:
                raw_broker_executions = empty_broker_executions()
                broker_executions = empty_broker_executions()
                broker_status = f"csv_path_parse_error: {exc}"
                broker_execution_diagnostics["ibkr_messages"].append(str(exc))
                st.error("IBKR Activity CSV path parse failed")
                st.exception(exc)
        else:
            broker_status = f"csv_path_not_found: {path}"
            broker_execution_diagnostics["ibkr_messages"].append(broker_status)

    try:
        sqlite_executions = load_sqlite_executions(sqlite_path, selected_date)
        sqlite_closed_trades = load_sqlite_closed_trades(sqlite_path, selected_date)
        sqlite_positions = load_sqlite_active_positions(sqlite_path, selected_date)
        sqlite_trade_pnl = load_sqlite_trade_pnl(sqlite_path, selected_date)
    except Exception as exc:
        st.error("SQLite reconciliation data load failed")
        st.exception(exc)

    try:
        broker_closed_trades = closed_trades_from_commission_reports(broker_executions, selected_date)
    except Exception as exc:
        broker_closed_trades = empty_closed_trades()
        st.error("Broker closed trade realized PnL reconstruction from IBKR commission reports failed")
        st.exception(exc)

    try:
        broker_fifo_estimated_trades = reconstruct_closed_trades_fifo(broker_executions, selected_date)
    except Exception as exc:
        broker_fifo_estimated_trades = empty_closed_trades()
        st.error("Broker FIFO estimated trade reconstruction failed")
        st.exception(exc)

    try:
        result = reconcile_broker_vs_sqlite(
            broker_executions,
            sqlite_executions,
            portfolio,
            sqlite_positions,
            broker_closed_trades,
            sqlite_closed_trades,
            sqlite_trade_pnl,
            selected_date=selected_date,
            broker_status=broker_status,
        )
    except Exception as exc:
        result = empty_reconciliation_result(selected_date, broker_status)
        st.error("Broker vs SQLite reconciliation failed")
        st.exception(exc)

    ibkr_connected = str(portfolio_status).startswith("OK")
    account = ""
    if not portfolio.empty and "account" in portfolio.columns:
        account = ", ".join(sorted(str(x) for x in portfolio["account"].dropna().unique() if str(x)))
    last_refresh = pd.Timestamp.utcnow().strftime("%d-%m-%Y %H:%M:%S UTC")

    st.subheader("Broker Status")
    status_cols = st.columns(4)
    status_cols[0].metric("Connection", "Connected" if ibkr_connected else "Disconnected")
    status_cols[1].metric("Account", account or "N/A")
    status_cols[2].metric("Selected Date", selected_date)
    status_cols[3].metric("Last Refresh", last_refresh)
    if not ibkr_connected:
        st.warning(f"IBKR connection unavailable: {portfolio_status}")
    if broker_status and not str(broker_status).startswith("OK"):
        st.warning(str(broker_status))

    st.divider()
    st.subheader("Broker Portfolio")
    if portfolio.empty:
        st.info("No portfolio positions")
    else:
        st.dataframe(portfolio, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Broker Executions")
    exec_diag_cols = st.columns(6)
    exec_diag_cols[0].metric("Raw API/CSV executions", int(broker_execution_diagnostics.get("raw_execution_count_before_filtering", len(raw_broker_executions)) or 0))
    exec_diag_cols[1].metric("After date filter", int(broker_execution_diagnostics.get("execution_count_after_date_filtering", len(broker_executions)) or 0))
    exec_diag_cols[2].metric("ib.fills()", int(broker_execution_diagnostics.get("fills_count", 0) or 0))
    exec_diag_cols[3].metric("reqExecutions", int(broker_execution_diagnostics.get("req_executions_count", 0) or 0))
    exec_diag_cols[4].metric("Commission reports", int(broker_execution_diagnostics.get("raw_commission_report_count", 0) or 0))
    exec_diag_cols[5].metric("Timezone", str(broker_execution_diagnostics.get("timezone_used", "UTC")))
    client_ids = broker_execution_diagnostics.get("unique_execution_client_ids") or []
    if client_ids:
        st.caption(f"Execution clientIds: {', '.join(str(x) for x in client_ids)}")
    if broker_executions.empty:
        st.info("No executions for selected date")
        st.warning("Broker Closed Trades require Broker Executions. If API executions are empty, upload Activity Statement / Trades CSV.")
        if use_api_executions and raw_broker_executions.empty and not csv_loaded:
            st.warning("IBKR_API_EXECUTIONS_SESSION_ONLY: IBKR API may return only executions from the current API session, not full historical selected-date activity.")
        if not csv_loaded and not use_api_executions:
            st.warning("CSV_REQUIRED_FOR_HISTORICAL_DATE")
    else:
        st.dataframe(broker_executions, width="stretch", hide_index=True)
    with st.expander("Raw Broker Executions Preview", expanded=False):
        dataframe_or_info(raw_broker_executions.head(50), "No raw broker executions returned before date filtering.", key="raw_broker_executions_preview")

    st.divider()
    st.subheader("Broker Closed Trades / Realized Trades")
    if broker_closed_trades.empty:
        st.info("No broker realized closed trades from IBKR commission reports for selected date.")
        if not broker_executions.empty:
            st.caption("Broker truth requires commissionReport.realizedPNL. FIFO estimate is shown separately when available.")
    else:
        st.caption("source=IBKR_COMMISSION_REPORT_REALIZED_PNL")
        st.dataframe(broker_closed_trades, width="stretch", hide_index=True)
    with st.expander("Estimated FIFO closed trades (not broker truth)", expanded=False):
        st.caption("source=BROKER_FIFO_RECONSTRUCTED; use only as an estimate/diagnostic, not realized broker PnL.")
        dataframe_or_info(broker_fifo_estimated_trades, "No FIFO-estimated closed trades available.")

    st.divider()
    st.subheader("Reconciliation Summary")
    status = result.summary.get("status")
    if status == "IBKR_RECONCILED":
        st.success(status)
    elif status == "CSV_REQUIRED_FOR_HISTORICAL_DATE":
        st.warning(status)
    elif status == "NOT_RECONCILED":
        st.warning(status)
    else:
        st.error(status)
    cols = st.columns(8)
    for col, (label, key) in zip(
        cols,
        [
            ("Matched", "matched_executions"),
            ("Missing SQLite", "missing_in_sqlite"),
            ("Extra SQLite", "extra_in_sqlite"),
            ("Position Mismatches", "position_mismatches"),
            ("Matched Trades", "matched_trades"),
            ("Missing Trades", "missing_trades"),
            ("Extra Trades", "extra_trades"),
            ("Trade Mismatches", "trade_mismatches"),
        ],
    ):
        col.metric(label, int(result.summary.get(key, 0) or 0))
    pnl_cols = st.columns(7)
    pnl_metrics = [
        ("Broker Gross", "broker_gross_pnl"),
        ("Broker Comms", "broker_commission"),
        ("Broker Net", "broker_net_pnl"),
        ("SQLite Gross", "sqlite_gross_pnl"),
        ("SQLite Comms", "sqlite_commission"),
        ("SQLite Net", "sqlite_net_pnl"),
        ("Difference", "net_pnl_difference"),
    ]
    for col, (label, key) in zip(pnl_cols, pnl_metrics):
        col.metric(label, money(float(result.summary.get(key, 0.0) or 0.0)))
    st.caption(
        f"broker_executions={len(broker_executions)} sqlite_executions={len(sqlite_executions)} "
        f"broker_closed_trades={len(broker_closed_trades)} sqlite_closed_trades={len(sqlite_closed_trades)} "
        f"portfolio_rows={len(portfolio)}"
    )
    st.caption(
        f"reconciliation_sqlite_trade_source={result.summary.get('reconciliation_sqlite_trade_source', '')} "
        f"runtime_trade_source={result.summary.get('runtime_trade_source', '')} "
        f"trusted_closed_count={result.summary.get('trusted_closed_count', 0)} "
        f"untrusted_carry_count={result.summary.get('untrusted_carry_count', 0)}"
    )
    if status == "CSV_REQUIRED_FOR_HISTORICAL_DATE":
        st.warning("CSV_REQUIRED_FOR_HISTORICAL_DATE")

    st.subheader("Mismatches")
    contributors = trade_difference_contributors(result)
    if not contributors.empty:
        st.caption("Top 10 symbols contributing most to reconciliation net difference")
        st.dataframe(contributors, width="stretch", hide_index=True)
    mismatch_tabs = st.tabs([
        "Missing Executions",
        "Extra Executions",
        "Execution Mismatches",
        "Position Mismatches",
        "Trade Mismatches",
        "Missing SQLite Trades",
        "Missing Broker Trades",
        "Trades / PnL",
    ])
    with mismatch_tabs[0]:
        dataframe_or_info(result.missing_in_sqlite, "No broker executions missing in SQLite.")
    with mismatch_tabs[1]:
        dataframe_or_info(result.extra_in_sqlite, "No extra SQLite executions.")
    with mismatch_tabs[2]:
        dataframe_or_info(result.execution_mismatches, "No quantity/price/commission mismatches.")
    with mismatch_tabs[3]:
        position_columns = [
            "symbol",
            "broker_qty",
            "sqlite_qty",
            "broker_avg_cost",
            "sqlite_avg_cost",
            "broker_market_value",
            "sqlite_market_value",
            "qty_difference",
            "cost_difference",
            "status",
        ]
        dataframe_or_info(select_columns(result.position_mismatches, position_columns), "No position mismatches.")
    with mismatch_tabs[4]:
        trade_mismatch_columns = [
            "symbol",
            "broker_trade_id",
            "sqlite_trade_id",
            "broker_qty",
            "sqlite_qty",
            "broker_entry_time",
            "sqlite_entry_time",
            "broker_exit_time",
            "sqlite_exit_time",
            "broker_gross",
            "sqlite_gross",
            "broker_commission",
            "sqlite_commission",
            "broker_net",
            "sqlite_net",
            "net_difference",
            "status",
        ]
        dataframe_or_info(select_columns(result.trade_mismatches, trade_mismatch_columns), "No closed trade quantity/pnl/commission mismatches.")
    with mismatch_tabs[5]:
        missing_columns = ["symbol", "qty", "entry_time", "exit_time", "gross", "commission", "net", "trade_id", "mismatch_type"]
        dataframe_or_info(select_columns(result.missing_trades, missing_columns), "No broker closed trades missing in SQLite.")
    with mismatch_tabs[6]:
        missing_columns = ["symbol", "qty", "entry_time", "exit_time", "gross", "commission", "net", "trade_id", "mismatch_type"]
        dataframe_or_info(select_columns(result.extra_trades, missing_columns), "No SQLite closed trades missing in broker.")
    with mismatch_tabs[7]:
        dataframe_or_info(result.pnl_comparison, "No PnL comparison available.")

    with st.expander("Diagnostics", expanded=False):
        st.write("event_loop_info", describe_asyncio_event_loop())
        st.write("portfolio_type", type(portfolio))
        st.write("portfolio_rows", len(portfolio))
        st.write("broker_executions_type", type(broker_executions))
        st.write("broker_executions_rows", len(broker_executions))
        st.write("raw_broker_executions_rows", len(raw_broker_executions))
        st.write("broker_execution_diagnostics", broker_execution_diagnostics)
        st.write("broker_closed_trades_rows", len(broker_closed_trades))
        st.write("broker_fifo_estimated_trades_rows", len(broker_fifo_estimated_trades))
        st.write("sqlite_closed_trades_rows", len(sqlite_closed_trades))
        st.write("sqlite_executions_type", type(sqlite_executions))
        st.write("sqlite_executions_rows", len(sqlite_executions))
        st.write("sqlite_positions_rows", len(sqlite_positions))
        st.write("api_executions_count", api_executions_count)
        st.write("csv_loaded", csv_loaded)
        st.write("portfolio_status", portfolio_status)
        st.write("broker_status", broker_status)
        if debug_broker:
            st.subheader("SQLite Executions")
            dataframe_or_info(sqlite_executions, "No SQLite executions for selected date.", key="broker_diag_sqlite_execs")

    export_csv = export_reconciliation_csv(reconciliation_export_frames(result))
    st.download_button(
        "Export reconciliation report CSV",
        data=export_csv,
        file_name=f"ibkr_reconciliation_{selected_date}.csv",
        mime="text/csv",
    )
    export_cols = st.columns(2)
    export_cols[0].download_button(
        "Export trade_mismatches.csv",
        data=select_columns(result.trade_mismatches, [
            "symbol", "broker_trade_id", "sqlite_trade_id", "broker_qty", "sqlite_qty",
            "broker_entry_time", "sqlite_entry_time", "broker_exit_time", "sqlite_exit_time",
            "broker_gross", "sqlite_gross", "broker_commission", "sqlite_commission",
            "broker_net", "sqlite_net", "net_difference", "status",
        ]).to_csv(index=False),
        file_name=f"trade_mismatches_{selected_date}.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "Export position_mismatches.csv",
        data=select_columns(result.position_mismatches, [
            "symbol", "broker_qty", "sqlite_qty", "broker_avg_cost", "sqlite_avg_cost",
            "broker_market_value", "sqlite_market_value", "qty_difference", "cost_difference", "status",
        ]).to_csv(index=False),
        file_name=f"position_mismatches_{selected_date}.csv",
        mime="text/csv",
    )


def render_readiness_card(title: str, value: object, status: str, details: list[str]) -> None:
    st.markdown(f"### {title}")
    cols = st.columns([1, 1])
    cols[0].metric("Value", value)
    with cols[1]:
        status_badge(status)
    for detail in details:
        st.caption(detail)


def render_operational_readiness_tab(sqlite_path: str, selected_session_date: str) -> None:
    st.header("Operational Readiness")
    st.caption("Pre-session/post-session operational flags. Broker portfolio is read live from IBKR; SQLite/file checks are local snapshots.")

    controls = st.expander("Readiness settings", expanded=False)
    with controls:
        readiness_date = st.date_input(
            "Selected/current trading date",
            value=pd.to_datetime(selected_session_date).date(),
            key="ops_readiness_date",
        ).isoformat()
        previous_session = previous_us_equity_trading_day(pd.to_datetime(readiness_date).date()).isoformat()
        history_dir = st.text_input("Universe 1m parquet dir", value="data/history/universe_1m", key="ops_history_dir")
        universe_csv = st.text_input("Universe CSV", value="data/universe/v68_final_daytrading_universe.csv", key="ops_universe_csv")
        collector_status_dir = st.text_input("Collector status dir", value="data/history", key="ops_collector_status_dir")
        top100_latest = st.text_input("Top100 latest CSV", value="data/universe/daily_top100_latest.csv", key="ops_top100_latest")
        expected_top100 = int(st.number_input("Top100 expected symbols", value=100, min_value=1, step=1, key="ops_expected_top100"))
        control_api_url = st.text_input("Control API base URL", value="http://127.0.0.1:8767", key="ops_control_api_url").rstrip("/")
        col1, col2, col3, col4 = st.columns(4)
        host = col1.text_input("IBKR host", value="127.0.0.1", key="ops_ibkr_host")
        port = int(col2.number_input("IBKR port", value=4002, step=1, key="ops_ibkr_port"))
        client_id = int(col3.number_input("Client ID", value=178, step=1, key="ops_ibkr_client_id"))
        timeout = float(col4.number_input("Timeout seconds", value=4.0, step=0.5, key="ops_ibkr_timeout"))

    refresh_clicked = st.button("Refresh broker status", key="ops_refresh_broker")
    if refresh_clicked:
        st.cache_data.clear()

    broker_portfolio = pd.DataFrame()
    broker_status_message = "NOT_LOADED"
    try:
        broker_portfolio, broker_status_message = fetch_ibkr_live_portfolio(host, port, client_id, timeout)
    except Exception as exc:
        broker_status_message = f"ERROR {exc!r}"
        st.error("Broker status refresh failed")
        st.exception(exc)
    broker_open_count = int(len(broker_portfolio)) if not broker_portfolio.empty else 0
    broker_status = "OK" if broker_open_count == 0 and str(broker_status_message).startswith("OK") else ("UNKNOWN" if not str(broker_status_message).startswith("OK") else "FAILED")

    sqlite_active = int(sqlite_scalar(sqlite_path, "SELECT COUNT(*) FROM positions WHERE active=1", default=0) or 0)
    sqlite_status = "OK" if sqlite_active == 0 else "FAILED"

    parquet_files = parquet_files_for_session(history_dir, previous_session)
    parquet_count = len(parquet_files)
    history_readiness = load_history_readiness_summary(
        history_dir=history_dir,
        universe_path=universe_csv,
        status_dir=collector_status_dir,
        session_date=previous_session,
        session_type="RTH",
    )
    universe_status = str(history_readiness.get("status") or "UNKNOWN")

    top100 = load_top100_readiness(
        top100_latest,
        expected_symbols=expected_top100,
        expected_source_session_date=previous_session,
    )
    top100_diag = load_top100_diagnostics_summary(Path(top100_latest).parent, previous_session)
    eod = load_eod_readiness(
        sqlite_path,
        readiness_date,
        broker_open_count=broker_open_count,
        sqlite_active_count=sqlite_active,
    )
    try:
        ops_snapshot = load_dashboard_snapshot(sqlite_path, DateWindow(readiness_date, readiness_date), "All")
    except Exception as exc:
        ops_snapshot = {"diagnostics": {}, "orphan_stale_positions": pd.DataFrame()}
        st.error("Operational readiness SQLite snapshot failed")
        st.exception(exc)
    orphan_stale_positions = ops_snapshot.get("orphan_stale_positions", pd.DataFrame())
    orphan_stale_count = int(len(orphan_stale_positions)) if isinstance(orphan_stale_positions, pd.DataFrame) else 0
    ready_for_next_session = all(
        status == "OK"
        for status in [
            broker_status,
            sqlite_status,
            universe_status,
            str(top100.get("status") or "UNKNOWN"),
            str(eod.get("status") or "UNKNOWN"),
        ]
    ) and orphan_stale_count == 0

    st.subheader("Ready for Next Session")
    ready_cols = st.columns([1, 3])
    with ready_cols[0]:
        status_badge("OK" if ready_for_next_session else "FAILED")
    ready_cols[1].caption(
        "Requires broker open positions=0, SQLite active positions=0, universe parquet OK, "
        "Top100 source session matching previous completed session, and EOD OK."
    )

    row1 = st.columns(2)
    with row1[0]:
        render_readiness_card(
            "Broker open positions",
            broker_open_count,
            broker_status,
            [
                f"source=IBKR API host={host} port={port} client_id={client_id}",
                f"message={broker_status_message}",
                "OK means broker has zero open positions before/after session.",
            ],
        )
    with row1[1]:
        render_readiness_card(
            "SQLite active positions",
            sqlite_active,
            sqlite_status,
            [
                "query=positions where active=1",
                "OK means SQLite has zero active positions before next session.",
            ],
        )

    row2 = st.columns(2)
    with row2[0]:
        render_readiness_card(
            "Universe data readiness",
            f"{history_readiness.get('complete_symbols', 0)}/{history_readiness.get('expected_symbols', 0)}",
            universe_status,
            [
                f"selected_date={readiness_date}",
                f"history_session_date={previous_session}",
                f"latest_completed_session={previous_session}",
                f"history_dir={history_dir}",
                f"collector_status_dir={collector_status_dir}",
                f"complete={history_readiness.get('complete_symbols')} no_data={history_readiness.get('no_data_symbols')} "
                f"partial={history_readiness.get('partial_symbols')} missing={history_readiness.get('missing_symbols')} "
                f"failed={history_readiness.get('failed_symbols')} completion_pct={history_readiness.get('completion_pct')}",
            ],
        )
    with row2[1]:
        render_readiness_card(
            "Top100 readiness",
            int(top100.get("symbols", 0) or 0),
            str(top100.get("status") or "UNKNOWN"),
            [
                f"path={top100.get('path')}",
                f"modified_at={top100.get('modified_at')}",
                f"top100_source_session_date={top100.get('top100_source_session_date')}",
                f"expected_source_session_date={top100.get('expected_source_session_date')}",
                f"diagnostics: missing={top100_diag.get('missing')} rejected={top100_diag.get('rejected')} "
                f"error={top100_diag.get('error')} excluded_ineligible={top100_diag.get('excluded_ineligible')}",
            ],
        )

    row3 = st.columns(2)
    with row3[0]:
        render_readiness_card(
            "EOD status",
            eod.get("last_final_status", "UNKNOWN"),
            str(eod.get("status") or "UNKNOWN"),
            [
                f"last_flatten_event={eod.get('last_flatten_event')} at={eod.get('last_flatten_at')}",
                f"last_final_at={eod.get('last_final_at')}",
                f"positions_verified_closed={eod.get('positions_verified_closed')}",
                f"broker_open_count={eod.get('broker_open_count')} sqlite_active_count={eod.get('sqlite_active_count')}",
                f"flat_confirmed={eod.get('flat_confirmed')} final_clean={eod.get('final_clean')}",
            ],
        )
    with row3[1]:
        render_readiness_card(
            "Orphan stale positions",
            orphan_stale_count,
            "OK" if orphan_stale_count == 0 else "FAILED",
            [
                f"oldest={ops_snapshot.get('diagnostics', {}).get('oldest_orphan_stale_position', '')}",
                f"age_days={ops_snapshot.get('diagnostics', {}).get('oldest_orphan_stale_position_age_days', 0.0)}",
                f"recommendation={ops_snapshot.get('diagnostics', {}).get('cleanup_recommendation', '')}",
            ],
        )

    row4 = st.columns(2)
    with row4[0]:
        health_code, health_body = get_json(f"{control_api_url}/health", timeout=3.0)
        health_status = "OK" if health_code == 200 else "UNKNOWN"
        render_readiness_card(
            "Control API",
            health_code,
            health_status,
            [
                f"url={control_api_url}/health",
                f"response={health_body[:240]}",
            ],
        )
    with row4[1]:
        st.markdown("### Cleanup Recommendation")
        if orphan_stale_count:
            st.error("Close stale orphan position")
            st.caption("Broker does not confirm this as a real open position. Cleanup marks stale SQLite rows inactive.")
        else:
            st.success("OK")

    with st.expander("Broker Portfolio Now", expanded=False):
        dataframe_or_info(broker_portfolio, "No broker portfolio positions.")
    with st.expander("Readiness raw diagnostics", expanded=False):
        st.json(
            {
                "readiness_date": readiness_date,
                "previous_completed_session": previous_session,
                "broker_open_positions": broker_open_count,
                "broker_status_message": broker_status_message,
                "sqlite_active_positions": sqlite_active,
                "parquet_count": parquet_count,
                "history_readiness": history_readiness,
                "top100": top100,
                "top100_diagnostics": top100_diag,
                "eod": eod,
                "orphan_stale_position_count": orphan_stale_count,
                "orphan_stale_positions": orphan_stale_positions.to_dict("records") if isinstance(orphan_stale_positions, pd.DataFrame) else [],
                "health_code": health_code,
                "health_body": health_body,
                "sample_parquet_files": [str(p) for p in parquet_files[:10]],
            }
        )

    st.divider()
    st.subheader("Manual Actions")
    st.warning("Manual actions can start long-running jobs. Use confirmation checkboxes intentionally.")
    action_cols = st.columns(4)

    with action_cols[0]:
        confirm_collector = st.checkbox("Confirm collector re-run", key="ops_confirm_collector")
        if st.button("Re-run universe collector for selected date", disabled=not confirm_collector, key="ops_run_collector"):
            payload = {
                "start_date": previous_session,
                "end_date": previous_session,
                "session_type": "RTH",
                "max_tasks": 3000,
                "max_attempts": 5,
                "force": True,
                "allow_live_session": False,
            }
            code, body = post_json(f"{control_api_url}/run_history_collector", payload, timeout=10.0)
            st.code(f"HTTP {code}\n{body}", language="json")

    with action_cols[1]:
        confirm_top100 = st.checkbox("Confirm Top100 rebuild", key="ops_confirm_top100")
        if st.button("Re-run Top100 build", disabled=not confirm_top100, key="ops_run_top100"):
            command = ["bash", str(REPO_ROOT / "scripts" / "build_daily_top100_premarket.sh"), previous_session]
            rc, output = run_dashboard_command(command, timeout=1200)
            st.code("$ " + " ".join(command) + f"\nreturncode={rc}\n{output}", language="bash")

    with action_cols[2]:
        confirm_reconciliation = st.checkbox("Confirm reconciliation refresh", key="ops_confirm_reconciliation")
        if st.button("Re-run reconciliation", disabled=not confirm_reconciliation, key="ops_run_reconciliation"):
            command = [
                sys.executable,
                "-m",
                "src.live_trading.order_lifecycle.reconciliation",
                "--json",
            ]
            rc, output = run_dashboard_command(command, timeout=120)
            code, body = get_json(f"{control_api_url}/health", timeout=5.0)
            st.code(
                "$ " + " ".join(command) + f"\nreturncode={rc}\n{output}\n\nCONTROL_API_HEALTH HTTP {code}\n{body}",
                language="json",
            )

    with action_cols[3]:
        confirm_cleanup = st.checkbox("Confirm stale orphan cleanup", key="ops_confirm_stale_orphan_cleanup")
        if st.button("Close stale orphan position", disabled=not confirm_cleanup or orphan_stale_count == 0, key="ops_cleanup_stale_orphan"):
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "cleanup_duplicate_stale_open_positions.py"),
                "--date",
                readiness_date,
                "--apply",
            ]
            rc, output = run_dashboard_command(command, timeout=120)
            st.code("$ " + " ".join(command) + f"\nreturncode={rc}\n{output}", language="bash")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Runtime Trading Dashboard")

    with st.sidebar:
        st.header("Data")
        sqlite_path = st.text_input("SQLite DB", value=resolve_sqlite_path(DEFAULT_SQLITE_PATH))
        sessions = cached_sessions(sqlite_path)
        today = utc_today()
        mode = st.radio("Session selector", ["Today", "Specific date", "Date range"], index=0)
        if mode == "Today":
            start_date = end_date = today
        elif mode == "Specific date":
            default = pd.to_datetime(sessions[0]).date() if sessions else date.today()
            selected = st.date_input("Date", value=default)
            start_date = end_date = selected.isoformat()
        else:
            default_end = pd.to_datetime(sessions[0]).date() if sessions else date.today()
            default_start = default_end - timedelta(days=7)
            start, end = st.date_input("Date range", value=(default_start, default_end))
            start_date, end_date = start.isoformat(), end.isoformat()
        strategies = ["All", *cached_strategies(sqlite_path, start_date, end_date)]
        strategy = st.selectbox("Strategy", strategies, index=0)
        include_reconstructed = st.toggle("Show execution-reconstructed trades", value=False, disabled=True)
        auto_refresh = st.toggle("Auto refresh current session", value=False)

    runtime_tab, broker_tab, readiness_tab = st.tabs([
        "Runtime Dashboard",
        "Broker Reality / IBKR Reconciliation",
        "Operational Readiness",
    ])
    with runtime_tab:
        render_runtime_tab(sqlite_path, start_date, end_date, strategy, include_reconstructed, auto_refresh)
    with broker_tab:
        render_broker_reality_tab(sqlite_path)
    with readiness_tab:
        render_operational_readiness_tab(sqlite_path, end_date)

    if auto_refresh and start_date == end_date == utc_today():
        st.caption("Auto refresh is intentionally disabled on Runtime Dashboard to avoid reading partial reducer state.")


if __name__ == "__main__":
    main()
