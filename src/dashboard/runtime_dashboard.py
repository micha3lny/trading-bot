from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.runtime_queries import (  # noqa: E402
    DateWindow,
    list_sessions,
    list_strategies,
    load_dashboard_snapshot,
    utc_today,
)
from src.dashboard.broker_reality import (  # noqa: E402
    ReconciliationResult,
    empty_broker_executions,
    fetch_ibkr_executions_for_date,
    fetch_ibkr_live_portfolio,
    load_sqlite_active_positions,
    load_sqlite_executions,
    load_sqlite_trade_pnl,
    parse_ibkr_activity_csv,
    reconcile_broker_vs_sqlite,
    reconciliation_export_frames,
)
from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, resolve_sqlite_path  # noqa: E402


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
def load_live_snapshot(sqlite_path: str, start_date: str, end_date: str, strategy: str, include_reconstructed: bool) -> dict:
    return load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), strategy, include_reconstructed=include_reconstructed)


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


def render_data_quality_summary(summary: dict) -> None:
    st.subheader("Closed Trade Data Quality")
    cols = st.columns(6)
    cols[0].metric("Closed", int(summary.get("closed_trades_count", 0)))
    cols[1].metric("Comm OK", int(summary.get("commission_ok", 0)))
    cols[2].metric("Comm Partial", int(summary.get("commission_partial", 0)))
    cols[3].metric("Comm Missing", int(summary.get("commission_missing", 0)))
    cols[4].metric("Peak Missing", int(summary.get("peak_missing", 0)))
    cols[5].metric("Warnings", int(summary.get("data_quality_warning_count", 0)))


def render_open_positions(df: pd.DataFrame, *, title: str = "Open Positions", prefix: str = "open") -> None:
    if title:
        st.subheader(title)
    if df.empty:
        st.info("No open positions in the selected window.")
        return
    cols = [
        "symbol", "qty", "buy", "now", "upnl", "now_pct", "peak_pct", "giveback_pct",
        "hold_minutes", "entry_time", "entry_source", "status", "strategy", "data_quality", "ibkr_confirmed",
    ]
    available = [col for col in cols if col in df.columns]
    out = filter_table(df[available].copy(), prefix).sort_values(["upnl", "symbol"], na_position="last")
    out["entry_time"] = out.apply(
        lambda row: f"ADOPTED {display_time(row['entry_time'])}" if str(row.get("entry_source") or "").upper() == "ADOPTED" else display_time(row["entry_time"]),
        axis=1,
    )
    for col in ("now", "upnl", "now_pct"):
        out[col] = out[col].map(display_optional_number)
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "qty": "Qty",
            "buy": "Buy",
            "now": "Now",
            "upnl": "UPNL",
            "now_pct": "Now %",
            "peak_pct": "Peak %",
            "giveback_pct": "Drop from Peak %",
            "hold_minutes": "Min",
            "entry_time": "Entry Time",
            "status": "Status",
            "strategy": "Strategy",
            "data_quality": "Data Quality",
            "ibkr_confirmed": "IBKR Confirmed",
        }
    )
    display_cols = ["Symbol", "Qty", "Buy", "Now", "UPNL", "Now %", "Peak %", "Drop from Peak %", "Min", "Entry Time", "Status", "Strategy"]
    if "Data Quality" in out.columns:
        display_cols.append("Data Quality")
    if "IBKR Confirmed" in out.columns:
        display_cols.append("IBKR Confirmed")
    out = out[display_cols]
    st.dataframe(
        style_pnl(out, ["UPNL", "Now %"]),
        width="stretch",
        hide_index=True,
    )


def render_open_position_sections(df: pd.DataFrame) -> None:
    if df.empty or "position_bucket" not in df.columns:
        render_open_positions(df)
        return
    today = df[df["position_bucket"].fillna("") == "today"].copy()
    carry = df[df["position_bucket"].fillna("") == "carry_stale"].copy()
    if today.empty:
        st.subheader("Today Open Positions")
        st.info("No today open positions in the selected window.")
    else:
        render_open_positions(today, title="Today Open Positions", prefix="today_open")
    if carry.empty:
        st.subheader("Carry / Stale Open Positions")
        st.info("No carry/stale open positions requiring verification.")
    else:
        render_open_positions(carry, title="Carry / Stale Open Positions", prefix="carry_open")


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


def render_closed_positions(df: pd.DataFrame) -> None:
    st.subheader("Closed Positions")
    if df.empty:
        st.info("No confirmed closed trades in trades table.")
        return
    cols = [
        "symbol", "entry_date", "exit_date", "qty", "ibkr_commission", "commission_status", "buy", "sell", "gross", "net_actual", "net_pct", "peak_pct",
        "drop_from_peak_pct", "hold_minutes", "exit_reason", "strategy",
        "entry_time", "exit_time", "data_quality",
    ]
    out = filter_table(df[cols].copy(), "closed").sort_values(["net_actual", "symbol"], na_position="last")
    out["entry_time"] = out["entry_time"].map(display_time)
    out["exit_time"] = out["exit_time"].map(display_time)
    out["peak_pct"] = out["peak_pct"].map(display_number_or_missing)
    out["drop_from_peak_pct"] = out["drop_from_peak_pct"].map(display_number_or_missing)
    out["hold_minutes"] = out["hold_minutes"].map(display_number_or_missing)
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
            "drop_from_peak_pct": "Drop from Peak %",
            "hold_minutes": "Min",
            "exit_reason": "Exit Reason",
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "data_quality": "Data Quality",
            "strategy": "Strategy",
        }
    )
    out = out[
        [
            "Symbol", "Quantity", "IBKR Comm", "Commission Status", "Buy", "Sell", "Gross", "Net", "Net %",
            "Peak %", "Drop from Peak %", "Min", "Exit Reason", "Entry Date", "Exit Date", "Entry Time", "Exit Time", "Strategy",
            "Data Quality",
        ]
    ]
    st.dataframe(
        style_pnl(out, ["Gross", "Net", "Net %"]),
        width="stretch",
        hide_index=True,
    )
    debug_cols = [
        "trade_id", "symbol", "entry_execution_count", "exit_execution_count",
        "confirmed_commission_execution_count", "expected_commission_execution_count",
        "peak_source", "peak_match_quality", "commission_source_detail", "closed_source", "data_quality",
    ]
    available_debug_cols = [col for col in debug_cols if col in df.columns]
    if available_debug_cols:
        with st.expander("Closed trade source diagnostics", expanded=False):
            debug = df[available_debug_cols].copy().rename(
                columns={
                    "trade_id": "Trade ID",
                    "symbol": "Symbol",
                    "entry_execution_count": "Entry Execs",
                    "exit_execution_count": "Exit Execs",
                    "confirmed_commission_execution_count": "Confirmed Comm Execs",
                    "expected_commission_execution_count": "Expected Comm Execs",
                    "peak_source": "Peak Source",
                    "peak_match_quality": "Peak Match Quality",
                    "commission_source_detail": "Commission Source Detail",
                    "closed_source": "Closed Source",
                    "data_quality": "Data Quality",
                }
            )
            st.dataframe(debug, width="stretch", hide_index=True)


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
    pos_cols = st.columns(6)
    position_labels = [
        ("SQLite Active Rows", "sqlite_active_positions_count"),
        ("Latest Active", "latest_active_positions_count"),
        ("Today Open", "today_open_positions_count"),
        ("Carry/Stale Open", "stale_carry_open_count"),
        ("Duplicate Symbols", "duplicate_active_symbol_count"),
        ("IBKR Positions", "ibkr_positions_count"),
    ]
    for col, (label, key) in zip(pos_cols, position_labels):
        col.metric(label, int(diag.get(key, 0)))
    closed_cols = st.columns(4)
    closed_labels = [
        ("Persisted Closed", "persisted_closed_trades_count"),
        ("Reconstructed Pairs", "reconstructed_execution_pairs_count"),
        ("Displayed Closed", "displayed_closed_trades_count"),
        ("Carried Closed Today", "carried_closed_today_count"),
    ]
    for col, (label, key) in zip(closed_cols, closed_labels):
        col.metric(label, int(diag.get(key, 0)))


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


def empty_reconciliation_result(selected_date: str, broker_status: str = "NOT_LOADED") -> ReconciliationResult:
    summary = {
        "status": "CSV_REQUIRED_FOR_HISTORICAL_DATE" if broker_status != "OK" else "NOT_RECONCILED",
        "broker_status": broker_status,
        "broker_execution_count": 0,
        "sqlite_execution_count": 0,
        "matched_executions": 0,
        "missing_in_sqlite": 0,
        "extra_in_sqlite": 0,
        "execution_mismatches": 0,
        "position_mismatches": 0,
    }
    return ReconciliationResult(
        summary=summary,
        matched=pd.DataFrame(),
        missing_in_sqlite=pd.DataFrame(),
        extra_in_sqlite=pd.DataFrame(),
        execution_mismatches=pd.DataFrame(),
        position_mismatches=pd.DataFrame(),
        pnl_comparison=pd.DataFrame(
            [
                {
                    "selected_date": selected_date,
                    "broker_realized_pnl": 0.0,
                    "broker_commission": 0.0,
                    "sqlite_gross": 0.0,
                    "sqlite_commission": 0.0,
                    "sqlite_net": 0.0,
                    "sqlite_closed_trades": 0,
                    "commission_delta": 0.0,
                }
            ]
        ),
    )


def render_runtime_tab(sqlite_path: str, start_date: str, end_date: str, strategy: str, include_reconstructed: bool, auto_refresh: bool) -> None:
    window = DateWindow(start_date, end_date)
    current = is_current_window(window)
    if current:
        snapshot = load_live_snapshot(sqlite_path, start_date, end_date, strategy, include_reconstructed)
    else:
        snapshot = load_historical_snapshot(sqlite_path, start_date, end_date, strategy, include_reconstructed)

    loaded_at = snapshot.get("loaded_at", "")
    st.caption(f"SQLite: `{snapshot.get('source')}` | window: {start_date} to {end_date} | strategy: {strategy} | loaded_at: {loaded_at}")

    render_summary(snapshot["summary"])
    st.divider()
    render_data_quality_summary(snapshot.get("data_quality_summary", {}))
    if int(snapshot.get("diagnostics", {}).get("execution_reconstruction_disabled", 0) or 0):
        st.warning("EXECUTION_RECONSTRUCTION_DISABLED_IN_DASHBOARD: executions exist, but no persisted closed trades exist for this window.")
    st.divider()
    render_open_position_sections(snapshot["open_positions"])
    st.divider()
    render_rejected_entries(snapshot.get("rejected_entries", pd.DataFrame()))
    st.divider()
    render_closed_positions(snapshot["closed_positions"])
    st.divider()
    render_exit_simulation(snapshot["exit_simulation"])
    st.divider()
    render_peak_charts(snapshot["closed_positions"])
    st.divider()
    render_diagnostics(snapshot["diagnostics"])

    if current and auto_refresh:
        st.markdown('<p class="small-note">Auto refresh active: rerendering every 5 seconds for current session.</p>', unsafe_allow_html=True)
        time.sleep(5)
        st.rerun()


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
        port = col2.number_input("IBKR port", value=7497, step=1, key="broker_port")
        client_id = col3.number_input("Client ID", value=177, step=1, key="broker_client_id")
        timeout = col4.number_input("Timeout seconds", value=4.0, step=0.5, key="broker_timeout")
        use_api_executions = st.toggle("Try IBKR API executions for selected date", value=False)
        uploaded_csv = st.file_uploader("IBKR Activity Statement / Trades CSV fallback", type=["csv", "txt"], key="broker_activity_csv")
        csv_path = st.text_input("Or Activity CSV file path on server", value="", key="broker_activity_csv_path")

    portfolio = pd.DataFrame()
    portfolio_status = "NOT_LOADED"
    broker_executions = empty_broker_executions()
    broker_status = "CSV_REQUIRED_FOR_HISTORICAL_DATE"
    sqlite_executions = empty_broker_executions()
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
            broker_executions, broker_status = fetch_ibkr_executions_for_date(host, int(port), int(client_id), selected_date, float(timeout))
            api_executions_count = len(broker_executions)
        except Exception as exc:
            broker_executions = empty_broker_executions()
            broker_status = f"ibkr_executions_exception: {exc}"
            st.error("IBKR execution request failed")
            st.exception(exc)
    if uploaded_csv is not None:
        try:
            broker_executions = parse_ibkr_activity_csv(uploaded_csv)
            broker_executions = broker_executions[broker_executions["execution_time"].map(lambda x: str(x)[:10]) == selected_date].reset_index(drop=True)
            broker_status = "OK"
            csv_loaded = True
        except Exception as exc:
            broker_executions = empty_broker_executions()
            broker_status = f"csv_upload_parse_error: {exc}"
            st.error("IBKR Activity CSV parse failed")
            st.exception(exc)
    elif csv_path.strip():
        path = Path(csv_path.strip()).expanduser()
        if path.exists():
            try:
                broker_executions = parse_ibkr_activity_csv(path.read_text())
                broker_executions = broker_executions[broker_executions["execution_time"].map(lambda x: str(x)[:10]) == selected_date].reset_index(drop=True)
                broker_status = "OK"
                csv_loaded = True
            except Exception as exc:
                broker_executions = empty_broker_executions()
                broker_status = f"csv_path_parse_error: {exc}"
                st.error("IBKR Activity CSV path parse failed")
                st.exception(exc)
        else:
            broker_status = f"csv_path_not_found: {path}"

    try:
        sqlite_executions = load_sqlite_executions(sqlite_path, selected_date)
        sqlite_positions = load_sqlite_active_positions(sqlite_path, selected_date)
        sqlite_trade_pnl = load_sqlite_trade_pnl(sqlite_path, selected_date)
    except Exception as exc:
        st.error("SQLite reconciliation data load failed")
        st.exception(exc)

    try:
        result = reconcile_broker_vs_sqlite(
            broker_executions,
            sqlite_executions,
            portfolio,
            sqlite_positions,
            sqlite_trade_pnl,
            selected_date=selected_date,
            broker_status=broker_status,
        )
    except Exception as exc:
        result = empty_reconciliation_result(selected_date, broker_status)
        st.error("Broker vs SQLite reconciliation failed")
        st.exception(exc)

    ibkr_connected = portfolio_status == "OK"
    account = ""
    if not portfolio.empty and "account" in portfolio.columns:
        account = ", ".join(sorted(str(x) for x in portfolio["account"].dropna().unique() if str(x)))

    st.subheader("Broker Debug")
    dbg = st.columns(7)
    dbg[0].metric("IBKR connected", "true" if ibkr_connected else "false")
    dbg[1].metric("Account", account or "N/A")
    dbg[2].metric("Client ID", int(client_id))
    dbg[3].metric("Selected date", selected_date)
    dbg[4].metric("API executions", api_executions_count)
    dbg[5].metric("SQLite executions", len(sqlite_executions))
    dbg[6].metric("CSV loaded", "true" if csv_loaded else "false")

    if debug_broker:
        with st.expander("Raw broker tab diagnostics", expanded=True):
            st.write("portfolio_type", type(portfolio))
            st.write("portfolio_rows", len(portfolio))
            st.write("broker_executions_type", type(broker_executions))
            st.write("broker_executions_rows", len(broker_executions))
            st.write("sqlite_executions_type", type(sqlite_executions))
            st.write("sqlite_executions_rows", len(sqlite_executions))
            st.write("sqlite_positions_rows", len(sqlite_positions))
            st.write("portfolio_status", portfolio_status)
            st.write("broker_status", broker_status)

    st.subheader("Broker Portfolio")
    st.caption(f"portfolio_status={portfolio_status}")
    if portfolio_status != "OK":
        st.warning(f"IBKR connection unavailable: {portfolio_status}")
    if portfolio.empty:
        st.info("No portfolio positions")
    else:
        st.dataframe(portfolio, width="stretch", hide_index=True)

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
    cols = st.columns(6)
    for col, (label, key) in zip(
        cols,
        [
            ("Broker Execs", "broker_execution_count"),
            ("SQLite Execs", "sqlite_execution_count"),
            ("Matched", "matched_executions"),
            ("Missing SQLite", "missing_in_sqlite"),
            ("Extra SQLite", "extra_in_sqlite"),
            ("Position Mismatches", "position_mismatches"),
        ],
    ):
        col.metric(label, int(result.summary.get(key, 0) or 0))
    st.caption(f"broker_status={broker_status}")
    if status == "CSV_REQUIRED_FOR_HISTORICAL_DATE":
        st.warning("CSV_REQUIRED_FOR_HISTORICAL_DATE")
    if broker_executions.empty:
        st.info("No executions for selected date")

    st.subheader("Broker Executions")
    dataframe_or_info(broker_executions, "No broker executions loaded. Use IBKR API or Activity CSV fallback.")

    st.subheader("SQLite Executions")
    dataframe_or_info(sqlite_executions, "No SQLite executions for selected date.")

    st.subheader("Mismatches")
    mismatch_tabs = st.tabs(["Missing in SQLite", "Extra in SQLite", "Execution Mismatches", "Position Mismatches", "Trades / PnL"])
    with mismatch_tabs[0]:
        dataframe_or_info(result.missing_in_sqlite, "No broker executions missing in SQLite.")
    with mismatch_tabs[1]:
        dataframe_or_info(result.extra_in_sqlite, "No extra SQLite executions.")
    with mismatch_tabs[2]:
        dataframe_or_info(result.execution_mismatches, "No quantity/price/commission mismatches.")
    with mismatch_tabs[3]:
        dataframe_or_info(result.position_mismatches, "No position mismatches.")
    with mismatch_tabs[4]:
        dataframe_or_info(result.pnl_comparison, "No PnL comparison available.")

    export_csv = export_reconciliation_csv(reconciliation_export_frames(result))
    st.download_button(
        "Export reconciliation report CSV",
        data=export_csv,
        file_name=f"ibkr_reconciliation_{selected_date}.csv",
        mime="text/csv",
    )


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
        include_reconstructed = st.toggle("Show execution-reconstructed trades", value=False)
        auto_refresh = st.toggle("Auto refresh current session", value=True)

    runtime_tab, broker_tab = st.tabs(["Runtime Dashboard", "Broker Reality / IBKR Reconciliation"])
    with runtime_tab:
        render_runtime_tab(sqlite_path, start_date, end_date, strategy, include_reconstructed, auto_refresh)
    with broker_tab:
        render_broker_reality_tab(sqlite_path)


if __name__ == "__main__":
    main()
