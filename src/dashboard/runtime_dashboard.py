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
def load_live_snapshot(sqlite_path: str, start_date: str, end_date: str, strategy: str) -> dict:
    return load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), strategy)


@st.cache_data(ttl=3600)
def load_historical_snapshot(sqlite_path: str, start_date: str, end_date: str, strategy: str) -> dict:
    return load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), strategy)


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


def render_open_positions(df: pd.DataFrame) -> None:
    st.subheader("Open Positions")
    if df.empty:
        st.info("No open positions in the selected window.")
        return
    cols = [
        "symbol", "qty", "buy", "now", "upnl", "now_pct", "peak_pct", "giveback_pct",
        "hold_minutes", "entry_time", "entry_source", "status", "strategy",
    ]
    out = filter_table(df[cols].copy(), "open").sort_values(["upnl", "symbol"], na_position="last")
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
        }
    )
    out = out[["Symbol", "Qty", "Buy", "Now", "UPNL", "Now %", "Peak %", "Drop from Peak %", "Min", "Entry Time", "Status", "Strategy"]]
    st.dataframe(
        style_pnl(out, ["UPNL", "Now %"]),
        width="stretch",
        hide_index=True,
    )


def render_closed_positions(df: pd.DataFrame) -> None:
    st.subheader("Closed Positions")
    if df.empty:
        st.info("No closed positions in the selected window.")
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
    cols = st.columns(7)
    labels = [
        ("Orphans", "orphans"),
        ("Missing IBKR", "missing_in_ibkr"),
        ("Partial Exits", "partial_exits"),
        ("Delayed Fills", "delayed_fills"),
        ("Risk Blocks", "risk_guard_blocks"),
        ("SQLite Failures", "sqlite_failures"),
        ("Reconnects", "reconnect_events"),
    ]
    for col, (label, key) in zip(cols, labels):
        col.metric(label, int(diag.get(key, 0)))
    pos_cols = st.columns(4)
    position_labels = [
        ("SQLite Active Rows", "sqlite_active_positions_count"),
        ("Latest Active", "latest_active_positions_count"),
        ("IBKR Positions", "ibkr_positions_count"),
        ("Stale Active Rows", "stale_active_positions_count"),
    ]
    for col, (label, key) in zip(pos_cols, position_labels):
        col.metric(label, int(diag.get(key, 0)))


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
        auto_refresh = st.toggle("Auto refresh current session", value=True)

    window = DateWindow(start_date, end_date)
    current = is_current_window(window)
    if current:
        snapshot = load_live_snapshot(sqlite_path, start_date, end_date, strategy)
    else:
        snapshot = load_historical_snapshot(sqlite_path, start_date, end_date, strategy)

    loaded_at = snapshot.get("loaded_at", "")
    st.caption(f"SQLite: `{snapshot.get('source')}` | window: {start_date} to {end_date} | strategy: {strategy} | loaded_at: {loaded_at}")

    render_summary(snapshot["summary"])
    st.divider()
    render_data_quality_summary(snapshot.get("data_quality_summary", {}))
    st.divider()
    render_open_positions(snapshot["open_positions"])
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


if __name__ == "__main__":
    main()
