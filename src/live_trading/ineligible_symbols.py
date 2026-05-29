from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SYMBOL_DENYLIST = "data/config/symbol_denylist.csv"
DEFAULT_RUNTIME_INELIGIBLE = "data/runtime/ineligible_symbols.json"

DENYLIST_COLUMNS = ["symbol", "reason", "source", "first_seen_at", "last_seen_at", "notes"]
PRODUCT_KEYWORDS = (
    "ETF",
    "ETN",
    "ETP",
    "FUND",
    "TRUST",
    "BULL",
    "BEAR",
    "2X",
    "3X",
    "ULTRA",
    "DAILY",
    "LEVERAGED",
    "INVERSE",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").upper().strip()


def _coerce_path(path: str | Path | None, default: str) -> Path:
    return Path(path or default)


def load_symbol_denylist(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = _coerce_path(path, DEFAULT_SYMBOL_DENYLIST)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            out[symbol] = {
                "symbol": symbol,
                "reason": row.get("reason") or "denylisted",
                "source": row.get("source") or "symbol_denylist",
                "first_seen_at": row.get("first_seen_at") or "",
                "last_seen_at": row.get("last_seen_at") or "",
                "notes": row.get("notes") or "",
            }
    return out


def write_symbol_denylist(rows: dict[str, dict[str, Any]], path: str | Path | None = None) -> None:
    p = _coerce_path(path, DEFAULT_SYMBOL_DENYLIST)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DENYLIST_COLUMNS)
        writer.writeheader()
        for symbol in sorted(rows):
            row = dict(rows[symbol])
            row["symbol"] = symbol
            writer.writerow({key: row.get(key, "") for key in DENYLIST_COLUMNS})


def add_symbol_denylist(
    symbol: Any,
    reason: str,
    *,
    source: str = "manual",
    notes: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)
    if not symbol:
        raise ValueError("symbol is required")
    rows = load_symbol_denylist(path)
    now = utc_now()
    existing = rows.get(symbol, {})
    rows[symbol] = {
        "symbol": symbol,
        "reason": reason,
        "source": source,
        "first_seen_at": existing.get("first_seen_at") or now,
        "last_seen_at": now,
        "notes": notes if notes != "" else existing.get("notes", ""),
    }
    write_symbol_denylist(rows, path)
    return rows[symbol]


def remove_symbol_denylist(symbol: Any, path: str | Path | None = None) -> bool:
    symbol = normalize_symbol(symbol)
    rows = load_symbol_denylist(path)
    existed = symbol in rows
    if existed:
        rows.pop(symbol, None)
        write_symbol_denylist(rows, path)
    return existed


def load_runtime_ineligible(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = _coerce_path(path, DEFAULT_RUNTIME_INELIGIBLE)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        rows = data.get("symbols", data)
    else:
        rows = data
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, dict):
        iterable = rows.values()
    elif isinstance(rows, list):
        iterable = rows
    else:
        iterable = []
    for row in iterable:
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        out[symbol] = {**row, "symbol": symbol}
    return out


def write_runtime_ineligible(rows: dict[str, dict[str, Any]], path: str | Path | None = None) -> None:
    p = _coerce_path(path, DEFAULT_RUNTIME_INELIGIBLE)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"symbols": {symbol: rows[symbol] for symbol in sorted(rows)}}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_runtime_ineligible(
    symbol: Any,
    *,
    con_id: Any = None,
    reason: str = "kid_priip_ineligible",
    ibkr_error_code: Any = None,
    raw_message: str = "",
    source: str = "ibkr_error_201",
    path: str | Path | None = None,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)
    if not symbol:
        raise ValueError("symbol is required")
    rows = load_runtime_ineligible(path)
    now = utc_now()
    existing = rows.get(symbol, {})
    row = {
        "symbol": symbol,
        "conId": con_id if con_id not in (None, "") else existing.get("conId"),
        "reason": reason,
        "ibkr_error_code": ibkr_error_code if ibkr_error_code not in (None, "") else existing.get("ibkr_error_code"),
        "first_seen_at": existing.get("first_seen_at") or now,
        "last_seen_at": now,
        "raw_message": raw_message or existing.get("raw_message", ""),
        "source": source,
    }
    rows[symbol] = row
    write_runtime_ineligible(rows, path)
    return row


def combined_ineligible_symbols(
    denylist_path: str | Path | None = None,
    runtime_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    combined = load_runtime_ineligible(runtime_path)
    for symbol, row in load_symbol_denylist(denylist_path).items():
        combined[symbol] = row
    return combined


def contract_metadata(contract: Any) -> dict[str, Any]:
    return {
        "symbol": normalize_symbol(getattr(contract, "symbol", "")),
        "conId": getattr(contract, "conId", None),
        "secType": getattr(contract, "secType", None),
        "longName": getattr(contract, "longName", None),
        "category": getattr(contract, "category", None),
        "industry": getattr(contract, "industry", None),
        "primaryExchange": getattr(contract, "primaryExchange", None),
        "tradingClass": getattr(contract, "tradingClass", None),
    }


def contract_ineligible_reason(contract: Any) -> str | None:
    metadata = contract_metadata(contract)
    sec_type = str(metadata.get("secType") or "STK").upper().strip()
    if sec_type and sec_type != "STK":
        return f"non_stock_sectype:{sec_type}"
    haystack = " ".join(str(value or "") for value in metadata.values()).upper()
    for keyword in PRODUCT_KEYWORDS:
        if keyword in haystack:
            return f"product_keyword:{keyword.lower()}"
    return None

