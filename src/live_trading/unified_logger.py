from __future__ import annotations

import gzip
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TextIO


DEFAULT_LOG_DIR = "data/logs"
LOG_PREFIX = "trading-bot-"
_INSTALLED = False
_LOG_DIR: Path | None = None


def utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_log_dir(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("TRADING_BOT_LOG_DIR") or DEFAULT_LOG_DIR)


def daily_log_path(log_dir: str | Path | None = None, day: date | None = None) -> Path:
    day = day or datetime.now(timezone.utc).date()
    return resolve_log_dir(log_dir) / f"{LOG_PREFIX}{day.isoformat()}.log"


def _coerce_timestamp(value: str | None) -> str:
    if not value:
        return utc_now_z()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return utc_now_z()


def _infer_component(event: str) -> str:
    upper = event.upper()
    if upper.startswith(("EOD_", "FRACTIONAL_ORPHAN")):
        return "EOD"
    if upper.startswith(("RISK_", "DISK_USAGE")):
        return "RISK"
    if upper.startswith(("CONTROL_API_", "CONTROL_")):
        return "CONTROL"
    if upper.startswith(("HISTORY_", "OVERNIGHT_")):
        return "COLLECTOR"
    if upper.startswith("DAILY_TOP100"):
        return "TOP100"
    if upper.startswith(("SQLITE_", "BACKFILL_SQLITE")):
        return "SQLITE"
    if upper.startswith(("RECONCILIATION_", "STARTUP_RECONCILIATION")):
        return "RECONCILIATION"
    if upper.startswith(("BUY_", "SELL_", "EXIT_", "ENTRY_", "ORDER_", "PAPER")):
        return "ORDER"
    if upper.startswith(("IBKR_", "RECONNECT_")):
        return "IBKR"
    if upper in {"BOT_START", "BOT_STOP", "BOT_CRASH"}:
        return "BOT"
    return "RUNTIME"


def _infer_level(event: str, stream_level: str = "INFO") -> str:
    upper = event.upper()
    if stream_level == "ERROR":
        return "ERROR"
    if any(token in upper for token in ("CRITICAL", "CRASH")):
        return "CRITICAL"
    if any(token in upper for token in ("FAILED", "FAILURE", "ERROR", "EXCEPTION")):
        return "ERROR"
    if any(token in upper for token in ("WARN", "WARNING", "GIVEUP", "MISSING", "BLOCK")):
        return "WARN"
    return "INFO"


_TS_LINE_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\S+)\s+(?P<event>[A-Za-z0-9_./:-]+)(?P<tail>.*)$")
_EVENT_LINE_RE = re.compile(r"^(?P<event>[A-Z][A-Z0-9_]+)(?P<tail>.*)$")


def normalize_log_line(line: str, *, stream_level: str = "INFO") -> str:
    stripped = line.rstrip("\n")
    if not stripped:
        return ""
    match = _TS_LINE_RE.match(stripped)
    if match:
        ts = _coerce_timestamp(match.group("ts"))
        event = match.group("event")
        tail = match.group("tail").strip()
    else:
        match = _EVENT_LINE_RE.match(stripped)
        ts = utc_now_z()
        if match:
            event = match.group("event")
            tail = match.group("tail").strip()
        else:
            event = "LOG"
            tail = "message=" + repr(stripped)
    level = _infer_level(event, stream_level=stream_level)
    component = _infer_component(event)
    return f"{ts} {level} {component} {event}" + (f" {tail}" if tail else "")


def append_unified_log(line: str, *, log_dir: str | Path | None = None, stream_level: str = "INFO") -> None:
    normalized = normalize_log_line(line, stream_level=stream_level)
    if not normalized:
        return
    try:
        path = daily_log_path(log_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(normalized + "\n")
            fh.flush()
    except Exception:
        return


def emit_unified_log_line(line: str, *, log_dir: str | Path | None = None, stream_level: str = "INFO") -> None:
    """Emit one runtime line to stdout/stderr and the unified log without tee duplication."""
    stream = sys.stderr if stream_level == "ERROR" else sys.stdout
    if isinstance(stream, _UnifiedLogTee):
        try:
            stream.wrapped.write(line + "\n")
            stream.wrapped.flush()
        except Exception:
            pass
        append_unified_log(line, log_dir=log_dir or stream.log_dir, stream_level=stream_level)
        return
    print(line, file=stream, flush=True)
    append_unified_log(line, log_dir=log_dir, stream_level=stream_level)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if not text or any(ch.isspace() for ch in text):
        return repr(text)
    return text


def log_event(component: str, event: str, level: str = "INFO", *, log_dir: str | Path | None = None, **fields: Any) -> None:
    tail = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
    line = f"{utc_now_z()} {level.upper()} {component.upper()} {event.upper()}" + (f" {tail}" if tail else "")
    try:
        path = daily_log_path(log_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception:
        return


class _UnifiedLogTee:
    def __init__(self, wrapped: TextIO, *, log_dir: Path, stream_level: str) -> None:
        self.wrapped = wrapped
        self.log_dir = log_dir
        self.stream_level = stream_level
        self._buffer = ""

    def write(self, text: str) -> int:
        written = self.wrapped.write(text)
        self.wrapped.flush()
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            append_unified_log(line, log_dir=self.log_dir, stream_level=self.stream_level)
        return written

    def flush(self) -> None:
        self.wrapped.flush()
        if self._buffer:
            append_unified_log(self._buffer, log_dir=self.log_dir, stream_level=self.stream_level)
            self._buffer = ""

    def isatty(self) -> bool:
        return self.wrapped.isatty()

    @property
    def encoding(self) -> str | None:
        return self.wrapped.encoding

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


def run_log_retention(log_dir: str | Path | None = None, *, keep_days: int = 14, delete_gz_after_days: int = 30) -> None:
    root = resolve_log_dir(log_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).date()
        for path in root.glob(f"{LOG_PREFIX}*.log"):
            try:
                day = datetime.strptime(path.stem.removeprefix(LOG_PREFIX), "%Y-%m-%d").date()
            except Exception:
                continue
            if (today - day).days > keep_days:
                gz_path = path.with_suffix(path.suffix + ".gz")
                if not gz_path.exists():
                    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                path.unlink(missing_ok=True)
        for path in root.glob(f"{LOG_PREFIX}*.log.gz"):
            try:
                name = path.name.removeprefix(LOG_PREFIX).removesuffix(".log.gz")
                day = datetime.strptime(name, "%Y-%m-%d").date()
            except Exception:
                continue
            if (today - day).days > delete_gz_after_days:
                path.unlink(missing_ok=True)
    except Exception:
        return


def install_unified_logger(log_dir: str | Path | None = None) -> Path:
    global _INSTALLED, _LOG_DIR
    resolved = resolve_log_dir(log_dir)
    os.environ["TRADING_BOT_LOG_DIR"] = str(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    run_log_retention(resolved)
    if not _INSTALLED:
        sys.stdout = _UnifiedLogTee(sys.stdout, log_dir=resolved, stream_level="INFO")  # type: ignore[assignment]
        sys.stderr = _UnifiedLogTee(sys.stderr, log_dir=resolved, stream_level="ERROR")  # type: ignore[assignment]
        _INSTALLED = True
    _LOG_DIR = resolved
    log_event("LOG", "UNIFIED_LOGGER_ACTIVE", log_dir=resolved, path=daily_log_path(resolved))
    return resolved


def unified_logger_installed() -> bool:
    return _INSTALLED


def current_git_commit(cwd: str | Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass
    return "unknown"


def disk_usage_pct(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return round((usage.used / usage.total) * 100.0, 2)


def monitor_disk_usage(
    path: str | Path,
    runtime_state: dict[str, Any] | None = None,
    *,
    warning_pct: float = 85.0,
    critical_pct: float = 95.0,
    block_pct: float = 98.0,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    result = {"usage_pct": 0.0, "level": "OK", "block_entries": False}
    try:
        pct = disk_usage_pct(path)
    except Exception as exc:
        log_event("RISK", "DISK_USAGE_CHECK_FAILED", "WARN", log_dir=log_dir, error=repr(exc), path=path)
        return result
    result["usage_pct"] = pct
    if pct >= block_pct:
        result["level"] = "CRITICAL"
        result["block_entries"] = True
    elif pct >= critical_pct:
        result["level"] = "CRITICAL"
    elif pct >= warning_pct:
        result["level"] = "WARNING"
    if runtime_state is not None:
        runtime_state["disk_usage_pct"] = pct
        runtime_state["disk_full_entries_blocked"] = bool(result["block_entries"])
    if result["level"] != "OK":
        event = "DISK_USAGE_CRITICAL" if result["level"] == "CRITICAL" else "DISK_USAGE_WARNING"
        level = "CRITICAL" if result["level"] == "CRITICAL" else "WARN"
        log_event("RISK", event, level, log_dir=log_dir, path=path, usage_pct=pct, block_entries=int(bool(result["block_entries"])))
    return result


def format_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).replace("\n", "\\n")
