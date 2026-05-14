from __future__ import annotations

import re
from pathlib import Path

P = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")

HELPER = r'''

def v68_history_utc_minutes(value: str) -> int:
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    return hh * 60 + mm


def v68_history_in_utc_window(start_hhmm: str, end_hhmm: str) -> bool:
    t = datetime.now(timezone.utc).time()
    now_min = t.hour * 60 + t.minute
    start = v68_history_utc_minutes(start_hhmm)
    end = v68_history_utc_minutes(end_hhmm)
    if end <= start:
        return now_min >= start or now_min < end
    return start <= now_min < end


def v68_history_trading_window_active(args: argparse.Namespace) -> bool:
    return v68_history_in_utc_window(
        getattr(args, "market_open_utc", "13:30"),
        getattr(args, "history_collector_market_close_utc", "20:00"),
    )


def run_v68_history_collector_once_if_due(args: argparse.Namespace, runtime_state: dict[str, Any]) -> int:
    if not getattr(args, "history_collector_enabled", False):
        return 0
    if v68_history_trading_window_active(args):
        return 0
    if not v68_history_in_utc_window(
        getattr(args, "history_collector_start_utc", "20:15"),
        getattr(args, "history_collector_end_utc", "23:00"),
    ):
        return 0

    today = datetime.now(timezone.utc).date().isoformat()
    session_type = str(getattr(args, "history_collector_session_type", "RTH")).upper()
    key = f"{today}_{session_type}"
    if runtime_state.get("history_collector_last_run_key") == key:
        return 0
    if runtime_state.get("history_collector_running"):
        return 0

    cmd = [
        sys.executable,
        "-m",
        "src.live_trading.data.v68_universe_1m_parquet_collector",
        "--start-date",
        str(getattr(args, "history_collector_start_date", "2026-01-01")),
        "--end-date",
        today,
        "--session-type",
        session_type,
        "--max-tasks",
        str(int(getattr(args, "history_collector_max_tasks", 500))),
        "--client-id",
        str(int(getattr(args, "history_collector_client_id", 168))),
        "--allow-outside-window",
    ]
    limit_symbols = int(getattr(args, "history_collector_limit_symbols", 0) or 0)
    if limit_symbols > 0:
        cmd.extend(["--limit-symbols", str(limit_symbols)])

    runtime_state["history_collector_running"] = True
    print(f"{now_utc()} HISTORY_COLLECTOR_LAUNCH key={key} cmd={' '.join(cmd)}", flush=True)
    try:
        completed = subprocess.run(cmd, check=False)
        print(f"{now_utc()} HISTORY_COLLECTOR_EXIT key={key} returncode={completed.returncode}", flush=True)
        if completed.returncode == 0:
            runtime_state["history_collector_last_run_key"] = key
        return int(completed.returncode)
    except Exception as exc:
        print(f"{now_utc()} HISTORY_COLLECTOR_LAUNCH_ERROR key={key} error={exc!r}", flush=True)
        return 1
    finally:
        runtime_state["history_collector_running"] = False
'''

PARSER_ARGS = '''    parser.add_argument("--history-collector-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-collector-start-date", default="2026-01-01")
    parser.add_argument("--history-collector-session-type", choices=["RTH", "EXT"], default="RTH")
    parser.add_argument("--history-collector-start-utc", default="20:15")
    parser.add_argument("--history-collector-end-utc", default="23:00")
    parser.add_argument("--history-collector-market-close-utc", default="20:00")
    parser.add_argument("--history-collector-max-tasks", type=int, default=500)
    parser.add_argument("--history-collector-limit-symbols", type=int, default=0)
    parser.add_argument("--history-collector-client-id", type=int, default=168)
'''


def strip_old_broken_calls(txt: str) -> str:
    txt = re.sub(r"^\s*run_history_collector_once_if_due\(args, runtime_state\)\s*\n", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*run_v68_history_collector_once_if_due\(args, v68_history_runtime_state\)\s*\n", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*run_v68_history_collector_once_if_due\(args, runtime_state\)\s*\n", "", txt, flags=re.MULTILINE)
    return txt


def add_imports(txt: str) -> str:
    if "import subprocess" not in txt:
        txt = txt.replace("import time\n", "import time\nimport subprocess\nimport sys\n", 1)
    elif "import sys" not in txt:
        txt = txt.replace("import subprocess\n", "import subprocess\nimport sys\n", 1)
    return txt


def add_helper(txt: str) -> str:
    if "def run_v68_history_collector_once_if_due(" in txt:
        return txt
    m = re.search(r"^def main\(\)\s*->\s*int:\s*$|^def main\(\)\s*:\s*$", txt, re.MULTILINE)
    if not m:
        raise SystemExit("main() marker not found")
    return txt[:m.start()] + HELPER + "\n\n" + txt[m.start():]


def add_parser_args(txt: str) -> str:
    if "--history-collector-enabled" in txt:
        return txt
    m = re.search(r"^(?P<indent>\s*)parser\s*=\s*argparse\.ArgumentParser\([^\n]*\)\s*$", txt, re.MULTILINE)
    if not m:
        raise SystemExit("ArgumentParser line not found")
    insert_at = txt.find("\n", m.end()) + 1
    return txt[:insert_at] + PARSER_ARGS + txt[insert_at:]


def line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def find_main_loop(lines: list[str]) -> tuple[int, int, int]:
    starts = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*while\s+time\.time\(\)\s*-\s*start\s*<\s*args\.duration_seconds\s*:\s*$", line):
            starts.append(i)
    if len(starts) != 1:
        raise SystemExit(f"expected exactly one main duration loop, found {len(starts)}")
    start_i = starts[0]
    base_indent = line_indent(lines[start_i])
    end_i = len(lines)
    for j in range(start_i + 1, len(lines)):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line_indent(lines[j]) <= base_indent:
            end_i = j
            break
    return start_i, end_i, base_indent


def add_runtime_state_and_loop_call(txt: str) -> str:
    if "v68_history_runtime_state = {}" in txt and "run_v68_history_collector_once_if_due(args, v68_history_runtime_state)" in txt:
        return txt

    lines = txt.splitlines(True)
    start_i, end_i, base_indent = find_main_loop(lines)
    base = " " * base_indent
    inner = " " * (base_indent + 4)

    if "v68_history_runtime_state = {}" not in txt:
        lines.insert(start_i, f"{base}v68_history_runtime_state = {{}}\n")
        start_i += 1
        end_i += 1

    # Refresh loop bounds after insertion.
    start_i, end_i, base_indent = find_main_loop(lines)
    sleep_indices = []
    for i in range(start_i + 1, end_i):
        if re.match(r"^\s*(ib|time)\.sleep\([^\n]+\)\s*$", lines[i]):
            sleep_indices.append(i)
    if not sleep_indices:
        raise SystemExit("no sleep call found inside main loop")

    insert_i = sleep_indices[-1]
    if "run_v68_history_collector_once_if_due(args, v68_history_runtime_state)" not in "".join(lines[start_i:end_i]):
        lines.insert(insert_i, f"{inner}run_v68_history_collector_once_if_due(args, v68_history_runtime_state)\n")

    return "".join(lines)


def main() -> None:
    original = P.read_text()
    txt = strip_old_broken_calls(original)
    txt = add_imports(txt)
    txt = add_helper(txt)
    txt = add_parser_args(txt)
    txt = add_runtime_state_and_loop_call(txt)

    if txt == original:
        print("no changes needed")
        return

    backup = P.with_suffix(".before_history_collector_safe_v3_patch.py")
    backup.write_text(original)
    P.write_text(txt)
    print(f"patched v67 safe v3; backup={backup}")


if __name__ == "__main__":
    main()
