from __future__ import annotations

import re
from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')

HELPER = r'''

def _v68_history_utc_minutes(value: str) -> int:
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    return hh * 60 + mm


def _v68_history_in_utc_window(start_hhmm: str, end_hhmm: str) -> bool:
    t = datetime.now(timezone.utc).time()
    now_min = t.hour * 60 + t.minute
    start = _v68_history_utc_minutes(start_hhmm)
    end = _v68_history_utc_minutes(end_hhmm)
    if end <= start:
        return now_min >= start or now_min < end
    return start <= now_min < end


def _v68_history_trading_window_active(args: argparse.Namespace) -> bool:
    start = getattr(args, "market_open_utc", "13:30")
    end = getattr(args, "history_collector_market_close_utc", "20:00")
    return _v68_history_in_utc_window(start, end)


def run_v68_history_collector_once_if_due(args: argparse.Namespace, runtime_state: dict[str, Any]) -> int:
    if not getattr(args, "history_collector_enabled", False):
        return 0
    if _v68_history_trading_window_active(args):
        return 0
    if not _v68_history_in_utc_window(
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


def add_imports(txt: str) -> str:
    if 'import subprocess' not in txt:
        txt = txt.replace('import time\n', 'import time\nimport subprocess\nimport sys\n', 1)
    elif 'import sys' not in txt:
        txt = txt.replace('import subprocess\n', 'import subprocess\nimport sys\n', 1)
    return txt


def add_parser_args(txt: str) -> str:
    if '--history-collector-enabled' in txt:
        return txt
    marker = 'parser = argparse.ArgumentParser'
    idx = txt.find(marker)
    if idx == -1:
        raise SystemExit('parser marker not found')
    line_end = txt.find('\n', idx)
    insert_at = line_end + 1
    args_block = (
        '    parser.add_argument("--history-collector-enabled", action=argparse.BooleanOptionalAction, default=False)\n'
        '    parser.add_argument("--history-collector-start-date", default="2026-01-01")\n'
        '    parser.add_argument("--history-collector-session-type", choices=["RTH", "EXT"], default="RTH")\n'
        '    parser.add_argument("--history-collector-start-utc", default="20:15")\n'
        '    parser.add_argument("--history-collector-end-utc", default="23:00")\n'
        '    parser.add_argument("--history-collector-market-close-utc", default="20:00")\n'
        '    parser.add_argument("--history-collector-max-tasks", type=int, default=500)\n'
        '    parser.add_argument("--history-collector-limit-symbols", type=int, default=0)\n'
        '    parser.add_argument("--history-collector-client-id", type=int, default=168)\n'
    )
    return txt[:insert_at] + args_block + txt[insert_at:]


def add_helper(txt: str) -> str:
    if 'def run_v68_history_collector_once_if_due(' in txt:
        return txt
    # Insert before parser or main, whichever exists first.
    candidates = [m.start() for m in re.finditer(r'\ndef (build_parser|parse_args|main)\(', txt)]
    if not candidates:
        raise SystemExit('helper insertion marker not found')
    pos = min(candidates)
    return txt[:pos] + HELPER + txt[pos:]


def add_runtime_state(txt: str) -> str:
    if 'v68_history_runtime_state = {}' in txt:
        return txt
    m = re.search(r'^(?P<indent>\s*)while\s+time\.time\(\)\s*-\s*start\s*<\s*args\.duration_seconds\s*:', txt, re.MULTILINE)
    if not m:
        raise SystemExit('main loop while marker not found')
    indent = m.group('indent')
    insert = f'{indent}v68_history_runtime_state = {{}}\n'
    return txt[:m.start()] + insert + txt[m.start():]


def add_loop_call(txt: str) -> str:
    if 'run_v68_history_collector_once_if_due(args, v68_history_runtime_state)' in txt:
        return txt
    heartbeat_idx = txt.find('heartbeat scanned=')
    if heartbeat_idx == -1:
        raise SystemExit('heartbeat marker not found; refusing unsafe patch')
    sleep_match = re.search(r'^(?P<indent>\s*)ib\.sleep\([^\n]+\)\s*$', txt[heartbeat_idx:], re.MULTILINE)
    if not sleep_match:
        raise SystemExit('ib.sleep after heartbeat not found; refusing unsafe patch')
    abs_start = heartbeat_idx + sleep_match.start()
    indent = sleep_match.group('indent')
    call = f'{indent}run_v68_history_collector_once_if_due(args, v68_history_runtime_state)\n'
    return txt[:abs_start] + call + txt[abs_start:]


def main() -> None:
    txt = P.read_text()
    original = txt
    txt = add_imports(txt)
    txt = add_helper(txt)
    txt = add_parser_args(txt)
    txt = add_runtime_state(txt)
    txt = add_loop_call(txt)
    if txt == original:
        print('no changes needed')
        return
    backup = P.with_suffix('.before_history_collector_safe_patch.py')
    backup.write_text(original)
    P.write_text(txt)
    print(f'patched v67 safely; backup={backup}')


if __name__ == '__main__':
    main()
