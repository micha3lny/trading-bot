from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def insert_after_line(txt: str, marker: str, insertion: str, label: str) -> str:
    idx = txt.find(marker)
    if idx == -1:
        raise SystemExit(f'marker not found: {label}: {marker!r}')
    end = txt.find('\n', idx)
    if end == -1:
        end = len(txt)
    return txt[: end + 1] + insertion + txt[end + 1:]


def main() -> None:
    txt = P.read_text()

    # Imports.
    if 'import subprocess' not in txt:
        txt = txt.replace('import time\n', 'import time\nimport subprocess\nimport sys\n', 1)
    elif 'import sys' not in txt:
        txt = txt.replace('import subprocess\n', 'import subprocess\nimport sys\n', 1)

    # CLI args.
    if '--history-collector-enabled' not in txt:
        markers = [
            '    parser.add_argument("--enable-eod-flatten"',
            '    parser.add_argument("--eod-flatten-utc"',
            '    parser.add_argument("--duration-seconds"',
        ]
        insertion = (
            '    parser.add_argument("--history-collector-enabled", action=argparse.BooleanOptionalAction, default=True)\n'
            '    parser.add_argument("--history-collector-start-date", default="2026-01-01")\n'
            '    parser.add_argument("--history-collector-session-type", choices=["RTH", "EXT"], default="RTH")\n'
            '    parser.add_argument("--history-collector-start-utc", default="20:15")\n'
            '    parser.add_argument("--history-collector-end-utc", default="23:00")\n'
            '    parser.add_argument("--history-collector-max-tasks", type=int, default=500)\n'
            '    parser.add_argument("--history-collector-limit-symbols", type=int, default=0)\n'
            '    parser.add_argument("--history-collector-client-id", type=int, default=168)\n'
        )
        for marker in markers:
            if marker in txt:
                txt = insert_after_line(txt, marker, insertion, 'history collector parser args')
                break
        else:
            raise SystemExit('no parser marker found')

    # Helper functions.
    if 'def run_history_collector_once_if_due(' not in txt:
        helper = r'''
def run_history_collector_once_if_due(args: argparse.Namespace, state: dict[str, Any]) -> int:
    if not getattr(args, "history_collector_enabled", True):
        return 0
    if not in_utc_window(getattr(args, "history_collector_start_utc", "20:15"), getattr(args, "history_collector_end_utc", "23:00")):
        return 0
    if is_trading_window():
        return 0

    today = datetime.now(timezone.utc).date().isoformat()
    key = f"{today}_{getattr(args, 'history_collector_session_type', 'RTH')}"
    if state.get("history_collector_last_run_key") == key:
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
        str(getattr(args, "history_collector_session_type", "RTH")),
        "--max-tasks",
        str(int(getattr(args, "history_collector_max_tasks", 500))),
        "--client-id",
        str(int(getattr(args, "history_collector_client_id", 168))),
        "--allow-outside-window",
    ]
    limit_symbols = int(getattr(args, "history_collector_limit_symbols", 0) or 0)
    if limit_symbols > 0:
        cmd.extend(["--limit-symbols", str(limit_symbols)])

    print(f"{now_utc()} HISTORY_COLLECTOR_LAUNCH key={key} cmd={' '.join(cmd)}", flush=True)
    try:
        completed = subprocess.run(cmd, check=False)
        print(f"{now_utc()} HISTORY_COLLECTOR_EXIT key={key} returncode={completed.returncode}", flush=True)
        if completed.returncode == 0:
            state["history_collector_last_run_key"] = key
        return int(completed.returncode)
    except Exception as exc:
        print(f"{now_utc()} HISTORY_COLLECTOR_LAUNCH_ERROR key={key} error={exc!r}", flush=True)
        return 1

'''
        markers = ['\ndef is_eod_flatten_time(', '\ndef send_exit_order(', '\ndef manage_exits(']
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, helper + marker, 1)
                break
        else:
            raise SystemExit('no helper insertion marker found')

    # Runtime state before main loop.
    if 'runtime_state = {' not in txt:
        markers = ['        last_eod_flatten_retry_ts = 0.0\n', '        start = time.time()\n', '    start = time.time()\n']
        insertion = '        runtime_state = {}\n'
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, insertion + marker, 1)
                break
        else:
            raise SystemExit('no runtime_state insertion marker found')

    # Call collector in main loop after exits/eod processing, once per day in window.
    if 'run_history_collector_once_if_due(args, runtime_state)' not in txt:
        markers = [
            '        record_strategy_equity(recorder, managed_positions, latest_snapshots)\n',
            '            record_strategy_equity(recorder, managed_positions, latest_snapshots)\n',
            '        persist_managed_positions(recorder, managed_positions)\n',
            '            persist_managed_positions(recorder, managed_positions)\n',
        ]
        for marker in markers:
            if marker in txt:
                indent = marker[: len(marker) - len(marker.lstrip())]
                txt = txt.replace(marker, marker + f'{indent}run_history_collector_once_if_due(args, runtime_state)\n', 1)
                break
        else:
            raise SystemExit('collector call insertion marker not found')

    P.write_text(txt)
    print('patched v67: integrated post-EOD v68 parquet history collector')


if __name__ == '__main__':
    main()
