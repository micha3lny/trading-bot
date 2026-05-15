from __future__ import annotations

from pathlib import Path

P = Path("src/live_trading/data/v68_universe_1m_parquet_collector.py")


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if new in txt:
        return txt
    if old not in txt:
        raise SystemExit(f"marker not found: {label}")
    return txt.replace(old, new, 1)


def main() -> None:
    txt = P.read_text()
    original = txt
    backup = P.with_suffix(".before_resilience_patch.py")
    backup.write_text(txt)

    if "import traceback" not in txt:
        txt = txt.replace("import time\n", "import time\nimport traceback\n", 1)

    txt = replace_once(
        txt,
        '''def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
''',
        '''def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp, index=False)
    except ImportError as exc:
        raise RuntimeError("Missing parquet engine. Install pyarrow in venv: pip install pyarrow") from exc
    tmp.replace(path)
''',
        "write_parquet",
    )

    txt = txt.replace(
        '    parser.add_argument("--postmarket-end-utc", default="23:00")',
        '    parser.add_argument("--postmarket-end-utc", default="15:00")',
    )
    txt = txt.replace(
        '    parser.add_argument("--premarket-start-utc", default="11:00")',
        '    parser.add_argument("--premarket-start-utc", default="20:15")',
    )
    txt = txt.replace(
        '    parser.add_argument("--premarket-end-utc", default="13:00")',
        '    parser.add_argument("--premarket-end-utc", default="15:00")',
    )

    txt = replace_once(
        txt,
        '''    print(
        f"{now_iso()} HISTORY_COLLECTOR_START symbols={len(symbols)} tasks={len(tasks)} "
        f"pending={len(pending)} start={start} end={end} session={args.session_type}",
        flush=True,
    )
''',
        '''    print(
        f"{now_iso()} HISTORY_COLLECTOR_START symbols={len(symbols)} tasks={len(tasks)} "
        f"pending={len(pending)} start={start} end={end} session={args.session_type} "
        f"output_dir={output_dir}",
        flush=True,
    )
''',
        "collector start log",
    )

    txt = replace_once(
        txt,
        '''            state, rows, error = collect_one(ib, task, output_dir, float(args.request_sleep_seconds))
            if state == "complete":
''',
        '''            try:
                state, rows, error = collect_one(ib, task, output_dir, float(args.request_sleep_seconds))
            except Exception as exc:
                state, rows, error = "failed", 0, f"unexpected: {exc!r}"
                tb = traceback.format_exc(limit=3).replace("\\n", " | ")
                print(
                    f"{now_iso()} HISTORY_SYMBOL_EXCEPTION {task.symbol} {task.session_date} "
                    f"error={exc!r} traceback={tb}",
                    flush=True,
                )

            if state == "complete":
''',
        "collect_one try/except",
    )

    txt = replace_once(
        txt,
        '''                print(f"{now_iso()} HISTORY_PROGRESS_SAVED idx={idx} pending={len(pending)}", flush=True)
''',
        '''                print(
                    f"{now_iso()} HISTORY_PROGRESS_SAVED idx={idx} pending={len(pending)} "
                    f"complete={completed} partial={partial} no_data={no_data} failed={failed}",
                    flush=True,
                )
''',
        "progress log",
    )

    txt = replace_once(
        txt,
        '''        f"no_data={no_data} failed={failed}",
''',
        '''        f"no_data={no_data} failed={failed} output_dir={output_dir}",
''',
        "done log",
    )

    if txt == original:
        print("no changes needed")
        return
    P.write_text(txt)
    print(f"patched v68 collector resilience; backup={backup}")


if __name__ == "__main__":
    main()
