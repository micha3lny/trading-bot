from __future__ import annotations

import re
from pathlib import Path

P = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")

IMPORT_LINE = "from src.live_trading.data.v68_history_hook import maybe_run_history_collector\n"

ARGS_BLOCK = '''    parser.add_argument("--history-collector-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-collector-start-date", default="2026-01-01")
    parser.add_argument("--history-collector-session-type", choices=["RTH", "EXT"], default="RTH")
    parser.add_argument("--history-collector-start-utc", default="20:15")
    parser.add_argument("--history-collector-end-utc", default="23:00")
    parser.add_argument("--history-collector-market-close-utc", default="20:00")
    parser.add_argument("--history-collector-max-tasks", type=int, default=500)
    parser.add_argument("--history-collector-limit-symbols", type=int, default=0)
    parser.add_argument("--history-collector-client-id", type=int, default=168)
'''


def main() -> None:
    txt = P.read_text()
    original = txt
    backup = P.with_suffix(".before_history_hook_small_patch.py")
    backup.write_text(original)

    # 1. Import hook after existing src.live_trading imports.
    if IMPORT_LINE not in txt:
        marker = "from src.live_trading.v62_live_data_recorder import LiveDataRecorder\n"
        if marker not in txt:
            raise SystemExit("import marker not found")
        txt = txt.replace(marker, marker + IMPORT_LINE, 1)

    # 2. Parser args after ArgumentParser line.
    if "--history-collector-enabled" not in txt:
        m = re.search(r"^(?P<indent>\s*)parser\s*=\s*argparse\.ArgumentParser\([^\n]*\)\s*$", txt, re.MULTILINE)
        if not m:
            raise SystemExit("ArgumentParser line not found")
        insert_at = txt.find("\n", m.end()) + 1
        txt = txt[:insert_at] + ARGS_BLOCK + txt[insert_at:]

    # 3. State before main loop. This exact block is used by v67.
    if "history_collector_state = {}" not in txt:
        old = "        start = time.time()\n        while time.time() - start < args.duration_seconds:\n"
        new = "        start = time.time()\n        history_collector_state = {}\n        while time.time() - start < args.duration_seconds:\n"
        if old not in txt:
            raise SystemExit("main loop start marker not found")
        txt = txt.replace(old, new, 1)

    # 4. Hook call before the sleep at the end of the main loop.
    if "maybe_run_history_collector(args, history_collector_state)" not in txt:
        matches = list(re.finditer(r"^\s*ib\.sleep\(args\.sleep_seconds\)\s*$", txt, re.MULTILINE))
        if not matches:
            raise SystemExit("ib.sleep(args.sleep_seconds) marker not found")
        m = matches[-1]
        indent = m.group(0)[: len(m.group(0)) - len(m.group(0).lstrip())]
        call = f"{indent}maybe_run_history_collector(args, history_collector_state)\n"
        txt = txt[:m.start()] + call + txt[m.start():]

    P.write_text(txt)
    print(f"patched v67 history hook; backup={backup}")


if __name__ == "__main__":
    main()
