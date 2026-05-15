from __future__ import annotations

import re
from pathlib import Path

P = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")

IMPORT_OLD = "from src.live_trading.control.control_api import start_control_api\n"
IMPORT_NEW = "from src.live_trading.control.control_api import process_control_api_commands, start_control_api\n"


def main() -> None:
    txt = P.read_text()
    original = txt

    backup = P.with_suffix(".before_control_api_queue_hook_patch.py")
    backup.write_text(original)

    # 1. Import process_control_api_commands next to start_control_api.
    if IMPORT_NEW not in txt:
        if IMPORT_OLD in txt:
            txt = txt.replace(IMPORT_OLD, IMPORT_NEW, 1)
        else:
            marker = "from src.live_trading.v62_live_data_recorder import LiveDataRecorder\n"
            if marker not in txt:
                raise SystemExit("import marker not found")
            txt = txt.replace(marker, marker + IMPORT_NEW, 1)

    # 2. Ensure runtime_state exists and has control_api_commands queue.
    if '"entries_blocked": False' not in txt:
        anchor = "    managed_positions: dict[str, ManagedPosition] = {}\n"
        if anchor not in txt:
            raise SystemExit("managed_positions anchor not found")
        txt = txt.replace(
            anchor,
            anchor + '    runtime_state = {"entries_blocked": False, "control_api_commands": []}\n',
            1,
        )
    elif '"control_api_commands"' not in txt:
        txt = txt.replace(
            'runtime_state = {"entries_blocked": False}',
            'runtime_state = {"entries_blocked": False, "control_api_commands": []}',
            1,
        )
        txt = txt.replace(
            "        \"entries_blocked\": False,\n    }",
            "        \"entries_blocked\": False,\n        \"control_api_commands\": [],\n    }",
            1,
        )

    # 3. Ensure start_control_api is wired before run metadata.
    if "start_control_api(" not in txt:
        anchor = "        recorder.record_run_metadata({\n"
        if anchor not in txt:
            raise SystemExit("record_run_metadata anchor not found")
        block = '''        control_api_server = start_control_api(
            ib=ib,
            recorder=recorder,
            managed_positions=managed_positions,
            runtime_state=runtime_state,
            record_lifecycle_fn=record_lifecycle,
            persist_managed_positions_fn=persist_managed_positions,
            host="127.0.0.1",
            port=8767,
        )

'''
        txt = txt.replace(anchor, block + anchor, 1)

    # 4. Insert queue processor in the main loop before sleep.
    call = '''        process_control_api_commands(
            ib=ib,
            recorder=recorder,
            managed_positions=managed_positions,
            runtime_state=runtime_state,
            record_lifecycle_fn=record_lifecycle,
            persist_managed_positions_fn=persist_managed_positions,
        )
'''
    if "process_control_api_commands(" not in txt.replace(IMPORT_NEW, ""):
        sleep_matches = list(re.finditer(r"^\s*ib\.sleep\(args\.sleep_seconds\)\s*$", txt, re.MULTILINE))
        if not sleep_matches:
            raise SystemExit("ib.sleep(args.sleep_seconds) marker not found")
        # last sleep should be the heartbeat/main-loop sleep
        m = sleep_matches[-1]
        indent = m.group(0)[: len(m.group(0)) - len(m.group(0).lstrip())]
        indented_call = "\n".join((indent + line if line else line) for line in call.splitlines()) + "\n"
        txt = txt[:m.start()] + indented_call + txt[m.start():]

    if txt == original:
        print("no changes needed")
        return

    P.write_text(txt)
    print(f"patched v67 control API queue hook; backup={backup}")


if __name__ == "__main__":
    main()
