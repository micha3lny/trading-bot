from pathlib import Path

p = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")
txt = p.read_text()
backup = p.with_suffix(".before_control_api_hook_patch.py")
backup.write_text(txt)

imp = "from src.live_trading.control.control_api import start_control_api\n"
marker = "from src.live_trading.v62_live_data_recorder import LiveDataRecorder\n"
if imp not in txt:
    txt = txt.replace(marker, marker + imp, 1)

if '"entries_blocked": False' not in txt:
    anchor = "    managed_positions: dict[str, ManagedPosition] = {}\n"
    txt = txt.replace(anchor, anchor + '    runtime_state = {"entries_blocked": False}\n', 1)

if "start_control_api(" not in txt:
    anchor = "        recorder.record_run_metadata({\n"
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

p.write_text(txt)
print(f"patched control api hook; backup={backup}")
