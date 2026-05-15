from __future__ import annotations

from pathlib import Path

P = Path("src/live_trading/v67_live_top100_expansion_paper_trader.py")


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if new in txt:
        return txt
    if old not in txt:
        raise SystemExit(f"marker not found: {label}")
    return txt.replace(old, new, 1)


def main() -> None:
    txt = P.read_text()
    original = txt
    backup = P.with_suffix(".before_hard_block_entries_patch.py")
    backup.write_text(txt)

    txt = txt.replace(
        'runtime_state = {"entries_blocked": False, "control_api_commands": [],',
        'runtime_state = {"entries_blocked": True, "control_api_commands": [],',
        1,
    )

    old_block = '''            entries_blocked = (
                not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
'''
    new_block = '''            time_entries_blocked = (
                not is_after_utc(args.new_entries_start_utc)
                or is_after_utc(args.no_new_entries_after_utc)
                or is_after_utc(args.eod_flatten_utc)
            )
            manual_entries_blocked = bool(runtime_state.get("entries_blocked", False))
            entries_blocked = time_entries_blocked or manual_entries_blocked
'''
    txt = replace_once(txt, old_block, new_block, "entries_blocked decision block")

    # There is a second heartbeat-local recomputation in the current file. Replace it too if still present.
    txt = txt.replace(old_block, new_block)

    if 'BUY_BLOCKED' not in txt:
        old_buy_if = '                if features["ready"] and not state.signal_sent and not has_active_position and not entries_blocked:\n'
        new_buy_if = '''                if features["ready"] and not state.signal_sent and not has_active_position and entries_blocked:
                    record_lifecycle(
                        recorder,
                        "BUY_BLOCKED",
                        symbol,
                        action="BUY",
                        price=features.get("entry_price"),
                        reason="entries_blocked_manual_or_time_window",
                        raw_json={**features, "manual_entries_blocked": bool(runtime_state.get("entries_blocked", False))},
                    )
                if features["ready"] and not state.signal_sent and not has_active_position and not entries_blocked:
'''
        txt = replace_once(txt, old_buy_if, new_buy_if, "buy blocked logging")

    if txt == original:
        print("no changes needed")
        return

    P.write_text(txt)
    print(f"patched v67 hard block entries after restart; backup={backup}")


if __name__ == "__main__":
    main()
