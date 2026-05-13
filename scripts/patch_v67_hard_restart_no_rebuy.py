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


def insert_before_line_containing(txt: str, needle: str, insertion_builder, label: str) -> str:
    lines = txt.splitlines(True)
    out = []
    patched = False
    for line in lines:
        if not patched and needle in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(insertion_builder(indent))
            patched = True
        out.append(line)
    if not patched:
        raise SystemExit(f'line containing marker not found: {label}: {needle!r}')
    return ''.join(out)


def main() -> None:
    txt = P.read_text()

    if '--restart-entry-guard-minutes' not in txt:
        markers = [
            '    parser.add_argument("--max-one-trade-per-symbol-per-day"',
            '    parser.add_argument("--position-usd"',
            '    parser.add_argument("--top-n"',
        ]
        for marker in markers:
            if marker in txt:
                txt = insert_after_line(
                    txt,
                    marker,
                    '    parser.add_argument("--restart-entry-guard-minutes", type=float, default=20.0)\n'
                    '    parser.add_argument("--restart-entry-guard-after-market-open", action=argparse.BooleanOptionalAction, default=True)\n'
                    '    parser.add_argument("--restart-entry-guard-block-replayed-signals", action=argparse.BooleanOptionalAction, default=True)\n',
                    'restart guard parser args',
                )
                break
        else:
            raise SystemExit('no parser insertion marker found')

    if 'def restart_entry_guard_active(' not in txt:
        helper = r'''
def utc_now_minutes() -> int:
    t = datetime.now(timezone.utc).time()
    return t.hour * 60 + t.minute


def utc_hhmm_to_minutes(value: str) -> int:
    hh, mm = [int(x) for x in str(value).strip().split(":", 1)]
    return hh * 60 + mm


def restart_entry_guard_active(process_started_monotonic: float, args: argparse.Namespace) -> tuple[bool, str]:
    if not getattr(args, "restart_entry_guard_after_market_open", True):
        return False, "disabled"
    guard_minutes = float(getattr(args, "restart_entry_guard_minutes", 20.0))
    if guard_minutes <= 0:
        return False, "disabled_zero_minutes"
    try:
        market_open_min = utc_hhmm_to_minutes(getattr(args, "market_open_utc", "13:30"))
        now_min = utc_now_minutes()
    except Exception as exc:
        return True, f"guard_fail_closed_bad_time error={exc!r}"
    if now_min < market_open_min:
        return False, "before_market_open"
    elapsed_min = (time.monotonic() - process_started_monotonic) / 60.0
    if elapsed_min < guard_minutes:
        return True, f"intraday_restart_guard elapsed_min={elapsed_min:.1f} guard_min={guard_minutes:.1f}"
    return False, f"guard_elapsed elapsed_min={elapsed_min:.1f} guard_min={guard_minutes:.1f}"

'''
        markers = ['\ndef is_eod_flatten_time(', '\ndef send_exit_order(', '\ndef manage_exits(']
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, helper + marker, 1)
                break
        else:
            raise SystemExit('no helper insertion marker found')

    if 'process_started_monotonic = time.monotonic()' not in txt:
        markers = ['    recorder = LiveDataRecorder(args.recorder_dir)', '    ib = IB()']
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, '    process_started_monotonic = time.monotonic()\n' + marker, 1)
                break
        else:
            raise SystemExit('no process start marker found')

    if 'restart_guard_block_entries, restart_guard_reason = restart_entry_guard_active' not in txt:
        markers = ['        entries_blocked = 0', '        ready_new = 0']
        for marker in markers:
            if marker in txt:
                txt = txt.replace(
                    marker,
                    '        restart_guard_block_entries, restart_guard_reason = restart_entry_guard_active(process_started_monotonic, args)\n'
                    '        if restart_guard_block_entries:\n'
                    '            print(f"{now_utc()} restart_entry_guard_active reason={restart_guard_reason}", flush=True)\n'
                    + marker,
                    1,
                )
                break
        else:
            raise SystemExit('no per-loop guard insertion marker found')

    if 'RESTART_GUARD_HARD_SKIP_BUY' not in txt:
        def build(indent: str) -> str:
            return (
                f'{indent}if restart_guard_block_entries:\n'
                f'{indent}    record_lifecycle(\n'
                f'{indent}        recorder,\n'
                f'{indent}        "RESTART_GUARD_HARD_SKIP_BUY",\n'
                f'{indent}        symbol,\n'
                f'{indent}        action="SKIP_BUY",\n'
                f'{indent}        price=features.get("entry_price") if isinstance(features, dict) else None,\n'
                f'{indent}        reason=restart_guard_reason,\n'
                f'{indent}        raw_json=features if isinstance(features, dict) else None,\n'
                f'{indent}    )\n'
                f'{indent}    entries_blocked += 1\n'
                f'{indent}    continue\n'
            )

        buy_markers = [
            'MarketOrder("BUY"',
            "MarketOrder('BUY'",
            'MarketOrder(action="BUY"',
        ]
        for marker in buy_markers:
            if marker in txt:
                txt = insert_before_line_containing(txt, marker, build, 'hard buy block')
                break
        else:
            raise SystemExit('no BUY MarketOrder marker found; cannot safely patch hard restart guard')

    if 'restart_guard={1 if restart_guard_block_entries else 0}' not in txt:
        txt = txt.replace(
            'entries_blocked={entries_blocked}',
            'entries_blocked={entries_blocked} restart_guard={1 if restart_guard_block_entries else 0}',
        )

    P.write_text(txt)
    print('patched hard restart no-rebuy guard: BUY orders are blocked during intraday restart cooldown')


if __name__ == '__main__':
    main()
