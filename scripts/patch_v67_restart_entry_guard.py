from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def main() -> None:
    txt = P.read_text()

    # 1) Add CLI options.
    if '--restart-entry-guard-minutes' not in txt:
        marker = '    parser.add_argument("--max-one-trade-per-symbol-per-day"'
        idx = txt.find(marker)
        if idx == -1:
            marker = '    parser.add_argument("--position-usd"'
            idx = txt.find(marker)
        if idx == -1:
            raise SystemExit('parser arg insertion marker not found')
        insert_at = txt.find('\n', idx) + 1
        txt = txt[:insert_at] + (
            '    parser.add_argument("--restart-entry-guard-minutes", type=float, default=20.0)\n'
            '    parser.add_argument("--restart-entry-guard-after-market-open", action=argparse.BooleanOptionalAction, default=True)\n'
            '    parser.add_argument("--restart-entry-guard-require-fresh-fill-check", action=argparse.BooleanOptionalAction, default=True)\n'
        ) + txt[insert_at:]

    # 2) Add helpers.
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
        return False, "guard_minutes_zero"
    try:
        market_open_min = utc_hhmm_to_minutes(getattr(args, "market_open_utc", "13:30"))
        now_min = utc_now_minutes()
    except Exception:
        return False, "bad_market_open"
    if now_min < market_open_min:
        return False, "before_market_open"
    elapsed_min = (time.monotonic() - process_started_monotonic) / 60.0
    if elapsed_min < guard_minutes:
        return True, f"intraday_restart_guard elapsed_min={elapsed_min:.1f} guard_min={guard_minutes:.1f}"
    return False, f"guard_elapsed elapsed_min={elapsed_min:.1f} guard_min={guard_minutes:.1f}"

'''
        txt = replace_once(txt, '\ndef is_eod_flatten_time(', helper + '\ndef is_eod_flatten_time(', 'insert restart guard helpers')

    # 3) Capture process start near main runtime.
    if 'process_started_monotonic = time.monotonic()' not in txt:
        markers = [
            '    recorder = LiveDataRecorder(args.recorder_dir)',
            '    ib = IB()',
        ]
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, '    process_started_monotonic = time.monotonic()\n' + marker, 1)
                break
        else:
            raise SystemExit('main process start marker not found')

    # 4) Gate entry logic. We patch the loop before new entries are evaluated by forcing ready_new to skip.
    # Look for the common entry condition marker.
    if 'restart_guard_block_entries' not in txt:
        marker = '        entries_blocked = 0'
        if marker not in txt:
            marker = '        ready_new = 0'
        guard_block = '''        restart_guard_block_entries, restart_guard_reason = restart_entry_guard_active(process_started_monotonic, args)
        if restart_guard_block_entries:
            print(f"{now_utc()} restart_entry_guard_active reason={restart_guard_reason}", flush=True)
'''
        txt = replace_once(txt, marker, guard_block + marker, 'insert restart guard loop check')

    # 5) Patch BUY condition(s): if features ready but guard active, log skip and do not submit.
    if 'RESTART_GUARD_SKIP_BUY' not in txt:
        # This codebase uses `if features["ready"]` or `if features.get("ready")` before buy.
        candidates = [
            'if features.get("ready"):',
            'if features["ready"]:',
            'if features.get("ready", False):',
        ]
        patched = False
        for cand in candidates:
            if cand in txt:
                replacement = cand + '''
                    if restart_guard_block_entries:
                        record_lifecycle(
                            recorder,
                            "RESTART_GUARD_SKIP_BUY",
                            symbol,
                            action="SKIP_BUY",
                            price=features.get("entry_price"),
                            reason=restart_guard_reason,
                            raw_json=features,
                        )
                        entries_blocked += 1
                        continue'''
                txt = txt.replace(cand, replacement, 1)
                patched = True
                break
        if not patched:
            # Fallback: block immediately before placeOrder BUY if condition marker changed.
            cand = 'trade = ib.placeOrder(contract, order)'
            if cand not in txt:
                raise SystemExit('buy condition/placeOrder marker not found')
            guard = '''if restart_guard_block_entries:
                            record_lifecycle(
                                recorder,
                                "RESTART_GUARD_SKIP_BUY",
                                symbol,
                                action="SKIP_BUY",
                                price=features.get("entry_price") if 'features' in locals() else None,
                                reason=restart_guard_reason,
                            )
                            entries_blocked += 1
                            continue
                        '''
            txt = txt.replace(cand, guard + cand, 1)

    # 6) Make heartbeat show guard state if heartbeat f-string contains eod_active.
    if 'restart_guard={1 if restart_guard_block_entries else 0}' not in txt:
        txt = txt.replace(
            'eod_active={1 if eod_active else 0}',
            'eod_active={1 if eod_active else 0} restart_guard={1 if restart_guard_block_entries else 0}',
        )

    P.write_text(txt)
    print('patched v67 intraday restart entry guard')


if __name__ == '__main__':
    main()
