from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def main() -> None:
    txt = P.read_text()

    # 1) Robust parser for IBKR bar timestamps with timezone offsets, e.g. 2026-05-12 09:30:00-04:00.
    old_parser = '''def _parse_bar_time_utc(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
'''
    new_parser = '''def _parse_bar_time_utc(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None

    raw = raw.replace("US/Eastern", "").replace("America/New_York", "").strip()

    for fmt in (
        "%Y%m%d  %H:%M:%S",
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None
'''
    if old_parser in txt:
        txt = txt.replace(old_parser, new_parser, 1)
    elif 'def _parse_bar_time_utc(value: str):' not in txt:
        raise SystemExit('bar parser function missing')

    # 2) Add targeted current-session backfill helper after backfill_recent_1m.
    if 'def backfill_current_session_1m(' not in txt:
        marker = '\ndef is_eod_flatten_time('
        helper = r'''
def current_session_candle_count(recorder: LiveDataRecorder, args: argparse.Namespace) -> int:
    path = recorder.path("candles_1m.csv")
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        today = datetime.now(timezone.utc).date()
        market_open = datetime(today.year, today.month, today.day, hh, mm, tzinfo=timezone.utc)
    except Exception:
        return 0
    count = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_bar_time_utc(row.get("bar_time", ""))
                if ts is not None and ts >= market_open:
                    count += 1
    except Exception:
        return 0
    return count


def backfill_current_session_1m(
    ib: IB,
    recorder: LiveDataRecorder,
    contracts: list[tuple[str, Any]],
    args: argparse.Namespace,
) -> int:
    if not getattr(args, "backfill_current_session_on_rebuild_miss", True):
        return 0

    try:
        hh, mm = [int(x) for x in str(args.market_open_utc).split(":", 1)]
        now = datetime.now(timezone.utc)
        market_open = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception as exc:
        print(f"{now_utc()} current_session_backfill_skipped reason=bad_market_open error={exc!r}", flush=True)
        return 0

    if now < market_open:
        print(f"{now_utc()} current_session_backfill_skipped reason=before_market_open", flush=True)
        return 0

    duration_seconds = max(60, int((now - market_open).total_seconds()) + 300)
    duration = f"{duration_seconds} S"
    keys = existing_candle_keys(recorder)
    total = 0
    subset = contracts[: max(0, int(getattr(args, "backfill_top_n", len(contracts))))]
    print(f"{now_utc()} current_session_backfill_start symbols={len(subset)} duration={duration}", flush=True)

    for symbol, contract in subset:
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            print(f"{now_utc()} current_session_backfill_error symbol={symbol} error={exc!r}", flush=True)
            continue

        rows = []
        for bar in bars:
            bar_time = str(getattr(bar, "date", ""))
            ts = _parse_bar_time_utc(bar_time)
            if ts is None or ts < market_open:
                continue
            key = (symbol, bar_time)
            if key in keys:
                continue
            keys.add(key)
            rows.append({
                "symbol": symbol,
                "bar_time": bar_time,
                "open": safe_float(getattr(bar, "open", None)),
                "high": safe_float(getattr(bar, "high", None)),
                "low": safe_float(getattr(bar, "low", None)),
                "close": safe_float(getattr(bar, "close", None)),
                "volume": safe_float(getattr(bar, "volume", None)),
                "wap": safe_float(getattr(bar, "average", None)),
                "trade_count": safe_float(getattr(bar, "barCount", None)),
                "source": "ibkr_current_session_backfill_1m",
                "recorded_at": now_utc(),
            })
        if rows:
            total += recorder.record_candles_1m(rows)
        ib.sleep(float(getattr(args, "backfill_pause_seconds", 0.15)))

    print(f"{now_utc()} current_session_backfill_done symbols={len(subset)} rows={total}", flush=True)
    return total

'''
        txt = replace_once(txt, marker, helper + marker, 'insert current session backfill helper')

    # 3) Add CLI flag for current-session rebuild miss backfill.
    if '--backfill-current-session-on-rebuild-miss' not in txt:
        old_arg = '    parser.add_argument("--backfill-pause-seconds", type=float, default=0.15)'
        new_arg = old_arg + '\n    parser.add_argument("--backfill-current-session-on-rebuild-miss", action=argparse.BooleanOptionalAction, default=True)'
        txt = replace_once(txt, old_arg, new_arg, 'backfill current-session arg')

    # 4) If rebuild gets 0 symbols but current-session candles are missing, fetch them and rebuild again.
    if 'current_session_backfilled_rows = 0' not in txt:
        old_block = '''        backfilled_rows = backfill_recent_1m(ib, recorder, contracts, args)
        state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        traded_symbols_today = load_traded_symbols_today(recorder)'''
        new_block = '''        backfilled_rows = backfill_recent_1m(ib, recorder, contracts, args)
        state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        current_session_backfilled_rows = 0
        if state_rebuild_count == 0 and current_session_candle_count(recorder, args) == 0:
            current_session_backfilled_rows = backfill_current_session_1m(ib, recorder, contracts, args)
            if current_session_backfilled_rows:
                state_rebuild_count = rebuild_symbol_states_from_1m_candles(recorder, states, args)
        traded_symbols_today = load_traded_symbols_today(recorder)'''
        txt = replace_once(txt, old_block, new_block, 'rebuild fallback block')

    # 5) Include metadata if record_run_metadata block has state_rebuild_count.
    if '"current_session_backfilled_1m_rows": current_session_backfilled_rows' not in txt:
        txt = txt.replace(
            '            "state_rebuild_count": state_rebuild_count,',
            '            "state_rebuild_count": state_rebuild_count,\n            "current_session_backfilled_1m_rows": current_session_backfilled_rows,',
            1,
        )

    P.write_text(txt)
    print('patched v67 current-session backfill on rebuild miss')


if __name__ == '__main__':
    main()
