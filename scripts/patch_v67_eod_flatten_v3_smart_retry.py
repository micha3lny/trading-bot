from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def main() -> None:
    txt = P.read_text()

    # 1) Add CLI knobs for safer EOD behavior.
    if '--eod-flatten-retry-seconds' not in txt:
        markers = [
            '    parser.add_argument("--eod-flatten-utc"',
            '    parser.add_argument("--enable-eod-flatten"',
            '    parser.add_argument("--duration-seconds"',
        ]
        for marker in markers:
            if marker in txt:
                line_end = txt.find('\n', txt.find(marker)) + 1
                txt = txt[:line_end] + (
                    '    parser.add_argument("--eod-flatten-retry-seconds", type=float, default=30.0)\n'
                    '    parser.add_argument("--eod-flatten-verify-retries", type=int, default=8)\n'
                    '    parser.add_argument("--eod-flatten-use-smart-contract", action=argparse.BooleanOptionalAction, default=True)\n'
                ) + txt[line_end:]
                break
        else:
            raise SystemExit('parser insertion marker not found')

    # 2) Add helper functions before send_exit_order.
    if 'def build_smart_stock_contract_for_close(' not in txt:
        helper = r'''
def build_smart_stock_contract_for_close(ib: IB, symbol: str, source_contract: Any | None = None) -> Any:
    """Return a qualified SMART stock contract for closing orders.

    IBKR portfolio contracts often contain conId/primaryExchange but no exchange.
    Sending orders with those raw contracts can fail with:
        Error 321: Missing order exchange.
    Therefore all close orders should use an explicit SMART stock contract.
    """
    symbol = str(symbol).upper().strip()
    currency = getattr(source_contract, "currency", None) or "USD"
    primary = getattr(source_contract, "primaryExchange", None) or ""
    contract = Stock(symbol, "SMART", currency, primaryExchange=primary)
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
    except Exception as exc:
        print(f"{now_utc()} smart_close_contract_qualify_error symbol={symbol} error={exc!r}", flush=True)
    return contract


def place_smart_close_order(
    ib: IB,
    symbol: str,
    quantity: int,
    action: str,
    source_contract: Any | None,
    outside_rth: bool = False,
) -> Any:
    contract = build_smart_stock_contract_for_close(ib, symbol, source_contract)
    order = MarketOrder(action, int(quantity))
    order.tif = "DAY"
    order.outsideRth = bool(outside_rth)
    return ib.placeOrder(contract, order)

'''
        txt = replace_once(txt, '\ndef send_exit_order(', helper + '\ndef send_exit_order(', 'insert smart close helpers')

    # 3) Patch send_exit_order to use SMART-qualified contract.
    old = '''    order = MarketOrder("SELL", pos.quantity)
    order.tif = "DAY"
    order.outsideRth = False
    trade = ib.placeOrder(pos.contract, order)
'''
    new = '''    order = MarketOrder("SELL", pos.quantity)
    order.tif = "DAY"
    order.outsideRth = False
    close_contract = build_smart_stock_contract_for_close(ib, pos.symbol, pos.contract)
    trade = ib.placeOrder(close_contract, order)
'''
    if old in txt:
        txt = txt.replace(old, new, 1)
    elif 'build_smart_stock_contract_for_close(ib, pos.symbol, pos.contract)' not in txt:
        raise SystemExit('send_exit_order SMART replacement marker not found')

    # 4) Patch control_flatten_positions if present: raw portfolio contract -> SMART contract.
    old_control = '''        order = MarketOrder(action, quantity)
        order.tif = "DAY"
        order.outsideRth = False
        trade = ib.placeOrder(contract, order)
'''
    new_control = '''        order = MarketOrder(action, quantity)
        order.tif = "DAY"
        order.outsideRth = False
        close_contract = build_smart_stock_contract_for_close(ib, symbol, contract)
        trade = ib.placeOrder(close_contract, order)
'''
    if old_control in txt:
        txt = txt.replace(old_control, new_control, 1)

    # 5) Add EOD retry state in main before loop.
    if 'last_eod_flatten_retry_ts = 0.0' not in txt:
        markers = ['        start = time.time()\n', '    start = time.time()\n']
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, '        last_eod_flatten_retry_ts = 0.0\n' + marker, 1)
                break
        else:
            raise SystemExit('start loop marker not found')

    # 6) Add retry call after manage_exits result is computed.
    if 'EOD_FLATTEN_RETRY' not in txt:
        marker_candidates = [
            '        exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args)\n',
            '            exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args)\n',
        ]
        retry_block = '''        if args.enable_eod_flatten and is_eod_flatten_time(args.eod_flatten_utc):
            now_mono = time.monotonic()
            if now_mono - last_eod_flatten_retry_ts >= float(getattr(args, "eod_flatten_retry_seconds", 30.0)):
                last_eod_flatten_retry_ts = now_mono
                retry_count = 0
                for symbol, pos in list(managed_positions.items()):
                    if not pos.active or pos.exit_sent or pos.quantity <= 0:
                        continue
                    snap = latest_snapshots.get(symbol) or {}
                    retry_price = safe_float(snap.get("price"))
                    record_lifecycle(
                        recorder,
                        "EOD_FLATTEN_RETRY",
                        symbol,
                        action="SELL",
                        quantity=pos.quantity,
                        price=retry_price,
                        reason="eod_retry_still_active",
                        entry_price=pos.entry_price,
                        peak_price=pos.peak_price,
                    )
                    send_exit_order(ib, recorder, pos, "v46_wide_trail_close_exit_eod_retry", retry_price)
                    retry_count += 1
                if retry_count:
                    print(f"{now_utc()} EOD_FLATTEN_RETRY sent={retry_count}", flush=True)
'''
        for marker in marker_candidates:
            if marker in txt:
                txt = txt.replace(marker, marker + retry_block, 1)
                break
        else:
            raise SystemExit('manage_exits call marker not found for EOD retry')

    P.write_text(txt)
    print('patched v67 EOD flatten v3: SMART close contracts + retry loop')


if __name__ == '__main__':
    main()
