from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


EOD_HELPERS = r'''
def eod_remaining_portfolio_positions(ib: IB) -> list[tuple[str, float]]:
    remaining: list[tuple[str, float]] = []
    try:
        for item in ib.portfolio():
            symbol = str(getattr(item.contract, "symbol", "")).upper()
            qty = safe_float(getattr(item, "position", None))
            if symbol and qty is not None and abs(qty) > 0:
                remaining.append((symbol, float(qty)))
    except Exception as exc:
        print(f"{now_utc()} EOD VERIFY ERROR error={exc!r}", flush=True)
    return remaining


def force_eod_flatten_all_portfolio_positions_v2(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    reason: str = "v67_forced_eod_flatten_all",
) -> int:
    sent = 0
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper()
        qty = safe_float(getattr(item, "position", None))
        if not symbol or qty is None or abs(qty) <= 0:
            continue

        managed = managed_positions.get(symbol)
        if managed is not None and managed.exit_sent:
            continue

        action = "SELL" if qty > 0 else "BUY"
        quantity = int(abs(qty))
        if quantity <= 0:
            print(f"{now_utc()} EOD SKIP FRACTIONAL symbol={symbol} qty={qty}", flush=True)
            continue

        original_contract = getattr(item, "contract", None)
        contract = contract_by_symbol.get(symbol)
        if contract is None:
            currency = getattr(original_contract, "currency", None) or "USD"
            primary = getattr(original_contract, "primaryExchange", None) or ""
            contract = Stock(symbol, "SMART", currency, primaryExchange=primary)
            try:
                ib.qualifyContracts(contract)
            except Exception as exc:
                print(f"{now_utc()} EOD QUALIFY ERROR symbol={symbol} error={exc!r}", flush=True)
                contract = original_contract
        if contract is None:
            print(f"{now_utc()} EOD SKIP NO_CONTRACT symbol={symbol} qty={qty}", flush=True)
            continue

        price = safe_float((latest_snapshots.get(symbol) or {}).get("price")) or safe_float(getattr(item, "marketPrice", None))
        avg_cost = safe_float(getattr(item, "averageCost", None)) or (managed.entry_price if managed else price)
        peak = managed.peak_price if managed else (price or avg_cost or 0.0)

        try:
            order = MarketOrder(action, quantity)
            order.tif = "DAY"
            order.outsideRth = False
            trade = ib.placeOrder(contract, order)
            pnl_pct = ((price / avg_cost - 1.0) * 100.0) if price and avg_cost and avg_cost > 0 else None
            record_lifecycle(
                recorder,
                "SELL_ORDER_SENT" if action == "SELL" else "BUY_TO_COVER_SENT",
                symbol,
                action=action,
                quantity=quantity,
                price=price,
                order_id=trade.order.orderId,
                reason=reason,
                entry_price=avg_cost,
                peak_price=peak,
                pnl_pct=pnl_pct,
                decision_last=price,
            )
            if managed is not None:
                managed.exit_sent = True
                managed.active = False
            sent += 1
            pnl_txt = f" pnl_pct={pnl_pct:.2f}" if pnl_pct is not None else ""
            print(
                f"EOD FORCE ORDER SENT symbol={symbol} action={action} qty={quantity} reason={reason} "
                f"entry={avg_cost if avg_cost else 0:.2f} price={price if price else 0:.2f}{pnl_txt} "
                f"orderId={trade.order.orderId}",
                flush=True,
            )
        except Exception as exc:
            print(f"{now_utc()} EOD FORCE ORDER ERROR symbol={symbol} action={action} qty={quantity} error={exc!r}", flush=True)
    return sent


def eod_flatten_verify_and_retry(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    print(f"{now_utc()} EOD FLATTEN START utc={getattr(args, 'eod_flatten_utc', '')}", flush=True)
    total_sent = 0
    retries = max(1, int(getattr(args, "eod_flatten_retries", 5)))
    sleep_s = float(getattr(args, "eod_flatten_retry_sleep_seconds", 15.0))

    for attempt in range(1, retries + 1):
        sent = force_eod_flatten_all_portfolio_positions_v2(
            ib=ib,
            recorder=recorder,
            contract_by_symbol=contract_by_symbol,
            managed_positions=managed_positions,
            latest_snapshots=latest_snapshots,
            reason=f"v67_forced_eod_flatten_attempt_{attempt}",
        )
        total_sent += sent
        ib.sleep(2)
        remaining = eod_remaining_portfolio_positions(ib)
        print(
            f"{now_utc()} EOD FLATTEN VERIFY attempt={attempt} sent={sent} remaining={len(remaining)} positions={remaining[:20]}",
            flush=True,
        )
        if not remaining:
            print(f"{now_utc()} EOD FLATTEN COMPLETE remaining=0 total_sent={total_sent}", flush=True)
            return total_sent
        if attempt < retries:
            ib.sleep(sleep_s)

    remaining = eod_remaining_portfolio_positions(ib)
    print(f"{now_utc()} EOD FLATTEN INCOMPLETE remaining={len(remaining)} positions={remaining[:50]}", flush=True)
    return total_sent

'''


def main() -> None:
    txt = P.read_text()

    txt = txt.replace('default="19:55"', 'default="19:45"')

    if '--eod-flatten-retries' not in txt:
        marker = '    parser.add_argument("--eod-flatten-utc"'
        idx = txt.find(marker)
        if idx == -1:
            raise SystemExit('eod flatten utc parser arg not found')
        insert_at = txt.find('\n', idx) + 1
        txt = txt[:insert_at] + (
            '    parser.add_argument("--eod-flatten-retries", type=int, default=5)\n'
            '    parser.add_argument("--eod-flatten-retry-sleep-seconds", type=float, default=15.0)\n'
            '    parser.add_argument("--eod-flatten-verify", action=argparse.BooleanOptionalAction, default=True)\n'
        ) + txt[insert_at:]

    if 'def eod_flatten_verify_and_retry(' not in txt:
        txt = replace_once(txt, '\ndef manage_exits(', EOD_HELPERS + '\ndef manage_exits(', 'insert eod v2 helpers')

    if 'eod_flatten_v2_sent_this_loop' not in txt:
        old = '        exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args, contract_by_symbol)'
        if old not in txt:
            old = '        exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args)'
        new = '''        eod_flatten_v2_sent_this_loop = 0
        if args.enable_eod_flatten and is_eod_flatten_time(args.eod_flatten_utc):
            eod_flatten_v2_sent_this_loop = eod_flatten_verify_and_retry(
                ib=ib,
                recorder=recorder,
                contract_by_symbol=contract_by_symbol,
                managed_positions=managed_positions,
                latest_snapshots=latest_snapshots,
                args=args,
            )
        exits_sent = manage_exits(ib, recorder, managed_positions, latest_snapshots, args, contract_by_symbol)
        exits_sent += eod_flatten_v2_sent_this_loop'''
        txt = replace_once(txt, old, new, 'patch manage_exits call')

    P.write_text(txt)
    print('patched EOD flatten v2: earlier 19:45, force all positions, verify/retry, explicit logs')


if __name__ == '__main__':
    main()
