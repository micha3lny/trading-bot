from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def main() -> None:
    txt = P.read_text()

    # 1) Add strict/original setup CLI thresholds if missing.
    if '--strict-setup-name' not in txt:
        markers = [
            '    parser.add_argument("--min-or-range-pct", type=float, default=5.0)',
            '    parser.add_argument("--min-or-range-pct", type=float, default=0.5)',
        ]
        for marker in markers:
            if marker in txt:
                strict_args = marker + '''
    parser.add_argument("--strict-setup-name", default="v67_original_600usd_setup")
    parser.add_argument("--strict-min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--strict-min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--strict-min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--strict-min-price", type=float, default=5.0)
    parser.add_argument("--strict-max-spread-bps", type=float, default=50.0)'''
                txt = txt.replace(marker, strict_args, 1)
                break
        else:
            raise SystemExit('min-or-range parser arg marker not found')

    # 2) Default re-entry enabled: max-one-trade/day off by default.
    if '--max-one-trade-per-symbol-per-day' in txt:
        txt = txt.replace(
            'parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=True)',
            'parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=False)',
        )
        txt = txt.replace(
            'parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default = True)',
            'parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=False)',
        )
    else:
        # Add the flag near max spread if it does not exist yet.
        marker = '    parser.add_argument("--max-spread-bps", type=float, default=50.0)'
        if marker in txt:
            txt = txt.replace(
                marker,
                marker + '\n    parser.add_argument("--max-one-trade-per-symbol-per-day", action=argparse.BooleanOptionalAction, default=False)',
                1,
            )

    # 3) Add strict setup calculation to feature payload.
    if 'strict_setup_ready = (' not in txt:
        marker = '    reasons: list[str] = []'
        strict_calc = '''    strict_setup_ready = (
        first_5m_high_pct is not None
        and first_15m_high_pct is not None
        and or_range_pct is not None
        and first_5m_high_pct >= getattr(args, "strict_min_first_5m_high_pct", 4.0)
        and first_15m_high_pct >= getattr(args, "strict_min_first_15m_high_pct", 6.5)
        and or_range_pct >= getattr(args, "strict_min_or_range_pct", 5.0)
        and price is not None
        and price >= getattr(args, "strict_min_price", 5.0)
        and (spread_bps is None or spread_bps <= getattr(args, "strict_max_spread_bps", 50.0))
    )

'''
        txt = replace_once(txt, marker, strict_calc + marker, 'insert strict setup calculation')

    if '"strict_setup_ready": strict_setup_ready' not in txt:
        old = '''        "ready": ready,
        "score": round(score, 4),'''
        new = '''        "ready": ready,
        "strict_setup_ready": strict_setup_ready,
        "strict_setup_name": getattr(args, "strict_setup_name", "v67_original_600usd_setup"),
        "strict_min_first_5m_high_pct": getattr(args, "strict_min_first_5m_high_pct", 4.0),
        "strict_min_first_15m_high_pct": getattr(args, "strict_min_first_15m_high_pct", 6.5),
        "strict_min_or_range_pct": getattr(args, "strict_min_or_range_pct", 5.0),
        "strict_min_price": getattr(args, "strict_min_price", 5.0),
        "strict_max_spread_bps": getattr(args, "strict_max_spread_bps", 50.0),
        "score": round(score, 4),'''
        txt = replace_once(txt, old, new, 'strict fields in feature return')

    # 4) Ensure BUY_ORDER_SENT rows carry raw_json=features when possible.
    # This is intentionally conservative; SIGNAL_READY has the strict tag regardless.
    if 'BUY_ORDER_SENT' in txt and 'raw_json=features' not in txt:
        markers = [
            '''                        spread_pct=(snap.get("spread_bps") or 0) / 100.0 if snap.get("spread_bps") is not None else None,
                    )''',
            '''                        spread_pct=spread_pct,
                    )''',
        ]
        for old in markers:
            if old in txt:
                txt = txt.replace(old, old.replace('\n                    )', '\n                        raw_json=features,\n                    )'), 1)
                break

    # 5) Harden EOD liquidation: force account portfolio longs too, not only managed_positions.
    if 'def force_eod_flatten_portfolio_longs(' not in txt:
        insert_before = '\ndef manage_exits('
        helper = r'''
def force_eod_flatten_portfolio_longs(
    ib: IB,
    recorder: LiveDataRecorder,
    contract_by_symbol: dict[str, Any],
    managed_positions: dict[str, ManagedPosition],
    latest_snapshots: dict[str, dict[str, Any]],
    reason: str = "v67_forced_eod_flatten_portfolio_long",
) -> int:
    sent = 0
    for item in ib.portfolio():
        symbol = str(getattr(item.contract, "symbol", "")).upper()
        qty = safe_float(getattr(item, "position", None))
        if not symbol or qty is None or qty <= 0:
            continue
        contract = contract_by_symbol.get(symbol) or getattr(item, "contract", None)
        if contract is None:
            continue

        managed = managed_positions.get(symbol)
        if managed is not None and managed.exit_sent:
            continue

        quantity = int(qty)
        price = safe_float((latest_snapshots.get(symbol) or {}).get("price")) or safe_float(getattr(item, "marketPrice", None))
        avg_cost = safe_float(getattr(item, "averageCost", None)) or (managed.entry_price if managed else price)
        peak = managed.peak_price if managed else (price or avg_cost or 0.0)

        try:
            order = MarketOrder("SELL", quantity)
            order.tif = "DAY"
            order.outsideRth = False
            trade = ib.placeOrder(contract, order)
            pnl_pct = ((price / avg_cost - 1.0) * 100.0) if price and avg_cost and avg_cost > 0 else None
            record_lifecycle(
                recorder,
                "SELL_ORDER_SENT",
                symbol,
                action="SELL",
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
                f"EOD FORCE SELL SENT symbol={symbol} qty={quantity} reason={reason} "
                f"entry={avg_cost if avg_cost else 0:.2f} price={price if price else 0:.2f}{pnl_txt} "
                f"orderId={trade.order.orderId}",
                flush=True,
            )
        except Exception as exc:
            print(f"{now_utc()} eod_force_sell_error symbol={symbol} qty={quantity} error={exc!r}", flush=True)
    return sent

'''
        txt = replace_once(txt, insert_before, helper + insert_before, 'insert force eod flatten helper')

    # 6) Patch manage_exits signature to receive contract_by_symbol, then call force flattener when EOD.
    old_sig = 'def manage_exits(ib: IB, recorder: LiveDataRecorder, managed_positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:'
    new_sig = 'def manage_exits(ib: IB, recorder: LiveDataRecorder, managed_positions: dict[str, ManagedPosition], latest_snapshots: dict[str, dict[str, Any]], args: argparse.Namespace, contract_by_symbol: dict[str, Any] | None = None) -> int:'
    if old_sig in txt:
        txt = txt.replace(old_sig, new_sig, 1)

    if 'force_eod_flatten_portfolio_longs(' in txt and 'eod_forced_extra_exits' not in txt:
        old = '''    for symbol, pos in list(managed_positions.items()):'''
        new = '''    if eod and contract_by_symbol is not None:
        extra = force_eod_flatten_portfolio_longs(
            ib=ib,
            recorder=recorder,
            contract_by_symbol=contract_by_symbol,
            managed_positions=managed_positions,
            latest_snapshots=latest_snapshots,
        )
        if extra:
            print(f"{now_utc()} eod_forced_extra_exits={extra}", flush=True)
            exits += extra

    for symbol, pos in list(managed_positions.items()):'''
        txt = replace_once(txt, old, new, 'call force eod flattener')

    # 7) Patch all manage_exits calls to pass contract_by_symbol if not already.
    txt = txt.replace(
        'manage_exits(ib, recorder, managed_positions, latest_snapshots, args)',
        'manage_exits(ib, recorder, managed_positions, latest_snapshots, args, contract_by_symbol)',
    )

    P.write_text(txt)
    print('patched v67 strict setup tags, disabled one-trade/day default, and hardened EOD flatten')


if __name__ == '__main__':
    main()
