from __future__ import annotations

from pathlib import Path

V67 = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')
REPORT = Path('src/live_trading/analytics/v67_daily_report.py')


def replace_once(txt: str, old: str, new: str, label: str) -> str:
    if old not in txt:
        raise SystemExit(f'marker not found: {label}')
    return txt.replace(old, new, 1)


def patch_v67() -> None:
    txt = V67.read_text()

    # Add configurable strict/original setup thresholds. Defaults match the original strict parser defaults.
    if '--strict-setup-name' not in txt:
        old = '    parser.add_argument("--min-or-range-pct", type=float, default=5.0)'
        new = old + '''
    parser.add_argument("--strict-setup-name", default="v67_original_600usd_setup")
    parser.add_argument("--strict-min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--strict-min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--strict-min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--strict-min-price", type=float, default=5.0)
    parser.add_argument("--strict-max-spread-bps", type=float, default=50.0)'''
        txt = replace_once(txt, old, new, 'strict setup parser args')

    # Compute strict_ready in the same place as the current relaxed ready flag.
    if 'strict_setup_ready = (' not in txt:
        old = '''    reasons: list[str] = []'''
        new = '''    strict_setup_ready = (
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

    reasons: list[str] = []'''
        txt = replace_once(txt, old, new, 'strict setup ready computation')

    if '"strict_setup_ready": strict_setup_ready' not in txt:
        old = '''        "ready": ready,
        "score": round(score, 4),'''
        new = '''        "ready": ready,
        "strict_setup_ready": strict_setup_ready,
        "strict_setup_name": getattr(args, "strict_setup_name", "v67_original_600usd_setup"),
        "strict_min_first_5m_high_pct": getattr(args, "strict_min_first_5m_high_pct", 4.0),
        "strict_min_first_15m_high_pct": getattr(args, "strict_min_first_15m_high_pct", 6.5),
        "strict_min_or_range_pct": getattr(args, "strict_min_or_range_pct", 5.0),
        "score": round(score, 4),'''
        txt = replace_once(txt, old, new, 'strict setup fields in features')

    # Try to attach the same feature payload to BUY_ORDER_SENT rows as raw_json.
    # If the exact marker does not exist in a future version, SIGNAL_READY will still carry the strict tag.
    if 'raw_json=features,' not in txt:
        candidate_markers = [
            '                        spread_pct=spread_pct,\n                    )',
            '                        spread_pct=spread_pct,\n                        fill_price=None,\n                    )',
            '                        spread_pct=spread_pct,\n                        raw_json=None,\n                    )',
        ]
        for old in candidate_markers:
            if old in txt:
                new = old.replace('\n                    )', '\n                        raw_json=features,\n                    )')
                txt = txt.replace(old, new, 1)
                break

    V67.write_text(txt)
    print('patched v67 with strict/original setup tag')


def patch_report() -> None:
    if not REPORT.exists():
        print('daily report not found, skipping report patch')
        return
    txt = REPORT.read_text()

    # Add json helper for strict tag extraction.
    if 'def raw_json_dict(' not in txt:
        old = '''def parse_dt(value: str | None):'''
        new = '''def raw_json_dict(row: dict) -> dict:
    try:
        raw = row.get("raw_json") or "{}"
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)
    except Exception:
        return {}


def parse_dt(value: str | None):'''
        txt = replace_once(txt, old, new, 'raw_json_dict helper')

    # Store strict tag in BUY rows, primarily from BUY raw_json if present.
    if '"strict_setup_ready": bool(raw_json_dict(b).get("strict_setup_ready", False))' not in txt:
        old = '''                        "buy_bucket": buy_time_bucket(buy_time),
                        "minutes_from_open": minutes_from_market_open(buy_time),'''
        new = '''                        "buy_bucket": buy_time_bucket(buy_time),
                        "minutes_from_open": minutes_from_market_open(buy_time),
                        "strict_setup_ready": bool(raw_json_dict(b).get("strict_setup_ready", False)),
                        "strict_setup_name": raw_json_dict(b).get("strict_setup_name", ""),'''
        txt = replace_once(txt, old, new, 'closed strict tag fields')

        old = '''                "buy_bucket": buy_time_bucket(buy_time),
                "minutes_from_open": minutes_from_market_open(buy_time),'''
        new = '''                "buy_bucket": buy_time_bucket(buy_time),
                "minutes_from_open": minutes_from_market_open(buy_time),
                "strict_setup_ready": bool(raw_json_dict(b).get("strict_setup_ready", False)),
                "strict_setup_name": raw_json_dict(b).get("strict_setup_name", ""),'''
        txt = replace_once(txt, old, new, 'open strict tag fields')

    # Add report section. This will show 0 strict buys until BUY_ORDER_SENT raw_json is available.
    # SIGNAL_READY already has the tag after the v67 patch, so future BUY rows should also have it if marker matched.
    if '=== Strict/original setup subset ===' not in txt:
        marker = '''    print("=== PnL by buy time bucket ===")'''
        section = '''    strict_closed = [x for x in closed if x.get("strict_setup_ready")]
    strict_open = [x for x in open_positions if x.get("strict_setup_ready")]
    strict_gross = sum(x["gross"] for x in strict_closed)
    strict_net = sum(x["net"] for x in strict_closed)
    strict_upnl = sum(x["unrealized"] for x in strict_open)
    strict_wins = [x for x in strict_closed if x["gross"] > 0]
    strict_win_rate = (len(strict_wins) / len(strict_closed) * 100) if strict_closed else 0.0
    print("=== Strict/original setup subset ===")
    print(f"Strict closed trades: {len(strict_closed)}")
    print(f"Strict open trades:   {len(strict_open)}")
    print(f"Strict win rate:      {strict_win_rate:.1f}%")
    print(f"Strict gross closed:  ${strict_gross:.2f}")
    print(f"Strict net closed:    ${strict_net:.2f}")
    print(f"Strict open UPNL:     ${strict_upnl:.2f}")
    print(f"Strict total est:     ${strict_net + strict_upnl:.2f}")
    print()

'''
        txt = replace_once(txt, marker, section + marker, 'strict report section')

    # Add STRICT column to tables if possible.
    if "{'STRICT':>6}" not in txt:
        txt = txt.replace(
            "{'SYM':<7} {'UTC':>5} {'MIN':>5} {'BUCKET':<13} {'QTY':>5}",
            "{'SYM':<7} {'UTC':>5} {'MIN':>5} {'BUCKET':<13} {'STRICT':>6} {'QTY':>5}",
        )
        txt = txt.replace(
            "f\"{x.get('buy_bucket',''):<13} \"\n            f\"{x['qty']:>5.0f} ",
            "f\"{x.get('buy_bucket',''):<13} \"\n            f\"{str(bool(x.get('strict_setup_ready'))):>6} \"\n            f\"{x['qty']:>5.0f} ",
        )

    REPORT.write_text(txt)
    print('patched daily report with strict/original setup subset')


def main() -> None:
    patch_v67()
    patch_report()


if __name__ == '__main__':
    main()
