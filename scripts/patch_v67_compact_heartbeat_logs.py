from __future__ import annotations

from pathlib import Path

P = Path('src/live_trading/v67_live_top100_expansion_paper_trader.py')


def main() -> None:
    txt = P.read_text()
    replacements = {
        'f"{now_utc()} heartbeat scanned={scanned}': 'f"heartbeat scanned={scanned}',
        'f"{now_utc()} heartbeat scanned=': 'f"heartbeat scanned=',
    }
    changed = False
    for old, new in replacements.items():
        if old in txt:
            txt = txt.replace(old, new)
            changed = True

    if not changed:
        print('No heartbeat timestamp pattern found; file left unchanged')
        return

    P.write_text(txt)
    print('patched compact heartbeat logs: removed internal now_utc() prefix from heartbeat lines')


if __name__ == '__main__':
    main()
