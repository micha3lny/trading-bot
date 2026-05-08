from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = "data/1m"
DEFAULT_FAILED_CACHE = "data/1m/fetch_1m_failed_symbols.csv"
DEFAULT_OUTPUT_DIR = "data/universe"

SPECIAL_SUFFIXES = (
    "W", "WS", "WT", "WTS", "WW", "WZ", "U", "R", "RT", "RIGHT", "UNIT",
    "P", "PR", "L", "Z",
)


def is_likely_special_issue(symbol: str) -> tuple[bool, str]:
    s = symbol.upper().strip()
    if not s:
        return True, "empty_symbol"

    # Common SPAC / warrant / unit / right / preferred / note patterns.
    if len(s) >= 5 and s.endswith("W"):
        return True, "likely_warrant_suffix_w"
    if len(s) >= 5 and s.endswith("WW"):
        return True, "likely_warrant_suffix_ww"
    if len(s) >= 5 and s.endswith("WS"):
        return True, "likely_warrant_suffix_ws"
    if len(s) >= 5 and s.endswith("WT"):
        return True, "likely_warrant_suffix_wt"
    if len(s) >= 5 and s.endswith("WZ"):
        return True, "likely_special_suffix_wz"
    if len(s) >= 5 and s.endswith("U"):
        return True, "likely_unit_suffix_u"
    if len(s) >= 5 and s.endswith("R"):
        return True, "likely_right_suffix_r"
    if len(s) >= 5 and s.endswith("P"):
        return True, "likely_preferred_suffix_p"

    # TMUSI/TMUSL/TMUSZ, notes/preferreds/special classes, not useful for intraday common-stock universe.
    if len(s) >= 5 and s[-1] in {"L", "Z", "O", "N"}:
        return True, "likely_note_or_special_class"

    return False, "common_or_retry_candidate"


def existing_rows(data_dir: Path, symbol: str) -> int:
    path = data_dir / f"{symbol}.csv"
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit IBKR failed cache and build retry lists")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--failed-cache", default=DEFAULT_FAILED_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    failed_path = Path(args.failed_cache)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== v63 failed cache audit ===")
    print(f"Data dir: {data_dir}")
    print(f"Failed cache: {failed_path}")

    if not failed_path.exists():
        print("No failed cache found")
        return 0

    failed = pd.read_csv(failed_path)
    if failed.empty or "symbol" not in failed.columns:
        print("Failed cache is empty or missing symbol column")
        return 0

    failed["symbol"] = failed["symbol"].astype(str).str.upper().str.strip()
    failed = failed.drop_duplicates(subset=["symbol"], keep="last")

    rows = []
    for _, row in failed.iterrows():
        symbol = row["symbol"]
        special, reason = is_likely_special_issue(symbol)
        local_rows = existing_rows(data_dir, symbol)
        rows.append({
            "symbol": symbol,
            "error": row.get("error", ""),
            "existing_rows": local_rows,
            "likely_special_issue": special,
            "classification_reason": reason,
            "retry_candidate": (not special) and local_rows == 0,
            "has_partial_data": local_rows > 0,
        })

    audit = pd.DataFrame(rows).sort_values(["retry_candidate", "likely_special_issue", "symbol"], ascending=[False, True, True])
    retry = audit[audit["retry_candidate"]].copy()
    special = audit[audit["likely_special_issue"]].copy()
    partial = audit[audit["has_partial_data"]].copy()

    audit_path = out_dir / "v63_failed_cache_audit.csv"
    retry_path = out_dir / "v63_retry_symbols.txt"
    special_path = out_dir / "v63_likely_special_failed.csv"
    partial_path = out_dir / "v63_partial_failed.csv"

    audit.to_csv(audit_path, index=False)
    special.to_csv(special_path, index=False)
    partial.to_csv(partial_path, index=False)
    retry_path.write_text("\n".join(retry["symbol"].tolist()) + ("\n" if not retry.empty else ""))

    print("\n=== Failed cache summary ===")
    print(f"failed_unique: {len(audit)}")
    print(f"retry_candidates_common_no_file: {len(retry)}")
    print(f"likely_special_or_garbage: {len(special)}")
    print(f"failed_but_has_partial_data: {len(partial)}")

    if not retry.empty:
        print("\n=== Retry candidates sample ===")
        print(retry[["symbol", "classification_reason", "error"]].head(50).to_string(index=False))

    print(f"\nSaved audit: {audit_path}")
    print(f"Saved retry symbols: {retry_path}")
    print(f"Saved likely special failed: {special_path}")
    print(f"Saved partial failed: {partial_path}")

    print("\nNext command for normal failed retry:")
    print("python -m src.data.fetch_1m_data --symbols-file data/universe/v63_retry_symbols.txt --days 90 --port 4002 --skip-existing-any --no-skip-failed-cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
