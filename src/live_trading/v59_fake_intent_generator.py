from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path


DEFAULT_INTENTS_OUTPUT = "data/live/order_intents.csv"
DEFAULT_SNAPSHOTS_OUTPUT = "data/live/live_signal_snapshots.csv"


FAKE_SYMBOLS = [
    {"symbol": "QQQ", "start": 520.00, "spread_bps": 0.8},
    {"symbol": "NVDA", "start": 180.00, "spread_bps": 4.0},
    {"symbol": "TSLA", "start": 280.00, "spread_bps": 2.0},
    {"symbol": "RKLB", "start": 25.00, "spread_bps": 12.0},
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_price(symbol_cfg: dict[str, object], i: int) -> float:
    base = float(symbol_cfg["start"])
    symbol = str(symbol_cfg["symbol"])

    # Deterministic mini scenarios:
    # QQQ: slow grind up.
    # NVDA: breakout then fade.
    # TSLA: strong breakout.
    # RKLB: choppy / higher spread.
    if symbol == "QQQ":
        return base * (1 + 0.0008 * i)
    if symbol == "NVDA":
        return base * (1 + 0.0020 * min(i, 8) - 0.0015 * max(i - 8, 0))
    if symbol == "TSLA":
        return base * (1 + 0.0030 * i)
    if symbol == "RKLB":
        return base * (1 + 0.0040 * min(i, 5) - 0.0030 * max(i - 5, 0))
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fake v58 order intents and snapshots for v59 pipeline testing.")
    parser.add_argument("--intents-output", default=DEFAULT_INTENTS_OUTPUT)
    parser.add_argument("--snapshots-output", default=DEFAULT_SNAPSHOTS_OUTPUT)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval-seconds", type=int, default=5)
    parser.add_argument("--base-position-usd", type=float, default=100.0)
    args = parser.parse_args()

    print("=== v59 fake intent generator ===")
    print(f"Writing intents: {args.intents_output}")
    print(f"Writing snapshots: {args.snapshots_output}")

    start_ts = datetime.now(UTC).replace(microsecond=0)
    snapshots: list[dict[str, object]] = []
    intents: list[dict[str, object]] = []

    for i in range(args.samples):
        ts = start_ts + timedelta(seconds=i * args.interval_seconds)
        for cfg in FAKE_SYMBOLS:
            symbol = str(cfg["symbol"])
            price = make_price(cfg, i)
            spread_bps = float(cfg["spread_bps"])
            spread = price * spread_bps / 10_000.0
            bid = price - spread / 2.0
            ask = price + spread / 2.0
            snapshots.append({
                "timestamp_utc": ts.isoformat(),
                "symbol": symbol,
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "mid": round(price, 4),
                "last": round(price, 4),
                "close": round(price, 4),
                "volume": 100000 + i * 1000,
                "bid_size": 100,
                "ask_size": 100,
                "spread": round(spread, 4),
                "spread_bps": spread_bps,
                "reference_price": round(price, 4),
                "ready": True,
                "reason": "fake_snapshot",
                "score": 8.0,
                "or_high": round(price * 0.998, 4),
                "or_low": round(price * 0.992, 4),
                "or_breakout_pct": 0.2,
                "momentum_5_pct": 0.1,
                "samples": i + 1,
            })

    # Generate only a few deterministic intents.
    intent_specs = [
        ("QQQ", 6, 8.0, 100.0),
        ("NVDA", 7, 9.0, 150.0),
        ("TSLA", 8, 10.0, 200.0),
        ("RKLB", 9, 7.0, 100.0),
    ]
    for symbol, sample_idx, score, position_usd in intent_specs:
        cfg = next(c for c in FAKE_SYMBOLS if c["symbol"] == symbol)
        ts = start_ts + timedelta(seconds=sample_idx * args.interval_seconds)
        price = make_price(cfg, sample_idx)
        spread_bps = float(cfg["spread_bps"])
        spread = price * spread_bps / 10_000.0
        bid = price - spread / 2.0
        ask = price + spread / 2.0
        qty = max(1, int(position_usd // ask))
        intents.append({
            "timestamp_utc": ts.isoformat(),
            "intent_type": "BUY_INTENT",
            "symbol": symbol,
            "side": "BUY",
            "quantity": qty,
            "position_usd": position_usd,
            "reference_price": round(ask, 4),
            "limit_price": round(ask * 1.0002, 2),
            "order_type": "MARKETABLE_LIMIT_INTENT",
            "score": score,
            "reason": "fake_pipeline_test",
            "or_high": round(price * 0.998, 4),
            "or_low": round(price * 0.992, 4),
            "or_breakout_pct": 0.2,
            "momentum_5_pct": 0.1,
            "intraday_from_first_pct": 1.0,
            "spread_bps": spread_bps,
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "last": round(price, 4),
            "mid": round(price, 4),
            "status": "NOT_SENT_TO_BROKER",
        })

    snapshot_fields = [
        "timestamp_utc", "symbol", "bid", "ask", "mid", "last", "close", "volume", "bid_size", "ask_size",
        "spread", "spread_bps", "reference_price", "ready", "reason", "score", "or_high", "or_low",
        "or_breakout_pct", "momentum_5_pct", "samples",
    ]
    intent_fields = [
        "timestamp_utc", "intent_type", "symbol", "side", "quantity", "position_usd", "reference_price",
        "limit_price", "order_type", "score", "reason", "or_high", "or_low", "or_breakout_pct",
        "momentum_5_pct", "intraday_from_first_pct", "spread_bps", "bid", "ask", "last", "mid", "status",
    ]

    write_csv(Path(args.snapshots_output), snapshots, snapshot_fields)
    write_csv(Path(args.intents_output), intents, intent_fields)

    print(f"Snapshots written: {len(snapshots)}")
    print(f"Intents written: {len(intents)}")
    print("Run next:")
    print("python -m src.live_trading.v59_simulated_paper_bot_runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
