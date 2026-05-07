from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_EQUITY = "data/live/v59_simulated_equity_curve.csv"
DEFAULT_EVENTS = "data/live/v59_simulated_trade_events.csv"
DEFAULT_KILL_SWITCH = "data/live/KILL_SWITCH"
DEFAULT_OUTPUT = "data/live/v60_risk_guard_status.csv"


@dataclass
class RiskGuardConfig:
    max_daily_loss_usd: float = 250.0
    max_open_positions: int = 3
    max_gross_exposure_usd: float = 1_000.0
    max_pending_intents: int = 20
    max_stale_seconds: int = 60
    stop_loss_cooldown_minutes: int = 30
    kill_switch_path: str = DEFAULT_KILL_SWITCH


@dataclass
class RiskGuardResult:
    timestamp_utc: str
    allow_trading: bool
    reasons: list[str] = field(default_factory=list)
    realized_net_pnl_usd: float = 0.0
    open_positions: int = 0
    open_value_usd: float = 0.0
    pending_intents: int = 0
    stale_seconds: float | None = None
    stop_loss_cooldowns: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "allow_trading": self.allow_trading,
            "reasons": ";".join(self.reasons),
            "realized_net_pnl_usd": self.realized_net_pnl_usd,
            "open_positions": self.open_positions,
            "open_value_usd": self.open_value_usd,
            "pending_intents": self.pending_intents,
            "stale_seconds": self.stale_seconds,
            "stop_loss_cooldowns": self.stop_loss_cooldowns,
        }


class RiskGuard:
    def __init__(self, config: RiskGuardConfig):
        self.config = config

    def evaluate(
        self,
        *,
        equity_df: pd.DataFrame,
        events_df: pd.DataFrame,
        now: datetime | None = None,
    ) -> RiskGuardResult:
        now = now or datetime.now(UTC)
        reasons: list[str] = []

        kill_switch = Path(self.config.kill_switch_path)
        if kill_switch.exists():
            reasons.append("kill_switch_active")

        latest_equity = equity_df.tail(1).copy() if not equity_df.empty else pd.DataFrame()

        realized_net_pnl = 0.0
        open_positions = 0
        open_value = 0.0
        pending_intents = 0
        stale_seconds = None

        if latest_equity.empty:
            reasons.append("no_equity_data")
        else:
            row = latest_equity.iloc[0]
            ts = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
            if pd.notna(ts):
                stale_seconds = max(0.0, (now - ts.to_pydatetime()).total_seconds())
                if stale_seconds > self.config.max_stale_seconds:
                    reasons.append("stale_market_or_equity_data")
            else:
                reasons.append("invalid_equity_timestamp")

            realized_net_pnl = float(pd.to_numeric(pd.Series([row.get("realized_net_pnl_usd", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            open_positions = int(pd.to_numeric(pd.Series([row.get("open_positions", 0)]), errors="coerce").fillna(0).iloc[0])
            open_value = float(pd.to_numeric(pd.Series([row.get("open_value_usd", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            pending_intents = int(pd.to_numeric(pd.Series([row.get("pending_intents", 0)]), errors="coerce").fillna(0).iloc[0])

            if realized_net_pnl <= -abs(self.config.max_daily_loss_usd):
                reasons.append("max_daily_loss_breached")
            if open_positions > self.config.max_open_positions:
                reasons.append("max_open_positions_breached")
            if open_value > self.config.max_gross_exposure_usd:
                reasons.append("max_gross_exposure_breached")
            if pending_intents > self.config.max_pending_intents:
                reasons.append("too_many_pending_intents")

        cooldown_count = 0
        if not events_df.empty:
            ev = events_df.copy()
            ev["timestamp_utc"] = pd.to_datetime(ev["timestamp_utc"], utc=True, errors="coerce")
            recent_cutoff = now - timedelta(minutes=self.config.stop_loss_cooldown_minutes)
            recent_stops = ev[
                (ev["event"] == "EXIT")
                & (ev["reason"] == "stop_loss")
                & (ev["timestamp_utc"] >= recent_cutoff)
            ]
            cooldown_count = len(recent_stops)
            if cooldown_count > 0:
                reasons.append("stop_loss_cooldown_active")

        return RiskGuardResult(
            timestamp_utc=now.isoformat(),
            allow_trading=len(reasons) == 0,
            reasons=reasons,
            realized_net_pnl_usd=realized_net_pnl,
            open_positions=open_positions,
            open_value_usd=open_value,
            pending_intents=pending_intents,
            stale_seconds=stale_seconds,
            stop_loss_cooldowns=cooldown_count,
        )


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="v60 risk guard / kill switch evaluator")
    parser.add_argument("--equity", default=DEFAULT_EQUITY)
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--kill-switch", default=DEFAULT_KILL_SWITCH)
    parser.add_argument("--max-daily-loss-usd", type=float, default=250.0)
    parser.add_argument("--max-open-positions", type=int, default=3)
    parser.add_argument("--max-gross-exposure-usd", type=float, default=1_000.0)
    parser.add_argument("--max-pending-intents", type=int, default=20)
    parser.add_argument("--max-stale-seconds", type=int, default=60)
    parser.add_argument("--stop-loss-cooldown-minutes", type=int, default=30)
    args = parser.parse_args()

    config = RiskGuardConfig(
        max_daily_loss_usd=args.max_daily_loss_usd,
        max_open_positions=args.max_open_positions,
        max_gross_exposure_usd=args.max_gross_exposure_usd,
        max_pending_intents=args.max_pending_intents,
        max_stale_seconds=args.max_stale_seconds,
        stop_loss_cooldown_minutes=args.stop_loss_cooldown_minutes,
        kill_switch_path=args.kill_switch,
    )

    equity = read_csv(args.equity)
    events = read_csv(args.events)
    guard = RiskGuard(config)
    result = guard.evaluate(equity_df=equity, events_df=events)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.to_dict()]).to_csv(out, index=False)

    print("=== v60 risk guard ===")
    print(f"allow_trading: {result.allow_trading}")
    print(f"reasons: {', '.join(result.reasons) if result.reasons else 'none'}")
    print(f"realized_net_pnl_usd: {result.realized_net_pnl_usd:.2f}")
    print(f"open_positions: {result.open_positions}")
    print(f"open_value_usd: {result.open_value_usd:.2f}")
    print(f"pending_intents: {result.pending_intents}")
    if result.stale_seconds is not None:
        print(f"stale_seconds: {result.stale_seconds:.1f}")
    print(f"stop_loss_cooldowns: {result.stop_loss_cooldowns}")
    print(f"Saved: {out}")

    print("\nUsage:")
    print(f"- Emergency stop: touch {args.kill_switch}")
    print(f"- Clear stop: rm {args.kill_switch}")
    return 0 if result.allow_trading else 2


if __name__ == "__main__":
    raise SystemExit(main())
