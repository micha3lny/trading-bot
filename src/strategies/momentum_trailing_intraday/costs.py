"""Trading cost model for intraday backtests.

The default model is deliberately conservative for small retail IBKR Pro trades:
- commission is modeled as a round-trip percentage of notional
- slippage is modeled as a round-trip percentage of notional

IBKR stock commissions vary by pricing plan, country, routing, volume, minimums,
and whether the account uses fixed or tiered pricing. Keeping this as percentage
inputs makes it easy to run sensitivity tests without pretending we know exact fills.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostModel:
    commission_round_trip_pct: float = 0.10
    slippage_round_trip_pct: float = 0.10

    @property
    def total_round_trip_pct(self) -> float:
        return self.commission_round_trip_pct + self.slippage_round_trip_pct


DEFAULT_COST_MODEL = CostModel()


def apply_cost_to_trade(trade, cost_model: CostModel = DEFAULT_COST_MODEL):
    """Return a BacktestTrade-like object with costs deducted from pnl/capital pnl."""
    net_pnl_pct = trade.pnl_pct - cost_model.total_round_trip_pct
    net_capital_pnl = trade.capital_pnl - (trade.position_weight * 10_000.0 * cost_model.total_round_trip_pct / 100.0)
    return replace(trade, pnl_pct=net_pnl_pct, capital_pnl=net_capital_pnl)


def apply_costs_to_trades(trades, cost_model: CostModel = DEFAULT_COST_MODEL):
    return [apply_cost_to_trade(trade, cost_model) for trade in trades]
