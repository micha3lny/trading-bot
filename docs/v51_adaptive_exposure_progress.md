# V51 Adaptive Exposure Progress

## Current best strategy

Current best-performing configuration:

- Strategy: `v51_aggressive_QQQ`
- Base engine: `v46 wide_trail`
- Market regime filter: `QQQ`
- Position sizing: adaptive exposure
- Dataset: last 90 trading days
- Entry: OR5 breakout

---

# Key insight from project evolution

The largest improvement did NOT come from micro-optimizing entries.

The largest improvement came from:

1. Removing low-quality / garbage momentum setups.
2. Filtering weak market regimes.
3. Dynamic position sizing based on setup quality and market strength.

Main discovery:

> Edge is stronger in allocation and exposure control than in tiny entry tweaks.

---

# Evolution summary

## Earlier versions

Earlier versions generated:

- too many trades,
- many garbage small-cap momentum names,
- weak risk-adjusted performance,
- unstable drawdowns.

The system was good at finding explosive momentum days, but poor at selecting when to increase or reduce exposure.

---

# v49 Market Regime

Added:

- QQQ / SPY / IWM 1m market data.
- VWAP regime filter.
- Market strength from open.
- Early market weakness detection.

Main finding:

Momentum continuation performs significantly better when:

- QQQ trades above VWAP,
- market is green from open,
- early market drawdown is limited.

v49 reduced:

- trade count,
- drawdown,
- weak market participation.

---

# v50 Adaptive Exposure

Added:

- setup quality scoring,
- market regime classification,
- dynamic position sizing.

Core concept:

- Strong market + high-quality setup => larger size.
- Weak market + weak setup => smaller or zero size.

Result:

- Profit improved substantially.
- Drawdown improved simultaneously.

This was the first major proof that dynamic exposure materially improves the system.

---

# v51 Aggressive Adaptive Exposure

v51 aggressively increased exposure for:

- `strong + A+`
- `strong + A`

and strongly reduced:

- `C` setups,
- bad market regimes.

## v51 Results

### Baseline fixed-size portfolio

- Profit: ~$2636
- Max drawdown: ~$531
- Avg position size: $1000

### v51 aggressive adaptive portfolio

- Profit: ~$4273
- Max drawdown: ~$378
- Avg position size: ~$1158
- Max position size: $4000

Key result:

- Profit improved by ~62% versus baseline.
- Drawdown improved at the same time.

This is currently the best result achieved in the project.

---

# Important findings from v51

## Strong market regime is extremely important

`strong` regime generated the majority of profits.

The best conditions:

- QQQ above VWAP,
- positive market momentum,
- strong opening continuation.

---

## Setup quality scoring works

Most profits came from:

- `A`
- `A+`

Low-quality `C` setups contributed very little.

This confirms:

- the scoring engine is meaningful,
- setup ranking contains real signal.

---

# Current system status

The strategy now includes:

- OR breakout entries,
- momentum continuation logic,
- market regime filtering,
- adaptive exposure,
- setup quality scoring,
- drawdown-aware allocation,
- realistic ETF-based market context.

The project now resembles a real systematic intraday momentum engine rather than a simple backtest.

---

# Next steps

## v52 / realistic execution simulation

Planned improvements:

- max concurrent positions,
- portfolio exposure limits,
- realistic cash management,
- slippage simulation,
- spread penalty,
- partial fills,
- execution queue,
- max daily loss stop,
- halt simulation.

Goal:

Move from signal-quality research into realistic live-trading portfolio simulation.

---

# Current conclusion

The strategy appears to have a real edge when:

- momentum market conditions are favorable,
- exposure is scaled intelligently,
- low-quality setups are suppressed.

The project focus should now move toward:

- execution realism,
- portfolio management,
- live trading infrastructure,
- risk management.
