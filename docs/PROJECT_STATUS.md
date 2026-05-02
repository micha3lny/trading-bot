# Project status

Last updated: 2026-05-02

## Current research phase

The current research focus is an intraday reversal/pullback strategy on local historical data:

- universe: 99 local symbols with 1D + 15m + 5m + 1m data,
- sample window used in the latest tests: roughly 90 trading days,
- strategy family: momentum trailing intraday / reversal pullback,
- current preferred entry source: v29 simple entry scan, optimized through v38,
- current exit source: v22/v30 smart intraday exit.

The intraday strategy is not finished, but it has produced a credible research candidate. It should be preserved as a separate intraday strategy and revisited later, especially after testing on more aggressive / more volatile exchanges or a larger data universe.

## What was done recently

### 1. Original multi-timeframe entry was too restrictive

Earlier versions used a strict sequence:

1. daily trend filter,
2. 15m breakout attempt,
3. 5m pullback,
4. 1m confirmation trigger,
5. v22 smart exit.

This produced too few entries for 99 symbols over about 90 days. Examples:

- v22/v23 style entries: only a few trades,
- v24 with looser regime/ADR: 11 trades,
- v25/v26 attempts either stayed too small or overfit badly.

Conclusion: the old MTF entry path was too narrow and likely entered after the move was already too filtered.

### 2. Entry was separated from exit

A dedicated entry scan was created to measure opportunity count without letting the exit logic hide the entry problem.

Important scanners:

- `reversal_pullback_entry_scan.py` — diagnostic version with balanced/loose/quality presets,
- `reversal_pullback_entry_scan_v29_simple.py` — simplified 5m/1m pullback/reclaim scanner.

The v29 simple scanner showed that opportunity count exists:

- quality preset: 1105 candidates, 61 active days, about 18.11 candidates per active day,
- balanced preset: 1566 candidates,
- loose preset: 2716 candidates.

Conclusion: the market data does contain many potential entries, but most raw entries are noisy and need context filters.

### 3. Raw v30 simple-entry backtest was too noisy

`backtest_reversal_pullback_v30_simple_entry_exit.py --preset quality` tested the large v29 quality entry pool with v22-style smart exit.

Result summary:

- signals: 1105,
- win rate: about 38%–42% depending on cost/output variant,
- average PnL after costs: negative,
- large cumulative drawdown,
- many stop-loss exits.

Conclusion: old-bot-like broad entry creates many opportunities but too much noise. The strategy cannot simply trade every simple reclaim.

### 4. Context filters v31–v35 found very small high-quality pockets

Several context filters were tested:

- v31: 23 trades, about break-even,
- v32: 4 trades, 100% win rate, too small,
- v33: 12 trades, positive but still small,
- v34/v35: small samples, sometimes positive but not enough entries.

Conclusion: high-quality pockets exist, but manual filtering was unstable and tended to overfit.

### 5. Entry timing audit confirmed the core problem

`reversal_pullback_entry_audit.py` was introduced to classify entries:

- good_timing,
- late_after_bounce,
- weak_followthrough,
- too_early_or_wrong.

On the broad quality pool, many trades were either late after the bounce or weak follow-through. The good-timing subset had much better MFE/MAE characteristics.

Conclusion: the problem is primarily entry timing and context, not just exit mechanics.

### 6. Early trigger attempts v36/v37 did not improve the strategy

v36 and v37 tested earlier reclaim-style entries. They increased or changed the entry set, but results were poor:

- v36: 44 signals, negative average PnL,
- v37: 10 signals, negative average PnL.

Conclusion: entering earlier is not automatically better. A good entry needs both timing and quality context.

### 7. Data-driven optimizer was added

`reversal_pullback_entry_optimizer.py` was added to stop guessing filters manually.

It:

- builds a broad candidate pool,
- simulates the current intraday exit on every candidate,
- calculates timing features such as pre-entry bounce and future MFE/MAE,
- runs a grid search over entry filters,
- saves CSVs under `data/backtests/`.

Important output from `--preset quality --min-sample 10`:

- broad pool: 1105 trades, 61 active days, average PnL about -0.071%, total about -77.95%, stop rate about 48.96%,
- best optimizer segment: 10 trades, 9 active days, 90% win rate, average PnL about +0.98% before exact v38 reproduction differences.

Best data-driven entry segment:

- daily trend: -7% to -3%,
- pullback proxy: 1.2% to 3.0%,
- 1m close strength: 0.80 to 0.90,
- entry risk: 4% to 8%,
- pre-15m low-to-entry: <= 2.0%,
- 5m close strength: <= 0.75,
- distance below opening-range high: 0% to 5%,
- noisy leveraged ETFs excluded.

### 8. v38 preserves the best intraday candidate

`backtest_reversal_pullback_v38_optimizer_winner.py` was added with three profiles:

- `strict` — exact optimizer winner,
- `sample20` — middle profile with more trades,
- `scaled` — looser validation profile.

Latest known results:

#### v38 strict

Command:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality
```

Summary:

- signals: 10,
- active days: 9,
- win rate: 90%,
- average PnL after costs: about +0.78%,
- total PnL: about +7.79%,
- max drawdown in pseudo equity: about -1.20%,
- only one stop-loss trade.

This is the best-performing version, but the sample is small.

#### v38 sample20

Command:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality --profile sample20
```

Summary:

- signals: 20,
- active days: 16,
- win rate: 70%,
- average PnL after costs: about +0.23%,
- total PnL: about +4.63%,
- max drawdown in pseudo equity: about -2.39%.

This profile is less profitable than strict, but more credible because it has a larger sample.

#### v38 scaled

Command:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality --profile scaled
```

Summary:

- signals: 29,
- active days: 22,
- win rate: about 55%,
- average PnL after costs: about -0.06%,
- total PnL: about -1.86%.

This profile suggests that loosening too much reintroduces noise.

## Current best intraday setup

The current best intraday research candidate is:

- file: `backtest_reversal_pullback_v38_optimizer_winner.py`,
- command: `python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality`,
- profile: `strict`,
- role: best high-quality intraday candidate, not final production strategy.

The more robust validation candidate is:

- same file,
- command: `python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality --profile sample20`,
- role: practical reference profile with more trades.

## Current interpretation

The intraday setup appears to work only in a narrow window:

- moderate daily selloff, not panic crash,
- clean pullback/reclaim,
- entry after a controlled bounce, not too early and not too late,
- 1m close strength strong but not extreme,
- risk window around 4%–8%, with best bins near 5%–6%,
- noisy leveraged ETFs should stay excluded.

The strategy should not be expanded by simply loosening all filters. Looser profiles quickly add stop-loss noise.

## Known weak points

- Strict profile has only 10 trades, so it may be overfit.
- Sample20 is positive but still small.
- Scaled profile turns slightly negative.
- Losses in sample20 are mostly stop-losses and concentrated in names such as BBAI, DNA, JOBY, RDDT, and one UUUU loss.
- Larger data history and more symbols/exchanges are needed before production use.

## Decision

Keep this intraday strategy as a separate research track.

Do not delete or overwrite v38. The intraday work should be paused here and revisited later with:

1. more historical data,
2. more aggressive / more volatile exchanges,
3. optional symbol hygiene tests,
4. stop-loss sensitivity tests,
5. out-of-sample validation.

## Next planned strategy

Start a second strategy based on similar entry logic, but without the forced same-day exit requirement.

Working name:

- multi-day reversal continuation,
- overnight reversal pullback,
- or swing continuation from intraday reclaim.

High-level idea:

- use a similar high-quality reclaim entry,
- keep intraday risk controls for the entry day,
- if the trade is positive/healthy near end of day, allow holding overnight,
- test max hold of 2–5 days,
- use separate exits and separate reporting from the intraday strategy.

This must be implemented as a second strategy, not as a replacement for the current intraday strategy.
