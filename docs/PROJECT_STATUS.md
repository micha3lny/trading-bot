# Project status

Last updated: 2026-05-03

## Current research focus

We are now focusing on **learning how to catch large momentum days** from the historical universe, especially symbols that produce:

- intraday high >= 10%,
- close-to-close daily gain >= 5% or >= 10%,
- strong early momentum / opening-range expansion.

The previous intraday reversal-pullback work is preserved, but the active research direction is now broader: understand the historical big winners first, then design a strategy that can catch them live.

## Data status

### Local daily opportunity scan

`daily_move_opportunity_scan` showed that the local daily database contains many large opportunities:

- total daily rows scanned: 74,031,
- max close-to-close daily return: 50.77%,
- max intraday high vs open: 67.05%,
- >= 5% close-to-close days: 3,366,
- >= 5% intraday-high days: 6,789,
- >= 10% close-to-close days: 686,
- >= 10% intraday-high days: 1,305.

Important symbols with many >=5% intraday opportunities include LUNR, DNA, SOUN, BBAI, MARA, RIOT, RXRX, SMCI, UPST, SEDG, ACHR, SOXS, SOXL, RKLB, COIN, AFRM, JOBY, UUUU, AMC, and MSTR.

### Current blocker: missing 1m data for most historical big winners

`big_momentum_research.py --min-intraday-high 10 --top 200` found:

- selected opportunities: 200,
- analyzed with 1m data: 2,
- missing 1m data: 198.

This means the daily database already tells us where the big opportunities were, but we cannot learn precise intraday entries/exits for most of them until 1m candles are backfilled.

The script writes:

- analyzed rows: `data/backtests/big_momentum_research_analyzed_intraday_ge_10_top200.csv`,
- missing 1m rows: `data/backtests/big_momentum_research_missing_1m_intraday_ge_10_top200.csv`,
- backfill shopping list: `data/backtests/big_momentum_research_symbols_to_backfill_intraday_ge_10_top200.csv`.

### IBKR 1m backfill

A new IBKR-based 1m backfill tool exists:

```bash
python -m src.data.fetch_1m_data --universe nasdaq --days 90 --port 4002
```

Current observed IBKR behavior:

- Gateway paper/simulated port is `4002`, not `7497`,
- error `1100` means temporary IBKR/Gateway connectivity loss,
- error `1102` means connectivity restored,
- error `162` often means IBKR has no HMDS data for a specific symbol/chunk,
- error `200` means IBKR cannot qualify the contract, often for warrants/units/invalid tickers.

The current goal is to fill `data/1m/` for the full active universe, especially the symbols listed by the big-momentum backfill CSV.

## What was done recently

### 1. Intraday reversal-pullback research was preserved

The best intraday reversal-pullback candidate remains v38:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality
```

Best strict result:

- signals: 10,
- active days: 9,
- win rate: 90%,
- average PnL after costs: about +0.78%,
- total PnL: about +7.79%,
- max drawdown in pseudo equity: about -1.20%.

More practical validation profile:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality --profile sample20
```

Result:

- signals: 20,
- active days: 16,
- win rate: 70%,
- average PnL after costs: about +0.23%,
- total PnL: about +4.63%,
- max drawdown: about -2.39%.

Decision: keep v38 as a separate intraday research track. Do not delete or overwrite it.

### 2. Overnight continuation v39 was tested

`backtest_reversal_pullback_v39_overnight.py` was created to test a similar entry without forced same-day exit.

Main observation:

- holding overnight did not materially improve the strategy on the current small sample,
- loosening stop-loss often increased win rate but worsened average loss and drawdown,
- lowering take-profit / trailing activation often cut winners too early,
- sample20 remained very sensitive to stop-loss clusters.

Current best command idea from v39 exploration:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v39_overnight \
  --preset quality \
  --profile sample20 \
  --stop-loss 2.0 \
  --take-profit 2.0 \
  --trailing-activation 1.5 \
  --trailing-stop 1.0 \
  --max-hold-days 2
```

But v39 is still exploratory and should not replace v38.

### 3. Big momentum research was added

`big_momentum_research.py` was added to analyze the daily winners and test diagnostic entries on the subset that already has 1m data.

Available 1m subset is tiny so far, but the first evidence is important:

- OR5 breakout and early pullback entries had high theoretical max PnL on the available rows,
- close-to-close holding captured much more of the big move than tight trailing exits,
- tight intraday trailing captured only small pieces of huge moves,
- these big days look more like momentum-continuation / opening-range expansion days than reversal-pullback days.

This suggests we need a separate strategy family for big momentum days.

## Current interpretation

The old v38/v39 strategy is trying to buy a controlled reversal after a red-day pullback. That is not the same pattern as the historical 10%+ momentum days.

The big winners often appear to be:

- gap / opening range / momentum expansion days,
- not clean selloff-reversal days,
- not necessarily visible to the v29/v38 broad candidate scanner,
- often missed because the scanner looks for pullback/reclaim context, not early momentum continuation.

Therefore the next strategy should not be a small tweak to v38. It should be a separate **Big Momentum Continuation** strategy.

## Current best strategy tracks

### Track A — Intraday Reversal Pullback

Status: preserved / paused.

Best file:

```text
src/strategies/momentum_trailing_intraday/backtest_reversal_pullback_v38_optimizer_winner.py
```

Best reference commands:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner --preset quality --profile sample20
```

### Track B — Overnight Reversal Continuation

Status: exploratory.

Best file:

```text
src/strategies/momentum_trailing_intraday/backtest_reversal_pullback_v39_overnight.py
```

Observation: no clear improvement over intraday yet.

### Track C — Big Momentum Continuation

Status: active research priority.

Goal: learn from historical daily winners, especially days with intraday high >= 10%.

Current files:

```text
src/analysis/daily_move_opportunity_scan.py
src/analysis/big_move_missed_entry_audit.py
src/analysis/momentum_day_pattern_analysis.py
src/analysis/big_momentum_research.py
src/data/fetch_1m_data.py
```

## Next steps

1. Continue IBKR 1m backfill for the universe:

```bash
python -m src.data.fetch_1m_data --universe nasdaq --days 90 --port 4002
```

2. After the backfill finishes, rerun:

```bash
python -m src.analysis.big_momentum_research --min-intraday-high 10 --top 200
python -m src.analysis.big_momentum_research --min-intraday-high 5 --top 500
```

3. Compare patterns on the now-larger 1m sample:

- opening range 5m / 15m / 30m breakout,
- first pullback after +3%,
- first pullback after +5%,
- hold-to-close vs trailing exit,
- time-to-high distribution,
- gap size distribution,
- drawdown after entry.

4. Design a new `big_momentum_continuation` backtest using the learned pattern.

5. Only after that, test live/paper behavior with real-time 1m candles.

## Important caution

Do not optimize by excluding individual losing tickers from history. The strategy should work live on unknown future symbols. Symbol filters should be structural only, for example:

- liquidity,
- price range,
- ADR / volatility,
- avoid warrants/rights/units if they cannot be traded cleanly,
- avoid invalid or unqualified IBKR contracts.
