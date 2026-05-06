# Momentum Research Status — May 2026

## Current Goal

The project focus has shifted from generic reversal strategies into a dedicated intraday momentum-continuation engine.

Main objective:
- detect stocks capable of large intraday expansion
- trade OR (opening range) breakouts and momentum continuation
- simulate execution as close as possible to real intraday trading
- eventually run the strategy fully live through IBKR

---

## Current Best Benchmark

### Best strategy so far

As of the latest research run, the best balanced live-like benchmark is:

`v46 trend exit / wide_trail`

Command:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v46_trend_exit \
  --recent-days 90 \
  --opening-range-minutes 5
```

Best current variant:

`wide_trail`

Observed result on local 90-day 1m dataset:

| Metric | Value |
| --- | ---: |
| Trades | 738 |
| Active days | 62 |
| Symbols | 278 |
| Win rate | 59.21% |
| Avg PnL/trade | +0.36% |
| Median PnL/trade | +0.45% |
| Total PnL points | +263.63 |
| Fixed $1000/trade simulated profit | ~$2636 |
| Max loss | -8.00% |
| Max win | +18.56% |

Interpretation:
- `wide_trail` is currently the best compromise.
- It earns more than tight trailing while keeping median positive.
- `close_exit` still earns more total PnL, but has a weak/near-zero median and depends more heavily on a few large runners.
- `v45_trail_only` has stronger median but cuts winners too aggressively.

Important:
- `total_pnl` is a sum of percentage trade results, not portfolio return.
- With a fixed $1000 position per trade, profit is approximately `total_pnl / 100 * 1000`.
- Therefore `total_pnl=263.63` means about `$2636` profit with $1000 position size on every trade.

---

## What Was Built

### 1. Large-scale 1m historical backfill from IBKR

We added a dedicated IBKR 1-minute downloader.

Current dataset:
- ~3500 symbols downloaded
- ~90 recent trading days
- minute candles stored locally in `data/1m`
- chunked downloading to respect IBKR HMDS limits
- failed symbols saved separately

The system now uses real broker market data instead of fake/generated candles.

---

### 2. Local 1m-only momentum research

New research flow:

`big_momentum_available_1m_research.py`

Purpose:
- analyze ONLY locally available 1m data
- avoid daily-only historical rows without minute candles
- build real momentum datasets from local files

The first scan over all files is expensive because:
- ~3500 CSV files
- ~250k+ sessions
- millions of minute rows

Future runs are expected to become faster after:
- caching
- symbol prefiltering
- watchlist engine

---

### 3. Clean momentum research filter

New cleaner research pipeline:

`big_momentum_available_1m_clean_research.py`

This removes low-quality setups and obvious garbage symbols.

Current quality filters:
- rows_1m >= 300
- abs(gap_pct) <= 30
- intraday_high_pct <= 80 for research sets only
- first_5m_high_pct <= 50
- exclude leveraged ETFs
- exclude weird / broken symbols

Important distinction:
- Research scripts may use final-day information to study known big movers.
- Live-like strategy scripts must not use final intraday high, open-to-close, or time-to-high as entry filters.

---

## Strategy Timeline

### v40 — OR breakout momentum engine

Implemented:

`backtest_momentum_or_breakout_v40.py`

Features:
- OR5 breakout entries
- stop loss
- trailing stop
- staged profit taking
- multiple exit variants
- clean-symbol filters
- initial momentum continuation testing

v40 showed that OR breakout works well on known big-momentum days, but some early experiments still used research-only filters.

---

### v41 — tradability filters and live-like baseline

Implemented:

`backtest_momentum_or_breakout_v41_tradable.py`

Main result:
- v41 became the first broad live-like baseline.
- It scanned ~270k symbol-day sessions.
- It produced about 2040 accepted entries.
- Result was close to breakeven.

Key lesson:
- OR breakout alone is not enough.
- Entry logic can catch momentum runners, but it also catches too many weak/fake breakouts.
- Stock/setup selection matters more than micro-tuning the entry.

---

### v42 — score-based momentum filter

Implemented:

`backtest_momentum_or_breakout_v42_score.py`

Goal:
- reduce trade count without imposing fixed daily trade limits.
- use a quality score instead of hard limiting trades/day.

Result:
- trade count dropped significantly.
- performance did not improve enough.

Lesson:
- the first momentum score mainly reduced activity.
- it did not separate winners from losers strongly enough.

---

### v43 — winners vs losers diagnostics

Implemented:

`v43_momentum_trade_diagnostics.py`

Purpose:
- compare winners vs losers from v41/v42/v45/v46 trade CSVs.
- analyze feature separation.
- inspect symbol repeatability.

Most important findings:
- winners tend to become trend days.
- losers are often early spike-and-fade moves.
- OR high alone does not distinguish winners from losers.
- time-to-high and open-to-close are strong diagnostic separators, but cannot be used as live entry filters.
- repeatable symbol behavior exists.

Repeatably strong symbols observed in diagnostics included examples such as:
- NBIS
- SATL
- LPTH
- INTC
- IBRX
- NVTS
- POET

Repeatably weak symbols included examples such as:
- MNDY
- OPEN
- SANA
- HIVE
- IMUX

Lesson:
- v47 should add rolling symbol quality instead of relying only on raw OR breakout properties.

---

### v44 — hard continuation confirmation

Implemented:

`backtest_momentum_or_breakout_v44_continuation.py`
`backtest_momentum_or_breakout_v44_live.py`

Goal:
- avoid fake breakouts by waiting for hold / pullback / second-push confirmation.

Result:
- research mode looked strong, but that was not fully live-like.
- after removing future-aware filters, the strategy became too restrictive.
- trade count collapsed too much.

Lesson:
- hard continuation confirmation helps conceptually, but it kills too many real opportunities.
- v45 should return to the broader v41-style OR breakout, with soft anti-fade scoring instead of hard confirmation.

---

### v45 — anti-fade scoring on broad OR breakout

Implemented:

`backtest_momentum_or_breakout_v45_antifade.py`

Goal:
- keep broad OR breakout entries.
- add live-like anti-fade score.
- reduce garbage trades without over-filtering.

Observed result:
- ~734 accepted entries.
- ~62 active days.
- ~276 symbols.
- no future filters in rejection reasons.

Main variants:

| Variant | Total PnL | Median | Comment |
| --- | ---: | ---: | --- |
| trail_only | +165.10 | +1.09% | Stable but cuts winners fast |
| close_exit | +396.46 | ~0% | Captures runners but less stable |

Lesson:
- entry/filtering improved.
- tight trailing created stability but capped large winners.
- close_exit revealed that larger runners still exist and are being cut too early.

---

### v46 — trend-aware exits on v45 entries

Implemented:

`backtest_momentum_or_breakout_v46_trend_exit.py`

Goal:
- keep v45 entry/selection.
- test better exits to recover larger momentum runners.

Variants tested:
- `v45_trail_only`
- `close_exit`
- `wide_trail`
- `tp_wide_trail`
- `breakeven_trend`
- `staged_runner`
- `ratchet`

Best balanced result:

`wide_trail`

Why `wide_trail` is the current best benchmark:
- higher total PnL than tight trailing.
- positive median.
- less dependent on a few large close-to-close winners than `close_exit`.
- better compromise between stability and upside.

Result summary:
- count: 738
- active days: 62
- symbols: 278
- win rate: 59.21%
- avg PnL: +0.36%
- median PnL: +0.45%
- total PnL: +263.63
- fixed $1000/trade profit: about +$2636

---

## Key Research Findings

### 1. Momentum continuation is real

Research results strongly suggest:
- large momentum days frequently continue after OR5 breakout.
- many stocks trend for large parts of the session.
- continuation setups outperform reversal-style thinking on these days.

---

### 2. Entry is not the only edge

The biggest performance drivers so far:
- stock/setup selection
- avoiding spike-and-fade names
- letting real runners breathe
- avoiding over-tight exits

---

### 3. Fixed daily trade limits are not desired

We do not want a rule like "max 5 trades/day".

Reason:
- some days should have 0 trades.
- some strong momentum days may have 10 valid setups.

Preferred approach:
- quality threshold / score.
- rolling symbol quality.
- premarket watchlist.
- risk/exposure limits instead of arbitrary trade count limits.

---

### 4. Symbol quality is the next edge candidate

Diagnostics show repeatable behavior by symbol.

Some symbols repeatedly produce follow-through.
Some repeatedly produce fake breakouts.

Therefore v47 should add:
- rolling symbol-quality score.
- prior-trade performance memory.
- no static blacklist as the first step.
- no future leakage.

---

## Next Step — v47

### v47 goal

Add a rolling symbol-quality layer on top of:

`v45 entries + v46 wide_trail exit`

The symbol-quality layer should use only information known before the current trade:
- previous trades for that symbol.
- rolling average PnL.
- rolling median PnL.
- previous stop-out rate.
- previous win rate.
- previous MFE/MAE behavior.

Important:
- the current trade must not influence its own symbol score.
- future trades must not influence current decisions.

Expected v47 behavior:
- keep new symbols initially allowed.
- progressively penalize symbols that repeatedly fail.
- reward symbols with repeated follow-through.
- reduce losses from repeatable bad tickers.
- avoid static hard blacklist initially.

---

## Production Direction

The live bot should eventually work as:

### Premarket phase
- scan tradable universe.
- filter garbage symbols.
- calculate premarket relative volume and gap.
- build momentum watchlist.

### Market open phase
- monitor watchlist.
- detect OR breakout.
- apply anti-fade score.
- apply rolling symbol quality.
- enter only high-quality setups.

### Position management
- use current best exit benchmark: `wide_trail`.
- later replace with portfolio/risk-aware sizing.

---

## Current Status

Current state:
- real IBKR 1m data ingestion works.
- large local 1m dataset exists.
- local 1m research pipeline works.
- clean momentum research pipeline works.
- OR breakout backtester works.
- no-lookahead live-like backtesting is now enforced for v45/v46.
- multiple exit models tested.
- current best benchmark is v46 `wide_trail`.
- next research step is v47 rolling symbol quality.

The project is now transitioning from:

research -> realistic execution modeling -> symbol quality / watchlist engine -> portfolio simulation -> live IBKR execution
