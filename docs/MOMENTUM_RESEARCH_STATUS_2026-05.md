# Momentum Research Status — May 2026

## Current Goal

The project focus has shifted from generic reversal strategies into a dedicated intraday momentum-continuation engine.

Main objective:
- detect stocks capable of large intraday expansion
- trade OR (opening range) breakouts and momentum continuation
- simulate execution as close as possible to real intraday trading
- eventually run the strategy fully live through IBKR

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
- intraday_high_pct <= 80
- first_5m_high_pct <= 50
- exclude leveraged ETFs
- exclude weird / broken symbols

This dramatically improved realism of the learning set.

---

## Key Research Findings

### Momentum continuation is real

Research results strongly suggest:
- large momentum days frequently continue after OR5 breakout
- many stocks trend until close
- continuation setups outperform reversal-style thinking on these days

Observed characteristics:
- average intraday move often 20%+
- many clean continuation days
- average OR5 breakout profitability was positive across thousands of trades

---

## Current Strategy Generation

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
- no future leakage during trade simulation

The engine simulates:
- candle-by-candle progression
- entries only after breakout confirmation
- exits based only on information available at that candle

This is significantly closer to real intraday trading.

---

## Important Result

One of the strongest observations:

`close_exit` outperformed aggressive trailing exits.

Interpretation:
- many momentum stocks continue trending most of the session
- early trailing exits may cut winners too fast
- momentum continuation may work better with wider management

This is now a major research direction.

---

## What We Learned About the Market

The dataset includes:
- normal NASDAQ / NYSE stocks
- microcaps
- low-float momentum runners
- biotech/news runners
- speculative momentum names
- some garbage and illiquid symbols

We intentionally analyze broad market data first.

Reason:
- the model must learn BOTH good and bad momentum behavior
- filtering too early could hide useful patterns

However:
- trading engine filters now progressively remove obvious garbage
- future live engine will focus only on tradable liquid candidates

---

## Planned v41 Improvements

v41 goals:
- stronger tradability filters
- liquidity filtering
- minimum volume thresholds
- spread approximations
- better symbol blacklist
- improved stop execution realism
- avoid impossible fills

Goal:
make simulations closer to real fills and real trading constraints.

---

## Planned v42 Improvements

### Daily Watchlist Engine

The live bot will NOT scan 3500 symbols continuously.

Instead:

### Premarket phase
- scan universe
- filter garbage symbols
- select top momentum candidates
- build watchlist (30–100 symbols)

Potential filters:
- premarket gap
- relative volume
- float
- liquidity
- news
- volatility
- price range

### Market open phase
- monitor only watchlist
- detect OR breakout
- manage positions live

This is the intended production architecture.

---

## Current Status

Current state:
- real IBKR 1m data ingestion works
- large local dataset exists
- clean momentum research pipeline works
- OR breakout backtester works
- no-lookahead simulation implemented
- multiple exit models tested
- continuation behavior confirmed statistically

The project is now transitioning from:

research -> realistic execution modeling -> live watchlist engine
