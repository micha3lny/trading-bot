# v44 continuation-filter strategy

## Problem discovered in v43

The v42 momentum score reduces trade count, but does NOT separate winners from losers strongly enough.

Key findings from diagnostics:

- OR breakout strength alone is not enough.
- Winners behave like trend days.
- Losers behave like early spike-and-fade days.
- Biggest separator is NOT first impulse.
- Biggest separator is continuation quality after breakout.

Examples:

Winner median:
- time_to_high ~= 263 min
- open_to_close ~= +15%
- intraday_high ~= +17%

Loser median:
- time_to_high ~= 13 min
- open_to_close ~= -1%
- intraday_high ~= +9%

Interpretation:

Losers usually:
- spike early,
- fail quickly,
- lose OR high,
- fade intraday.

Winners usually:
- hold breakout,
- trend for hours,
- keep momentum after OR.

---

# v44 hypothesis

We should NOT optimize pure OR breakout score.

We should trade:

"OR breakout + continuation confirmation"

instead of:

"OR breakout only"

---

# Planned v44 filters

## 1. OR hold filter

After breakout:
- price must hold above OR high for X minutes
- reject immediate rejection candles

Goal:
remove fake breakout spikes.

---

## 2. VWAP hold filter

Require:
- price above VWAP after breakout
- no fast reclaim below VWAP

Goal:
avoid weak momentum.

---

## 3. Second impulse confirmation

Require:
- higher low after breakout
- second push continuation

Pattern:
- breakout
- pullback
- higher low
- continuation

This is likely the biggest edge candidate.

---

## 4. Spike-fade rejection

Reject:
- giant 1-2 candle vertical spikes
- candles with huge wick rejection
- immediate reversal after breakout

Goal:
avoid low-liquidity traps.

---

## 5. Better entry timing

v43 showed:

Bad:
- 10-15 min after open

Better:
- 5-10 min
- 15-30 min

v44 should explicitly test timing windows.

---

## 6. Symbol quality memory

Some symbols repeatedly perform better.

Examples:
- SATL
- NBIS
- LWLG
- NVTS
- POET

Some repeatedly fail.

Examples:
- ONDS
- RCAT
- OPEN
- MRNA

v44 should introduce:
- rolling symbol-quality score
- dynamic trust score
- NOT static blacklist

---

# Most important next experiment

Instead of:
- buy instantly on OR breakout

Test:

1. OR breakout
2. Wait for pullback
3. Require hold above OR high
4. Require higher low
5. Enter on second push

Hypothesis:
- fewer trades
- lower win size
- MUCH higher consistency
- much lower stop-outs

---

# Long-term direction

The strategy is evolving toward:

"intraday momentum continuation detection"

NOT:

"simple breakout scalping"

That is important because:
- trend continuation matters more than initial spike.
- quality of continuation is the real edge.
