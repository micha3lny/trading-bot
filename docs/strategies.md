# Trading Strategies

## 🎯 Current strategy direction

System is evolving into **two separate strategy families**:

1. Reversal Pullback (intraday / controlled bounce)
2. Big Momentum Continuation (new, main focus)

These are fundamentally different and must not be mixed.

---

## 1. Reversal Pullback (v29–v39)

### Idea

Trade intraday reversal after controlled pullback on a red day.

### Entry (best segment from optimizer)

- daily trend: -7% to -3%
- pullback: 1.2%–3%
- 1m close strength: 0.80–0.90
- entry risk: 4%–8%
- avoid extended bounce

### Exit (typical baseline)

- stop-loss: 1%–2%
- take-profit: 2%–3%
- trailing: ~1% after 1%–1.5% move

### What works

- controlled selloff
- clean reclaim
- good timing (not early / not late)

### What fails

- weak follow-through
- late entries
- noisy / low-quality tickers

### Profiles

- strict → high edge, very small sample
- sample20 → balanced (reference)
- scaled → too noisy

### Status

🟡 Preserved, not primary focus

This strategy captures a **specific niche pattern** and works, but it does not capture the largest historical winners.

---

## 2. Reversal Pullback Overnight (v39)

### Idea

Same entry as intraday reversal, but allow holding overnight.

### Key parameters tested

- larger stop-loss (1.5%–2%)
- delayed trailing activation
- max hold: 2–5 days

### Observations

- no clear improvement vs intraday version yet
- stop-loss clusters still dominate losses
- holding overnight does not automatically increase edge

### Status

🟡 Experimental

---

## 3. 🚀 Big Momentum Continuation (NEW MAIN FOCUS)

### Goal

Learn how to capture **large daily winners (>=5% / >=10%)** that appear in historical data.

### Key insight

From audits and scans:

- most 10%+ days are NOT reversal pullback days
- they are:
  - opening range breakouts
  - momentum expansion days
  - gap + continuation

### Typical pattern (based on available data)

- first 5–15 minutes already move several percent
- first 30 minutes often reach ~10% move
- pullbacks after +3% or +5% are shallow but tradable
- holding to close captures much more than tight trailing

### Candidate entry types

1. OR5 breakout (first 5m high break)
2. OR15 breakout
3. early pullback after +3%
4. pullback after +5%

### Candidate exits

- loose trailing (e.g. 15% activation / 10% trail)
- time-based exit
- hold-to-close

### Why current system misses them

- scanner looks for pullback on red days
- does not consider momentum continuation
- does not trigger on early breakout

### Required data

- 1m candles for full universe
- especially for days with intraday high >= 10%

### Status

🔴 PRIMARY RESEARCH TRACK

---

## Strategy architecture (important)

Each strategy must follow:

```text
ranking → entry → exit
```

But **ranking will differ between strategies**:

### Reversal Pullback ranking

- oversold
- controlled pullback

### Momentum Continuation ranking

- gap
- volume spike
- early momentum
- volatility

---

## Next implementation target

New strategy module:

```text
src/strategies/momentum_continuation/
  backtest_big_momentum_continuation_v1.py
```

Based on:

- OR breakout logic
- early momentum filters
- separate exit logic (not v38 exit)

---

## Important principle

Do NOT try to force one strategy to do everything.

- Reversal Pullback = niche edge
- Momentum Continuation = big winners

Both should coexist as independent systems.
