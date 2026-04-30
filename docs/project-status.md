# Project Status / Start Here

## Aktualny status

Projekt: `trading-bot`

Repozytorium:

```text
https://github.com/micha3lny/trading-bot
```

---
## 📊 Data Layer Update (April 2026)

### New datasets available
- 1D candles
- 15m candles
- 5m candles
- 1m candles (NEW)

### Coverage
- ~99 symbols
- Full intraday coverage (90 days)

### Impact
- Enabled precise entry timing (v17+)
- Allowed shift to MTF strategies
- Significantly increased research depth
## 🚨 NOWY KIERUNEK (AKTUALNY)

### Strategia: Reversal Pullback (MTF)

Opis:
- 15m: oversold + breakout attempt
- 5m: pullback entry (mean reversion)

To NIE jest już breakout strategy.
To jest:

```text
reversal after failed breakout
```

---

## 🥇 BEST CONFIG (stan na teraz)

### v12 (baseline)

Parametry:

- 15m:
  - trend <= -5%
  - breakout: 0.5% – 2.5%
  - confirmation_cs: 0.4 – 0.95

- 5m entry:
  - pullback: 0.3% – 2.5%
  - close_strength <= 0.60
  - max below OR: 0.75%

- Exit:
  - stop: ~1.2%
  - trailing

- Costs: included

### Wynik (research, pseudo equity):

- Signals: ~49
- Winrate: ~39%
- Avg trade: ~0.15%
- Total PnL: ~+7%

- Equity:
```text
Initial: 10000
Final:   ~10650
Return:  ~6.5%
DD:      ~-14%
```

---

## 🔍 Kluczowe odkrycia

1. Momentum NIE działa
2. Follow-through NIE działa
3. Pullback działa
4. Najlepsze wejścia to:

```text
low close_strength (panic candles)
```

---

## 🚧 Aktualne eksperymenty

### v13

Cel:
- tylko panic entry (cs <= 0.35)
- breakout >= 1.0
- większy stop
- time-based exit

---

## GŁÓWNY PROBLEM

```text
wysoki drawdown (~14%)
```

---

## NEXT STEP

- poprawa entry (panic only)
- zmiana exit (time-based)
- stabilizacja equity

---

## START

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v12_mtf_filtered
```
## 📅 2026-04 — Reversal Pullback MTF (1m / 5m / 15m)

### 🎯 Cel
Zbudowanie strategii intraday typu **reversal momentum** z użyciem:
- 15m → setup (kontekst)
- 5m → pullback
- 1m → precyzyjny entry

---

## 🧪 Iteracje

### v12 — baseline (MTF pullback)
- 15m oversold + breakout attempt
- 5m pullback entry
- trailing stop / session exit

**Wynik:**
- Total PnL: ~7%
- Win rate: ~38%
- Max DD: ~14%

✅ Pierwszy stabilny edge

---

### v15 — full universe + volatility filter
- wszystkie symbole (ok. 99)
- filtr zmienności (ADR >= 4%)

**Wynik:**
- Total PnL: **~6.3%**
- Win rate: ~38%
- Max DD: ~16%

✅ **Najlepszy setup do tej pory**

---

### v16 — microstructure filter (5m)
- dodatkowy filtr lokalnej stabilizacji

**Wynik:**
- Total PnL: ~-10%

❌ overfitting / za agresywne filtrowanie

---

### v17 — 1m entry trigger
- wejście dopiero na 1m reversal:
  - close_strength >= 0.55
  - świeca wzrostowa

**Wynik:**
- Total PnL: ~1.6%
- Win rate: ~31%
- Max DD: ~19%

⚠️ Lepszy timing, ale:
- za dużo złych trade’ów
- problem leży w **setupie, nie entry**

---

## 🔍 Kluczowe wnioski

### 1. Breakout size
- 0.5–1.0% → negatywne
- >1.0% → pozytywne

👉 małe breakouty = noise

---

### 2. Close strength
- silne świece → dobre reversale
- słabe → fake

---

### 3. Entry ≠ główny problem
- 1m poprawia timing
- ale nie poprawia jakości trade’ów

👉 problem = **selekcja setupów**

---

## 🏆 Najlepszy setup (aktualnie)

### 🔥 v15 — Reversal Pullback MTF

**Parametry:**
- 15m:
  - trend <= -5%
  - breakout_attempt: 0.5%–2.5%
- 5m:
  - pullback: 0.3%–2.5%
  - close_strength <= 0.6
- universe:
  - pełny (99 symboli)
  - volatility filter (ADR >= 4%)

**Wynik:**
- Return: ~5–6% / 90 dni (po kosztach)
- Avg win: ~2%
- Avg loss: ~-1%
- Edge: **few big winners**

---

## 🧠 Obecne zrozumienie strategii

To NIE jest strategia:
- dla wszystkich spółek
- dla każdego breakoutu

To jest:
👉 **selective reversal strategy**

Działa tylko gdy:
- jest mocny ruch (breakout)
- spółka jest bardzo zmienna
- pojawia się prawdziwy pullback (nie fake)

---

## 🚀 Następny krok

### v18 — hard filters (w trakcie)

Plan:
- breakout >= 1.0%
- 1m entry strength >= 0.8
- opcjonalnie: ADR >= 6%

Cel:
- mniej trade’ów
- wyższa jakość
- stabilniejszy equity curve

---

## 📌 Uwagi

- testy na pełnym universe są OK
- entry powinno filtrować trade’y (nie universe)
- kluczowe: balans między ilością a jakością sygnałów
