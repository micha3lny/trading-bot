# Project Status / Start Here

## Aktualny status

Projekt: `trading-bot`

Repozytorium:

```text
https://github.com/micha3lny/trading-bot
```

---

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