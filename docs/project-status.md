# Project Status / Start Here

## Aktualny status

Projekt: `trading-bot`

Repozytorium:

```text
https://github.com/micha3lny/trading-bot
```

Cel projektu: bot tradingowy pod IBKR, który docelowo będzie obsługiwał:

- bazę spółek,
- dane historyczne,
- ranking spółek per strategia,
- backtesting,
- paper trading,
- live trading przez IBKR,
- portfolio,
- statystyki,
- powiadomienia.

---

## Co już zostało zrobione

### 1. Dokumentacja główna

Utworzono i uzupełniono:

```text
README.md
```

### 2. Phase 1: Market Data Foundation

```text
docs/phase-1-market-data.md
```

### 3. IBKR setup

```text
docs/ibkr-setup.md
```

### 4. Strategie

```text
docs/strategies.md
```

---

## Historia prac nad Momentum Trailing Intraday (90D intraday)

### Pipeline
- 1D + 15m intraday
- 90 dni historii
- parquet storage
- ranking → entry → exit → portfolio sim

---

## Eksperymenty

### 1. 30 spółek / 30 dni
- winrate ~73%
- return ~1.5%
👉 Wniosek: overfitting

### 2. 30 spółek / 90 dni
- return ~ -0.30%
👉 brak edge

### 3. Trailing tuning
- brak wpływu
👉 NIE jest bottleneck

### 4. Universe 99 spółek
- return ~ +0.65%
- DD ~ -2.7%
👉 poprawa, ale mała

### 5. OR range filter
- return ~ -1%
👉 NIE działa

### 6. Aggressive universe
- return ~ -0.74%
👉 NIE działa

### 7. Luzowanie entry
- więcej tradów
- gorsza jakość
👉 NIE działa

---

## Najlepsza konfiguracja

- Universe: ~99 spółek
- Selection: daily_trend + breakout
- stop: 1%
- trailing: 0.8 / 1.2

Wynik:
- return ~0.6–1%
- DD ~2–3%

---

## GŁÓWNY PROBLEM

👉 brak follow-through po breakout

---

## NEXT STEP

### Follow-through entry

Zamiast:
- wejście na breakout

Zrobić:
1. breakout
2. kolejna świeca potwierdza
3. entry

---

## START NEXT SESSION

```bash
python -m src.strategies.momentum_trailing_intraday.backtest
```