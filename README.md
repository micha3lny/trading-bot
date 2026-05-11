# 📊 Trading Bot

Automated trading bot for IBKR with strategy ranking, backtesting, paper trading, live execution, portfolio tracking, and notifications.

---

## ✅ Aktualny status — 2026-05-11

Projekt przeszedł z etapu backtestów do pierwszego realnego uruchomienia na Raspberry Pi z IBKR Gateway i paper account.

### Uruchomione i potwierdzone

- Raspberry Pi działa jako execution node.
- IB Gateway działa przez GUI/VNC i udostępnia API na porcie `4002`.
- Repo działa na Macu i Raspberry przez GitHub.
- Universe został rozszerzony do `2184` symboli w `data/universe/v62_symbols_wide.txt`.
- Na Raspberry trzymamy tylko pliki potrzebne do live execution, a pełną historię świec trzymamy na Macu.
- Działa live recorder zapisujący dane do `data/live/recorder/<YYYY-MM-DD>/`.
- Działa monitor tekstowy portfolio: `src.live_trading.v63_live_portfolio_monitor`.
- Działa account recorder: `src.live_trading.v66_ibkr_account_recorder`.
- Działa paper trader: `src.live_trading.v67_live_top100_expansion_paper_trader`.

### Najlepszy aktualny setup strategiczny

Najlepszy potwierdzony setup z ostatnich backtestów:

```bash
python -m src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v59_daily_top_universe \
  --top-n 100 \
  --apply-live-safe-expansion
```

Wynik testowy dla tego wariantu:

- około `+614.72 USD` netto,
- `26` transakcji,
- `76.92%` win rate,
- testowany jako live-safe TOP100 + expansion.

### Aktualny live/paper stack

```text
IB Gateway / Paper Account
    ↓
v67 live top100 expansion paper trader
    ↓
TOP100 z alpha rankingu
    ↓
live snapshots: bid / ask / last / spread / volume
    ↓
feature engine: first 5m, first 15m, OR range, spread, price
    ↓
signal / order intent / paper execution
    ↓
recorder CSV
    ↓
portfolio monitor
```

### Obecny tryb działania v67

`v67_live_top100_expansion_paper_trader.py`:

- subskrybuje TOP100 spółek z `data/universe/v64_universe_alpha_ranked.csv`,
- liczy live-safe expansion,
- zapisuje `market_snapshots.csv`, `spread_snapshots.csv`, `selection_events.csv`, `signal_snapshots.csv`, `order_intents.csv`,
- po spełnieniu warunków wysyła paper BUY przez IBKR `MarketOrder`,
- działa z poziomu `tmux`,
- logowany jest do `data/live/recorder/<date>/v67_trader.log`.

Uruchomienie na Raspberry:

```bash
cd ~/trading-bot
source venv/bin/activate

python -m src.live_trading.v67_live_top100_expansion_paper_trader \
  --host 127.0.0.1 \
  --port 4002 \
  --client-id 67 \
  --top-n 100 \
  --duration-seconds 28800 \
  2>&1 | tee -a data/live/recorder/$(date -u +%F)/v67_trader.log
```

### Ważne ograniczenia / TODO

- Domyślne logi v67 są jeszcze zbyt ubogie; potrzebujemy logować `best_score`, `best_symbol`, TOP kandydatów i powody odrzuceń.
- Trzeba dodać live agregację świec 1m do `candles_1m.csv`.
- Trzeba potwierdzić pełną ścieżkę: sygnał → order intent → paper fill → portfolio snapshot.
- Trzeba dopiąć exit management i zweryfikować SELL w paper tradingu.
- Trzeba dodać autostart po reboot: Gateway → wait for API → bot → portfolio recorder.
- Trzeba dodać backfill po restarcie, żeby uzupełniać brakujące świece.
- Trzeba dodać automatyczny sync danych live z Raspberry na Maca.

---

## 🎯 Cel projektu

System do automatycznego i półautomatycznego handlu akcjami przez IBKR, który:

- analizuje rynek, początkowo NASDAQ, później również inne giełdy,
- buduje ranking spółek osobno dla każdej strategii,
- wybiera najlepsze kandydaty do tradingu,
- podejmuje decyzje o kupnie i sprzedaży,
- działa w trybie manualnym albo automatycznym,
- przechodzi przez etapy: backtest → paper trading → live trading.

---

## 🧱 Architektura logiczna

```text
Market Data
↓
Universe / baza spółek
↓
Strategie
↓
Ranking 0–100 per strategia
↓
Tryb manual / auto
↓
Entry / Exit
↓
Risk Management
↓
Order Manager
↓
Broker API / IBKR
↓
Portfolio
↓
Statystyki + Powiadomienia
```

---

## 📊 Baza spółek / Universe

Baza spółek będzie zawierać między innymi:

- ticker,
- giełdę,
- walutę,
- sektor,
- market cap,
- średni wolumen,
- status aktywności,
- informację, czy spółka jest dopuszczona do tradingu.

Na start zakładamy NASDAQ, ale system ma być rozszerzalny o inne giełdy.

---

## 📈 Dane historyczne

System będzie przechowywał dane historyczne potrzebne do analizy, rankingu i backtestów.

Wstępne założenia:

- `1D` — minimum 3 lata,
- `1H` — 1–2 lata,
- `15m / 5m` — kilka miesięcy,
- `1m` — dla day tradingu.

Dane mają być:

- zasilone początkowo,
- aktualizowane codziennie,
- wykorzystywane zarówno do rankingu, jak i do backtestów.

---

## 🧠 Strategie

System ma obsługiwać wiele strategii w jednym frameworku.

Na start planowane są dwie strategie:

1. Day Trading Strategy,
2. Swing Trading Strategy.

Każda strategia zawiera:

- ranking kandydatów,
- logikę wejścia — kiedy kupić,
- logikę wyjścia — kiedy sprzedać,
- parametry strategii,
- zasady zarządzania pozycją.

---

## 🏆 Ranking

Ranking jest liczony osobno dla każdej strategii.

Przykład:

```text
Day Trading Strategy:
AAPL → 91
NVDA → 88
TSLA → 82

Swing Trading Strategy:
MSFT → 89
AMD  → 84
META → 81
```

Założenia:

- skala od 0 do 100,
- ranking jest częścią strategii,
- ranking korzysta z danych historycznych i wskaźników,
- po rankingu wybierane jest top X spółek do obserwacji albo tradingu.

---

## 🎮 Tryby działania

### AUTO

Bot:

- generuje ranking,
- wybiera top X spółek,
- obserwuje rynek,
- sam podejmuje decyzje o wejściu,
- sam zarządza wyjściem według strategii.

### MANUAL

Bot:

- generuje ranking,
- pokazuje propozycję spółek,
- czeka na zatwierdzenie użytkownika,
- po zatwierdzeniu handluje tylko zatwierdzonymi spółkami.

Manual oznacza zgodę na handel wybranymi spółkami, a nie ręczne klikanie każdej transakcji.

---

## ⚡ Day Trading

Założenia:

- codzienny ranking,
- wybór top X spółek,
- bot może otworzyć pozycję na tej samej spółce kilka razy w ciągu dnia, jeśli warunki strategii są spełnione,
- bot decyduje, kiedy wejść w pozycję,
- sprzedaż odbywa się według algorytmu strategii.

Przykładowe limity:

- maksymalna liczba transakcji na spółkę dziennie,
- cooldown po zamknięciu pozycji,
- maksymalna strata dzienna na spółkę,
- maksymalna strata dzienna całego bota.

---

## 📉 Swing Trading

Założenia:

- pozycje mogą być trzymane kilka dni albo tygodni,
- strategia będzie oparta głównie o dane dzienne,
- liczba transakcji będzie mniejsza niż w day tradingu,
- strategia ma służyć jako stabilniejszy i wolniejszy model handlu.

---

## 🧪 Backtesting

Backtesting sprawdza strategię na danych historycznych.

Ma odpowiadać na pytania:

- czy strategia zarabiała w przeszłości,
- jaki był zysk lub strata,
- jaki był winrate,
- jaki był maksymalny drawdown,
- ile było transakcji,
- jakie były najlepsze i najgorsze okresy.

Backtesting jest wymagany przed paper tradingiem i live tradingiem.

---

## 🧾 Paper Trading

Paper trading to test strategii na aktualnym rynku bez użycia prawdziwych pieniędzy.

Bot:

- działa na realnych danych,
- generuje realne sygnały,
- symuluje kupno i sprzedaż,
- prowadzi wirtualny portfel,
- liczy wyniki strategii.

Paper trading jest etapem po backteście i przed live tradingiem.

---

## 💰 Live Trading / IBKR

Po przejściu backtestu i paper tradingu strategia może zostać aktywowana na realnym rachunku IBKR.

System ma korzystać z API Interactive Brokers.

---

## ⚠️ Risk Management

Risk Management jest nadrzędnym modułem bezpieczeństwa.

Przykładowe zasady:

- maksymalna kwota na jedną pozycję,
- maksymalna liczba otwartych pozycji,
- stop-loss,
- take-profit,
- trailing stop,
- maksymalna strata dzienna,
- blokada handlu po serii strat,
- limit ekspozycji na jedną spółkę.

Strategia nie powinna móc ominąć risk managera.

---

## 📦 Portfolio Manager

Portfolio Manager śledzi:

- otwarte pozycje,
- średnią cenę zakupu,
- ilość akcji,
- P/L,
- dostępną gotówkę,
- historię transakcji,
- wynik per strategia.

---

## 📬 Powiadomienia

Na start planowane są powiadomienia email.

Docelowo możliwe kanały:

- Telegram,
- Slack,
- panel webowy.

Powiadomienia mają obejmować:

- wygenerowany ranking,
- prośbę o zatwierdzenie w trybie manualnym,
- kupno,
- sprzedaż,
- wynik transakcji,
- błędy,
- alerty bezpieczeństwa.

---

## 📊 Statystyki

System ma raportować:

- liczbę transakcji,
- transakcje zyskowne i stratne,
- winrate,
- zysk/stratę,
- wynik per strategia,
- wynik per spółka,
- maksymalny drawdown,
- najlepsze i najgorsze transakcje.

---

## 🔁 Dzienny workflow

```text
1. Aktualizacja danych
2. Generowanie rankingu per strategia
3. Wybór top X spółek
4. Tryb AUTO albo MANUAL
5. Obserwacja rynku
6. Wejście w pozycję, gdy warunki są spełnione
7. Zarządzanie wyjściem według strategii
8. Aktualizacja portfolio
9. Zapis statystyk
10. Wysłanie powiadomień
```

---

## 🚀 Roadmap

1. Dokumentacja i założenia projektu,
2. baza spółek,
3. moduł danych historycznych,
4. pierwsza strategia day tradingowa,
5. backtesting,
6. paper trading,
7. powiadomienia,
8. integracja z IBKR,
9. live trading,
10. strategia swing tradingowa,
11. panel webowy / Telegram / Slack.

---

## 📌 Kluczowe decyzje projektowe

- Jeden system obsługujący wiele strategii.
- Ranking osobny dla każdej strategii.
- Tryb manual/auto po wygenerowaniu rankingu.
- Day trading i swing trading w tym samym frameworku.
- Bot decyduje o wejściu w pozycję.
- Wyjście z pozycji jest zarządzane przez algorytm strategii.
- Możliwość wielu wejść w tę samą spółkę jednego dnia.
- Risk manager jest obowiązkowy i nadrzędny wobec strategii.
