# 📊 Trading Bot

Automated trading bot for IBKR with strategy ranking, backtesting, paper trading, live execution, portfolio tracking, and notifications.

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
