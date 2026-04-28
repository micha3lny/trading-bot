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

Zawiera ogólną wizję systemu, architekturę i roadmapę.

### 2. Phase 1: Market Data Foundation

Utworzono:

```text
docs/phase-1-market-data.md
```

Dokument opisuje pierwszy etap projektu:

- połączenie z IBKR,
- baza spółek,
- dane historyczne,
- lokalny storage,
- aktualizacja danych.

### 3. IBKR setup

Utworzono:

```text
docs/ibkr-setup.md
```

Ustalony setup:

```text
IB Gateway
Paper Trading
IB API
host: 127.0.0.1
port: 4002
client id: 1
```

Bot na tym etapie nie składa żadnych zleceń.

### 4. Strategie

Utworzono:

```text
docs/strategies.md
```

Zdefiniowane strategie:

1. Momentum Trailing Intraday
2. Momentum Trailing Overnight
3. Swing Trend Momentum

Najważniejsza ustalona zasada dla Momentum Trailing Intraday:

```text
najpierw działa zwykły stop-loss
trailing stop aktywuje się dopiero po osiągnięciu minimalnego zysku
```

Wstępne parametry intraday:

```text
initial_stop_loss_pct = 1.2
trailing_activation_profit_pct = 1.5
trailing_stop_pct = 1.7
force_exit_before_market_close = true
```

---

## Co działa technicznie

### 1. Lokalny test połączenia z IBKR

Plik:

```text
src/ibkr/test_connection.py
```

Cel:

- połączyć się z IB Gateway / TWS,
- pobrać czas serwera,
- odczytać konta,
- rozłączyć się,
- nie składać zleceń.

Test wykonany lokalnie przez użytkownika zakończył się sukcesem.

Wynik przykładowy:

```text
Connecting to IBKR...
Connected: True
Server time: 2026-04-28 10:03:58+00:00
Accounts: ['DUM541958']
Disconnected
```

### 2. Pobranie danych historycznych AAPL

Plik:

```text
src/data/fetch_aapl_history.py
```

Cel:

- pobrać dane historyczne AAPL,
- interwał: 1D,
- zakres: 3 lata,
- zapisać lokalnie do Parquet.

Dane zostały poprawnie pobrane i zapisane lokalnie:

```text
data/market_data/AAPL_1D.parquet
```

### 3. Pobieranie danych dla 30 spółek

Plik:

```text
src/data/fetch_top30.py
```

Cel:

- pobrać dane daily 1D z ostatnich 3 lat dla początkowego universe 30 spółek,
- zapisać lokalnie do Parquet.

Ważna poprawka:

```text
SQ został zastąpiony przez XYZ
```

Powód: Square / Block zmienił ticker na XYZ.

---

## Jak uruchomić projekt lokalnie

### 1. Pobranie repo

```bash
git clone https://github.com/micha3lny/trading-bot.git
cd trading-bot
```

Jeśli repo już istnieje lokalnie:

```bash
git pull
```

### 2. Środowisko Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Konfiguracja `.env`

```bash
cp .env.example .env
```

Domyślne ustawienia:

```env
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
```

### 4. Uruchomienie IB Gateway

Należy uruchomić lokalnie:

```text
IB Gateway
```

Ustawienia:

```text
Trading Mode: Paper Trading
API Type: IB API
Socket port: 4002
Allow connections from localhost only: enabled
Trusted IP: 127.0.0.1
```

### 5. Test połączenia

```bash
python src/ibkr/test_connection.py
```

### 6. Pobranie AAPL 1D / 3 lata

```bash
python src/data/fetch_aapl_history.py
```

### 7. Pobranie top 30 universe 1D / 3 lata

```bash
python src/data/fetch_top30.py
```

---

## Lokalny storage danych

Dane historyczne są zapisywane lokalnie, nie w repozytorium:

```text
data/market_data/*.parquet
```

Tego folderu nie należy commitować do GitHub.

Do repo trafia tylko:

- kod,
- dokumentacja,
- konfiguracja przykładowa,
- testy.

Lokalnie zostają:

- `.env`,
- `data/`,
- cache,
- logi,
- wyniki backtestów.

---

## Aktualne ustalenia o danych

Na tym etapie pobierane są tylko dane:

```text
bar size: 1 day
history: 3 Y
whatToShow: TRADES
useRTH: true
```

Do pierwszego rankingu strategii Momentum Trailing Intraday potrzebujemy rozszerzyć dane o intraday:

```text
1D + intraday, np. 15m / 5m
```

Użytkownik zdecydował, że ranking strategii ma od razu uwzględniać dane dzienne + intraday.

---

## Najbliższy następny krok

Następny etap:

```text
B: ranking oparty o 1D + intraday
```

Czyli należy zbudować pipeline:

1. pobieranie danych 1D,
2. pobieranie danych intraday, np. 15m lub 5m,
3. zapis do Parquet,
4. loader danych,
5. pierwsza wersja rankingu dla Momentum Trailing Intraday.

Ranking musi być per strategia, nie ogólny.

---

## Ważne zasady projektowe

- Nie robimy ogólnego rankingu spółek.
- Ranking jest zawsze częścią konkretnej strategii.
- Strategia składa się z: ranking → entry → exit.
- Bot nie kupuje od razu po rankingu; ranking wybiera kandydatów do obserwacji.
- Entry decyduje, kiedy kupić na podstawie bieżących świec.
- Exit decyduje, kiedy sprzedać.
- W Phase 1 nie składamy zleceń.
- Najpierw backtest, potem paper trading, dopiero potem live trading.
