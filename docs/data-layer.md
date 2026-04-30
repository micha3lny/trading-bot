# Data Layer / Market Data Architecture

## Cel

Ten dokument opisuje aktualną warstwę danych używaną w projekcie `trading-bot`.

Warstwa danych jest teraz przygotowana pod strategie typu **MTF** (*multi-timeframe*), czyli takie, które używają różnych interwałów świec do różnych decyzji tradingowych.

---

## Dostępne interwały danych

Aktualnie projekt obsługuje i lokalnie zapisuje dane dla następujących interwałów:

```text
1D   - dane dzienne
15m  - świece intraday 15-minutowe
5m   - świece intraday 5-minutowe
1m   - świece intraday 1-minutowe
```

---

## Struktura plików

### Dane dzienne

```text
data/market_data/{SYMBOL}_1D.parquet
```

Przykład:

```text
data/market_data/AAPL_1D.parquet
```

### Dane intraday

```text
data/market_data_intraday/{SYMBOL}_{INTERVAL}.parquet
```

Przykłady:

```text
data/market_data_intraday/AAPL_15m.parquet
data/market_data_intraday/AAPL_5m.parquet
data/market_data_intraday/AAPL_1m.parquet
```

---

## Aktualne pokrycie danych

Na obecnym etapie pobrano dane dla pełnego universe około 99 symboli.

Dostępne są:

```text
1D + 15m + 5m + 1m
```

Dane intraday obejmują około:

```text
90 dni historii
```

---

## Jak używamy interwałów w strategiach

Aktualny główny kierunek researchu to strategie typu:

```text
Reversal Pullback MTF
```

### 1D

Używane do:

- trendu dziennego,
- średnich kroczących,
- dziennej zmienności,
- filtrowania symboli po zmienności.

Przykładowe użycie:

```text
trend <= -5%
avg_daily_range >= 4%
```

---

### 15m

Używane do wykrywania setupu.

Rola:

```text
15m = kontekst / setup
```

Przykładowo:

- oversold context,
- opening range,
- breakout attempt,
- confirmation candle.

---

### 5m

Używane do wykrywania pullbacku.

Rola:

```text
5m = struktura cofki
```

Przykładowo:

- pullback po breakout attempt,
- close_strength świecy cofki,
- risk od opening range low,
- kontrola czy cena nie rozpada się zbyt głęboko pod OR high.

---

### 1m

Używane do precyzyjnego wejścia.

Rola:

```text
1m = execution / trigger wejścia
```

Wprowadzone po testach v17.

Celem 1m nie jest generowanie setupu, tylko poprawa timingu wejścia po tym, jak setup został już wykryty na 15m i pullback na 5m.

Przykładowe warunki testowane w v17:

```text
1m close_strength >= 0.55
1m close > previous close
```

---

## Aktualny model MTF

Docelowy flow strategii wygląda obecnie tak:

```text
1D   -> filtr kontekstu i zmienności
15m  -> setup
5m   -> pullback
1m   -> entry trigger
exit -> trailing / session close / testowane warianty
```

---

## Ważne wnioski z testów

### 1. 15m nie wystarcza do entry

Testy v6/v7 pokazały, że pullback entry na samych świecach 15m jest zbyt grube i często nie znajduje sygnałów.

Wniosek:

```text
15m nadaje się do setupu, ale nie do precyzyjnego wejścia.
```

---

### 2. 5m poprawia strukturę wejścia

Dodanie 5m umożliwiło testy pullback entry i doprowadziło do pierwszych dodatnich wyników w v12/v15.

Wniosek:

```text
5m jest dobrym interwałem do wykrycia cofki.
```

---

### 3. 1m poprawia timing, ale nie rozwiązuje selekcji setupów

Test v17 pokazał, że 1m może poprawić precyzję wejścia, ale nie wystarcza do naprawy słabej selekcji setupów.

Wniosek:

```text
1m powinno być używane jako trigger, nie jako główny filtr strategii.
```

---

## Aktualne skrypty pobierania danych

### 15m / 5m

```bash
python -m src.data.fetch_top30_intraday
python -m src.data.fetch_top30_intraday_5m
```

### 1m

Ze względu na limity i timeouty IBKR, dane 1m powinny być pobierane wersją chunked:

```bash
python -m src.data.fetch_top30_intraday_1m_chunked
```

Ten skrypt pobiera 1m dane kawałkami, aby uniknąć timeoutów na dużym zapytaniu 90D.

---

## Aktualny status

Warstwa danych jest gotowa do dalszych testów MTF:

```text
1D + 15m + 5m + 1m
```

Główny problem nie leży już w braku danych, tylko w:

```text
selekcji setupów / filtrowaniu jakości sygnałów
```

---

## Następne kroki

Najbliższy kierunek researchu:

```text
v18 - hard filters / quality setup selection
```

Planowane testy:

- breakout >= 1.0%,
- 1m close_strength >= 0.8,
- opcjonalnie volatility filter >= 6%,
- dalsza selekcja setupów przed wejściem.
