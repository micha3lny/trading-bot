# Trading Strategies

## Cel dokumentu

Ten dokument opisuje strategie planowane w systemie. Każda strategia składa się z trzech części:

```text
ranking → entry → exit
```

- `ranking` wybiera najlepsze spółki do obserwacji,
- `entry` decyduje, kiedy wejść w pozycję,
- `exit` decyduje, kiedy wyjść z pozycji.

Ranking jest liczony osobno dla każdej strategii.

---
## 🔄 Reversal Pullback MTF (current main research direction)

### Timeframes
- 15m → setup (trend + breakout attempt)
- 5m → pullback detection
- 1m → entry timing (microstructure)

### Strategy flow
1. Identify oversold condition on 15m
2. Detect breakout attempt
3. Wait for controlled pullback on 5m
4. Enter on 1m reversal signal

### Notes
- Strategy is **selective**, not universal
- Works best on high-volatility symbols
- Relies on few large winners

### Status
🚧 In active development (v12–v17 tested)
## 1. Momentum Trailing Intraday

### Cel

Strategia day tradingowa. Bot kupuje i sprzedaje tego samego dnia.

Celem jest znalezienie spółek z silnym momentum, wejście w odpowiednim momencie i prowadzenie pozycji tak długo, jak cena rośnie.

### Ranking

Strategia ocenia spółki w skali `0–100` na podstawie między innymi:

- momentum z ostatnich dni,
- wolumenu względem średniej,
- zmienności,
- gapu na otwarciu,
- trendu dziennego,
- siły względem rynku,
- płynności,
- informacji giełdowych, np. earnings/news — etap późniejszy.

Ranking służy do wyboru top X spółek, które bot będzie obserwował w danym dniu.

### Entry

Bot nie kupuje od razu po rankingu.

Po wyborze top X spółek obserwuje bieżące świece i szuka sygnału wejścia, np.:

- wybicie lokalnego high,
- odbicie od VWAP / średniej,
- rosnący wolumen,
- świeca potwierdzająca momentum,
- brak gwałtownego odwrócenia.

Dokładna logika entry będzie doprecyzowana i testowana w backtestingu / paper tradingu.

### Exit

Strategia używa klasycznego stop-lossu oraz trailing stopu.

Ustalona zasada:

```text
najpierw działa zwykły stop-loss
trailing stop aktywuje się dopiero po osiągnięciu minimalnego zysku
```

Wstępne parametry:

```text
initial_stop_loss_pct = 1.2
trailing_activation_profit_pct = 1.5
trailing_stop_pct = 1.7
force_exit_before_market_close = true
```

Interpretacja:

- jeśli po kupnie cena spadnie o `1.2%`, bot sprzedaje,
- jeśli cena wzrośnie minimum o `1.5%`, aktywuje się trailing stop,
- po aktywacji trailing stop przesuwa się za najwyższą osiągniętą ceną,
- bot sprzedaje, gdy cena spadnie o `1.7%` od maksimum po aktywacji trailing stopu,
- jeśli pozycja nadal jest otwarta pod koniec sesji, bot zamyka ją przed końcem dnia.

### Dodatkowe limity

Planowane parametry bezpieczeństwa:

- maksymalna liczba transakcji na spółkę dziennie,
- cooldown po zamknięciu pozycji,
- maksymalna strata dzienna na spółkę,
- maksymalna strata dzienna całego bota,
- maksymalna liczba jednocześnie otwartych pozycji.

---

## 2. Momentum Trailing Overnight

### Cel

Strategia bliźniacza do Momentum Trailing Intraday, ale bez przymusu zamknięcia pozycji tego samego dnia.

### Różnica względem strategii intraday

```text
force_exit_before_market_close = false
```

Pozycja może przejść overnight.

### Wstępne parametry

```text
initial_stop_loss_pct = 1.5
trailing_activation_profit_pct = 2.0
trailing_stop_pct = 2.5
max_holding_days = 5
```

Parametry będą testowane na danych historycznych.

---

## 3. Swing Trend Momentum

### Cel

Strategia swing tradingowa. Pozycje mogą być trzymane kilka dni lub tygodni.

Strategia ma szukać spółek w stabilnym trendzie wzrostowym.

### Ranking

Wstępne kryteria:

- trend na danych dziennych,
- momentum z kilku tygodni,
- wolumen,
- stabilność trendu,
- siła względem rynku,
- unikanie spółek o zbyt niskiej płynności.

### Entry / Exit

Do doprecyzowania po zbudowaniu fundamentu danych oraz pierwszej strategii intraday.

---

## Kolejność implementacji

1. Momentum Trailing Intraday
2. Backtest Momentum Trailing Intraday
3. Paper trading Momentum Trailing Intraday
4. Momentum Trailing Overnight
5. Swing Trend Momentum
