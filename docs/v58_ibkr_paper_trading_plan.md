# V58 IBKR Paper Trading Plan

## Cel

Przechodzimy z fazy backtest/research do fazy:

- IBKR paper trading,
- real-time scanner,
- real order/fill logging,
- pomiaru rzeczywistych kosztow wykonania,
- walidacji strategii na rynku live bez ryzyka realnych pieniedzy.

Strategia pozostaje intraday-only:

- kupno po sygnale momentum / OR breakout,
- sprzedaz tego samego dnia,
- brak overnight positions.

---

## Aktualny status po v56/v57

### Backtest brutto

System ma dodatni edge brutto, ale wynik jest mocno wrazliwy na koszty wykonania.

### v53 portfolio constraints

Symulacja z limitem kapitalu:

- starting cash: $20,000
- max gross exposure: $20,000
- max positions: 8
- accepted trades: 279
- gross profit: ok. $1,666

### v54 execution costs

Po dodaniu konserwatywnych kosztow:

- gross profit: ok. $1,666
- execution costs: ok. $1,622
- net profit: ok. $44

Wniosek:

- edge brutto istnieje,
- ale koszty moga go prawie calkowicie zjesc,
- najwieksza niewiadoma to realne fill quality.

### v56 cost sensitivity

Wynik netto dla roznych modeli kosztow:

| Scenario | Net profit | Return on $20k |
| --- | ---: | ---: |
| ultra optimistic | ~$751 | ~3.75% |
| optimistic | ~$622 | ~3.11% |
| moderate | ~$430 | ~2.15% |
| conservative | ~$44 | ~0.22% |
| extreme | ~-$406 | ~-2.03% |

Wniosek:

- strategia nie wymaga idealnych filli,
- ale jest bardzo wrazliwa na spread/slippage,
- paper trading jest potrzebny, aby zmierzyc realne koszty.

### v57 scaled portfolio

Symulacja dla konta $25k i pozycji $500-$2000:

- trades: 279
- gross profit: ok. $1,595
- execution costs: ok. $1,423
- net profit: ok. $172
- net return on $25k: ok. 0.69%
- net win rate: ok. 54%

Wniosek:

- strategia pozostaje dodatnia przy moderate-cost assumptions,
- ale profit/drawdown nadal nie uzasadnia real-money deployment,
- paper trading jest nastepnym krokiem.

---

## Najwazniejsze wnioski

### 1. PDT nie jest problemem przy kapitale > $25k

Strategia robi day trade'y, wiec na koncie margin w USA obowiazuje PDT rule.

Jesli equity > $25,000, ograniczenie PDT nie powinno blokowac handlu.

### 2. Problemem nie sa same prowizje

Najwiekszy koszt to nie tylko prowizja brokera.

Najwieksze niewiadome:

- spread,
- slippage,
- fill latency,
- partial fills,
- jakosc wykonania na small/mid caps.

### 3. Paper trading jest konieczny

Backtest nie zna realnego order booka.

Paper trading pozwoli zmierzyc:

- expected entry price vs actual fill,
- expected exit price vs actual fill,
- spread at entry/exit,
- latency,
- rejected orders,
- partial fills.

---

## V58 zakres implementacji

### 1. IBKR paper connection

Bot powinien laczyc sie z IB Gateway / TWS paper account.

Domyslne porty:

- IB Gateway paper: 4002
- IB Gateway live: 4001
- TWS paper: 7497
- TWS live: 7496

Na start uzywamy paper only.

---

### 2. Real-time market data

Bot powinien pobierac:

- live 1m bars,
- bid/ask snapshot jezeli dostepne,
- QQQ/SPY/IWM market regime,
- aktualne ceny kandydatow.

---

### 3. Scanner

Scanner powinien:

- budowac opening range,
- wykrywac OR breakout,
- liczyc v45/v50-style setup quality,
- liczyc market regime,
- sortowac kandydatow wedlug rank.

---

### 4. Portfolio manager

Na start bardzo konserwatywne limity paper:

- position size: $100-$250,
- max positions: 3,
- max gross exposure: $1,000,
- max daily loss: $50-$100,
- no overnight positions.

Dopiero po zebraniu danych o fillach zwiekszamy size.

---

### 5. Execution engine

Na poczatek:

- marketable limit orders,
- brak aggressive market orderow poza emergency exit,
- stop loss,
- wide trailing exit,
- end-of-day flatten.

Bot musi umiec:

- wyslac order,
- sprawdzic status,
- zapisac fill,
- anulowac order,
- awaryjnie zamknac pozycje.

---

### 6. Execution logger

Kazdy order powinien zapisac:

- timestamp sygnalu,
- symbol,
- side,
- expected price,
- bid,
- ask,
- order price,
- fill price,
- fill quantity,
- commission jezeli dostepne,
- realized slippage,
- spread estimate,
- reason for entry/exit.

Output:

- CSV na start,
- SQLite pozniej.

---

## Kryteria sukcesu paper tradingu

Paper trading ma odpowiedziec:

1. Czy realny slippage jest blizej optimistic/moderate czy conservative/extreme?
2. Czy spread zabija male trade'y?
3. Czy fill'e sa stabilne?
4. Czy system potrafi zarzadzac pozycjami intraday?
5. Czy daily drawdown jest akceptowalny?
6. Czy strategia ma sens po realnych kosztach?

---

## Decyzja

Status:

- ready for paper trading implementation,
- not ready for real-money live trading,
- no more blind backtest tuning before measuring real execution.

Nastepny etap:

- v58 paper trading bot,
- v59 execution analytics,
- v60 premarket/watchlist automation.
