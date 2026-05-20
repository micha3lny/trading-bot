# v67/v68 Daily Top100 Ranking

## Cel

Codziennie po zebraniu świeżych świec 1m dla pełnego universe budujemy plik Top100 na następną sesję live.

Runtime v67 nadal dostaje zwykły CSV przez `--alpha-rank-csv`. Ten etap nie zmienia startu bota, orderów ani logiki strategii.

Stabilny plik dla systemd/live startup:

```text
data/universe/daily_top100_latest.csv
```

## Co robi obecny runtime

`src/live_trading/v67_live_top100_expansion_paper_trader.py` ładuje symbole przez `load_top_symbols(...)`.

Reguły są proste:

- CSV musi mieć kolumnę `symbol`.
- Jeśli istnieje `alpha_score`, runtime sortuje malejąco po `alpha_score`.
- Jeśli istnieje `last_close`, runtime może odfiltrować symbole poniżej `--min-price`.
- Potem bierze pierwsze `--top-n` unikalnych symboli.

Dlatego nowy daily builder zapisuje zarówno `score`, jak i kompatybilny alias `alpha_score`.

## Co robił v64 alpha ranker

`src/data/v64_universe_alpha_ranker.py` czytał historyczne CSV z `data/1m` i liczył ranking z kolumn OHLCV:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Główne metryki:

- `last_close`
- `median_dollar_volume`
- `avg_dollar_volume`
- `median_1m_range_bps`
- `p90_1m_range_bps`
- `avg_abs_1m_return_bps`
- `momentum_day_frequency`
- `expansion_bar_frequency`
- `positive_followthrough_frequency`
- score składowe: completeness, liquidity, volatility, momentum, followthrough

To był ranking bardziej historyczny i szeroki. Daily Top100 jest krótszym pre-market rankingiem z ostatniej zwalidowanej sesji RTH.

## Status v68 universe

Produkcyny plik universe to:

```text
data/universe/v68_final_daytrading_universe.csv
```

Lokalny repozytorium zwykle nie zawiera `data/`, więc nie da się z kodu potwierdzić, czy ten konkretny plik jest posortowany po liquidity, starym alpha score, czy ręcznej kolejności. Obecny loader sortuje go po `alpha_score` tylko wtedy, gdy ta kolumna istnieje. W przeciwnym razie kolejność pliku jest kolejnością runtime.

Daily builder usuwa tę niejednoznaczność, bo zawsze zapisuje `rank`, `score` i `alpha_score`.

## Dane wejściowe

Collector v68 zapisuje pliki:

```text
data/history/universe_1m/session_type=RTH/symbol=SYMBOL/year=YYYY/month=MM/day=DD.parquet
```

Wymagane kolumny:

- `bar_time_utc`
- `open`
- `high`
- `low`
- `close`
- `volume`

Opcjonalnie używane:

- `wap`

## Ranking features

Daily builder używa tylko lokalnych parquetów, bez IBKR:

Top100 jest rankingiem prawdopodobieństwa runnera, nie rankingiem największej płynności. Celem jest znaleźć czyste small/mid caps z wysoką szansą na tradowalny intraday expansion move, typowo +5% do +10%+ od open/high, a nie zdominować listę przez SPY/QQQ/NVDA/MSFT.

Ranking faworyzuje:

- recent momentum z ostatniej sesji RTH
- `intraday_high_pct`
- `close_open_pct`, czyli close vs open
- `range_pct`
- `median_1m_range_bps`
- `avg_abs_1m_return_bps`
- `gap_pct`, jeśli jest poprzednia sesja
- `multi_day_return_pct`, jeśli są wcześniejsze sesje
- wysoką kompletność danych
- filtry minimum price, bars, volume i dollar volume

Liquidity działa głównie jako gate bezpieczeństwa. Symbol musi mieć wystarczający `volume` i `dollar_volume`, ale po przejściu filtra liquidity ma tylko lekką wagę w score. Dollar volume jest log-scaled i capowany, więc mega caps nie dostają automatycznej przewagi tylko dlatego, że obracają miliardami.

Final score jest ważony:

- 45% `momentum_score`
- 25% `range_pct`
- 15% volatility score z median/average 1m movement
- 5% `close_open_pct`
- 5% data completeness
- 5% `liquidity_score`

To jest ranking kandydatów do obserwacji live, nie gwarancja wejścia. Finalne wejście dalej robi strategia v67 na bieżących danych.

Builder odrzuca oczywiste junk suffixes:

- warrants
- units
- rights
- preferred/special suffix

## Komenda build

Przykład: ranking sesji z piątku pod plik używany w poniedziałek.

```bash
python -m src.live_trading.ranking.daily_top100_builder \
  --date 2026-05-15 \
  --universe data/universe/v68_final_daytrading_universe.csv \
  --history-dir data/history/universe_1m \
  --output data/universe/daily_top100_2026-05-16.csv \
  --latest-output data/universe/daily_top100_latest.csv \
  --diagnostics-output data/universe/daily_top100_2026-05-16_diagnostics.csv \
  --top-n 100
```

Builder zawsze zapisuje dated output. `daily_top100_latest.csv` jest aktualizowany atomowo tylko wtedy, gdy build zakończy się poprawnie i wynik ma minimum 100 wierszy. Jeśli wynik ma mniej niż 100 wierszy, stary latest zostaje nietknięty, a proces kończy się kodem niezerowym.

Przy dużych brakach danych builder nie spamuje pełną listą symboli w logu. Domyślnie pokazuje pierwsze 50 missing/rejected i zapisuje pełny raport diagnostyczny:

```text
data/universe/daily_top100_YYYY-MM-DD_diagnostics.csv
```

Raport ma kolumny:

```text
date,symbol,status,reason
```

Output CSV jest kompatybilny z `--alpha-rank-csv`:

```text
rank,symbol,score,alpha_score,final_score,momentum_score,liquidity_score,last_close,dollar_volume,day_return_pct,close_open_pct,intraday_high_pct,range_pct,volume,gap_pct,median_1m_range_bps,avg_abs_1m_return_bps,multi_day_return_pct,reason,components_json
```

## SQLite audit store

Domyślnie builder zapisuje też snapshot do:

```text
data/runtime/rankings.sqlite
```

Tabela:

```text
daily_rankings(date, rank, symbol, score, components_json, created_at)
```

CSV zostaje artefaktem runtime. SQLite jest tylko audytem i wygodnym miejscem do porównań rankingów między dniami.

Można wyłączyć SQLite:

```bash
python -m src.live_trading.ranking.daily_top100_builder ... --no-sqlite
```

## Pre-market flow

Collector i ranking są osobnymi logicznie krokami. Runtime v67 może je teraz automatycznie odpalać poza RTH.

1. Po sesji, w weekend albo poza godzinami handlu uruchom history collector dla pełnego 2463-symbolowego universe i sesji RTH.
2. Sprawdź, czy parquet-y są zapisane w `data/history/universe_1m`.
3. Premarket uruchom daily Top100 builder dla ostatniej kompletnej sesji albo pozwól runtime zrobić to automatycznie o `12:45 UTC`.
4. Builder zapisuje dated CSV oraz, jeśli wynik jest valid, atomowo aktualizuje `data/universe/daily_top100_latest.csv`.
5. Live bot czyta stabilny latest przez `--alpha-rank-csv data/universe/daily_top100_latest.csv`. To jest teraz domyślna ścieżka runtime.

Domyślna automatyzacja w v67:

```text
20:15 UTC overnight collector
23:00 UTC overnight collector
03:00 UTC overnight collector
07:00 UTC overnight collector
12:45 UTC daily Top100 build
```

Collector jest inkrementalny: pomija kompletne parquet days i pobiera tylko brakujące albo niekompletne symbol-days. `daily_top100_latest.csv` jest aktualizowany tylko po valid buildzie z minimum 100 wierszami; jeśli build wyprodukuje mniej, poprzedni latest zostaje na miejscu.

Oczekiwane logi automatyzacji:

```text
OVERNIGHT_COLLECTOR_START
OVERNIGHT_COLLECTOR_DONE
OVERNIGHT_COLLECTOR_SKIPPED
DAILY_TOP100_BUILD_START
DAILY_TOP100_BUILD_DONE
DAILY_TOP100_BUILD_FAILED
```

Wrapper premarket:

```bash
scripts/build_daily_top100_premarket.sh 2026-05-15
```

Bez argumentu wrapper wybiera poprzedni dzień roboczy względem daty systemowej.

Inspekcja top 20:

```bash
python -c 'import pandas as pd; df=pd.read_csv("data/universe/daily_top100_latest.csv"); print(df.head(20).to_string(index=False))'
```

Przykład konfiguracji live bota:

```bash
python -u -m src.live_trading.v67_live_top100_expansion_paper_trader \
  --host 127.0.0.1 \
  --port 4002 \
  --client-id 67 \
  --alpha-rank-csv data/universe/daily_top100_latest.csv \
  --top-n 100
```

## Safety notes

- Builder nie łączy się z IBKR.
- Brak danych dla pojedynczego symbolu jest logowany jako `DAILY_TOP100_MISSING_DATA` i nie przerywa całego runu.
- Pełna lista braków jest zapisywana w diagnostics CSV, żeby było wiadomo co dociągnąć collectorem.
- Jeśli valid symboli jest mniej niż `--top-n`, builder zapisuje mniej wierszy i loguje warning.
- `daily_top100_latest.csv` nie jest aktualizowany, jeśli output ma mniej niż 100 wierszy.
- Runtime default `--alpha-rank-csv` wskazuje na `data/universe/daily_top100_latest.csv`.
