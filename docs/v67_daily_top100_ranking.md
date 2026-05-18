# v67/v68 Daily Top100 Ranking

## Cel

Codziennie po zebraniu świeżych świec 1m dla pełnego universe budujemy plik Top100 na następną sesję live.

Runtime v67 nadal dostaje zwykły CSV przez `--alpha-rank-csv`. Ten etap nie zmienia startu bota, orderów ani logiki strategii.

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

- recent momentum z ostatniej sesji RTH
- `intraday_high_pct`
- `day_return_pct`, czyli close vs open
- `volume`
- `dollar_volume`
- `range_pct`
- `median_1m_range_bps`
- `gap_pct`, jeśli jest poprzednia sesja
- `multi_day_return_pct`, jeśli są wcześniejsze sesje
- filtry minimum price, bars, volume i dollar volume

Score jest ważony:

- 30% intraday high
- 20% day return
- 20% liquidity / dollar volume
- 10% daily range
- 10% median 1m range
- 5% gap
- 5% multi-day momentum

To jest ranking kandydatów do obserwacji live, nie gwarancja wejścia. Finalne wejście dalej robi strategia v67 na bieżących danych.

## Komenda build

Przykład: ranking sesji z piątku pod plik używany w poniedziałek.

```bash
python -m src.live_trading.ranking.daily_top100_builder \
  --date 2026-05-15 \
  --universe data/universe/v68_final_daytrading_universe.csv \
  --history-dir data/history/universe_1m \
  --output data/universe/daily_top100_2026-05-16.csv \
  --top-n 100
```

Output CSV jest kompatybilny z `--alpha-rank-csv`:

```text
rank,symbol,score,alpha_score,last_close,dollar_volume,day_return_pct,intraday_high_pct,range_pct,volume,gap_pct,median_1m_range_bps,avg_abs_1m_return_bps,multi_day_return_pct,reason,components_json
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

1. Po sesji uruchom history collector dla pełnego universe i sesji RTH.
2. Sprawdź, czy parquet-y są zapisane w `data/history/universe_1m`.
3. Uruchom daily Top100 builder dla ostatniej kompletnej sesji.
4. Zweryfikuj top 20.
5. Przy starcie bota podaj wygenerowany CSV przez `--alpha-rank-csv`.

Inspekcja top 20:

```bash
python -c 'import pandas as pd; df=pd.read_csv("data/universe/daily_top100_2026-05-16.csv"); print(df.head(20).to_string(index=False))'
```

## Safety notes

- Builder nie łączy się z IBKR.
- Brak danych dla pojedynczego symbolu jest logowany jako `DAILY_TOP100_MISSING_DATA` i nie przerywa całego runu.
- Jeśli valid symboli jest mniej niż `--top-n`, builder zapisuje mniej wierszy i loguje warning.
- Ten etap nie zmienia `v67_live_top100_expansion_paper_trader.py`.
- Ten etap nie zmienia order lifecycle, EOD flatten ani control API.

