# 📊 Trading Bot

Automated trading bot for IBKR with strategy ranking, backtesting, paper trading, live execution, portfolio tracking, analytics, and operational controls.

---

## ✅ Aktualny status — 2026-05-13

Projekt przeszedł z etapu prostego paper tradingu do etapu:

```text
production-style paper trading infrastructure
```

System działa już jako:
- live execution node na Raspberry Pi,
- z pełnym IBKR Gateway,
- reconnect watchdogiem,
- recorderem,
- analytics,
- telemetry,
- emergency control API.

---

## 🚀 Co działa i zostało potwierdzone

### Infrastructure

- Raspberry Pi działa jako execution node.
- IBKR Gateway działa przez GUI/VNC.
- API działa na porcie `4002`.
- Repo synchronizowane przez GitHub.
- System działa w `systemd`.
- Autorestart bota działa.
- Reconnect watchdog działa.
- Backfill po reconnect działa.

---

### Universe / Market Data

- Universe rozszerzony do ~2184 symboli.
- TOP100 wybierane dynamicznie.
- Działa live recorder.
- Działa agregacja świec 1m.
- Działa current-session backfill.
- Działa universe-wide 1m backfill.
- Zapisywane są candles dla pełnego universe.

Recorder zapisuje:

- `candles_1m.csv`
- `trade_lifecycle.csv`
- `fills.csv`
- `strategy_equity.csv`
- `portfolio_snapshots.csv`
- `run_metadata.csv`

---

### Strategy / Trading

Działa:

- `v67_live_top100_expansion_paper_trader`
- momentum expansion ranking
- breakout entries
- trailing exits
- EOD flatten support
- restart recovery
- managed positions restore
- multiple entries per symbol/day

---

### Strict setup tracking

Najlepszy historyczny setup (~600 USD netto) został zachowany jako osobny tag telemetryczny.

Strict thresholds:

- 5m >= 4.0%
- 15m >= 6.5%
- OR >= 5.0%
- spread <= 50bps

Każdy trade posiada:

- `strict_setup=True`
- `strict_setup=False`

Pozwala to:

- eksperymentować z poluzowanymi parametrami,
- ale nadal mierzyć performance oryginalnego setupu.

---

### Analytics

Działa:

- daily report
- strict setup analytics
- peak / giveback analytics
- pnl bucket analytics
- missed runners report
- runner telemetry
- MFE tracking
- entry timing analytics
- open/unrealized pnl

Raporty:

```bash
python -m src.live_trading.analytics.v67_daily_report --date YYYY-MM-DD
```

```bash
python -m src.live_trading.analytics.v67_missed_runners_report --date YYYY-MM-DD
```

---

### Missed runners analytics

System analizuje wszystkie spółki, które zrobiły:

```text
+10% intraday move
```

Dla każdej spółki raport pokazuje:

- open price
- high price
- high timestamp
- whether bought
- buy timestamp
- buy price
- sell price
- pnl
- first 5m expansion
- first 15m expansion
- OR expansion
- rejection reason

To pozwala analizować:

- dlaczego runner został pominięty,
- czy kupujemy za późno,
- czy kupujemy near peak.

---

### Control API (NEW)

Bot posiada lokalne operational API.

Endpoints:

```bash
curl http://127.0.0.1:8767/health
```

```bash
curl -X POST http://127.0.0.1:8767/flatten_all_positions
```

```bash
curl -X POST "http://127.0.0.1:8767/flatten_symbol?symbol=QUBT"
```

```bash
curl -X POST http://127.0.0.1:8767/pause_entries
```

```bash
curl -X POST http://127.0.0.1:8767/resume_entries
```

Cel:

- emergency flatten
- lifecycle consistency
- operational safety
- avoiding ghost positions
- safer restart handling

---

## ⚠️ Aktualne problemy / TODO

### HIGH PRIORITY

#### 1. Trade reconciliation engine

Potrzebny jest:

- IBKR executions reconciliation
- portfolio reconciliation
- ghost trade cleanup
- restart reconciliation

Status:

```text
NOT DONE
```

---

#### 2. Reliable EOD flatten

Potrzebne:

- retry logic
- verification pass
- stuck order handling
- guaranteed flatten before close

Status:

```text
PARTIALLY DONE
```

---

#### 3. Lifecycle recorder migration

Obecnie:

```text
trade_lifecycle.csv
```

Problem:

embedded JSON potrafi uszkodzić CSV parsing.

Planned:

```text
JSONL structured recorder
```

---

#### 4. Restart safety

Potrzebne:

- restart cooldown
- stale order cleanup
- duplicate entry prevention
- stronger adoption logic

---

## 🎯 Aktualny cel projektu

Przejście z:

```text
paper trading prototype
```

na:

```text
production-grade live trading infrastructure
```

przed przejściem na:

```text
real IBKR account
```

---

## 📌 Najważniejsze wnioski po pierwszych live sesjach

### Strategicznie

- strict setup wygląda bardzo obiecująco,
- część relaxed entries wygląda jak buying near peak,
- peak/giveback analytics okazały się bardzo wartościowe,
- missed runner telemetry daje dużo insightów.

---

### Technicznie

Największe ryzyka nie są już strategiczne.

Największe ryzyka:

- restart consistency
- lifecycle consistency
- reconciliation
- IBKR gateway edge cases
- EOD reliability

---

## 🧠 Architektura logiczna

```text
Universe (2184 symbols)
    ↓
1m candles
    ↓
TOP100 ranking
    ↓
Feature engine
    ↓
Expansion filters
    ↓
Entry signals
    ↓
IBKR paper execution
    ↓
Managed positions
    ↓
Trailing exits
    ↓
Recorder / analytics
    ↓
Daily reports
```

---

## 🚀 Long-term roadmap

Planned:

- reconciliation engine
- JSONL recorder
- web dashboard
- Discord / Telegram alerts
- Prometheus/Grafana monitoring
- dynamic position sizing
- volatility regime filters
- ML ranking layer
- multi-strategy framework
- live account rollout
