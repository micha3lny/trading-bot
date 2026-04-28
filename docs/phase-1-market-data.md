# Phase 1: Market Data Foundation

## 🎯 Cel
Zbudowanie fundamentu systemu danych rynkowych:
- połączenie z IBKR API
- baza spółek (universe)
- pobieranie danych historycznych
- zapis danych lokalnie
- codzienna aktualizacja

⚠️ Na tym etapie NIE realizujemy:
- składania zleceń
- tradingu
- strategii

---

## 🔌 IBKR API (read-only)

Bot ma:
- połączyć się z IBKR (TWS / Gateway)
- pobierać dane rynkowe
- pobierać dane historyczne świec

Na start:
- tylko tryb paper / demo
- tylko read-only (brak orderów)

---

## 📊 Baza spółek (Universe)

Tabela (SQLite):

- ticker
- exchange
- currency
- sector (opcjonalnie)
- market_cap (opcjonalnie)
- avg_volume (opcjonalnie)
- active (bool)

Na start:
- NASDAQ
- możliwość rozszerzenia

---

## 📈 Dane historyczne

Zakres:

- 1D → min. 3 lata
- 1H → 1–2 lata
- 5m / 15m → kilka miesięcy
- 1m → opcjonalnie (później)

Źródło:
- IBKR API

---

## 💾 Storage

- SQLite → metadata (spółki, konfiguracja)
- Parquet → dane świec

Struktura plików:

```
data/
  ├── universe.db
  └── market_data/
        ├── AAPL_1D.parquet
        ├── AAPL_5m.parquet
```

---

## 🔄 Aktualizacja danych

Codzienny proces:

1. sprawdź ostatnią datę danych
2. pobierz brakujące świece
3. zapisz do Parquet
4. waliduj spójność danych

---

## 🧪 Walidacja

Bot powinien wykrywać:
- brakujące dane
- duplikaty
- luki w czasie

---

## 🚀 Outcome

Po zakończeniu Phase 1 bot potrafi:
- połączyć się z IBKR
- zarządzać listą spółek
- pobierać dane historyczne
- przechowywać dane
- aktualizować dane codziennie

To jest fundament pod:
- strategie
- backtesting
- paper trading
