# IBKR Setup

## Cel
Konfiguracja połączenia bota z Interactive Brokers przez IB Gateway / TWS.

Na start używamy wyłącznie:
- Paper Trading
- IB API
- trybu read-only dla danych rynkowych i historycznych

Na tym etapie bot NIE składa żadnych zleceń.

---

## Preferowane narzędzie

Do pracy bota preferowany jest:

```text
IB Gateway
```

TWS może służyć pomocniczo do ręcznej kontroli, podglądu rachunku i debugowania.

---

## Tryb

Na start:

```text
Trading Mode: Paper Trading
API Type: IB API
```

Nie używamy Live Trading do czasu przejścia etapów:

```text
backtest → paper trading → live trading
```

---

## Porty IBKR

Domyślne porty:

```text
TWS Paper:       7497
TWS Live:        7496
IB Gateway Paper: 4002
IB Gateway Live:  4001
```

Na start używamy:

```text
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
IB_TRADING_MODE=paper
```

---

## Zmienne środowiskowe

Planowane zmienne:

```env
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
IB_TRADING_MODE=paper
```

Danych logowania do IBKR nie zapisujemy w repozytorium.

---

## Wymagania w IB Gateway / TWS

Należy upewnić się, że:

- Gateway/TWS jest uruchomiony,
- zalogowano się do Paper Trading,
- wybrano API Type = IB API,
- API jest włączone w ustawieniach,
- port zgadza się z konfiguracją bota,
- localhost / 127.0.0.1 jest dozwolony.

---

## Pierwszy test połączenia

Pierwszy moduł bota powinien tylko:

1. połączyć się z IBKR,
2. odczytać informację o kontach,
3. odczytać czas serwera,
4. rozłączyć się,
5. nie składać żadnych zleceń.

---

## Zasady bezpieczeństwa

- Żadnych loginów ani haseł w repozytorium.
- Żadnych zleceń w Phase 1.
- Domyślny tryb to paper.
- Live trading będzie wymagał osobnej, jawnej konfiguracji.
