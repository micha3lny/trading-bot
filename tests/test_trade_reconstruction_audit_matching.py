import pandas as pd

from scripts.audit_trade_reconstruction import match_flex_rows_to_sqlite


def flex_row(symbol, side, qty, price, trade_id):
    return {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "price": price,
        "time": "2026-07-15T00:00:00+00:00",
        "execution_id": trade_id,
    }


def sqlite_row(symbol, side, qty, price, execution_id):
    return {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "price": price,
        "session_date": "2026-07-15",
        "executed_at": "2026-07-15T13:36:15Z",
        "execution_id": execution_id,
    }


def run_match(flex_rows, sqlite_rows):
    return match_flex_rows_to_sqlite(pd.DataFrame(flex_rows), pd.DataFrame(sqlite_rows))


def test_aaoi_exact_one_to_one_without_matching_trade_id():
    _assignments, sqlite_to_flex, statuses = run_match(
        [flex_row("AAOI", "BUY", 8, 119.56, "FLEX-AAOI-1")],
        [sqlite_row("AAOI", "BUY", 8, 119.56, "00025b45.aaoi")],
    )

    assert statuses == {0: "exact_match"}
    assert list(sqlite_to_flex) == ["00025b45.aaoi"]


def test_abtc_one_flex_sell_matches_split_sqlite_sells_by_aggregate_quantity():
    _assignments, sqlite_to_flex, statuses = run_match(
        [flex_row("ABTC", "SELL", 176, 1.23, "FLEX-ABTC-S")],
        [
            sqlite_row("ABTC", "SELL", 174, 1.23, "SQL-ABTC-S1"),
            sqlite_row("ABTC", "SELL", 2, 1.23, "SQL-ABTC-S2"),
        ],
    )

    assert statuses == {0: "aggregate_match"}
    assert set(sqlite_to_flex) == {"SQL-ABTC-S1", "SQL-ABTC-S2"}


def test_allt_split_flex_buys_match_one_sqlite_buy_by_aggregate_quantity():
    _assignments, sqlite_to_flex, statuses = run_match(
        [
            flex_row("ALLT", "BUY", 107, 3.45, "FLEX-ALLT-B1"),
            flex_row("ALLT", "BUY", 6, 3.45, "FLEX-ALLT-B2"),
        ],
        [sqlite_row("ALLT", "BUY", 113, 3.45, "SQL-ALLT-B")],
    )

    assert statuses == {0: "aggregate_match", 1: "aggregate_match"}
    assert list(sqlite_to_flex) == ["SQL-ALLT-B"]
    assert len(sqlite_to_flex["SQL-ALLT-B"]) == 2


def test_ampl_split_flex_sells_match_one_sqlite_sell_by_aggregate_quantity():
    _assignments, sqlite_to_flex, statuses = run_match(
        [
            flex_row("AMPL", "SELL", 1, 9.87, "FLEX-AMPL-S1"),
            flex_row("AMPL", "SELL", 100, 9.87, "FLEX-AMPL-S2"),
        ],
        [sqlite_row("AMPL", "SELL", 101, 9.87, "SQL-AMPL-S")],
    )

    assert statuses == {0: "aggregate_match", 1: "aggregate_match"}
    assert list(sqlite_to_flex) == ["SQL-AMPL-S"]


def test_poet_partial_fill_aggregate_respects_price_tolerance():
    _assignments, sqlite_to_flex, statuses = run_match(
        [flex_row("POET", "BUY", 50, 4.005, "FLEX-POET-B")],
        [
            sqlite_row("POET", "BUY", 20, 4.00, "SQL-POET-B1"),
            sqlite_row("POET", "BUY", 30, 4.01, "SQL-POET-B2"),
        ],
    )

    assert statuses == {0: "aggregate_match"}
    assert set(sqlite_to_flex) == {"SQL-POET-B1", "SQL-POET-B2"}
