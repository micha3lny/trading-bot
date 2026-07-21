# Regression analysis: full test suite after SHS session scoping

Date: 2026-07-21
Scope: classify failures reported after accepted SHS fixes (`ae398af`, `1642fd5`).

## Known verified facts

- `tests/test_historical_analysis_modules.py` passes on Raspberry Pi.
- `tests/test_shs_session_scoping.py` passes on Raspberry Pi with `PYTHONPATH=$PWD`.
- SHS/session commits touched only:
  - `src/live_trading/analysis/should_have_signaled_investigator.py`
  - `src/live_trading/analysis/signal_replay_analyzer.py`
  - `tests/test_historical_analysis_modules.py`
  - `tests/test_shs_session_scoping.py`
- Direct dependency search shows the reported failing groups do not import the SHS session-scoping helpers directly.

## Pytest import configuration

Root-cause for requiring `export PYTHONPATH=$PWD` is missing pytest project configuration. `pytest.ini` should set:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

This is test-runner configuration only; it does not change analyzer or trading logic.

## Failure Group Classification

### RuntimeDashboardQueries

Affected tests: tests under `tests/test_runtime_dashboard_queries.py`, exact failed test names require the full pytest output.

Classification: unrelated existing regression, not caused by SHS/session commits.

Root cause: dashboard query tests exercise canonical closed-trade display, peak quality masking, execution lookup, and SQLite-backed snapshot loading. The likely shared cause is drift after the canonical FIFO / trade peak / closed-position query changes. These tests import `src.dashboard.runtime_queries` and `SQLiteRuntimeStore`, not SHS modules.

Likely commit family: dashboard/SQLite/peak commits after SHS, not `ae398af` or `1642fd5`.

Confidence: high that it is not SHS; medium on exact root cause until exact failing assertions are available.

Recommended fix order: handle after SQLite schema fixture failures, because many dashboard tests depend on a correctly initialized temporary runtime DB.

### SQLiteRuntimeStore

Affected tests: tests under `tests/test_sqlite_runtime_store.py`, exact failed test names require the full pytest output.

Classification: unrelated existing regression or regression from canonical FIFO / trade peak store integration, not SHS.

Root cause: repeated `OperationalError("no such table: trades")` points to a schema initialization / fixture mismatch. `SQLiteRuntimeStore(init=True)` creates `trades`, but some tests or helpers may use raw `sqlite3.connect`, `SQLiteRuntimeStore(init=False)`, or a minimal fixture DB while newer code now assumes `trades` exists because trade peak repair and canonical FIFO hooks are imported/executed from the store layer.

Interpretation of `no such table: trades`: most likely missing schema initialization or optional feature incorrectly assumed mandatory in helper code. It is unlikely to be caused by SHS, because SHS commits do not touch `sqlite_store.py` or dashboard query code.

Likely commit family: canonical FIFO / trade peak persistence integration (`sqlite_store.py`, `trade_peak_rebuilder.py`), not `ae398af` or `1642fd5`.

Confidence: high.

Recommended fix order: first. Make temporary DB helpers consistently initialize schema, or guard optional trade/peak queries with table-existence checks where callers intentionally support partial/minimal DBs.

### CommissionLedger

Affected tests: tests under `tests/test_ibkr_commission_ledger.py`, exact failed test names require the full pytest output.

Classification: indirectly exposed by recorder/session work if failures mention changed recorder paths; otherwise unrelated existing regression from SQLite/finalize trade changes.

Root cause candidates:

- If failures are row-count/path assertions in `fills.csv`, inspect the recorder session rotation change. `record_recent_fills` and `record_commission_report` now pass fill row context into `recorder.path("fills.csv", row=...)`; this should preserve same-session writes, but any fake fill timestamp that resolves to a different date than the test recorder session can now rotate correctly and change test file location.
- If failures are SQLite/finalization assertions, the likely root cause is `finalize_pending_trades` / canonical FIFO integration, not SHS.

Likely commit: recorder rotation commit `a7652d9` only for file-location failures; otherwise SQLite/FIFO commits.

Confidence: medium without exact failure assertions.

Recommended fix order: after SQLite schema fixture failures. If only fake timestamp/session mismatch, update the test fixture to read from the resolved session directory or set fake execution session date explicitly.

### DailyReport / PostSessionDiagnostics

Affected tests: tests under `tests/test_post_session_diagnostics.py`, exact failed test names require the full pytest output.

Classification: likely indirectly exposed by recorder/session work for files like `eod_summary.json`, `eod_pending.json`, `trade_lifecycle.csv`; otherwise unrelated existing regression in daily report/SQLite closed-trade semantics.

Root cause candidates:

- Direct file writes using `recorder.path(...)` without row context still intentionally read the current recorder session. Writes now rotate only when a row/session context is supplied.
- Tests using synthetic timestamps that belong to a different session than the recorder's initial `session_date` can now write into a different session directory. This is correct runtime behavior but may require fixture alignment.
- Daily report tests that use SQLite executions/trades may be affected by canonical FIFO / peak persistence assumptions.

Likely commit: `a7652d9` if the failed assertion is about recorder output file location; otherwise canonical FIFO / trade peak commits.

Confidence: medium.

Recommended fix order: after SQLite store failures and after identifying whether the failed assertions are path/row-count or accounting values.

### ControlApi overnight scheduler

Affected tests: `tests/test_control_api.py` overnight scheduler tests, exact failed test names require the full pytest output.

Classification: unrelated existing regression, not SHS.

Root cause: these tests exercise overnight collector scheduling, history repair range decisions, and backlog/daily mode selection. They do not depend on SHS evidence matching. Failures likely come from history-repair/lookback/startup-collector changes or market-calendar changes, not SHS.

Likely commit family: history collector / market calendar / overnight scheduler changes, not `ae398af`, `1642fd5`, or recorder rotation.

Confidence: high not SHS; medium on exact code path without assertion output.

Recommended fix order: last unless it blocks production startup. Fix one scheduler policy mismatch rather than individual assertions.

## SHS shared-helper impact check

`ae398af` and `1642fd5` changed strict session scoping in:

- `row_belongs_to_session`
- `symbol_rows`
- `build_symbol_index`
- `sources_for_symbol`
- runtime signal-ready provenance handling

These helpers are used by SHS/NBAS/offline runtime forensic analyzers. They are not on the dashboard, SQLite store, commission ledger, daily report, or control API code paths. Therefore, failures in those groups should not be classified as genuine regressions introduced by SHS/session commits unless their exact stack trace enters one of the SHS modules above.

## Minimum fix set proposed after analysis

Do not shotgun-fix 31 individual tests. The minimum likely fix set is:

1. Add pytest root configuration so `pytest` can import `src` and `scripts` without `PYTHONPATH=$PWD`.
2. Resolve SQLite schema/fixture mismatch around `trades` once. This should collapse many `no such table: trades` failures.
3. For recorder-session-related test failures only, align fixtures with authoritative event session date or read from the resolved session directory.
4. Fix dashboard closed-position/peak query expectations once after SQLite fixture stability.
5. Fix overnight scheduler policy mismatch as one grouped change.

## Evidence gap

This report classifies by subsystem and inspected dependencies. To classify every failing test by exact name, provide the full pytest failure output, preferably:

```bash
pytest -q --tb=short | tee data/analysis/full_suite_failures_2026-07-21.txt
```

Then each failed test can be mapped one-by-one into the categories above.
