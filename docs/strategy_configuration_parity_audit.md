# Strategy Configuration Parity Audit

## Inventory and classification

| Path | Threshold use | Classification | Parity action |
|---|---|---|---|
| `src/live_trading/analysis/common.py` | `4.0/6.5/5.0` helper defaults | historical library defaults | Kept for direct programmatic compatibility; not accepted as session evidence by CLI analyzers. |
| `src/live_trading/analysis/missed_runners_analyzer.py` | `failed_first5`, `failed_first15`, `failed_or_range`, `should_have_signaled` | live-session forensic | Resolve exact-session metadata or complete CLI triple; emit provenance. |
| `src/live_trading/analysis/strategy_coverage_report.py` | coverage diagnostics call missed-runner signal replay | live-session KPI | Same resolver and provenance. |
| `src/live_trading/analysis/offline_runtime_pre_signal_analyzer.py` | offline/runtime gate parity | live-session forensic | Same resolver and provenance. |
| `src/live_trading/analysis/signal_case_trace.py` | one-symbol signal verdict | live-session forensic | Same resolver; provenance in trace. |
| `src/live_trading/analysis/buy_decision_trace.py` | one-symbol BUY gate verdict | live-session forensic | Same resolver; provenance in trace and summary CSV. |
| `src/live_trading/analysis/signal_opportunity_forensics.py` | live-equivalent gates plus legacy breakout comparison | mixed | Session gates resolve metadata/CLI; legacy breakout remains explicitly diagnostic. |
| `src/live_trading/analysis/full_session_replay_v67.py` | live, low-threshold and legacy profiles | mixed | `live` resolves session config; named historical profiles retain their definitions; all outputs emit provenance. |
| `src/live_trading/analysis/top100_buy_analyzer.py` | current gate filter and causal replay | live-session analysis with named experiments | Current filter resolves session config; `late_bloomer_separate_setup` retains its historical `0.5/1.0/<4.0` definition. |
| `src/live_trading/analysis/candidate_lifetime_analyzer.py` | baseline and counterfactual gate lifetimes | live-session plus explicit counterfactuals | Baseline resolves session config; every row emits provenance. |
| `src/live_trading/analysis/should_have_signaled_investigator.py` | consumes missed-runner classifications | downstream forensic | Preserves upstream provenance; does not invent thresholds. |
| `src/strategies/momentum_trailing_intraday/backtest_momentum_or_breakout_v59_daily_top_universe.py` | defaults `4.0/6.5` | historical backtest | Kept unchanged; its CLI is the experiment definition, not session-runtime evidence. |
| `src/strategies/momentum_trailing_intraday/backtest_momentum_or_breakout_v60_filter_scenarios.py` | named strict/relaxed grids | historical scenario study | Kept unchanged. |
| `scripts/backtest_v67_replay_last_days.py` | explicit low-threshold replay defaults | historical experiment | Kept unchanged. |
| `scripts/patch_v67_strict_setup_tag.py`, `scripts/patch_v67_strict_peak_analytics.py`, `scripts/patch_v67_eod_strict_reentry.py` | explicit `strict_*` values | migration/patch tooling | Kept unchanged because the values define the historical strict tag. |

## Live-session analyzers

These outputs make claims about the v67 decisions for a specific session and must resolve thresholds from session-scoped `run_metadata.csv` or from a complete explicit CLI triple:

- `missed_runners_analyzer.py`: emits `failed_first5`, `failed_first15`, `failed_or_range`, and `should_have_signaled`.
- `strategy_coverage_report.py`: derives `missed_should_have_signaled` through the missed-runner replay.
- `offline_runtime_pre_signal_analyzer.py`: compares offline signal gates with runtime evidence.
- `signal_case_trace.py` and `buy_decision_trace.py`: forensic live-decision traces.
- `signal_opportunity_forensics.py`: live-equivalent opportunity gates.
- `full_session_replay_v67.py` profile `live`: claims current/session live parity.
- downstream SHS/NBAS reports: consume classifications created by `missed_runners_analyzer.py` and must preserve its effective configuration fields.

The static constants in `analysis/common.py` describe the historical strict defaults. They are not evidence that a particular runtime session used those values.

## Explicit analytical profiles

- `full_session_replay_v67.py` profile `low_threshold_causal`: intentionally uses first5 `0.5`, first15 `1.0`, and retains the profile OR gate.
- `full_session_replay_v67.py` profile `legacy_offline`: intentionally reproduces the historical offline model and is marked non-causal where applicable.
- `backtest_v67_replay_last_days.py`: explicit historical/experimental CLI defaults `0.5/1.0`; it does not claim runtime-session parity.
- `backtest_momentum_or_breakout_v60_filter_scenarios.py`: named scenario grid containing strict and relaxed variants; constants are scenario definitions.

## Historical strict tooling

- `patch_v67_strict_setup_tag.py`, `patch_v67_strict_peak_analytics.py`, and `patch_v67_eod_strict_reentry.py` explicitly label the historical strict setup `4.0/6.5/5.0`.
- `src/live_trading/analytics/v67_missed_runners_report.py` is a legacy recorder report with its own explicit low-threshold CLI model; it is not part of the canonical daily pipeline and does not claim exact v67 session parity.
- Older strategy backtests retain their own named/default configurations as historical experiments.

## Provenance contract

Live-session outputs must contain:

- `effective_min_first5`
- `effective_min_first15`
- `effective_min_or_range`
- `config_source`

Resolution order is complete CLI triple (`config_source=cli_explicit`), then exact-session runtime metadata (`config_source=run_metadata`). Missing metadata is an error instead of a silent fallback. `top100_source_date`, named analytical profiles, and analyzer module constants are not valid evidence of the runtime configuration for a session.

The live trader writes these values to session-scoped `run_metadata.csv` at startup and again on a process-surviving session boundary. `run_daily_analysis.py` also accepts the complete triple and forwards it only to registered analyzers that evaluate these gates. Partial overrides are rejected.
