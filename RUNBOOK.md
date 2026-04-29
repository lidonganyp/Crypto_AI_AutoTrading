# CryptoAI v3 Runbook

## 1. Purpose

This runbook describes how to operate the active CryptoAI v3 project in a repeatable way. It is focused on automation-first paper trading, then controlled progression toward guarded live execution.

For a delivered-scope snapshot, also see [RELEASE_NOTES_v3_BASELINE.md](./RELEASE_NOTES_v3_BASELINE.md).

## 2. Environment

Recommended startup steps:

```powershell
cd D:\CODE\CryptoAI
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

Containerized startup:

```powershell
docker compose up -d --build
```

Check the main configuration inputs before first run:

- `.env`
- `config/__init__.py`
- database path in `app.db_path`
- runtime mode in `app.runtime_mode`
- live-order gate in `app.allow_live_orders`
- symbol list in `exchange.symbols`
- Feishu alert webhooks if you want IM alerting

Current operating baseline:

- execution pool is auto-rebuilt from model readiness + recent edge
- runtime thresholds may change cycle to cycle because learning overrides are refreshed automatically
- learning overrides can now tighten `xgboost_probability_threshold`, `final_score_threshold`, `min_liquidity_ratio`, and `sentiment_weight`
- blocked low-quality setups can pause new entries without requiring an operator to manually remove symbols
- new entries still require `LLM suggested_action = OPEN_LONG`
- OKX public market endpoints are not consistently reachable on this host; cached OHLCV / recent-close fallback is expected during paper validation
- the dashboard is intentionally reduced to the core automation pages: `Overview`, `Settings`, `Predictions`, `Ops`

Recommended Feishu settings:

- `FEISHU_WEBHOOK_URL`: normal info/warning alerts
- `FEISHU_CRITICAL_WEBHOOK_URL`: critical alerts such as circuit breaker or manual recovery required

Security note:

- do not keep real exchange or LLM keys in `.env` inside the workspace long-term
- prefer system environment variables, deployment secrets, or an external secret manager
- if real keys were ever written into `.env`, rotate them before continued use

Environment variable notes:

- prefer `DB_PATH` when you want to override the runtime SQLite file
- `APP_DB_PATH` is still accepted as an alias, but `DB_PATH` should be the operator-facing example
- `RUNTIME_MODE` and `APP_RUNTIME_MODE` are both accepted
- `ALLOW_LIVE_ORDERS` and `APP_ALLOW_LIVE_ORDERS` are both accepted

## 3. Safe Execution Order

Use this sequence for a fresh environment or after major strategy changes:

1. `python main.py init-system`
2. fill `.env` with valid keys or export them via secure environment variables
3. `python main.py backfill 180`
4. `python main.py execution-rebuild`
5. `python main.py train`
6. `python main.py walkforward BTC/USDT`
7. `python main.py backtest BTC/USDT`
8. `python main.py health`
9. `python main.py metrics`
10. `python main.py once`
11. `python main.py report`
12. `python main.py validate BTC/USDT,ETH/USDT`

If results are stable, move to continuous paper trading:

```powershell
python main.py loop
```

Or use scheduled execution:

```powershell
python main.py daemon
```

Safe isolated one-off cycle example:

```bash
DB_PATH=/tmp/cryptoai-paper-smoke.db RUNTIME_MODE=paper ALLOW_LIVE_ORDERS=false python main.py once
```

## 4. Command Guide

Main commands:

- `python main.py once`: run one full trading cycle
- `python main.py loop`: run the engine continuously using `analysis_interval_seconds`
- `python main.py init-system`: reset runtime database tables and generated artifacts for a clean start; the command now creates an automatic backup under `data/backups/pre-init-system-<timestamp>/`
- `python main.py cleanup-data`: clean generated artifacts while preserving models and current-day runtime files
- `python main.py train`: train models for configured symbols
- `python main.py walkforward [symbol]`: run walk-forward evaluation
- `python main.py backtest [symbol]`: run v3 backtest
- `python main.py validate [symbols]`: run the accelerated validation sprint report for current candidate symbols
- `python main.py backfill [days]`: backfill configured symbols and timeframes
- `python main.py execution-pool`: inspect current execution / active / model-ready pools
- `python main.py execution-rebuild`: force a rebuild of the execution pool from recent edge
- `python main.py reconcile`: compare internal state with executor/account view
- `python main.py approve-recovery`: clear manual recovery block after operator review
- `python main.py report`: generate daily and weekly reports
- `python main.py health`: generate health report
- `python main.py guards`: generate guardrail and alert summary
- `python main.py drift`: compare backtest, walk-forward and runtime drift
- `python main.py metrics`: generate performance metrics report
- `python main.py maintenance`: apply retention cleanup
- `python main.py failures`: generate failure summary
- `python main.py incidents`: generate incident summary
- `python main.py ops`: generate compact operational overview
- `python main.py schedule [job_name]`: run a single scheduler job and persist the result
- `python main.py daemon`: start the blocking scheduler

## 5. Scheduler Behavior

The current scheduler is APScheduler-based and interval-driven. It auto-registers only jobs supported by the active engine instance.

Default scheduled jobs:

- analysis cycle
- health check
- guard report
- drift report
- ops overview
- training
- walk-forward
- reports
- reconciliation
- maintenance
- failure report
- incident report

Relevant config keys are in `config/__init__.py` under `SchedulerSettings`.

If a schedule value is `0` or below, that job is skipped.

Current operator note:

- `loop` cadence is controlled by `analysis_interval_seconds`
- if you restart `loop`, it will inherit the current `.env` cadence, current execution pool state, and current learning runtime overrides

## 6. Daily Operator Checklist

Run or verify these items at least once per day:

1. Confirm latest `health` report exists and status is acceptable.
2. Confirm latest `guards` report exists.
3. Confirm latest `drift` report exists.
4. Confirm latest `ops` report exists.
5. Review `scheduler_runs` for failed jobs.
6. Review `reconciliation_runs` and ensure mismatch count is `0`.
7. Review latest `account_snapshots` for drawdown, cooldown, and circuit breaker state.
8. Review `runtime_settings_effective`, `runtime_settings_learning_details`, and `runtime_settings_override_conflicts`.
9. Review `execution_events` for rejected orders, timeouts, abnormal closes, latency blocks, setup auto-pause, or cross-validation failures.
10. Generate `report`, `guards`, `drift`, `metrics`, and `failures` if not already scheduled.
11. During validation sprint mode, run `validate BTC/USDT,ETH/USDT` or the current target symbols after meaningful threshold or pool changes.

For a cold start after reset:

1. confirm `init-system` completed without unexpected skipped files
2. run `backfill`, `watchlist-refresh`, `train`, and `health`
3. verify the dashboard shows empty historical state before new accumulation begins

## 7. Incident Triage

When the system behaves abnormally, use this order:

1. `python main.py health`
2. `python main.py guards`
3. `python main.py drift`
4. `python main.py ops`
5. `python main.py reconcile`
6. `python main.py failures`
7. `python main.py incidents`

Focus first on:

- circuit breaker activation
- rising drawdown
- repeated scheduler failures
- reconciliation mismatch
- persistent order rejection due to slippage or balance constraints
- repeated OKX public market endpoint failures; if cycles still complete using cached data, treat this as degraded-but-running rather than a full outage

## 8. Reconciliation Workflow

Recommended daily workflow:

1. Run `python main.py reconcile`
2. Inspect the latest row in `reconciliation_runs`
3. If `mismatch_count > 0`, pause further live-style execution and inspect:
   - `positions`
   - `orders`
   - `execution_events`
   - latest account snapshot
4. Do not resume live execution until the mismatch source is understood
5. If the system entered manual recovery mode, run `python main.py approve-recovery` only after operator confirmation

## 9. Training And Evaluation Policy

Recommended cadence:

- incremental training: daily or when enough new market data accumulates
- walk-forward: daily after training
- backtest: after meaningful strategy or feature changes
- validation sprint: after execution-pool rebuilds, threshold changes, or learning-layer behavior changes
- report review: daily and weekly

Do not treat backtest profitability as permission for immediate live deployment. The project target remains:

- paper trading first
- small-capital guarded rollout second
- scale only after stable out-of-sample behavior

Current operator target:

- let the automation layer rebuild the execution pool instead of maintaining a hand-picked single-symbol pool
- accumulate valid `prediction_runs` and closed-trade samples so learning overrides and blocked setups can become more selective
- keep manual overrides unlocked unless there is a deliberate reason to pin a field

## 10. Live Mode Warning

`LiveTrader` is currently a guarded execution path with slippage checks and adapter structure, but it is not the same as a fully production-ready exchange execution stack.

Before using live mode, verify:

- `RUNTIME_MODE=live`
- `ALLOW_LIVE_ORDERS=true`
- exchange adapter order placement is fully implemented
- balance checks are correct
- reconciliation is stable
- notifications are wired to real channels, preferably Feishu webhooks
- manual intervention procedures are documented
- recent paper-trading cycles have produced enough closed-trade and prediction-evaluation samples under the current automation baseline

## 11. Validation

Before shipping a code change, run:

```powershell
python -m unittest discover -s tests -v
python -m py_compile main.py
```
