# CryptoAI v3

CryptoAI v3 is the active architecture for this repository. It is a rule-driven crypto trading system with LLM-assisted research, XGBoost prediction, strict risk controls, paper/live execution paths, reconciliation, reporting, and operator tooling.

The source-of-truth requirements document is [xuqiu/CryptoAIv3.md](./xuqiu/CryptoAIv3.md).

## Current Scope

- v3 runtime is active in [main.py](./main.py) and [core/engine.py](./core/engine.py)
- feature, prediction, training, backtest, walk-forward, execution, reconciliation, scheduler, and report artifacts are persisted into SQLite
- research inputs include CoinDesk / Jin10 / CryptoPanic summaries, Fear & Greed, LunarCrush, and on-chain context when configured
- execution pool is rebuilt automatically from model readiness + recent edge instead of being treated as a fixed static list
- runtime thresholds are auto-tightened by the learning layer using `prediction_runs` and `reflections`
- blocked low-quality setups can be auto-paused before new entries are allowed, including recent symbol-level negative setups
- paper trading is functional
- live trading remains guarded by default; `RUNTIME_MODE=live` only selects the live executor, while `ALLOW_LIVE_ORDERS=true` is required to place real orders
- dashboard and operational reports are available for daily use, but the UI is intentionally reduced to the automation-critical pages only

## Environment Variables

Runtime environment variables accepted by the active settings layer:

- `DB_PATH`
- `APP_DB_PATH`
- `RUNTIME_MODE`
- `APP_RUNTIME_MODE`
- `ALLOW_LIVE_ORDERS`
- `APP_ALLOW_LIVE_ORDERS`

Operational guidance:

- prefer `DB_PATH` when you want to point the engine at a non-default SQLite file
- treat `APP_DB_PATH` as a compatibility alias, not the primary example to copy into run commands
- use `RUNTIME_MODE=paper` for paper execution
- use both `RUNTIME_MODE=live` and `ALLOW_LIVE_ORDERS=true` before expecting real orders

## Current Runtime Baseline

The active runtime baseline is now dynamic rather than hard-coded:

- the execution pool is auto-rebuilt and auto-filtered by recent symbol edge
- runtime overrides come from three layers: `default`, `manual`, and `learning`
- learning overrides can automatically tighten `xgboost_probability_threshold`, `final_score_threshold`, `min_liquidity_ratio`, and `sentiment_weight`
- blocked setups can prevent new entries even when the raw model stack is otherwise permissive
- new entries still require explicit `LLM suggested_action = OPEN_LONG`
- `python main.py validate ...` remains the fast validation path for threshold and pool changes
- OKX public market data is still treated as degradable infrastructure; cached OHLCV / recent close fallback is expected during paper validation when the public endpoints are unstable

For the current effective values on a given machine, use:

- the `Settings` page in `dashboard.py`
- `runtime_settings_effective` in `system_state`
- `python main.py ops`

## Baseline Status

See [RELEASE_NOTES_v3_BASELINE.md](./RELEASE_NOTES_v3_BASELINE.md) for the current delivered baseline and known boundaries.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
copy .env.example .env
```

For reproducible deployments, prefer locking the environment with `pip-compile` or an equivalent lockfile workflow instead of relying only on floating minimum versions.

## Container Deployment

```powershell
docker compose up -d --build
```

This starts:

- `cryptoai-engine`: scheduler/engine service
- `cryptoai-dashboard`: Streamlit dashboard on `http://localhost:8501`

The default compose file now binds the dashboard to `127.0.0.1:8501` only. If you need remote access, put it behind an authenticated reverse proxy or use an SSH tunnel instead of exposing Streamlit directly.

Optional alert channels in `.env`:

- `FEISHU_WEBHOOK_URL`
- `FEISHU_WEBHOOK_SECRET`
- `FEISHU_CRITICAL_WEBHOOK_URL`
- `FEISHU_CRITICAL_WEBHOOK_SECRET`
- `LANGUAGE=zh` or `LANGUAGE=en`

For a small Linux cloud server such as Tencent Cloud `2 vCPU / 4 GB RAM`, use the lightweight profile in [DEPLOY_TENCENT_LINUX.md](./DEPLOY_TENCENT_LINUX.md):

```bash
docker compose -f docker-compose.tencent-lite.yml up -d --build
docker compose -f docker-compose.tencent-lite.yml --profile dashboard up -d --build
```

The lightweight Tencent Cloud profile binds the dashboard to `127.0.0.1:8501` by default. Expose it externally only through an authenticated reverse proxy or SSH tunnel.

## Core Commands

```powershell
python main.py once
python main.py loop
python main.py train
python main.py report
python main.py walkforward BTC/USDT
python main.py backfill 180
python main.py backtest BTC/USDT
python main.py validate BTC/USDT,ETH/USDT
python main.py reconcile
python main.py approve-recovery
python main.py health
python main.py guards
python main.py drift
python main.py metrics
python main.py alpha
python main.py attribution
python main.py maintenance
python main.py failures
python main.py incidents
python main.py ops
python main.py execution-pool
python main.py execution-rebuild
python main.py schedule once
python main.py daemon
```

Example isolated paper run against a temporary database:

```bash
DB_PATH=/tmp/cryptoai-smoke.db RUNTIME_MODE=paper ALLOW_LIVE_ORDERS=false python main.py once
```

`python main.py init-system` now creates an automatic SQLite backup in `data/backups/pre-init-system-<timestamp>/` before clearing runtime tables and generated artifacts.

## Dashboard

```powershell
streamlit run dashboard.py
```

Dashboard pages include:

- Overview
- Settings
- Predictions
- Ops

Manual and low-frequency operator controls still exist in code and CLI, but they are intentionally hidden from the default dashboard navigation.

## Active Vs Legacy

Active path:

- `main.py`
- `core/engine.py`
- `config/__init__.py`
- `dashboard.py`
- `execution/*` used by the active runtime
- `monitor/*` used by the active runtime

Experimental or older modules may still exist in the repo, but new work should extend the active v3 runtime path unless a deliberate refactor is being done.

## Operations

See [RUNBOOK.md](./RUNBOOK.md) for:

- recommended execution sequence
- automation-first operating model
- scheduler usage
- reconciliation workflow
- incident triage
- daily operator checklist

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Old project paths and pre-v3 assumptions are deprecated.
- Tests use `unittest`.
- All datetime handling in the active runtime path should remain timezone-aware (`timezone.utc`).
- On this host, the project `.venv` is the only reliable runtime Python. System Python may not have the required dependencies.
