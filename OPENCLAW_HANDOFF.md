# CryptoAI Handoff For OpenClaw

## Repo

- Project root: `/home/lidong/lidong/CryptoAI`
- Single source of truth requirements doc: [`xuqiu/CryptoAIv3.md`](/home/lidong/lidong/CryptoAI/xuqiu/CryptoAIv3.md)
- Legacy `PRD.md` has been deleted on purpose

## Current Runtime

Last checked by Codex on `2026-03-28`:

- `main.py loop` is running
- `streamlit run dashboard.py` is running
- `python main.py health` returned `status: ok`
- `execution_symbols = ["BTC/USDT"]`
- `active_symbols = ["BTC/USDT"]`
- `broken_model_symbols = {}`
- `manual_recovery_required` is not set

Use these commands for fresh state, not this file:

```bash
cd /home/lidong/lidong/CryptoAI
./.venv/bin/python main.py health
./.venv/bin/python main.py ops
./.venv/bin/python main.py metrics
./.venv/bin/python main.py execution-pool
./.venv/bin/python main.py failures
```

## Important Recent Changes

### Runtime / Safety

- LLM clients ignore host proxy env vars (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`)
- LLM runtime failures now use short backoff instead of retrying every symbol / every cycle
- Live equity now uses exchange quote balance plus managed position value
- Partial close realized PnL is included in account equity, daily/weekly PnL, and daily report
- Reconciliation no longer treats extra external balances/orders as mismatch by default
- Health stale-stream logic now respects `analysis_interval_seconds`

### Models

- Model writes are atomic (`temp file + replace`)
- Loop now runs lightweight model maintenance / self-heal after cycles
- BTC model corruption caused by tests has been repaired

### Validation Sample Expansion

- `near_miss_shadow_enabled = true`
- `paper_canary_enabled = true` in current `.env`
- Near-miss shadow now records cases where model score is close but trade is blocked by risk/review
- Current live data showed `BTC/USDT` being blocked mainly by `insufficient liquidity`

## Current Strategy Reality

- System is operational as a paper runtime
- Signal generation is active (`prediction_runs`, `prediction_evaluations` exist)
- Strategy validation is still in progress because recent periods had few or zero true opens
- New near-miss shadow logic was added specifically to break the “no trades => no validation” loop

## What OpenClaw Should Check First

1. Read [`xuqiu/CryptoAIv3.md`](/home/lidong/lidong/CryptoAI/xuqiu/CryptoAIv3.md)
2. Read this file
3. Run `health`, `ops`, `metrics`, `execution-pool`
4. Inspect:
   - `prediction_runs`
   - `prediction_evaluations`
   - `shadow_trade_runs`
   - `execution_events` with `paper_canary_open`
5. Treat “0 real opens” as a strategy-validation question, not a runtime-failure question

## Known Open Questions

- Whether near-miss shadow starts accumulating enough samples over the next few days
- Whether paper canary starts opening small validation positions under current market conditions
- Whether strategy thresholds are still too strict even after near-miss / canary support

