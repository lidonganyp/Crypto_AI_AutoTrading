# NextGen Evolution

This directory is a greenfield scaffold for a new trading system whose goal is **active alpha creation, fast validation, staged promotion, and capital allocation**.

It exists because the current root runtime is a conservative paper-trading framework, not a true self-evolving profit engine.

## Design Goal

Build toward a system that can:

1. Generate many candidate strategies instead of only tightening thresholds on one strategy.
2. Validate candidates across backtest, walk-forward, shadow, paper, and live stages.
3. Promote only the candidates that keep edge after costs, drawdown, and latency penalties.
4. Allocate capital across winners rather than forcing one monolithic strategy.

## Fused Architecture

- `alpha_factory.py`
  Generates candidate strategy genomes from seed strategy families.
  Inspired mainly by `vectorbt` parameter sweeps and `freqtrade` strategy iteration loops.

- `validation.py`
  Scores each candidate on expectancy, drawdown, cost drag, latency, and regime consistency.
  Inspired by `freqtrade` dry-run / backtest discipline and `LEAN`'s separate research/optimization/live workflow.

- `promotion.py`
  Handles stage transitions: `reject -> shadow -> paper -> live`.
  Inspired by `LEAN` deployment flow and `NautilusTrader` research/live parity.

- `allocator.py`
  Assigns capital only to promoted candidates using bounded weights.
  Inspired by `LEAN`'s portfolio-model separation.

- `engine.py`
  Orchestrates generation, validation, promotion, and allocation.
  Intended to become the future integration seam with exchange adapters and market data.

## Why This Is Separate

This directory is intentionally isolated from the current `main.py` runtime path so you can explore a more aggressive alpha-seeking architecture without destabilizing the production-like paper loop.

## Current Status

This is a scaffold, not a finished trading engine. What is implemented now:

- candidate generation
- SQLite OHLCV feed adapter over the existing CryptoAI database
- experiment lab for intraday candidate evaluation on real candles
- promotion registry for persisted experiment runs, candidate scores, and allocations
- validation scoring
- promotion stages
- capital allocation
- autonomy planner / repair / director scaffold
- rollout state machine and paper execution bridge
- guarded limited-live execution bridge
- live runtime/operator adapter over existing v3 settings and system_state
- reusable live-cycle runner that v3 can call directly for scheduled nextgen sync or emergency flatten
- runtime evidence collector over real trades / PnL ledger
- autonomy cycle persistence for execution and repair decisions
- source-selection rationale in `INSPIRATIONS.md`

What still needs real implementation:

- richer signal primitives that actually mine alpha
- event-driven live execution runtime
- exchange adapter layer
- cost model calibrated by venue
- deeper feature store
- shadow/paper/live promotion state transition logic beyond experiment-level persistence
- degradation telemetry sourced from real fills, slippage, and reconciliation
- real operator controls and exchange credentials management for explicitly-enabled live rollout

## Quick Demo

```bash
python -m nextgen_evolution
python -m nextgen_evolution autonomy
python -m nextgen_evolution autonomy-paper
python -m nextgen_evolution autonomy-live
python -m nextgen_evolution autonomy-live --enable-live
python -m nextgen_evolution autonomy-live --disable-live
python -m nextgen_evolution autonomy-live --kill-live
python -m nextgen_evolution autonomy-live --clear-live-kill-switch
```

If `data/cryptoai.db` exists and contains `5m` candles, the module now runs a real-data experiment first.
That demo also writes the experiment result into the SQLite-backed promotion registry tables.
`autonomy` mode additionally builds and persists one autonomy cycle containing execution directives and repair plans.
`autonomy-paper` goes one step further and applies symbol-level paper execution intents through the existing `PaperTrader` shell.
`autonomy-live` applies the same autonomy output through a guarded `LiveTrader` bridge.
It now also reads the existing v3 operator surface:

- `APP_RUNTIME_MODE`
- `ALLOW_LIVE_ORDERS`
- exchange credentials from `.env`
- `manual_recovery_required` / `manual_recovery_approved`
- `model_degradation_status`
- nextgen-specific live kill switch in `system_state`

`--enable-live` and `--disable-live` now persist the nextgen live operator request in `system_state`, and optional `--live-whitelist` / `--live-max-active-runtimes` updates are persisted alongside it.
Even with `--enable-live`, it stays dry-run unless those gates are all satisfied and the symbol is whitelisted.
When kill switch, manual recovery, or model disable is active, the live bridge now prioritizes flattening nextgen-managed live positions instead of opening anything new.
Both autonomy modes now also snapshot runtime evidence from actual managed trades when historical execution exists.
The same guarded path is now callable from `CryptoAIV2Engine.run_nextgen_autonomy_live()`, and v3 can trigger emergency flatten from its scheduler / guard surfaces without shelling out to `python -m nextgen_evolution`.
The scheduler-facing path now reads the persisted nextgen live operator request by default, so explicit arming survives beyond a single CLI invocation.
Current v3 reports now also surface nextgen live requested/effective state, force-flatten state, kill switch state, and latest nextgen live run details through the existing ops / guard reporting layer.

## Autonomy Scaffold

The current autonomy scaffold is intentionally constrained.
It does not let an LLM place trades freely.
It does three narrower things:

- turn experiment output into runtime snapshots
- decide promote / keep / scale down / exit actions
- generate repair plans for degraded strategies and persist them

Main modules:

- `director.py`
  Builds runtime snapshots from experiment results and hands them to the planner.
- `planner.py`
  Produces execution directives for `shadow`, `paper`, and `live` stages.
- `repair.py`
  Produces repair candidates when drawdown, expectancy, or cost signals degrade.
- `promotion_registry.py`
  Persists autonomy cycles, execution directives, and repair plans for auditability.
- `rollout.py`
  Keeps deployment transitions staged instead of jumping directly from `paper` to unrestricted `live`.
- `execution_bridge.py`
  Converts autonomy output into symbol-level paper execution intents with blast-radius caps.
- `live_bridge.py`
  Converts autonomy output into limited-live/live intents with whitelist gates, runtime caps, rollback closes, operator-gated entry blocks, and emergency flatten handling.
- `live_runtime.py`
  Resolves effective live mode from settings, credentials, kill switch, and recovery state, then separately decides whether entries are allowed and whether emergency managed closes stay armed.
- `runtime_evidence.py`
  Aggregates real runtime evidence from managed trades, positions, and `pnl_ledger` so the planner can react to actual degradation.

## Recommended Next Build Steps

1. Add an `experiment_lab` that runs large candidate batches over intraday datasets.
2. Add a `connector` layer that standardizes order book, trades, balances, and orders.
3. Add a persistent `promotion_registry` to record candidate stage transitions.
4. Replace the toy scoring formula with a real objective that includes fill quality and slippage.
