# CryptoAI Autonomy Rebuild Plan

## Decision

Do not delete the current `v3` runtime first.

Use `v3` as the execution, exchange, risk, reconciliation, and observability baseline.
Build the autonomous planning stack in `nextgen_evolution/` until it can survive shadow and paper with enough evidence.

Deleting `v3` now would remove the only production-like safety shell in the repository while still leaving the hardest problems unsolved:

- strategy discovery
- promotion discipline
- failure rollback
- order safety
- operator visibility
- capital controls

## Target Architecture

The target system should not be a single fixed strategy and should not be a free-running LLM trader.
It should be an autonomous portfolio system with five hard boundaries:

1. Planner
   Produces experiment, promotion, repair, and capital directives from observed evidence.
2. Strategy Factory
   Generates new or repaired candidate genomes from seed families and failure signatures.
3. Validation Ladder
   Forces every candidate through backtest, walk-forward, shadow, paper, and limited-live gates.
4. Execution Shell
   Owns exchange adapters, order routing, reconciliation, and live kill switches.
5. Registry And Audit
   Persists why the system promoted, scaled down, exited, or retired a strategy.

## What "Autonomous" Means Here

Autonomous does not mean "let the LLM trade freely".
It means the system can repeatedly do these steps without manual parameter editing:

- detect degraded live or paper performance
- reduce risk automatically
- mutate or repair the affected strategy
- revalidate the repaired candidate
- only re-promote it after evidence improves
- keep capital concentrated in winners and away from damaged lineages

## Current Build Direction

The repo now has the correct place to build this in `nextgen_evolution/`:

- `AutonomyPlanner`
- `StrategyRepairEngine`
- `AutonomousDirector`

These modules define the minimum control loop needed for a self-repairing runtime:

- deployment decisions
- scale-up / scale-down decisions
- quarantine decisions
- repair plan generation
- revalidation staging

## Migration Order

1. Keep `main.py` + `core/engine.py` as the live execution shell.
2. Run `nextgen_evolution/` as shadow autonomy only.
3. Persist planner cycles, repair plans, and rollout decisions into the registry.
4. Add paper-order intent generation from autonomy directives.
5. Add limited-live budgeted execution with hard caps and immediate rollback.
   Status: implemented in `nextgen_evolution/live_bridge.py`, still dry-run by default until explicit enable + whitelist.
   Operator gate: now also wired to existing v3 settings / credentials / `manual_recovery_required` / nextgen live kill switch through `nextgen_evolution/live_runtime.py`.
   Emergency behavior: live gate failures now block new entries and can force-flatten nextgen-managed live positions during kill switch / manual recovery / model disable states.
   Integration status: v3 now exposes `run_nextgen_autonomy_live()`, scheduler can register `nextgen_live`, and guard/manual-recovery/model-disable paths can trigger the same flatten flow without relying on the CLI wrapper.
   Operator persistence: nextgen live requested state, optional whitelist, and active-runtime cap now persist in `system_state`, so scheduler-driven autonomy-live can stay armed across runs without weakening the emergency flatten path.
   Operator visibility: existing ops / guard reports now surface nextgen live requested state, effective gate state, kill switch / force-flatten status, and the latest nextgen live execution outcome.
6. Only then consider replacing parts of `v3` orchestration.

## Commercial Standard

For commercial use, the bar is not "a profitable backtest".
The minimum standard is:

- reproducible research lineage
- deterministic promotion rules
- bounded capital allocation
- exchange and order idempotency
- intraday kill switch
- post-trade audit trail
- rolling degradation detection
- repair before re-promotion

## Non-Negotiable Constraint

No architecture can guarantee stable profitability.
The correct engineering target is:

- faster discovery
- faster rejection of weak strategies
- faster repair of damaged strategies
- lower live failure blast radius
- better evidence before capital expansion
