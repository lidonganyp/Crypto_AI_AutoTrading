# CryptoAI v3 Refactor Plan

## Goal

Refactor the active runtime without breaking paper/live safety so the system can:

- remove redundant logic and stale paths
- reduce operational risk in the main loop
- stay responsive with larger SQLite tables and longer runtime history
- make future strategy changes cheaper and safer

## Refactor Principles

- keep behavior stable unless a change is explicitly risk-reducing
- prefer extraction and boundary cleanup over large rewrites
- every phase must end with passing regression tests
- optimize the storage and query layer before changing strategy behavior

## Phase 0: Baseline And Safety Rails

- keep `paper` loop as the primary verification path
- preserve existing model file compatibility and current DB schema compatibility
- use full test-suite checkpoints after each phase
- treat `core/engine.py`, `core/storage.py`, `dashboard.py`, and runtime DB state as migration-sensitive

## Phase 1: Storage And Query Performance

- add missing SQLite indexes for runtime, reporting, dashboard, and shadow-analysis queries
- reduce lock-related instability with safer connection settings such as `busy_timeout`
- centralize the heaviest recurring queries instead of duplicating raw SQL in pages and services
- identify stale-report dependencies and replace them with live aggregation where needed

## Phase 2: Engine Decomposition

- split `run_once()` into explicit stages:
  - cycle preflight
  - execution-pool refresh
  - active symbol analysis
  - shadow observation analysis
  - position management
  - matured evaluation and reporting
- extract symbol snapshot building from `core/engine.py`
- extract shadow evaluation / shadow trade bookkeeping into a dedicated runtime service
- move runtime override orchestration into a dedicated learning/runtime coordinator

## Phase 3: Dashboard Decomposition

- split the dashboard into page modules and shared query helpers
- separate DB reads, formatting, and Streamlit rendering
- keep only operator-critical pages in the default navigation
- ensure key metrics are fresh even when report artifacts lag behind

## Phase 4: Test And Naming Cleanup

- split `tests/test_v2_architecture.py` by domain:
  - engine
  - execution
  - learning
  - monitor
  - dashboard
- keep compatibility-sensitive internal names stable until runtime/data migrations are complete
- then clean internal `v2` naming that no longer reflects the active version

## Phase 5: Live-Readiness Hardening

- add explicit performance and freshness checks for large tables
- add paper-to-live operational checklists tied to runtime metrics
- tighten report freshness, failure visibility, and reconciliation observability

## Execution Order

1. Complete Phase 1 fully before further feature work.
2. Execute Phase 2 in small extractions with no behavior drift.
3. Execute Phase 3 after engine boundaries are stable.
4. Finish with Phase 4 and Phase 5 before expanding real-live scope.

## Current Status

- Phase 0: completed baseline assessment
- Phase 1: in progress
