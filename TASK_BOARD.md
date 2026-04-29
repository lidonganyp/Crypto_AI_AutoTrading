# CryptoAI v3 Task Board

Last updated: 2026-03-21

## Current Baseline

- [x] v3 runtime is the active runtime path
- [x] paper trading path is functional
- [x] guarded live trading path exists with market/limit execution, polling, timeout cancel, and slippage checks
- [x] XGBoost training, prediction, holdout, and walk-forward evaluation are wired
- [x] multi-source research inputs are wired: CoinDesk, Jin10, CryptoPanic, Fear & Greed, LunarCrush, Glassnode, CoinMetrics
- [x] risk controls cover daily/weekly/drawdown/cooldown/circuit-breaker paths
- [x] manual recovery flow exists
- [x] drift, guard, failure, incident, ops, health, and metrics reports are available
- [x] dashboard reflects current v3 operational state

## Completed High-Priority Gaps

- [x] aware datetime usage across active runtime path
- [x] partial close PnL handling
- [x] order state machine
- [x] reconciliation with mismatch ratio threshold
- [x] model degradation detection and runtime threshold adjustment
- [x] market latency warning and circuit breaker
- [x] cross-source validation before trading
- [x] bearish news hard block
- [x] funding-rate block
- [x] Feishu webhook notifications

## Remaining Enhancements

- [ ] richer order-book derived features such as large-order imbalance / large trade net flow
- [ ] stronger turnover-rate features with reliable circulating-supply inputs
- [ ] multi-exchange runtime adapter beyond current OKX-first path
- [ ] long-horizon archival / rotation policy for 5-year storage objectives
- [ ] stricter real-time multi-source consistency checks at ingestion time
- [ ] optional A/B rollout framework for model gray release

## Active Guidance

- Treat `main.py`, `core/engine.py`, `config/__init__.py`, `dashboard.py`, and the active monitor/execution modules as the source of truth.
- Do not route new work through older orchestration paths unless explicitly refactoring them.
- Prefer extending the active v3 runtime path instead of reviving older experimental modules.
