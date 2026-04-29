# CryptoAI v3 Baseline Release Notes

Date: 2026-03-21

## Scope

This baseline reflects the active `D:\CODE\CryptoAI` v3 architecture aligned to `xuqiu/CryptoAIv3.md` as far as practical within the current repository.

## Delivered

- deterministic runtime engine with SQLite persistence
- feature pipeline expanded to 50+ features
- XGBoost training, prediction, holdout, walk-forward, and backtest support
- drift, guard, health, metrics, ops, failure, and incident reporting
- guarded live execution with:
  - slippage checks
  - market order path
  - limit fallback path
  - polling confirmation
  - timeout cancel / retry
- reconciliation with ratio threshold escalation
- manual recovery approval flow
- model degradation detection with runtime threshold tightening / disable mode
- market latency warning and circuit breaker
- multi-source validation before trading
- news and sentiment inputs from:
  - CoinDesk
  - Jin10
  - CryptoPanic
  - Fear & Greed
  - LunarCrush
- on-chain inputs from:
  - Glassnode
  - CoinMetrics
- Feishu webhook notification support

## Important Boundaries

- The live path is guarded but still not equivalent to a fully production-hardened exchange OMS.
- Some older modules remain in the repository for reference or partial legacy use, but they are not the primary v3 runtime path.
- News and on-chain integrations depend on external API availability and keys.

## Suggested Next Phase

- stabilize long-term operations on paper mode
- add richer market microstructure features
- add multi-exchange runtime redundancy
- strengthen ingestion-time consistency checks
