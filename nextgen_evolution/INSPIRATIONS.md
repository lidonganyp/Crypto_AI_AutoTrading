# Inspiration Sources

This directory is a clean-room scaffold inspired by the architecture strengths of several mature open-source trading systems. It does **not** copy source code from those repositories.

## Selected Repositories

Snapshot date for repo metadata: `2026-04-09`.

| Project | Stars | Why it was selected | Strength adopted here |
| --- | ---: | --- | --- |
| `freqtrade/freqtrade` | 48.5k | Strong crypto workflow, dry-run, backtesting, hyperopt, FreqAI hooks | research-to-paper loop, candidate evaluation ladder, operator-first tooling |
| `nautechsystems/nautilus_trader` | 21.7k | Deterministic event-driven runtime spanning research and live | parity-first architecture, single execution semantics across modes |
| `QuantConnect/Lean` | 18.3k | Mature algorithm framework with separate research, optimize, live flows | clear separation of research, optimization, and deployment stages |
| `hummingbot/hummingbot` | 18.0k | Crypto-native exchange connector depth, including CEX/DEX connectors | adapter-first exchange layer, execution portability |
| `polakowo/vectorbt` | 7.1k | Extremely fast parameter exploration and large-scale signal testing | alpha factory / experiment lab for rapid candidate generation |

## Fusion Principles

1. Use `vectorbt` ideas for large-batch alpha search, not for live execution.
2. Use `freqtrade` ideas for dry-run, candidate scoring, and paper validation discipline.
3. Use `NautilusTrader` ideas for deterministic event flow and research/live parity.
4. Use `LEAN` ideas for splitting alpha generation, validation, promotion, and allocation into distinct layers.
5. Use `Hummingbot` ideas for exchange portability and order-routing adapters.

## What This Scaffold Explicitly Avoids

- No source-code copy/paste from GPL/LGPL/Apache/MIT projects.
- No claim that parameter self-tuning is equal to alpha discovery.
- No reuse of the current `CryptoAI` conservative single-strategy runtime as the primary control loop for this directory.
