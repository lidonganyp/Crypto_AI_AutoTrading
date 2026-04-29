"""Configuration for the next-generation evolution scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvolutionConfig:
    experiment_budget: int = 24
    mutation_per_seed: int = 3
    mutation_scale: float = 0.12
    min_trade_count: int = 24
    max_drawdown_pct: float = 12.0
    max_cost_drag_pct: float = 0.35
    max_latency_ms: float = 250.0
    min_regime_consistency: float = 0.55
    high_frequency_trade_count_target: int = 96
    high_frequency_deployment_reward_cap: float = 0.10
    shadow_threshold: float = 0.18
    paper_threshold: float = 0.32
    live_threshold: float = 0.50
    max_live_weight: float = 0.35
    max_paper_weight: float = 0.20
    max_allocations_per_lineage: int = 1
    max_portfolio_symbol_weight: float = 0.40
    max_portfolio_family_weight: float = 0.45
    max_portfolio_positions: int = 6
    stale_snapshot_after_minutes: int = 10
    health_max_drawdown_pct: float = 8.0
    health_min_equity_ratio: float = 0.96
    health_max_gross_exposure_ratio: float = 0.80
    autonomy_repair_drawdown_pct: float = 8.0
    autonomy_repair_expectancy_floor: float = 0.0
    autonomy_min_runtime_trades: int = 24
    autonomy_max_repair_queue: int = 4
    autonomy_repair_retire_after_failures: int = 2
    autonomy_repair_promote_after_successes: int = 2
    autonomy_repair_history_limit: int = 40
    autonomy_overstay_soft_multiplier: float = 1.25
    autonomy_overstay_hard_multiplier: float = 2.0
    autonomy_profit_lock_min_return_pct: float = 3.0
    autonomy_profit_lock_soft_retrace_pct: float = 35.0
    autonomy_profit_lock_hard_retrace_pct: float = 65.0
    autonomy_profit_lock_soft_scale_down_factor: float = 0.75
    autonomy_profit_lock_deep_scale_down_factor: float = 0.40
    autonomy_live_scale_down_factor: float = 0.50
    autonomy_live_scale_up_factor: float = 1.15
    autonomy_scale_up_score_bonus: float = 0.12
    autonomy_runtime_override_decay_rate: float = 0.25
    autonomy_runtime_override_cooldown_decay_step: float = 0.5
    autonomy_limited_live_cycles: int = 2
    autonomy_limited_live_max_weight: float = 0.05
    autonomy_live_blast_radius_capital_pct: float = 0.10
    autonomy_live_enabled: bool = False
    autonomy_live_whitelist: tuple[str, ...] = ()
    autonomy_live_max_active_runtimes: int = 1
    autonomy_live_require_explicit_enable: bool = True
