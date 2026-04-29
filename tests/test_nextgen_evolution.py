import unittest
from dataclasses import replace

from nextgen_evolution import (
    AutonomyDirective,
    EvolutionConfig,
    NextGenEvolutionEngine,
    PortfolioAllocator,
    StrategyPrimitive,
    ValidationPipeline,
)
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.models import PromotionStage, ScoreCard, StrategyGenome, ValidationMetrics
from nextgen_evolution.models import RuntimeEvidenceSnapshot
from nextgen_evolution.models import RuntimeLifecycleState, RuntimeState
from nextgen_evolution.models import ExecutionAction, ExecutionDirective
from nextgen_evolution.runtime_override_policy import (
    build_repair_reentry_notes,
    compose_runtime_policy_notes,
)
from datetime import datetime, timedelta, timezone


class NextGenEvolutionTests(unittest.TestCase):
    def test_nextgen_evolution_generates_seed_and_mutations(self):
        engine = NextGenEvolutionEngine(
            EvolutionConfig(experiment_budget=4, mutation_per_seed=2)
        )
        primitives = [
            StrategyPrimitive(
                family="microtrend_breakout",
                base_params={"lookback": 20.0, "vol_filter": 1.2},
            )
        ]

        population = engine.propose_population(primitives)

        self.assertEqual(len(population), 3)
        self.assertEqual(population[0].strategy_id, "microtrend_breakout:seed")
        self.assertEqual(population[1].mutation_of, "microtrend_breakout:seed")
        self.assertEqual(population[2].mutation_of, "microtrend_breakout:seed")

    def test_nextgen_evolution_promotes_and_allocates_strong_candidates(self):
        engine = NextGenEvolutionEngine()
        primitives = [
            StrategyPrimitive(
                family="microtrend_breakout",
                base_params={"lookback": 20.0, "vol_filter": 1.2, "stop": 0.6},
            ),
            StrategyPrimitive(
                family="mean_revert_imbalance",
                base_params={"zscore": 2.0, "inventory_bias": 0.4, "stop": 0.5},
            ),
        ]
        metrics = {
            "microtrend_breakout:seed": ValidationMetrics(
                backtest_expectancy=0.55,
                walkforward_expectancy=0.42,
                shadow_expectancy=0.31,
                live_expectancy=0.18,
                max_drawdown_pct=5.0,
                trade_count=150,
                cost_drag_pct=0.12,
                latency_ms=70.0,
                regime_consistency=0.82,
            ),
            "mean_revert_imbalance:seed": ValidationMetrics(
                backtest_expectancy=0.28,
                walkforward_expectancy=0.19,
                shadow_expectancy=0.11,
                live_expectancy=0.02,
                max_drawdown_pct=8.0,
                trade_count=120,
                cost_drag_pct=0.15,
                latency_ms=60.0,
                regime_consistency=0.69,
            ),
        }

        promoted, allocations = engine.build_deployment_plan(
            metrics_by_strategy=metrics,
            primitives=primitives,
            total_capital=10000.0,
        )

        self.assertTrue(promoted)
        self.assertIn(promoted[0].stage, {PromotionStage.PAPER, PromotionStage.LIVE})
        self.assertTrue(allocations)
        self.assertGreater(allocations[0].allocated_capital, 0)

    def test_nextgen_evolution_rejects_high_drawdown_candidate(self):
        engine = NextGenEvolutionEngine()
        primitives = [
            StrategyPrimitive(
                family="fragile_momentum",
                base_params={"lookback": 12.0, "stop": 0.4},
            )
        ]
        metrics = {
            "fragile_momentum:seed": ValidationMetrics(
                backtest_expectancy=0.60,
                walkforward_expectancy=0.48,
                shadow_expectancy=0.20,
                live_expectancy=0.10,
                max_drawdown_pct=25.0,
                trade_count=100,
                cost_drag_pct=0.12,
                latency_ms=50.0,
                regime_consistency=0.80,
            )
        }

        promoted, allocations = engine.build_deployment_plan(
            metrics_by_strategy=metrics,
            primitives=primitives,
            total_capital=5000.0,
        )

        self.assertEqual(promoted, [])
        self.assertEqual(allocations, [])

    def test_nextgen_evolution_allocates_only_one_variant_per_lineage(self):
        engine = NextGenEvolutionEngine(
            EvolutionConfig(
                experiment_budget=3,
                mutation_per_seed=2,
                max_allocations_per_lineage=1,
            )
        )
        primitives = [
            StrategyPrimitive(
                family="volatility_reclaim",
                base_params={"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
            )
        ]
        metrics = {
            "volatility_reclaim:seed": ValidationMetrics(
                backtest_expectancy=0.48,
                walkforward_expectancy=0.36,
                shadow_expectancy=0.24,
                live_expectancy=0.12,
                max_drawdown_pct=5.0,
                trade_count=48,
                cost_drag_pct=0.10,
                latency_ms=50.0,
                regime_consistency=0.82,
            ),
            "volatility_reclaim:mut1": ValidationMetrics(
                backtest_expectancy=0.46,
                walkforward_expectancy=0.35,
                shadow_expectancy=0.23,
                live_expectancy=0.11,
                max_drawdown_pct=5.3,
                trade_count=47,
                cost_drag_pct=0.11,
                latency_ms=50.0,
                regime_consistency=0.80,
            ),
            "volatility_reclaim:mut2": ValidationMetrics(
                backtest_expectancy=0.45,
                walkforward_expectancy=0.34,
                shadow_expectancy=0.22,
                live_expectancy=0.11,
                max_drawdown_pct=5.5,
                trade_count=46,
                cost_drag_pct=0.11,
                latency_ms=50.0,
                regime_consistency=0.79,
            ),
        }

        promoted, allocations = engine.build_deployment_plan(
            metrics_by_strategy=metrics,
            primitives=primitives,
            total_capital=10000.0,
        )

        self.assertEqual(len(promoted), 3)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].strategy_id, "volatility_reclaim:seed")
        self.assertIn("lineage_diversified", allocations[0].reasons)

    def test_validation_penalizes_sample_count_gradually(self):
        pipeline = ValidationPipeline(EvolutionConfig(min_trade_count=24))
        genome = StrategyPrimitive(
            family="trend_pullback_continuation",
            base_params={"hold_bars": 6.0},
        )
        candidate = NextGenEvolutionEngine().propose_population([genome])[0]
        base_metrics = ValidationMetrics(
            backtest_expectancy=0.52,
            walkforward_expectancy=0.41,
            shadow_expectancy=0.33,
            live_expectancy=0.24,
            max_drawdown_pct=4.0,
            trade_count=0,
            cost_drag_pct=0.11,
            latency_ms=40.0,
            regime_consistency=0.84,
        )

        low_sample = pipeline.score(
            candidate,
            replace(base_metrics, trade_count=6),
        )
        mid_sample = pipeline.score(
            candidate,
            replace(base_metrics, trade_count=18),
        )
        full_sample = pipeline.score(
            candidate,
            replace(base_metrics, trade_count=24),
        )

        self.assertIn("insufficient_trade_count", low_sample.reasons)
        self.assertIn("insufficient_trade_count", mid_sample.reasons)
        self.assertNotIn("insufficient_trade_count", full_sample.reasons)
        self.assertLess(low_sample.total_score, mid_sample.total_score)
        self.assertLess(mid_sample.total_score, full_sample.total_score)

    def test_validation_rewards_high_frequency_candidates_when_cost_adjusted_edge_is_healthy(self):
        pipeline = ValidationPipeline(
            EvolutionConfig(
                min_trade_count=24,
                high_frequency_trade_count_target=96,
                high_frequency_deployment_reward_cap=0.10,
            )
        )
        genome = NextGenEvolutionEngine().propose_population(
            [
                StrategyPrimitive(
                    family="microtrend_breakout",
                    base_params={"lookback": 18.0, "hold_bars": 6.0},
                )
            ]
        )[0]
        base_metrics = ValidationMetrics(
            backtest_expectancy=0.30,
            walkforward_expectancy=0.24,
            shadow_expectancy=0.16,
            live_expectancy=0.12,
            max_drawdown_pct=4.0,
            trade_count=24,
            cost_drag_pct=0.10,
            latency_ms=40.0,
            regime_consistency=0.82,
        )

        baseline = pipeline.score(genome, base_metrics)
        high_frequency = pipeline.score(
            genome,
            replace(base_metrics, trade_count=96),
        )

        self.assertNotIn("high_frequency_deployment_bonus", baseline.reasons)
        self.assertIn("high_frequency_deployment_bonus", high_frequency.reasons)
        self.assertGreater(high_frequency.deployment_score, baseline.deployment_score)
        self.assertGreater(high_frequency.total_score, baseline.total_score)

    def test_validation_does_not_reward_high_frequency_candidates_when_cost_drag_is_too_high(self):
        pipeline = ValidationPipeline(
            EvolutionConfig(
                min_trade_count=24,
                max_cost_drag_pct=0.35,
                high_frequency_trade_count_target=96,
                high_frequency_deployment_reward_cap=0.10,
            )
        )
        genome = NextGenEvolutionEngine().propose_population(
            [
                StrategyPrimitive(
                    family="microtrend_breakout",
                    base_params={"lookback": 18.0, "hold_bars": 6.0},
                )
            ]
        )[0]
        healthy_metrics = ValidationMetrics(
            backtest_expectancy=0.30,
            walkforward_expectancy=0.24,
            shadow_expectancy=0.16,
            live_expectancy=0.12,
            max_drawdown_pct=4.0,
            trade_count=96,
            cost_drag_pct=0.10,
            latency_ms=40.0,
            regime_consistency=0.82,
        )

        healthy = pipeline.score(genome, healthy_metrics)
        expensive = pipeline.score(
            genome,
            replace(healthy_metrics, cost_drag_pct=0.30),
        )

        self.assertIn("high_frequency_deployment_bonus", healthy.reasons)
        self.assertNotIn("high_frequency_deployment_bonus", expensive.reasons)
        self.assertLess(expensive.deployment_score, healthy.deployment_score)
        self.assertLess(expensive.total_score, healthy.total_score)

    def test_portfolio_allocator_diversifies_across_symbols_and_families(self):
        allocator = PortfolioAllocator(
            EvolutionConfig(
                max_allocations_per_lineage=1,
                max_portfolio_symbol_weight=0.40,
                max_portfolio_family_weight=0.45,
                max_portfolio_positions=4,
            )
        )
        eth_vol = ScoreCard(
            genome=StrategyGenome("volatility_reclaim:seed", "volatility_reclaim", {}),
            stage=PromotionStage.PAPER,
            edge_score=0.4,
            robustness_score=0.8,
            deployment_score=0.5,
            total_score=0.42,
            reasons=["promote_paper"],
        )
        eth_trend = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation:seed",
                "trend_pullback_continuation",
                {},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.38,
            robustness_score=0.8,
            deployment_score=0.48,
            total_score=0.36,
            reasons=["promote_paper"],
        )
        sol_vol = ScoreCard(
            genome=StrategyGenome("volatility_reclaim:seed", "volatility_reclaim", {}),
            stage=PromotionStage.LIVE,
            edge_score=0.62,
            robustness_score=0.82,
            deployment_score=0.67,
            total_score=0.58,
            reasons=["promote_live"],
        )
        results = [
            ExperimentResult(
                symbol="ETH/USDT:USDT",
                timeframe="5m",
                scorecards=[eth_vol, eth_trend],
                promoted=[eth_vol, eth_trend],
                allocations=[],
                candle_count=300,
            ),
            ExperimentResult(
                symbol="SOL/USDT:USDT",
                timeframe="5m",
                scorecards=[sol_vol],
                promoted=[sol_vol],
                allocations=[],
                candle_count=300,
            ),
        ]

        allocations = allocator.allocate(results, total_capital=10000.0)

        self.assertEqual(len(allocations), 3)
        self.assertEqual(allocations[0].symbol, "SOL/USDT:USDT")
        self.assertLessEqual(
            sum(item.allocated_capital for item in allocations if item.symbol == "ETH/USDT:USDT"),
            4000.0,
        )
        self.assertLessEqual(
            sum(item.allocated_capital for item in allocations if item.family == "volatility_reclaim"),
            4500.0,
        )

    def test_portfolio_allocator_consumes_runtime_weight_override_into_allocated_capital(self):
        allocator = PortfolioAllocator(EvolutionConfig())
        card = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                "trend_pullback_continuation",
                {},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.42,
            robustness_score=0.8,
            deployment_score=0.42,
            total_score=0.42,
            reasons=["repair_reentry_candidate"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[card],
            promoted=[card],
            allocations=[],
            candle_count=300,
        )
        previous_state = RuntimeState(
            runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.42,
            allocated_capital=800.0,
            desired_capital=800.0,
            current_capital=0.0,
            current_weight=0.08,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                }
            ),
        )

        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].allocated_capital, 1100.0)
        self.assertEqual(allocations[0].weight, 0.11)
        self.assertIn(
            PortfolioAllocator.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON,
            allocations[0].reasons,
        )

    def test_portfolio_allocator_uses_current_runtime_recovery_to_relax_weight_override(self):
        allocator = PortfolioAllocator(
            EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            )
        )
        runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation@BTC_USDT_USDT_5m:repair"
        card = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                "trend_pullback_continuation",
                {},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.42,
            robustness_score=0.8,
            deployment_score=0.42,
            total_score=0.42,
            reasons=["repair_reentry_candidate"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[card],
            promoted=[card],
            allocations=[],
            candle_count=300,
        )
        previous_state = RuntimeState(
            runtime_id=runtime_id,
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.42,
            allocated_capital=800.0,
            desired_capital=800.0,
            current_capital=0.0,
            current_weight=0.08,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                },
                runtime_override_state={
                    "cycles_since_refresh": 0,
                },
            ),
        )
        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            previous_states=[previous_state],
            runtime_evidence={
                runtime_id: RuntimeEvidenceSnapshot(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    open_position=True,
                    current_capital=550.0,
                    unrealized_pnl=50.0,
                    total_net_pnl=50.0,
                    current_drawdown_pct=0.0,
                    closed_trade_count=0,
                    win_rate=0.0,
                    consecutive_losses=0,
                    health_status="active",
                )
            },
        )

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].allocated_capital, 1325.0)
        self.assertEqual(allocations[0].weight, 0.1325)

    def test_portfolio_allocator_freezes_hold_recovery_runtime_at_previous_risk_budget(self):
        allocator = PortfolioAllocator(EvolutionConfig())
        runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation@BTC_USDT_USDT_5m:repair"
        card = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                "trend_pullback_continuation",
                {},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.42,
            robustness_score=0.8,
            deployment_score=0.42,
            total_score=0.42,
            reasons=["repair_reentry_candidate"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[card],
            promoted=[card],
            allocations=[],
            candle_count=300,
        )
        previous_state = RuntimeState(
            runtime_id=runtime_id,
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.42,
            allocated_capital=1000.0,
            desired_capital=400.0,
            current_capital=0.0,
            current_weight=0.10,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                }
            ),
        )
        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            previous_states=[previous_state],
            runtime_evidence={
                runtime_id: RuntimeEvidenceSnapshot(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    open_position=False,
                    current_capital=0.0,
                    realized_pnl=-20.0,
                    unrealized_pnl=0.0,
                    total_net_pnl=-20.0,
                    current_drawdown_pct=6.0,
                    closed_trade_count=2,
                    win_rate=0.0,
                    consecutive_losses=2,
                    health_status="degraded",
                )
            },
        )

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].allocated_capital, 400.0)
        self.assertEqual(allocations[0].weight, 0.04)
        self.assertIn(
            PortfolioAllocator.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON,
            allocations[0].reasons,
        )
        self.assertIn(
            PortfolioAllocator.RUNTIME_OVERRIDE_HOLD_CAP_REASON,
            allocations[0].reasons,
        )

    def test_portfolio_allocator_holds_capital_in_reserve_during_active_reentry_cooldown(self):
        allocator = PortfolioAllocator(EvolutionConfig())
        runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation@BTC_USDT_USDT_5m:repair"
        source_runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation:seed"
        card = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                "trend_pullback_continuation",
                {},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.42,
            robustness_score=0.8,
            deployment_score=0.42,
            total_score=0.42,
            reasons=["repair_reentry_candidate"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[card],
            promoted=[card],
            allocations=[],
            candle_count=300,
            notes=compose_runtime_policy_notes(
                repair_reentry_notes=build_repair_reentry_notes(
                    source_runtime_id=source_runtime_id,
                    runtime_overrides={
                        "entry_cooldown_bars_multiplier": 2.0,
                    },
                )
            ),
        )

        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            latest_close_time_by_runtime={
                source_runtime_id: datetime.now(timezone.utc) - timedelta(minutes=5),
            },
        )

        self.assertEqual(allocations, [])

    def test_portfolio_allocator_consumes_profit_lock_staged_exit_into_allocated_capital(self):
        allocator = PortfolioAllocator(EvolutionConfig())
        runtime_id = "BTC/USDT:USDT|5m|profit_lock_runtime:seed"
        card = ScoreCard(
            genome=StrategyGenome(
                "profit_lock_runtime:seed",
                "profit_lock_runtime",
                {},
            ),
            stage=PromotionStage.LIVE,
            edge_score=0.58,
            robustness_score=0.8,
            deployment_score=0.58,
            total_score=0.57,
            reasons=["promote_live"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[card],
            promoted=[card],
            allocations=[],
            candle_count=300,
        )

        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.SCALE_DOWN,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=0.75,
                        reasons=["profit_lock_harvest"],
                    )
                ]
            ),
        )

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].allocated_capital, 2625.0)
        self.assertEqual(allocations[0].weight, 0.2625)
        self.assertIn(
            PortfolioAllocator.STAGED_EXIT_CAP_REASON,
            allocations[0].reasons,
        )

    def test_portfolio_allocator_reenters_gradually_from_previous_profit_lock_staged_exit(self):
        allocator = PortfolioAllocator(EvolutionConfig())
        runtime_id = "BTC/USDT:USDT|5m|profit_lock_runtime:seed"
        card = ScoreCard(
            genome=StrategyGenome(
                "profit_lock_runtime:seed",
                "profit_lock_runtime",
                {},
            ),
            stage=PromotionStage.LIVE,
            edge_score=0.58,
            robustness_score=0.8,
            deployment_score=0.58,
            total_score=0.57,
            reasons=["promote_live"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[card],
            promoted=[card],
            allocations=[],
            candle_count=300,
        )
        previous_state = RuntimeState(
            runtime_id=runtime_id,
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="profit_lock_runtime:seed",
            family="profit_lock_runtime",
            lifecycle_state=RuntimeLifecycleState.LIVE,
            promotion_stage=PromotionStage.LIVE,
            target_stage=PromotionStage.LIVE,
            last_directive_action=ExecutionAction.KEEP,
            score=0.57,
            allocated_capital=1400.0,
            desired_capital=1400.0,
            current_capital=1400.0,
            current_weight=0.14,
            notes=compose_runtime_policy_notes(
                staged_exit_state={
                    "mode": "profit_lock",
                    "phase": "deep_harvest",
                    "target_multiplier": 0.4,
                    "trigger_count": 2,
                    "recovery_count": 0,
                    "last_reason": "profit_lock_harvest",
                }
            ),
        )

        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            previous_states=[previous_state],
            runtime_evidence={
                runtime_id: RuntimeEvidenceSnapshot(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    open_position=True,
                    current_capital=700.0,
                    realized_pnl=0.0,
                    unrealized_pnl=50.0,
                    total_net_pnl=50.0,
                    current_drawdown_pct=0.0,
                    max_drawdown_pct=0.0,
                    closed_trade_count=0,
                    win_rate=0.0,
                    consecutive_losses=0,
                    health_status="active",
                )
            },
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        reasons=["live_strategy_stable"],
                    )
                ]
            ),
        )

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].allocated_capital, 2450.0)
        self.assertEqual(allocations[0].weight, 0.245)
        self.assertIn(
            PortfolioAllocator.STAGED_EXIT_CAP_REASON,
            allocations[0].reasons,
        )


if __name__ == "__main__":
    unittest.main()
