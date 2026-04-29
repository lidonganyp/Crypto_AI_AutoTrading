import unittest
from dataclasses import replace

from nextgen_evolution import (
    AutonomousDirector,
    AutonomyPlanner,
    EvolutionConfig,
    ExecutionAction,
    RepairActionType,
    RepairFeedbackSummary,
    StrategyRepairEngine,
)
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.models import (
    CapitalAllocation,
    RuntimeLifecycleState,
    RuntimeState,
    PromotionStage,
    ScoreCard,
    StrategyGenome,
    StrategyRuntimeSnapshot,
    ValidationMetrics,
)
from nextgen_evolution.runtime_override_policy import (
    build_repair_reentry_notes,
    compose_runtime_policy_notes,
)


class NextGenAutonomyTests(unittest.TestCase):
    def test_planner_quarantines_degraded_live_strategy_and_builds_repair(self):
        config = EvolutionConfig(
            live_threshold=0.50,
            autonomy_repair_drawdown_pct=8.0,
            autonomy_repair_expectancy_floor=0.0,
        )
        planner = AutonomyPlanner(config)
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    {"lookback": 18.0, "hold_bars": 6.0, "stop": 0.5},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.55,
                robustness_score=0.70,
                deployment_score=0.58,
                total_score=0.56,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.40,
                walkforward_expectancy=-0.05,
                shadow_expectancy=0.04,
                live_expectancy=-0.08,
                max_drawdown_pct=11.0,
                trade_count=48,
                cost_drag_pct=0.18,
                latency_ms=45.0,
                regime_consistency=0.72,
            ),
            allocated_capital=5000.0,
            current_weight=0.35,
            realized_pnl=-320.0,
            current_drawdown_pct=11.5,
            consecutive_losses=3,
            health_status="degraded",
        )

        directive = planner.plan([snapshot])

        self.assertEqual(len(directive.execution), 1)
        self.assertEqual(directive.execution[0].action, ExecutionAction.EXIT)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.SHADOW)
        self.assertEqual(directive.quarantined, ["trend_pullback_continuation:seed"])
        self.assertEqual(len(directive.repairs), 1)
        self.assertEqual(directive.repairs[0].action, RepairActionType.QUARANTINE)
        self.assertLess(
            directive.repairs[0].candidate_genome.params["hold_bars"],
            snapshot.scorecard.genome.params["hold_bars"],
        )

    def test_planner_tracks_repairs_dropped_by_queue_limit(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                autonomy_repair_drawdown_pct=8.0,
                autonomy_repair_expectancy_floor=0.0,
                autonomy_max_repair_queue=2,
            )
        )
        snapshots = []
        for strategy_id in (
            "runtime_a:seed",
            "runtime_b:seed",
            "runtime_c:seed",
        ):
            snapshots.append(
                StrategyRuntimeSnapshot(
                    scorecard=ScoreCard(
                        genome=StrategyGenome(
                            strategy_id,
                            "trend_pullback_continuation",
                            {"lookback": 18.0, "hold_bars": 6.0, "stop": 0.5},
                        ),
                        stage=PromotionStage.LIVE,
                        edge_score=0.55,
                        robustness_score=0.70,
                        deployment_score=0.58,
                        total_score=0.56,
                        reasons=["promote_live"],
                    ),
                    metrics=ValidationMetrics(
                        backtest_expectancy=0.40,
                        walkforward_expectancy=-0.05,
                        shadow_expectancy=0.04,
                        live_expectancy=-0.08,
                        max_drawdown_pct=11.0,
                        trade_count=48,
                        cost_drag_pct=0.18,
                        latency_ms=45.0,
                        regime_consistency=0.72,
                    ),
                    allocated_capital=5000.0,
                    current_weight=0.35,
                    realized_pnl=-320.0,
                    current_drawdown_pct=11.5,
                    consecutive_losses=3,
                    health_status="degraded",
                )
            )

        directive = planner.plan(snapshots)

        self.assertEqual(len(directive.execution), 3)
        self.assertEqual(len(directive.repairs), 2)
        self.assertEqual(directive.notes["repair_queue_requested_size"], 3)
        self.assertEqual(directive.notes["repair_queue_size"], 2)
        self.assertEqual(directive.notes["repair_queue_dropped_count"], 1)
        self.assertEqual(len(directive.notes["repair_queue_dropped_runtime_ids"]), 1)
        self.assertIn(
            directive.notes["repair_queue_dropped_runtime_ids"][0],
            {"runtime_a:seed", "runtime_b:seed", "runtime_c:seed"},
        )

    def test_planner_scales_up_healthy_live_strategy(self):
        config = EvolutionConfig(
            live_threshold=0.50,
            autonomy_scale_up_score_bonus=0.10,
            autonomy_repair_expectancy_floor=0.0,
        )
        planner = AutonomyPlanner(config)
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "volatility_reclaim:seed",
                    "volatility_reclaim",
                    {"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.65,
                robustness_score=0.82,
                deployment_score=0.71,
                total_score=0.64,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.56,
                walkforward_expectancy=0.41,
                shadow_expectancy=0.31,
                live_expectancy=0.22,
                max_drawdown_pct=4.0,
                trade_count=64,
                cost_drag_pct=0.11,
                latency_ms=55.0,
                regime_consistency=0.84,
            ),
            allocated_capital=4000.0,
            current_weight=0.25,
            realized_pnl=420.0,
            current_drawdown_pct=2.5,
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_UP)
        self.assertGreater(
            directive.execution[0].capital_multiplier,
            1.0,
        )
        self.assertEqual(directive.repairs, [])

    def test_planner_blocks_live_scale_up_while_runtime_override_hold_is_active(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_scale_up_score_bonus=0.10,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "volatility_reclaim:seed",
                    "volatility_reclaim",
                    {"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.65,
                robustness_score=0.82,
                deployment_score=0.71,
                total_score=0.64,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.56,
                walkforward_expectancy=0.41,
                shadow_expectancy=0.31,
                live_expectancy=0.22,
                max_drawdown_pct=4.0,
                trade_count=64,
                cost_drag_pct=0.11,
                latency_ms=55.0,
                regime_consistency=0.84,
            ),
            allocated_capital=4000.0,
            current_weight=0.25,
            realized_pnl=420.0,
            current_drawdown_pct=2.5,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                },
                runtime_override_state={
                    "recovery_mode": "hold",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.KEEP)
        self.assertIn("runtime_override_hold_blocks_expansion", directive.execution[0].reasons)
        self.assertIn("runtime_override_recovery_mode:hold", directive.execution[0].reasons)

    def test_planner_accelerates_live_scale_up_during_reentry_recovery(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_scale_up_score_bonus=0.10,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "volatility_reclaim:seed",
                    "volatility_reclaim",
                    {"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.65,
                robustness_score=0.82,
                deployment_score=0.71,
                total_score=0.64,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.56,
                walkforward_expectancy=0.41,
                shadow_expectancy=0.31,
                live_expectancy=0.22,
                max_drawdown_pct=4.0,
                trade_count=64,
                cost_drag_pct=0.11,
                latency_ms=55.0,
                regime_consistency=0.84,
            ),
            allocated_capital=4000.0,
            current_weight=0.25,
            realized_pnl=420.0,
            current_drawdown_pct=2.5,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.7,
                },
                runtime_override_state={
                    "recovery_mode": "accelerate",
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "recovery",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_UP)
        self.assertIn("repair_reentry_recovery_accelerated", directive.execution[0].reasons)
        self.assertIn("runtime_override_accelerate_allows_scale_up", directive.execution[0].reasons)
        self.assertIn("runtime_override_recovery_mode:accelerate", directive.execution[0].reasons)

    def test_planner_uses_soft_profit_lock_scale_down_factor(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                autonomy_repair_expectancy_floor=0.0,
                autonomy_profit_lock_min_return_pct=5.0,
                autonomy_profit_lock_soft_retrace_pct=35.0,
                autonomy_profit_lock_hard_retrace_pct=80.0,
                autonomy_profit_lock_soft_scale_down_factor=0.75,
                autonomy_profit_lock_deep_scale_down_factor=0.40,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.58,
                robustness_score=0.78,
                deployment_score=0.61,
                total_score=0.57,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.42,
                walkforward_expectancy=0.28,
                shadow_expectancy=0.18,
                live_expectancy=0.09,
                max_drawdown_pct=4.0,
                trade_count=36,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.81,
            ),
            allocated_capital=1000.0,
            realized_pnl=0.0,
            unrealized_pnl=60.0,
            current_drawdown_pct=1.0,
            notes={
                "peak_unrealized_return_pct": 10.0,
                "profit_retrace_pct": 40.0,
            },
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_DOWN)
        self.assertIn("profit_lock_harvest", directive.execution[0].reasons)
        self.assertEqual(directive.execution[0].capital_multiplier, 0.75)

    def test_planner_uses_deep_profit_lock_scale_down_factor(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                autonomy_repair_expectancy_floor=0.0,
                autonomy_profit_lock_min_return_pct=5.0,
                autonomy_profit_lock_soft_retrace_pct=35.0,
                autonomy_profit_lock_hard_retrace_pct=80.0,
                autonomy_profit_lock_soft_scale_down_factor=0.75,
                autonomy_profit_lock_deep_scale_down_factor=0.40,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "trend_exhaustion:seed",
                    "trend_exhaustion",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.58,
                robustness_score=0.78,
                deployment_score=0.61,
                total_score=0.57,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.42,
                walkforward_expectancy=0.28,
                shadow_expectancy=0.18,
                live_expectancy=0.09,
                max_drawdown_pct=4.0,
                trade_count=36,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.81,
            ),
            allocated_capital=1000.0,
            realized_pnl=0.0,
            unrealized_pnl=30.0,
            current_drawdown_pct=1.0,
            notes={
                "peak_unrealized_return_pct": 10.0,
                "profit_retrace_pct": 65.0,
            },
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_DOWN)
        self.assertIn("profit_lock_harvest", directive.execution[0].reasons)
        self.assertEqual(directive.execution[0].capital_multiplier, 0.40)

    def test_planner_keeps_profit_lock_staged_exit_active_without_fresh_retrace_signal(self):
        planner = AutonomyPlanner(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.58,
                robustness_score=0.78,
                deployment_score=0.61,
                total_score=0.57,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.42,
                walkforward_expectancy=0.28,
                shadow_expectancy=0.18,
                live_expectancy=0.09,
                max_drawdown_pct=4.0,
                trade_count=36,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.81,
            ),
            allocated_capital=1000.0,
            realized_pnl=0.0,
            unrealized_pnl=20.0,
            current_drawdown_pct=1.0,
            notes=compose_runtime_policy_notes(
                staged_exit_state={
                    "mode": "profit_lock",
                    "phase": "harvest",
                    "target_multiplier": 0.75,
                    "trigger_count": 1,
                    "recovery_count": 0,
                    "last_reason": "profit_lock_harvest",
                }
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_DOWN)
        self.assertEqual(directive.execution[0].capital_multiplier, 0.75)
        self.assertIn("profit_lock_staged_exit_active", directive.execution[0].reasons)
        self.assertIn("profit_lock_phase:harvest", directive.execution[0].reasons)

    def test_planner_suppresses_live_scale_up_while_profit_lock_reentry_is_active(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_scale_up_score_bonus=0.10,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.66,
                robustness_score=0.82,
                deployment_score=0.71,
                total_score=0.64,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.56,
                walkforward_expectancy=0.41,
                shadow_expectancy=0.31,
                live_expectancy=0.22,
                max_drawdown_pct=4.0,
                trade_count=64,
                cost_drag_pct=0.11,
                latency_ms=55.0,
                regime_consistency=0.84,
            ),
            allocated_capital=4000.0,
            current_weight=0.25,
            realized_pnl=420.0,
            current_drawdown_pct=2.5,
            notes=compose_runtime_policy_notes(
                staged_exit_state={
                    "mode": "profit_lock",
                    "phase": "reentry",
                    "target_multiplier": 0.7,
                    "trigger_count": 2,
                    "recovery_count": 1,
                    "last_reason": "profit_lock_reentry",
                }
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.KEEP)
        self.assertIn("profit_lock_reentry_active", directive.execution[0].reasons)
        self.assertIn("profit_lock_phase:reentry", directive.execution[0].reasons)
        self.assertEqual(directive.repairs, [])

    def test_planner_keeps_paper_reentry_in_probation_before_live_promotion(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "repair_runtime:seed",
                    "repair_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.60,
                robustness_score=0.80,
                deployment_score=0.62,
                total_score=0.61,
                reasons=["repair_reentry_candidate"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.45,
                walkforward_expectancy=0.31,
                shadow_expectancy=0.21,
                live_expectancy=0.15,
                max_drawdown_pct=4.0,
                trade_count=48,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.83,
            ),
            allocated_capital=1000.0,
            realized_pnl=20.0,
            unrealized_pnl=10.0,
            current_drawdown_pct=1.0,
            notes=compose_runtime_policy_notes(
                base_notes={
                    "closed_trade_count": 3,
                    "win_rate": 0.67,
                    "total_net_pnl": 30.0,
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "probation",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.KEEP)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.PAPER)
        self.assertIn("repair_reentry_probation_active", directive.execution[0].reasons)

    def test_planner_promotes_paper_reentry_to_live_after_recovery_successes(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_repair_expectancy_floor=0.0,
                autonomy_repair_promote_after_successes=2,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "repair_runtime:seed",
                    "repair_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.60,
                robustness_score=0.80,
                deployment_score=0.62,
                total_score=0.61,
                reasons=["repair_reentry_candidate"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.45,
                walkforward_expectancy=0.31,
                shadow_expectancy=0.21,
                live_expectancy=0.15,
                max_drawdown_pct=4.0,
                trade_count=48,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.83,
            ),
            allocated_capital=1000.0,
            realized_pnl=20.0,
            unrealized_pnl=10.0,
            current_drawdown_pct=1.0,
            health_status="active",
            notes=compose_runtime_policy_notes(
                base_notes={
                    "closed_trade_count": 2,
                    "win_rate": 0.50,
                    "total_net_pnl": 30.0,
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "recovery",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.PROMOTE_TO_LIVE)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.LIVE)
        self.assertIn("repair_reentry_recovery_complete", directive.execution[0].reasons)

    def test_planner_blocks_paper_reentry_live_promotion_while_runtime_override_hold_is_active(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_repair_expectancy_floor=0.0,
                autonomy_repair_promote_after_successes=2,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "repair_runtime:seed",
                    "repair_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.60,
                robustness_score=0.80,
                deployment_score=0.62,
                total_score=0.61,
                reasons=["repair_reentry_candidate"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.45,
                walkforward_expectancy=0.31,
                shadow_expectancy=0.21,
                live_expectancy=0.15,
                max_drawdown_pct=4.0,
                trade_count=48,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.83,
            ),
            allocated_capital=1000.0,
            realized_pnl=20.0,
            unrealized_pnl=10.0,
            current_drawdown_pct=1.0,
            health_status="active",
            notes=compose_runtime_policy_notes(
                base_notes={
                    "closed_trade_count": 2,
                    "win_rate": 0.50,
                    "total_net_pnl": 30.0,
                },
                repair_reentry_notes=build_repair_reentry_notes(
                    source_runtime_id="BTC/USDT:USDT|5m|repair_runtime:seed",
                ),
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                },
                runtime_override_state={
                    "recovery_mode": "hold",
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "recovery",
                    "source_runtime_id": "BTC/USDT:USDT|5m|repair_runtime:seed",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.KEEP)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.PAPER)
        self.assertIn("runtime_override_hold_blocks_expansion", directive.execution[0].reasons)
        self.assertIn("runtime_override_recovery_mode:hold", directive.execution[0].reasons)
        self.assertNotIn("repair_reentry_recovery_complete", directive.execution[0].reasons)

    def test_planner_accelerates_paper_reentry_live_promotion(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_repair_expectancy_floor=0.0,
                autonomy_repair_promote_after_successes=2,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "repair_runtime:seed",
                    "repair_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.60,
                robustness_score=0.80,
                deployment_score=0.62,
                total_score=0.61,
                reasons=["repair_reentry_candidate"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.45,
                walkforward_expectancy=0.31,
                shadow_expectancy=0.21,
                live_expectancy=0.15,
                max_drawdown_pct=4.0,
                trade_count=48,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.83,
            ),
            allocated_capital=1000.0,
            realized_pnl=20.0,
            unrealized_pnl=10.0,
            current_drawdown_pct=1.0,
            health_status="active",
            notes=compose_runtime_policy_notes(
                base_notes={
                    "closed_trade_count": 1,
                    "win_rate": 1.0,
                    "total_net_pnl": 30.0,
                },
                repair_reentry_notes=build_repair_reentry_notes(
                    source_runtime_id="BTC/USDT:USDT|5m|repair_runtime:seed",
                ),
                runtime_overrides={
                    "max_weight_multiplier": 0.6,
                },
                runtime_override_state={
                    "recovery_mode": "accelerate",
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "recovery",
                    "source_runtime_id": "BTC/USDT:USDT|5m|repair_runtime:seed",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.PROMOTE_TO_LIVE)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.LIVE)
        self.assertIn("repair_reentry_recovery_accelerated", directive.execution[0].reasons)
        self.assertIn("runtime_override_recovery_mode:accelerate", directive.execution[0].reasons)

    def test_planner_release_mode_clears_paper_reentry_observation_gate(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                live_threshold=0.50,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "repair_runtime:seed",
                    "repair_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.60,
                robustness_score=0.80,
                deployment_score=0.62,
                total_score=0.61,
                reasons=["repair_reentry_candidate"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.45,
                walkforward_expectancy=0.31,
                shadow_expectancy=0.21,
                live_expectancy=0.15,
                max_drawdown_pct=4.0,
                trade_count=48,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.83,
            ),
            allocated_capital=1000.0,
            realized_pnl=20.0,
            unrealized_pnl=10.0,
            current_drawdown_pct=1.0,
            health_status="active",
            notes=compose_runtime_policy_notes(
                base_notes={
                    "closed_trade_count": 2,
                    "win_rate": 1.0,
                    "total_net_pnl": 30.0,
                },
                runtime_override_state={
                    "recovery_mode": "release",
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "probation",
                    "release_ready": True,
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.PROMOTE_TO_LIVE)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.LIVE)
        self.assertIn("promotion_requirements_met", directive.execution[0].reasons)
        self.assertIn("runtime_override_recovery_mode:release", directive.execution[0].reasons)
        self.assertNotIn("repair_reentry_probation_active", directive.execution[0].reasons)

    def test_planner_promotes_shadow_reentry_to_paper_after_recovery(self):
        planner = AutonomyPlanner(
            EvolutionConfig(
                paper_threshold=0.32,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "repair_runtime:seed",
                    "repair_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.SHADOW,
                edge_score=0.36,
                robustness_score=0.76,
                deployment_score=0.35,
                total_score=0.37,
                reasons=["repair_reentry_candidate"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.36,
                walkforward_expectancy=0.27,
                shadow_expectancy=0.19,
                live_expectancy=0.08,
                max_drawdown_pct=4.0,
                trade_count=40,
                cost_drag_pct=0.10,
                latency_ms=35.0,
                regime_consistency=0.83,
            ),
            allocated_capital=500.0,
            realized_pnl=10.0,
            unrealized_pnl=0.0,
            current_drawdown_pct=1.0,
            health_status="active",
            notes=compose_runtime_policy_notes(
                base_notes={
                    "total_net_pnl": 10.0,
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "recovery",
                },
            ),
        )

        directive = planner.plan([snapshot])

        self.assertEqual(directive.execution[0].action, ExecutionAction.PROMOTE_TO_PAPER)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.PAPER)
        self.assertIn("repair_reentry_recovery_complete", directive.execution[0].reasons)

    def test_director_carries_previous_profit_lock_staged_exit_state_into_planner(self):
        director = AutonomousDirector(EvolutionConfig())
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=StrategyGenome(
                        "profit_lock_runtime:seed",
                        "profit_lock_runtime",
                        {"lookback": 18.0, "hold_bars": 6.0},
                    ),
                    stage=PromotionStage.LIVE,
                    edge_score=0.58,
                    robustness_score=0.78,
                    deployment_score=0.61,
                    total_score=0.57,
                    reasons=["promote_live"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=200,
            metrics_by_strategy={
                "profit_lock_runtime:seed": ValidationMetrics(
                    backtest_expectancy=0.42,
                    walkforward_expectancy=0.28,
                    shadow_expectancy=0.18,
                    live_expectancy=0.09,
                    max_drawdown_pct=4.0,
                    trade_count=36,
                    cost_drag_pct=0.10,
                    latency_ms=35.0,
                    regime_consistency=0.81,
                )
            },
        )

        directive = director.plan_from_experiments(
            [result],
            previous_states=[
                RuntimeState(
                    runtime_id="BTC/USDT:USDT|5m|profit_lock_runtime:seed",
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=1000.0,
                    desired_capital=750.0,
                    current_capital=750.0,
                    current_weight=0.075,
                    notes=compose_runtime_policy_notes(
                        staged_exit_state={
                            "mode": "profit_lock",
                            "phase": "harvest",
                            "target_multiplier": 0.75,
                            "trigger_count": 1,
                            "recovery_count": 0,
                            "last_reason": "profit_lock_harvest",
                        }
                    ),
                )
            ],
        )

        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_DOWN)
        self.assertEqual(directive.execution[0].capital_multiplier, 0.75)
        self.assertIn("profit_lock_staged_exit_active", directive.execution[0].reasons)

        snapshots = director.build_runtime_snapshots(
            [result],
            previous_states=[
                RuntimeState(
                    runtime_id="BTC/USDT:USDT|5m|profit_lock_runtime:seed",
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=1000.0,
                    desired_capital=750.0,
                    current_capital=750.0,
                    current_weight=0.075,
                    notes=compose_runtime_policy_notes(
                        staged_exit_state={
                            "mode": "profit_lock",
                            "phase": "harvest",
                            "target_multiplier": 0.75,
                            "trigger_count": 1,
                            "recovery_count": 0,
                            "last_reason": "profit_lock_harvest",
                        }
                    ),
                )
            ],
        )
        self.assertEqual(
            snapshots[0].notes["runtime_lifecycle_policy"]["staged_exit"]["phase"],
            "harvest",
        )

    def test_repair_engine_raises_selectivity_for_high_cost_candidate(self):
        config = EvolutionConfig(max_cost_drag_pct=0.35)
        repair_engine = StrategyRepairEngine(config)
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "microtrend_breakout:seed",
                    "microtrend_breakout",
                    {
                        "lookback": 18.0,
                        "breakout_buffer": 0.002,
                        "momentum_floor": 0.001,
                        "hold_bars": 6.0,
                    },
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.22,
                robustness_score=0.55,
                deployment_score=0.20,
                total_score=0.24,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.18,
                walkforward_expectancy=0.05,
                shadow_expectancy=0.04,
                live_expectancy=0.01,
                max_drawdown_pct=6.0,
                trade_count=32,
                cost_drag_pct=0.28,
                latency_ms=90.0,
                regime_consistency=0.76,
            ),
        )

        repair = repair_engine.propose(snapshot)

        self.assertIn(repair.action, {RepairActionType.RAISE_SELECTIVITY, RepairActionType.MUTATE_AND_REVALIDATE})
        self.assertGreater(
            repair.candidate_genome.params["breakout_buffer"],
            snapshot.scorecard.genome.params["breakout_buffer"],
        )
        self.assertGreaterEqual(
            repair.candidate_genome.params["momentum_floor"],
            snapshot.scorecard.genome.params["momentum_floor"],
        )

    def test_repair_engine_retires_lineage_after_repeated_failed_repairs(self):
        repair_engine = StrategyRepairEngine(
            EvolutionConfig(autonomy_repair_retire_after_failures=2)
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "microtrend_breakout:seed",
                    "microtrend_breakout",
                    {"lookback": 18.0, "breakout_buffer": 0.002, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.18,
                robustness_score=0.52,
                deployment_score=0.16,
                total_score=0.20,
                reasons=["promote_shadow"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.15,
                walkforward_expectancy=-0.03,
                shadow_expectancy=0.01,
                live_expectancy=-0.02,
                max_drawdown_pct=8.0,
                trade_count=26,
                cost_drag_pct=0.20,
                latency_ms=45.0,
                regime_consistency=0.70,
            ),
        )

        repair = repair_engine.propose(
            snapshot,
            feedback=RepairFeedbackSummary(
                source_strategy_id="microtrend_breakout:seed",
                source_runtime_id="BTC/USDT:USDT|5m|microtrend_breakout:seed",
                attempts=2,
                failures=2,
                consecutive_failures=2,
                probation_required=True,
                retire_recommended=True,
            ),
        )

        self.assertEqual(repair.action, RepairActionType.RETIRE)
        self.assertIsNone(repair.candidate_genome)
        self.assertEqual(repair.validation_stage, PromotionStage.REJECT)
        self.assertIn("repair_lineage_exhausted", repair.reasons)

    def test_repair_engine_carries_runtime_recovery_mode_into_retire_plan(self):
        repair_engine = StrategyRepairEngine(
            EvolutionConfig(autonomy_repair_retire_after_failures=2)
        )
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "microtrend_breakout:seed",
                    "microtrend_breakout",
                    {"lookback": 18.0, "breakout_buffer": 0.002, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.18,
                robustness_score=0.52,
                deployment_score=0.16,
                total_score=0.20,
                reasons=["promote_shadow"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.15,
                walkforward_expectancy=-0.03,
                shadow_expectancy=0.01,
                live_expectancy=-0.02,
                max_drawdown_pct=8.0,
                trade_count=26,
                cost_drag_pct=0.20,
                latency_ms=45.0,
                regime_consistency=0.70,
            ),
            notes=compose_runtime_policy_notes(
                runtime_override_state={
                    "recovery_mode": "release",
                }
            ),
        )

        repair = repair_engine.propose(
            snapshot,
            feedback=RepairFeedbackSummary(
                source_strategy_id="microtrend_breakout:seed",
                source_runtime_id="BTC/USDT:USDT|5m|microtrend_breakout:seed",
                attempts=2,
                failures=2,
                consecutive_failures=2,
                probation_required=True,
                retire_recommended=True,
            ),
        )

        self.assertEqual(repair.action, RepairActionType.RETIRE)
        self.assertIn("repair_runtime_override_recovery_mode:release", repair.reasons)

    def test_repair_engine_uses_autonomy_outcome_feedback_bias(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    {
                        "lookback": 18.0,
                        "breakout_buffer": 0.002,
                        "momentum_floor": 0.001,
                        "hold_bars": 6.0,
                    },
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.22,
                robustness_score=0.55,
                deployment_score=0.20,
                total_score=0.24,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.18,
                walkforward_expectancy=0.05,
                shadow_expectancy=0.04,
                live_expectancy=0.01,
                max_drawdown_pct=6.0,
                trade_count=32,
                cost_drag_pct=0.10,
                latency_ms=90.0,
                regime_consistency=0.76,
            ),
        )

        repair = repair_engine.propose(
            snapshot,
            feedback=RepairFeedbackSummary(
                source_strategy_id="trend_pullback_continuation:seed",
                source_runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                preferred_action=RepairActionType.RAISE_SELECTIVITY,
                notes={
                    "autonomy_outcomes": {
                        "profit_lock_harvest_count": 2,
                        "profit_lock_net_pnl": 50.0,
                    }
                },
            ),
        )

        self.assertEqual(repair.action, RepairActionType.RAISE_SELECTIVITY)
        self.assertIn("repair_feedback_selectivity_bias", repair.reasons)
        self.assertIn("repair_feedback_profit_lock_harvest_bias", repair.reasons)
        self.assertEqual(repair.runtime_overrides["max_weight_multiplier"], 0.85)
        self.assertEqual(repair.runtime_overrides["entry_cooldown_bars_multiplier"], 1.10)
        self.assertEqual(repair.runtime_overrides["take_profit_bias"], 1.10)
        self.assertLess(
            repair.candidate_genome.params["hold_bars"],
            snapshot.scorecard.genome.params["hold_bars"],
        )
        self.assertGreater(
            repair.candidate_genome.params["breakout_buffer"],
            snapshot.scorecard.genome.params["breakout_buffer"],
        )

    def test_repair_engine_uses_profit_lock_exit_outcome_to_tighten_runtime_and_capital(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "trend_exhaustion:seed",
                    "trend_exhaustion",
                    {"lookback": 18.0, "stop": 0.02, "hold_bars": 8.0},
                ),
                stage=PromotionStage.LIVE,
                edge_score=0.20,
                robustness_score=0.50,
                deployment_score=0.18,
                total_score=0.22,
                reasons=["promote_live"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.16,
                walkforward_expectancy=0.04,
                shadow_expectancy=0.03,
                live_expectancy=0.01,
                max_drawdown_pct=5.0,
                trade_count=30,
                cost_drag_pct=0.10,
                latency_ms=55.0,
                regime_consistency=0.78,
            ),
        )

        repair = repair_engine.propose(
            snapshot,
            feedback=RepairFeedbackSummary(
                source_strategy_id="trend_exhaustion:seed",
                source_runtime_id="BTC/USDT:USDT|5m|trend_exhaustion:seed",
                preferred_action=RepairActionType.TIGHTEN_RISK,
                notes={
                    "autonomy_outcomes": {
                        "profit_lock_exit_count": 1,
                    }
                },
            ),
        )

        self.assertEqual(repair.action, RepairActionType.TIGHTEN_RISK)
        self.assertIn("repair_feedback_risk_bias", repair.reasons)
        self.assertIn("repair_feedback_profit_lock_exit_bias", repair.reasons)
        self.assertEqual(repair.runtime_overrides["take_profit_bias"], 0.95)
        self.assertEqual(repair.runtime_overrides["max_weight_multiplier"], 0.45)
        self.assertEqual(repair.capital_multiplier, 0.45)
        self.assertLess(
            repair.candidate_genome.params["stop"],
            snapshot.scorecard.genome.params["stop"],
        )
        self.assertLess(
            repair.candidate_genome.params["hold_bars"],
            snapshot.scorecard.genome.params["hold_bars"],
        )

    def test_repair_engine_uses_forced_exit_outcomes_to_extend_cooldown(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "breakout_runtime:seed",
                    "breakout_runtime",
                    {"lookback": 18.0, "cooldown_bars": 4.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.20,
                robustness_score=0.50,
                deployment_score=0.18,
                total_score=0.22,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.16,
                walkforward_expectancy=0.04,
                shadow_expectancy=0.03,
                live_expectancy=0.01,
                max_drawdown_pct=5.0,
                trade_count=30,
                cost_drag_pct=0.10,
                latency_ms=55.0,
                regime_consistency=0.78,
            ),
        )

        repair = repair_engine.propose(
            snapshot,
            feedback=RepairFeedbackSummary(
                source_strategy_id="breakout_runtime:seed",
                source_runtime_id="BTC/USDT:USDT|5m|breakout_runtime:seed",
                preferred_action=RepairActionType.TIGHTEN_RISK,
                notes={
                    "autonomy_outcomes": {
                        "forced_exit_count": 2,
                    }
                },
            ),
        )

        self.assertIn("repair_feedback_forced_exit_bias", repair.reasons)
        self.assertEqual(repair.runtime_overrides["entry_cooldown_bars_multiplier"], 1.35)
        self.assertEqual(repair.runtime_overrides["max_weight_multiplier"], 0.40)
        self.assertEqual(repair.capital_multiplier, 0.40)

    def test_repair_engine_uses_success_history_to_raise_validation_stage(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "volatility_reclaim:seed",
                    "volatility_reclaim",
                    {"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.22,
                robustness_score=0.60,
                deployment_score=0.24,
                total_score=0.26,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.20,
                walkforward_expectancy=0.08,
                shadow_expectancy=0.05,
                live_expectancy=0.01,
                max_drawdown_pct=7.0,
                trade_count=28,
                cost_drag_pct=0.18,
                latency_ms=55.0,
                regime_consistency=0.78,
            ),
        )

        repair = repair_engine.propose(
            snapshot,
            feedback=RepairFeedbackSummary(
                source_strategy_id="volatility_reclaim:seed",
                source_runtime_id="ETH/USDT:USDT|5m|volatility_reclaim:seed",
                attempts=2,
                successes=2,
                preferred_action=RepairActionType.RAISE_SELECTIVITY,
                suggested_validation_stage=PromotionStage.PAPER,
            ),
        )

        self.assertEqual(repair.validation_stage, PromotionStage.PAPER)
        self.assertEqual(repair.action, RepairActionType.RAISE_SELECTIVITY)
        self.assertIn("repair_lineage_recovered", repair.reasons)

    def test_repair_engine_uses_runtime_override_hold_mode_to_force_conservative_repair(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "mean_revert_runtime:seed",
                    "mean_revert_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0, "stop": 0.02},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.24,
                robustness_score=0.56,
                deployment_score=0.22,
                total_score=0.25,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.18,
                walkforward_expectancy=0.05,
                shadow_expectancy=0.04,
                live_expectancy=0.02,
                max_drawdown_pct=5.0,
                trade_count=30,
                cost_drag_pct=0.10,
                latency_ms=55.0,
                regime_consistency=0.80,
            ),
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                },
                runtime_override_state={
                    "recovery_mode": "hold",
                },
            ),
        )

        repair = repair_engine.propose(snapshot)

        self.assertEqual(repair.action, RepairActionType.TIGHTEN_RISK)
        self.assertEqual(repair.validation_stage, PromotionStage.SHADOW)
        self.assertEqual(repair.priority, 3)
        self.assertEqual(repair.runtime_overrides["max_weight_multiplier"], 0.5)
        self.assertEqual(repair.runtime_overrides["entry_cooldown_bars_multiplier"], 1.25)
        self.assertEqual(repair.capital_multiplier, 0.5)
        self.assertIn("repair_runtime_override_recovery_mode:hold", repair.reasons)
        self.assertIn("repair_runtime_override_hold_bias", repair.reasons)
        self.assertLess(
            repair.candidate_genome.params["hold_bars"],
            snapshot.scorecard.genome.params["hold_bars"],
        )

    def test_repair_engine_deprioritizes_repair_when_runtime_override_accelerates(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        base_snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "trend_runtime:seed",
                    "trend_runtime",
                    {"lookback": 18.0, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.22,
                robustness_score=0.55,
                deployment_score=0.20,
                total_score=0.24,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.18,
                walkforward_expectancy=-0.03,
                shadow_expectancy=0.04,
                live_expectancy=0.01,
                max_drawdown_pct=6.0,
                trade_count=32,
                cost_drag_pct=0.10,
                latency_ms=90.0,
                regime_consistency=0.76,
            ),
        )
        accelerate_snapshot = replace(
            base_snapshot,
            notes=compose_runtime_policy_notes(
                runtime_override_state={
                    "recovery_mode": "accelerate",
                }
            ),
        )

        baseline = repair_engine.propose(base_snapshot)
        accelerated = repair_engine.propose(accelerate_snapshot)

        self.assertEqual(baseline.priority, 2)
        self.assertEqual(accelerated.priority, 1)
        self.assertIn("repair_runtime_override_recovery_mode:accelerate", accelerated.reasons)
        self.assertIn("repair_runtime_override_accelerate_relief", accelerated.reasons)

    def test_repair_engine_deprioritizes_repair_further_when_runtime_override_releases(self):
        repair_engine = StrategyRepairEngine(EvolutionConfig())
        base_snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "volatility_runtime:seed",
                    "volatility_runtime",
                    {"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.20,
                robustness_score=0.52,
                deployment_score=0.18,
                total_score=0.22,
                reasons=["promote_paper"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.16,
                walkforward_expectancy=-0.04,
                shadow_expectancy=0.03,
                live_expectancy=-0.01,
                max_drawdown_pct=5.0,
                trade_count=30,
                cost_drag_pct=0.30,
                latency_ms=55.0,
                regime_consistency=0.78,
            ),
        )
        release_snapshot = replace(
            base_snapshot,
            notes=compose_runtime_policy_notes(
                runtime_override_state={
                    "recovery_mode": "release",
                }
            ),
        )

        baseline = repair_engine.propose(base_snapshot)
        released = repair_engine.propose(release_snapshot)

        self.assertEqual(baseline.priority, 5)
        self.assertEqual(released.priority, 3)
        self.assertIn("repair_runtime_override_recovery_mode:release", released.reasons)
        self.assertIn("repair_runtime_override_release_relief", released.reasons)

    def test_director_plans_live_promotion_from_experiment_results(self):
        director = AutonomousDirector(
            EvolutionConfig(
                paper_threshold=0.30,
                live_threshold=0.50,
                autonomy_min_runtime_trades=24,
            )
        )
        scorecard = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation:seed",
                "trend_pullback_continuation",
                {"lookback": 18.0, "hold_bars": 6.0},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.52,
            robustness_score=0.78,
            deployment_score=0.61,
            total_score=0.57,
            reasons=["promote_paper"],
        )
        result = ExperimentResult(
            symbol="BTC/USDT",
            timeframe="5m",
            scorecards=[scorecard],
            promoted=[scorecard],
            allocations=[
                CapitalAllocation(
                    strategy_id="trend_pullback_continuation:seed",
                    stage=PromotionStage.PAPER,
                    allocated_capital=2000.0,
                    weight=0.20,
                    reasons=["paper"],
                )
            ],
            candle_count=300,
            metrics_by_strategy={
                "trend_pullback_continuation:seed": ValidationMetrics(
                    backtest_expectancy=0.48,
                    walkforward_expectancy=0.36,
                    shadow_expectancy=0.28,
                    live_expectancy=0.16,
                    max_drawdown_pct=5.0,
                    trade_count=42,
                    cost_drag_pct=0.10,
                    latency_ms=35.0,
                    regime_consistency=0.81,
                )
            },
        )

        directive = director.plan_from_experiments([result])

        self.assertEqual(len(directive.execution), 1)
        self.assertEqual(
            directive.execution[0].action,
            ExecutionAction.PROMOTE_TO_LIVE,
        )

    def test_director_carries_runtime_override_hold_mode_into_planner_reasons(self):
        director = AutonomousDirector(
            EvolutionConfig(
                paper_threshold=0.30,
                live_threshold=0.50,
                autonomy_min_runtime_trades=24,
                autonomy_repair_expectancy_floor=0.0,
            )
        )
        scorecard = ScoreCard(
            genome=StrategyGenome(
                "trend_pullback_continuation@BTC_USDT_5m:repair",
                "trend_pullback_continuation",
                {"lookback": 18.0, "hold_bars": 6.0},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.52,
            robustness_score=0.78,
            deployment_score=0.61,
            total_score=0.57,
            reasons=["repair_reentry_candidate"],
        )
        runtime_id = "BTC/USDT|5m|trend_pullback_continuation@BTC_USDT_5m:repair"
        result = ExperimentResult(
            symbol="BTC/USDT",
            timeframe="5m",
            scorecards=[scorecard],
            promoted=[scorecard],
            allocations=[
                CapitalAllocation(
                    strategy_id="trend_pullback_continuation@BTC_USDT_5m:repair",
                    stage=PromotionStage.PAPER,
                    allocated_capital=2000.0,
                    weight=0.20,
                    reasons=["paper"],
                )
            ],
            candle_count=300,
            metrics_by_strategy={
                "trend_pullback_continuation@BTC_USDT_5m:repair": ValidationMetrics(
                    backtest_expectancy=0.48,
                    walkforward_expectancy=0.36,
                    shadow_expectancy=0.28,
                    live_expectancy=0.16,
                    max_drawdown_pct=5.0,
                    trade_count=42,
                    cost_drag_pct=0.10,
                    latency_ms=35.0,
                    regime_consistency=0.81,
                )
            },
        )
        previous_state = RuntimeState(
            runtime_id=runtime_id,
            symbol="BTC/USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation@BTC_USDT_5m:repair",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.57,
            allocated_capital=2000.0,
            desired_capital=800.0,
            current_capital=800.0,
            current_weight=0.08,
            notes=compose_runtime_policy_notes(
                repair_reentry_notes=build_repair_reentry_notes(
                    source_runtime_id="BTC/USDT|5m|trend_pullback_continuation:seed",
                ),
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "recovery",
                    "source_runtime_id": "BTC/USDT|5m|trend_pullback_continuation:seed",
                },
            ),
        )

        directive = director.plan_from_experiments(
            [result],
            previous_states=[previous_state],
            runtime_overrides={
                runtime_id: {
                    "health_status": "active",
                    "current_drawdown_pct": 5.0,
                    "consecutive_losses": 2,
                    "notes": {
                        "total_net_pnl": -15.0,
                        "closed_trade_count": 2,
                        "win_rate": 0.0,
                    },
                }
            },
        )

        self.assertEqual(directive.execution[0].action, ExecutionAction.KEEP)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.PAPER)
        self.assertIn("runtime_override_recovery_mode:hold", directive.execution[0].reasons)
        self.assertIn("repair_reentry_probation_active", directive.execution[0].reasons)

    def test_director_namespaces_runtime_ids_across_symbols(self):
        director = AutonomousDirector(EvolutionConfig())
        shared_scorecard = ScoreCard(
            genome=StrategyGenome(
                "volatility_reclaim:seed",
                "volatility_reclaim",
                {"lookback": 20.0, "hold_bars": 8.0},
            ),
            stage=PromotionStage.PAPER,
            edge_score=0.40,
            robustness_score=0.75,
            deployment_score=0.45,
            total_score=0.41,
            reasons=["promote_paper"],
        )
        metrics = ValidationMetrics(
            backtest_expectancy=0.35,
            walkforward_expectancy=0.24,
            shadow_expectancy=0.18,
            live_expectancy=0.08,
            max_drawdown_pct=5.0,
            trade_count=32,
            cost_drag_pct=0.10,
            latency_ms=40.0,
            regime_consistency=0.80,
        )
        results = [
            ExperimentResult(
                symbol="BTC/USDT",
                timeframe="5m",
                scorecards=[shared_scorecard],
                promoted=[shared_scorecard],
                allocations=[],
                candle_count=200,
                metrics_by_strategy={"volatility_reclaim:seed": metrics},
            ),
            ExperimentResult(
                symbol="ETH/USDT",
                timeframe="5m",
                scorecards=[shared_scorecard],
                promoted=[shared_scorecard],
                allocations=[],
                candle_count=200,
                metrics_by_strategy={"volatility_reclaim:seed": metrics},
            ),
        ]

        snapshots = director.build_runtime_snapshots(results)

        self.assertEqual(len(snapshots), 2)
        self.assertNotEqual(snapshots[0].directive_id, snapshots[1].directive_id)
        self.assertIn("BTC/USDT", snapshots[0].directive_id)
        self.assertIn("ETH/USDT", snapshots[1].directive_id)

    def test_planner_retires_runtime_when_repair_feedback_is_exhausted(self):
        planner = AutonomyPlanner(EvolutionConfig())
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=StrategyGenome(
                    "microtrend_breakout:seed",
                    "microtrend_breakout",
                    {"lookback": 18.0, "breakout_buffer": 0.002, "hold_bars": 6.0},
                ),
                stage=PromotionStage.PAPER,
                edge_score=0.18,
                robustness_score=0.50,
                deployment_score=0.14,
                total_score=0.17,
                reasons=["promote_shadow"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.10,
                walkforward_expectancy=-0.04,
                shadow_expectancy=0.01,
                live_expectancy=-0.03,
                max_drawdown_pct=8.0,
                trade_count=24,
                cost_drag_pct=0.21,
                latency_ms=45.0,
                regime_consistency=0.71,
            ),
            runtime_id="BTC/USDT:USDT|5m|microtrend_breakout:seed",
        )

        directive = planner.plan(
            [snapshot],
            repair_feedback={
                snapshot.directive_id: RepairFeedbackSummary(
                    source_strategy_id="microtrend_breakout:seed",
                    source_runtime_id=snapshot.directive_id,
                    attempts=3,
                    failures=3,
                    consecutive_failures=3,
                    probation_required=True,
                    retire_recommended=True,
                )
            },
        )

        self.assertEqual(directive.execution[0].action, ExecutionAction.OBSERVE)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.REJECT)
        self.assertEqual(directive.retired, [snapshot.directive_id])
        self.assertEqual(len(directive.repairs), 1)
        self.assertEqual(directive.repairs[0].action, RepairActionType.RETIRE)


if __name__ == "__main__":
    unittest.main()
