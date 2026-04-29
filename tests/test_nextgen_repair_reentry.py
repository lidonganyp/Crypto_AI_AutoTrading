import unittest

from nextgen_evolution import RepairActionType, RepairPlan, RepairReentryPlanner
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.models import (
    PromotionStage,
    ScoreCard,
    StrategyGenome,
    ValidationMetrics,
)
from nextgen_evolution.repair_cycle import RepairExecutionResult
from nextgen_evolution.runtime_override_policy import (
    lifecycle_policy_repair_reentry_notes,
    lifecycle_policy_runtime_overrides,
)


class NextGenRepairReentryTests(unittest.TestCase):
    def test_reentry_planner_caps_stage_to_validation_target(self):
        candidate = StrategyGenome(
            "microtrend_breakout@BTC_USDT_USDT_5m:repair",
            "microtrend_breakout",
            {"lookback": 18.0, "breakout_buffer": 0.0022, "hold_bars": 5.0},
            mutation_of="microtrend_breakout:seed",
            tags=("repair",),
        )
        raw_result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=candidate,
                    stage=PromotionStage.LIVE,
                    edge_score=0.62,
                    robustness_score=0.86,
                    deployment_score=0.70,
                    total_score=0.65,
                    reasons=["promote_live"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=250,
            metrics_by_strategy={
                candidate.strategy_id: ValidationMetrics(
                    backtest_expectancy=0.48,
                    walkforward_expectancy=0.36,
                    shadow_expectancy=0.28,
                    live_expectancy=0.19,
                    max_drawdown_pct=4.5,
                    trade_count=38,
                    cost_drag_pct=0.10,
                    latency_ms=35.0,
                    regime_consistency=0.84,
                )
            },
            notes={"repair_validation": {"source_runtime_id": "BTC/USDT:USDT|5m|microtrend_breakout:seed"}},
            registry_run_id=42,
        )
        planner = RepairReentryPlanner()

        results, directives = planner.plan(
            [
                RepairExecutionResult(
                    plan=RepairPlan(
                        strategy_id="BTC/USDT:USDT|5m|microtrend_breakout:seed",
                        action=RepairActionType.MUTATE_AND_REVALIDATE,
                        priority=3,
                        candidate_genome=candidate,
                        validation_stage=PromotionStage.SHADOW,
                        capital_multiplier=0.5,
                        runtime_overrides={"max_weight_multiplier": 0.4, "take_profit_bias": 1.1},
                        reasons=["drawdown_breach"],
                    ),
                    source_runtime_id="BTC/USDT:USDT|5m|microtrend_breakout:seed",
                    source_strategy_id="microtrend_breakout:seed",
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    experiment=raw_result,
                    repair_execution_id=7,
                )
            ]
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(directives), 1)
        self.assertEqual(results[0].scorecards[0].stage, PromotionStage.SHADOW)
        self.assertEqual(results[0].promoted[0].stage, PromotionStage.SHADOW)
        self.assertEqual(
            lifecycle_policy_repair_reentry_notes(results[0].notes)["effective_target_stage"],
            PromotionStage.SHADOW.value,
        )
        self.assertEqual(
            lifecycle_policy_runtime_overrides(results[0].notes)["max_weight_multiplier"],
            0.4,
        )
        self.assertNotIn("repair_reentry", results[0].notes)
        self.assertEqual(directives[0].action.value, "promote_to_shadow")
        self.assertEqual(directives[0].target_stage, PromotionStage.SHADOW)
        self.assertEqual(
            directives[0].strategy_id,
            "BTC/USDT:USDT|5m|microtrend_breakout@BTC_USDT_USDT_5m:repair",
        )


if __name__ == "__main__":
    unittest.main()
