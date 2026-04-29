import unittest
import json

from nextgen_evolution import EvolutionConfig, RepairActionType, RepairFeedbackEngine
from nextgen_evolution.models import PromotionStage


class NextGenRepairFeedbackTests(unittest.TestCase):
    def test_feedback_engine_flags_probation_and_retirement_after_consecutive_failures(self):
        engine = RepairFeedbackEngine(
            EvolutionConfig(
                autonomy_repair_retire_after_failures=2,
                autonomy_repair_promote_after_successes=2,
            )
        )

        feedback = engine.build(
            [
                {
                    "id": 1,
                    "created_at": "2026-04-11T00:00:00+00:00",
                    "source_runtime_id": "BTC/USDT:USDT|5m|microtrend_breakout:seed",
                    "source_strategy_id": "microtrend_breakout:seed",
                    "candidate_strategy_id": "microtrend_breakout@BTC_USDT_USDT_5m:repair1",
                    "action": RepairActionType.MUTATE_AND_REVALIDATE.value,
                    "status": PromotionStage.REJECT.value,
                },
                {
                    "id": 2,
                    "created_at": "2026-04-11T00:05:00+00:00",
                    "source_runtime_id": "BTC/USDT:USDT|5m|microtrend_breakout:seed",
                    "source_strategy_id": "microtrend_breakout:seed",
                    "candidate_strategy_id": "microtrend_breakout@BTC_USDT_USDT_5m:repair2",
                    "action": RepairActionType.RAISE_SELECTIVITY.value,
                    "status": "no_score",
                },
            ],
            runtime_ids=["BTC/USDT:USDT|5m|microtrend_breakout:seed"],
        )

        summary = feedback["BTC/USDT:USDT|5m|microtrend_breakout:seed"]
        self.assertEqual(summary.attempts, 2)
        self.assertEqual(summary.failures, 2)
        self.assertEqual(summary.consecutive_failures, 2)
        self.assertTrue(summary.probation_required)
        self.assertTrue(summary.retire_recommended)
        self.assertEqual(summary.suggested_validation_stage, PromotionStage.SHADOW)

    def test_feedback_engine_can_unlock_faster_validation_after_repeated_successes(self):
        engine = RepairFeedbackEngine(
            EvolutionConfig(
                autonomy_repair_retire_after_failures=2,
                autonomy_repair_promote_after_successes=2,
            )
        )

        feedback = engine.build(
            [
                {
                    "id": 1,
                    "created_at": "2026-04-11T00:00:00+00:00",
                    "source_runtime_id": "ETH/USDT:USDT|5m|volatility_reclaim:seed",
                    "source_strategy_id": "volatility_reclaim:seed",
                    "candidate_strategy_id": "volatility_reclaim@ETH_USDT_USDT_5m:repair1",
                    "action": RepairActionType.TIGHTEN_RISK.value,
                    "status": PromotionStage.PAPER.value,
                },
                {
                    "id": 2,
                    "created_at": "2026-04-11T00:05:00+00:00",
                    "source_runtime_id": "ETH/USDT:USDT|5m|volatility_reclaim:seed",
                    "source_strategy_id": "volatility_reclaim:seed",
                    "candidate_strategy_id": "volatility_reclaim@ETH_USDT_USDT_5m:repair2",
                    "action": RepairActionType.RAISE_SELECTIVITY.value,
                    "status": PromotionStage.PAPER.value,
                },
            ],
            strategy_ids=["volatility_reclaim:seed"],
        )

        summary = feedback["volatility_reclaim:seed"]
        self.assertEqual(summary.successes, 2)
        self.assertFalse(summary.probation_required)
        self.assertFalse(summary.retire_recommended)
        self.assertEqual(summary.suggested_validation_stage, PromotionStage.PAPER)
        self.assertEqual(summary.preferred_action, RepairActionType.RAISE_SELECTIVITY)

    def test_feedback_engine_uses_profit_lock_harvest_outcomes_to_bias_selectivity(self):
        engine = RepairFeedbackEngine(EvolutionConfig())

        feedback = engine.build(
            [],
            runtime_ids=["BTC/USDT:USDT|5m|trend_pullback_continuation:seed"],
            autonomy_outcome_rows=[
                {
                    "id": 1,
                    "event_type": "close",
                    "net_pnl": 45.0,
                    "metadata_json": json.dumps(
                        {
                            "runtime_id": "BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                            "strategy_id": "trend_pullback_continuation:seed",
                            "reason": "nextgen_autonomy_profit_lock_reduce",
                        }
                    ),
                },
                {
                    "id": 2,
                    "event_type": "close",
                    "net_pnl": 32.0,
                    "metadata_json": json.dumps(
                        {
                            "runtime_id": "BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                            "strategy_id": "trend_pullback_continuation:seed",
                            "reason": "nextgen_autonomy_live_profit_lock_reduce",
                        }
                    ),
                },
            ],
        )

        summary = feedback["BTC/USDT:USDT|5m|trend_pullback_continuation:seed"]
        self.assertEqual(summary.preferred_action, RepairActionType.RAISE_SELECTIVITY)
        self.assertEqual(
            summary.notes["autonomy_outcomes"]["profit_lock_harvest_count"],
            2,
        )
        self.assertGreater(
            summary.notes["autonomy_outcomes"]["profit_lock_net_pnl"],
            0.0,
        )

    def test_feedback_engine_uses_profit_lock_exit_outcomes_to_bias_risk_tightening(self):
        engine = RepairFeedbackEngine(EvolutionConfig())

        feedback = engine.build(
            [],
            strategy_ids=["trend_exhaustion:seed"],
            autonomy_outcome_rows=[
                {
                    "id": 1,
                    "event_type": "close",
                    "net_pnl": 18.0,
                    "metadata_json": json.dumps(
                        {
                            "runtime_id": "BTC/USDT:USDT|5m|trend_exhaustion:seed",
                            "strategy_id": "trend_exhaustion:seed",
                            "reason": "nextgen_autonomy_live_close",
                            "reasons": ["profit_lock_exit"],
                        }
                    ),
                }
            ],
        )

        summary = feedback["trend_exhaustion:seed"]
        self.assertEqual(summary.preferred_action, RepairActionType.TIGHTEN_RISK)
        self.assertEqual(
            summary.notes["autonomy_outcomes"]["profit_lock_exit_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
